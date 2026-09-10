"""
异步帧处理器 - 参考 smart_engine OptimizedAsyncProcessor

检测、推流、告警三线程分离，技能实例通过接口注入，可被任意算法复用。
"""
import logging
import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

from app.services.video.ffmpeg_streamer import FFmpegStreamerBase
from app.services.video.types import AlertCallback, AlertPredicate, default_alert_predicate

logger = logging.getLogger(__name__)


class AsyncFrameProcessor:
    """
    通用异步帧处理器。

    skill 需实现:
      - process(frame, fence_config=None) -> SkillResult
      - 可选 _draw_detections_on_frame / draw_detections_on_frame
    """

    def __init__(
        self,
        task_id: str = "default",
        max_queue_size: int = 2,
        alarm_interval: float = 1.0,
        scene_id: str = "",
        on_alert: Optional[AlertCallback] = None,
        alert_predicate: AlertPredicate = default_alert_predicate,
    ):
        self.task_id = task_id
        self.max_queue_size = max_queue_size
        self.alarm_interval = alarm_interval
        self.scene_id = scene_id
        self.on_alert = on_alert
        self.alert_predicate = alert_predicate

        self.frame_buffer: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.alert_buffer: queue.Queue = queue.Queue(maxsize=8)

        self.running = False
        self.skill_instance = None
        self.task_config: Dict[str, Any] = {}
        self.streamer: Optional[FFmpegStreamerBase] = None

        self._detection_thread: Optional[threading.Thread] = None
        self._streaming_thread: Optional[threading.Thread] = None
        self._alert_thread: Optional[threading.Thread] = None

        self.latest_detection_result = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_raw_frame: Optional[np.ndarray] = None
        self._result_lock = threading.RLock()
        self._last_alert_time = 0.0

        self.stats: Dict[str, Any] = {
            "frames_captured": 0,
            "frames_detected": 0,
            "frames_streamed": 0,
            "frames_dropped": 0,
            "alerts_triggered": 0,
            "detection_fps": 0.0,
            "streaming_fps": 0.0,
            "avg_detection_time": 0.0,
        }
        self._detection_times: list = []
        self._last_stats_update = time.time()

    def start(
        self,
        skill_instance,
        task_config: Optional[Dict[str, Any]] = None,
        streamer: Optional[FFmpegStreamerBase] = None,
    ):
        self.skill_instance = skill_instance
        self.task_config = task_config or {}
        self.streamer = streamer
        self.running = True

        if self.streamer and not self.streamer.start():
            detail = ""
            try:
                detail = (self.streamer.stats or {}).get("last_error") or ""
            except Exception:
                detail = ""
            if detail:
                raise RuntimeError(f"推流器启动失败: {detail[:800]}")
            raise RuntimeError(
                "推流器启动失败（常见原因：未安装 FFmpeg、NVENC 不可用、"
                "无法连接推流地址 ZLM）。请查看上方 FFmpeg 相关日志。"
            )

        self._detection_thread = threading.Thread(
            target=self._detection_worker, daemon=True,
            name=f"Detection-{self.task_id}",
        )
        self._detection_thread.start()

        if self.streamer:
            self._streaming_thread = threading.Thread(
                target=self._streaming_worker, daemon=True,
                name=f"Streaming-{self.task_id}",
            )
            self._streaming_thread.start()

        self._alert_thread = threading.Thread(
            target=self._alert_worker, daemon=True,
            name=f"Alert-{self.task_id}",
        )
        self._alert_thread.start()
        logger.info("任务 %s 异步帧处理器已启动", self.task_id)

    def stop(self):
        self.running = False
        if self.streamer:
            self.streamer.stop()
        logger.info("任务 %s 已停止, stats=%s", self.task_id, self.stats)

    def put_frame(self, frame: np.ndarray) -> bool:
        if not self.running:
            return False
        try:
            if self.frame_buffer.full():
                try:
                    self.frame_buffer.get_nowait()
                    self.stats["frames_dropped"] += 1
                except queue.Empty:
                    pass
            self.frame_buffer.put(
                {"frame": frame, "timestamp": time.time()}, block=False
            )
            # 采集侧同步更新原始帧，检测失败时推流线程仍可推送画面
            with self._result_lock:
                self.latest_raw_frame = frame
            self.stats["frames_captured"] += 1
            return True
        except queue.Full:
            self.stats["frames_dropped"] += 1
            return False

    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        with self._result_lock:
            if self.latest_detection_result is None:
                return None
            return {
                "result": self.latest_detection_result,
                "frame": self.latest_annotated_frame,
            }

    def _detection_worker(self):
        fence_config = self.task_config.get("fence_config")

        while self.running:
            try:
                item = self.frame_buffer.get(timeout=1.0)
            except queue.Empty:
                continue

            frame = item["frame"]
            t0 = time.time()
            try:
                result = self.skill_instance.process(frame, fence_config)
                detect_duration = time.time() - t0
                self._detection_times.append(time.time())
                if len(self._detection_times) > 100:
                    self._detection_times = self._detection_times[-50:]

                if not result.success:
                    logger.warning("检测失败: %s", result.error_message)
                    # 推理失败时仍尝试画 count_line/bypass_line，避免 FLV 只有原流无标注
                    overlay_data = result.data if isinstance(result.data, dict) else {}
                    try:
                        annotated = self._draw(frame, overlay_data)
                    except Exception:
                        annotated = frame
                    with self._result_lock:
                        self.latest_raw_frame = frame
                        self.latest_annotated_frame = annotated
                    continue

                draw_t0 = time.time()
                annotated = self._draw(frame, result.data)
                draw_duration = time.time() - draw_t0
                process_duration = detect_duration + draw_duration

                with self._result_lock:
                    self.latest_detection_result = result
                    self.latest_annotated_frame = annotated
                    self.stats["avg_detection_time"] = process_duration

                alert_data = dict(result.data)
                skill_inst = getattr(self, "skill_instance", None)
                if skill_inst is not None:
                    cfg = getattr(skill_inst, "config", None) or {}
                    skill_name = cfg.get("name") if isinstance(cfg, dict) else None
                    if skill_name:
                        alert_data.setdefault("skill_name", skill_name)
                alert_data["detect_time_ms"] = round(detect_duration * 1000, 2)
                alert_data["draw_time_ms"] = round(draw_duration * 1000, 2)
                alert_data["process_time_ms"] = round(process_duration * 1000, 2)
                alert_data["alert_image_enabled"] = bool(
                    self.task_config.get("alert_image_enabled", True)
                )
                alert_data["alert_video_enabled"] = bool(
                    self.task_config.get("alert_video_enabled", False)
                )
                timing_ms = alert_data.get("timing_ms")
                if isinstance(timing_ms, dict):
                    for key, value in timing_ms.items():
                        alert_data.setdefault(key, value)

                alert_item = {
                    "data": alert_data,
                    # 上传带检测框/线段的标注帧，便于告警回溯
                    "raw_frame": annotated.copy(),
                    "timestamp": time.time(),
                }
                try:
                    if self.alert_buffer.full():
                        self.alert_buffer.get_nowait()
                    self.alert_buffer.put(alert_item, block=False)
                except queue.Full:
                    pass

                self.stats["frames_detected"] += 1
                self._update_stats()

            except Exception as e:
                logger.error("检测线程出错: %s", e)
                time.sleep(0.1)

    def _streaming_worker(self):
        fps = self.streamer.fps if self.streamer else 15.0
        target_interval = 1.0 / fps
        adaptive_interval = target_interval
        last_push = time.time()
        consecutive_failures = 0
        stream_count = 0
        stats_t = time.time()

        while self.running:
            now = time.time()
            if now - last_push < adaptive_interval:
                time.sleep(max(0.001, adaptive_interval - (now - last_push)))
                continue

            with self._result_lock:
                frame = self.latest_annotated_frame
                if frame is None:
                    frame = self.latest_raw_frame

            if frame is None or not self.streamer or not self.streamer.is_running:
                time.sleep(0.05)
                continue

            if self.streamer.push_frame(frame):
                self.stats["frames_streamed"] += 1
                stream_count += 1
                last_push = now
                consecutive_failures = 0
                adaptive_interval = max(target_interval, adaptive_interval * 0.99)
            else:
                consecutive_failures += 1
                if consecutive_failures > 3:
                    adaptive_interval = min(adaptive_interval * 1.2, target_interval * 2)
                if consecutive_failures > 10 and consecutive_failures % 20 == 0:
                    self.streamer.reset_restart_count()
                time.sleep(0.05)

            if now - stats_t >= 3.0 and stream_count > 0:
                self.stats["streaming_fps"] = stream_count / (now - stats_t)
                stream_count = 0
                stats_t = now

    def _alert_worker(self):
        while self.running:
            try:
                item = self.alert_buffer.get(timeout=1.0)
            except queue.Empty:
                continue

            data = item["data"]
            if not self.alert_predicate(data):
                continue

            now = time.time()
            if now - self._last_alert_time < self.alarm_interval:
                continue

            self._last_alert_time = now
            self.stats["alerts_triggered"] += 1

            if self.on_alert:
                try:
                    self.on_alert(data, item["raw_frame"], self.scene_id)
                except Exception as e:
                    logger.error("告警回调失败: %s", e)
            else:
                timing_ms = data.get("timing_ms") if isinstance(data.get("timing_ms"), dict) else {}
                timing_suffix = ""
                if timing_ms:
                    timing_suffix = " " + " ".join(
                        f"{key}={float(value):.2f}" for key, value in timing_ms.items()
                    )
                logger.info(
                    "告警 scene=%s count=%s enter_count=%s enter_count_change=%s "
                    "count_exit_violation=%s bypass_violation=%s "
                    "process_time_ms=%.2f detect_time_ms=%.2f draw_time_ms=%.2f%s",
                    self.scene_id,
                    data.get("count", 0),
                    data.get("enter_count", 0),
                    data.get("has_enter_count_change", False),
                    data.get("has_count_exit_violation", False),
                    data.get("has_bypass_violation", False),
                    float(data.get("process_time_ms", 0.0)),
                    float(data.get("detect_time_ms", 0.0)),
                    float(data.get("draw_time_ms", 0.0)),
                    timing_suffix,
                )

    def _draw(self, frame: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        skill = self.skill_instance
        if hasattr(skill, "draw_detections_on_frame") and callable(skill.draw_detections_on_frame):
            return skill.draw_detections_on_frame(frame.copy(), data)
        if hasattr(skill, "_draw_detections_on_frame") and callable(skill._draw_detections_on_frame):
            return skill._draw_detections_on_frame(frame.copy(), data)
        return self._default_draw(frame.copy(), data)

    @staticmethod
    def _default_draw(frame: np.ndarray, data: Dict[str, Any]) -> np.ndarray:
        for det in data.get("detections", []):
            bbox = det.get("bbox", [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.get('class_name', '?')}: {det.get('confidence', 0):.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame

    def _update_stats(self):
        now = time.time()
        if now - self._last_stats_update < 2.0:
            return
        cutoff = now - 5.0
        recent = [t for t in self._detection_times if t >= cutoff]
        self.stats["detection_fps"] = len(recent) / 5.0 if recent else 0.0
        self._last_stats_update = now
