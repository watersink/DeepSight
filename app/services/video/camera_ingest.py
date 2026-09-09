"""摄像头级 ingest：单路解码 + 进程内多 skill fan-out。"""
from __future__ import annotations

import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from app.core.config import settings
from app.plugins.skill_registry import create_skill, merge_skill_config_overrides
from app.services.video.async_processor import AsyncFrameProcessor
from app.services.video.ffmpeg_streamer import create_streamer
from app.services.video.frame_reader import FrameReader
from app.services.video.types import AlertCallback, AlertPredicate

logger = logging.getLogger(__name__)


@dataclass
class SkillBranch:
    branch_id: str
    config: Dict[str, Any]
    processor: AsyncFrameProcessor
    streamer: Any
    skill_name: str = ""
    scene_id: str = ""
    lock: threading.Lock = field(default_factory=threading.Lock)


class CameraFanoutPipeline:
    """
    单路 FrameReader，按分支 fan-out 到多个 AsyncFrameProcessor。
    每个 skill 独立队列 / 识别 / 推流 / 告警。
    """

    def __init__(
        self,
        source_url: str,
        fps: float = 15.0,
        reconnect_delay: float = 2.0,
        queue_size: int = 4,
        on_alert: Optional[AlertCallback] = None,
        alert_predicate: Optional[AlertPredicate] = None,
        stop_event: Optional[Any] = None,
        control_queue: Optional[Any] = None,
    ):
        self.source_url = source_url
        self.fps = fps
        self.reconnect_delay = reconnect_delay
        self.queue_size = queue_size
        self.on_alert = on_alert
        self.alert_predicate = alert_predicate
        self._stop_event = stop_event
        self._control_queue = control_queue

        self._reader: Optional[FrameReader] = None
        self._width = 0
        self._height = 0
        self._branches: Dict[str, SkillBranch] = {}
        self._branches_lock = threading.RLock()
        self._shutdown = False

    @property
    def branch_count(self) -> int:
        with self._branches_lock:
            return len(self._branches)

    def _open_reader(self) -> bool:
        self._reader = FrameReader(
            source_url=self.source_url,
            target_fps=self.fps,
            reconnect_delay=self.reconnect_delay,
        )
        if not self._reader.open():
            return False
        self._width, self._height = self._reader.get_resolution()
        if self._width % 2 or self._height % 2:
            logger.warning(
                "视频分辨率 %dx%d 含奇数边，推流将自动调整为偶数",
                self._width,
                self._height,
            )
        return True

    def add_branch(self, branch_id: str, config: Dict[str, Any]) -> None:
        with self._branches_lock:
            if branch_id in self._branches:
                raise ValueError(f"分支已存在: {branch_id}")
            if self._width <= 0 or self._height <= 0:
                raise RuntimeError("解码尚未就绪，无法添加 skill 分支")

            skill_name = config.get("skill_name", settings.DEFAULT_SKILL_NAME)
            skill_config = merge_skill_config_overrides(
                config.get("skill_config"),
                enter_count=config.get("enter_count"),
                gate_direction=config.get("gate_direction"),
                count_line=config.get("count_line"),
                bypass_line=config.get("bypass_line"),
                mine_code=config.get("mine_code"),
                camera_code=config.get("camera_code"),
            )
            skill = create_skill(skill_name, skill_config)

            resolved = settings.resolve_stream_start(config)
            scene_id = resolved.get("scene_id", settings.DEFAULT_SCENE_ID)
            out_fps = float(resolved.get("out_fps", settings.DEFAULT_OUT_FPS))
            output_format = resolved.get("output_format", settings.DEFAULT_OUTPUT_FORMAT)
            alarm_interval = float(
                resolved.get("alarm_interval", settings.DEFAULT_ALARM_INTERVAL)
            )
            queue_size = int(resolved.get("queue_size", self.queue_size))
            push_stream = bool(resolved.get("push_annotated_stream", False))
            alert_image_enabled = bool(resolved.get("alert_image_enabled", True))
            alert_video_enabled = bool(resolved.get("alert_video_enabled", False))

            streamer = None
            if push_stream:
                streamer = create_streamer(
                    output_url=resolved["out_url"],
                    fps=out_fps,
                    width=self._width,
                    height=self._height,
                    output_format=output_format,
                    use_hardware_encoding=resolved.get(
                        "use_hardware_encoding", settings.STREAM_USE_HARDWARE_ENCODING
                    ),
                    bitrate=resolved.get("bitrate", settings.STREAM_BITRATE),
                    buffer_size=resolved.get("buffer_size", settings.STREAM_BUFFER_SIZE),
                )
            else:
                logger.info(
                    "未开启画框推流，跳过 FFmpeg scene=%s skill=%s",
                    scene_id,
                    skill_name,
                )

            processor_kwargs = {
                "task_id": f"{scene_id}:{skill_name}:{branch_id[:8]}",
                "max_queue_size": queue_size,
                "alarm_interval": alarm_interval,
                "scene_id": scene_id,
                "on_alert": self.on_alert,
            }
            if self.alert_predicate is not None:
                processor_kwargs["alert_predicate"] = self.alert_predicate

            processor = AsyncFrameProcessor(**processor_kwargs)
            task_config = {
                "fence_config": resolved.get("fence_config"),
                "alert_image_enabled": alert_image_enabled,
                "alert_video_enabled": alert_video_enabled,
                "push_annotated_stream": push_stream,
            }
            processor.start(
                skill_instance=skill,
                task_config=task_config,
                streamer=streamer,
            )

            self._branches[branch_id] = SkillBranch(
                branch_id=branch_id,
                config=resolved,
                processor=processor,
                streamer=streamer,
                skill_name=skill_name,
                scene_id=scene_id,
            )
            logger.info(
                "已添加 skill 分支 branch=%s scene=%s skill=%s（当前 %d 路）",
                branch_id,
                scene_id,
                skill_name,
                len(self._branches),
            )

    def remove_branch(self, branch_id: str) -> bool:
        with self._branches_lock:
            branch = self._branches.pop(branch_id, None)
        if not branch:
            return False
        try:
            branch.processor.stop()
        except Exception:
            logger.exception("停止分支 processor 失败 branch=%s", branch_id)
        logger.info(
            "已移除 skill 分支 branch=%s scene=%s skill=%s（剩余 %d 路）",
            branch_id,
            branch.scene_id,
            branch.skill_name,
            self.branch_count,
        )
        return True

    def _drain_control_queue(self) -> None:
        if self._control_queue is None:
            return
        import queue as queue_mod

        while True:
            try:
                msg = self._control_queue.get_nowait()
            except queue_mod.Empty:
                break
            except Exception:
                logger.exception("读取控制队列失败")
                break

            if not isinstance(msg, dict):
                continue
            cmd = msg.get("cmd")
            if cmd == "add":
                try:
                    self.add_branch(str(msg["branch_id"]), dict(msg.get("config") or {}))
                except Exception:
                    logger.exception(
                        "添加 skill 分支失败 branch=%s", msg.get("branch_id")
                    )
            elif cmd == "remove":
                self.remove_branch(str(msg.get("branch_id") or ""))
                if self.branch_count == 0:
                    logger.info("无剩余 skill 分支，准备退出 ingest")
                    self._shutdown = True
            elif cmd == "stop":
                self._shutdown = True

    def _fanout(self, frame) -> None:
        with self._branches_lock:
            branches = list(self._branches.values())
        n = len(branches)
        if n == 0:
            return
        if n == 1:
            branches[0].processor.put_frame(frame)
            return
        for branch in branches:
            # 多 skill 必须拷贝，避免检测/画框线程互相踩帧
            branch.processor.put_frame(frame.copy())

    def _install_signal_handlers(self) -> None:
        if self._stop_event is not None:
            return

        def _handler(signum, _frame):
            logger.info("收到停止信号 (%s)", signum)
            self._shutdown = True

        signal.signal(signal.SIGINT, _handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handler)

    def run(self, initial_branches: Optional[list] = None) -> None:
        if not self._open_reader():
            raise RuntimeError(f"无法打开视频源: {self.source_url}")

        for item in initial_branches or []:
            self.add_branch(str(item["branch_id"]), dict(item.get("config") or {}))

        if self.branch_count == 0:
            raise RuntimeError("摄像头 ingest 未配置任何 skill 分支")

        self._install_signal_handlers()
        logger.info(
            "摄像头 ingest 已启动 source=%s branches=%d",
            self.source_url,
            self.branch_count,
        )

        last_stats_log = time.time()
        try:
            while not self._shutdown:
                if self._stop_event is not None and self._stop_event.is_set():
                    logger.info("收到停止事件，退出摄像头 ingest")
                    break

                self._drain_control_queue()
                if self._shutdown:
                    break
                if self.branch_count == 0:
                    break

                frame = self._reader.read() if self._reader else None
                if frame is None:
                    time.sleep(0.1)
                    continue

                self._fanout(frame)

                now = time.time()
                if now - last_stats_log >= 10.0:
                    with self._branches_lock:
                        summary = [
                            (
                                b.scene_id,
                                b.skill_name,
                                b.processor.stats.get("frames_detected", 0),
                                b.processor.stats.get("frames_streamed", 0),
                            )
                            for b in self._branches.values()
                        ]
                    logger.info(
                        "ingest 状态 branches=%s detail=%s",
                        self.branch_count,
                        summary,
                    )
                    last_stats_log = now
        finally:
            self.stop()

    def stop(self) -> None:
        with self._branches_lock:
            branch_ids = list(self._branches.keys())
        for bid in branch_ids:
            self.remove_branch(bid)
        if self._reader:
            self._reader.release()
            self._reader = None
        logger.info("摄像头 ingest 已停止 source=%s", self.source_url)
