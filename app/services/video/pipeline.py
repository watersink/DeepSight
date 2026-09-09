import logging
import signal
import time
from typing import Any, Optional

from app.services.video.async_processor import AsyncFrameProcessor
from app.services.video.ffmpeg_streamer import create_streamer
from app.services.video.frame_reader import FrameReader
from app.services.video.types import AlertCallback, AlertPredicate, StreamConfig

logger = logging.getLogger(__name__)


class VideoStreamPipeline:
    """
    视频流处理管道（可共享复用）。

    用法:
        pipeline = VideoStreamPipeline(config, skill)
        pipeline.run()  # 阻塞运行，Ctrl+C 退出
    """

    def __init__(
        self,
        config: StreamConfig,
        skill_instance: Any,
        on_alert: Optional[AlertCallback] = None,
        alert_predicate: Optional[AlertPredicate] = None,
        stop_event: Optional[Any] = None,
    ):
        self.config = config
        self.skill = skill_instance
        self.on_alert = on_alert
        self.alert_predicate = alert_predicate
        self._stop_event = stop_event

        self._reader: Optional[FrameReader] = None
        self._processor: Optional[AsyncFrameProcessor] = None
        self._streamer = None
        self._shutdown = False

    def _setup(self) -> bool:
        cfg = self.config
        self._reader = FrameReader(
            source_url=cfg.source_url,
            target_fps=cfg.fps,
            reconnect_delay=cfg.reconnect_delay,
        )
        if not self._reader.open():
            return False

        width, height = self._reader.get_resolution()
        if width % 2 or height % 2:
            logger.warning(
                "视频分辨率 %dx%d 含奇数边，推流将自动调整为偶数",
                width, height,
            )
        self._streamer = create_streamer(
            output_url=cfg.output_url,
            fps=cfg.fps,
            width=width,
            height=height,
            output_format=cfg.output_format,
            use_hardware_encoding=cfg.use_hardware_encoding,
            bitrate=cfg.bitrate,
            buffer_size=cfg.buffer_size,
        )

        task_config = {"fence_config": cfg.fence_config}
        task_config.update(cfg.extra)

        processor_kwargs = {
            "task_id": cfg.scene_id or "stream",
            "max_queue_size": cfg.queue_size,
            "alarm_interval": cfg.alarm_interval,
            "scene_id": cfg.scene_id,
            "on_alert": self.on_alert,
        }
        if self.alert_predicate is not None:
            processor_kwargs["alert_predicate"] = self.alert_predicate

        self._processor = AsyncFrameProcessor(**processor_kwargs)
        self._processor.start(
            skill_instance=self.skill,
            task_config=task_config,
            streamer=self._streamer,
        )
        return True

    def _install_signal_handlers(self):
        if self._stop_event is not None:
            return

        def _handler(signum, _frame):
            logger.info("收到停止信号 (%s)", signum)
            self._shutdown = True

        signal.signal(signal.SIGINT, _handler)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, _handler)

    def run(self):
        """阻塞运行主循环，直到收到停止信号或视频源永久失败。"""
        if not self._setup():
            raise RuntimeError(f"无法打开视频源: {self.config.source_url}")

        self._install_signal_handlers()
        logger.info("视频流管道已启动 scene=%s", self.config.scene_id)

        last_stats_log = time.time()
        try:
            while not self._shutdown:
                if self._stop_event is not None and self._stop_event.is_set():
                    logger.info("收到停止事件，准备退出 scene=%s", self.config.scene_id)
                    break

                frame = self._reader.read()
                if frame is None:
                    time.sleep(0.1)
                    continue
                self._processor.put_frame(frame)

                now = time.time()
                if now - last_stats_log >= 10.0:
                    s = self.stats
                    streamer_stats = self._streamer.stats if self._streamer else {}
                    logger.info(
                        "推流状态: captured=%d detected=%d streamed=%d dropped=%d "
                        "ffmpeg_sent=%d ffmpeg_errors=%d last_error=%s",
                        s.get("frames_captured", 0),
                        s.get("frames_detected", 0),
                        s.get("frames_streamed", 0),
                        s.get("frames_dropped", 0),
                        streamer_stats.get("frames_sent", 0),
                        streamer_stats.get("errors", 0),
                        streamer_stats.get("last_error"),
                    )
                    last_stats_log = now
        finally:
            self.stop()

    def stop(self):
        if self._reader:
            self._reader.release()
        if self._processor:
            self._processor.stop()
        logger.info("视频流管道已停止")

    @property
    def stats(self):
        if self._processor:
            return self._processor.stats
        return {}

    @property
    def processor(self) -> Optional[AsyncFrameProcessor]:
        return self._processor
