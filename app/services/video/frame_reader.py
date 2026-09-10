"""
视频帧读取器 - 参考 smart_engine 采集循环

支持 RTSP/RTMP/本地文件，断线重连与降帧采样。
解码优先：CUDA/NVDEC → QSV → OpenCV 软解。
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.services.video.decoder import (
    build_ffmpeg_decode_command,
    probe_stream_meta,
    select_hw_decode_backend,
)

logger = logging.getLogger(__name__)


class FrameReader:
    """从视频源读取帧，按目标帧率节流并降采样。"""

    def __init__(
        self,
        source_url: str,
        target_fps: float,
        reconnect_delay: float = 2.0,
        use_hardware_decoding: Optional[bool] = None,
    ):
        self.source_url = source_url
        self.target_fps = max(target_fps, 0.1)
        self.reconnect_delay = reconnect_delay
        self.frame_interval = 1.0 / self.target_fps
        if use_hardware_decoding is None:
            use_hardware_decoding = bool(
                getattr(settings, "STREAM_USE_HARDWARE_DECODING", True)
            )
        self.use_hardware_decoding = use_hardware_decoding

        self._cap: Optional[cv2.VideoCapture] = None
        self._ffmpeg: Optional[subprocess.Popen] = None
        self._backend: str = "opencv"  # cuda | qsv | opencv
        self._frame_nbytes = 0

        self._source_fps = self.target_fps
        self._width = 0
        self._height = 0
        self._last_frame_time = 0.0
        self._frame_counter = 0
        self._sample_step = 1
        self._stderr_thread: Optional[threading.Thread] = None
        self._last_ffmpeg_error = ""

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def decode_backend(self) -> str:
        return self._backend

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        try:
            if not proc.stderr:
                return
            for line_bytes in iter(proc.stderr.readline, b""):
                line = line_bytes.decode("utf-8", errors="ignore").rstrip()
                if line:
                    self._last_ffmpeg_error = line[:500]
                    lower = line.lower()
                    if any(k in lower for k in ("error", "failed", "invalid", "denied")):
                        logger.warning("FFmpeg decode: %s", line[:300])
        except Exception:
            pass

    def _close_ffmpeg(self) -> None:
        if self._ffmpeg is None:
            return
        proc = self._ffmpeg
        self._ffmpeg = None
        try:
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            proc.kill()
            proc.wait(timeout=2)
        except Exception:
            pass
        if (
            self._stderr_thread
            and self._stderr_thread.is_alive()
            and self._stderr_thread is not threading.current_thread()
        ):
            self._stderr_thread.join(timeout=1)
        self._stderr_thread = None

    def _close_opencv(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _open_opencv(self) -> bool:
        self._close_ffmpeg()
        self._cap = cv2.VideoCapture(self.source_url)
        if not self._cap.isOpened():
            logger.error("无法打开视频源(OpenCV): %s", self.source_url)
            return False

        self._source_fps = self._cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        if self._source_fps <= 0:
            self._source_fps = self.target_fps
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self._width <= 0 or self._height <= 0:
            logger.error("OpenCV 未能获取有效分辨率: %s", self.source_url)
            self._close_opencv()
            return False

        self._backend = "opencv"
        self._frame_nbytes = self._width * self._height * 3
        self._sample_step = max(1, int(round(self._source_fps / self.target_fps)))
        self._frame_counter = 0
        logger.info(
            "视频源(软解 OpenCV): %s, %dx%d, 源帧率=%.1f, 目标=%.1f, 采样步长=%d",
            self.source_url,
            self._width,
            self._height,
            self._source_fps,
            self.target_fps,
            self._sample_step,
        )
        return True

    def _open_ffmpeg(self, backend: str) -> bool:
        self._close_ffmpeg()
        self._close_opencv()

        width, height, fps = probe_stream_meta(self.source_url)
        if width <= 0 or height <= 0:
            logger.warning(
                "硬解前探测分辨率失败，回退 OpenCV: %s", self.source_url
            )
            return False

        self._width = width
        self._height = height
        self._source_fps = fps if fps and fps > 0 else self.target_fps
        self._frame_nbytes = self._width * self._height * 3
        self._sample_step = max(1, int(round(self._source_fps / self.target_fps)))
        self._frame_counter = 0

        cmd = build_ffmpeg_decode_command(self.source_url, backend)
        logger.info("启动 FFmpeg 硬解(%s): %s", backend, " ".join(cmd))
        try:
            flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            self._ffmpeg = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_nbytes * 2,
                creationflags=flags,
            )
        except FileNotFoundError:
            logger.error("FFmpeg 未安装或不在 PATH，无法硬解")
            return False
        except Exception:
            logger.exception("启动 FFmpeg 硬解失败")
            return False

        self._stderr_thread = threading.Thread(
            target=self._drain_stderr,
            args=(self._ffmpeg,),
            daemon=True,
            name=f"FFmpegDecodeStderr-{id(self)}",
        )
        self._stderr_thread.start()
        time.sleep(0.3)
        if self._ffmpeg.poll() is not None:
            err = self._last_ffmpeg_error or "进程已退出"
            logger.error("FFmpeg 硬解启动失败(%s): %s", backend, err)
            self._close_ffmpeg()
            return False

        self._backend = backend
        logger.info(
            "视频源(硬解 %s): %s, %dx%d, 源帧率=%.1f, 目标=%.1f, 采样步长=%d",
            backend.upper(),
            self.source_url,
            self._width,
            self._height,
            self._source_fps,
            self.target_fps,
            self._sample_step,
        )
        return True

    def open(self) -> bool:
        backend = select_hw_decode_backend(self.use_hardware_decoding)
        if backend:
            if self._open_ffmpeg(backend):
                return True
            logger.warning("硬解(%s)失败，回退 OpenCV 软解", backend)

        return self._open_opencv()

    def _reconnect(self) -> bool:
        self._close_ffmpeg()
        self._close_opencv()
        logger.warning(
            "视频源断开，%.1fs 后重连: %s", self.reconnect_delay, self.source_url
        )
        time.sleep(self.reconnect_delay)
        return self.open()

    def _read_ffmpeg_frame(self) -> Optional[np.ndarray]:
        proc = self._ffmpeg
        if proc is None or proc.stdout is None:
            return None
        if proc.poll() is not None:
            return None

        buf = bytearray()
        remaining = self._frame_nbytes
        while remaining > 0:
            chunk = proc.stdout.read(remaining)
            if not chunk:
                return None
            buf.extend(chunk)
            remaining -= len(chunk)

        try:
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(
                (self._height, self._width, 3)
            )
            return frame.copy()
        except Exception:
            logger.exception("解析 FFmpeg 原始帧失败")
            return None

    def _read_opencv_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None
        return True, frame

    def read(self) -> Optional[np.ndarray]:
        now = time.time()
        if now - self._last_frame_time < self.frame_interval:
            time.sleep(max(0.001, self.frame_interval - (now - self._last_frame_time)))

        while True:
            if self._backend in {"cuda", "qsv"}:
                if self._ffmpeg is None or self._ffmpeg.poll() is not None:
                    if not self._reconnect():
                        return None
                    continue
                frame = self._read_ffmpeg_frame()
                if frame is None:
                    if not self._reconnect():
                        return None
                    continue
            else:
                if self._cap is None or not self._cap.isOpened():
                    if not self._reconnect():
                        return None
                    continue
                ok, frame = self._read_opencv_frame()
                if not ok or frame is None:
                    if not self._reconnect():
                        return None
                    continue

            self._frame_counter += 1
            if self._frame_counter % self._sample_step != 0:
                continue

            self._last_frame_time = time.time()
            return frame

    def release(self):
        self._close_ffmpeg()
        self._close_opencv()

    def get_resolution(self) -> Tuple[int, int]:
        return self._width, self._height
