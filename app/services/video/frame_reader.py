"""
视频帧读取器 - 参考 smart_engine 采集循环

支持 RTSP/RTMP/本地文件，断线重连与降帧采样。
"""
import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FrameReader:
    """从视频源读取帧，按目标帧率节流并降采样。"""

    def __init__(
        self,
        source_url: str,
        target_fps: float,
        reconnect_delay: float = 2.0,
    ):
        self.source_url = source_url
        self.target_fps = max(target_fps, 0.1)
        self.reconnect_delay = reconnect_delay
        self.frame_interval = 1.0 / self.target_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._source_fps = self.target_fps
        self._width = 0
        self._height = 0
        self._last_frame_time = 0.0
        self._frame_counter = 0
        self._sample_step = 1

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def source_fps(self) -> float:
        return self._source_fps

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.source_url)
        if not self._cap.isOpened():
            logger.error("无法打开视频源: %s", self.source_url)
            return False

        self._source_fps = self._cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        if self._source_fps <= 0:
            self._source_fps = self.target_fps

        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._sample_step = max(1, int(round(self._source_fps / self.target_fps)))
        self._frame_counter = 0

        logger.info(
            "视频源: %s, %dx%d, 源帧率=%.1f, 目标=%.1f, 采样步长=%d",
            self.source_url, self._width, self._height,
            self._source_fps, self.target_fps, self._sample_step,
        )
        return True

    def _reconnect(self) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.warning("视频源断开，%.1fs 后重连: %s", self.reconnect_delay, self.source_url)
        time.sleep(self.reconnect_delay)
        return self.open()

    def read(self) -> Optional[np.ndarray]:
        now = time.time()
        if now - self._last_frame_time < self.frame_interval:
            time.sleep(max(0.001, self.frame_interval - (now - self._last_frame_time)))

        while True:
            if self._cap is None or not self._cap.isOpened():
                if not self._reconnect():
                    return None

            ret, frame = self._cap.read()
            if not ret or frame is None:
                if not self._reconnect():
                    return None
                continue

            self._frame_counter += 1
            if self._frame_counter % self._sample_step != 0:
                continue

            self._last_frame_time = time.time()
            return frame

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def get_resolution(self) -> Tuple[int, int]:
        return self._width, self._height
