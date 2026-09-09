"""基于 ZLMediaKit startRecordTask 的事件视频截取与截图服务"""
import base64
import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2

from app.core.config import settings
from app.services.zlm_client import ZLMClient, ZLMClientError, zlm_client

logger = logging.getLogger(__name__)

_STREAM_SUFFIX_RE = re.compile(
    r"(\.live\.flv|\.flv|\.m3u8|\.mp4|\.ts|\.fmp4)$",
    re.IGNORECASE,
)


@dataclass
class VideoClipResult:
    app: str
    stream: str
    vhost: str
    record_rel_path: str
    video_server_path: str
    video_url: str
    video_base64: str
    image_base64: str
    image_content_type: str
    back_ms: int
    forward_ms: int
    duration_sec: Optional[float]


class VideoClipService:
    def __init__(self, client: Optional[ZLMClient] = None):
        self.client = client or zlm_client

    @staticmethod
    def parse_stream_url(video_url: str) -> Tuple[str, str]:
        """
        从流地址解析 ZLMediaKit 的 app / stream。

        支持示例：
        - rtmp://host:1935/scene_001/person_count_detector
        - rtsp://host:8554/scene_001/person_count_detector
        - http://host:8080/scene_001/person_count_detector.live.flv
        - http://host:8080/scene_001/person_count_detector/hls.m3u8
        """
        url = (video_url or "").strip()
        if not url:
            raise ZLMClientError("video_url 不能为空")

        parsed = urlparse(url)
        if not parsed.scheme or not parsed.path:
            raise ZLMClientError(f"无法解析 video_url: {video_url}")

        parts = [p for p in parsed.path.split("/") if p]
        # 跳过 ZLM HTTP 播放前缀，如 /rtp/... 以外的常见 index
        if parts and parts[0] in {"index", "api"}:
            raise ZLMClientError(f"video_url 不是有效的流播放地址: {video_url}")

        if len(parts) < 2:
            raise ZLMClientError(
                f"video_url 路径需包含 app/stream，例如 "
                f"`rtmp://host/app/stream`，当前: {video_url}"
            )

        app = parts[0]
        stream = parts[1]

        # http://host/app/stream/hls.m3u8 或 .../stream/0.ts
        if len(parts) >= 3 and _STREAM_SUFFIX_RE.search(parts[-1] or ""):
            stream = parts[1]
        else:
            stream = _STREAM_SUFFIX_RE.sub("", stream)

        if not app or not stream:
            raise ZLMClientError(f"无法从 video_url 解析 app/stream: {video_url}")

        return app, stream

    def capture_event_clip(
        self,
        video_url: str,
        back_ms: int,
        forward_ms: int,
    ) -> VideoClipResult:
        app, stream = self.parse_stream_url(video_url)
        vhost = settings.ZLM_DEFAULT_VHOST
        rel_path = self._build_record_rel_path()

        logger.info(
            "事件截取 video_url=%s -> app=%s stream=%s back_ms=%s forward_ms=%s",
            video_url,
            app,
            stream,
            back_ms,
            forward_ms,
        )

        if not self.client.is_stream_online(app, stream, vhost):
            raise ZLMClientError(f"流不在线: {vhost}/{app}/{stream}（来自 {video_url}）")

        server_path = self.client.start_record_task(
            app=app,
            stream=stream,
            path=rel_path,
            back_ms=back_ms,
            forward_ms=forward_ms,
            vhost=vhost,
        )

        wait_ms = max(forward_ms, 0) + settings.ZLM_RECORD_WAIT_BUFFER_MS
        if wait_ms > 0:
            logger.info("等待事件录制完成 %d ms", wait_ms)
            time.sleep(wait_ms / 1000.0)

        self._wait_video_ready(server_path, app, stream, rel_path)
        video_bytes = self._load_video_bytes(server_path, app, stream, rel_path)

        duration_sec = self._probe_duration(video_bytes)
        image_bytes = self._extract_middle_frame(video_bytes, duration_sec)

        video_b64 = base64.b64encode(video_bytes).decode("ascii")
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        video_url = settings.build_record_http_url(app, stream, rel_path)

        return VideoClipResult(
            app=app,
            stream=stream,
            vhost=vhost,
            record_rel_path=rel_path,
            video_server_path=server_path,
            video_url=video_url,
            video_base64=video_b64,
            image_base64=image_b64,
            image_content_type="image/jpeg",
            back_ms=back_ms,
            forward_ms=forward_ms,
            duration_sec=duration_sec,
        )

    @staticmethod
    def _build_record_rel_path() -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return f"events/{stamp}_{uuid.uuid4().hex[:8]}.mp4"

    def _load_video_bytes(
        self,
        server_path: str,
        app: str,
        stream: str,
        rel_path: str,
    ) -> bytes:
        local_path = Path(server_path)
        if local_path.is_file():
            return local_path.read_bytes()

        video_url = settings.build_record_http_url(app, stream, rel_path)
        return self.client.download_bytes(video_url)

    def _wait_video_ready(
        self,
        server_path: str,
        app: str,
        stream: str,
        rel_path: str,
        timeout_sec: float = 30.0,
        poll_interval: float = 0.5,
    ) -> None:
        """轮询直到录像文件大小稳定（写入完成）。"""
        deadline = time.monotonic() + timeout_sec
        local_path = Path(server_path)
        last_size = -1
        stable_count = 0

        while time.monotonic() < deadline:
            size = None
            if local_path.is_file():
                size = local_path.stat().st_size
            else:
                try:
                    data = self.client.download_bytes(
                        settings.build_record_http_url(app, stream, rel_path),
                        timeout=5.0,
                    )
                    size = len(data)
                except ZLMClientError:
                    size = None

            if size and size > 0:
                if size == last_size:
                    stable_count += 1
                    if stable_count >= 2:
                        return
                else:
                    stable_count = 0
                    last_size = size

            time.sleep(poll_interval)

        raise ZLMClientError("等待录像文件就绪超时")

    @staticmethod
    def _probe_duration(video_bytes: bytes) -> Optional[float]:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    tmp_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (OSError, ValueError) as e:
            logger.warning("ffprobe 获取时长失败: %s", e)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return None

    @staticmethod
    def _extract_middle_frame(video_bytes: bytes, duration_sec: Optional[float]) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        try:
            if duration_sec and duration_sec > 0:
                timestamp = duration_sec / 2.0
                image_bytes = VideoClipService._ffmpeg_snapshot(tmp_path, timestamp)
                if image_bytes:
                    return image_bytes

            return VideoClipService._opencv_snapshot(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    @staticmethod
    def _ffmpeg_snapshot(video_path: str, timestamp_sec: float) -> Optional[bytes]:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as img_tmp:
            img_path = img_tmp.name
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{timestamp_sec:.3f}",
                    "-i",
                    video_path,
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    img_path,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                logger.warning("ffmpeg 截图失败: %s", result.stderr[-500:])
                return None
            return Path(img_path).read_bytes()
        except OSError as e:
            logger.warning("ffmpeg 不可用: %s", e)
            return None
        finally:
            try:
                os.unlink(img_path)
            except OSError:
                pass

    @staticmethod
    def _opencv_snapshot(video_path: str) -> bytes:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ZLMClientError("无法打开录像文件以截取图片")

        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ZLMClientError("截取视频中间帧失败")

            ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise ZLMClientError("JPEG 编码失败")
            return encoded.tobytes()
        finally:
            cap.release()


video_clip_service = VideoClipService()
