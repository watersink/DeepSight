"""FFmpeg 编码器配置 - 参考 smart_engine rtsp_streamer"""
import logging
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)

_NVENC_AVAILABLE: Optional[bool] = None


def check_nvenc_available() -> bool:
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if "h264_nvenc" not in result.stdout:
            _NVENC_AVAILABLE = False
            return False
        test = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=black:s=256x256:d=0.1",
                "-c:v", "h264_nvenc", "-frames:v", "1",
                "-f", "null", "-",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _NVENC_AVAILABLE = test.returncode == 0
    except Exception:
        _NVENC_AVAILABLE = False
    if _NVENC_AVAILABLE:
        logger.info("检测到 NVENC 硬件编码可用")
    else:
        logger.info("NVENC 不可用，将使用软件编码")
    return _NVENC_AVAILABLE


def get_nvenc_encoder_options(
    fps: float, bitrate: str = "2M", buffer_size: str = "4M"
) -> List[str]:
    return [
        "-pix_fmt", "yuv420p",
        "-c:v", "h264_nvenc",
        "-preset", "p1",
        "-tune", "ull",
        "-profile:v", "baseline",
        "-b:v", bitrate,
        "-maxrate", bitrate,
        "-bufsize", buffer_size,
        "-g", str(int(fps)),
        "-bf", "0",
    ]


def get_libx264_encoder_options(
    fps: float,
    crf: int = 23,
    bitrate: str = "1M",
    buffer_size: str = "2M",
) -> List[str]:
    return [
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-crf", str(crf),
        "-profile:v", "baseline",
        "-level", "3.1",
        "-maxrate", bitrate,
        "-bufsize", buffer_size,
        "-g", str(int(fps)),
        "-bf", "0",
    ]


def get_encoder_options(
    fps: float,
    use_hardware: bool,
    bitrate: str = "2M",
    buffer_size: str = "2M",
) -> List[str]:
    if use_hardware and check_nvenc_available():
        return get_nvenc_encoder_options(fps, bitrate, buffer_size)
    return get_libx264_encoder_options(fps, bitrate=bitrate, buffer_size=buffer_size)
