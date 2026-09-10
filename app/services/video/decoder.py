"""FFmpeg 硬件解码探测与命令构建（NVDEC/CUDA 优先，其次 QSV）。"""
from __future__ import annotations

import logging
import subprocess
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_CUDA_AVAILABLE: Optional[bool] = None
_QSV_AVAILABLE: Optional[bool] = None


def _run_ffmpeg(args: List[str], timeout: float = 12.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ffmpeg", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def check_cuda_hwaccel_available() -> bool:
    """检测 FFmpeg 是否可用 CUDA/NVDEC 硬解。"""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return _CUDA_AVAILABLE
    try:
        listed = _run_ffmpeg(["-hide_banner", "-hwaccels"], timeout=5)
        if "cuda" not in (listed.stdout or "").lower():
            _CUDA_AVAILABLE = False
            return False
        test = _run_ffmpeg(
            [
                "-hide_banner",
                "-loglevel",
                "error",
                "-hwaccel",
                "cuda",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=256x256:rate=1",
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            timeout=15,
        )
        _CUDA_AVAILABLE = test.returncode == 0
    except Exception:
        _CUDA_AVAILABLE = False
    if _CUDA_AVAILABLE:
        logger.info("检测到 CUDA/NVDEC 硬件解码可用")
    else:
        logger.info("CUDA/NVDEC 不可用")
    return _CUDA_AVAILABLE


def check_qsv_hwaccel_available() -> bool:
    """检测 FFmpeg 是否可用 Intel QSV 硬解。"""
    global _QSV_AVAILABLE
    if _QSV_AVAILABLE is not None:
        return _QSV_AVAILABLE
    try:
        listed = _run_ffmpeg(["-hide_banner", "-hwaccels"], timeout=5)
        if "qsv" not in (listed.stdout or "").lower():
            _QSV_AVAILABLE = False
            return False
        test = _run_ffmpeg(
            [
                "-hide_banner",
                "-loglevel",
                "error",
                "-hwaccel",
                "qsv",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=256x256:rate=1",
                "-frames:v",
                "1",
                "-f",
                "null",
                "-",
            ],
            timeout=15,
        )
        _QSV_AVAILABLE = test.returncode == 0
    except Exception:
        _QSV_AVAILABLE = False
    if _QSV_AVAILABLE:
        logger.info("检测到 QSV 硬件解码可用")
    else:
        logger.info("QSV 不可用")
    return _QSV_AVAILABLE


def select_hw_decode_backend(use_hardware: bool = True) -> Optional[str]:
    """
    选择硬解后端：优先 NVIDIA CUDA/NVDEC，其次 Intel QSV。
    返回 'cuda' | 'qsv' | None（软解）。
    """
    if not use_hardware:
        return None
    if check_cuda_hwaccel_available():
        return "cuda"
    if check_qsv_hwaccel_available():
        return "qsv"
    return None


def build_input_latency_options(source_url: str) -> List[str]:
    """低延迟拉流参数（避免使用各版本 FFmpeg 差异大的选项）。"""
    url = (source_url or "").strip().lower()
    # +nobuffer 兼容性优于裸 nobuffer；不使用已废弃/移除的 -vsync
    opts: List[str] = ["-fflags", "+nobuffer+genpts"]
    if url.startswith("rtsp://"):
        opts.extend(["-rtsp_transport", "tcp"])
    return opts


def build_ffmpeg_decode_command(
    source_url: str,
    backend: Optional[str],
) -> List[str]:
    """
    构建输出 raw BGR24 的 FFmpeg 解码命令（stdout 管道）。
    backend: cuda / qsv / None(软解)
    """
    cmd: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
    ]
    cmd.extend(build_input_latency_options(source_url))

    if backend == "cuda":
        cmd.extend(["-hwaccel", "cuda"])
    elif backend == "qsv":
        cmd.extend(["-hwaccel", "qsv"])

    cmd.extend(
        [
            "-i",
            source_url,
            "-an",
            "-sn",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "pipe:1",
        ]
    )
    return cmd


def probe_stream_meta(source_url: str) -> Tuple[int, int, float]:
    """
    用 OpenCV 探测分辨率与帧率（打开后立即释放）。
    失败返回 (0, 0, 0.0)。
    """
    import cv2

    cap = cv2.VideoCapture(source_url)
    if not cap.isOpened():
        return 0, 0, 0.0
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return width, height, fps
    finally:
        cap.release()
