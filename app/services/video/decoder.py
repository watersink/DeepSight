"""FFmpeg 解码探测与命令构建（CUDA/NVDEC → FFmpeg 软解；QSV 仅在实测可用时启用）。"""
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
    """
    检测 QSV 是否能把帧下载为系统内存 BGR24（本项目 rawvideo 管道所需）。

    仅检测 -hwaccel qsv 不够：很多环境 lavfi/null 能过，但对真实流 + bgr24 会报
    Function not implemented。
    """
    global _QSV_AVAILABLE
    if _QSV_AVAILABLE is not None:
        return _QSV_AVAILABLE
    try:
        listed = _run_ffmpeg(["-hide_banner", "-hwaccels"], timeout=5)
        if "qsv" not in (listed.stdout or "").lower():
            _QSV_AVAILABLE = False
            return False
        # 与真实解码路径一致：qsv → hwdownload → bgr24
        test = _run_ffmpeg(
            [
                "-hide_banner",
                "-loglevel",
                "error",
                "-hwaccel",
                "qsv",
                "-hwaccel_output_format",
                "qsv",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=256x256:rate=1",
                "-an",
                "-sn",
                "-vf",
                "hwdownload,format=nv12,format=bgr24",
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ],
            timeout=15,
        )
        _QSV_AVAILABLE = test.returncode == 0
        if not _QSV_AVAILABLE:
            err = (test.stderr or test.stdout or "").strip()[:300]
            logger.info("QSV 无法输出 BGR24 raw，将跳过硬解(%s)", err or "probe failed")
    except Exception:
        _QSV_AVAILABLE = False
    if _QSV_AVAILABLE:
        logger.info("检测到 QSV 硬件解码可用（含 BGR24 下载）")
    else:
        logger.info("QSV 不可用或不适合 raw BGR 管道")
    return _QSV_AVAILABLE


def mark_decode_backend_unavailable(backend: str) -> None:
    """运行时失败后拉黑，避免重连死循环。"""
    global _CUDA_AVAILABLE, _QSV_AVAILABLE
    b = (backend or "").strip().lower()
    if b == "cuda":
        _CUDA_AVAILABLE = False
        logger.warning("已禁用 CUDA 硬解（运行时失败）")
    elif b == "qsv":
        _QSV_AVAILABLE = False
        logger.warning("已禁用 QSV 硬解（运行时失败）")


def list_ffmpeg_decode_backends(use_hardware: bool = True) -> List[str]:
    """
    返回按优先级排列的 FFmpeg 解码后端：
    cuda → qsv(仅探测通过) → soft（纯软解）
    """
    backends: List[str] = []
    if use_hardware:
        if check_cuda_hwaccel_available():
            backends.append("cuda")
        if check_qsv_hwaccel_available():
            backends.append("qsv")
    backends.append("soft")
    return backends


def select_hw_decode_backend(use_hardware: bool = True) -> Optional[str]:
    """兼容旧接口：返回第一个硬解后端，没有则 None（调用方应再用 soft）。"""
    for b in list_ffmpeg_decode_backends(use_hardware):
        if b in {"cuda", "qsv"}:
            return b
    return None


def build_input_latency_options(source_url: str) -> List[str]:
    """低延迟拉流参数（避免使用各版本 FFmpeg 差异大的选项）。"""
    url = (source_url or "").strip().lower()
    opts: List[str] = ["-fflags", "+nobuffer+genpts"]
    if url.startswith("rtsp://") or url.startswith("rtsps://"):
        opts.extend(["-rtsp_transport", "tcp"])
    return opts


def build_ffmpeg_decode_command(
    source_url: str,
    backend: Optional[str],
) -> List[str]:
    """
    构建输出 raw BGR24 的 FFmpeg 解码命令（stdout 管道）。
    backend: cuda / qsv / soft / None(等同 soft)
    """
    b = (backend or "soft").strip().lower()
    if b in {"", "none", "sw", "software"}:
        b = "soft"

    cmd: List[str] = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
    ]
    cmd.extend(build_input_latency_options(source_url))

    if b == "cuda":
        cmd.extend(["-hwaccel", "cuda"])
    elif b == "qsv":
        # 必须显式下载到系统内存，否则 rawvideo/bgr24 常报 Function not implemented
        cmd.extend(["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"])

    cmd.extend(["-i", source_url, "-map", "0:v:0", "-an", "-sn"])

    if b == "qsv":
        cmd.extend(["-vf", "hwdownload,format=nv12,format=bgr24"])

    cmd.extend(
        [
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
