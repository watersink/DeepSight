"""
视频流处理共享模块

参考 smart_engine 架构，提供可在多个算法/任务间复用的组件：

- FrameReader          视频源读取（断线重连、降帧）
- FFmpegRTMPStreamer   RTMP 推流
- FFmpegRTSPStreamer   RTSP 推流
- AsyncFrameProcessor  检测/推流/告警三线程处理器
- VideoStreamPipeline  一站式管道编排
- StreamConfig         统一配置
"""
from app.services.video.types import (
    AlertCallback,
    AlertPredicate,
    StreamConfig,
    default_alert_predicate,
)
from app.services.video.frame_reader import FrameReader
from app.services.video.ffmpeg_streamer import (
    FFmpegStreamerBase,
    FFmpegRTMPStreamer,
    FFmpegRTSPStreamer,
    create_streamer,
    cleanup_all_streamers,
)
from app.services.video.async_processor import AsyncFrameProcessor
from app.services.video.pipeline import VideoStreamPipeline
from app.services.video.camera_ingest import CameraFanoutPipeline
from app.services.video.encoder import check_nvenc_available
from app.services.video.decoder import (
    check_cuda_hwaccel_available,
    check_qsv_hwaccel_available,
    list_ffmpeg_decode_backends,
    select_hw_decode_backend,
)

__all__ = [
    "AlertCallback",
    "AlertPredicate",
    "StreamConfig",
    "default_alert_predicate",
    "FrameReader",
    "FFmpegStreamerBase",
    "FFmpegRTMPStreamer",
    "FFmpegRTSPStreamer",
    "create_streamer",
    "cleanup_all_streamers",
    "AsyncFrameProcessor",
    "VideoStreamPipeline",
    "CameraFanoutPipeline",
    "check_nvenc_available",
    "check_cuda_hwaccel_available",
    "check_qsv_hwaccel_available",
    "list_ffmpeg_decode_backends",
    "select_hw_decode_backend",
]
