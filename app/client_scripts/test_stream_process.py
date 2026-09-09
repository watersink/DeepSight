"""
人数检测 RTMP 推流入口（命令行）

使用 app.services.video 共享模块，业务逻辑与视频流基础设施解耦。

运行（在 code 目录下）:
  python app/client_scripts/stream_process.py
  python app/client_scripts/stream_process.py --in_url ./test_images_videos/person_gouzi.mp4
"""
import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import pytz

# client_scripts -> app -> code
CODE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_ROOT))

from app.core.config import settings
from app.plugins.skill_registry import create_skill
from app.services.video import StreamConfig, VideoStreamPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _default_alert_handler(data, raw_frame, scene_id):
    """示例告警回调，可扩展为写库、上传 MinIO 等"""
    tz = pytz.timezone("Asia/Shanghai")
    ts = datetime.now(tz).strftime("%Y%m%d-%H:%M:%S.%f")
    pic_name = f"person_count/{scene_id}_{ts}.jpg"
    logger.info(
        "触发人数告警 scene=%s count=%d enter_count=%d pic=%s shape=%s "
        "process_time_ms=%.2f detect_time_ms=%.2f draw_time_ms=%.2f",
        scene_id,
        data.get("count", 0),
        data.get("enter_count", 0),
        pic_name,
        raw_frame.shape,
        float(data.get("process_time_ms", 0.0)),
        float(data.get("detect_time_ms", 0.0)),
        float(data.get("draw_time_ms", 0.0)),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="人数检测视频流推流")
    parser.add_argument(
        "--in_url",
        default=settings.resolve_test_video_path(CODE_ROOT),
        help="输入视频流地址",
    )
    parser.add_argument(
        "--out_url",
        default=None,
        help=(
            "输出推流地址（默认: "
            f"{settings.build_push_url()}"
            "，即 {scene_id}/{skill_name}）"
        ),
    )
    parser.add_argument("--out_fps", type=int, default=settings.DEFAULT_OUT_FPS, help="输出帧率")
    parser.add_argument(
        "--alarm_interval",
        type=int,
        default=settings.DEFAULT_ALARM_INTERVAL,
        help="告警间隔（秒）",
    )
    parser.add_argument("--scene_id", default=settings.DEFAULT_SCENE_ID, help="场景 ID")
    parser.add_argument(
        "--skill_name",
        default=settings.DEFAULT_SKILL_NAME,
        help="技能名称",
    )
    parser.add_argument(
        "--output_format",
        default=settings.DEFAULT_OUTPUT_FORMAT,
        choices=["rtmp", "rtsp"],
        help="推流协议",
    )
    parser.add_argument(
        "--queue_size",
        type=int,
        default=settings.STREAM_QUEUE_SIZE,
        help="帧缓冲队列大小",
    )
    parser.add_argument(
        "--no_hw_encode",
        action="store_true",
        help="禁用 NVENC 硬件编码，使用 libx264 软件编码",
    )
    parser.add_argument("--bitrate", default=settings.STREAM_BITRATE)
    parser.add_argument("--buffer_size", default=settings.STREAM_BUFFER_SIZE)
    parser.add_argument("--reconnect_delay", type=float, default=settings.STREAM_RECONNECT_DELAY)
    return parser.parse_args()


def main():
    args = parse_args()

    output_url = args.out_url or settings.build_push_url(
        args.output_format,
        scene_id=args.scene_id,
        skill_name=args.skill_name,
    )

    config = StreamConfig(
        source_url=args.in_url,
        output_url=output_url,
        fps=args.out_fps,
        alarm_interval=args.alarm_interval,
        scene_id=args.scene_id,
        queue_size=args.queue_size,
        output_format=args.output_format,
        use_hardware_encoding=not args.no_hw_encode,
        bitrate=args.bitrate,
        buffer_size=args.buffer_size,
        reconnect_delay=args.reconnect_delay,
    )

    skill = create_skill(args.skill_name)
    pipeline = VideoStreamPipeline(
        config=config,
        skill_instance=skill,
        on_alert=_default_alert_handler,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
