"""推送一条视频质量异常测试数据到煤安平台（POST /mine/ai/videoAnomaly）。

用法::

    cd DeepSight
    python scripts/push_video_anomaly_test.py                  # 真实推送（含证据图片）
    python scripts/push_video_anomaly_test.py --dry-run        # 只打印请求体，不推送
    python scripts/push_video_anomaly_test.py --no-image       # 只推数据，跳过图片上传
    python scripts/push_video_anomaly_test.py --camera-code 34020000001320000004 \
        --analysis-case 0003 --mine-code 123456789012

说明：
- 图片经 app.services.coal_video_anomaly_push 上传文件服务（MinIO），URL 写入 local_files；
- 未配置 COAL_OAUTH_CLIENT_ID / COAL_OAUTH_CLIENT_SECRET 时鉴权会失败（日志可见）。
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services import coal_video_anomaly_push as anomaly_push  # noqa: E402

DEFAULT_IMAGE = ROOT / "test_images_videos" / "video_anomaly_blocked_demo.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="推送一条视频质量异常测试数据")
    p.add_argument("--camera-code", default="34020000001320000001", help="摄像仪编码")
    p.add_argument("--analysis-case", default="0001", help="识别结果，默认 0001 画面遮挡")
    p.add_argument("--data-time", default="", help="数据生成时间，默认当前时间")
    p.add_argument("--mine-code", default="", help="煤矿编码（12 位）；默认取配置兜底")
    p.add_argument("--image", default=str(DEFAULT_IMAGE), help="证据图片路径；不存在则只用 Base64")
    p.add_argument("--no-image", action="store_true", help="不上传图片（localFiles 为空数组）")
    p.add_argument("--dry-run", action="store_true", help="只打印请求体，不真正推送")
    p.add_argument("--skip-camera-check", action="store_true", help="跳过摄像仪有效性校验")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    args = parse_args()

    data_time = args.data_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    images_base64 = []
    if not args.no_image:
        image_path = Path(args.image)
        if image_path.is_file():
            b64 = base64.b64encode(image_path.read_bytes()).decode()
            images_base64 = [f"data:image/png;base64,{b64}"]
            print(f"证据图片: {image_path} ({image_path.stat().st_size} 字节, base64 {len(b64)} 字符)")
        else:
            print(f"证据图片不存在，跳过图片: {image_path}")

    print("\n===== 待推送请求体（推送前占位）=====")
    preview = {
        "cameraCode": args.camera_code,
        "analysisType": anomaly_push.ANALYSIS_TYPE_VIDEO_ANOMALY,
        "analysisCase": args.analysis_case,
        "dataTime": data_time,
    }
    if images_base64:
        preview["imagesBase64"] = [images_base64[0][:48] + "...（Base64 已省略）"]
    if args.mine_code:
        preview["mineCode"] = args.mine_code
    print(json.dumps([preview], ensure_ascii=False, indent=2))

    if args.dry_run:
        print("\n[dry-run] 未真正推送。")
        return 0

    print(f"\n推送地址: {settings.COAL_VIDEO_ANOMALY_PUSH_URL}")
    result = anomaly_push.push_video_anomaly(
        camera_code=args.camera_code,
        analysis_case=args.analysis_case,
        data_time=data_time,
        images_base64=images_base64,
        mine_code=args.mine_code or None,
        valid_camera_codes=None if not args.skip_camera_check else {args.camera_code},
    )

    print("\n===== 推送结果 =====")
    print(json.dumps(result, ensure_ascii=False, indent=2) if result else "None（未推送或推送失败，详见上方日志）")
    return 0 if result and result.get("code") == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
