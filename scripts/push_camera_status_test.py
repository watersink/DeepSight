"""推送摄像仪状态（在线/离线）到煤安平台（POST /mine/ai/cameraStatus）。

用法::

    cd DeepSight

    # 采集本机全部摄像仪在线状态（ZLM 判在线）并推送
    python scripts/push_camera_status_test.py --all

    # 指定摄像仪编码与状态推送
    python scripts/push_camera_status_test.py --camera-code 34020000001320000001 --status 1
    python scripts/push_camera_status_test.py --camera-code 34020000001320000001 --status 0

    # 多台一次推送（可重复 --camera-code，配合 --status 逐台指定）
    python scripts/push_camera_status_test.py \
        --camera-code 34020000001320000001=1 \
        --camera-code 34020000001320000004=0

    # 只打印请求体，不推送
    python scripts/push_camera_status_test.py --all --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services import coal_camera_status_push as status_push  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="推送摄像仪在线/离线状态")
    p.add_argument(
        "--camera-code",
        action="append",
        default=[],
        metavar="CODE[=0|1]",
        help="摄像仪编码，可带 =0/=1 指定该台状态；可重复传多次",
    )
    p.add_argument("--status", default=None, help="统一状态：1=在线，0=离线")
    p.add_argument("--all", action="store_true", help="采集本机全部摄像仪状态并推送")
    p.add_argument("--data-time", default="", help="数据生成时间，默认当前时间")
    p.add_argument("--mine-code", default="", help="煤矿编码（12 位）；默认取配置兜底")
    p.add_argument("--dry-run", action="store_true", help="只打印请求体，不真正推送")
    p.add_argument("--skip-camera-check", action="store_true", help="跳过摄像仪有效性校验")
    return p.parse_args()


def build_records(args: argparse.Namespace) -> list[dict]:
    """把命令行参数转成待推送记录（含采集模式）。"""
    if args.all:
        collected = status_push.collect_camera_statuses()
        if not collected:
            print("采集失败或没有可判定状态的摄像仪（详见日志）")
            return []
        print(f"采集到 {len(collected)} 台摄像仪状态：")
        for r in collected:
            label = "在线" if r["cameraStatus"] == status_push.CAMERA_STATUS_ONLINE else "离线"
            print(f"  {r['cameraCode']}  {r['cameraStatus']} ({label})  mineCode={r.get('mineCode') or '(兜底)'}")
        return collected

    records = []
    for raw in args.camera_code:
        code, _, inline_status = raw.partition("=")
        status = inline_status.strip() or (args.status or "")
        records.append({"cameraCode": code.strip(), "cameraStatus": status})
    return records


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    args = parse_args()

    if not args.all and not args.camera_code:
        print("请指定 --all，或用 --camera-code 指定摄像仪编码（详见 --help）")
        return 2

    records = build_records(args)
    if not records:
        return 1

    data_time = args.data_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("\n===== 待推送请求体 =====")
    preview = []
    for r in records:
        item = {
            "cameraCode": r["cameraCode"],
            "cameraStatus": status_push.normalize_camera_status(r.get("cameraStatus")) or r.get("cameraStatus"),
            "dataTime": data_time,
        }
        # mineCode 取"该台的"值 > 命令行显式值 > 全局兜底；
        # 采集模式下 r 里已带各摄像仪自己的 mineCode，优先用它，避免与
        # /mine/ai/camera 推送的 mineCode 不一致
        mc = (
            str(r.get("mineCode") or "").strip()
            or args.mine_code
            or status_push._default_mine_code()
        )
        if mc:
            item["mineCode"] = mc
        preview.append(item)
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    if args.dry_run:
        print("\n[dry-run] 未真正推送。")
        return 0

    print(f"\n推送地址: {settings.COAL_CAMERA_STATUS_PUSH_URL}")

    # 采集模式已自带有效 cameraCode，无需再查库校验
    valid = None
    if args.all:
        valid = {r["cameraCode"] for r in records}
    elif args.skip_camera_check:
        valid = {r["cameraCode"] for r in records}

    result = status_push.push_camera_statuses(
        [
            {
                "cameraCode": r["cameraCode"],
                "cameraStatus": r["cameraStatus"],
                "dataTime": data_time,
                # 显式指定才覆盖，否则由服务层取该台/兜底的 mineCode
                "mineCode": str(r.get("mineCode") or "").strip() or args.mine_code or None,
            }
            for r in records
        ],
        valid_camera_codes=valid,
    )

    print("\n===== 推送结果 =====")
    print(
        json.dumps(result, ensure_ascii=False, indent=2)
        if result
        else "None（未推送或推送失败，详见上方日志）"
    )
    return 0 if result and result.get("code") == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
