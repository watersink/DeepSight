"""推送人员计数 / 井下人数 / 重要运输设备人数到煤安平台。

覆盖接口：
- ``countingRecog``        人员计数识别数据-推送（含证据图片）
- ``undergroundCount``     井下人数不符数据-推送
- ``importPersonCount``    重要运输设备实时人数-推送

用法::

    cd DeepSight

    # 人员计数识别（含证据图片）
    python scripts/push_person_count_test.py counting --camera-code 34020000001320000001 \
        --analysis-type 04 --person-count 2

    # 井下人数不符
    python scripts/push_person_count_test.py underground --camera-code 34020000001320000001 \
        --person-count 25

    # 重要运输设备实时人数（同一 request 可推入井+出井两条）
    python scripts/push_person_count_test.py transport \
        --position-code 12345678901234 --position-name "斜井（架空乘人）" \
        --camera-code 34020000001320000001 --direction 01 --person-count 10

    # 只打印请求体，不推送
    python scripts/push_person_count_test.py counting --person-count 3 --dry-run
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import settings  # noqa: E402
from app.services import coal_counting_recog_push as cr  # noqa: E402
from app.services import coal_import_person_count_push as ip  # noqa: E402
from app.services import coal_underground_count_push as uc  # noqa: E402
from app.services.coal_push_common import default_mine_code  # noqa: E402

DEFAULT_IMAGE = ROOT / "test_images_videos" / "video_anomaly_blocked_demo.png"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="人员计数 / 井下人数 / 运输设备人数推送")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--camera-code", default="34020000001320000001")
        sp.add_argument("--person-count", type=int, default=None, help="人数（自动补零）")
        sp.add_argument("--data-time", default="")
        sp.add_argument("--mine-code", default="")
        sp.add_argument("--dry-run", action="store_true")
        sp.add_argument("--skip-camera-check", action="store_true")

    def add_list_mode(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--list4",
            action="store_true",
            help="一次请求推送 4 条（列表形式），各条按测试矩阵取不同取值",
        )
        sp.add_argument(
            "--camera-codes",
            default="34020000001320000001,34020000001320000001,34020000001320000004,34020000001320000005",
            help="--list4 使用的摄像仪编码（逗号分隔，按序分配给 4 条）",
        )

    sp1 = sub.add_parser("counting", help="人员计数识别 /mine/ai/countingRecog")
    common(sp1)
    add_list_mode(sp1)
    sp1.add_argument("--analysis-type", default="04", help="04/05/06/07/08")
    sp1.add_argument("--analysis-case", default="", help="如 0002；与 --person-count 二选一")
    sp1.add_argument("--image", default=str(DEFAULT_IMAGE), help="证据图片路径")
    sp1.add_argument("--no-image", action="store_true")

    sp2 = sub.add_parser("underground", help="井下人数 /mine/ai/undergroundCount")
    common(sp2)
    add_list_mode(sp2)
    sp2.add_argument("--analysis-case", default="", help="如 0025")

    sp3 = sub.add_parser("transport", help="运输设备人数 /mine/ai/importPersonCount")
    common(sp3)
    add_list_mode(sp3)
    sp3.add_argument("--position-code", default="12345678901234")
    sp3.add_argument("--position-name", default="斜井（架空乘人）")
    sp3.add_argument("--direction", default="01", help="01=入井，02=出井")
    sp3.add_argument("--both-directions", action="store_true", help="同请求推入井+出井两条")
    sp3.add_argument(
        "--position-codes",
        default="12345678901234,12345678901234,12345678901235,12345678901235",
        help="--list4 使用的点位编码（逗号分隔，按序分配给 4 条）",
    )
    sp3.add_argument(
        "--position-names",
        default="斜井（架空乘人）,斜井（架空乘人）,副井口（罐笼）,副井口（罐笼）",
        help="--list4 使用的点位名称（逗号分隔）",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# 列表模式：4 条测试数据矩阵
# ---------------------------------------------------------------------------

_LIST_CAMERA_CODES_FALLBACK = (
    "34020000001320000001",
    "34020000001320000001",
    "34020000001320000004",
    "34020000001320000005",
)


def _split(text: str, default: tuple) -> list:
    items = [s.strip() for s in str(text or "").split(",") if s.strip()]
    return items or list(default)


def _pick(seq: list, index: int) -> str:
    return seq[index % len(seq)]


def _base_time(data_time: str) -> datetime:
    if data_time:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(data_time, fmt)
            except ValueError:
                continue
    return datetime.now()


def build_list4_records(args: argparse.Namespace, images: list) -> tuple:
    """构造 4 条列表数据；返回 ``(records, 展示用摘要行)``。"""
    codes = _split(getattr(args, "camera_codes", ""), _LIST_CAMERA_CODES_FALLBACK)
    mine_code = args.mine_code or None
    base = _base_time(args.data_time)
    times = [(base.replace(microsecond=0) + timedelta(seconds=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(4)]
    records, rows = [], []

    if args.cmd == "counting":
        matrix = [("04", "0002", "非常规通道入井"), ("05", "0001", "非常规通道出井"),
                  ("06", "0003", "入井闸机出闸"), ("08", "0007", "无轨胶轮车上下车")]
        for i, (atype, case, label) in enumerate(matrix):
            rec = {
                "cameraCode": _pick(codes, i),
                "analysisType": atype,
                "analysisCase": case,
                "dataTime": times[i],
                "mineCode": mine_code,
            }
            if images:
                rec["imagesBase64"] = images
            records.append(rec)
            rows.append(f"  {i+1}. {rec['cameraCode']}  type={atype} case={case} ({label})  图片={len(images)} 张")

    elif args.cmd == "underground":
        matrix = [("0025", 25), ("0018", 18), ("0009", 9), ("0032", 32)]
        for i, (case, count) in enumerate(matrix):
            records.append({
                "cameraCode": _pick(codes, i),
                "analysisCase": case,
                "dataTime": times[i],
                "mineCode": mine_code,
            })
            rows.append(f"  {i+1}. {_pick(codes, i)}  研判人数={count} case={case}")

    else:  # transport
        pcodes = _split(getattr(args, "position_codes", ""), ("12345678901234",) * 2 + ("12345678901235",) * 2)
        pnames = _split(getattr(args, "position_names", ""), ("斜井（架空乘人）",) * 2 + ("副井口（罐笼）",) * 2)
        matrix = [(pcodes[0] if len(pcodes) > 0 else "12345678901234", pnames[0], "01", 10),
                  (pcodes[1] if len(pcodes) > 1 else "12345678901234", pnames[1], "02", 8),
                  (pcodes[2] if len(pcodes) > 2 else "12345678901235", pnames[2], "01", 14),
                  (pcodes[3] if len(pcodes) > 3 else "12345678901235", pnames[3], "02", 12)]
        for i, (pcode, pname, direction, count) in enumerate(matrix):
            records.append({
                "positionName": pname,
                "positionCode": pcode,
                "cameraCode": _pick(codes, i),
                "direction": direction,
                "personCount": count,
                "mineCode": mine_code,
            })
            rows.append(f"  {i+1}. {pname} {pcode}  camera={_pick(codes, i)}  direction={direction} personCount={count}")

    return records, rows


def main() -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    args = parse_args()
    data_time = args.data_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mine_code = args.mine_code or None
    valid = {args.camera_code} if args.skip_camera_check else None

    # ------------------------------------------------------------------
    # 列表模式：一次请求推送 4 条
    # ------------------------------------------------------------------
    if getattr(args, "list4", False):
        images = []
        if args.cmd == "counting" and not args.no_image:
            img = Path(args.image)
            if img.is_file():
                images = [f"data:image/png;base64,{base64.b64encode(img.read_bytes()).decode()}"]
                print(f"证据图片: {img} ({img.stat().st_size} 字节)")

        records, rows = build_list4_records(args, images)
        if args.skip_camera_check:
            valid = {r["cameraCode"] for r in records}
        url_map = {
            "counting": settings.COAL_COUNTING_RECOG_PUSH_URL,
            "underground": settings.COAL_UNDERGROUND_COUNT_PUSH_URL,
            "transport": settings.COAL_IMPORT_PERSON_COUNT_PUSH_URL,
        }
        print(f"\n一次请求推送 {len(records)} 条（JSON 数组）：")
        print("\n".join(rows))
        preview = [
            {k: (["<Base64>"] if k == "imagesBase64" else v) for k, v in r.items() if v is not None}
            for r in records
        ]
        print("\n请求体（列表形式）:")
        print(json.dumps(preview, ensure_ascii=False, indent=2))

        if args.dry_run:
            print("\n[dry-run] 未推送。")
            return 0

        print(f"\n推送地址: {url_map[args.cmd]}")
        if args.cmd == "counting":
            res = cr.push_counting_recog_records(records, valid_camera_codes=valid)
        elif args.cmd == "underground":
            res = uc.push_underground_count_records(records, valid_camera_codes=valid)
        else:
            res = ip.push_import_person_count_records(records, valid_camera_codes=valid)

        print("\n===== 推送结果 =====")
        print(json.dumps(res, ensure_ascii=False, indent=2) if res
              else "None（未推送或推送失败，详见上方日志）")
        return 0 if res and res.get("code") == 200 else 1

    if args.cmd == "counting":
        images = []
        if not args.no_image:
            img = Path(args.image)
            if img.is_file():
                b64 = base64.b64encode(img.read_bytes()).decode()
                images = [f"data:image/png;base64,{b64}"]
                print(f"证据图片: {img} ({img.stat().st_size} 字节)")
            else:
                print(f"证据图片不存在，跳过: {img}")
        case = args.analysis_case or cr.format_analysis_case(args.person_count)
        if not case:
            print("请指定 --person-count 或 --analysis-case")
            return 2
        preview = {
            "cameraCode": args.camera_code,
            "analysisType": args.analysis_type,
            "analysisCase": case,
            "dataTime": data_time,
        }
        if images:
            preview["imagesBase64"] = [f"<{len(images[0])} 字符 Base64>"]
        print("待推送:", json.dumps([preview], ensure_ascii=False))
        if args.dry_run:
            print("[dry-run] 未推送。")
            return 0
        print(f"推送地址: {settings.COAL_COUNTING_RECOG_PUSH_URL}")
        res = cr.push_counting_recog(
            camera_code=args.camera_code,
            analysis_type=args.analysis_type,
            analysis_case=case,
            data_time=data_time,
            images_base64=images,
            mine_code=mine_code,
            valid_camera_codes=valid,
        )

    elif args.cmd == "underground":
        case = args.analysis_case or uc.format_analysis_case(args.person_count)
        if case is None:
            print("请指定 --person-count 或 --analysis-case")
            return 2
        case = f"{int(case):04d}"
        print("待推送:", json.dumps(
            [{"cameraCode": args.camera_code, "analysisCase": case, "dataTime": data_time}],
            ensure_ascii=False))
        if args.dry_run:
            print("[dry-run] 未推送。")
            return 0
        print(f"推送地址: {settings.COAL_UNDERGROUND_COUNT_PUSH_URL}")
        res = uc.push_underground_count(
            camera_code=args.camera_code,
            person_count=args.person_count,
            analysis_case=case,
            data_time=data_time,
            mine_code=mine_code,
            valid_camera_codes=valid,
        )

    else:  # transport
        count = args.person_count if args.person_count is not None else 10
        records = [
            {
                "positionName": args.position_name,
                "positionCode": args.position_code,
                "cameraCode": args.camera_code,
                "direction": args.direction,
                "personCount": count,
                "mineCode": mine_code,
            }
        ]
        if args.both_directions:
            records.append({**records[0], "direction": "02", "personCount": max(count - 2, 0)})
        preview = []
        for r in records:
            item = {k: v for k, v in r.items() if v is not None}
            item["mineCode"] = r.get("mineCode") or default_mine_code()
            if not item["mineCode"]:
                item.pop("mineCode")
            preview.append(item)
        print("待推送:", json.dumps(preview, ensure_ascii=False))
        if args.dry_run:
            print("[dry-run] 未推送。")
            return 0
        print(f"推送地址: {settings.COAL_IMPORT_PERSON_COUNT_PUSH_URL}")
        res = ip.push_import_person_count_records(records, valid_camera_codes=valid)

    print("\n===== 推送结果 =====")
    print(json.dumps(res, ensure_ascii=False, indent=2) if res
          else "None（未推送或推送失败，详见上方日志）")
    return 0 if res and res.get("code") == 200 else 1


if __name__ == "__main__":
    raise SystemExit(main())
