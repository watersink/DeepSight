"""煤安平台井下人数不符数据推送。

接口：``POST /mine/ai/undergroundCount``（MT/T 1201.6-2023）
- 请求体为 JSON 数组，一次可推送多条；
- 无图片字段；``analysisCase`` 为研判分析人数，4 位补零（``0001``=1 人、``0025``=25 人）；
- ``cameraCode`` 必须是摄像仪基本信息中的有效摄像仪（默认按本地 Camera 表校验）。

服务端处理（文档 8.3，由平台侧完成，本项目只负责推送）：
- 每条数据均写入 ``mine_underground_count``；
- 调人员定位系统取当前井下定位卡数写入 ``ps_person_card_count``（接口未定，暂为 0）；
- 冗余规则：允许差距 = 两者较大值的 10%（向上取整）且不少于 10 人，
  超出即产生 ``analysis_type=12`` 报警；
- 比对结果经 SSE（事件名 ``undergroundCount``）推送前端。

鉴权与失败处理同其它推送模块：401/403 自动刷新 token 重试，失败只记日志。
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Set

from app.services.coal_push_common import (
    apply_mine_code,
    build_result,
    camera_code_valid,
    cfg_bool,
    cfg_float,
    cfg_str,
    load_valid_camera_codes,
    now_text,
    post_json_array,
)

logger = logging.getLogger(__name__)

_LABEL = "井下人数推送"

MAX_PERSON_COUNT = 9999


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _push_enabled() -> bool:
    return cfg_bool("COAL_UNDERGROUND_COUNT_PUSH_ENABLED", True)


def _push_url() -> str:
    return cfg_str("COAL_UNDERGROUND_COUNT_PUSH_URL")


def _push_timeout() -> float:
    return cfg_float("COAL_UNDERGROUND_COUNT_PUSH_TIMEOUT", 15.0)


# ---------------------------------------------------------------------------
# analysisCase / 冗余规则
# ---------------------------------------------------------------------------

def format_analysis_case(person_count: Any) -> Optional[str]:
    """研判分析人数 -> ``0025`` 形式的 analysisCase；非法返回 None。"""
    try:
        count = int(person_count)
    except (TypeError, ValueError):
        return None
    if count < 0:
        return None
    if count > MAX_PERSON_COUNT:
        logger.warning(
            "井下人数推送：人数 %s 超出 4 位上限，按 %s 提交", count, MAX_PERSON_COUNT
        )
        count = MAX_PERSON_COUNT
    return f"{count:04d}"


def parse_analysis_case(analysis_case: Any) -> Optional[int]:
    """``0025`` -> 25；非法返回 None。"""
    text = str(analysis_case or "").strip()
    if not text.isdigit():
        return None
    return int(text)


def redundant_allowance(ai_count: int, card_count: int) -> int:
    """服务端冗余规则（本地复算，便于自检/日志）：较大值的 10% 向上取整，且不少于 10。"""
    return max(math.ceil(max(ai_count, card_count) * 0.10), 10)


def is_count_mismatch(ai_count: int, card_count: int) -> bool:
    """差距是否超出冗余范围（超出即平台会产生 analysis_type=12 报警）。"""
    return abs(ai_count - card_count) > redundant_allowance(ai_count, card_count)


# ---------------------------------------------------------------------------
# 构造 payload
# ---------------------------------------------------------------------------

def build_underground_count_item(
    *,
    camera_code: str,
    analysis_case: Any,
    data_time: Optional[str] = None,
    mine_code: Optional[str] = None,
) -> Dict[str, Any]:
    """构造单条井下人数推送字段（camelCase）。"""
    item: Dict[str, Any] = {
        "cameraCode": str(camera_code or "").strip(),
        "analysisCase": str(analysis_case or "").strip(),
        "dataTime": str(data_time or "").strip() or now_text(),
    }
    apply_mine_code(item, mine_code)
    return item


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def push_underground_count_records(
    records: Iterable[Dict[str, Any]],
    *,
    valid_camera_codes: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """批量推送井下人数记录。

    ``records`` 每项字段：cameraCode / analysisCase（``0025`` 或人数）/ dataTime /
    mineCode（可选）。失败只记日志，不抛异常。
    """
    if not _push_enabled():
        logger.info("跳过井下人数推送：COAL_UNDERGROUND_COUNT_PUSH_ENABLED=false")
        return None

    url = _push_url()
    if not url:
        logger.warning("跳过井下人数推送：COAL_UNDERGROUND_COUNT_PUSH_URL 未配置")
        return None

    raw_items = [it for it in (records or []) if isinstance(it, dict) and it]
    if not raw_items:
        logger.warning("跳过井下人数推送：无有效记录")
        return None

    if valid_camera_codes is None:
        valid_camera_codes = load_valid_camera_codes()

    payload: List[Dict[str, Any]] = []
    skipped = 0
    for it in raw_items:
        camera_code = str(it.get("cameraCode") or it.get("camera_code") or "").strip()
        if not camera_code:
            skipped += 1
            logger.warning("跳过一条井下人数推送：cameraCode 为空")
            continue
        if not camera_code_valid(camera_code, valid_camera_codes):
            skipped += 1
            logger.warning(
                "跳过一条井下人数推送：cameraCode=%s 不在摄像仪基本信息中", camera_code
            )
            continue

        case = it.get("analysisCase")
        if case is None:
            case = it.get("analysis_case")
        if case is None:
            case = it.get("personCount") or it.get("person_count")
        case_text = str(case or "").strip()
        count = parse_analysis_case(case_text)
        if count is None:
            skipped += 1
            logger.warning(
                "跳过一条井下人数推送：analysisCase=%r 非法（应为 4 位人数编码）cameraCode=%s",
                case,
                camera_code,
            )
            continue

        payload.append(
            build_underground_count_item(
                camera_code=camera_code,
                analysis_case=f"{count:04d}",
                data_time=it.get("dataTime") or it.get("data_time"),
                mine_code=it.get("mineCode") or it.get("mine_code"),
            )
        )

    if not payload:
        logger.warning(
            "跳过井下人数推送：无有效记录 skipped=%s（摄像仪校验=%s）",
            skipped,
            "已启用" if valid_camera_codes is not None else "跳过",
        )
        return None

    try:
        body = post_json_array(
            url,
            payload,
            timeout=_push_timeout(),
            label=_LABEL,
            success_summary="cameraCodes=%s cases=%s"
            % (
                [it.get("cameraCode") for it in payload],
                [it.get("analysisCase") for it in payload],
            ),
        )
    except Exception:
        logger.exception(
            "井下人数推送异常 count=%s cameraCodes=%s",
            len(payload),
            [it.get("cameraCode") for it in payload],
        )
        return None

    return build_result(body, pushed=len(payload), skipped=skipped)


def push_underground_count(
    *,
    camera_code: str,
    person_count: Any = None,
    analysis_case: Any = None,
    data_time: Optional[str] = None,
    mine_code: Optional[str] = None,
    valid_camera_codes: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """推送单条井下人数（便捷入口）。

    可传 ``person_count=25`` 或 ``analysis_case="0025"``。
    """
    case = str(analysis_case or "").strip() or format_analysis_case(person_count)
    if not case:
        logger.warning(
            "跳过井下人数推送：analysisCase/person_count 均为空 cameraCode=%s", camera_code
        )
        return None
    return push_underground_count_records(
        [
            {
                "cameraCode": camera_code,
                "analysisCase": case,
                "dataTime": data_time,
                "mineCode": mine_code,
            }
        ],
        valid_camera_codes=valid_camera_codes,
    )
