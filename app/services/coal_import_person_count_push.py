"""煤安平台重要运输设备实时人数推送。

接口：``POST /mine/ai/importPersonCount``（MT/T 1201.6-2023）
推送重要运输设备（架空乘人装置、罐笼、无轨胶轮车、电机车、单轨吊等）监控点位的
实时人数，按 ``positionCode`` 全量替换并 SSE 实时推送前端大屏。一次可推送多条。

字段：
- ``positionName``   监控点位名称（必填）
- ``positionCode``   监控点位编码（参照附录 A.1 矿井位置编码，14 位，必填）
- ``cameraCode``     摄像仪编码（MT/T 1201.6-2023，必填）
- ``direction``      出入井类型：``01``=入井方向，``02``=出井方向（必填）
- ``personCount``    Integer 人数（必填）
- ``mineCode``       煤矿编码，12 位数字（可选，缺省取全局配置）

幂等策略：按 ``positionCode`` 先删除后新增；同一监控点位可在一个请求内推送
入井、出井两条（``direction`` 不同、``cameraCode`` 不同）。故本模块默认按
``positionCode + direction`` 去重（后者覆盖前者）。

鉴权与失败处理同其它推送模块：401/403 自动刷新 token 重试，失败只记日志。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Set

from app.services.coal_push_common import (
    apply_mine_code,
    build_result,
    camera_code_valid,
    cfg_bool,
    cfg_float,
    cfg_str,
    dedupe_by_key,
    load_valid_camera_codes,
    post_json_array,
)

logger = logging.getLogger(__name__)

_LABEL = "重要运输设备人数推送"

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DIRECTION_IN = "01"    # 入井方向
DIRECTION_OUT = "02"   # 出井方向

VALID_DIRECTIONS: Set[str] = {DIRECTION_IN, DIRECTION_OUT}
DIRECTION_LABELS = {DIRECTION_IN: "入井方向", DIRECTION_OUT: "出井方向"}

_POSITION_CODE_RE = re.compile(r"^\d{14}$")
_DEDUPE_FIELDS = ("positionCode", "direction")


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _push_enabled() -> bool:
    return cfg_bool("COAL_IMPORT_PERSON_COUNT_PUSH_ENABLED", True)


def _push_url() -> str:
    return cfg_str("COAL_IMPORT_PERSON_COUNT_PUSH_URL")


def _push_timeout() -> float:
    return cfg_float("COAL_IMPORT_PERSON_COUNT_PUSH_TIMEOUT", 15.0)


# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------

def normalize_direction(value: Any) -> Optional[str]:
    """规整出入井方向为 ``"01"`` / ``"02"``；非法返回 None。"""
    if value is None:
        return None
    text = str(value).strip()
    if text in VALID_DIRECTIONS:
        return text
    lowered = text.lower()
    if lowered in {"in", "inbound", "入", "入井", "入井方向"}:
        return DIRECTION_IN
    if lowered in {"out", "outbound", "出", "出井", "出井方向"}:
        return DIRECTION_OUT
    return None


def normalize_person_count(value: Any) -> Optional[int]:
    """规整人数为非负整数；非法返回 None。"""
    if isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    if count < 0:
        return None
    return count


# ---------------------------------------------------------------------------
# 构造 payload
# ---------------------------------------------------------------------------

def build_import_person_count_item(
    *,
    position_name: str,
    position_code: str,
    camera_code: str,
    direction: Any,
    person_count: Any,
    mine_code: Optional[str] = None,
) -> Dict[str, Any]:
    """构造单个监控点位的实时人数字段（camelCase）。"""
    dir_value = normalize_direction(direction)
    if dir_value is None:
        logger.warning(
            "重要运输设备人数推送：direction=%r 非法（应为 01/02），仍按原值提交",
            direction,
        )
        dir_value = str(direction or "").strip()

    count = normalize_person_count(person_count)
    if count is None:
        logger.warning(
            "重要运输设备人数推送：personCount=%r 非法（应为非负整数），仍按原值提交",
            person_count,
        )
        count = person_count

    item: Dict[str, Any] = {
        "positionName": str(position_name or "").strip(),
        "positionCode": str(position_code or "").strip(),
        "cameraCode": str(camera_code or "").strip(),
        "direction": dir_value,
        "personCount": count,
    }
    apply_mine_code(item, mine_code)
    return item


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def push_import_person_count_records(
    records: Iterable[Dict[str, Any]],
    *,
    valid_camera_codes: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """批量推送重要运输设备实时人数（按 positionCode 全量替换）。

    ``records`` 每项字段：positionName / positionCode / cameraCode / direction /
    personCount / mineCode(可选)。失败只记日志，不抛异常。
    """
    if not _push_enabled():
        logger.info(
            "跳过重要运输设备人数推送：COAL_IMPORT_PERSON_COUNT_PUSH_ENABLED=false"
        )
        return None

    url = _push_url()
    if not url:
        logger.warning(
            "跳过重要运输设备人数推送：COAL_IMPORT_PERSON_COUNT_PUSH_URL 未配置"
        )
        return None

    raw_items = [it for it in (records or []) if isinstance(it, dict) and it]
    if not raw_items:
        logger.warning("跳过重要运输设备人数推送：无有效记录")
        return None

    if valid_camera_codes is None:
        valid_camera_codes = load_valid_camera_codes()

    payload: List[Dict[str, Any]] = []
    skipped = 0
    for it in raw_items:
        def _pick(*names: str) -> Any:
            for name in names:
                value = it.get(name)
                if value is not None and str(value).strip() != "":
                    return value
            return None

        position_name = str(_pick("positionName", "position_name") or "").strip()
        position_code = str(_pick("positionCode", "position_code") or "").strip()
        camera_code = str(_pick("cameraCode", "camera_code") or "").strip()

        if not position_name:
            skipped += 1
            logger.warning(
                "跳过一条重要运输设备人数推送：positionName 为空 positionCode=%s",
                position_code or "(空)",
            )
            continue
        if not position_code:
            skipped += 1
            logger.warning(
                "跳过一条重要运输设备人数推送：positionCode 为空 positionName=%s",
                position_name,
            )
            continue
        if not _POSITION_CODE_RE.match(position_code):
            logger.warning(
                "重要运输设备人数推送：positionCode=%s 不是 14 位数字（仍按原值提交）",
                position_code,
            )
        if not camera_code:
            skipped += 1
            logger.warning(
                "跳过一条重要运输设备人数推送：cameraCode 为空 positionCode=%s",
                position_code,
            )
            continue
        if not camera_code_valid(camera_code, valid_camera_codes):
            skipped += 1
            logger.warning(
                "跳过一条重要运输设备人数推送：cameraCode=%s 不在摄像仪基本信息中",
                camera_code,
            )
            continue

        direction = _pick("direction")
        if normalize_direction(direction) is None:
            skipped += 1
            logger.warning(
                "跳过一条重要运输设备人数推送：direction=%r 非法（应为 01/02）positionCode=%s",
                direction,
                position_code,
            )
            continue

        person_count = _pick("personCount", "person_count")
        if normalize_person_count(person_count) is None:
            skipped += 1
            logger.warning(
                "跳过一条重要运输设备人数推送：personCount=%r 非法 positionCode=%s",
                person_count,
                position_code,
            )
            continue

        payload.append(
            build_import_person_count_item(
                position_name=position_name,
                position_code=position_code,
                camera_code=camera_code,
                direction=direction,
                person_count=person_count,
                mine_code=_pick("mineCode", "mine_code"),
            )
        )

    if not payload:
        logger.warning(
            "跳过重要运输设备人数推送：无有效记录 skipped=%s（摄像仪校验=%s）",
            skipped,
            "已启用" if valid_camera_codes is not None else "跳过",
        )
        return None

    # 幂等：同一 positionCode + direction 只保留最后一条
    payload = dedupe_by_key(payload, _DEDUPE_FIELDS, label=_LABEL)

    try:
        body = post_json_array(
            url,
            payload,
            timeout=_push_timeout(),
            label=_LABEL,
            success_summary="positionCodes=%s directions=%s counts=%s"
            % (
                [it.get("positionCode") for it in payload],
                [it.get("direction") for it in payload],
                [it.get("personCount") for it in payload],
            ),
        )
    except Exception:
        logger.exception(
            "重要运输设备人数推送异常 count=%s positionCodes=%s",
            len(payload),
            [it.get("positionCode") for it in payload],
        )
        return None

    return build_result(body, pushed=len(payload), skipped=skipped)


def push_import_person_count(
    *,
    position_code: str,
    camera_code: str,
    person_count: Any,
    position_name: str = "",
    direction: Any = DIRECTION_IN,
    mine_code: Optional[str] = None,
    valid_camera_codes: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """推送单个监控点位的实时人数（便捷入口）。"""
    return push_import_person_count_records(
        [
            {
                "positionName": position_name or position_code,
                "positionCode": position_code,
                "cameraCode": camera_code,
                "direction": direction,
                "personCount": person_count,
                "mineCode": mine_code,
            }
        ],
        valid_camera_codes=valid_camera_codes,
    )
