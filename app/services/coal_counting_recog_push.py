"""煤安平台人员计数识别数据推送（含证据图片）。

接口：``POST /mine/ai/countingRecog``（MT/T 1201.6-2023）
- 请求体为 JSON 数组，一次可推送多条；
- ``analysisType``：``04``=非常规通道入井，``05``=非常规通道出井，
  ``06``=入井闸机出闸，``07``=出井闸机入闸，``08``=无轨胶轮车上车点上下车人数识别；
- ``analysisCase``：识别结果，4 位补零（``0001``=1 人、``0002``=2 人……）；
- ``imagesBase64`` 可空，图片规则同 6.3（见 ``coal_image_upload``）；
- ``cameraCode`` 必须是摄像仪基本信息中的有效摄像仪（默认按本地 Camera 表校验）。

注：本模块走煤安平台 ``/mine/ai/countingRecog``（数组 + 图片）。
历史遗留的 ``stream_alert.push_counting_recog_upload`` 走的是
``COUNTING_RECOG_UPLOAD_URL``（单对象、无图片），两者互不影响。

鉴权与失败处理同其它推送模块：401/403 自动刷新 token 重试，失败只记日志。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.services.coal_image_upload import build_local_files, normalize_images
from app.services.coal_push_common import (
    apply_mine_code,
    build_result,
    camera_code_valid,
    cfg_bool,
    cfg_float,
    cfg_str,
    default_mine_code,
    load_valid_camera_codes,
    now_text,
    post_json_array,
)

logger = logging.getLogger(__name__)

_LABEL = "人员计数识别推送"

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

ANALYSIS_TYPE_UNUSUAL_IN = "04"        # 非常规通道入井
ANALYSIS_TYPE_UNUSUAL_OUT = "05"       # 非常规通道出井
ANALYSIS_TYPE_GATE_OUT = "06"          # 入井闸机出闸
ANALYSIS_TYPE_GATE_IN = "07"           # 出井闸机入闸
ANALYSIS_TYPE_VEHICLE = "08"           # 无轨胶轮车上车点上下车人数识别

VALID_ANALYSIS_TYPES: Set[str] = {
    ANALYSIS_TYPE_UNUSUAL_IN,
    ANALYSIS_TYPE_UNUSUAL_OUT,
    ANALYSIS_TYPE_GATE_OUT,
    ANALYSIS_TYPE_GATE_IN,
    ANALYSIS_TYPE_VEHICLE,
}

ANALYSIS_TYPE_LABELS = {
    ANALYSIS_TYPE_UNUSUAL_IN: "非常规通道入井",
    ANALYSIS_TYPE_UNUSUAL_OUT: "非常规通道出井",
    ANALYSIS_TYPE_GATE_OUT: "入井闸机出闸",
    ANALYSIS_TYPE_GATE_IN: "出井闸机入闸",
    ANALYSIS_TYPE_VEHICLE: "无轨胶轮车上车点上下车人数",
}

MAX_PERSON_COUNT = 9999


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

def _push_enabled() -> bool:
    return cfg_bool("COAL_COUNTING_RECOG_PUSH_ENABLED", True)


def _push_url() -> str:
    return cfg_str("COAL_COUNTING_RECOG_PUSH_URL")


def _push_timeout() -> float:
    return cfg_float("COAL_COUNTING_RECOG_PUSH_TIMEOUT", 30.0)


def _image_prefix() -> str:
    return cfg_str("COAL_COUNTING_RECOG_IMAGE_PREFIX", "coal/counting_recog")


# ---------------------------------------------------------------------------
# analysisCase 规则：人数 -> 4 位补零
# ---------------------------------------------------------------------------

def format_analysis_case(person_count: Any) -> Optional[str]:
    """人数 -> ``0001`` 形式的 analysisCase；非法返回 None。"""
    try:
        count = int(person_count)
    except (TypeError, ValueError):
        return None
    if count <= 0:
        return None
    if count > MAX_PERSON_COUNT:
        logger.warning(
            "人员计数识别推送：人数 %s 超出 4 位上限，按 %s 提交", count, MAX_PERSON_COUNT
        )
        count = MAX_PERSON_COUNT
    return f"{count:04d}"


# ---------------------------------------------------------------------------
# 构造 payload
# ---------------------------------------------------------------------------

def build_counting_recog_item(
    *,
    camera_code: str,
    analysis_type: Any,
    analysis_case: Any,
    data_time: Optional[str] = None,
    images_base64: Optional[Sequence[str]] = None,
    local_files: Optional[Sequence[str]] = None,
    mine_code: Optional[str] = None,
) -> Dict[str, Any]:
    """构造单条人员计数识别推送字段（camelCase）。"""
    code = str(camera_code or "").strip()

    atype = str(analysis_type or "").strip()
    if atype and atype not in VALID_ANALYSIS_TYPES:
        logger.warning(
            "人员计数识别推送：analysisType=%s 不在枚举内（仍按原值提交）", atype
        )

    case = str(analysis_case or "").strip()

    dt = str(data_time or "").strip() or now_text()

    item: Dict[str, Any] = {
        "cameraCode": code,
        "analysisType": atype,
        "analysisCase": case,
        "dataTime": dt,
    }

    imgs = normalize_images(images_base64)
    if imgs:
        item["imagesBase64"] = imgs

    # imagesBase64 为空时跳过图片上传，localFiles 为空数组
    files = [str(u).strip() for u in (local_files or []) if str(u or "").strip()]
    item["localFiles"] = files

    apply_mine_code(item, mine_code)
    return item


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def push_counting_recog_records(
    records: Iterable[Dict[str, Any]],
    *,
    valid_camera_codes: Optional[Set[str]] = None,
    upload_images: bool = True,
) -> Optional[Dict[str, Any]]:
    """批量推送人员计数识别记录（含证据图片）。

    ``records`` 每项字段：cameraCode / analysisType(04~08) / analysisCase /
    dataTime(可选) / imagesBase64(可选) / mineCode(可选)。
    失败只记日志，不抛异常。
    """
    if not _push_enabled():
        logger.info("跳过人员计数识别推送：COAL_COUNTING_RECOG_PUSH_ENABLED=false")
        return None

    url = _push_url()
    if not url:
        logger.warning("跳过人员计数识别推送：COAL_COUNTING_RECOG_PUSH_URL 未配置")
        return None

    raw_items = [it for it in (records or []) if isinstance(it, dict) and it]
    if not raw_items:
        logger.warning("跳过人员计数识别推送：无有效记录")
        return None

    if valid_camera_codes is None:
        valid_camera_codes = load_valid_camera_codes()

    payload: List[Dict[str, Any]] = []
    skipped = 0
    uploaded = 0
    for it in raw_items:
        camera_code = str(it.get("cameraCode") or it.get("camera_code") or "").strip()
        if not camera_code:
            skipped += 1
            logger.warning("跳过一条人员计数识别推送：cameraCode 为空")
            continue
        if not camera_code_valid(camera_code, valid_camera_codes):
            skipped += 1
            logger.warning(
                "跳过一条人员计数识别推送：cameraCode=%s 不在摄像仪基本信息中", camera_code
            )
            continue

        analysis_type = str(
            it.get("analysisType") or it.get("analysis_type") or ""
        ).strip()
        if not analysis_type:
            skipped += 1
            logger.warning(
                "跳过一条人员计数识别推送：analysisType 为空 cameraCode=%s", camera_code
            )
            continue

        analysis_case = it.get("analysisCase")
        if analysis_case is None:
            analysis_case = it.get("analysis_case")
        case = str(analysis_case or "").strip()
        if not case:
            skipped += 1
            logger.warning(
                "跳过一条人员计数识别推送：analysisCase 为空 cameraCode=%s", camera_code
            )
            continue

        data_time = str(it.get("dataTime") or it.get("data_time") or "").strip()
        images = it.get("imagesBase64") or it.get("images_base64") or []

        local_files = it.get("localFiles") or it.get("local_files") or []
        if upload_images and not local_files:
            local_files = build_local_files(
                images,
                prefix=_image_prefix(),
                camera_code=camera_code,
                data_time=data_time,
            )
        uploaded += len(local_files)

        payload.append(
            build_counting_recog_item(
                camera_code=camera_code,
                analysis_type=analysis_type,
                analysis_case=case,
                data_time=data_time,
                images_base64=images,
                local_files=local_files,
                mine_code=it.get("mineCode") or it.get("mine_code"),
            )
        )

    if not payload:
        logger.warning(
            "跳过人员计数识别推送：无有效记录 skipped=%s（摄像仪校验=%s）",
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
            success_summary="cameraCodes=%s types=%s cases=%s imageCounts=%s"
            % (
                [it.get("cameraCode") for it in payload],
                [it.get("analysisType") for it in payload],
                [it.get("analysisCase") for it in payload],
                [len(it.get("imagesBase64") or []) for it in payload],
            ),
        )
    except Exception:
        logger.exception(
            "人员计数识别推送异常 count=%s cameraCodes=%s",
            len(payload),
            [it.get("cameraCode") for it in payload],
        )
        return None

    return build_result(body, pushed=len(payload), skipped=skipped, extra={"uploaded": uploaded})


def push_counting_recog(
    *,
    camera_code: str,
    analysis_type: Any,
    analysis_case: Any = None,
    person_count: Any = None,
    data_time: Optional[str] = None,
    images_base64: Optional[Sequence[str]] = None,
    mine_code: Optional[str] = None,
    valid_camera_codes: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """推送单条人员计数识别记录（便捷入口）。

    ``analysis_case`` 可直接给 ``"0002"``；也可给 ``person_count=2`` 自动补零。
    """
    case = str(analysis_case or "").strip() or format_analysis_case(person_count)
    if not case:
        logger.warning(
            "跳过人员计数识别推送：analysisCase/person_count 均为空 cameraCode=%s",
            camera_code,
        )
        return None
    return push_counting_recog_records(
        [
            {
                "cameraCode": camera_code,
                "analysisType": analysis_type,
                "analysisCase": case,
                "dataTime": data_time,
                "imagesBase64": images_base64,
                "mineCode": mine_code,
            }
        ],
        valid_camera_codes=valid_camera_codes,
    )
