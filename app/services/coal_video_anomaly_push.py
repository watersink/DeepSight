"""煤安平台视频质量异常数据推送（含证据图片）。

接口：``POST /mine/ai/videoAnomaly``（MT/T 1201.6-2023）
- 请求体为 JSON 数组，一次可推送多条；
- ``analysisType`` 固定为 ``03``（视频质量异常）；
- ``cameraCode`` 必须是摄像仪基本信息中的有效摄像仪（默认按本地 Camera 表校验）；
- ``imagesBase64`` 可空，单张支持纯 Base64 或 ``data:image/xxx;base64,`` 前缀；
- 服务端将 Base64 解码后逐张上传至文件服务（本项目复用 MinIO），
  返回的 URL 以 JSON 数组写入 ``local_files``；
- 单张图片上传失败只记录日志，不影响本条记录入库。

鉴权：Authorization 每次经 ``app.auth.client_token`` 获取（401/403 自动刷新重试）。
所有失败只记日志、不抛异常，避免阻断上层业务流程。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import httpx

from app.auth.client_token import ClientTokenError, get_auth_header
from app.core.config import settings
from app.services.coal_image_upload import (
    build_local_files as _build_local_files,
    decode_base64_image as _decode_base64_image,
    normalize_images as _normalize_images,
    upload_image as _upload_image,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

ANALYSIS_TYPE_VIDEO_ANOMALY = "03"

ANALYSIS_CASE_SCREEN_BLOCKED = "0001"        # 画面遮挡
ANALYSIS_CASE_CAMERA_MOVED = "0002"          # 摄像仪挪动
ANALYSIS_CASE_TOO_DARK = "0003"              # 画面过暗
ANALYSIS_CASE_OVEREXPOSED = "0004"           # 画面过曝
ANALYSIS_CASE_BLURRY = "0005"                # 图像模糊
ANALYSIS_CASE_FROZEN = "0006"                # 画面冻结
ANALYSIS_CASE_VIDEO_LOST = "0007"            # 视频丢失
ANALYSIS_CASE_SHAKY = "0008"                 # 画面抖动
ANALYSIS_CASE_RESOLUTION_ABNORMAL = "0009"   # 分辨率异常

VALID_ANALYSIS_CASES: Set[str] = {
    ANALYSIS_CASE_SCREEN_BLOCKED,
    ANALYSIS_CASE_CAMERA_MOVED,
    ANALYSIS_CASE_TOO_DARK,
    ANALYSIS_CASE_OVEREXPOSED,
    ANALYSIS_CASE_BLURRY,
    ANALYSIS_CASE_FROZEN,
    ANALYSIS_CASE_VIDEO_LOST,
    ANALYSIS_CASE_SHAKY,
    ANALYSIS_CASE_RESOLUTION_ABNORMAL,
}

_MINE_CODE_RE = re.compile(r"^\d{12}$")


# ---------------------------------------------------------------------------
# 配置读取
# ---------------------------------------------------------------------------

def _cfg(name: str, default: Any = None) -> Any:
    return getattr(settings, name, default)


def _push_enabled() -> bool:
    return bool(_cfg("COAL_VIDEO_ANOMALY_PUSH_ENABLED", True))


def _push_url() -> str:
    return str(_cfg("COAL_VIDEO_ANOMALY_PUSH_URL", "") or "").strip()


def _push_timeout() -> float:
    try:
        return float(_cfg("COAL_VIDEO_ANOMALY_PUSH_TIMEOUT", 30.0) or 30.0)
    except (TypeError, ValueError):
        return 30.0


def _image_prefix() -> str:
    prefix = str(_cfg("COAL_VIDEO_ANOMALY_IMAGE_PREFIX", "") or "").strip()
    return (prefix or "coal/video_anomaly").strip("/")


def _default_mine_code() -> str:
    for name in (
        "MINE_CODE",
        "COUNTING_RECOG_MINE_CODE",
        "COAL_VIDEO_ANOMALY_MINE_CODE",
    ):
        value = str(_cfg(name, "") or "").strip()
        if value:
            return value
    return ""


# ---------------------------------------------------------------------------
# 图片处理：Base64 解码 -> 上传文件服务 -> localFiles
# （通用逻辑见 coal_image_upload，此处保留同名薄封装以兼容既有调用）
# ---------------------------------------------------------------------------

def decode_base64_image(raw: Any) -> Optional[tuple[bytes, str, str]]:
    """将单张图片 Base64 解码为 ``(bytes, ext, mime)``；失败返回 None。"""
    return _decode_base64_image(raw)


def upload_anomaly_image(
    raw: Any,
    *,
    camera_code: str,
    data_time: str,
) -> Optional[str]:
    """解码并上传单张证据图片，返回可访问 URL；失败只记日志并返回 None。"""
    return _upload_image(
        raw,
        prefix=_image_prefix(),
        camera_code=camera_code,
        data_time=data_time,
    )


def build_local_files(
    images_base64: Optional[Sequence[str]],
    *,
    camera_code: str,
    data_time: str,
) -> List[str]:
    """逐张上传图片，返回文件服务 URL 数组（顺序与入参一致，失败项丢弃）。"""
    urls: List[str] = []
    for raw in images_base64 or []:
        url = upload_anomaly_image(raw, camera_code=camera_code, data_time=data_time)
        if url:
            urls.append(url)
    return urls


# ---------------------------------------------------------------------------
# 构造 payload
# ---------------------------------------------------------------------------

def build_video_anomaly_item(
    *,
    camera_code: str,
    analysis_case: str,
    data_time: Optional[str] = None,
    images_base64: Optional[Sequence[str]] = None,
    local_files: Optional[Sequence[str]] = None,
    mine_code: Optional[str] = None,
) -> Dict[str, Any]:
    """构造单条视频质量异常推送字段（camelCase）。"""
    code = str(camera_code or "").strip()
    case = str(analysis_case or "").strip()
    if case and case not in VALID_ANALYSIS_CASES:
        logger.warning(
            "视频质量异常推送：analysisCase=%s 不在枚举内（仍按原值提交）", case
        )

    dt = str(data_time or "").strip() or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    item: Dict[str, Any] = {
        "cameraCode": code,
        "analysisType": ANALYSIS_TYPE_VIDEO_ANOMALY,
        "analysisCase": case,
        "dataTime": dt,
    }

    imgs = [str(i).strip() for i in (images_base64 or []) if str(i or "").strip()]
    if imgs:
        item["imagesBase64"] = imgs

    # imagesBase64 为空时跳过图片上传，localFiles 为空数组
    files = [str(u).strip() for u in (local_files or []) if str(u or "").strip()]
    item["localFiles"] = files

    mc = str(mine_code or "").strip() or _default_mine_code()
    if mc:
        if not _MINE_CODE_RE.match(mc):
            logger.warning(
                "视频质量异常推送：mineCode=%s 不是 12 位数字（仍按原值提交）", mc
            )
        item["mineCode"] = mc

    return item


# ---------------------------------------------------------------------------
# 摄像仪有效性校验（摄像仪基本信息）
# ---------------------------------------------------------------------------

def load_valid_camera_codes() -> Optional[Set[str]]:
    """读取本地摄像仪基本信息中的有效 cameraCode 集合。

    查询失败返回 None（表示无法校验，调用方跳过校验而不是全部丢弃）。
    """
    try:
        from app.db import SessionLocal
        from app.db.models import Camera

        db = SessionLocal()
        try:
            rows = db.query(Camera.camera_code).all()
        finally:
            db.close()
    except Exception:
        logger.exception("视频质量异常推送：读取摄像仪基本信息失败，跳过 cameraCode 校验")
        return None

    codes: Set[str] = set()
    for (code,) in rows:
        value = str(code or "").strip()
        if value:
            codes.add(value)
    return codes


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def push_video_anomaly_records(
    records: Iterable[Dict[str, Any]],
    *,
    valid_camera_codes: Optional[Set[str]] = None,
    upload_images: bool = True,
) -> Optional[Dict[str, Any]]:
    """批量推送视频质量异常记录（含证据图片）。

    ``records`` 每项字段：cameraCode / analysisCase / dataTime /
    imagesBase64(可选) / mineCode(可选)。失败只记日志，不抛异常。

    成功返回 ``{"code": ..., "msg": ..., "pushed": n, "uploaded": m, "skipped": k}``；
    未启用或未配置 URL 时返回 None。
    """
    if not _push_enabled():
        logger.info("跳过视频质量异常推送：COAL_VIDEO_ANOMALY_PUSH_ENABLED=false")
        return None

    url = _push_url()
    if not url:
        logger.warning("跳过视频质量异常推送：COAL_VIDEO_ANOMALY_PUSH_URL 未配置")
        return None

    raw_items = [it for it in (records or []) if isinstance(it, dict) and it]
    if not raw_items:
        logger.warning("跳过视频质量异常推送：无有效记录")
        return None

    # 摄像仪有效性校验：无效 cameraCode 的记录直接丢弃
    if valid_camera_codes is None:
        valid_camera_codes = load_valid_camera_codes()

    payload: List[Dict[str, Any]] = []
    skipped = 0
    uploaded = 0
    for it in raw_items:
        camera_code = str(it.get("cameraCode") or it.get("camera_code") or "").strip()
        if not camera_code:
            skipped += 1
            logger.warning("跳过一条视频质量异常推送：cameraCode 为空")
            continue
        if valid_camera_codes is not None and camera_code not in valid_camera_codes:
            skipped += 1
            logger.warning(
                "跳过一条视频质量异常推送：cameraCode=%s 不在摄像仪基本信息中",
                camera_code,
            )
            continue

        analysis_case = str(
            it.get("analysisCase") or it.get("analysis_case") or ""
        ).strip()
        if not analysis_case:
            skipped += 1
            logger.warning(
                "跳过一条视频质量异常推送：analysisCase 为空 cameraCode=%s",
                camera_code,
            )
            continue

        data_time = str(it.get("dataTime") or it.get("data_time") or "").strip()
        images = it.get("imagesBase64") or it.get("images_base64") or []

        # localFiles 优先取外部传入，否则由本地图片上传结果填充
        local_files = it.get("localFiles") or it.get("local_files") or []
        if upload_images and not local_files:
            local_files = build_local_files(
                images, camera_code=camera_code, data_time=data_time
            )
        uploaded += len(local_files)

        payload.append(
            build_video_anomaly_item(
                camera_code=camera_code,
                analysis_case=analysis_case,
                data_time=data_time,
                images_base64=images,
                local_files=local_files,
                mine_code=it.get("mineCode") or it.get("mine_code"),
            )
        )

    if not payload:
        logger.warning(
            "跳过视频质量异常推送：无有效记录 skipped=%s（摄像仪校验=%s）",
            skipped,
            "已启用" if valid_camera_codes is not None else "跳过",
        )
        return None

    try:
        body = _post_video_anomaly_payload(url, payload, timeout=_push_timeout())
    except Exception:
        logger.exception(
            "视频质量异常推送异常 count=%s cameraCodes=%s",
            len(payload),
            [it.get("cameraCode") for it in payload],
        )
        return None

    result: Dict[str, Any] = {
        "code": (body or {}).get("code"),
        "msg": (body or {}).get("msg"),
        "pushed": len(payload),
        "uploaded": uploaded,
        "skipped": skipped,
    }
    if body is not None:
        result["body"] = body
    return result


def push_video_anomaly(
    *,
    camera_code: str,
    analysis_case: str,
    data_time: Optional[str] = None,
    images_base64: Optional[Sequence[str]] = None,
    mine_code: Optional[str] = None,
    valid_camera_codes: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """推送单条视频质量异常记录（便捷入口）。"""
    return push_video_anomaly_records(
        [
            {
                "cameraCode": camera_code,
                "analysisCase": analysis_case,
                "dataTime": data_time,
                "imagesBase64": images_base64,
                "mineCode": mine_code,
            }
        ],
        valid_camera_codes=valid_camera_codes,
    )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _post_video_anomaly_payload(
    url: str,
    payload: List[Dict[str, Any]],
    *,
    timeout: float,
) -> Optional[Dict[str, Any]]:
    camera_codes = [it.get("cameraCode") for it in payload]

    def _do_request(*, force_token: bool) -> httpx.Response:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **get_auth_header(force=force_token),
        }
        auth = str(headers.get("Authorization") or "")
        logger.info(
            "视频质量异常推送 Authorization 前缀检查 starts_with_Bearer=%s "
            "auth_len=%s auth_preview=%r count=%s",
            auth.lower().startswith("bearer "),
            len(auth),
            (auth[:16] + "...") if len(auth) > 16 else auth,
            len(payload),
        )
        with httpx.Client(timeout=timeout) as client:
            return client.post(url, json=payload, headers=headers)

    try:
        # 每次推送前强制拉取最新 client_token（app.auth.client_token）
        resp = _do_request(force_token=True)
        # 若平台仍返回鉴权失败，再强制刷新一次后重试
        if resp.status_code in {401, 403}:
            logger.warning(
                "视频质量异常推送鉴权失败 status=%s，再次刷新 token 后重试 count=%s",
                resp.status_code,
                len(payload),
            )
            resp = _do_request(force_token=True)
    except ClientTokenError as exc:
        logger.error("视频质量异常推送失败：获取 client_token 失败 err=%s", exc)
        return None
    except httpx.RequestError as exc:
        logger.error("视频质量异常推送失败：无法连接 %s err=%s", url, exc)
        return None

    text = (resp.text or "")[:500]
    body: Optional[Dict[str, Any]] = None
    try:
        parsed = resp.json()
        if isinstance(parsed, dict):
            body = parsed
    except ValueError:
        body = None

    biz_code = body.get("code") if body else None
    biz_msg = body.get("msg") if body else None

    if resp.status_code >= 400 or (biz_code is not None and int(biz_code) != 200):
        hint = ""
        if biz_code is not None and int(biz_code) == 401:
            hint = (
                "（Authorization 已按 Bearer 提交；请向煤安平台确认该 "
                "client_id 是否已授权 /coal/mine/ai/videoAnomaly，"
                "或凭证/环境是否匹配）"
            )
        logger.error(
            "视频质量异常推送失败 count=%s cameraCodes=%s http=%s code=%s msg=%s%s body=%s",
            len(payload),
            camera_codes,
            resp.status_code,
            biz_code,
            biz_msg,
            hint,
            text,
        )
        return body

    logger.info(
        "视频质量异常推送成功 count=%s cameraCodes=%s cases=%s imageCounts=%s msg=%s",
        len(payload),
        camera_codes,
        [it.get("analysisCase") for it in payload],
        [len(it.get("imagesBase64") or []) for it in payload],
        biz_msg or "操作成功",
    )
    return body
