"""煤安平台摄像仪配置推送。

摄像头创建/更新成功后，将 MT/T 基本信息 POST 到
``COAL_CAMERA_PUSH_URL``；Authorization 每次经 ``app.auth.client_token`` 获取。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import httpx

from app.auth.client_token import ClientTokenError, get_auth_header
from app.core.config import settings
from app.db.models import Camera

logger = logging.getLogger(__name__)


def build_camera_push_item(camera: Camera) -> Dict[str, Any]:
    """将本地 Camera 转为煤安推送字段（camelCase）。"""
    mine_code = str(camera.mine_code or "").strip() or str(
        settings.COUNTING_RECOG_MINE_CODE or ""
    ).strip()
    data_time = str(camera.data_time or "").strip()
    if not data_time:
        data_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    item: Dict[str, Any] = {
        "cameraCode": str(camera.camera_code or "").strip(),
        "positionType": str(getattr(camera, "position_type", None) or "").strip(),
        "positionDesc": str(getattr(camera, "position_desc", None) or "").strip(),
        "analysisType": str(getattr(camera, "analysis_type", None) or "").strip(),
        "dataTime": data_time,
    }

    ps = str(getattr(camera, "ps_station_code", None) or "").strip()
    if ps:
        item["psStationCode"] = ps

    basic = str(getattr(camera, "basic_image_base64", None) or "").strip()
    if basic:
        item["basicImageBase64"] = basic

    if mine_code:
        item["mineCode"] = mine_code

    return item


def push_camera_config(camera: Camera) -> Optional[Dict[str, Any]]:
    """推送单路摄像头配置到煤安平台。

    成功返回响应 JSON；URL 未配置 / 关闭推送时返回 None。
    失败只记日志，不抛到上层（避免阻断本地保存）。
    """
    if not bool(getattr(settings, "COAL_CAMERA_PUSH_ENABLED", True)):
        logger.info(
            "跳过摄像仪配置推送：COAL_CAMERA_PUSH_ENABLED=false camera_id=%s",
            getattr(camera, "id", None),
        )
        return None

    url = str(settings.COAL_CAMERA_PUSH_URL or "").strip()
    if not url:
        logger.warning(
            "跳过摄像仪配置推送：COAL_CAMERA_PUSH_URL 未配置 camera_id=%s",
            getattr(camera, "id", None),
        )
        return None

    item = build_camera_push_item(camera)
    if not item.get("cameraCode"):
        logger.warning(
            "跳过摄像仪配置推送：cameraCode 为空 camera_id=%s",
            getattr(camera, "id", None),
        )
        return None

    payload: List[Dict[str, Any]] = [item]
    timeout = float(settings.COAL_CAMERA_PUSH_TIMEOUT or 15.0)

    try:
        return _post_camera_payload(url, payload, timeout=timeout, camera=camera)
    except Exception:
        logger.exception(
            "摄像仪配置推送异常 camera_id=%s cameraCode=%s",
            getattr(camera, "id", None),
            item.get("cameraCode"),
        )
        return None


def build_camera_push_payload(cameras: Iterable[Camera]) -> List[Dict[str, Any]]:
    """构造待推送的摄像仪基础信息 JSON 数组（不发送）。

    cameraCode 为空的摄像仪会被跳过（接口要求必填）。
    """
    payload: List[Dict[str, Any]] = []
    seen: set = set()
    for camera in cameras or []:
        item = build_camera_push_item(camera)
        code = str(item.get("cameraCode") or "").strip()
        if not code:
            logger.warning(
                "跳过摄像仪基础信息推送：cameraCode 为空 camera_id=%s",
                getattr(camera, "id", None),
            )
            continue
        if code in seen:
            logger.warning(
                "跳过摄像仪基础信息推送：单次请求内 cameraCode=%s 重复，"
                "按服务端幂等策略仅保留最后一条",
                code,
            )
            payload = [it for it in payload if it.get("cameraCode") != code]
        seen.add(code)
        payload.append(item)
    return payload


def dry_run_camera_push(camera: Camera) -> Optional[Dict[str, Any]]:
    """只构造单台摄像仪的基础信息报文，不发送（用于核对字段/编码）。"""
    item = build_camera_push_item(camera)
    if not item.get("cameraCode"):
        logger.warning(
            "构造摄像仪基础信息报文失败：cameraCode 为空 camera_id=%s",
            getattr(camera, "id", None),
        )
        return None
    return item


def _post_camera_payload(
    url: str,
    payload: List[Dict[str, Any]],
    *,
    timeout: float,
    camera: Camera,
) -> Optional[Dict[str, Any]]:
    item = payload[0] if payload else {}
    camera_id = getattr(camera, "id", None)
    camera_code = item.get("cameraCode")

    def _do_request(*, force_token: bool) -> httpx.Response:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **get_auth_header(force=force_token),
        }
        auth = str(headers.get("Authorization") or "")
        logger.info(
            "摄像仪配置推送 Authorization 前缀检查 starts_with_Bearer=%s "
            "auth_len=%s auth_preview=%r camera_id=%s",
            auth.lower().startswith("bearer "),
            len(auth),
            (auth[:16] + "...") if len(auth) > 16 else auth,
            camera_id,
        )
        with httpx.Client(timeout=timeout) as client:
            return client.post(url, json=payload, headers=headers)

    try:
        resp = _do_request(force_token=False)
        # token 失效时强制刷新再试一次
        if resp.status_code in {401, 403}:
            logger.warning(
                "摄像仪配置推送鉴权失败 status=%s，刷新 token 后重试 camera_id=%s",
                resp.status_code,
                camera_id,
            )
            resp = _do_request(force_token=True)
    except ClientTokenError as exc:
        logger.error(
            "摄像仪配置推送失败：获取 client_token 失败 camera_id=%s err=%s",
            camera_id,
            exc,
        )
        return None
    except httpx.RequestError as exc:
        logger.error(
            "摄像仪配置推送失败：无法连接 %s camera_id=%s err=%s",
            url,
            camera_id,
            exc,
        )
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
                "client_id 是否已授权 /coal/mine/ai/camera，或凭证/环境是否匹配）"
            )
        logger.error(
            "摄像仪配置推送失败 camera_id=%s cameraCode=%s http=%s code=%s msg=%s%s body=%s",
            camera_id,
            camera_code,
            resp.status_code,
            biz_code,
            biz_msg,
            hint,
            text,
        )
        return body

    logger.info(
        "摄像仪配置推送成功 camera_id=%s cameraCode=%s positionType=%s "
        "analysisType=%s mineCode=%s hasImage=%s msg=%s",
        camera_id,
        camera_code,
        item.get("positionType"),
        item.get("analysisType"),
        item.get("mineCode"),
        bool(item.get("basicImageBase64")),
        biz_msg or "操作成功",
    )
    return body
