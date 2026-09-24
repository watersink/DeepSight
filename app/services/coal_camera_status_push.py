"""煤安平台摄像仪状态推送（在线/离线）。

接口：``POST /mine/ai/cameraStatus``（MT/T 1201.6-2023）
- 请求体为 JSON 数组，一次可推送多台；
- 无图片字段；
- ``cameraCode`` 必须是摄像仪基本信息中的有效摄像仪（默认按本地 Camera 表校验）；
- 幂等策略：服务端按 ``cameraCode`` 先删除后新增，全量替换。
  故单次请求内同一 ``cameraCode`` 只保留最后一条（保留最后状态），并记日志提示。

状态来源：调用 ``app.services.zlm_client.zlm_client.list_online_media_keys()``
判断流是否在线（取不到 ZLM 结果时不推送，避免把全部摄像仪误报为离线）。
也支持直接给定状态推送（``push_camera_statuses``）。

鉴权：Authorization 每次经 ``app.auth.client_token`` 获取（401/403 自动刷新重试）。
所有失败只记日志、不抛异常，避免阻断上层业务流程。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import httpx

from app.auth.client_token import ClientTokenError, get_auth_header
from app.core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

CAMERA_STATUS_OFFLINE = "0"   # 离线
CAMERA_STATUS_ONLINE = "1"    # 在线

VALID_CAMERA_STATUSES: Set[str] = {CAMERA_STATUS_OFFLINE, CAMERA_STATUS_ONLINE}

_MINE_CODE_RE = re.compile(r"^\d{12}$")


# ---------------------------------------------------------------------------
# 配置读取
# ---------------------------------------------------------------------------

def _cfg(name: str, default: Any = None) -> Any:
    return getattr(settings, name, default)


def _push_enabled() -> bool:
    return bool(_cfg("COAL_CAMERA_STATUS_PUSH_ENABLED", True))


def _push_url() -> str:
    return str(_cfg("COAL_CAMERA_STATUS_PUSH_URL", "") or "").strip()


def _push_timeout() -> float:
    try:
        return float(_cfg("COAL_CAMERA_STATUS_PUSH_TIMEOUT", 15.0) or 15.0)
    except (TypeError, ValueError):
        return 15.0


def _default_mine_code() -> str:
    for name in (
        "MINE_CODE",
        "COUNTING_RECOG_MINE_CODE",
        "COAL_CAMERA_STATUS_MINE_CODE",
    ):
        value = str(_cfg(name, "") or "").strip()
        if value:
            return value
    return ""


def normalize_camera_status(value: Any) -> Optional[str]:
    """把入参状态规整为 ``"0"`` / ``"1"``；非法返回 None。"""
    if value is None:
        return None
    if isinstance(value, bool):
        return CAMERA_STATUS_ONLINE if value else CAMERA_STATUS_OFFLINE
    if isinstance(value, int):
        return str(value) if str(value) in VALID_CAMERA_STATUSES else None
    text = str(value).strip()
    if text in VALID_CAMERA_STATUSES:
        return text
    lowered = text.lower()
    if lowered in {"online", "true", "yes", "on"}:
        return CAMERA_STATUS_ONLINE
    if lowered in {"offline", "false", "no", "off"}:
        return CAMERA_STATUS_OFFLINE
    if lowered in {"在线", "上线"}:
        return CAMERA_STATUS_ONLINE
    if lowered in {"离线", "下线"}:
        return CAMERA_STATUS_OFFLINE
    return None


# ---------------------------------------------------------------------------
# 构造 payload
# ---------------------------------------------------------------------------

def build_camera_status_item(
    *,
    camera_code: str,
    camera_status: Any,
    data_time: Optional[str] = None,
    mine_code: Optional[str] = None,
) -> Dict[str, Any]:
    """构造单台摄像仪状态推送字段（camelCase）。"""
    code = str(camera_code or "").strip()
    status = normalize_camera_status(camera_status)
    if status is None:
        logger.warning(
            "摄像仪状态推送：cameraStatus=%r 非法（应为 0/1），仍按原值提交",
            camera_status,
        )
        status = str(camera_status or "").strip()

    dt = str(data_time or "").strip() or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    item: Dict[str, Any] = {
        "cameraCode": code,
        "cameraStatus": status,
        "dataTime": dt,
    }

    mc = str(mine_code or "").strip() or _default_mine_code()
    if mc:
        if not _MINE_CODE_RE.match(mc):
            logger.warning(
                "摄像仪状态推送：mineCode=%s 不是 12 位数字（仍按原值提交）", mc
            )
        item["mineCode"] = mc

    return item


def dedupe_by_camera_code(
    items: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """同一 cameraCode 只保留最后一条（全量替换语义，后者覆盖前者）。"""
    order: List[str] = []
    latest: Dict[str, Dict[str, Any]] = {}
    for item in items:
        code = str(item.get("cameraCode") or "").strip()
        if code not in latest:
            order.append(code)
        else:
            logger.warning(
                "摄像仪状态推送：单次请求内 cameraCode=%s 重复，"
                "按服务端幂等策略仅保留最后一条",
                code,
            )
        latest[code] = item
    return [latest[code] for code in order if code]


# ---------------------------------------------------------------------------
# 摄像仪有效性校验 / 在线状态来源
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
        logger.exception("摄像仪状态推送：读取摄像仪基本信息失败，跳过 cameraCode 校验")
        return None

    codes: Set[str] = set()
    for (code,) in rows:
        value = str(code or "").strip()
        if value:
            codes.add(value)
    return codes


def load_online_media_keys() -> Optional[Set[str]]:
    """获取 ZLM 在线流 key 集合；失败返回 None。"""
    try:
        from app.services.zlm_client import zlm_client

        return zlm_client.list_online_media_keys()
    except Exception:
        logger.exception("摄像仪状态推送：获取 ZLM 在线流列表失败")
        return None


def collect_camera_statuses() -> Optional[List[Dict[str, Any]]]:
    """从本地摄像仪基本信息 + ZLM 在线流，汇总各摄像仪在线状态。

    与 ``mgmt_service._camera_to_dict`` 取值口径一致：优先用 Camera 的
    ``zlm_app`` / ``zlm_stream``，缺失时从 ``in_url`` 解析 RTMP/RTSP 地址。

    返回 ``[{"cameraCode": ..., "cameraStatus": "0"/"1", "mineCode": ...}, ...]``；
    读取摄像仪或 ZLM 失败时返回 None（调用方不应把失败当成"全部离线"）。
    """
    try:
        from app.db import SessionLocal
        from app.db.models import Camera

        db = SessionLocal()
        try:
            rows = (
                db.query(
                    Camera.camera_code,
                    Camera.mine_code,
                    Camera.in_url,
                    Camera.zlm_app,
                    Camera.zlm_stream,
                )
                .filter(Camera.camera_code != "")
                .all()
            )
        finally:
            db.close()
    except Exception:
        logger.exception("摄像仪状态推送：读取摄像仪基本信息失败")
        return None

    try:
        from app.services.zlm_client import parse_stream_url, zlm_client

        online_keys = zlm_client.list_online_media_keys()
    except Exception:
        logger.exception("摄像仪状态推送：获取 ZLM 在线流列表失败")
        return None

    out: List[Dict[str, Any]] = []
    no_stream = 0
    for camera_code, mine_code, in_url, zlm_app, zlm_stream in rows:
        code = str(camera_code or "").strip()
        if not code:
            continue

        app = str(zlm_app or "").strip()
        stream = str(zlm_stream or "").strip()
        if not (app and stream):
            parsed = parse_stream_url(str(in_url or ""))
            if not parsed:
                no_stream += 1
                continue
            app, stream = parsed["app"], parsed["stream"]

        key = zlm_client.media_key(app, stream)
        out.append(
            {
                "cameraCode": code,
                # 带上该摄像仪自己的 mineCode，避免推送时回落到全局兜底值，
                # 导致与 /mine/ai/camera 推送的 mineCode 不一致
                "mineCode": str(mine_code or "").strip() or None,
                "cameraStatus": (
                    CAMERA_STATUS_ONLINE
                    if key in online_keys
                    else CAMERA_STATUS_OFFLINE
                ),
            }
        )

    if no_stream:
        logger.warning(
            "摄像仪状态推送：%s 台摄像仪无有效流地址，已跳过（无法判定在线状态）",
            no_stream,
        )
    return out


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def push_camera_statuses(
    statuses: Iterable[Dict[str, Any]],
    *,
    valid_camera_codes: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """批量推送摄像仪在线/离线状态。

    ``statuses`` 每项字段：cameraCode / cameraStatus（0/1）/ dataTime(可选) /
    mineCode(可选)。失败只记日志，不抛异常。

    成功返回 ``{"code":..., "msg":..., "pushed":n, "skipped":k, "online":x, "offline":y}``；
    未启用或未配置 URL 时返回 None。
    """
    if not _push_enabled():
        logger.info("跳过摄像仪状态推送：COAL_CAMERA_STATUS_PUSH_ENABLED=false")
        return None

    url = _push_url()
    if not url:
        logger.warning("跳过摄像仪状态推送：COAL_CAMERA_STATUS_PUSH_URL 未配置")
        return None

    raw_items = [it for it in (statuses or []) if isinstance(it, dict) and it]
    if not raw_items:
        logger.warning("跳过摄像仪状态推送：无有效记录")
        return None

    if valid_camera_codes is None:
        valid_camera_codes = load_valid_camera_codes()

    payload: List[Dict[str, Any]] = []
    skipped = 0
    for it in raw_items:
        camera_code = str(it.get("cameraCode") or it.get("camera_code") or "").strip()
        if not camera_code:
            skipped += 1
            logger.warning("跳过一条摄像仪状态推送：cameraCode 为空")
            continue
        if valid_camera_codes is not None and camera_code not in valid_camera_codes:
            skipped += 1
            logger.warning(
                "跳过一条摄像仪状态推送：cameraCode=%s 不在摄像仪基本信息中",
                camera_code,
            )
            continue

        status = it.get("cameraStatus")
        if status is None:
            status = it.get("camera_status")
        if normalize_camera_status(status) is None:
            skipped += 1
            logger.warning(
                "跳过一条摄像仪状态推送：cameraStatus=%r 非法（应为 0/1）cameraCode=%s",
                status,
                camera_code,
            )
            continue

        payload.append(
            build_camera_status_item(
                camera_code=camera_code,
                camera_status=status,
                data_time=it.get("dataTime") or it.get("data_time"),
                mine_code=it.get("mineCode") or it.get("mine_code"),
            )
        )

    if not payload:
        logger.warning(
            "跳过摄像仪状态推送：无有效记录 skipped=%s（摄像仪校验=%s）",
            skipped,
            "已启用" if valid_camera_codes is not None else "跳过",
        )
        return None

    # 幂等：单次请求内同一 cameraCode 只保留最后一条
    payload = dedupe_by_camera_code(payload)

    try:
        body = _post_camera_status_payload(url, payload, timeout=_push_timeout())
    except Exception:
        logger.exception(
            "摄像仪状态推送异常 count=%s cameraCodes=%s",
            len(payload),
            [it.get("cameraCode") for it in payload],
        )
        return None

    online = sum(1 for it in payload if it.get("cameraStatus") == CAMERA_STATUS_ONLINE)
    result: Dict[str, Any] = {
        "code": (body or {}).get("code"),
        "msg": (body or {}).get("msg"),
        "pushed": len(payload),
        "skipped": skipped,
        "online": online,
        "offline": len(payload) - online,
    }
    if body is not None:
        result["body"] = body
    return result


def push_camera_status(
    *,
    camera_code: str,
    camera_status: Any,
    data_time: Optional[str] = None,
    mine_code: Optional[str] = None,
    valid_camera_codes: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """推送单台摄像仪状态（便捷入口）。"""
    return push_camera_statuses(
        [
            {
                "cameraCode": camera_code,
                "cameraStatus": camera_status,
                "dataTime": data_time,
                "mineCode": mine_code,
            }
        ],
        valid_camera_codes=valid_camera_codes,
    )


def push_all_camera_statuses() -> Optional[Dict[str, Any]]:
    """采集本地全部摄像仪的在线状态并推送（ZLM 判在线）。

    适合定时任务/手动触发；采集失败返回 None 且不推送。
    """
    statuses = collect_camera_statuses()
    if not statuses:
        logger.warning("跳过摄像仪状态推送：未采集到摄像仪状态（或采集失败）")
        return None
    logger.info("摄像仪状态推送：采集到 %s 台摄像仪状态", len(statuses))
    return push_camera_statuses(statuses)


def push_camera_obj_status(
    camera: Any,
    *,
    camera_status: Any,
    mine_code: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """按本地 Camera ORM 对象推送其状态（cameraCode / mineCode 取自对象）。"""
    camera_code = str(getattr(camera, "camera_code", "") or "").strip()
    if not camera_code:
        logger.warning(
            "跳过摄像仪状态推送：Camera.camera_code 为空 camera_id=%s",
            getattr(camera, "id", None),
        )
        return None
    return push_camera_status(
        camera_code=camera_code,
        camera_status=camera_status,
        mine_code=mine_code or getattr(camera, "mine_code", None),
    )


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _post_camera_status_payload(
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
            "摄像仪状态推送 Authorization 前缀检查 starts_with_Bearer=%s "
            "auth_len=%s auth_preview=%r count=%s",
            auth.lower().startswith("bearer "),
            len(auth),
            (auth[:16] + "...") if len(auth) > 16 else auth,
            len(payload),
        )
        with httpx.Client(timeout=timeout) as client:
            return client.post(url, json=payload, headers=headers)

    try:
        resp = _do_request(force_token=False)
        # token 失效时强制刷新再试一次
        if resp.status_code in {401, 403}:
            logger.warning(
                "摄像仪状态推送鉴权失败 status=%s，刷新 token 后重试 count=%s",
                resp.status_code,
                len(payload),
            )
            resp = _do_request(force_token=True)
    except ClientTokenError as exc:
        logger.error("摄像仪状态推送失败：获取 client_token 失败 err=%s", exc)
        return None
    except httpx.RequestError as exc:
        logger.error("摄像仪状态推送失败：无法连接 %s err=%s", url, exc)
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
                "client_id 是否已授权 /coal/mine/ai/cameraStatus，"
                "或凭证/环境是否匹配）"
            )
        logger.error(
            "摄像仪状态推送失败 count=%s cameraCodes=%s http=%s code=%s msg=%s%s body=%s",
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
        "摄像仪状态推送成功 count=%s cameraCodes=%s statuses=%s msg=%s",
        len(payload),
        camera_codes,
        [it.get("cameraStatus") for it in payload],
        biz_msg or "操作成功",
    )
    return body
