"""煤安平台 AI 数据推送共用工具。

各推送模块（videoAnomaly / cameraStatus / countingRecog / undergroundCount /
importPersonCount）共用的配置读取、摄像仪有效性校验、HTTP 提交与响应判定逻辑。

约定：
- 配置统一走 ``app.core.config.settings``，缺字段时用默认值兜底；
- 鉴权统一走 ``app.auth.client_token``（401/403 自动刷新重试一次）；
- 所有失败只记日志、不抛异常，避免阻断上层业务流程。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import httpx

from app.auth.client_token import ClientTokenError, get_auth_header
from app.core.config import settings

logger = logging.getLogger(__name__)

MINE_CODE_RE = re.compile(r"^\d{12}$")
CAMERA_CODE_RE = re.compile(r"^\d{20}$")


# ---------------------------------------------------------------------------
# 配置读取
# ---------------------------------------------------------------------------

def cfg(name: str, default: Any = None) -> Any:
    return getattr(settings, name, default)


def cfg_bool(name: str, default: bool = True) -> bool:
    return bool(cfg(name, default))


def cfg_str(name: str, default: str = "") -> str:
    return str(cfg(name, default) or "").strip()


def cfg_float(name: str, default: float) -> float:
    try:
        return float(cfg(name, default) or default)
    except (TypeError, ValueError):
        return default


def default_mine_code() -> str:
    """全局兜底煤矿编码（请求未带 mineCode 时使用）。"""
    for name in ("MINE_CODE", "COUNTING_RECOG_MINE_CODE"):
        value = cfg_str(name)
        if value:
            return value
    return ""


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def apply_mine_code(item: Dict[str, Any], mine_code: Optional[str]) -> None:
    """写入 mineCode：优先入参，其次全局兜底；非 12 位仅告警不拦截。"""
    mc = str(mine_code or "").strip() or default_mine_code()
    if not mc:
        return
    if not MINE_CODE_RE.match(mc):
        logger.warning("推送：mineCode=%s 不是 12 位数字（仍按原值提交）", mc)
    item["mineCode"] = mc


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
        logger.exception("推送：读取摄像仪基本信息失败，跳过 cameraCode 校验")
        return None

    codes: Set[str] = set()
    for (code,) in rows:
        value = str(code or "").strip()
        if value:
            codes.add(value)
    return codes


def camera_code_valid(
    camera_code: str,
    valid_camera_codes: Optional[Set[str]],
) -> bool:
    """校验 cameraCode 是否在摄像仪基本信息中（None 表示无法校验，放行）。"""
    if valid_camera_codes is None:
        return True
    return str(camera_code or "").strip() in valid_camera_codes


# ---------------------------------------------------------------------------
# 去重（全量替换语义：单次请求内同一键只保留最后一条）
# ---------------------------------------------------------------------------

def dedupe_by_key(
    items: List[Dict[str, Any]],
    key_fields: tuple,
    *,
    label: str,
) -> List[Dict[str, Any]]:
    """按 ``key_fields`` 组合去重，保留最后一条并保持首次出现顺序。"""
    order: List[tuple] = []
    latest: Dict[tuple, Dict[str, Any]] = {}
    for item in items:
        key = tuple(str(item.get(f) or "").strip() for f in key_fields)
        if key in latest:
            logger.warning(
                "%s：单次请求内 %s=%s 重复，按服务端幂等策略仅保留最后一条",
                label,
                "+".join(key_fields),
                "|".join(key),
            )
        else:
            order.append(key)
        latest[key] = item
    return [latest[key] for key in order]


# ---------------------------------------------------------------------------
# HTTP 提交（含 token 失效重试）
# ---------------------------------------------------------------------------

def post_json_array(
    url: str,
    payload: List[Dict[str, Any]],
    *,
    timeout: float,
    label: str,
    success_summary: str = "",
) -> Optional[Dict[str, Any]]:
    """POST JSON 数组到煤安平台，返回响应 body（dict）或 None。

    - Authorization 每次经 client_token 获取；
    - 401/403 时强制刷新 token 重试一次；
    - HTTP>=400 或业务 code != 200 记错误日志并返回 body；
    - 所有异常只记日志，不抛出。
    """

    def _do_request(*, force_token: bool) -> httpx.Response:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            **get_auth_header(force=force_token),
        }
        auth = str(headers.get("Authorization") or "")
        logger.info(
            "%s Authorization 前缀检查 starts_with_Bearer=%s auth_len=%s "
            "auth_preview=%r count=%s",
            label,
            auth.lower().startswith("bearer "),
            len(auth),
            (auth[:16] + "...") if len(auth) > 16 else auth,
            len(payload),
        )
        with httpx.Client(timeout=timeout) as client:
            return client.post(url, json=payload, headers=headers)

    try:
        resp = _do_request(force_token=False)
        if resp.status_code in {401, 403}:
            logger.warning(
                "%s 鉴权失败 status=%s，刷新 token 后重试 count=%s",
                label,
                resp.status_code,
                len(payload),
            )
            resp = _do_request(force_token=True)
    except ClientTokenError as exc:
        logger.error("%s 失败：获取 client_token 失败 err=%s", label, exc)
        return None
    except httpx.RequestError as exc:
        logger.error("%s 失败：无法连接 %s err=%s", label, url, exc)
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
                "（Authorization 已按 Bearer 提交；请向煤安平台确认该 client_id "
                "是否已授权该接口，或凭证/环境是否匹配）"
            )
        logger.error(
            "%s 失败 http=%s code=%s msg=%s%s body=%s",
            label,
            resp.status_code,
            biz_code,
            biz_msg,
            hint,
            text,
        )
        return body

    logger.info(
        "%s 成功 count=%s %s msg=%s",
        label,
        len(payload),
        success_summary,
        biz_msg or "操作成功",
    )
    return body


def build_result(
    body: Optional[Dict[str, Any]],
    *,
    pushed: int,
    skipped: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """统一返回结构：code / msg / pushed / skipped（+ 业务统计）。"""
    result: Dict[str, Any] = {
        "code": (body or {}).get("code"),
        "msg": (body or {}).get("msg"),
        "pushed": pushed,
        "skipped": skipped,
    }
    if extra:
        result.update(extra)
    if body is not None:
        result["body"] = body
    return result


def skip_log(label: str, code_field: str, code: str, reason: str) -> None:
    logger.warning("跳过一条%s：%s=%s %s", label, code_field, code, reason)
