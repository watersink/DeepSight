"""OAuth2 客户端模式（client_credentials）获取 client_token。

供需要调用煤安（coal/mine）业务接口的模块统一获取访问令牌。
静态配置（client_id / client_secret / scope / token 地址）写入根目录 .env。

使用示例::

    from app.auth.client_token import get_client_token, get_auth_header

    token = get_client_token()          # 形如 "Bearer xxxxx"
    headers = get_auth_header()         # {"Authorization": "Bearer xxxxx"}
"""
from __future__ import annotations

import logging
import threading
import time

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

_GRANT_TYPE = "client_credentials"

# 进程内缓存：避免每次调用业务接口都重新请求 token。
_cache = {"token": None, "fetched_at": 0.0}
_lock = threading.Lock()


class ClientTokenError(RuntimeError):
    """client_token 获取失败。"""


def _request_client_token() -> str:
    """向授权服务请求 client_token，返回形如 ``Bearer xxxxx`` 的完整令牌。"""
    url = str(settings.COAL_OAUTH_TOKEN_URL or "").strip()
    if not url:
        raise ClientTokenError("COAL_OAUTH_TOKEN_URL 未配置")

    client_id = str(settings.COAL_OAUTH_CLIENT_ID or "").strip()
    client_secret = str(settings.COAL_OAUTH_CLIENT_SECRET or "").strip()
    scope = str(settings.COAL_OAUTH_SCOPE or "").strip() or "mine"
    if not client_id or not client_secret:
        raise ClientTokenError(
            "COAL_OAUTH_CLIENT_ID / COAL_OAUTH_CLIENT_SECRET 未配置，"
            "请在项目根目录 .env 中填写煤安平台 OAuth2 客户端凭证"
        )

    data = {
        "grant_type": _GRANT_TYPE,
        "client_id": client_id,
        "client_secret": client_secret,
        "scope": scope,
    }

    logger.info(
        "开始请求 client_token url=%s client_id=%s scope=%s",
        url,
        client_id,
        scope,
    )
    try:
        with httpx.Client(timeout=settings.COAL_OAUTH_TIMEOUT) as client:
            resp = client.post(
                url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except httpx.RequestError as exc:
        logger.error("client_token 请求失败：无法连接 url=%s err=%s", url, exc)
        raise ClientTokenError(f"无法连接 client_token 接口 ({url}): {exc}") from exc

    if resp.status_code >= 400:
        logger.error(
            "client_token 请求失败 http=%s url=%s body=%s",
            resp.status_code,
            url,
            (resp.text or "")[:300],
        )
        raise ClientTokenError(
            f"client_token 接口返回 {resp.status_code}: {resp.text[:300]}"
        )

    try:
        body = resp.json()
    except ValueError as exc:
        logger.error(
            "client_token 请求失败：响应非 JSON http=%s body=%s",
            resp.status_code,
            (resp.text or "")[:300],
        )
        raise ClientTokenError(
            f"client_token 响应不是合法 JSON: {resp.text[:300]}"
        ) from exc

    # 兼容业务包装：{"code":200,"data":{"client_token":"..."}} 或扁平 {"client_token":"..."}
    token = ""
    biz_code = None
    biz_msg = None
    if isinstance(body, dict):
        biz_code = body.get("code")
        biz_msg = body.get("msg")
        token = str(body.get("client_token") or "").strip()
        if not token and isinstance(body.get("data"), dict):
            token = str(body["data"].get("client_token") or "").strip()
        if not token and isinstance(body.get("data"), str):
            token = str(body.get("data") or "").strip()
        if not token:
            token = str(body.get("access_token") or "").strip()
        # 业务失败码（HTTP 仍可能是 200）
        if biz_code is not None and int(biz_code) != 200 and not token:
            logger.error(
                "client_token 请求失败 http=%s code=%s msg=%s body=%s",
                resp.status_code,
                biz_code,
                biz_msg,
                (resp.text or "")[:300],
            )
            raise ClientTokenError(
                f"client_token 业务失败 code={biz_code} msg={biz_msg!r}"
            )

    if not token:
        logger.error(
            "client_token 请求失败：响应缺少 token 字段 http=%s body=%s",
            resp.status_code,
            (resp.text or "")[:300],
        )
        raise ClientTokenError(
            f"client_token 响应缺少 client_token 字段: {resp.text[:300]}"
        )
    # 统一成 "Bearer xxx"，便于直接放进 Authorization
    if not token.lower().startswith("bearer "):
        token = f"Bearer {token}"

    # 日志只打长度与尾号，避免完整 token 落盘
    token_tail = token[-6:] if len(token) >= 6 else token
    logger.info(
        "client_token 请求成功 http=%s code=%s msg=%s client_id=%s "
        "token_len=%s token_tail=%s",
        resp.status_code,
        biz_code if biz_code is not None else 200,
        biz_msg or "ok",
        client_id,
        len(token),
        token_tail,
    )
    return token


def get_client_token(force: bool = False) -> str:
    """获取 client_token（带进程内缓存）。

    返回形如 ``Bearer xxxxx`` 的完整令牌，可直接放入 ``Authorization`` 请求头。
    服务端响应不含有效期，本地按 ``COAL_OAUTH_TOKEN_TTL_SECONDS`` 复用，
    临近过期自动刷新；``force=True`` 时立即重新获取。
    """
    now = time.time()
    with _lock:
        token = _cache["token"]
        age = now - _cache["fetched_at"]
        ttl = settings.COAL_OAUTH_TOKEN_TTL_SECONDS
        margin = settings.COAL_OAUTH_TOKEN_REFRESH_MARGIN
        if token and not force and age < max(ttl - margin, 1):
            logger.info(
                "复用缓存 client_token age=%.1fs ttl=%ss",
                age,
                ttl,
            )
            return token

        token = _request_client_token()
        _cache["token"] = token
        _cache["fetched_at"] = now
        return token


def get_auth_header(force: bool = False) -> dict:
    """返回可直接用于业务接口的 ``Authorization`` 请求头。"""
    return {"Authorization": get_client_token(force=force)}
