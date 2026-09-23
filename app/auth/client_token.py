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

    data = {
        "grant_type": _GRANT_TYPE,
        "client_id": settings.COAL_OAUTH_CLIENT_ID,
        "client_secret": settings.COAL_OAUTH_CLIENT_SECRET,
        "scope": settings.COAL_OAUTH_SCOPE,
    }

    try:
        with httpx.Client(timeout=settings.COAL_OAUTH_TIMEOUT) as client:
            resp = client.post(
                url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
    except httpx.RequestError as exc:
        raise ClientTokenError(f"无法连接 client_token 接口 ({url}): {exc}") from exc

    if resp.status_code >= 400:
        raise ClientTokenError(
            f"client_token 接口返回 {resp.status_code}: {resp.text[:300]}"
        )

    try:
        body = resp.json()
    except ValueError as exc:
        raise ClientTokenError(
            f"client_token 响应不是合法 JSON: {resp.text[:300]}"
        ) from exc

    token = (body.get("client_token") or "").strip()
    if not token:
        raise ClientTokenError(
            f"client_token 响应缺少 client_token 字段: {resp.text[:300]}"
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
            return token

        token = _request_client_token()
        _cache["token"] = token
        _cache["fetched_at"] = now
        return token


def get_auth_header(force: bool = False) -> dict:
    """返回可直接用于业务接口的 ``Authorization`` 请求头。"""
    return {"Authorization": get_client_token(force=force)}
