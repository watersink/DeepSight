"""Redis 缓存（不可用时静默降级）"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

_client = None
_client_failed = False


def get_redis():
    global _client, _client_failed
    if not settings.REDIS_ENABLED or _client_failed:
        return None
    if _client is not None:
        return _client
    try:
        import redis

        kwargs = {
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
            "db": settings.REDIS_DB,
            "decode_responses": True,
            "socket_connect_timeout": 1.5,
            "socket_timeout": 1.5,
        }
        if settings.REDIS_PASSWORD:
            kwargs["password"] = settings.REDIS_PASSWORD
        client = redis.Redis(**kwargs)
        client.ping()
        _client = client
        logger.info(
            "Redis 已连接 %s:%s db=%s",
            settings.REDIS_HOST,
            settings.REDIS_PORT,
            settings.REDIS_DB,
        )
        return _client
    except Exception as e:
        _client_failed = True
        logger.warning("Redis 不可用，缓存降级为直查数据库: %s", e)
        return None


def cache_get(key: str) -> Optional[Any]:
    client = get_redis()
    if client is None:
        return None
    try:
        raw = client.get(key)
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        return None


def cache_set(key: str, value: Any, ttl_seconds: int = 60) -> None:
    client = get_redis()
    if client is None:
        return
    try:
        client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False, default=str))
    except Exception:
        pass


def cache_delete(*keys: str) -> None:
    client = get_redis()
    if client is None or not keys:
        return
    try:
        client.delete(*keys)
    except Exception:
        pass
