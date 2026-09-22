"""大模型助手配置与 OpenAI 兼容流式对话（支持多模型）。"""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import LlmSettings

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_SYSTEM_PROMPT = (
    "你是 DeepSight 视频监控平台的智能助手，帮助用户理解告警、事件与系统使用。"
    "回答简洁、准确。"
)


def _mask_api_key(api_key: str) -> str:
    key = (api_key or "").strip()
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:4]}{'*' * min(16, len(key) - 8)}{key[-4:]}"


def settings_to_dict(row: LlmSettings, *, include_secret: bool = False) -> Dict[str, Any]:
    api_key = str(row.api_key or "")
    name = str(row.name or "").strip() or str(row.model or DEFAULT_MODEL)
    data = {
        "id": int(row.id),
        "name": name,
        "enabled": bool(row.enabled),
        "base_url": str(row.base_url or DEFAULT_BASE_URL),
        "model": str(row.model or DEFAULT_MODEL),
        "temperature": float(row.temperature if row.temperature is not None else 0.7),
        "max_tokens": int(row.max_tokens or 2048),
        "system_prompt": str(row.system_prompt or ""),
        "has_api_key": bool(api_key.strip()),
        "api_key_masked": _mask_api_key(api_key),
        "ready": bool(api_key.strip() and (row.base_url or "").strip() and (row.model or "").strip()),
        "configured": bool(
            row.enabled
            and api_key.strip()
            and (row.base_url or "").strip()
            and (row.model or "").strip()
        ),
    }
    if include_secret:
        data["api_key"] = api_key
    return data


def option_to_dict(row: LlmSettings) -> Dict[str, Any]:
    """助手页下拉用，不含密钥。"""
    name = str(row.name or "").strip() or str(row.model or DEFAULT_MODEL)
    return {
        "id": int(row.id),
        "name": name,
        "model": str(row.model or DEFAULT_MODEL),
        "base_url": str(row.base_url or DEFAULT_BASE_URL),
    }


def ensure_llm_settings(db: Session) -> None:
    """兼容旧单行配置：补齐 name；不再强制插入默认行。"""
    rows = list(db.scalars(select(LlmSettings)).all())
    changed = False
    for row in rows:
        if not str(getattr(row, "name", "") or "").strip():
            row.name = str(row.model or DEFAULT_MODEL)[:128]
            changed = True
    if changed:
        db.commit()


def list_llm_settings(db: Session) -> List[LlmSettings]:
    ensure_llm_settings(db)
    return list(
        db.scalars(select(LlmSettings).order_by(LlmSettings.id.desc())).all()
    )


def get_llm_settings(db: Session, llm_id: int) -> Optional[LlmSettings]:
    return db.get(LlmSettings, llm_id)


def list_enabled_llm_options(db: Session) -> List[LlmSettings]:
    """助手可选：已启用且具备地址/模型/密钥。"""
    ensure_llm_settings(db)
    rows = list(
        db.scalars(
            select(LlmSettings)
            .where(LlmSettings.enabled.is_(True))
            .order_by(LlmSettings.id.desc())
        ).all()
    )
    return [
        row
        for row in rows
        if str(row.api_key or "").strip()
        and str(row.base_url or "").strip()
        and str(row.model or "").strip()
    ]


def _validate_fields(
    *,
    name: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    api_key: Optional[str] = None,
    require_name: bool = False,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if name is not None or require_name:
        label = (name or "").strip()
        if not label:
            raise ValueError("显示名称不能为空")
        if len(label) > 128:
            raise ValueError("显示名称过长")
        out["name"] = label
    if base_url is not None:
        url = base_url.strip().rstrip("/")
        if not url:
            raise ValueError("接口地址不能为空")
        if len(url) > 512:
            raise ValueError("接口地址过长")
        out["base_url"] = url
    if model is not None:
        model_name = model.strip()
        if not model_name:
            raise ValueError("模型名称不能为空")
        out["model"] = model_name[:128]
    if temperature is not None:
        if temperature < 0 or temperature > 2:
            raise ValueError("temperature 需在 0–2 之间")
        out["temperature"] = float(temperature)
    if max_tokens is not None:
        if max_tokens < 1 or max_tokens > 128000:
            raise ValueError("max_tokens 不合法")
        out["max_tokens"] = int(max_tokens)
    if system_prompt is not None:
        out["system_prompt"] = system_prompt.strip()[:8000]
    if api_key is not None:
        key = api_key.strip()
        if key and "*" not in key:
            if len(key) > 512:
                raise ValueError("API Key 过长")
            out["api_key"] = key
    return out


def create_llm_settings(
    db: Session,
    *,
    name: str,
    base_url: str,
    model: str,
    api_key: str = "",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    enabled: bool = True,
) -> LlmSettings:
    fields = _validate_fields(
        name=name,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        api_key=api_key or None,
        require_name=True,
    )
    row = LlmSettings(
        name=fields["name"],
        enabled=bool(enabled),
        base_url=fields["base_url"],
        api_key=fields.get("api_key", ""),
        model=fields["model"],
        temperature=fields.get("temperature", 0.7),
        max_tokens=fields.get("max_tokens", 2048),
        system_prompt=fields.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def update_llm_settings(
    db: Session,
    row: LlmSettings,
    *,
    name: Optional[str] = None,
    enabled: Optional[bool] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    system_prompt: Optional[str] = None,
    clear_api_key: bool = False,
) -> LlmSettings:
    fields = _validate_fields(
        name=name,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        api_key=None if clear_api_key else api_key,
    )
    if "name" in fields:
        row.name = fields["name"]
    if enabled is not None:
        row.enabled = bool(enabled)
    if "base_url" in fields:
        row.base_url = fields["base_url"]
    if clear_api_key:
        row.api_key = ""
    elif "api_key" in fields:
        row.api_key = fields["api_key"]
    if "model" in fields:
        row.model = fields["model"]
    if "temperature" in fields:
        row.temperature = fields["temperature"]
    if "max_tokens" in fields:
        row.max_tokens = fields["max_tokens"]
    if "system_prompt" in fields:
        row.system_prompt = fields["system_prompt"]
    db.commit()
    db.refresh(row)
    return row


def delete_llm_settings(db: Session, row: LlmSettings) -> None:
    db.delete(row)
    db.commit()


def _chat_completions_url(base_url: str) -> str:
    root = (base_url or "").strip().rstrip("/")
    if root.endswith("/chat/completions"):
        return root
    return f"{root}/chat/completions"


def build_chat_messages(
    row: LlmSettings, messages: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    cleaned: List[Dict[str, str]] = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant", "system"} or not content:
            continue
        cleaned.append({"role": role, "content": content[:12000]})
    if not cleaned:
        raise ValueError("请输入对话内容")
    if not any(m["role"] == "user" for m in cleaned):
        raise ValueError("至少需要一条用户消息")
    recent = cleaned[-24:]
    system = str(row.system_prompt or "").strip()
    if system:
        recent = [m for m in recent if m["role"] != "system"]
        return [{"role": "system", "content": system}] + recent
    return recent


async def stream_chat_completion(
    row: LlmSettings, messages: List[Dict[str, str]]
) -> AsyncIterator[Dict[str, Any]]:
    """向上游发起流式 Chat Completions，产出统一 delta/done/error 事件。"""
    if not row.enabled:
        yield {"event": "error", "data": {"message": "该模型已停用，请选择其他模型或先启用"}}
        return
    api_key = str(row.api_key or "").strip()
    if not api_key:
        yield {"event": "error", "data": {"message": "未配置 API Key"}}
        return
    base_url = str(row.base_url or "").strip()
    if not base_url:
        yield {"event": "error", "data": {"message": "未配置接口地址"}}
        return

    try:
        payload_messages = build_chat_messages(row, messages)
    except ValueError as e:
        yield {"event": "error", "data": {"message": str(e)}}
        return

    body = {
        "model": str(row.model or DEFAULT_MODEL),
        "messages": payload_messages,
        "stream": True,
        "temperature": float(row.temperature if row.temperature is not None else 0.7),
        "max_tokens": int(row.max_tokens or 2048),
    }
    url = _chat_completions_url(base_url)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code >= 400:
                    err_text = (await resp.aread()).decode("utf-8", errors="replace")[:800]
                    yield {
                        "event": "error",
                        "data": {
                            "message": f"上游返回 {resp.status_code}: {err_text or resp.reason_phrase}"
                        },
                    }
                    return
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") if isinstance(chunk, dict) else None
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0] if isinstance(choices[0], dict) else {}
                    delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
                    content = delta.get("content")
                    if content:
                        yield {"event": "delta", "data": {"content": str(content)}}
                    finish = choice.get("finish_reason")
                    if finish:
                        yield {"event": "done", "data": {"finish_reason": str(finish)}}
                        return
        yield {"event": "done", "data": {"finish_reason": "stop"}}
    except httpx.TimeoutException:
        yield {"event": "error", "data": {"message": "请求大模型超时"}}
    except httpx.HTTPError as e:
        logger.exception("大模型流式请求失败")
        yield {"event": "error", "data": {"message": f"网络错误: {e}"}}
    except Exception as e:
        logger.exception("大模型流式对话异常")
        yield {"event": "error", "data": {"message": str(e) or "对话失败"}}
