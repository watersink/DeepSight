"""大模型配置与 SSE 流式对话 API（多模型）"""
from __future__ import annotations

import json
from typing import Annotated, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_admin
from app.db import get_db
from app.db.models import User
from app.services import llm_settings_service as svc

router = APIRouter(tags=["大模型助手"])


class LlmSettingsOut(BaseModel):
    id: int
    name: str
    enabled: bool
    base_url: str
    model: str
    temperature: float
    max_tokens: int
    system_prompt: str
    has_api_key: bool
    api_key_masked: str
    ready: bool
    configured: bool


class LlmSettingsListOut(BaseModel):
    items: List[LlmSettingsOut]


class LlmSettingsCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    base_url: str = Field(..., min_length=1, max_length=512)
    api_key: str = Field(default="", max_length=512)
    model: str = Field(..., min_length=1, max_length=128)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=2048, ge=1, le=128000)
    system_prompt: str = Field(default="", max_length=8000)
    enabled: bool = True


class LlmSettingsUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128)
    enabled: Optional[bool] = None
    base_url: Optional[str] = Field(None, max_length=512)
    api_key: Optional[str] = Field(None, max_length=512)
    model: Optional[str] = Field(None, max_length=128)
    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1, le=128000)
    system_prompt: Optional[str] = Field(None, max_length=8000)
    clear_api_key: bool = False


class LlmOptionOut(BaseModel):
    id: int
    name: str
    model: str
    base_url: str


class LlmOptionsOut(BaseModel):
    items: List[LlmOptionOut]


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(..., min_length=1, max_length=12000)


class ChatRequest(BaseModel):
    messages: List[ChatMessageIn] = Field(..., min_length=1, max_length=40)
    llm_id: int = Field(..., description="选用的大模型配置 ID")


def _sse_pack(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


class LlmSettingsSnapshot:
    """脱离 Session 的配置快照，供流式请求使用。"""

    def __init__(
        self,
        *,
        enabled: bool,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
    ) -> None:
        self.enabled = enabled
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt


@router.get(
    "/llm-settings",
    response_model=LlmSettingsListOut,
    summary="大模型配置列表（管理员）",
)
def list_llm_settings(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    items = [LlmSettingsOut(**svc.settings_to_dict(row)) for row in svc.list_llm_settings(db)]
    return LlmSettingsListOut(items=items)


@router.post(
    "/llm-settings",
    response_model=LlmSettingsOut,
    summary="新增大模型配置（管理员）",
)
def create_llm_settings(
    body: LlmSettingsCreate,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    try:
        row = svc.create_llm_settings(
            db,
            name=body.name,
            base_url=body.base_url,
            model=body.model,
            api_key=body.api_key,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            system_prompt=body.system_prompt or svc.DEFAULT_SYSTEM_PROMPT,
            enabled=body.enabled,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return LlmSettingsOut(**svc.settings_to_dict(row))


@router.get(
    "/llm-settings/options",
    response_model=LlmOptionsOut,
    summary="已启用大模型列表（登录用户，供助手选择）",
)
def list_llm_options(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    items = [
        LlmOptionOut(**svc.option_to_dict(row))
        for row in svc.list_enabled_llm_options(db)
    ]
    return LlmOptionsOut(items=items)


@router.patch(
    "/llm-settings/{llm_id}",
    response_model=LlmSettingsOut,
    summary="更新大模型配置（管理员）",
)
def patch_llm_settings(
    llm_id: int,
    body: LlmSettingsUpdate,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    row = svc.get_llm_settings(db, llm_id)
    if row is None:
        raise HTTPException(status_code=404, detail="大模型配置不存在")
    try:
        row = svc.update_llm_settings(
            db,
            row,
            name=body.name,
            enabled=body.enabled,
            base_url=body.base_url,
            api_key=body.api_key,
            model=body.model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            system_prompt=body.system_prompt,
            clear_api_key=body.clear_api_key,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return LlmSettingsOut(**svc.settings_to_dict(row))


@router.delete(
    "/llm-settings/{llm_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除大模型配置（管理员）",
)
def delete_llm_settings(
    llm_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    row = svc.get_llm_settings(db, llm_id)
    if row is None:
        raise HTTPException(status_code=404, detail="大模型配置不存在")
    svc.delete_llm_settings(db, row)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/llm/chat/sse",
    summary="大模型流式对话（SSE）",
    description="兼容 OpenAI Chat Completions；事件：delta / done / error。需指定 llm_id。",
)
async def llm_chat_sse(
    body: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(get_current_user)],
):
    row = svc.get_llm_settings(db, body.llm_id)
    if row is None:
        raise HTTPException(status_code=404, detail="大模型配置不存在")
    if not row.enabled:
        raise HTTPException(status_code=400, detail="该模型已停用")
    snap = LlmSettingsSnapshot(
        enabled=bool(row.enabled),
        base_url=str(row.base_url or ""),
        api_key=str(row.api_key or ""),
        model=str(row.model or ""),
        temperature=float(row.temperature if row.temperature is not None else 0.7),
        max_tokens=int(row.max_tokens or 2048),
        system_prompt=str(row.system_prompt or ""),
    )
    messages = [m.model_dump() for m in body.messages]

    async def event_generator():
        yield _sse_pack("connected", {"ok": True, "llm_id": body.llm_id})
        async for item in svc.stream_chat_completion(snap, messages):
            yield _sse_pack(item["event"], item.get("data") or {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
