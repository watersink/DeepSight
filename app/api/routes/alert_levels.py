"""报警等级配置 API"""
from typing import Annotated, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.db import get_db
from app.db.models import User
from app.services import alert_level_service as svc

router = APIRouter(tags=["报警等级"])


class AlertLevelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    type_key: str
    name_zh: str
    category: str
    level: int
    sort_order: int


class AlertLevelUpdate(BaseModel):
    level: int = Field(..., ge=1, le=3, description="1 严重 / 2 警告 / 3 提示")


@router.get(
    "/alert-levels",
    response_model=List[AlertLevelOut],
    summary="报警等级配置列表（管理员）",
)
def list_alert_levels(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    return [AlertLevelOut.model_validate(row) for row in svc.list_alert_levels(db)]


@router.patch(
    "/alert-levels/{type_key}",
    response_model=AlertLevelOut,
    summary="修改某一类型的报警等级（管理员）",
)
def patch_alert_level(
    type_key: str,
    body: AlertLevelUpdate,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    try:
        row = svc.update_alert_level(db, type_key, body.level)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return AlertLevelOut.model_validate(row)
