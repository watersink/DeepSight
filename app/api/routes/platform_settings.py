"""平台基础配置 API"""
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.api.deps import require_admin
from app.db import get_db
from app.db.models import User
from app.services import platform_settings_service as svc

router = APIRouter(tags=["基础配置"])


class PlatformSettingsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    platform_name: str
    slogan: str
    logo_url: Optional[str] = None


class PlatformSettingsUpdate(BaseModel):
    platform_name: Optional[str] = Field(None, min_length=1, max_length=128)
    slogan: Optional[str] = Field(None, max_length=255)
    clear_logo: bool = False


@router.get(
    "/platform-settings",
    response_model=PlatformSettingsOut,
    summary="获取平台基础配置（公开，登录页可用）",
)
def get_platform_settings(db: Annotated[Session, Depends(get_db)]):
    row = svc.get_platform_settings(db)
    return PlatformSettingsOut.model_validate(row)


@router.patch(
    "/platform-settings",
    response_model=PlatformSettingsOut,
    summary="更新平台名称/口号（管理员）",
)
def patch_platform_settings(
    body: PlatformSettingsUpdate,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    try:
        row = svc.update_platform_settings(
            db,
            platform_name=body.platform_name,
            slogan=body.slogan,
            clear_logo=body.clear_logo,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return PlatformSettingsOut.model_validate(row)


@router.post(
    "/platform-settings/logo",
    response_model=PlatformSettingsOut,
    summary="上传平台 Logo（管理员）",
)
async def upload_platform_logo(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
    file: UploadFile = File(...),
):
    data = await file.read()
    try:
        row = svc.upload_platform_logo(
            db, data, file.content_type or "application/octet-stream"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Logo 上传失败: {e}",
        ) from e
    return PlatformSettingsOut.model_validate(row)
