"""平台基础配置：名称、口号、Logo"""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models import PlatformSettings

DEFAULT_NAME = "AI视频监控平台"
DEFAULT_SLOGAN = "深瞳洞察每一帧 · 云端决策每一秒"
SETTINGS_ID = 1


def ensure_platform_settings(db: Session) -> PlatformSettings:
    row = db.get(PlatformSettings, SETTINGS_ID)
    if row is None:
        row = PlatformSettings(
            id=SETTINGS_ID,
            platform_name=DEFAULT_NAME,
            slogan=DEFAULT_SLOGAN,
            logo_url=None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
    return row


def get_platform_settings(db: Session) -> PlatformSettings:
    return ensure_platform_settings(db)


def update_platform_settings(
    db: Session,
    *,
    platform_name: Optional[str] = None,
    slogan: Optional[str] = None,
    logo_url: Optional[str] = None,
    clear_logo: bool = False,
) -> PlatformSettings:
    row = ensure_platform_settings(db)
    if platform_name is not None:
        name = platform_name.strip()
        if not name:
            raise ValueError("平台名称不能为空")
        if len(name) > 128:
            raise ValueError("平台名称过长")
        row.platform_name = name
    if slogan is not None:
        row.slogan = slogan.strip()[:255]
    if clear_logo:
        row.logo_url = None
    elif logo_url is not None:
        row.logo_url = logo_url.strip() or None
    db.commit()
    db.refresh(row)
    return row


def upload_platform_logo(db: Session, data: bytes, content_type: str) -> PlatformSettings:
    from app.services.minio_client import minio_storage

    ct = (content_type or "image/png").split(";")[0].strip().lower()
    allowed = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/svg+xml": ".svg",
    }
    if ct not in allowed:
        raise ValueError("仅支持 png / jpg / webp / gif / svg 图片")
    if not data:
        raise ValueError("空文件")
    if len(data) > 2 * 1024 * 1024:
        raise ValueError("Logo 文件不能超过 2MB")

    ext = allowed[ct]
    object_name = f"platform/logo_{uuid.uuid4().hex}{ext}"
    url = minio_storage.upload_bytes(data, object_name, ct)
    return update_platform_settings(db, logo_url=url)
