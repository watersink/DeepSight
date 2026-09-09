"""第三方接入配置 CRUD"""
from __future__ import annotations

import re
import secrets
from typing import List, Optional, Tuple

from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from app.db.models import PartnerIntegration


_CODE_RE = re.compile(r"^[a-zA-Z0-9_-]{2,64}$")


def _new_api_key() -> str:
    return f"pk_{secrets.token_urlsafe(32)}"


def _new_webhook_secret() -> str:
    return f"whsec_{secrets.token_urlsafe(24)}"


def _page(items: list, total: int, page: int, page_size: int) -> Tuple[list, dict]:
    return items, {
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def list_partners(
    db: Session, page: int = 1, page_size: int = 20
) -> Tuple[List[PartnerIntegration], dict]:
    total = db.scalar(select(func.count()).select_from(PartnerIntegration)) or 0
    rows = list(
        db.scalars(
            select(PartnerIntegration)
            .order_by(desc(PartnerIntegration.id))
            .offset((page - 1) * page_size)
            .limit(page_size)
        ).all()
    )
    return _page(rows, int(total), page, page_size)


def get_partner(db: Session, partner_id: int) -> Optional[PartnerIntegration]:
    return db.get(PartnerIntegration, partner_id)


def get_partner_by_api_key(db: Session, api_key: str) -> Optional[PartnerIntegration]:
    key = (api_key or "").strip()
    if not key:
        return None
    return db.scalar(
        select(PartnerIntegration).where(
            PartnerIntegration.api_key == key,
            PartnerIntegration.enabled.is_(True),
        )
    )


def list_enabled_partners(db: Session) -> List[PartnerIntegration]:
    return list(
        db.scalars(
            select(PartnerIntegration).where(PartnerIntegration.enabled.is_(True))
        ).all()
    )


def create_partner(db: Session, data: dict) -> PartnerIntegration:
    code = str(data.get("code") or "").strip()
    if not _CODE_RE.match(code):
        raise ValueError("code 仅允许字母数字、下划线、短横线，长度 2–64")
    exists = db.scalar(
        select(PartnerIntegration.id).where(PartnerIntegration.code == code)
    )
    if exists:
        raise ValueError(f"接入编码已存在: {code}")

    row = PartnerIntegration(
        name=str(data.get("name") or "").strip(),
        code=code,
        webhook_url=str(data.get("webhook_url") or "").strip(),
        webhook_secret=_new_webhook_secret(),
        api_key=_new_api_key(),
        enabled=bool(data.get("enabled", True)),
        subscribe_categories=data.get("subscribe_categories"),
        subscribe_recognition_types=data.get("subscribe_recognition_types"),
        subscribe_scene_ids=data.get("subscribe_scene_ids"),
        remark=str(data.get("remark") or ""),
    )
    if not row.name:
        raise ValueError("名称不能为空")
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def update_partner(
    db: Session, row: PartnerIntegration, data: dict
) -> PartnerIntegration:
    if "name" in data and data["name"] is not None:
        name = str(data["name"]).strip()
        if not name:
            raise ValueError("名称不能为空")
        row.name = name
    if "webhook_url" in data and data["webhook_url"] is not None:
        row.webhook_url = str(data["webhook_url"]).strip()
    if "enabled" in data and data["enabled"] is not None:
        row.enabled = bool(data["enabled"])
    if "subscribe_categories" in data:
        row.subscribe_categories = data["subscribe_categories"]
    if "subscribe_recognition_types" in data:
        row.subscribe_recognition_types = data["subscribe_recognition_types"]
    if "subscribe_scene_ids" in data:
        row.subscribe_scene_ids = data["subscribe_scene_ids"]
    if "remark" in data and data["remark"] is not None:
        row.remark = str(data["remark"])
    if data.get("rotate_api_key"):
        row.api_key = _new_api_key()
    if data.get("rotate_webhook_secret"):
        row.webhook_secret = _new_webhook_secret()
    db.commit()
    db.refresh(row)
    return row


def delete_partner(db: Session, row: PartnerIntegration) -> None:
    db.delete(row)
    db.commit()
