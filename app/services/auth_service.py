"""用户与登录业务"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from app.db.models import User

logger = logging.getLogger(__name__)

VALID_ROLES = {"admin", "user"}


def _page(items: list, total: int, page: int, page_size: int) -> Tuple[list, dict]:
    return items, {"total": total, "page": page, "page_size": page_size}


def user_to_dict(row: User) -> dict:
    return {
        "id": row.id,
        "username": row.username,
        "display_name": row.display_name,
        "role": row.role,
        "enabled": row.enabled,
        "remark": row.remark,
        "last_login_at": row.last_login_at,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    return db.get(User, user_id)


def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.scalar(select(User).where(User.username == username.strip()))


def authenticate(db: Session, username: str, password: str) -> Optional[User]:
    user = get_user_by_username(db, username)
    if not user or not user.enabled:
        return None
    if not verify_password(password, user.password_hash):
        return None
    user.last_login_at = datetime.now()
    db.commit()
    db.refresh(user)
    return user


def issue_token(user: User) -> dict:
    token = create_access_token(
        subject=str(user.id),
        extra={"username": user.username, "role": user.role},
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "user": user_to_dict(user),
    }


def list_users(
    db: Session, page: int = 1, page_size: int = 20
) -> Tuple[List[dict], dict]:
    total = db.scalar(select(func.count()).select_from(User)) or 0
    rows = db.scalars(
        select(User)
        .order_by(User.id.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return _page([user_to_dict(r) for r in rows], int(total), page, page_size)


def create_user(db: Session, data: dict) -> User:
    username = str(data["username"]).strip()
    if not username:
        raise ValueError("用户名不能为空")
    if get_user_by_username(db, username):
        raise ValueError(f"用户名已存在: {username}")
    role = str(data.get("role") or "user").strip().lower()
    if role not in VALID_ROLES:
        raise ValueError("role 仅支持 admin / user")
    password = str(data.get("password") or "")
    if len(password) < 6:
        raise ValueError("密码至少 6 位")
    row = User(
        username=username,
        display_name=str(data.get("display_name") or username).strip(),
        password_hash=hash_password(password),
        role=role,
        enabled=bool(data.get("enabled", True)),
        remark=str(data.get("remark") or ""),
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def update_user(db: Session, row: User, data: dict) -> User:
    if "display_name" in data and data["display_name"] is not None:
        row.display_name = str(data["display_name"]).strip()
    if "role" in data and data["role"] is not None:
        role = str(data["role"]).strip().lower()
        if role not in VALID_ROLES:
            raise ValueError("role 仅支持 admin / user")
        row.role = role
    if "enabled" in data and data["enabled"] is not None:
        row.enabled = bool(data["enabled"])
    if "remark" in data and data["remark"] is not None:
        row.remark = str(data["remark"])
    if "password" in data and data["password"]:
        password = str(data["password"])
        if len(password) < 6:
            raise ValueError("密码至少 6 位")
        row.password_hash = hash_password(password)
    db.commit()
    db.refresh(row)
    return row


def delete_user(db: Session, row: User, *, actor_id: int) -> None:
    if row.id == actor_id:
        raise ValueError("不能删除当前登录用户")
    admin_count = db.scalar(
        select(func.count()).select_from(User).where(User.role == "admin", User.enabled.is_(True))
    ) or 0
    if row.role == "admin" and admin_count <= 1:
        raise ValueError("不能删除最后一个启用的管理员")
    db.delete(row)
    db.commit()


def change_password(db: Session, user: User, old_password: str, new_password: str) -> None:
    if not verify_password(old_password, user.password_hash):
        raise ValueError("原密码不正确")
    if len(new_password) < 6:
        raise ValueError("新密码至少 6 位")
    user.password_hash = hash_password(new_password)
    db.commit()


def seed_default_admin() -> None:
    """库为空时创建默认管理员。"""
    from app.db import SessionLocal

    db = SessionLocal()
    try:
        exists = db.scalar(select(func.count()).select_from(User)) or 0
        if exists:
            return
        username = (settings.AUTH_DEFAULT_ADMIN_USERNAME or "admin").strip()
        password = settings.AUTH_DEFAULT_ADMIN_PASSWORD or "Admin@123"
        row = User(
            username=username,
            display_name="系统管理员",
            password_hash=hash_password(password),
            role="admin",
            enabled=True,
            remark="系统首次启动自动创建，请尽快修改密码",
        )
        db.add(row)
        db.commit()
        logger.info("已创建默认管理员账号: %s", username)
    except Exception:
        db.rollback()
        logger.exception("创建默认管理员失败")
        raise
    finally:
        db.close()
