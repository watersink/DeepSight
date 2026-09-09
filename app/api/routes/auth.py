"""认证与用户管理 API"""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.auth_schemas import (
    ChangePasswordRequest,
    LoginRequest,
    PageMeta,
    TokenResponse,
    UserCreate,
    UserListResponse,
    UserOut,
    UserUpdate,
)
from app.api.deps import get_current_user, require_admin
from app.db import get_db
from app.db.models import User
from app.services import auth_service as auth_svc

router = APIRouter(tags=["认证与用户"])


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="登录",
    description="用户名密码登录，返回 JWT access_token。",
)
def login(body: LoginRequest, db: Annotated[Session, Depends(get_db)]):
    user = auth_svc.authenticate(db, body.username, body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误，或账号已禁用",
        )
    return TokenResponse(**auth_svc.issue_token(user))


@router.get("/auth/me", response_model=UserOut, summary="当前用户信息")
def me(user: Annotated[User, Depends(get_current_user)]):
    return UserOut.model_validate(auth_svc.user_to_dict(user))


@router.post("/auth/change-password", summary="修改当前用户密码")
def change_password(
    body: ChangePasswordRequest,
    db: Annotated[Session, Depends(get_db)],
    user: Annotated[User, Depends(get_current_user)],
):
    try:
        auth_svc.change_password(db, user, body.old_password, body.new_password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True}


@router.get(
    "/users",
    response_model=UserListResponse,
    summary="用户列表（管理员）",
)
def list_users(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
):
    items, meta = auth_svc.list_users(db, page, page_size)
    return UserListResponse(
        items=[UserOut.model_validate(i) for i in items],
        meta=PageMeta(**meta),
    )


@router.post(
    "/users",
    response_model=UserOut,
    status_code=201,
    summary="创建用户（管理员）",
)
def create_user(
    body: UserCreate,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    try:
        row = auth_svc.create_user(db, body.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return UserOut.model_validate(auth_svc.user_to_dict(row))


@router.patch(
    "/users/{user_id}",
    response_model=UserOut,
    summary="更新用户（管理员）",
)
def update_user(
    user_id: int,
    body: UserUpdate,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    row = auth_svc.get_user_by_id(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="用户不存在")
    try:
        row = auth_svc.update_user(db, row, body.model_dump(exclude_unset=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return UserOut.model_validate(auth_svc.user_to_dict(row))


@router.delete("/users/{user_id}", summary="删除用户（管理员）")
def delete_user(
    user_id: int,
    db: Annotated[Session, Depends(get_db)],
    actor: Annotated[User, Depends(require_admin)],
):
    row = auth_svc.get_user_by_id(db, user_id)
    if not row:
        raise HTTPException(status_code=404, detail="用户不存在")
    try:
        auth_svc.delete_user(db, row, actor_id=actor.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True}
