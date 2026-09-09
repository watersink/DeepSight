"""认证与用户管理 Schema"""
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    display_name: str
    role: str
    enabled: bool
    remark: str
    last_login_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=6, max_length=128)


class UserCreate(BaseModel):
    username: str = Field(min_length=2, max_length=64)
    password: str = Field(min_length=6, max_length=128)
    display_name: str = Field(default="", max_length=128)
    role: Literal["admin", "user"] = "user"
    enabled: bool = True
    remark: str = Field(default="", max_length=255)


class UserUpdate(BaseModel):
    display_name: Optional[str] = Field(default=None, max_length=128)
    role: Optional[Literal["admin", "user"]] = None
    enabled: Optional[bool] = None
    remark: Optional[str] = Field(default=None, max_length=255)
    password: Optional[str] = Field(default=None, min_length=6, max_length=128)


class PageMeta(BaseModel):
    total: int
    page: int
    page_size: int


class UserListResponse(BaseModel):
    items: List[UserOut]
    meta: PageMeta
