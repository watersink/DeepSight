"""管理台数据模型"""
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class User(Base):
    """管理台登录用户"""

    __tablename__ = "mgmt_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    display_name: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    # admin=用户管理+全部操作；user=业务操作（无用户管理）
    role: Mapped[str] = mapped_column(String(16), default="user", nullable=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    remark: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Mine(Base):
    """煤矿（树顶层）"""

    __tablename__ = "mgmt_mines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    code: Mapped[str] = mapped_column(String(16), default="", nullable=False, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    remark: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    sites: Mapped[List["Site"]] = relationship(
        "Site", back_populates="mine", cascade="all, delete-orphan"
    )


class Site(Base):
    """地点/区域（挂在煤矿下）"""

    __tablename__ = "mgmt_sites"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mine_id: Mapped[int] = mapped_column(
        ForeignKey("mgmt_mines.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    code: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    remark: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    mine: Mapped["Mine"] = relationship("Mine", back_populates="sites")
    cameras: Mapped[List["Camera"]] = relationship(
        "Camera", back_populates="site"
    )


class Camera(Base):
    __tablename__ = "mgmt_cameras"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    camera_code: Mapped[str] = mapped_column(String(32), default="", nullable=False)
    mine_code: Mapped[str] = mapped_column(String(16), default="", nullable=False)
    # 归属地点；历史数据可为空，未挂树
    site_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("mgmt_sites.id", ondelete="SET NULL"), nullable=True, index=True
    )
    in_url: Mapped[str] = mapped_column(String(512), nullable=False)
    # push=已在 ZLM 上推流，仅登记地址；proxy=摄像头原始地址经 addStreamProxy 拉入 ZLM
    ingest_mode: Mapped[str] = mapped_column(
        String(16), default="push", nullable=False
    )
    # 摄像头原始流地址（proxy 模式必填）
    source_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    zlm_app: Mapped[str] = mapped_column(String(64), default="", nullable=False)
    zlm_stream: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    # addStreamProxy 返回的 key，删除/更新时 delStreamProxy
    proxy_key: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    remark: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    site: Mapped[Optional["Site"]] = relationship("Site", back_populates="cameras")


class AlgorithmConfig(Base):
    __tablename__ = "mgmt_algorithm_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    skill_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    gate_direction: Mapped[str] = mapped_column(String(8), default="IN", nullable=False)
    enter_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    count_line: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    bypass_line: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    extra_params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    remark: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class TaskConfig(Base):
    __tablename__ = "mgmt_task_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    camera_id: Mapped[int] = mapped_column(
        ForeignKey("mgmt_cameras.id"), nullable=False, index=True
    )
    algorithm_config_id: Mapped[int] = mapped_column(
        ForeignKey("mgmt_algorithm_configs.id"), nullable=False, index=True
    )
    scene_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    output_format: Mapped[str] = mapped_column(String(16), default="rtmp", nullable=False)
    out_fps: Mapped[int] = mapped_column(Integer, default=15, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # 报警图片：默认需要；报警视频截取：默认不需要；AI 画框推流：默认不需要
    alert_image_enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    alert_video_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    push_annotated_stream: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    # 运行时间窗：{enabled, timezone, days[1-7], start_time, end_time}
    schedule: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    last_runtime_task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    remark: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )

    camera: Mapped["Camera"] = relationship("Camera")
    algorithm_config: Mapped["AlgorithmConfig"] = relationship("AlgorithmConfig")


class AlertRecord(Base):
    __tablename__ = "mgmt_alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    alert_uid: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    scene_id: Mapped[str] = mapped_column(String(64), default="", nullable=False, index=True)
    skill_name: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    recognition_types: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    enter_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    image_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    video_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    # alert=违规报警(04-07)；event=正常事件(01/02、画面人数等)
    category: Mapped[str] = mapped_column(
        String(16), default="alert", nullable=False, index=True
    )
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="new", nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, index=True
    )


class PartnerIntegration(Base):
    """第三方接入配置：Webhook 推送 + Open API 拉取"""

    __tablename__ = "mgmt_partners"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    code: Mapped[str] = mapped_column(
        String(64), unique=True, nullable=False, index=True
    )
    webhook_url: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    webhook_secret: Mapped[str] = mapped_column(String(128), default="", nullable=False)
    api_key: Mapped[str] = mapped_column(
        String(128), unique=True, nullable=False, index=True
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    # 订阅类别：["alert"] / ["event"] / ["alert","event"]；空则默认 ["alert"]
    subscribe_categories: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    # 订阅识别类型：["04","05"]；空列表/null 表示不限
    subscribe_recognition_types: Mapped[Optional[Any]] = mapped_column(
        JSON, nullable=True
    )
    # 订阅场景：["scene_001"]；空列表/null 表示不限
    subscribe_scene_ids: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    remark: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


class PlatformSettings(Base):
    """平台基础配置（单行，id 固定为 1）"""

    __tablename__ = "mgmt_platform_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    platform_name: Mapped[str] = mapped_column(
        String(128), default="AI视频监控平台", nullable=False
    )
    slogan: Mapped[str] = mapped_column(
        String(255),
        default="深瞳洞察每一帧 · 云端决策每一秒",
        nullable=False,
    )
    logo_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
