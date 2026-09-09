"""管理台 API Schema"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------- 煤矿 / 地点 / 摄像头树 ----------

class MineCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    code: str = Field(default="", max_length=16, description="煤矿编码，建议 12 位")
    enabled: bool = True
    remark: str = Field(default="", max_length=255)


class MineUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    code: Optional[str] = Field(default=None, max_length=16)
    enabled: Optional[bool] = None
    remark: Optional[str] = Field(default=None, max_length=255)


class MineOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    code: str
    enabled: bool
    remark: str
    created_at: datetime
    updated_at: datetime
    site_count: int = 0
    camera_count: int = 0


class SiteCreate(BaseModel):
    mine_id: int
    name: str = Field(min_length=1, max_length=128)
    code: str = Field(default="", max_length=32)
    enabled: bool = True
    remark: str = Field(default="", max_length=255)


class SiteUpdate(BaseModel):
    mine_id: Optional[int] = None
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    code: Optional[str] = Field(default=None, max_length=32)
    enabled: Optional[bool] = None
    remark: Optional[str] = Field(default=None, max_length=255)


class SiteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    mine_id: int
    name: str
    code: str
    enabled: bool
    remark: str
    created_at: datetime
    updated_at: datetime
    camera_count: int = 0
    mine_name: Optional[str] = None
    mine_code: Optional[str] = None


class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    # push：可填 in_url，或填 zlm_app+zlm_stream；proxy：填 source_url
    ingest_mode: Literal["push", "proxy"] = "push"
    in_url: str = Field(default="", max_length=512)
    source_url: Optional[str] = Field(default=None, max_length=512)
    zlm_app: str = Field(default="", max_length=64)
    zlm_stream: str = Field(default="", max_length=128)
    site_id: Optional[int] = None
    camera_code: str = Field(default="", max_length=32)
    mine_code: str = Field(default="", max_length=16)
    enabled: bool = True
    remark: str = Field(default="", max_length=255)


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    ingest_mode: Optional[Literal["push", "proxy"]] = None
    in_url: Optional[str] = Field(default=None, max_length=512)
    source_url: Optional[str] = Field(default=None, max_length=512)
    zlm_app: Optional[str] = Field(default=None, max_length=64)
    zlm_stream: Optional[str] = Field(default=None, max_length=128)
    site_id: Optional[int] = None
    camera_code: Optional[str] = Field(default=None, max_length=32)
    mine_code: Optional[str] = Field(default=None, max_length=16)
    enabled: Optional[bool] = None
    remark: Optional[str] = Field(default=None, max_length=255)


class CameraOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    in_url: str
    ingest_mode: str = "push"
    source_url: Optional[str] = None
    zlm_app: str = ""
    zlm_stream: str = ""
    proxy_key: Optional[str] = None
    camera_code: str
    mine_code: str
    site_id: Optional[int] = None
    site_name: Optional[str] = None
    mine_id: Optional[int] = None
    mine_name: Optional[str] = None
    enabled: bool
    remark: str
    created_at: datetime
    updated_at: datetime
    # ZLM getMediaList 判定；非流地址为 null
    online: Optional[bool] = None
    stream_app: Optional[str] = None
    stream_id: Optional[str] = None
    online_check_error: Optional[str] = None
    flv_url: Optional[str] = None


class CameraTreeSiteNode(BaseModel):
    id: int
    name: str
    code: str
    enabled: bool
    cameras: List[CameraOut] = Field(default_factory=list)


class CameraTreeMineNode(BaseModel):
    id: int
    name: str
    code: str
    enabled: bool
    sites: List[CameraTreeSiteNode] = Field(default_factory=list)


class CameraTreeResponse(BaseModel):
    mines: List[CameraTreeMineNode]
    unassigned: List[CameraOut] = Field(
        default_factory=list, description="未挂到地点的摄像头"
    )


class MineListResponse(BaseModel):
    items: List[MineOut]


class SiteListResponse(BaseModel):
    items: List[SiteOut]


class AlgorithmConfigCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    skill_name: str = Field(min_length=1, max_length=128)
    gate_direction: Literal["IN", "OUT"] = "IN"
    enter_count: int = 0
    count_line: Optional[Dict[str, Any]] = None
    bypass_line: Optional[Dict[str, Any]] = None
    extra_params: Optional[Dict[str, Any]] = None
    enabled: bool = True
    remark: str = Field(default="", max_length=255)


class AlgorithmConfigUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    skill_name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    gate_direction: Optional[Literal["IN", "OUT"]] = None
    enter_count: Optional[int] = None
    count_line: Optional[Dict[str, Any]] = None
    bypass_line: Optional[Dict[str, Any]] = None
    extra_params: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None
    remark: Optional[str] = Field(default=None, max_length=255)


class AlgorithmConfigOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    skill_name: str
    gate_direction: str
    enter_count: int
    count_line: Optional[Dict[str, Any]] = None
    bypass_line: Optional[Dict[str, Any]] = None
    extra_params: Optional[Dict[str, Any]] = None
    enabled: bool
    remark: str
    created_at: datetime
    updated_at: datetime


class TaskSchedule(BaseModel):
    """任务运行时间段（按周几 + 每日起止时间）。"""

    enabled: bool = False
    timezone: str = Field(default="Asia/Shanghai", max_length=64)
    days: List[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6, 7],
        description="ISO 星期：周一=1 … 周日=7",
    )
    start_time: str = Field(default="09:00", description="开始时间 HH:MM")
    end_time: str = Field(default="18:00", description="结束时间 HH:MM（可跨午夜）")


class TaskConfigCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    camera_id: int
    algorithm_config_id: int
    scene_id: str = Field(min_length=1, max_length=64)
    output_format: Literal["rtmp", "rtsp"] = "rtmp"
    out_fps: int = Field(default=15, ge=1, le=60)
    enabled: bool = True
    alert_image_enabled: bool = Field(default=True, description="是否保存报警图片")
    alert_video_enabled: bool = Field(default=False, description="是否截取报警视频")
    push_annotated_stream: bool = Field(
        default=False, description="是否推送 AI 画框识别结果视频"
    )
    schedule: Optional[TaskSchedule] = None
    remark: str = Field(default="", max_length=255)


class TaskConfigUpdate(BaseModel):
    name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    camera_id: Optional[int] = None
    algorithm_config_id: Optional[int] = None
    scene_id: Optional[str] = Field(default=None, min_length=1, max_length=64)
    output_format: Optional[Literal["rtmp", "rtsp"]] = None
    out_fps: Optional[int] = Field(default=None, ge=1, le=60)
    enabled: Optional[bool] = None
    alert_image_enabled: Optional[bool] = None
    alert_video_enabled: Optional[bool] = None
    push_annotated_stream: Optional[bool] = None
    schedule: Optional[TaskSchedule] = None
    remark: Optional[str] = Field(default=None, max_length=255)


class TaskConfigOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    camera_id: int
    algorithm_config_id: int
    scene_id: str
    output_format: str
    out_fps: int
    enabled: bool
    alert_image_enabled: bool = True
    alert_video_enabled: bool = False
    push_annotated_stream: bool = False
    schedule: Optional[Dict[str, Any]] = None
    schedule_active: bool = False
    last_runtime_task_id: Optional[str] = None
    remark: str
    created_at: datetime
    updated_at: datetime
    camera_name: Optional[str] = None
    algorithm_name: Optional[str] = None
    skill_name: Optional[str] = None
    runtime_status: Optional[str] = None
    flv_url: Optional[str] = None


class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    alert_uid: str
    scene_id: str
    skill_name: str
    recognition_types: Optional[List[str]] = None
    message: str
    count: int
    enter_count: int
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    category: str = "alert"
    payload: Optional[Dict[str, Any]] = None
    status: str
    created_at: datetime


class AlertStatusUpdate(BaseModel):
    status: Literal["new", "acked", "closed"]


class PageMeta(BaseModel):
    total: int
    page: int
    page_size: int


class CameraListResponse(BaseModel):
    items: List[CameraOut]
    meta: PageMeta


class AlgorithmConfigListResponse(BaseModel):
    items: List[AlgorithmConfigOut]
    meta: PageMeta


class TaskConfigListResponse(BaseModel):
    items: List[TaskConfigOut]
    meta: PageMeta


class AlertListResponse(BaseModel):
    items: List[AlertOut]
    meta: PageMeta


class PartnerIntegrationCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    code: str = Field(..., min_length=1, max_length=64, description="接入方唯一编码")
    webhook_url: str = Field("", max_length=512, description="Webhook 回调 URL")
    enabled: bool = True
    subscribe_categories: Optional[List[str]] = Field(
        default=None, description='订阅类别，如 ["alert"]；空则默认仅报警'
    )
    subscribe_recognition_types: Optional[List[str]] = Field(
        default=None, description='订阅识别类型，如 ["04","05"]；空表示不限'
    )
    subscribe_scene_ids: Optional[List[str]] = Field(
        default=None, description="订阅场景 ID；空表示不限"
    )
    remark: str = ""


class PartnerIntegrationUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=128)
    webhook_url: Optional[str] = Field(None, max_length=512)
    enabled: Optional[bool] = None
    subscribe_categories: Optional[List[str]] = None
    subscribe_recognition_types: Optional[List[str]] = None
    subscribe_scene_ids: Optional[List[str]] = None
    remark: Optional[str] = None
    rotate_api_key: bool = Field(False, description="为 true 时重新生成 API Key")
    rotate_webhook_secret: bool = Field(
        False, description="为 true 时重新生成 Webhook 签名密钥"
    )


class PartnerIntegrationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    code: str
    webhook_url: str
    webhook_secret: str
    api_key: str
    enabled: bool
    subscribe_categories: Optional[List[str]] = None
    subscribe_recognition_types: Optional[List[str]] = None
    subscribe_scene_ids: Optional[List[str]] = None
    remark: str
    created_at: datetime
    updated_at: datetime


class PartnerIntegrationListResponse(BaseModel):
    items: List[PartnerIntegrationOut]
    meta: PageMeta


class TritonModelOut(BaseModel):
    name: str
    version: Optional[str] = None
    state: Optional[str] = None
    ready: bool = False


class TritonModelListResponse(BaseModel):
    server_url: str
    server_live: bool
    server_ready: bool
    server_name: Optional[str] = None
    server_version: Optional[str] = None
    items: List[TritonModelOut]
    error: Optional[str] = None
