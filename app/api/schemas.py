"""API 请求/响应模型（OpenAPI / Swagger 文档）"""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import settings

# ---------------------------------------------------------------------------
# 通用
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """HTTP 错误响应（FastAPI 默认 detail 字段）"""

    detail: str = Field(description="错误说明")


class HealthResponse(BaseModel):
    """服务健康检查响应"""

    status: str = Field(description="服务状态，正常为 ok", examples=["ok"])
    service: str = Field(description="服务名称")
    running_tasks: int = Field(description="当前运行中的推流任务数量", ge=0)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "service": "Video Detection Stream",
                    "running_tasks": 1,
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# 技能
# ---------------------------------------------------------------------------

class SkillInfoResponse(BaseModel):
    """已注册技能信息"""

    skill_name: str = Field(description="技能唯一标识，用于启动推流时指定")
    name_zh: Optional[str] = Field(default=None, description="技能中文名称")
    description: Optional[str] = Field(default=None, description="技能功能描述")
    required_models: List[str] = Field(
        default_factory=list,
        description="依赖的 Triton 模型名称列表",
    )
    cover_image: Optional[str] = Field(
        default=None,
        description="平铺展示封面图路径（前端静态资源，如 /skills/xxx.png）",
    )


class SkillListResponse(BaseModel):
    """技能列表响应"""

    skills: List[SkillInfoResponse] = Field(description="可用技能列表")


# ---------------------------------------------------------------------------
# 视频流推流
# ---------------------------------------------------------------------------

class CountLinePoint(BaseModel):
    """过线计数线坐标点（像素）"""

    x: float = Field(description="横坐标（像素）", examples=[432])
    y: float = Field(description="纵坐标（像素）", examples=[328])


class CountLineSegment(BaseModel):
    """单条计数线段及其对应进入侧参考点"""

    line: List[CountLinePoint] = Field(
        description="计数线段两端点",
        min_length=2,
        examples=[[{"x": 432, "y": 328}, {"x": 723, "y": 229}]],
    )
    enter_point: CountLinePoint = Field(
        description="与该线段对应的进入侧参考点",
        examples=[{"x": 513, "y": 250}],
    )


class CountLineConfig(BaseModel):
    """
    过线计数线配置（可选，像素坐标，不做归一化）。

    线段与 enter_point 一一对应；仅部分技能需要（如 `person_count_detector26`）。
    """

    line: Optional[List[CountLinePoint]] = Field(
        default=None,
        description="单条计数线段（兼容旧格式，需配合顶层 enter_point）",
        min_length=2,
    )
    enter_point: Optional[CountLinePoint] = Field(
        default=None,
        description="单条计数线对应的进入侧参考点（兼容旧格式）",
    )
    lines: Optional[List[CountLineSegment]] = Field(
        default=None,
        description="多条计数线段，每项包含 line 与对应的 enter_point",
    )


class BypassLineSegment(BaseModel):
    """单条绕行线段及其对应参考点"""

    line: List[CountLinePoint] = Field(
        description="绕行线段两端点",
        min_length=2,
        examples=[[{"x": 154, "y": 421}, {"x": 434, "y": 337}]],
    )
    bypass_point: CountLinePoint = Field(
        description="与该线段对应的绕行侧参考点",
        examples=[{"x": 280, "y": 348}],
    )


class BypassLineConfig(BaseModel):
    """
    非计数绕行线配置（可选，像素坐标，不做归一化）。

    线段与 bypass_point 一一对应；穿过任一段视为绕行违规。
    """

    line: Optional[List[CountLinePoint]] = Field(
        default=None,
        description="单条绕行线段（兼容旧格式，需配合顶层 bypass_point）",
        min_length=2,
    )
    bypass_point: Optional[CountLinePoint] = Field(
        default=None,
        description="单条绕行线对应的参考点（兼容旧格式）",
    )
    lines: Optional[List[BypassLineSegment]] = Field(
        default=None,
        description="多条绕行线段，每项包含 line 与对应的 bypass_point",
    )


_COUNT_LINE_EXAMPLE = {
    "lines": [
        {
            "line": [{"x": 432, "y": 328}, {"x": 723, "y": 229}],
            "enter_point": {"x": 513, "y": 250},
        },
    ],
}

_BYPASS_LINE_EXAMPLE = {
    "lines": [
        {
            "line": [{"x": 154, "y": 421}, {"x": 434, "y": 337}],
            "bypass_point": {"x": 280, "y": 348},
        },
    ],
}


class StreamStartRequest(BaseModel):
    """启动推流任务请求"""

    in_url: str = Field(
        default_factory=settings.resolve_test_video_path,
        description="输入视频源地址（RTSP / RTMP / 本地文件路径）",
        examples=["test_images_videos/person_gouzi.mp4"],
    )
    out_url: Optional[str] = Field(
        default=None,
        description=(
            "输出推流地址；省略时自动拼装为 "
            f"{settings.build_push_url()}（即 {{scene_id}}/{{skill_name}}）"
        ),
    )
    out_fps: int = Field(
        default=settings.DEFAULT_OUT_FPS,
        ge=1,
        le=60,
        description="输出帧率",
    )
    alarm_interval: int = Field(
        default=settings.DEFAULT_ALARM_INTERVAL,
        ge=0,
        description="告警间隔（秒），0 表示不限制",
    )
    scene_id: str = Field(
        default=settings.DEFAULT_SCENE_ID,
        description="场景 ID，对应 ZLMediaKit 推流 app 段",
        examples=["scene_001"],
    )
    output_format: Literal["rtmp", "rtsp"] = Field(
        default=settings.DEFAULT_OUTPUT_FORMAT,
        description="推流协议",
    )
    queue_size: int = Field(
        default=settings.STREAM_QUEUE_SIZE,
        ge=1,
        le=32,
        description="帧缓冲队列大小",
    )
    use_hardware_encoding: bool = Field(
        default=settings.STREAM_USE_HARDWARE_ENCODING,
        description="是否使用 NVENC 硬件编码",
    )
    fence_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="电子围栏配置（技能相关 JSON 对象）",
    )
    bitrate: str = Field(default=settings.STREAM_BITRATE, description="推流码率，如 2M")
    buffer_size: str = Field(default=settings.STREAM_BUFFER_SIZE, description="推流缓冲区大小")
    reconnect_delay: float = Field(
        default=settings.STREAM_RECONNECT_DELAY,
        ge=0.5,
        description="视频源断线重连间隔（秒）",
    )
    skill_name: str = Field(
        default=settings.DEFAULT_SKILL_NAME,
        description="技能名称，须为已注册技能",
        examples=["person_count_detector26", "person_count_detector"],
    )
    skill_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="技能可选覆盖配置（合并到技能 DEFAULT_CONFIG）",
    )
    enter_count: Optional[int] = Field(
        default=None,
        description=(
            "过线进入人数初始值（如 0）。部分技能需要（如 person_count_detector26），"
            "不需要的技能可省略；传入时写入 skill_config.params.enter_count"
        ),
        examples=[0],
    )
    gate_direction: Optional[str] = Field(
        default=None,
        description=(
            "闸机方向：IN=入闸机口，OUT=出闸机口。"
            "影响画面 enter_point 标注文字；部分技能需要（如 person_count_detector26）"
        ),
        examples=["IN", "OUT"],
    )
    mine_code: Optional[str] = Field(
        default=None,
        description=(
            "煤矿编码（12位数字）。用于人数识别结果上传；"
            "省略时使用配置项 COUNTING_RECOG_MINE_CODE"
        ),
        examples=["140921021337"],
    )
    camera_code: Optional[str] = Field(
        default=None,
        description=(
            "摄像仪编码（20位，参照 MT/T 1201.6-2023）。用于人数识别结果上传；"
            "省略时使用配置项 COUNTING_RECOG_CAMERA_CODE"
        ),
        examples=["14092102133701010101"],
    )
    count_line: Optional[CountLineConfig] = Field(
        default=None,
        description=(
            "过线计数线配置（像素坐标）。person_count_detector26 必填；"
            "未上传或为空将启动失败。"
            "`lines` 中每段 line 与 enter_point 一一对应"
        ),
        examples=[_COUNT_LINE_EXAMPLE],
    )
    bypass_line: Optional[BypassLineConfig] = Field(
        default=None,
        description=(
            "非计数绕行线配置（可选，像素坐标）。未上传或为空则不做绕行检测；"
            "上传后穿过任一段视为绕行违规。"
            "`lines` 中每段 line 与 bypass_point 一一对应"
        ),
        examples=[_BYPASS_LINE_EXAMPLE],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "in_url": "test_images_videos/person_gouzi.mp4",
                    "scene_id": settings.DEFAULT_SCENE_ID,
                    "skill_name": settings.DEFAULT_SKILL_NAME,
                    "output_format": settings.DEFAULT_OUTPUT_FORMAT,
                    "out_fps": settings.DEFAULT_OUT_FPS,
                },
                {
                    "in_url": "rtmp://10.1.3.21:1935/live/stream01",
                    "scene_id": "scene_001",
                    "skill_name": "person_count_detector26",
                    "output_format": "rtmp",
                    "out_fps": settings.DEFAULT_OUT_FPS,
                    "enter_count": 0,
                    "gate_direction": "IN",
                    "count_line": _COUNT_LINE_EXAMPLE,
                    "bypass_line": _BYPASS_LINE_EXAMPLE,
                },
            ]
        }
    )


class StreamTaskResponse(BaseModel):
    """推流任务状态"""

    task_id: str = Field(description="任务唯一 ID")
    status: str = Field(
        description="任务状态：pending / running / stopped / failed",
        examples=["running"],
    )
    scene_id: Optional[str] = Field(default=None, description="场景 ID")
    skill_name: Optional[str] = Field(default=None, description="技能名称")
    in_url: Optional[str] = Field(default=None, description="输入视频源")
    out_url: Optional[str] = Field(default=None, description="输出推流地址")
    flv_url: Optional[str] = Field(
        default=None,
        description=(
            "out_url 对应的 HTTP-FLV 播放地址。"
            "例：rtmp://host/scene_001/xxx → http://host:8080/scene_001/xxx.live.flv"
        ),
        examples=[
            f"http://{settings.ZLM_HOST}:{settings.ZLM_HTTP_PORT}/scene_001/person_count_detector26.live.flv"
        ],
    )
    pid: Optional[int] = Field(default=None, description="子进程 PID")
    created_at: Optional[str] = Field(default=None, description="创建时间（ISO 8601）")
    started_at: Optional[str] = Field(default=None, description="启动时间（ISO 8601）")
    stopped_at: Optional[str] = Field(default=None, description="停止时间（ISO 8601）")
    error_message: Optional[str] = Field(default=None, description="失败时的错误信息")
    message: Optional[str] = Field(default=None, description="接口操作提示信息")


class StreamTaskListResponse(BaseModel):
    """推流任务列表"""

    total: int = Field(description="任务总数", ge=0)
    tasks: List[StreamTaskResponse] = Field(description="任务列表")


# ---------------------------------------------------------------------------
# 事件视频截取
# ---------------------------------------------------------------------------

class VideoClipRequest(BaseModel):
    """
    事件视频截取请求。

    仅需传入流地址与前后录制时长；服务端会从 `video_url` 解析出
    ZLMediaKit 所需的 `app` / `stream`，再调用 `startRecordTask`。
    """

    id: int = Field(
        ...,
        description="告警记录 ID，截取完成后回写至 coal/alarm/update",
        examples=[1],
    )
    video_url: str = Field(
        ...,
        description=(
            "目标视频流地址（RTMP / RTSP / HTTP-FLV / HLS）。"
            "路径需包含 app/stream，例如 "
            f"`{settings.build_push_url()}`"
        ),
        examples=[
            settings.build_push_url(),
            f"rtsp://{settings.ZLM_HOST}:{settings.ZLM_RTSP_PORT}/{settings.DEFAULT_SCENE_ID}/{settings.DEFAULT_SKILL_NAME}",
            f"http://{settings.ZLM_HOST}:{settings.ZLM_HTTP_PORT}/{settings.DEFAULT_SCENE_ID}/{settings.DEFAULT_SKILL_NAME}.live.flv",
        ],
    )
    back_ms: int = Field(
        default=10000,
        ge=0,
        description="回溯录制时长（毫秒），从 GOP 缓存中截取历史画面",
    )
    forward_ms: int = Field(
        default=10000,
        ge=0,
        description="后续录制时长（毫秒），从调用时刻起继续录制",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "id": 1,
                    "video_url": settings.build_push_url(),
                    "back_ms": 10000,
                    "forward_ms": 10000,
                }
            ]
        }
    )


class VideoClipResponse(BaseModel):
    """事件视频截取响应（含视频与中间帧截图 Base64）"""

    app: str = Field(description="应用名")
    stream: str = Field(description="流 ID")
    vhost: str = Field(description="虚拟主机")
    record_rel_path: str = Field(description="录像相对路径（相对 ZLM record 目录）")
    video_server_path: str = Field(description="录像在流媒体服务器上的绝对路径")
    video_url: str = Field(
        description="录像 MP4 HTTP 访问地址（事件回放）",
        examples=[f"http://{settings.ZLM_HOST}:{settings.ZLM_HTTP_PORT}/record/scene_001/person_count_detector/events/demo.mp4"],
    )
    video_base64: str = Field(
        description="录像 MP4 文件 Base64 编码",
        examples=["<base64-encoded-mp4>"],
    )
    image_base64: str = Field(
        description="视频中间时刻截图 Base64 编码（JPEG）",
        examples=["<base64-encoded-jpeg>"],
    )
    image_content_type: str = Field(
        default="image/jpeg",
        description="截图 MIME 类型",
    )
    video_minio_url: str = Field(
        description="录像在 MinIO 中的 HTTP 访问地址",
        examples=[
            f"http://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET_NAME}/"
            "events/scene_001/person_count_detector/demo.mp4"
        ],
    )
    image_minio_url: str = Field(
        description="中间帧截图在 MinIO 中的 HTTP 访问地址",
        examples=[
            f"http://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET_NAME}/"
            "events/scene_001/person_count_detector/demo.jpg"
        ],
    )
    back_ms: int = Field(description="回溯录制时长（毫秒）")
    forward_ms: int = Field(description="后续录制时长（毫秒）")
    duration_sec: Optional[float] = Field(
        default=None,
        description="实际录像时长（秒），由 ffprobe 探测",
    )
    message: Optional[str] = Field(default=None, description="操作提示信息")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "app": "scene_001",
                    "stream": "person_count_detector",
                    "vhost": "__defaultVhost__",
                    "record_rel_path": "events/20260713_120000_a1b2c3d4.mp4",
                    "video_server_path": "/www/record/scene_001/person_count_detector/events/20260713_120000_a1b2c3d4.mp4",
                    "video_url": f"http://{settings.ZLM_HOST}:{settings.ZLM_HTTP_PORT}/record/scene_001/person_count_detector/events/20260713_120000_a1b2c3d4.mp4",
                    "video_base64": "<base64-encoded-mp4>",
                    "image_base64": "<base64-encoded-jpeg>",
                    "image_content_type": "image/jpeg",
                    "video_minio_url": (
                        f"http://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET_NAME}/"
                        "events/scene_001/person_count_detector/20260713_120000_a1b2c3d4.mp4"
                    ),
                    "image_minio_url": (
                        f"http://{settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET_NAME}/"
                        "events/scene_001/person_count_detector/20260713_120000_a1b2c3d4.jpg"
                    ),
                    "back_ms": 10000,
                    "forward_ms": 10000,
                    "duration_sec": 20.5,
                    "message": "事件视频截取完成，并已上传 MinIO",
                }
            ]
        }
    )
