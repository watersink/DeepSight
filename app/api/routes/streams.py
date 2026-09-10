"""视频流推流 API 路由"""
import asyncio
import json
import logging
from typing import Annotated, Optional

from fastapi import APIRouter, Body, HTTPException, Path, Query, Request, status
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ErrorResponse,
    StreamStartRequest,
    StreamTaskListResponse,
    StreamTaskResponse,
)
from app.core.config import settings
from app.plugins.skill_registry import SkillNotFoundError, resolve_skill_class
from app.services.alert_hub import alert_hub
from app.services.runtime_gateway import WorkerApiError, runtime_gateway

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/streams", tags=["AI视频识别任务"])

_COMMON_ERRORS = {
    400: {"model": ErrorResponse, "description": "请求参数无效或技能不存在"},
    404: {"model": ErrorResponse, "description": "任务不存在"},
    409: {"model": ErrorResponse, "description": "任务冲突（如相同配置任务已在运行）"},
    500: {"model": ErrorResponse, "description": "服务内部错误"},
}


def _to_response(data: dict, message: str = None) -> StreamTaskResponse:
    flv_url = settings.build_flv_play_url_from_out_url(data.get("out_url"))
    if not flv_url and data.get("scene_id") and data.get("skill_name"):
        flv_url = settings.build_flv_play_url(data["scene_id"], data["skill_name"])
    payload = {
        "task_id": data.get("task_id"),
        "status": data.get("status"),
        "scene_id": data.get("scene_id"),
        "skill_name": data.get("skill_name"),
        "in_url": data.get("in_url"),
        "out_url": data.get("out_url"),
        "pid": data.get("pid"),
        "worker_id": data.get("worker_id"),
        "worker_name": data.get("worker_name"),
        "created_at": data.get("created_at"),
        "started_at": data.get("started_at"),
        "stopped_at": data.get("stopped_at"),
        "error_message": data.get("error_message"),
    }
    return StreamTaskResponse(**payload, flv_url=flv_url, message=message)


@router.post(
    "/start",
    response_model=StreamTaskResponse,
    summary="启动AI视频识别任务",
    description=(
        "在独立子进程中启动视频流 AI 检测推流任务。\n\n"
        "- 通过 `skill_name` 指定检测技能\n"
        "- 省略 `out_url` 时自动拼装推流地址："
        f"`{settings.build_push_url()}`\n"
        "- 响应含 `flv_url`（由 `out_url` 推导的 HTTP-FLV 播放地址）\n"
        "- 推流目标为 ZLMediaKit（RTMP / RTSP）\n\n"
        "**可选技能参数（按技能使用，不需要可省略）：**\n"
        "- `enter_count`：过线进入人数初始值（如 `0`）\n"
        "- `gate_direction`：闸机方向 `IN`（入闸）/ `OUT`（出闸），影响画面标注\n"
        "- `mine_code` / `camera_code`：煤矿编码（12位）与摄像仪编码（20位），用于识别结果上传\n"
        "- `count_line`：过线计数线像素坐标（person_count_detector26 必填；空则启动失败）\n"
        "- `bypass_line`：非计数绕行线像素坐标（可选；未传或为空则不做绕行检测）"
    ),
    responses={
        200: {"description": "视频AI检测任务已启动"},
        400: _COMMON_ERRORS[400],
        409: _COMMON_ERRORS[409],
        500: _COMMON_ERRORS[500],
    },
)
def start_stream(
    request: Annotated[
        StreamStartRequest,
        Body(
            openapi_examples={
                "basic": {
                    "summary": "基础启动",
                    "description": "使用默认技能与测试视频",
                    "value": {
                        "in_url": "test_images_videos/person_gouzi.mp4",
                        "scene_id": settings.DEFAULT_SCENE_ID,
                        "skill_name": settings.DEFAULT_SKILL_NAME,
                        "output_format": settings.DEFAULT_OUTPUT_FORMAT,
                        "out_fps": settings.DEFAULT_OUT_FPS,
                    },
                },
                "yolo26_count_line": {
                    "summary": "YOLO26 过线计数",
                    "description": (
                        "person_count_detector26：指定 enter_count 初始值、"
                        "gate_direction（IN/OUT）、count_line 与 bypass_line（绕行线）"
                    ),
                    "value": {
                        "in_url": "rtmp://10.1.3.21:1935/live/stream01",
                        "scene_id": "scene_001",
                        "skill_name": "person_count_detector26",
                        "output_format": "rtmp",
                        "out_fps": settings.DEFAULT_OUT_FPS,
                        "enter_count": 0,
                        "gate_direction": "IN",
                        "count_line": {
                            "lines": [
                                {
                                    "line": [
                                        {"x": 432, "y": 328},
                                        {"x": 723, "y": 229},
                                    ],
                                    "enter_point": {"x": 513, "y": 250},
                                },
                            ],
                        },
                        "bypass_line": {
                            "lines": [
                                {
                                    "line": [
                                        {"x": 154, "y": 421},
                                        {"x": 434, "y": 337},
                                    ],
                                    "bypass_point": {"x": 280, "y": 348},
                                },
                            ],
                        },
                    },
                },
            }
        ),
    ],
):
    try:
        resolve_skill_class(request.skill_name)
    except SkillNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    try:
        dumped = request.model_dump()
        worker_id = dumped.pop("worker_id", None)
        payload = settings.resolve_stream_start(dumped)
        task = runtime_gateway.start_task(payload, worker_id=worker_id)
    except WorkerApiError as e:
        code = status.HTTP_409_CONFLICT if e.status_code == 409 else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except Exception as e:
        logger.exception("启动视频AI检测任务失败")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return _to_response(task, message="视频AI检测任务已启动")


@router.post(
    "/stop/{task_id}",
    response_model=StreamTaskResponse,
    summary="停止AI视频识别任务",
    description="停止指定 `task_id` 的推流子进程并更新任务状态。",
    responses={
        200: {"description": "推流任务已停止"},
        404: _COMMON_ERRORS[404],
        500: _COMMON_ERRORS[500],
    },
)
def stop_stream(
    task_id: Annotated[str, Path(description="推流任务 ID，由启动接口返回")],
):
    try:
        task = runtime_gateway.stop_task(task_id)
    except WorkerApiError as e:
        code = status.HTTP_404_NOT_FOUND if e.status_code == 404 else status.HTTP_502_BAD_GATEWAY
        raise HTTPException(status_code=code, detail=str(e))
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"任务不存在: {task_id}")
    except Exception as e:
        logger.exception("停止推流任务失败 task_id=%s", task_id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return _to_response(task, message="推流任务已停止")


@router.get(
    "",
    response_model=StreamTaskListResponse,
    summary="查询所有AI视频识别任务",
    description="返回当前服务管理的全部推流任务及其状态。",
    responses={200: {"description": "任务列表"}},
)
def list_streams():
    tasks = runtime_gateway.list_tasks()
    items = [_to_response(t) for t in tasks]
    return StreamTaskListResponse(total=len(items), tasks=items)


@router.get(
    "/alerts/sse",
    summary="SSE 订阅人数告警",
    description=(
        "以 Server-Sent Events 实时推送人数告警 INFO。\n\n"
        "- 事件名：`person_count_alert`\n"
        "- 可选查询参数 `scene_id` 仅订阅指定场景\n"
        "- 连接后会先收到 `connected`，之后约每 15s 发送注释心跳\n"
        "- 前端可用 `EventSource` 连接本接口"
    ),
    responses={
        200: {
            "description": "SSE 事件流（text/event-stream）",
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                    "example": (
                        "event: connected\n"
                        'data: {"ok":true}\n\n'
                        "event: person_count_alert\n"
                        'data: {"type":"person_count_alert","level":"INFO","scene_id":"scene_001",'
                        '"count":3,"enter_count":0,"pic":"person_count/scene_001_....jpg",'
                        '"message":"触发人数告警 scene=scene_001 count=3 enter_count=0 ..."}\n\n'
                    ),
                }
            },
        }
    },
)
async def stream_alerts_sse(
    request: Request,
    scene_id: Optional[str] = Query(
        default=None,
        description="仅推送该场景的告警；省略则推送全部场景",
        examples=["scene_001"],
    ),
):
    async def event_generator_with_heartbeat():
        yield _sse_event("connected", {"ok": True, "scene_id": scene_id})
        agen = alert_hub.subscribe(scene_id=scene_id)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(agen.__anext__(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                except StopAsyncIteration:
                    break
                yield _sse_event("person_count_alert", event)
        except asyncio.CancelledError:
            raise
        finally:
            await agen.aclose()
            logger.info("SSE 告警订阅断开 scene_id=%s", scene_id)

    return StreamingResponse(
        event_generator_with_heartbeat(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


@router.get(
    "/{task_id}",
    response_model=StreamTaskResponse,
    summary="查询单个AI视频识别任务状态",
    description="根据 `task_id` 查询推流任务的详细状态（含 PID、起止时间等）。",
    responses={
        200: {"description": "任务详情"},
        404: _COMMON_ERRORS[404],
    },
)
def get_stream(
    task_id: Annotated[str, Path(description="推流任务 ID")],
):
    task = runtime_gateway.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"任务不存在: {task_id}")
    return _to_response(task)
