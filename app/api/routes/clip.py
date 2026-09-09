"""事件视频截取 API（基于 ZLMediaKit startRecordTask）"""
import base64
import json
import logging
import time
import uuid
import urllib.error
import urllib.request
from pathlib import PurePosixPath

from fastapi import APIRouter, HTTPException, status

from app.api.schemas import ErrorResponse, VideoClipRequest, VideoClipResponse
from app.core.config import settings
from app.services.minio_client import MinioClientError, minio_storage
from app.services.video_clip_service import video_clip_service
from app.services.zlm_client import ZLMClientError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clip", tags=["事件视频截取"])


class AlarmUpdateError(Exception):
    """告警回写接口调用失败"""


def _update_coal_alarm(*, alarm_id: int, violation_image: str, violation_video: str) -> None:
    """将截取结果回写至煤安告警更新接口。"""
    payload = {
        "id": alarm_id,
        "violationImage": violation_image,
        "violationVideo": violation_video,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        settings.COAL_ALARM_UPDATE_URL,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=settings.COAL_ALARM_UPDATE_TIMEOUT) as resp:
            resp_body = resp.read().decode("utf-8", errors="replace")
            logger.info(
                "告警回写成功 id=%s status=%s body=%s",
                alarm_id,
                getattr(resp, "status", None),
                resp_body[:200],
            )
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise AlarmUpdateError(f"告警回写 HTTP {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise AlarmUpdateError(
            f"无法连接告警回写接口 ({settings.COAL_ALARM_UPDATE_URL}): {e.reason}"
        ) from e


def _upload_clip_assets_to_minio(
    *,
    app: str,
    stream: str,
    record_rel_path: str,
    video_base64: str,
    image_base64: str,
    image_content_type: str,
) -> tuple[str, str]:
    """将截取视频与中间帧上传至 MinIO，返回 (video_minio_url, image_minio_url)。"""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    uniq = uuid.uuid4().hex[:8]
    stem = PurePosixPath(record_rel_path).stem or f"{stamp}_{uniq}"
    base_prefix = f"events/{app}/{stream}/{stem}"

    video_bytes = base64.b64decode(video_base64)
    image_bytes = base64.b64decode(image_base64)

    video_minio_url = minio_storage.upload_bytes(
        video_bytes,
        object_name=f"{base_prefix}.mp4",
        content_type="video/mp4",
    )
    image_minio_url = minio_storage.upload_bytes(
        image_bytes,
        object_name=f"{base_prefix}.jpg",
        content_type=image_content_type or "image/jpeg",
    )
    return video_minio_url, image_minio_url


@router.post(
    "/capture",
    response_model=VideoClipResponse,
    summary="截取事件视频并生成中间帧截图",
    description=(
        "封装 ZLMediaKit 事件录像接口，同步返回视频与中间帧截图。\n\n"
        "**请求参数**：`id`、`video_url`、`back_ms`、`forward_ms`。\n\n"
        "**底层 API**：`GET /index/api/startRecordTask`\n\n"
        "**流程**：\n"
        "1. 从 `video_url` 解析 `app` / `stream`，并检查流是否在线\n"
        "2. 调用 ZLMediaKit 进行回溯 + 前向录制\n"
        f"3. 等待 `forward_ms + {settings.ZLM_RECORD_WAIT_BUFFER_MS}ms` 并确认文件写入完成\n"
        "4. 从流媒体服务器获取 MP4，用 ffmpeg 截取中间时刻帧\n"
        "5. 将视频与截图上传至 MinIO\n"
        f"6. 调用 `{settings.COAL_ALARM_UPDATE_URL}` 回写 `violationImage` / `violationVideo`\n"
        "7. 返回 Base64 与 MinIO 访问地址\n\n"
        "**注意**：本接口为同步阻塞，耗时约 `forward_ms + 2s`；"
        f"需配置 `ZLM_SECRET` 及 ZLMediaKit 地址（当前 `{settings.ZLM_HOST}:{settings.ZLM_HTTP_PORT}`）；"
        f"MinIO 默认桶 `{settings.MINIO_BUCKET_NAME}` @ `{settings.MINIO_ENDPOINT}`。"
    ),
    responses={
        200: {"description": "截取成功，返回视频与截图 Base64 及 MinIO 地址"},
        400: {"model": ErrorResponse, "description": "流地址无效、流不在线、ZLM API 失败或参数错误"},
        500: {"model": ErrorResponse, "description": "文件读取、截图、MinIO 上传、告警回写或服务内部错误"},
    },
)
def capture_event_clip(request: VideoClipRequest):
    try:
        result = video_clip_service.capture_event_clip(
            video_url=request.video_url,
            back_ms=request.back_ms,
            forward_ms=request.forward_ms,
        )
        video_minio_url, image_minio_url = _upload_clip_assets_to_minio(
            app=result.app,
            stream=result.stream,
            record_rel_path=result.record_rel_path,
            video_base64=result.video_base64,
            image_base64=result.image_base64,
            image_content_type=result.image_content_type,
        )
        _update_coal_alarm(
            alarm_id=request.id,
            violation_image=image_minio_url,
            violation_video=video_minio_url,
        )
    except ZLMClientError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except MinioClientError as e:
        logger.exception("事件素材上传 MinIO 失败 video_url=%s", request.video_url)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except AlarmUpdateError as e:
        logger.exception("告警回写失败 id=%s video_url=%s", request.id, request.video_url)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.exception("事件视频截取失败 video_url=%s", request.video_url)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    return VideoClipResponse(
        app=result.app,
        stream=result.stream,
        vhost=result.vhost,
        record_rel_path=result.record_rel_path,
        video_server_path=result.video_server_path,
        video_url=result.video_url,
        video_base64=result.video_base64,
        image_base64=result.image_base64,
        image_content_type=result.image_content_type,
        video_minio_url=video_minio_url,
        image_minio_url=image_minio_url,
        back_ms=result.back_ms,
        forward_ms=result.forward_ms,
        duration_sec=result.duration_sec,
        message="事件视频截取完成，并已上传 MinIO",
    )
