"""摄像仪取流开放接口（独立文件，供平台直接调用）。

用途：平台按**摄像仪编码**查询「人员计数流」的播放地址（HTTP-FLV / HLS），
前端拿到地址后可直接播放（浏览器不支持 RTSP，故 RTSP 仅作附加字段）。

- 路径：``GET /api/v1/camera/play-url``（独立文件，与其它路由互不影响）
- 查不到人员计数流时返回空（``flvUrl`` / ``hlsUrl`` 为 null，``streams`` 为 ``[]``）
"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.db import get_db
from app.db.models import Camera, TaskConfig
from app.services.zlm_client import parse_stream_url

logger = logging.getLogger(__name__)

router = APIRouter(tags=["摄像仪取流（免鉴权）"])

# ---------------------------------------------------------------------------
# 人员计数技能白名单（与 stream_alert 保持同一来源，避免漏改）
# ---------------------------------------------------------------------------

_DEFAULT_PERSON_COUNT_SKILLS = frozenset(
    {
        "person_presence_detector26",
        "person_count_detector26",
        "person_count_detector",
        "boarding_detector",
        "non_fixed_parking_boarding_detector",
    }
)


def _person_count_skills() -> frozenset:
    """人员计数类技能名（优先复用 stream_alert 的白名单）。"""
    try:
        from app.services.stream_alert import _PERSON_COUNT_SKILLS

        return _PERSON_COUNT_SKILLS
    except Exception:
        return _DEFAULT_PERSON_COUNT_SKILLS


def _extra_prefixes() -> List[str]:
    raw = str(getattr(settings, "PERSON_COUNT_STREAM_EXTRA", "") or "")
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _matches_extra_prefix(stream: str) -> bool:
    name = str(stream or "").strip().lower()
    if not name:
        return False
    return any(name.startswith(p) or p in name for p in _extra_prefixes())


def _resolve_app_stream(camera: Camera) -> Optional[Dict[str, str]]:
    """取摄像仪的 ZLM app/stream：优先 zlm_app/zlm_stream，其次从 in_url 解析。"""
    app = str(getattr(camera, "zlm_app", None) or "").strip()
    stream = str(getattr(camera, "zlm_stream", None) or "").strip()
    if app and stream:
        return {"app": app, "stream": stream}
    parsed = parse_stream_url(str(getattr(camera, "in_url", None) or ""))
    if parsed:
        return {
            "app": str(parsed.get("app") or ""),
            "stream": str(parsed.get("stream") or ""),
        }
    return None


def _person_count_camera_ids(db: Session) -> set:
    """使用人员计数技能的任务所关联的摄像仪 id 集合。"""
    skills = _person_count_skills()
    ids: set = set()
    for task in db.scalars(
        select(TaskConfig).options(joinedload(TaskConfig.algorithm_config))
    ).all():
        algo = getattr(task, "algorithm_config", None)
        skill = str(getattr(algo, "skill_name", None) or "").strip()
        if skill in skills and task.camera_id:
            ids.add(int(task.camera_id))
    return ids


def _build_urls(app: str, stream: str) -> Dict[str, str]:
    """拼装播放地址：FLV / HLS / RTSP。"""
    app_s = str(app or "").strip().strip("/")
    stream_s = str(stream or "").strip().strip("/")
    return {
        "flvUrl": settings.build_flv_play_url(app_s, stream_s),
        "hlsUrl": f"{settings.zlm_http_base_url}/{app_s}/{stream_s}/hls.m3u8",
        "rtspUrl": settings.build_zlm_pull_url(app_s, stream_s, output_format="rtsp"),
    }


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------


@router.get(
    "/camera/play-url",
    summary="按摄像仪编码获取人员计数流播放地址（FLV / HLS）",
)
def get_camera_play_url(
    db: Annotated[Session, Depends(get_db)],
    camera_code: str = Query(
        ..., alias="cameraCode", description="摄像仪编码（MT/T 1201.6-2023）"
    ),
) -> Dict[str, Any]:
    """平台直接调用，无需 API Key。

    返回该摄像仪「人员计数流」的播放地址：

    - ``flvUrl``：HTTP-FLV，前端用 flv.js 播放（推荐）
    - ``hlsUrl``：HLS（m3u8），前端用 hls.js / 原生播放
    - ``rtspUrl``：RTSP 拉流地址（浏览器不支持，备用）
    - ``streams``：该摄像仪全部人员计数流（入井/出井等多条时按数组返回）

    只返回**人员计数**用途的流；过暗/过曝/模糊/挪移/遮挡等用途不返回。
    """
    code = str(camera_code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="cameraCode 不能为空")

    cameras = (
        db.scalars(select(Camera).where(Camera.camera_code == code).order_by(Camera.id))
        .unique()
        .all()
    )
    if not cameras:
        logger.info("取流接口：cameraCode=%s 未找到摄像仪", code)
        return {"cameraCode": code, "flvUrl": None, "hlsUrl": None, "streams": []}

    person_count_ids = _person_count_camera_ids(db)
    require_task = bool(getattr(settings, "PERSON_COUNT_STREAM_REQUIRE_TASK", True))

    streams: List[Dict[str, Any]] = []
    seen: set = set()
    for cam in cameras:
        resolved = _resolve_app_stream(cam)
        if not resolved or not resolved.get("app") or not resolved.get("stream"):
            continue
        key = (resolved["app"], resolved["stream"])
        if key in seen:
            continue

        by_task = int(getattr(cam, "id", 0) or 0) in person_count_ids
        by_extra = _matches_extra_prefix(resolved["stream"])
        matched = by_task if require_task else (by_task or by_extra)
        if not matched:
            continue

        seen.add(key)
        urls = _build_urls(resolved["app"], resolved["stream"])
        streams.append(
            {
                "app": resolved["app"],
                "stream": resolved["stream"],
                "streamName": f"{resolved['app']}/{resolved['stream']}",
                "flvUrl": urls["flvUrl"],
                "hlsUrl": urls["hlsUrl"],
                "rtspUrl": urls["rtspUrl"],
                "positionDesc": str(getattr(cam, "position_desc", None) or "").strip(),
                "cameraId": getattr(cam, "id", None),
            }
        )

    if not streams:
        logger.info(
            "取流接口：cameraCode=%s 没有人员计数流（关联摄像仪 %s 条）",
            code,
            len(cameras),
        )

    first = streams[0] if streams else {}
    return {
        "cameraCode": code,
        "flvUrl": first.get("flvUrl"),
        "hlsUrl": first.get("hlsUrl"),
        "rtspUrl": first.get("rtspUrl"),
        "streams": streams,
    }
