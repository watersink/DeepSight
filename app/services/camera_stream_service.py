"""摄像仪取流服务：按摄像仪编码解析「人员计数流」的播放地址。

对外接口（``GET /open/v1/camera/rtsp``）的业务实现，路由层只做鉴权与参数校验。
调用方是**前端**，因此主要返回浏览器可直接播放的 **HTTP-FLV / HLS** 地址
（RTSP 浏览器不支持，仅作为附加字段返回）。

设计要点：
- ZLM 上的流与摄像仪**不是一对一**：同一个 ``cameraCode`` 可能对应多条流
  （入井/过暗/挪移/遮挡等不同用途），因此本服务只挑**人员计数流**；
- 「人员计数流」判定：
  1. 该流被**关联到人员计数技能的任务**使用（``_PERSON_COUNT_SKILLS``）；
  2. 或流名命中 ``settings.PERSON_COUNT_STREAM_EXTRA`` 配置的前缀（兜底）；
- **没有人员计数流时返回空**（不猜测、不降级到其他用途的流）。

播放地址（均基于 ``ZLM_PUBLIC_HOST`` 与对外端口拼装）：
- FLV：``http://{host}:{httpPort}/{app}/{stream}.live.flv``
- HLS：``http://{host}:{httpPort}/{app}/{stream}/hls.m3u8``
- RTSP：``rtsp://{host}:{rtspPort}/{app}/{stream}``（前端一般用不到）
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.db.models import Camera, TaskConfig
from app.services.zlm_client import parse_stream_url

logger = logging.getLogger(__name__)


def _person_count_skills() -> frozenset:
    """人员计数类技能名（复用 stream_alert 的白名单，保持单一来源）。"""
    try:
        from app.services.stream_alert import _PERSON_COUNT_SKILLS

        return _PERSON_COUNT_SKILLS
    except Exception:
        logger.warning("取流服务：无法加载人员计数技能白名单，使用内置默认值")
        return frozenset(
            {
                "person_presence_detector26",
                "person_count_detector26",
                "person_count_detector",
                "boarding_detector",
                "non_fixed_parking_boarding_detector",
            }
        )


def _extra_prefixes() -> List[str]:
    """配置里额外视为人员计数流的前缀（去空、小写）。"""
    raw = str(getattr(settings, "PERSON_COUNT_STREAM_EXTRA", "") or "")
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _stream_name_matches_extra(stream: str) -> bool:
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
        return {"app": str(parsed.get("app") or ""), "stream": str(parsed.get("stream") or "")}
    return None


def _person_count_camera_ids(db: Session) -> set:
    """反查「使用人员计数技能的任务」关联的摄像仪 id 集合。"""
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


def _build_play_urls(app: str, stream: str) -> Dict[str, str]:
    """拼装前端可播放的地址（FLV / HLS）+ RTSP。"""
    app_s = str(app or "").strip().strip("/")
    stream_s = str(stream or "").strip().strip("/")
    http_base = settings.zlm_http_base_url
    return {
        "flvUrl": settings.build_flv_play_url(app_s, stream_s),
        "hlsUrl": f"{http_base}/{app_s}/{stream_s}/hls.m3u8",
        "rtspUrl": settings.build_zlm_pull_url(
            app_s, stream_s, output_format="rtsp"
        ),
    }


def list_person_count_streams(db: Session, camera_code: str) -> List[Dict[str, Any]]:
    """列出该摄像仪编码对应的「人员计数流」。

    返回 ``[{"app","stream","streamName","flvUrl","hlsUrl","rtspUrl",...}]``；
    **没有人员计数流时返回空列表**。
    """
    code = str(camera_code or "").strip()
    if not code:
        return []

    cameras = (
        db.scalars(
            select(Camera).where(Camera.camera_code == code).order_by(Camera.id)
        )
        .unique()
        .all()
    )
    if not cameras:
        logger.info("取流服务：cameraCode=%s 未找到摄像仪", code)
        return []

    person_count_ids = _person_count_camera_ids(db)
    require_task = bool(
        getattr(settings, "PERSON_COUNT_STREAM_REQUIRE_TASK", True)
    )

    out: List[Dict[str, Any]] = []
    seen: set = set()
    for cam in cameras:
        resolved = _resolve_app_stream(cam)
        if not resolved or not resolved.get("app") or not resolved.get("stream"):
            continue
        key = (resolved["app"], resolved["stream"])
        if key in seen:
            continue

        by_task = int(getattr(cam, "id", 0) or 0) in person_count_ids
        by_extra = _stream_name_matches_extra(resolved["stream"])
        if require_task:
            # 必须关联人员计数任务（严格口径，避免把过暗/挪移等流当人员计数）
            is_person_count = by_task
        else:
            # 放宽：任务关联 或 流名命中配置前缀
            is_person_count = by_task or by_extra
        if not is_person_count:
            continue

        seen.add(key)
        urls = _build_play_urls(resolved["app"], resolved["stream"])
        out.append(
            {
                "app": resolved["app"],
                "stream": resolved["stream"],
                "streamName": f"{resolved['app']}/{resolved['stream']}",
                # 前端播放地址（浏览器不支持 RTSP，优先用 flvUrl）
                "flvUrl": urls["flvUrl"],
                "hlsUrl": urls["hlsUrl"],
                "rtspUrl": urls["rtspUrl"],
                "positionDesc": str(getattr(cam, "position_desc", None) or "").strip(),
                "cameraId": getattr(cam, "id", None),
            }
        )

    if not out:
        logger.info(
            "取流服务：cameraCode=%s 没有人员计数流（关联摄像仪 %s 条）",
            code,
            len(cameras),
        )
    return out


def get_person_count_rtsp(db: Session, camera_code: str) -> Optional[Dict[str, Any]]:
    """返回单条人员计数流（取第一条）；没有则返回 None。"""
    streams = list_person_count_streams(db, camera_code)
    return streams[0] if streams else None
