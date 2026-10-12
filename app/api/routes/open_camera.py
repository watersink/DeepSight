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


def _person_count_task_streams(
    db: Session, camera_ids: set
) -> Dict[int, List[Dict[str, Any]]]:
    """反查这些摄像仪上「人员计数任务」的 **AI 画框流**坐标，按摄像仪 id 归组。

    AI 画框流 = ``{task.scene_id}/{algo.skill_name}``：任务把带检测框的画面推到
    这个 app/stream（见 ``settings.build_push_url``），与前端任务面板的
    ``flv_url`` 完全同源。**这才是平台要看的"有框"视频**；
    摄像仪自身的 ``in_url``（如 ``live/guanlong``）只是原始源流，没有框。
    """
    skills = _person_count_skills()
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for task in db.scalars(
        select(TaskConfig)
        .options(joinedload(TaskConfig.algorithm_config))
        .order_by(TaskConfig.id)
    ).all():
        algo = getattr(task, "algorithm_config", None)
        skill = str(getattr(algo, "skill_name", None) or "").strip()
        if skill not in skills:
            continue
        camera_id = int(getattr(task, "camera_id", 0) or 0)
        if not camera_id or camera_id not in camera_ids:
            continue
        scene_id = str(getattr(task, "scene_id", None) or "").strip()
        if not scene_id:
            continue
        grouped.setdefault(camera_id, []).append(
            {
                "app": scene_id,
                "stream": skill,
                "sceneId": scene_id,
                "skillName": skill,
                "taskId": getattr(task, "id", None),
                "taskName": str(getattr(task, "name", None) or ""),
                # 没开「AI 识别结果推流」的任务不会产出画框流
                "pushAnnotatedStream": bool(
                    getattr(task, "push_annotated_stream", False)
                ),
            }
        )
    return grouped


def _build_urls(app: str, stream: str) -> Dict[str, str]:
    """拼装播放地址：FLV / HLS / RTSP。"""
    app_s = str(app or "").strip().strip("/")
    stream_s = str(stream or "").strip().strip("/")
    return {
        "flvUrl": settings.build_flv_play_url(app_s, stream_s),
        "hlsUrl": f"{settings.zlm_http_base_url}/{app_s}/{stream_s}/hls.m3u8",
        "rtspUrl": settings.build_zlm_pull_url(app_s, stream_s, output_format="rtsp"),
    }


def _zlm_online_streams() -> Optional[set]:
    """查询 ZLM 当前在线的 (app, stream) 集合；查询失败返回 None（表示不确定）。"""
    try:
        from app.services.zlm_client import zlm_client

        keys = zlm_client.list_online_media_keys()
    except Exception:
        logger.warning("取流接口：查询 ZLM 在线流失败，跳过在线判断", exc_info=True)
        return None

    # list_online_media_keys 返回 "vhost|app|stream"，转成 (app, stream)
    online: set = set()
    for key in keys or set():
        parts = str(key).split("|")
        if len(parts) >= 3:
            online.add((parts[-2], parts[-1]))
    return online


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
    - ``streams``：该摄像仪全部人员计数流；每项含 ``source``
      （``ai``=任务画框流，带检测框 / ``source``=摄像仪原始源流，无框）与 ``online``

    优先返回**任务的 AI 画框流** ``{scene_id}/{skill_name}``（与前端任务面板
    ``flv_url`` 同源，**带检测框**）；任务没跑或没开「AI 识别结果推流」时，
    回落到摄像仪原始源流兜底。只返回**人员计数**用途的流，过暗/过曝/模糊/
    挪移/遮挡等用途不返回。
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
        return {
            "cameraCode": code,
            "flvUrl": None,
            "hlsUrl": None,
            "rtspUrl": None,
            "online": None,
            "hint": f"未找到摄像仪编码 {code}：平台侧编码与本系统摄像仪不一致",
            "streams": [],
        }

    person_count_ids = _person_count_camera_ids(db)
    require_task = bool(getattr(settings, "PERSON_COUNT_STREAM_REQUIRE_TASK", True))
    online_keys = _zlm_online_streams()
    task_streams = _person_count_task_streams(db, {int(c.id) for c in cameras})

    streams: List[Dict[str, Any]] = []
    seen: set = set()

    def _append(
        cam: Camera, app: str, stream: str, source: str, extra: Dict[str, Any]
    ) -> None:
        key = (app, stream)
        if key in seen or not app or not stream:
            return
        seen.add(key)
        urls = _build_urls(app, stream)
        item: Dict[str, Any] = {
            "app": app,
            "stream": stream,
            "streamName": f"{app}/{stream}",
            "flvUrl": urls["flvUrl"],
            "hlsUrl": urls["hlsUrl"],
            "rtspUrl": urls["rtspUrl"],
            # ai = 任务画框流（有检测框）；source = 摄像仪原始源流（无框）
            "source": source,
            # 流当前是否在 ZLM 在线（None=查询失败，无法判断）
            "online": (None if online_keys is None else (key in online_keys)),
            "positionDesc": str(getattr(cam, "position_desc", None) or "").strip(),
            "cameraId": getattr(cam, "id", None),
        }
        item.update(extra)
        streams.append(item)

    for cam in cameras:
        # ① 优先：任务的 AI 画框流 {scene_id}/{skill_name}（有检测框，平台要的就是它）
        for ts in task_streams.get(int(cam.id), []):
            if not ts.get("pushAnnotatedStream"):
                continue
            _append(
                cam,
                str(ts["app"]),
                str(ts["stream"]),
                "ai",
                {
                    "sceneId": ts.get("sceneId"),
                    "skillName": ts.get("skillName"),
                    "taskId": ts.get("taskId"),
                    "taskName": ts.get("taskName"),
                },
            )

        # ② 兜底：摄像仪原始源流（任务没跑 / 没开画框推流时至少还能看到画面）
        resolved = _resolve_app_stream(cam)
        if not resolved or not resolved.get("app") or not resolved.get("stream"):
            continue
        by_task = int(getattr(cam, "id", 0) or 0) in person_count_ids
        by_extra = _matches_extra_prefix(resolved["stream"])
        matched = by_task if require_task else (by_task or by_extra)
        if not matched:
            continue
        _append(
            cam,
            resolved["app"],
            resolved["stream"],
            "source",
            {"sceneId": None, "skillName": None},
        )

    if not streams:
        logger.info(
            "取流接口：cameraCode=%s 没有人员计数流（关联摄像仪 %s 条）",
            code,
            len(cameras),
        )

    # 平台播放器取用的是顶层 rtspUrl / flvUrl / hlsUrl，排序规则（越靠前越优先）：
    #   1) 在线 > 离线（给离线流平台会报「视频流连接超时」）
    #   2) 同为在线/离线时，AI 画框流 > 摄像仪原始源流（平台要的是有检测框的视频）
    streams.sort(
        key=lambda s: (
            0 if s.get("online") else 1,
            0 if s.get("source") == "ai" else 1,
        )
    )
    first = streams[0] if streams else {}
    online = first.get("online")
    hint = None
    if streams and online is False:
        if first.get("source") == "ai":
            hint = (
                "AI 画框流已配置但当前未推流（对应检测任务未在运行），"
                "播放地址暂时不可用"
            )
        else:
            hint = (
                "人员计数流已配置但当前未推流（ZLM 上不存在该流），"
                "播放地址暂时不可用；请确认对应检测任务已在运行"
            )
    elif streams and first.get("source") == "source" and any(
        s.get("source") == "ai" for s in streams
    ):
        hint = "AI 画框流当前不在线，暂用摄像仪原始源流（无检测框）"
    return {
        "cameraCode": code,
        "flvUrl": first.get("flvUrl"),
        "hlsUrl": first.get("hlsUrl"),
        "rtspUrl": first.get("rtspUrl"),
        "online": online,
        "hint": hint,
        "streams": streams,
    }
