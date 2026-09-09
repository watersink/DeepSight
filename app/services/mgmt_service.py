"""管理台业务服务"""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import desc, select
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.db.models import AlertRecord, AlgorithmConfig, Camera, Mine, Site, TaskConfig
from app.plugins.skill_registry import resolve_skill_class
from app.services.redis_cache import cache_delete
from app.services.stream_task_manager import stream_task_manager
from app.services.zlm_client import ZLMClientError, parse_stream_url, zlm_client

logger = logging.getLogger(__name__)

CACHE_CAMERAS = "mgmt:cameras:all"
CACHE_ALGOS = "mgmt:algos:all"

# 识别类型归类：04-07 报警；01/02 与画面人数等为事件
ALERT_RECOGNITION_TYPES = frozenset({"04", "05", "06", "07"})
EVENT_RECOGNITION_TYPES = frozenset({"01", "02"})
PRESENCE_SKILL_NAMES = frozenset({"person_presence_detector26"})


def classify_record_category(
    recognition_types: Optional[List[Any]] = None,
    *,
    skill_name: str = "",
    has_person_count_change: bool = False,
) -> str:
    """返回 alert | event。"""
    types = {
        str(t).strip()
        for t in (recognition_types or [])
        if str(t).strip()
    }
    if types & ALERT_RECOGNITION_TYPES:
        return "alert"
    if types & EVENT_RECOGNITION_TYPES:
        return "event"
    if skill_name in PRESENCE_SKILL_NAMES or has_person_count_change:
        return "event"
    return "event" if types else "alert"


def _normalize_type_codes(recognition_types: Optional[List[Any]]) -> List[str]:
    return sorted(
        {
            str(t).strip()
            for t in (recognition_types or [])
            if str(t).strip()
        }
    )


def _page(items: list, total: int, page: int, page_size: int) -> Tuple[list, dict]:
    return items, {"total": total, "page": page, "page_size": page_size}


def _camera_to_dict(row: Camera, *, online_keys=None, zlm_error: Optional[str] = None) -> dict:
    site = getattr(row, "site", None)
    mine = getattr(site, "mine", None) if site is not None else None
    data = {
        "id": row.id,
        "name": row.name,
        "in_url": row.in_url,
        "ingest_mode": getattr(row, "ingest_mode", None) or "push",
        "source_url": getattr(row, "source_url", None),
        "zlm_app": getattr(row, "zlm_app", None) or "",
        "zlm_stream": getattr(row, "zlm_stream", None) or "",
        "proxy_key": getattr(row, "proxy_key", None),
        "camera_code": row.camera_code,
        "mine_code": row.mine_code,
        "site_id": row.site_id,
        "site_name": site.name if site is not None else None,
        "mine_id": mine.id if mine is not None else None,
        "mine_name": mine.name if mine is not None else None,
        "enabled": row.enabled,
        "remark": row.remark,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "online": None,
        "stream_app": None,
        "stream_id": None,
        "online_check_error": None,
        "flv_url": None,
    }
    app = (data["zlm_app"] or "").strip()
    stream = (data["zlm_stream"] or "").strip()
    parsed = parse_stream_url(row.in_url)
    if app and stream:
        data["stream_app"] = app
        data["stream_id"] = stream
        data["flv_url"] = settings.build_flv_play_url(app, stream)
    elif parsed:
        data["stream_app"] = parsed["app"]
        data["stream_id"] = parsed["stream"]
        data["flv_url"] = settings.build_flv_play_url(parsed["app"], parsed["stream"])
        app, stream = parsed["app"], parsed["stream"]
    else:
        data["online_check_error"] = "非 RTMP/RTSP 流地址，跳过在线检测"
        return data
    if zlm_error:
        data["online_check_error"] = zlm_error
        return data
    if online_keys is None:
        return data
    key = zlm_client.media_key(app, stream)
    data["online"] = key in online_keys
    return data


def _sync_camera_mine_code(db: Session, data: dict) -> dict:
    """若指定 site_id，用所属煤矿 code 回填 mine_code（未显式传入时）。"""
    site_id = data.get("site_id")
    if not site_id:
        return data
    site = db.get(Site, int(site_id))
    if site is None:
        raise ValueError(f"地点不存在: {site_id}")
    mine = db.get(Mine, site.mine_id)
    if mine and not str(data.get("mine_code") or "").strip():
        data = {**data, "mine_code": mine.code or ""}
    return data


# ---------- mines / sites ----------

def list_mines(db: Session) -> List[dict]:
    from sqlalchemy import func

    rows = db.scalars(select(Mine).order_by(Mine.id)).all()
    site_counts = dict(
        db.execute(
            select(Site.mine_id, func.count())
            .group_by(Site.mine_id)
        ).all()
    )
    cam_counts = dict(
        db.execute(
            select(Site.mine_id, func.count(Camera.id))
            .select_from(Site)
            .outerjoin(Camera, Camera.site_id == Site.id)
            .group_by(Site.mine_id)
        ).all()
    )
    return [
        {
            "id": m.id,
            "name": m.name,
            "code": m.code,
            "enabled": m.enabled,
            "remark": m.remark,
            "created_at": m.created_at,
            "updated_at": m.updated_at,
            "site_count": int(site_counts.get(m.id) or 0),
            "camera_count": int(cam_counts.get(m.id) or 0),
        }
        for m in rows
    ]


def create_mine(db: Session, data: dict) -> Mine:
    row = Mine(**data)
    db.add(row)
    db.commit()
    db.refresh(row)
    cache_delete(CACHE_CAMERAS)
    return row


def get_mine(db: Session, mine_id: int) -> Optional[Mine]:
    return db.get(Mine, mine_id)


def update_mine(db: Session, row: Mine, data: dict) -> Mine:
    for k, v in data.items():
        if v is not None:
            setattr(row, k, v)
    db.commit()
    db.refresh(row)
    cache_delete(CACHE_CAMERAS)
    return row


def delete_mine(db: Session, row: Mine) -> None:
    db.delete(row)
    db.commit()
    cache_delete(CACHE_CAMERAS)


def list_sites(db: Session, mine_id: Optional[int] = None) -> List[dict]:
    from sqlalchemy import func

    stmt = select(Site).options(joinedload(Site.mine)).order_by(Site.id)
    if mine_id is not None:
        stmt = stmt.where(Site.mine_id == mine_id)
    rows = db.scalars(stmt).unique().all()
    cam_counts = dict(
        db.execute(
            select(Camera.site_id, func.count()).group_by(Camera.site_id)
        ).all()
    )
    return [
        {
            "id": s.id,
            "mine_id": s.mine_id,
            "name": s.name,
            "code": s.code,
            "enabled": s.enabled,
            "remark": s.remark,
            "created_at": s.created_at,
            "updated_at": s.updated_at,
            "camera_count": int(cam_counts.get(s.id) or 0),
            "mine_name": s.mine.name if s.mine else None,
            "mine_code": s.mine.code if s.mine else None,
        }
        for s in rows
    ]


def create_site(db: Session, data: dict) -> Site:
    if not db.get(Mine, data["mine_id"]):
        raise ValueError("煤矿不存在")
    row = Site(**data)
    db.add(row)
    db.commit()
    db.refresh(row)
    cache_delete(CACHE_CAMERAS)
    return row


def get_site(db: Session, site_id: int) -> Optional[Site]:
    return db.get(Site, site_id)


def update_site(db: Session, row: Site, data: dict) -> Site:
    if data.get("mine_id") is not None and not db.get(Mine, data["mine_id"]):
        raise ValueError("煤矿不存在")
    for k, v in data.items():
        if v is not None:
            setattr(row, k, v)
    db.commit()
    db.refresh(row)
    cache_delete(CACHE_CAMERAS)
    return row


def delete_site(db: Session, row: Site) -> None:
    db.delete(row)
    db.commit()
    cache_delete(CACHE_CAMERAS)


def get_camera_tree(db: Session) -> dict:
    """煤矿 → 地点 → 摄像头 树，附带在线状态。"""
    import logging

    logger = logging.getLogger(__name__)
    online_keys = None
    zlm_error = None
    try:
        online_keys = zlm_client.list_online_media_keys()
    except Exception as e:
        zlm_error = str(e)
        logger.warning("查询 ZLM 在线流失败: %s", e)

    mines = db.scalars(
        select(Mine)
        .options(joinedload(Mine.sites).joinedload(Site.cameras))
        .order_by(Mine.id)
    ).unique().all()

    mine_nodes = []
    for mine in mines:
        site_nodes = []
        for site in sorted(mine.sites, key=lambda s: s.id):
            cams = [
                _camera_to_dict(c, online_keys=online_keys, zlm_error=zlm_error)
                for c in sorted(site.cameras, key=lambda c: c.id)
            ]
            site_nodes.append(
                {
                    "id": site.id,
                    "name": site.name,
                    "code": site.code,
                    "enabled": site.enabled,
                    "cameras": cams,
                }
            )
        mine_nodes.append(
            {
                "id": mine.id,
                "name": mine.name,
                "code": mine.code,
                "enabled": mine.enabled,
                "sites": site_nodes,
            }
        )

    unassigned_rows = db.scalars(
        select(Camera).where(Camera.site_id.is_(None)).order_by(Camera.id)
    ).all()
    unassigned = [
        _camera_to_dict(c, online_keys=online_keys, zlm_error=zlm_error)
        for c in unassigned_rows
    ]
    return {"mines": mine_nodes, "unassigned": unassigned}


# ---------- cameras ----------

def list_cameras(
    db: Session,
    page: int = 1,
    page_size: int = 50,
    site_id: Optional[int] = None,
    mine_id: Optional[int] = None,
) -> Tuple[List[dict], dict]:
    from sqlalchemy import func
    import logging

    logger = logging.getLogger(__name__)

    stmt = select(Camera).options(joinedload(Camera.site).joinedload(Site.mine))
    count_stmt = select(func.count()).select_from(Camera)
    if site_id is not None:
        stmt = stmt.where(Camera.site_id == site_id)
        count_stmt = count_stmt.where(Camera.site_id == site_id)
    if mine_id is not None:
        stmt = stmt.join(Site, Camera.site_id == Site.id).where(Site.mine_id == mine_id)
        count_stmt = (
            select(func.count())
            .select_from(Camera)
            .join(Site, Camera.site_id == Site.id)
            .where(Site.mine_id == mine_id)
        )

    total = db.scalar(count_stmt) or 0
    rows = (
        db.scalars(
            stmt.order_by(desc(Camera.id))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .unique()
        .all()
    )
    online_keys = None
    zlm_error = None
    try:
        online_keys = zlm_client.list_online_media_keys()
    except ZLMClientError as e:
        zlm_error = str(e)
        logger.warning("查询 ZLM 在线流失败: %s", e)
    except Exception as e:
        zlm_error = str(e)
        logger.exception("查询 ZLM 在线流异常")

    items = [
        _camera_to_dict(r, online_keys=online_keys, zlm_error=zlm_error) for r in rows
    ]
    return _page(items, int(total), page, page_size)


def enrich_camera(row: Camera) -> dict:
    """单条摄像头附带在线状态。"""
    import logging

    logger = logging.getLogger(__name__)
    # 确保 site/mine 可用
    if row.site_id and row.site is None:
        # lazy load
        _ = row.site
    online_keys = None
    zlm_error = None
    try:
        online_keys = zlm_client.list_online_media_keys()
    except Exception as e:
        zlm_error = str(e)
        logger.warning("查询 ZLM 在线流失败: %s", e)
    return _camera_to_dict(row, online_keys=online_keys, zlm_error=zlm_error)


def _default_zlm_stream_id(data: dict) -> str:
    code = str(data.get("camera_code") or "").strip()
    if code:
        return re.sub(r"[^0-9A-Za-z_\-]", "_", code)[:64]
    name = str(data.get("name") or "cam").strip()
    slug = re.sub(r"[^0-9A-Za-z_\-]", "_", name)[:40] or "cam"
    return f"{slug}_{uuid.uuid4().hex[:8]}"


def _prepare_camera_stream_fields(
    data: dict, *, existing: Optional[Camera] = None
) -> dict:
    """
    规范化摄像头取流字段：
    - push：已在 ZLM 推流，用 in_url 或 zlm_app+zlm_stream
    - proxy：摄像头原始地址，调用 addStreamProxy 写入 ZLM，再生成 in_url
    """
    out = dict(data)
    mode = str(
        out.get("ingest_mode")
        or (getattr(existing, "ingest_mode", None) if existing else None)
        or "push"
    ).strip().lower()
    if mode not in {"push", "proxy"}:
        raise ValueError("ingest_mode 仅支持 push / proxy")
    out["ingest_mode"] = mode

    if mode == "push":
        in_url = str(out.get("in_url") or "").strip()
        if not in_url and existing is not None and "in_url" not in data:
            in_url = str(existing.in_url or "").strip()
        app = str(out.get("zlm_app") or "").strip()
        stream = str(out.get("zlm_stream") or "").strip()
        if not app and existing is not None and "zlm_app" not in data:
            app = str(existing.zlm_app or "").strip()
        if not stream and existing is not None and "zlm_stream" not in data:
            stream = str(existing.zlm_stream or "").strip()

        if app and stream:
            out["zlm_app"] = app
            out["zlm_stream"] = stream
            out["in_url"] = settings.build_zlm_pull_url(app, stream)
        elif in_url:
            parsed = parse_stream_url(in_url)
            out["in_url"] = in_url
            if parsed:
                out["zlm_app"] = parsed["app"]
                out["zlm_stream"] = parsed["stream"]
            else:
                out.setdefault("zlm_app", "")
                out.setdefault("zlm_stream", "")
        else:
            raise ValueError(
                "推流接入模式请填写 ZLM 流地址(in_url)，或填写 zlm_app + zlm_stream"
            )
        # 切到 push 时清掉旧代理
        if existing and existing.proxy_key and (
            existing.ingest_mode == "proxy" or data.get("ingest_mode") == "push"
        ):
            try:
                zlm_client.del_stream_proxy(existing.proxy_key)
            except Exception as e:
                logger.warning("删除旧拉流代理失败 key=%s: %s", existing.proxy_key, e)
        out["proxy_key"] = None
        out["source_url"] = None
        return out

    # ---- proxy 模式 ----
    source = str(out.get("source_url") or "").strip()
    if not source and existing is not None and "source_url" not in data:
        source = str(existing.source_url or "").strip()
    if not source:
        raise ValueError("拉流代理模式必须填写摄像头原始地址 source_url")

    app = str(out.get("zlm_app") or "").strip()
    if not app and existing is not None and "zlm_app" not in data:
        app = str(existing.zlm_app or "").strip()
    app = app or "live"

    stream = str(out.get("zlm_stream") or "").strip()
    if not stream and existing is not None and "zlm_stream" not in data:
        stream = str(existing.zlm_stream or "").strip()
    if not stream:
        stream = _default_zlm_stream_id(
            {
                "camera_code": out.get("camera_code")
                or (existing.camera_code if existing else ""),
                "name": out.get("name") or (existing.name if existing else "cam"),
            }
        )

    need_new_proxy = True
    if existing and existing.ingest_mode == "proxy" and existing.proxy_key:
        same_src = source == str(existing.source_url or "").strip()
        same_app = app == str(existing.zlm_app or "").strip()
        same_stream = stream == str(existing.zlm_stream or "").strip()
        if same_src and same_app and same_stream:
            need_new_proxy = False
            out["proxy_key"] = existing.proxy_key
        else:
            try:
                zlm_client.del_stream_proxy(existing.proxy_key)
            except Exception as e:
                logger.warning(
                    "更新前删除旧拉流代理失败 key=%s: %s", existing.proxy_key, e
                )

    if need_new_proxy:
        key = zlm_client.add_stream_proxy(app, stream, source)
        out["proxy_key"] = key

    out["source_url"] = source
    out["zlm_app"] = app
    out["zlm_stream"] = stream
    out["in_url"] = settings.build_zlm_pull_url(app, stream)
    return out


def create_camera(db: Session, data: dict) -> Camera:
    data = _sync_camera_mine_code(db, dict(data))
    data = _prepare_camera_stream_fields(data)
    # 仅保留模型字段
    allowed = {
        "name",
        "in_url",
        "ingest_mode",
        "source_url",
        "zlm_app",
        "zlm_stream",
        "proxy_key",
        "camera_code",
        "mine_code",
        "site_id",
        "enabled",
        "remark",
    }
    payload = {k: data[k] for k in allowed if k in data}
    row = Camera(**payload)
    db.add(row)
    db.commit()
    db.refresh(row)
    cache_delete(CACHE_CAMERAS)
    return row


def get_camera(db: Session, camera_id: int) -> Optional[Camera]:
    return db.scalars(
        select(Camera)
        .options(joinedload(Camera.site).joinedload(Site.mine))
        .where(Camera.id == camera_id)
    ).first()


def update_camera(db: Session, row: Camera, data: dict) -> Camera:
    data = _sync_camera_mine_code(db, dict(data))
    # 若涉及取流相关字段，重新规范化
    stream_keys = {
        "ingest_mode",
        "in_url",
        "source_url",
        "zlm_app",
        "zlm_stream",
        "camera_code",
        "name",
    }
    if stream_keys.intersection(data.keys()):
        merged = {
            "ingest_mode": data.get("ingest_mode", row.ingest_mode),
            "in_url": data.get("in_url", row.in_url),
            "source_url": data.get("source_url", row.source_url),
            "zlm_app": data.get("zlm_app", row.zlm_app),
            "zlm_stream": data.get("zlm_stream", row.zlm_stream),
            "camera_code": data.get("camera_code", row.camera_code),
            "name": data.get("name", row.name),
        }
        prepared = _prepare_camera_stream_fields(merged, existing=row)
        for k in (
            "ingest_mode",
            "in_url",
            "source_url",
            "zlm_app",
            "zlm_stream",
            "proxy_key",
        ):
            data[k] = prepared.get(k)

    for k, v in data.items():
        if v is not None or k in {"site_id", "source_url", "proxy_key"}:
            setattr(row, k, v)
    db.commit()
    db.refresh(row)
    cache_delete(CACHE_CAMERAS)
    return row


def delete_camera(db: Session, row: Camera) -> None:
    key = getattr(row, "proxy_key", None)
    if key:
        try:
            zlm_client.del_stream_proxy(str(key))
        except Exception as e:
            logger.warning("删除摄像头时 delStreamProxy 失败 key=%s: %s", key, e)
    db.delete(row)
    db.commit()
    cache_delete(CACHE_CAMERAS)


# ---------- algorithm configs ----------

def list_algorithms(
    db: Session, page: int = 1, page_size: int = 50
) -> Tuple[List[AlgorithmConfig], dict]:
    from sqlalchemy import func

    total = db.scalar(select(func.count()).select_from(AlgorithmConfig)) or 0
    rows = (
        db.scalars(
            select(AlgorithmConfig)
            .order_by(desc(AlgorithmConfig.id))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .all()
    )
    return _page(list(rows), int(total), page, page_size)


def create_algorithm(db: Session, data: dict) -> AlgorithmConfig:
    resolve_skill_class(data["skill_name"])
    row = AlgorithmConfig(**data)
    db.add(row)
    db.commit()
    db.refresh(row)
    cache_delete(CACHE_ALGOS)
    return row


def get_algorithm(db: Session, algo_id: int) -> Optional[AlgorithmConfig]:
    return db.get(AlgorithmConfig, algo_id)


def update_algorithm(db: Session, row: AlgorithmConfig, data: dict) -> AlgorithmConfig:
    if data.get("skill_name"):
        resolve_skill_class(data["skill_name"])
    for k, v in data.items():
        setattr(row, k, v)
    db.commit()
    db.refresh(row)
    cache_delete(CACHE_ALGOS)
    return row


def delete_algorithm(db: Session, row: AlgorithmConfig) -> None:
    db.delete(row)
    db.commit()
    cache_delete(CACHE_ALGOS)


# ---------- task configs ----------

def _enrich_task(row: TaskConfig) -> dict:
    from app.services.task_scheduler import is_within_schedule

    schedule = row.schedule
    data = {
        "id": row.id,
        "name": row.name,
        "camera_id": row.camera_id,
        "algorithm_config_id": row.algorithm_config_id,
        "scene_id": row.scene_id,
        "output_format": row.output_format,
        "out_fps": row.out_fps,
        "enabled": row.enabled,
        "alert_image_enabled": bool(getattr(row, "alert_image_enabled", True)),
        "alert_video_enabled": bool(getattr(row, "alert_video_enabled", False)),
        "push_annotated_stream": bool(getattr(row, "push_annotated_stream", False)),
        "schedule": schedule,
        "schedule_active": bool(schedule and schedule.get("enabled") and is_within_schedule(schedule)),
        "last_runtime_task_id": row.last_runtime_task_id,
        "remark": row.remark,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "camera_name": row.camera.name if row.camera else None,
        "algorithm_name": row.algorithm_config.name if row.algorithm_config else None,
        "skill_name": row.algorithm_config.skill_name if row.algorithm_config else None,
        "runtime_status": None,
        "flv_url": None,
    }
    if row.last_runtime_task_id:
        task = stream_task_manager.get_task(row.last_runtime_task_id)
        if task:
            d = stream_task_manager.enrich_task_dict(task)
            data["runtime_status"] = d.get("status")
            if data["push_annotated_stream"]:
                data["flv_url"] = settings.build_flv_play_url_from_out_url(d.get("out_url"))
                if not data["flv_url"] and d.get("scene_id") and d.get("skill_name"):
                    data["flv_url"] = settings.build_flv_play_url(
                        d["scene_id"], d["skill_name"]
                    )
            data["ingest_branches"] = d.get("ingest_branches")
    return data


def list_tasks(db: Session, page: int = 1, page_size: int = 50) -> Tuple[List[dict], dict]:
    from sqlalchemy import func

    total = db.scalar(select(func.count()).select_from(TaskConfig)) or 0
    rows = (
        db.scalars(
            select(TaskConfig)
            .options(
                joinedload(TaskConfig.camera),
                joinedload(TaskConfig.algorithm_config),
            )
            .order_by(desc(TaskConfig.id))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .unique()
        .all()
    )
    return _page([_enrich_task(r) for r in rows], int(total), page, page_size)


def create_task(db: Session, data: dict) -> dict:
    from app.services.task_scheduler import normalize_schedule

    if not db.get(Camera, data["camera_id"]):
        raise ValueError("摄像头不存在")
    if not db.get(AlgorithmConfig, data["algorithm_config_id"]):
        raise ValueError("算法配置不存在")
    if "schedule" in data:
        data["schedule"] = normalize_schedule(data.get("schedule"))
    # 兼容旧请求体中的 alert_rules 字段
    data.pop("alert_rules", None)
    row = TaskConfig(**data)
    db.add(row)
    db.commit()
    db.refresh(row)
    row = db.scalars(
        select(TaskConfig)
        .options(
            joinedload(TaskConfig.camera),
            joinedload(TaskConfig.algorithm_config),
        )
        .where(TaskConfig.id == row.id)
    ).first()
    return _enrich_task(row)


def get_task(db: Session, task_id: int) -> Optional[TaskConfig]:
    return db.scalars(
        select(TaskConfig)
        .options(
            joinedload(TaskConfig.camera),
            joinedload(TaskConfig.algorithm_config),
        )
        .where(TaskConfig.id == task_id)
    ).first()


def update_task(db: Session, row: TaskConfig, data: dict) -> dict:
    from app.services.task_scheduler import normalize_schedule

    if data.get("camera_id") and not db.get(Camera, data["camera_id"]):
        raise ValueError("摄像头不存在")
    if data.get("algorithm_config_id") and not db.get(
        AlgorithmConfig, data["algorithm_config_id"]
    ):
        raise ValueError("算法配置不存在")
    if "schedule" in data:
        data["schedule"] = normalize_schedule(data.get("schedule"))
    data.pop("alert_rules", None)
    for k, v in data.items():
        if v is not None or k == "schedule":
            setattr(row, k, v)
    db.commit()
    refreshed = get_task(db, row.id)
    return _enrich_task(refreshed)


def delete_task(db: Session, row: TaskConfig) -> None:
    db.delete(row)
    db.commit()


def build_stream_payload(row: TaskConfig) -> Dict[str, Any]:
    cam = row.camera
    algo = row.algorithm_config
    if not cam or not algo:
        raise ValueError("任务关联的摄像头或算法配置缺失")
    if not cam.enabled:
        raise ValueError("摄像头已禁用")
    if not algo.enabled:
        raise ValueError("算法配置已禁用")
    if not row.enabled:
        raise ValueError("任务配置已禁用")

    resolve_skill_class(algo.skill_name)
    payload: Dict[str, Any] = {
        "in_url": cam.in_url,
        "camera_id": cam.id,
        "scene_id": row.scene_id,
        "skill_name": algo.skill_name,
        "output_format": row.output_format,
        "out_fps": row.out_fps,
        "enter_count": algo.enter_count,
        "gate_direction": algo.gate_direction,
        "mine_code": cam.mine_code or None,
        "camera_code": cam.camera_code or None,
        "alert_image_enabled": bool(getattr(row, "alert_image_enabled", True)),
        "alert_video_enabled": bool(getattr(row, "alert_video_enabled", False)),
        "push_annotated_stream": bool(getattr(row, "push_annotated_stream", False)),
    }
    if algo.count_line:
        payload["count_line"] = algo.count_line
    if algo.bypass_line:
        payload["bypass_line"] = algo.bypass_line
    if algo.extra_params:
        payload["skill_config"] = {"params": algo.extra_params}
    return settings.resolve_stream_start(payload)


def start_task_config(db: Session, row: TaskConfig) -> dict:
    from app.services.task_scheduler import assert_manual_start_allowed

    assert_manual_start_allowed(row)
    payload = build_stream_payload(row)
    runtime = stream_task_manager.start_task(payload)
    row.last_runtime_task_id = runtime.task_id
    db.commit()
    refreshed = get_task(db, row.id)
    return _enrich_task(refreshed)


def stop_task_config(db: Session, row: TaskConfig) -> dict:
    if not row.last_runtime_task_id:
        raise ValueError("该任务尚未启动过运行时实例")
    stream_task_manager.stop_task(row.last_runtime_task_id)
    refreshed = get_task(db, row.id)
    return _enrich_task(refreshed)


# ---------- alerts ----------

def _clone_alert_record(row: AlertRecord) -> AlertRecord:
    """在 Session 内拷贝为未绑定实例，避免 commit 后 DetachedInstanceError。"""
    types = row.recognition_types
    if isinstance(types, list):
        types = list(types)
    payload = row.payload
    if isinstance(payload, dict):
        payload = dict(payload)
    return AlertRecord(
        id=int(row.id),
        alert_uid=str(row.alert_uid or ""),
        scene_id=str(row.scene_id or ""),
        skill_name=str(row.skill_name or ""),
        recognition_types=types,
        message=str(row.message or ""),
        count=int(row.count or 0),
        enter_count=int(row.enter_count or 0),
        image_url=row.image_url,
        video_url=row.video_url,
        category=str(row.category or "alert"),
        payload=payload,
        status=str(row.status or "new"),
        created_at=row.created_at,
    )


def persist_alert_event(event: Dict[str, Any]) -> Optional[AlertRecord]:
    """供告警子进程调用：将标准化告警/事件写入 MySQL。"""
    from app.db import session_scope

    uid = (
        str(event.get("alert_id") or "").strip()
        or f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    category = str(event.get("category") or "").strip().lower()
    if category not in {"alert", "event"}:
        category = classify_record_category(
            event.get("recognition_types"),
            skill_name=str(event.get("skill_name") or ""),
            has_person_count_change=bool(event.get("has_person_count_change")),
        )
    try:
        with session_scope() as db:
            existing = db.scalars(
                select(AlertRecord).where(AlertRecord.alert_uid == uid)
            ).first()
            if existing:
                return _clone_alert_record(existing)
            row = AlertRecord(
                alert_uid=uid,
                scene_id=str(event.get("scene_id") or ""),
                skill_name=str(event.get("skill_name") or ""),
                recognition_types=event.get("recognition_types") or [],
                message=str(event.get("message") or ""),
                count=int(event.get("count") or 0),
                enter_count=int(event.get("enter_count") or 0),
                image_url=event.get("image_minio_url") or event.get("pic_url"),
                video_url=event.get("video_minio_url") or event.get("video_url"),
                category=category,
                payload=event,
                status="new",
            )
            db.add(row)
            db.flush()
            db.refresh(row)
            # 脱离会话前取出字段，避免后续访问过期对象
            alert_pk = int(row.id)
            scene_id = row.scene_id
            skill_name = row.skill_name
            has_video = bool(row.video_url)
            row_category = row.category
            detached = _clone_alert_record(row)
        # 仅违规报警且任务开启「报警视频」时异步截取证据视频
        want_video = bool(event.get("alert_video_enabled", False))
        if (
            row_category == "alert"
            and want_video
            and not has_video
            and scene_id
            and skill_name
        ):
            schedule_alert_evidence_clip(alert_pk, scene_id, skill_name)
        return detached
    except Exception:
        # 子进程写库失败不影响推流主链路
        import logging

        logging.getLogger(__name__).exception("告警落库失败 uid=%s", uid)
        return None


def persist_classified_records(event: Dict[str, Any]) -> List[AlertRecord]:
    """
    按识别类型拆分落库：04-07 → 报警；01/02 / 画面人数 → 事件。
    同帧既有过线又有违规时各写一条。
    """
    import copy

    types = _normalize_type_codes(event.get("recognition_types"))
    skill_name = str(event.get("skill_name") or "")
    base_uid = (
        str(event.get("alert_id") or "").strip()
        or f"rec_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
    )
    alert_types = [t for t in types if t in ALERT_RECOGNITION_TYPES]
    event_types = [t for t in types if t in EVENT_RECOGNITION_TYPES]
    rows: List[AlertRecord] = []

    def _persist_one(
        *,
        category: str,
        recognition_types: List[str],
        uid_suffix: str,
    ) -> None:
        item = copy.deepcopy(event)
        item["category"] = category
        item["recognition_types"] = recognition_types
        item["alert_id"] = f"{base_uid}_{uid_suffix}"
        item["type"] = "violation_alert" if category == "alert" else "recognition_event"
        row = persist_alert_event(item)
        if row is not None:
            rows.append(row)

    if alert_types:
        _persist_one(category="alert", recognition_types=alert_types, uid_suffix="alert")
    if event_types:
        _persist_one(category="event", recognition_types=event_types, uid_suffix="event")

    # 画面人数等：无 01/02/04-07 编码，但仍需记事件
    if (
        not alert_types
        and not event_types
        and (
            skill_name in PRESENCE_SKILL_NAMES
            or bool(event.get("has_person_count_change"))
        )
    ):
        _persist_one(category="event", recognition_types=[], uid_suffix="event")

    # 兜底：有违规标志但编码缺失时仍记报警
    if (
        not rows
        and (
            bool(event.get("has_bypass_violation"))
            or bool(event.get("has_count_exit_violation"))
        )
    ):
        _persist_one(category="alert", recognition_types=types, uid_suffix="alert")

    return rows


def schedule_alert_evidence_clip(
    alert_id: int,
    scene_id: str,
    skill_name: str,
    *,
    back_ms: int = 5000,
    forward_ms: int = 5000,
) -> None:
    """后台截取约 10s 证据视频并回写 video_url（不阻塞告警主流程）。"""
    import logging
    import threading

    log = logging.getLogger(__name__)

    def _run() -> None:
        try:
            attach_alert_evidence_clip(
                alert_id,
                scene_id,
                skill_name,
                back_ms=back_ms,
                forward_ms=forward_ms,
            )
        except Exception:
            log.exception(
                "告警证据视频截取失败 alert_id=%s scene=%s skill=%s",
                alert_id,
                scene_id,
                skill_name,
            )

    threading.Thread(
        target=_run,
        name=f"alert-clip-{alert_id}",
        daemon=True,
    ).start()


def attach_alert_evidence_clip(
    alert_id: int,
    scene_id: str,
    skill_name: str,
    *,
    back_ms: int = 5000,
    forward_ms: int = 5000,
) -> Optional[str]:
    """
    从推流地址截取前后合计约 10s 的 MP4，上传 MinIO，写回 mgmt_alerts.video_url。
    """
    import base64
    import logging
    import time

    from app.db import session_scope
    from app.services.minio_client import minio_storage
    from app.services.video_clip_service import video_clip_service

    log = logging.getLogger(__name__)
    stream_url = settings.build_push_url(scene_id=scene_id, skill_name=skill_name)
    result = video_clip_service.capture_event_clip(
        video_url=stream_url,
        back_ms=back_ms,
        forward_ms=forward_ms,
    )
    stamp = time.strftime("%Y%m%d_%H%M%S")
    object_name = f"alerts/{scene_id}/{skill_name}/{alert_id}_{stamp}.mp4"
    video_minio_url = minio_storage.upload_bytes(
        base64.b64decode(result.video_base64),
        object_name=object_name,
        content_type="video/mp4",
    )

    with session_scope() as db:
        row = db.get(AlertRecord, alert_id)
        if row is None:
            log.warning("告警不存在，跳过视频回写 alert_id=%s", alert_id)
            return video_minio_url
        row.video_url = video_minio_url
        payload = dict(row.payload or {})
        payload["video_minio_url"] = video_minio_url
        payload["video_zlm_url"] = result.video_url
        payload["video_back_ms"] = back_ms
        payload["video_forward_ms"] = forward_ms
        row.payload = payload

    log.info(
        "告警证据视频已回写 alert_id=%s url=%s duration≈%ss",
        alert_id,
        video_minio_url,
        (back_ms + forward_ms) / 1000.0,
    )
    return video_minio_url


def list_alerts(
    db: Session,
    page: int = 1,
    page_size: int = 20,
    scene_id: Optional[str] = None,
    status: Optional[str] = None,
    category: Optional[str] = None,
    skill_name: Optional[str] = None,
    recognition_type: Optional[str] = None,
    time_from: Optional[datetime] = None,
    time_to: Optional[datetime] = None,
) -> Tuple[List[AlertRecord], dict]:
    from sqlalchemy import String, cast, func

    stmt = select(AlertRecord)
    count_stmt = select(func.count()).select_from(AlertRecord)

    def _apply(s):
        if scene_id:
            s = s.where(AlertRecord.scene_id.like(f"%{scene_id.strip()}%"))
        if status:
            s = s.where(AlertRecord.status == status)
        if category:
            s = s.where(AlertRecord.category == category)
        if skill_name:
            s = s.where(AlertRecord.skill_name.like(f"%{skill_name.strip()}%"))
        if recognition_type:
            code = recognition_type.strip()
            # recognition_types 为 JSON 数组，按包含匹配
            s = s.where(cast(AlertRecord.recognition_types, String).like(f'%"{code}"%'))
        if time_from is not None:
            s = s.where(AlertRecord.created_at >= time_from)
        if time_to is not None:
            s = s.where(AlertRecord.created_at <= time_to)
        return s

    stmt = _apply(stmt)
    count_stmt = _apply(count_stmt)
    total = db.scalar(count_stmt) or 0
    rows = (
        db.scalars(
            stmt.order_by(desc(AlertRecord.id))
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .all()
    )
    return _page(list(rows), int(total), page, page_size)


def get_alert(db: Session, alert_id: int) -> Optional[AlertRecord]:
    return db.get(AlertRecord, alert_id)


def update_alert_status(db: Session, row: AlertRecord, status: str) -> AlertRecord:
    row.status = status
    db.commit()
    db.refresh(row)
    return row


def list_triton_models() -> Dict[str, Any]:
    """从 Triton 仓库索引拉取已部署模型列表。"""
    from app.services.triton_client import triton_client

    result: Dict[str, Any] = {
        "server_url": settings.TRITON_URL,
        "server_live": False,
        "server_ready": False,
        "server_name": None,
        "server_version": None,
        "items": [],
        "error": None,
    }
    try:
        result["server_live"] = bool(triton_client.is_server_live())
        result["server_ready"] = bool(triton_client.is_server_ready())
    except Exception as e:
        result["error"] = f"无法连接 Triton: {e}"
        return result

    if not result["server_live"]:
        result["error"] = f"Triton 服务不可达: {settings.TRITON_URL}"
        return result

    meta = triton_client.get_server_metadata() or {}
    result["server_name"] = meta.get("name")
    result["server_version"] = meta.get("version")

    index = triton_client.get_model_repository_index()
    if index is None:
        result["error"] = "获取模型仓库索引失败"
        return result

    models = index.get("models") if isinstance(index, dict) else None
    if not isinstance(models, list):
        # 部分版本可能直接返回 list
        models = index if isinstance(index, list) else []

    items: List[Dict[str, Any]] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "").strip()
        if not name:
            continue
        version = m.get("version")
        version_str = "" if version is None else str(version)
        state = m.get("state")
        ready = False
        try:
            ready = bool(triton_client.is_model_ready(name, version_str))
        except Exception:
            ready = str(state or "").upper() == "READY"
        items.append(
            {
                "name": name,
                "version": version_str or None,
                "state": str(state) if state is not None else ("READY" if ready else None),
                "ready": ready,
            }
        )

    items.sort(key=lambda x: (x["name"], x.get("version") or ""))
    result["items"] = items
    return result
