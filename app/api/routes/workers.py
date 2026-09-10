"""Worker 节点管理 API（供管理台选择摄像头绑定的算力节点）。"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, status

from app.api.deps import get_current_user
from app.db.models import User
from app.services.runtime_gateway import WorkerApiError, runtime_gateway
from app.services.worker_nodes import list_worker_nodes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workers", tags=["Worker节点"])


@router.get("", summary="列出可用 Worker 节点")
def list_workers(_user: Annotated[User, Depends(get_current_user)]):
    items: List[Dict[str, Any]] = []
    for node in list_worker_nodes():
        entry: Dict[str, Any] = {
            "id": node.id,
            "name": node.name,
            "url": node.url,
            "local": node.is_local,
            "online": None,
            "running_tasks": None,
            "triton_url": None,
            "error": None,
        }
        try:
            health = runtime_gateway.health(node.id)
            entry["online"] = True
            entry["running_tasks"] = health.get("running_tasks")
            entry["triton_url"] = health.get("triton_url")
        except Exception as exc:
            entry["online"] = False
            entry["error"] = str(exc)
        items.append(entry)
    return {"items": items, "total": len(items)}


@router.get("/{worker_id}/skills", summary="查询指定 Worker 可用算法/技能")
def worker_skills(
    worker_id: Annotated[str, Path()],
    _user: Annotated[User, Depends(get_current_user)],
):
    nodes = {n.id for n in list_worker_nodes()}
    if worker_id not in nodes:
        raise HTTPException(status_code=404, detail=f"未知 Worker: {worker_id}")
    try:
        skills = runtime_gateway.list_skills(worker_id)
    except WorkerApiError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    except Exception as e:
        logger.exception("获取 Worker 技能失败 worker=%s", worker_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
    return {"worker_id": worker_id, "skills": skills}


@router.get("/{worker_id}/models", summary="查询指定 Worker 上的 Triton 模型")
def worker_models(
    worker_id: Annotated[str, Path()],
    _user: Annotated[User, Depends(get_current_user)],
):
    nodes = {n.id for n in list_worker_nodes()}
    if worker_id not in nodes:
        raise HTTPException(status_code=404, detail=f"未知 Worker: {worker_id}")
    try:
        return runtime_gateway.list_models(worker_id)
    except WorkerApiError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    except Exception as e:
        logger.exception("获取 Worker 模型失败 worker=%s", worker_id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/{worker_id}/health", summary="探测 Worker 健康状态")
def worker_health(
    worker_id: Annotated[str, Path()],
    _user: Annotated[User, Depends(get_current_user)],
):
    nodes = {n.id for n in list_worker_nodes()}
    if worker_id not in nodes:
        raise HTTPException(status_code=404, detail=f"未知 Worker: {worker_id}")
    try:
        return runtime_gateway.health(worker_id)
    except WorkerApiError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
