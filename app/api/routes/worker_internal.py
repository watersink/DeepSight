"""Worker 内部 API：供 API 机远程启停编解码任务。"""
from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Path, Query, status

from app.core.config import settings
from app.plugins.skill_registry import SkillNotFoundError, list_available_skills, resolve_skill_class
from app.services.stream_task_manager import stream_task_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/internal/v1", tags=["Worker内部接口"])


def verify_worker_token(
    x_worker_token: Annotated[Optional[str], Header()] = None,
) -> None:
    expected = (settings.WORKER_TOKEN or "").strip()
    if not expected:
        return
    if (x_worker_token or "").strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 Worker Token",
        )


def _to_dict(task) -> Dict[str, Any]:
    return stream_task_manager.enrich_task_dict(task)


@router.get("/health", summary="Worker 健康检查")
def health(_: None = Depends(verify_worker_token)):
    tasks = stream_task_manager.list_tasks()
    running = 0
    for t in tasks:
        data = _to_dict(t)
        if data.get("status") in {"running", "starting"}:
            running += 1
    return {
        "status": "ok",
        "role": "worker",
        "running_tasks": running,
        "triton_url": settings.TRITON_URL,
    }


@router.get("/skills", summary="本 Worker 可用技能")
def skills(_: None = Depends(verify_worker_token)):
    return {"skills": list_available_skills()}


@router.get("/models", summary="本 Worker 关联 Triton 上的模型")
def models(_: None = Depends(verify_worker_token)):
    from app.services.mgmt_service import list_triton_models

    data = list_triton_models()
    data["worker_role"] = "worker"
    return data


@router.post("/tasks/start", summary="启动推流任务")
def start_task(
    payload: Dict[str, Any],
    _: None = Depends(verify_worker_token),
):
    skill_name = str(payload.get("skill_name") or "").strip()
    if not skill_name:
        raise HTTPException(status_code=400, detail="缺少 skill_name")
    try:
        resolve_skill_class(skill_name)
    except SkillNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    config = settings.resolve_stream_start(payload)
    try:
        task = stream_task_manager.start_task(config)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except Exception as e:
        logger.exception("Worker 启动任务失败")
        raise HTTPException(status_code=500, detail=str(e)) from e
    return _to_dict(task)


@router.post("/tasks/{task_id}/stop", summary="停止推流任务")
def stop_task(
    task_id: Annotated[str, Path()],
    timeout: float = Query(default=10.0, ge=1.0, le=120.0),
    _: None = Depends(verify_worker_token),
):
    try:
        task = stream_task_manager.stop_task(task_id, timeout=timeout)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    except Exception as e:
        logger.exception("Worker 停止任务失败 task_id=%s", task_id)
        raise HTTPException(status_code=500, detail=str(e)) from e
    return _to_dict(task)


@router.get("/tasks", summary="列出本机推流任务")
def list_tasks(_: None = Depends(verify_worker_token)):
    tasks = [_to_dict(t) for t in stream_task_manager.list_tasks()]
    return {"total": len(tasks), "tasks": tasks}


@router.get("/tasks/{task_id}", summary="查询单个推流任务")
def get_task(
    task_id: Annotated[str, Path()],
    _: None = Depends(verify_worker_token),
):
    task = stream_task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    return _to_dict(task)
