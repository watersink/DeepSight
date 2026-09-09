"""视频流推流任务进程管理（摄像头级 ingest + 进程内 skill fan-out）"""
from __future__ import annotations

import logging
import multiprocessing
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from app.services.alert_hub import alert_hub
from app.services.stream_worker import run_camera_ingest_worker

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


def camera_ingest_key(config: Dict[str, Any]) -> str:
    """同一摄像头（或同一 in_url）共用一个 ingest 进程。"""
    cam_id = config.get("camera_id")
    if cam_id is not None and str(cam_id).strip() != "":
        return f"cam:{cam_id}"
    in_url = str(config.get("in_url") or "").strip()
    if in_url:
        return f"url:{in_url}"
    scene_id = str(config.get("scene_id") or "").strip()
    return f"scene:{scene_id or 'default'}"


@dataclass
class StreamTask:
    """对外仍按「一个 skill 分支 = 一个任务」暴露。"""

    task_id: str
    config: Dict[str, Any]
    camera_key: str
    status: TaskStatus = TaskStatus.STARTING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "scene_id": self.config.get("scene_id"),
            "skill_name": self.config.get("skill_name"),
            "in_url": self.config.get("in_url"),
            "out_url": self.config.get("out_url"),
            "camera_id": self.config.get("camera_id"),
            "camera_key": self.camera_key,
            "pid": None,  # 由 manager 填充
            "created_at": self.created_at,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "error_message": self.error_message,
        }


@dataclass
class CameraIngestSession:
    camera_key: str
    source_url: str
    process: multiprocessing.Process
    stop_event: multiprocessing.Event
    control_queue: multiprocessing.Queue
    branch_ids: Set[str] = field(default_factory=set)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    pid: Optional[int] = None

    def is_alive(self) -> bool:
        return bool(self.process and self.process.is_alive())


class StreamTaskManager:
    """管理摄像头 ingest 子进程，并在进程内 fan-out 多个 skill。"""

    _instance: Optional["StreamTaskManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tasks: Dict[str, StreamTask] = {}
                    cls._instance._sessions: Dict[str, CameraIngestSession] = {}
                    cls._instance._tasks_lock = threading.Lock()
        return cls._instance

    def _refresh_task_status(self, task: StreamTask) -> None:
        session = self._sessions.get(task.camera_key)
        alive = bool(session and session.is_alive() and task.task_id in session.branch_ids)
        if task.status == TaskStatus.STOPPING and not alive:
            task.status = TaskStatus.STOPPED
            task.stopped_at = task.stopped_at or datetime.now().isoformat()
            return
        if task.status in (TaskStatus.STOPPED,):
            return
        if alive:
            if task.status == TaskStatus.STARTING:
                task.status = TaskStatus.RUNNING
            return
        if task.status in (TaskStatus.RUNNING, TaskStatus.STARTING):
            if session and session.process and session.process.exitcode not in (0, None):
                task.status = TaskStatus.FAILED
                task.error_message = f"进程退出码: {session.process.exitcode}"
            else:
                task.status = TaskStatus.STOPPED
            task.stopped_at = task.stopped_at or datetime.now().isoformat()

    def _find_scene_conflict(self, scene_id: str, exclude_task_id: Optional[str] = None) -> Optional[str]:
        if not scene_id:
            return None
        for tid, task in self._tasks.items():
            if exclude_task_id and tid == exclude_task_id:
                continue
            if task.config.get("scene_id") != scene_id:
                continue
            self._refresh_task_status(task)
            if task.status in (TaskStatus.RUNNING, TaskStatus.STARTING):
                return tid
        return None

    def start_task(self, config: Dict[str, Any]) -> StreamTask:
        scene_id = config.get("scene_id")
        camera_key = camera_ingest_key(config)
        source_url = str(config.get("in_url") or "").strip()
        if not source_url:
            raise ValueError("缺少 in_url，无法启动摄像头 ingest")

        with self._tasks_lock:
            conflict = self._find_scene_conflict(str(scene_id or ""))
            if conflict:
                raise ValueError(f"场景 {scene_id} 已有运行中的推流任务: {conflict}")

            task_id = str(uuid.uuid4())
            task = StreamTask(
                task_id=task_id,
                config=dict(config),
                camera_key=camera_key,
                status=TaskStatus.STARTING,
            )

            session = self._sessions.get(camera_key)
            if session and not session.is_alive():
                # 僵尸会话清理
                for bid in list(session.branch_ids):
                    old = self._tasks.get(bid)
                    if old:
                        old.status = TaskStatus.STOPPED
                        old.stopped_at = old.stopped_at or datetime.now().isoformat()
                self._sessions.pop(camera_key, None)
                session = None

            if session and session.is_alive():
                try:
                    session.control_queue.put(
                        {"cmd": "add", "branch_id": task_id, "config": dict(config)},
                        timeout=5,
                    )
                except Exception as exc:
                    raise RuntimeError(f"向摄像头 ingest 添加 skill 失败: {exc}") from exc
                session.branch_ids.add(task_id)
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now().isoformat()
                self._tasks[task_id] = task
                logger.info(
                    "已向现有 ingest 添加 skill task_id=%s camera=%s scene=%s skill=%s pid=%s",
                    task_id,
                    camera_key,
                    scene_id,
                    config.get("skill_name"),
                    session.pid,
                )
                return task

            stop_event = multiprocessing.Event()
            control_queue: multiprocessing.Queue = multiprocessing.Queue()
            ingest_meta = {
                "source_url": source_url,
                "fps": config.get("out_fps"),
                "reconnect_delay": config.get("reconnect_delay"),
                "queue_size": config.get("queue_size"),
            }
            process = multiprocessing.Process(
                target=run_camera_ingest_worker,
                args=(
                    ingest_meta,
                    [{"branch_id": task_id, "config": dict(config)}],
                    stop_event,
                    control_queue,
                    alert_hub.mp_queue,
                ),
                name=f"ingest-{camera_key}",
                daemon=False,
            )
            process.start()
            session = CameraIngestSession(
                camera_key=camera_key,
                source_url=source_url,
                process=process,
                stop_event=stop_event,
                control_queue=control_queue,
                branch_ids={task_id},
                pid=process.pid,
            )
            self._sessions[camera_key] = session
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()
            self._tasks[task_id] = task
            logger.info(
                "已启动摄像头 ingest task_id=%s camera=%s scene=%s skill=%s pid=%s",
                task_id,
                camera_key,
                scene_id,
                config.get("skill_name"),
                process.pid,
            )
            return task

    def stop_task(self, task_id: str, timeout: float = 10.0) -> StreamTask:
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise KeyError(f"任务不存在: {task_id}")
            session = self._sessions.get(task.camera_key)

        if not session or not session.is_alive() or task_id not in session.branch_ids:
            task.status = TaskStatus.STOPPED
            task.stopped_at = task.stopped_at or datetime.now().isoformat()
            if session:
                session.branch_ids.discard(task_id)
                if not session.branch_ids and not session.is_alive():
                    with self._tasks_lock:
                        self._sessions.pop(task.camera_key, None)
            return task

        task.status = TaskStatus.STOPPING
        remaining = len(session.branch_ids) - 1

        if remaining > 0:
            try:
                session.control_queue.put(
                    {"cmd": "remove", "branch_id": task_id},
                    timeout=5,
                )
            except Exception:
                logger.exception("发送 remove 指令失败 task_id=%s，将停止整个 ingest", task_id)
                remaining = 0

        if remaining > 0:
            session.branch_ids.discard(task_id)
            task.status = TaskStatus.STOPPED
            task.stopped_at = datetime.now().isoformat()
            logger.info(
                "已从 ingest 移除 skill task_id=%s camera=%s remaining=%d",
                task_id,
                task.camera_key,
                len(session.branch_ids),
            )
            return task

        # 最后一个分支：停止整个摄像头 ingest 进程
        session.branch_ids.discard(task_id)
        if session.stop_event:
            session.stop_event.set()
        try:
            session.control_queue.put({"cmd": "stop"}, timeout=1)
        except Exception:
            pass

        session.process.join(timeout=timeout)
        if session.process.is_alive():
            logger.warning(
                "ingest %s 未在 %ss 内退出，强制终止", task.camera_key, timeout
            )
            session.process.terminate()
            session.process.join(timeout=3)

        task.status = TaskStatus.STOPPED
        task.stopped_at = datetime.now().isoformat()
        with self._tasks_lock:
            # 同进程其它残留分支一并标记停止
            for bid in list(session.branch_ids):
                other = self._tasks.get(bid)
                if other:
                    other.status = TaskStatus.STOPPED
                    other.stopped_at = other.stopped_at or datetime.now().isoformat()
            session.branch_ids.clear()
            self._sessions.pop(task.camera_key, None)

        logger.info("摄像头 ingest 已停止 task_id=%s camera=%s", task_id, task.camera_key)
        return task

    def get_task(self, task_id: str) -> Optional[StreamTask]:
        with self._tasks_lock:
            task = self._tasks.get(task_id)
            if not task:
                return None
            self._refresh_task_status(task)
            return task

    def list_tasks(self) -> List[StreamTask]:
        with self._tasks_lock:
            tasks = list(self._tasks.values())
            for t in tasks:
                self._refresh_task_status(t)
            return tasks

    def stop_all(self, timeout: float = 10.0):
        with self._tasks_lock:
            # 优先按 session 停，避免重复
            session_keys = list(self._sessions.keys())
            task_ids = list(self._tasks.keys())

        for key in session_keys:
            session = self._sessions.get(key)
            if not session:
                continue
            if session.stop_event:
                session.stop_event.set()
            try:
                session.control_queue.put({"cmd": "stop"}, timeout=1)
            except Exception:
                pass
            if session.process.is_alive():
                session.process.join(timeout=timeout)
                if session.process.is_alive():
                    session.process.terminate()
                    session.process.join(timeout=3)

        for task_id in task_ids:
            task = self._tasks.get(task_id)
            if task:
                task.status = TaskStatus.STOPPED
                task.stopped_at = task.stopped_at or datetime.now().isoformat()

        with self._tasks_lock:
            self._sessions.clear()

    def enrich_task_dict(self, task: StreamTask) -> Dict[str, Any]:
        data = task.to_dict()
        session = self._sessions.get(task.camera_key)
        alive = bool(session and session.is_alive() and task.task_id in session.branch_ids)
        data["pid"] = session.pid if alive and session else None
        data["status"] = task.status.value
        data["ingest_branches"] = len(session.branch_ids) if session else 0
        return data


stream_task_manager = StreamTaskManager()
