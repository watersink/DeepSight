"""运行时任务网关：本机 stream_task_manager 或远程 Worker HTTP。"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import httpx

from app.core.config import settings
from app.plugins.skill_registry import list_available_skills
from app.services.stream_task_manager import StreamTask, stream_task_manager
from app.services.worker_nodes import (
    WorkerNode,
    get_worker_node,
    has_local_worker,
    list_worker_nodes,
)

logger = logging.getLogger(__name__)


class WorkerApiError(RuntimeError):
    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


class RuntimeGateway:
    """统一启停/查询推流运行时，并记录 task_id → worker_id。"""

    _instance: Optional["RuntimeGateway"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._task_workers: Dict[str, str] = {}
                    cls._instance._map_lock = threading.Lock()
        return cls._instance

    def _bind(self, task_id: str, worker_id: str) -> None:
        with self._map_lock:
            self._task_workers[task_id] = worker_id

    def _unbind(self, task_id: str) -> None:
        with self._map_lock:
            self._task_workers.pop(task_id, None)

    def bound_worker_id(self, task_id: str) -> Optional[str]:
        with self._map_lock:
            return self._task_workers.get(task_id)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = (settings.WORKER_TOKEN or "").strip()
        if token:
            headers["X-Worker-Token"] = token
        return headers

    def _remote(self, node: WorkerNode, method: str, path: str, **kwargs) -> Any:
        url = node.url.rstrip("/") + path
        timeout = settings.WORKER_HTTP_TIMEOUT
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.request(method, url, headers=self._headers(), **kwargs)
        except httpx.RequestError as exc:
            raise WorkerApiError(f"无法连接 Worker [{node.id}] {node.url}: {exc}") from exc

        if resp.status_code >= 400:
            detail = resp.text
            try:
                body = resp.json()
                detail = body.get("detail") or body.get("message") or detail
            except Exception:
                pass
            raise WorkerApiError(
                f"Worker [{node.id}] 返回 {resp.status_code}: {detail}",
                status_code=resp.status_code,
            )
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    def _enrich_local(self, task: StreamTask, worker_id: str) -> Dict[str, Any]:
        data = stream_task_manager.enrich_task_dict(task)
        data["worker_id"] = worker_id
        node = get_worker_node(worker_id)
        data["worker_name"] = node.name
        return data

    def start_task(
        self,
        config: Dict[str, Any],
        worker_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        node = get_worker_node(worker_id)
        if node.is_local:
            task = stream_task_manager.start_task(config)
            data = self._enrich_local(task, node.id)
            self._bind(data["task_id"], node.id)
            return data

        payload = dict(config)
        body = self._remote(node, "POST", "/internal/v1/tasks/start", json=payload)
        if not isinstance(body, dict) or not body.get("task_id"):
            raise WorkerApiError(f"Worker [{node.id}] 启动响应无效")
        body["worker_id"] = node.id
        body["worker_name"] = node.name
        self._bind(str(body["task_id"]), node.id)
        return body

    def stop_task(
        self,
        task_id: str,
        worker_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        wid = (worker_id or "").strip() or self.bound_worker_id(task_id)
        node = get_worker_node(wid)
        if node.is_local:
            task = stream_task_manager.stop_task(task_id, timeout=timeout)
            data = self._enrich_local(task, node.id)
            if data.get("status") in {"stopped", "failed"}:
                self._unbind(task_id)
            return data

        body = self._remote(
            node,
            "POST",
            f"/internal/v1/tasks/{task_id}/stop",
            params={"timeout": timeout},
        )
        if not isinstance(body, dict):
            body = {"task_id": task_id, "status": "stopped", "worker_id": node.id}
        body["worker_id"] = node.id
        body["worker_name"] = node.name
        self._unbind(task_id)
        return body

    def get_task(
        self,
        task_id: str,
        worker_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        wid = (worker_id or "").strip() or self.bound_worker_id(task_id)
        if not wid:
            # 未知绑定时先查本机，再扫远程
            local = stream_task_manager.get_task(task_id)
            if local:
                data = self._enrich_local(local, get_worker_node("local").id)
                self._bind(task_id, data["worker_id"])
                return data
            for node in list_worker_nodes():
                if node.is_local:
                    continue
                try:
                    body = self._remote(node, "GET", f"/internal/v1/tasks/{task_id}")
                except WorkerApiError:
                    continue
                if isinstance(body, dict) and body.get("task_id"):
                    body["worker_id"] = node.id
                    body["worker_name"] = node.name
                    self._bind(task_id, node.id)
                    return body
            return None

        node = get_worker_node(wid)
        if node.is_local:
            task = stream_task_manager.get_task(task_id)
            if not task:
                return None
            return self._enrich_local(task, node.id)

        try:
            body = self._remote(node, "GET", f"/internal/v1/tasks/{task_id}")
        except WorkerApiError as exc:
            if exc.status_code == 404:
                return None
            raise
        if not isinstance(body, dict):
            return None
        body["worker_id"] = node.id
        body["worker_name"] = node.name
        return body

    def list_tasks(self, worker_id: Optional[str] = None) -> List[Dict[str, Any]]:
        nodes = list_worker_nodes()
        if worker_id:
            nodes = [get_worker_node(worker_id)]

        items: List[Dict[str, Any]] = []
        for node in nodes:
            if node.is_local:
                for task in stream_task_manager.list_tasks():
                    data = self._enrich_local(task, node.id)
                    self._bind(data["task_id"], node.id)
                    items.append(data)
                continue
            try:
                body = self._remote(node, "GET", "/internal/v1/tasks")
            except WorkerApiError:
                logger.exception("列举 Worker 任务失败 worker=%s", node.id)
                continue
            tasks = body.get("tasks") if isinstance(body, dict) else body
            if not isinstance(tasks, list):
                continue
            for item in tasks:
                if not isinstance(item, dict):
                    continue
                item = dict(item)
                item["worker_id"] = node.id
                item["worker_name"] = node.name
                tid = item.get("task_id")
                if tid:
                    self._bind(str(tid), node.id)
                items.append(item)
        return items

    def health(self, worker_id: Optional[str] = None) -> Dict[str, Any]:
        node = get_worker_node(worker_id)
        if node.is_local:
            tasks = stream_task_manager.list_tasks()
            running = sum(
                1
                for t in tasks
                if stream_task_manager.enrich_task_dict(t).get("status")
                in {"running", "starting"}
            )
            return {
                "worker_id": node.id,
                "worker_name": node.name,
                "status": "ok",
                "local": True,
                "running_tasks": running,
                "triton_url": settings.TRITON_URL,
            }
        body = self._remote(node, "GET", "/internal/v1/health")
        if not isinstance(body, dict):
            body = {"status": "ok"}
        body["worker_id"] = node.id
        body["worker_name"] = node.name
        body["local"] = False
        return body

    def list_skills(self, worker_id: Optional[str] = None) -> List[Dict[str, Any]]:
        node = get_worker_node(worker_id)
        if node.is_local:
            return list_available_skills()
        body = self._remote(node, "GET", "/internal/v1/skills")
        if isinstance(body, dict):
            skills = body.get("skills") or body.get("items") or []
        else:
            skills = body
        return skills if isinstance(skills, list) else []

    def list_models(self, worker_id: Optional[str] = None) -> Dict[str, Any]:
        """查询指定 Worker 本机 TRITON_URL 上的模型。"""
        node = get_worker_node(worker_id)
        if node.is_local:
            from app.services.mgmt_service import list_triton_models

            data = list_triton_models()
            data["worker_id"] = node.id
            data["worker_name"] = node.name
            return data
        body = self._remote(node, "GET", "/internal/v1/models")
        if not isinstance(body, dict):
            body = {"items": [], "error": "无效响应"}
        body["worker_id"] = node.id
        body["worker_name"] = node.name
        return body

    def load_model(
        self,
        model_name: str,
        worker_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """在指定 Worker 的 Triton 上加载模型。"""
        node = get_worker_node(worker_id)
        if node.is_local:
            from app.services.mgmt_service import load_triton_model

            data = load_triton_model(model_name)
            data["worker_id"] = node.id
            data["worker_name"] = node.name
            return data
        body = self._remote(
            node,
            "POST",
            f"/internal/v1/models/{quote(model_name, safe='')}/load",
        )
        if not isinstance(body, dict):
            body = {"ok": True, "model_name": model_name}
        body["worker_id"] = node.id
        body["worker_name"] = node.name
        return body

    def unload_model(
        self,
        model_name: str,
        worker_id: Optional[str] = None,
        *,
        unload_dependents: bool = False,
    ) -> Dict[str, Any]:
        """在指定 Worker 的 Triton 上卸载模型。"""
        node = get_worker_node(worker_id)
        if node.is_local:
            from app.services.mgmt_service import unload_triton_model

            data = unload_triton_model(
                model_name, unload_dependents=unload_dependents
            )
            data["worker_id"] = node.id
            data["worker_name"] = node.name
            return data
        body = self._remote(
            node,
            "POST",
            f"/internal/v1/models/{quote(model_name, safe='')}/unload",
            params={"unload_dependents": str(unload_dependents).lower()},
        )
        if not isinstance(body, dict):
            body = {
                "ok": True,
                "model_name": model_name,
                "unload_dependents": unload_dependents,
            }
        body["worker_id"] = node.id
        body["worker_name"] = node.name
        return body

    def control_model_aggregated(
        self,
        model_name: str,
        action: str,
        *,
        worker_id: Optional[str] = None,
        unload_dependents: bool = False,
    ) -> Dict[str, Any]:
        """
        对模型执行 load/unload。

        worker_id 为空时，对所有在线且仓库中含该模型的 Worker 执行。
        """
        name = (model_name or "").strip()
        if not name:
            raise ValueError("model_name 不能为空")
        action = (action or "").strip().lower()
        if action not in {"load", "unload"}:
            raise ValueError("action 仅支持 load / unload")

        targets: List[WorkerNode]
        if worker_id:
            targets = [get_worker_node(worker_id)]
        else:
            targets = []
            for node in list_worker_nodes():
                try:
                    raw = self.list_models(node.id)
                except WorkerApiError:
                    continue
                if raw.get("error"):
                    continue
                names = {
                    str(m.get("name") or "").strip()
                    for m in (raw.get("items") or [])
                    if isinstance(m, dict)
                }
                if name in names:
                    targets.append(node)
            if not targets:
                raise ValueError(
                    f"未在任何在线 Worker 的 Triton 仓库中找到模型: {name}"
                )

        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        for node in targets:
            try:
                if action == "load":
                    results.append(self.load_model(name, node.id))
                else:
                    results.append(
                        self.unload_model(
                            name,
                            node.id,
                            unload_dependents=unload_dependents,
                        )
                    )
            except WorkerApiError as exc:
                errors.append(f"{node.id}: {exc}")
            except Exception as exc:
                errors.append(f"{node.id}: {exc}")

        # 校验操作是否真正改变了就绪状态
        for item in results:
            still_ready = bool(item.get("ready"))
            wid = item.get("worker_id") or "?"
            if action == "unload" and still_ready:
                errors.append(
                    f"{wid}: 卸载请求已发送，但模型仍处于就绪（请确认 "
                    f"--model-control-mode=explicit）"
                )
            if action == "load" and not still_ready:
                errors.append(f"{wid}: 加载请求已发送，但模型尚未就绪")

        if errors and not results:
            raise WorkerApiError("; ".join(errors), status_code=502)
        if errors and action == "unload":
            # 卸载未生效时视为失败，避免前端误显示成功
            raise WorkerApiError("; ".join(errors), status_code=502)

        return {
            "ok": not errors,
            "action": action,
            "model_name": name,
            "unload_dependents": unload_dependents if action == "unload" else False,
            "results": results,
            "errors": errors or None,
        }

    def list_models_aggregated(self) -> Dict[str, Any]:
        """汇总所有 Worker 上的 Triton 模型，按模型名合并部署节点。"""
        nodes = list_worker_nodes()
        workers_meta: List[Dict[str, Any]] = []
        by_name: Dict[str, Dict[str, Any]] = {}
        errors: List[str] = []

        for node in nodes:
            try:
                raw = self.list_models(node.id)
            except WorkerApiError as exc:
                errors.append(f"{node.id}: {exc}")
                workers_meta.append(
                    {
                        "worker_id": node.id,
                        "worker_name": node.name,
                        "server_url": None,
                        "online": False,
                        "error": str(exc),
                    }
                )
                continue

            workers_meta.append(
                {
                    "worker_id": node.id,
                    "worker_name": node.name,
                    "server_url": raw.get("server_url"),
                    "server_live": raw.get("server_live"),
                    "server_ready": raw.get("server_ready"),
                    "online": bool(raw.get("server_live")),
                    "error": raw.get("error"),
                }
            )
            if raw.get("error"):
                errors.append(f"{node.id}: {raw.get('error')}")

            for m in raw.get("items") or []:
                if not isinstance(m, dict):
                    continue
                name = str(m.get("name") or "").strip()
                if not name:
                    continue
                entry = by_name.get(name)
                ready_now = bool(m.get("ready"))
                state_now = m.get("state")
                if not entry:
                    entry = {
                        "name": name,
                        "version": m.get("version"),
                        "state": state_now,
                        "ready": ready_now,
                        "workers": [],
                    }
                    by_name[name] = entry
                else:
                    # 任一 Worker/版本就绪则整体就绪；全部离线时同步为离线
                    if ready_now:
                        entry["ready"] = True
                    elif not entry.get("workers"):
                        entry["ready"] = False
                    if state_now:
                        # 优先保留 READY；否则用最新状态覆盖
                        if ready_now or str(entry.get("state") or "").upper() != "READY":
                            entry["state"] = state_now
                    if m.get("version") and (
                        not entry.get("version") or ready_now
                    ):
                        entry["version"] = m.get("version")

                wtag = {
                    "worker_id": node.id,
                    "worker_name": node.name,
                    "triton_url": raw.get("server_url"),
                    "ready": ready_now,
                    "version": m.get("version"),
                    "state": state_now,
                }
                existing_w = next(
                    (
                        x
                        for x in entry["workers"]
                        if x.get("worker_id") == node.id
                    ),
                    None,
                )
                if existing_w is None:
                    entry["workers"].append(wtag)
                else:
                    # 同 Worker 多版本：任一版本就绪则该节点就绪
                    existing_w["ready"] = bool(existing_w.get("ready")) or ready_now
                    if state_now:
                        existing_w["state"] = state_now
                    if m.get("version"):
                        existing_w["version"] = m.get("version")

                # 按各 Worker 汇总重算整体 ready
                entry["ready"] = any(
                    bool(w.get("ready")) for w in entry["workers"]
                )
                if not entry["ready"] and entry.get("state"):
                    if str(entry["state"]).upper() == "READY":
                        entry["state"] = "UNAVAILABLE"

        items = sorted(by_name.values(), key=lambda x: x["name"])
        any_live = any(bool(w.get("online")) for w in workers_meta)
        server_urls = [
            str(w.get("server_url"))
            for w in workers_meta
            if w.get("server_url")
        ]
        return {
            "server_url": "; ".join(dict.fromkeys(server_urls)) if server_urls else settings.TRITON_URL,
            "server_live": any_live,
            "server_ready": any(
                bool(w.get("server_ready")) for w in workers_meta
            ),
            "server_name": None,
            "server_version": None,
            "items": items,
            "workers": workers_meta,
            "error": "; ".join(errors) if errors and not items else (
                "; ".join(errors) if errors and not any_live else None
            ),
        }

    def stop_all_local(self) -> None:
        if has_local_worker():
            stream_task_manager.stop_all()


runtime_gateway = RuntimeGateway()
