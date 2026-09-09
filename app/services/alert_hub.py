"""人数告警跨进程汇聚 + SSE 订阅扇出"""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import queue
import threading
from typing import Any, AsyncIterator, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertHub:
    """主进程单例：从子进程队列取告警，扇出给 SSE 订阅者。"""

    _instance: Optional["AlertHub"] = None
    _init_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._mp_queue: multiprocessing.Queue = multiprocessing.Queue(maxsize=256)
        self._subscribers: List[asyncio.Queue] = []
        self._sub_lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._bridge_thread: Optional[threading.Thread] = None
        self._running = False

    @property
    def mp_queue(self) -> multiprocessing.Queue:
        return self._mp_queue

    def start(self) -> None:
        if self._running:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._running = True
        self._bridge_thread = threading.Thread(
            target=self._bridge_loop,
            name="alert-hub-bridge",
            daemon=True,
        )
        self._bridge_thread.start()
        logger.info("告警 SSE 汇聚已启动")

    def stop(self) -> None:
        self._running = False
        if self._bridge_thread and self._bridge_thread.is_alive():
            self._bridge_thread.join(timeout=2.0)
        self._bridge_thread = None
        logger.info("告警 SSE 汇聚已停止")

    def _bridge_loop(self) -> None:
        while self._running:
            try:
                event = self._mp_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception:
                logger.exception("读取告警队列失败")
                continue
            loop = self._loop
            if loop is None or not loop.is_running():
                continue
            try:
                asyncio.run_coroutine_threadsafe(self._fanout(event), loop)
            except Exception:
                logger.exception("投递告警到事件循环失败")

    async def _fanout(self, event: Dict[str, Any]) -> None:
        async with self._sub_lock:
            subscribers = list(self._subscribers)
        for q in subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(event)
                except asyncio.QueueFull:
                    pass

    async def subscribe(
        self,
        scene_id: Optional[str] = None,
        maxsize: int = 64,
    ) -> AsyncIterator[Dict[str, Any]]:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        async with self._sub_lock:
            self._subscribers.append(q)
        try:
            while True:
                event = await q.get()
                if scene_id and event.get("scene_id") != scene_id:
                    continue
                yield event
        finally:
            async with self._sub_lock:
                if q in self._subscribers:
                    self._subscribers.remove(q)


alert_hub = AlertHub()
