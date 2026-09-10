"""Worker 节点配置解析与查询。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.core.config import settings


@dataclass(frozen=True)
class WorkerNode:
    id: str
    name: str
    url: str

    @property
    def is_local(self) -> bool:
        u = (self.url or "").strip().lower()
        return u in {"", "local", "local://", "inplace"}


def _parse_nodes(raw: str) -> List[WorkerNode]:
    text = (raw or "").strip()
    if not text:
        return [WorkerNode(id="local", name="本机", url="local")]

    parts: List[str] = []
    for chunk in text.replace("\r", "\n").split("\n"):
        chunk = chunk.strip()
        if not chunk or chunk.startswith("#"):
            continue
        parts.extend(p.strip() for p in chunk.split(";") if p.strip())

    nodes: List[WorkerNode] = []
    seen = set()
    for part in parts:
        bits = [b.strip() for b in part.split("|")]
        if len(bits) == 1:
            wid, name, url = bits[0], bits[0], bits[0]
        elif len(bits) == 2:
            wid, name, url = bits[0], bits[0], bits[1]
        else:
            wid, name, url = bits[0], bits[1], bits[2]
        if not wid:
            continue
        if wid in seen:
            continue
        seen.add(wid)
        if url.lower() in {"local", "inplace"}:
            url = "local"
        nodes.append(WorkerNode(id=wid, name=name or wid, url=url))

    if not nodes:
        return [WorkerNode(id="local", name="本机", url="local")]
    return nodes


def list_worker_nodes() -> List[WorkerNode]:
    return _parse_nodes(settings.WORKER_NODES)


def get_worker_node(worker_id: Optional[str] = None) -> WorkerNode:
    nodes = list_worker_nodes()
    by_id: Dict[str, WorkerNode] = {n.id: n for n in nodes}
    wid = (worker_id or "").strip() or (settings.DEFAULT_WORKER_ID or "").strip()
    if wid and wid in by_id:
        return by_id[wid]
    if settings.DEFAULT_WORKER_ID in by_id:
        return by_id[settings.DEFAULT_WORKER_ID]
    return nodes[0]


def has_local_worker() -> bool:
    return any(n.is_local for n in list_worker_nodes())
