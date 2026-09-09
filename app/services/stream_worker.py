"""视频流处理子进程入口（供 multiprocessing 调用）

支持：
- 摄像头级 ingest + 进程内多 skill fan-out（推荐）
- 兼容旧的单 skill 启动参数（自动包成单分支）
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

CODE_ROOT = Path(__file__).resolve().parent.parent.parent


def _ensure_code_root_on_path():
    root = str(CODE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def run_camera_ingest_worker(
    ingest_meta: Dict[str, Any],
    initial_branches: List[Dict[str, Any]],
    stop_event,
    control_queue=None,
    alert_queue=None,
) -> None:
    """摄像头级解码 + 多 skill fan-out 子进程。"""
    _ensure_code_root_on_path()

    from app.core.config import settings
    from app.services.stream_alert import configure_alert_queue, default_alert_handler
    from app.services.video.camera_ingest import CameraFanoutPipeline

    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    if alert_queue is not None:
        configure_alert_queue(alert_queue)

    source_url = ingest_meta.get("source_url") or ingest_meta.get("in_url")
    if not source_url:
        raise ValueError("ingest 缺少 source_url/in_url")

    pipeline = CameraFanoutPipeline(
        source_url=source_url,
        fps=float(
            ingest_meta.get("fps")
            or ingest_meta.get("out_fps")
            or settings.DEFAULT_OUT_FPS
        ),
        reconnect_delay=float(
            ingest_meta.get("reconnect_delay") or settings.STREAM_RECONNECT_DELAY
        ),
        queue_size=int(
            ingest_meta.get("queue_size") or settings.STREAM_QUEUE_SIZE
        ),
        on_alert=default_alert_handler,
        stop_event=stop_event,
        control_queue=control_queue,
    )

    logger.info(
        "子进程启动摄像头 ingest source=%s branches=%d pid=%s",
        source_url,
        len(initial_branches or []),
        os.getpid(),
    )
    try:
        pipeline.run(initial_branches=initial_branches or [])
    except Exception:
        logger.exception("摄像头 ingest 子进程异常退出 source=%s", source_url)
        raise
    finally:
        logger.info("摄像头 ingest 子进程已退出 source=%s", source_url)


def run_stream_worker(
    config_dict: Dict[str, Any],
    stop_event,
    alert_queue=None,
    control_queue=None,
    branch_id: Optional[str] = None,
) -> None:
    """兼容入口：将单配置包装为摄像头 ingest 单分支。"""
    bid = branch_id or "branch-0"
    ingest_meta = {
        "source_url": config_dict.get("in_url"),
        "fps": config_dict.get("out_fps"),
        "reconnect_delay": config_dict.get("reconnect_delay"),
        "queue_size": config_dict.get("queue_size"),
    }
    run_camera_ingest_worker(
        ingest_meta=ingest_meta,
        initial_branches=[{"branch_id": bid, "config": config_dict}],
        stop_event=stop_event,
        control_queue=control_queue,
        alert_queue=alert_queue,
    )
