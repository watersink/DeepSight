"""
DeepSight Worker 入口：编解码 + 调用 Triton（不托管前端/管理台）。

启动:
  python -m app.worker_main

或:
  uvicorn app.worker_main:app --host 0.0.0.0 --port 8100

配置:
  与 API 相同，只加载项目根目录 .env（各机器自行准备/重命名配置文件）
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from app.api.routes.worker_internal import router as worker_router
from app.core.config import settings
from app.services.alert_hub import alert_hub
from app.services.minio_client import MinioClientError, minio_storage
from app.services.stream_task_manager import stream_task_manager

log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "Worker 启动 host=%s port=%s triton=%s",
        settings.WORKER_HOST,
        settings.WORKER_PORT,
        settings.TRITON_URL,
    )
    try:
        from app.db import init_db

        init_db()
        logger.info("Worker MySQL 已就绪（告警落库）")
    except Exception:
        logger.exception("Worker MySQL 初始化失败（告警落库可能不可用）")

    alert_hub.start()
    try:
        bucket = minio_storage.ensure_bucket()
        logger.info("Worker MinIO 已就绪: %s", bucket)
    except MinioClientError:
        logger.exception("Worker MinIO 初始化失败（截图上传可能不可用）")

    yield

    logger.info("Worker 关闭，停止本机推流任务...")
    stream_task_manager.stop_all()
    alert_hub.stop()


app = FastAPI(
    title=f"{settings.PROJECT_NAME} Worker",
    description="远程 Worker：视频解码 / AI 技能 / FFmpeg 编码推流",
    version=settings.PROJECT_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.include_router(worker_router)


def main() -> None:
    uvicorn.run(
        "app.worker_main:app",
        host=settings.WORKER_HOST,
        port=int(settings.WORKER_PORT),
        reload=False,
    )


if __name__ == "__main__":
    main()
