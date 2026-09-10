"""
视频检测推流 FastAPI 服务入口

启动:
  cd code
  python -m app.main

或:
  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from app.api.routes.auth import router as auth_router
from app.api.routes.clip import router as clip_router
from app.api.routes.mgmt import router as mgmt_router
from app.api.routes.open_api import router as open_api_router
from app.api.routes.platform_settings import router as platform_settings_router
from app.api.routes.skills import router as skills_router
from app.api.routes.streams import router as streams_router
from app.api.routes.workers import router as workers_router
from app.api.schemas import HealthResponse
from app.core.config import settings
from app.services.alert_hub import alert_hub
from app.services.minio_client import MinioClientError, minio_storage
from app.services.redis_cache import get_redis
from app.services.runtime_gateway import runtime_gateway
from app.services.worker_nodes import list_worker_nodes

log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

API = settings.API_V1_STR

OPENAPI_TAGS = [
    {
        "name": "健康检查",
        "description": "服务存活探测与运行中推流任务统计。",
    },
    {
        "name": "技能",
        "description": "查询当前已注册、可用于推流的 AI 检测技能。",
    },
    {
        "name": "AI视频识别任务",
        "description": (
            "启动 / 停止 AI 视频识别推流任务，查询任务状态，"
            f"以及 SSE 订阅人数告警。推流目标为 ZLMediaKit（`{settings.ZLM_HOST}`）。\n\n"
            "启动接口可选参数：`enter_count`（过线人数初始值，如 `0`）、"
            "`gate_direction`（`IN`/`OUT` 闸机方向）、"
            "`count_line`（过线计数线像素坐标）；仅部分技能需要，其它技能可省略。"
        ),
    },
    {
        "name": "事件视频截取",
        "description": (
            "基于 ZLMediaKit `startRecordTask` 截取回溯 + 前向事件视频，"
            "并生成中间帧截图（Base64 返回）。"
        ),
    },
    {
        "name": "Worker节点",
        "description": "列出算力 Worker、探测健康状态，并查询各节点可用技能。",
    },
    {
        "name": "管理台",
        "description": "摄像头配置、算法配置、任务配置与报警管理（MySQL 持久化）。",
    },
    {
        "name": "开放接口",
        "description": (
            "第三方接入：使用 X-API-Key 拉取告警详情与媒体；"
            "实时通知通过管理台配置的 Webhook 推送（HMAC 签名）。"
        ),
    },
    {
        "name": "基础配置",
        "description": "平台名称、口号与 Logo（公开读取；管理员可改）。",
    },
]

SWAGGER_DESCRIPTION = (
    f"""{settings.PROJECT_DESCRIPTION}

## 接口一览

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `GET` | `{API}/skills` | 列出可用技能 |
| `POST` | `{API}/streams/start` | 启动 AI 视频识别任务 |
| `POST` | `{API}/streams/stop/{{task_id}}` | 停止 AI 视频识别任务 |
| `GET` | `{API}/streams` | 查询所有 AI 视频识别任务 |
| `GET` | `{API}/streams/{{task_id}}` | 查询单个 AI 视频识别任务 |
| `GET` | `{API}/streams/alerts/sse` | SSE 实时订阅人数告警 |
| `POST` | `{API}/clip/capture` | 截取事件视频并返回中间帧截图 |

## 快速开始

1. `GET {API}/skills` — 查看可用技能
2. `POST {API}/streams/start` — 启动 AI 视频识别任务（子进程）
3. `GET {API}/streams/alerts/sse` — 前端 EventSource 订阅人数告警
4. `GET {API}/streams` — 查看任务列表
5. `POST {API}/clip/capture` — 截取事件视频（需流在线）
6. `POST {API}/streams/stop/{{task_id}}` — 停止任务

## 启动可选参数（按技能）

| 参数 | 说明 | 适用技能 |
|------|------|----------|
| `enter_count` | 过线进入人数初始值（如 `0`），默认可不传 | `person_count_detector26` |
| `gate_direction` | 闸机方向：`IN`（入闸）/ `OUT`（出闸），影响画面参考点标注 | `person_count_detector26` |
| `count_line` | 过线计数线（像素）：`lines` 中每段 `line` 与 `enter_point` 一一对应 | `person_count_detector26` |
| `bypass_line` | 绕行线（像素）：`lines` 中每段 `line` 与 `bypass_point` 一一对应 | `person_count_detector26` |

`count_line` 示例（线段与点一一对应）：

"""
    + """
```json
{
  "lines": [
    {
      "line": [{"x": 432, "y": 328}, {"x": 723, "y": 229}],
      "enter_point": {"x": 513, "y": 250}
    },
    {
      "line": [{"x": 100, "y": 200}, {"x": 300, "y": 150}],
      "enter_point": {"x": 180, "y": 170}
    }
  ]
}
```

`bypass_line` 示例（线段与点一一对应）：

```json
{
  "lines": [
    {
      "line": [{"x": 154, "y": 421}, {"x": 434, "y": 337}],
      "bypass_point": {"x": 280, "y": 348}
    }
  ]
}
```

"""
    + f"""
## 推流地址约定

默认推流地址：`{settings.build_push_url()}`

即 `{{协议}}://{settings.ZLM_HOST}:{{端口}}/{{scene_id}}/{{skill_name}}`

## 事件截取约定

- 请求需 `id`、`video_url`、`back_ms`、`forward_ms`
- 服务端从 `video_url`（如 `rtmp://host/app/stream`）解析 `app` / `stream`
- 底层调用 ZLMediaKit：`GET /index/api/startRecordTask`
- 上传 MinIO 后回写 `COAL_ALARM_UPDATE_URL`（`id` / `violationImage` / `violationVideo`）
- 需配置环境变量 `ZLM_SECRET`（ZLMediaKit `config.ini` 中 `[api] secret`）

## 文档入口

- Swagger UI: `/docs`
- ReDoc: `/redoc`
- OpenAPI JSON: `/openapi.json`
"""
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("%s 服务启动", settings.PROJECT_NAME)
    logger.info("Swagger UI: http://0.0.0.0:%s/docs", settings.REST_PORT)
    try:
        from app.db import init_db

        init_db()
        logger.info(
            "MySQL 表已就绪 %s:%s/%s",
            settings.MYSQL_HOST,
            settings.MYSQL_PORT,
            settings.MYSQL_DB,
        )
    except Exception:
        logger.exception(
            "MySQL 初始化失败 %s:%s/%s（管理台 API 将不可用）",
            settings.MYSQL_HOST,
            settings.MYSQL_PORT,
            settings.MYSQL_DB,
        )
    get_redis()
    alert_hub.start()
    logger.info(
        "已加载 Worker 节点: %s",
        ", ".join(f"{n.id}({n.url})" for n in list_worker_nodes()),
    )
    try:
        from app.services.task_scheduler import start_scheduler

        start_scheduler()
    except Exception:
        logger.exception("APScheduler 启动失败（任务时间窗自动启停将不可用）")
    try:
        from app.services.webhook_dispatch import start_webhook_mq_worker
        from app.services.internal_mq import ensure_topology_once

        if settings.RABBITMQ_WEBHOOK_ENABLED:
            ensure_topology_once()
            start_webhook_mq_worker()
        else:
            logger.info("已关闭内部 MQ Webhook 投递（RABBITMQ_WEBHOOK_ENABLED=false）")
    except Exception:
        logger.exception(
            "Webhook MQ Worker 启动失败 host=%s:%s（将在入队失败时回退直推）",
            settings.RABBITMQ_HOST,
            settings.RABBITMQ_PORT,
        )
    try:
        bucket = minio_storage.ensure_bucket()
        logger.info("MinIO 存储桶已就绪: %s @ %s", bucket, settings.MINIO_ENDPOINT)
    except MinioClientError:
        logger.exception(
            "MinIO 存储桶初始化失败 bucket=%s endpoint=%s（截取上传将不可用）",
            settings.MINIO_BUCKET_NAME,
            settings.MINIO_ENDPOINT,
        )
    yield
    logger.info("服务关闭，正在停止本机推流任务...")
    try:
        from app.services.webhook_dispatch import stop_webhook_mq_worker

        stop_webhook_mq_worker()
    except Exception:
        logger.exception("Webhook MQ Worker 停止异常")
    try:
        from app.services.task_scheduler import stop_scheduler

        stop_scheduler()
    except Exception:
        logger.exception("APScheduler 停止异常")
    runtime_gateway.stop_all_local()
    alert_hub.stop()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=SWAGGER_DESCRIPTION,
    version=settings.PROJECT_VERSION,
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "docExpansion": "list",
        "defaultModelsExpandDepth": 2,
        "tryItOutEnabled": True,
        "persistAuthorization": True,
        "filter": True,
        "displayRequestDuration": True,
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(streams_router, prefix=settings.API_V1_STR)
app.include_router(skills_router, prefix=settings.API_V1_STR)
app.include_router(workers_router, prefix=settings.API_V1_STR)
app.include_router(clip_router, prefix=settings.API_V1_STR)
app.include_router(mgmt_router, prefix=settings.API_V1_STR)
app.include_router(platform_settings_router, prefix=settings.API_V1_STR)
app.include_router(open_api_router)

_DIST = settings.frontend_dist_path
_INDEX = _DIST / "index.html"
_HAS_FRONTEND = _INDEX.is_file()


@app.get("/", include_in_schema=False)
def root():
    if _HAS_FRONTEND:
        return FileResponse(_INDEX)
    return RedirectResponse(url="/docs")


@app.get(
    "/health",
    tags=["健康检查"],
    summary="健康检查",
    response_model=HealthResponse,
    description="检查服务是否存活，并返回当前运行中的推流任务数量。",
    responses={200: {"description": "服务正常"}},
)
def health_check() -> HealthResponse:
    running = sum(
        1
        for t in runtime_gateway.list_tasks()
        if t.get("status") in {"running", "starting"}
    )
    return HealthResponse(
        status="ok",
        service=settings.PROJECT_NAME,
        running_tasks=running,
    )


if _HAS_FRONTEND:
    assets_dir = _DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="frontend-assets")

    skills_dir = _DIST / "skills"
    if skills_dir.is_dir():
        app.mount("/skills", StaticFiles(directory=str(skills_dir)), name="skill-covers")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        from fastapi import HTTPException

        if (
            full_path == "health"
            or full_path.startswith("api/")
            or full_path.startswith("assets/")
            or full_path.startswith("skills/")
            or full_path in {"docs", "redoc", "openapi.json"}
            or full_path.startswith("docs/")
        ):
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = _DIST / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_INDEX)


if __name__ == "__main__":
    import multiprocessing
    import uvicorn

    multiprocessing.freeze_support()
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.REST_PORT,
        reload=settings.DEBUG,
    )
