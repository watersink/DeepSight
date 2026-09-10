"""本项目配置"""
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import quote, urlparse

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

_CODE_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    # FastAPI 服务
    API_V1_STR: str = Field(default="/api/v1", description="API 路由前缀")
    PROJECT_NAME: str = Field(default="Video Detection Stream", description="项目名称")
    PROJECT_DESCRIPTION: str = Field(
        default="视频流检测推流服务",
        description="项目描述",
    )
    PROJECT_VERSION: str = Field(default="1.0.0", description="项目版本")
    REST_PORT: int = Field(default=8000, description="REST API 端口")
    REST_HOST: str = Field(default="127.0.0.1", description="REST API 主机（客户端默认连接地址）")

    DEBUG: bool = Field(default=False, description="调试模式")
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")

    # Triton 推理
    TRITON_URL: str = Field(default="10.1.3.21:8201", description="Triton gRPC 地址 host:port")

    # 远程 Worker（编解码 + 调 Triton）
    # 格式: id|显示名|url，多项用分号或换行分隔；url=local 表示本机进程内执行
    # 示例: local|本机|local;gpu1|GPU-1|http://10.1.3.21:8100;gpu2|GPU-2|http://10.1.3.22:8100
    WORKER_NODES: str = Field(
        default="local|本机|local",
        description="Worker 节点列表（id|name|url，分号分隔）",
    )
    WORKER_TOKEN: str = Field(
        default="",
        description="API 调用 Worker 内部接口的共享密钥（可空=不校验）",
    )
    WORKER_HOST: str = Field(default="0.0.0.0", description="worker_main 监听地址")
    WORKER_PORT: int = Field(default=8100, description="worker_main 监听端口")
    DEFAULT_WORKER_ID: str = Field(
        default="local",
        description="任务未指定 worker_id 时的默认节点",
    )
    WORKER_HTTP_TIMEOUT: float = Field(
        default=30.0,
        ge=1.0,
        description="调用远程 Worker HTTP 超时（秒）",
    )

    # ZLMediaKit 流媒体引擎
    ZLM_HOST: str = Field(default="10.1.3.21", description="ZLMediaKit 主机地址")
    ZLM_RTMP_PORT: int = Field(default=1935, description="ZLMediaKit RTMP 端口")
    ZLM_RTSP_PORT: int = Field(default=8554, description="ZLMediaKit RTSP 端口")
    ZLM_HTTP_PORT: int = Field(default=8080, description="ZLMediaKit HTTP 端口")
    ZLM_SECRET: str = Field(
        default="b1CXeSHhB1AcYV5Hmf9e9h7nyHXsI9Tm",
        description="ZLMediaKit HTTP API 密钥（config.ini [api] secret）",
    )
    ZLM_DEFAULT_VHOST: str = Field(
        default="__defaultVhost__",
        description="ZLMediaKit 默认虚拟主机",
    )
    ZLM_RECORD_WAIT_BUFFER_MS: int = Field(
        default=2000,
        ge=0,
        description="事件录制 forward_ms 结束后的额外等待时间（毫秒），确保文件写入完成",
    )

    # 视频AI识别默认参数
    DEFAULT_OUTPUT_FORMAT: Literal["rtmp", "rtsp"] = Field(
        default="rtmp",
        description="默认推流协议",
    )
    DEFAULT_OUT_FPS: int = Field(default=15, description="默认输出帧率")
    DEFAULT_ALARM_INTERVAL: int = Field(default=1, description="默认告警间隔（秒）")
    DEFAULT_SCENE_ID: str = Field(default="scene_001", description="默认场景 ID")
    DEFAULT_SKILL_NAME: str = Field(default="person_count_detector26", description="默认技能名称")
    DEFAULT_TEST_VIDEO: str = Field(
        default="test_images_videos/person_gouzi.mp4",
        description="默认测试视频（相对 code 目录）",
    )
    STREAM_QUEUE_SIZE: int = Field(default=4, description="默认帧缓冲队列大小")
    STREAM_USE_HARDWARE_ENCODING: bool = Field(default=True, description="默认是否硬件编码")
    STREAM_USE_HARDWARE_DECODING: bool = Field(
        default=True,
        description="默认是否硬件解码（优先 CUDA/NVDEC，其次 QSV，失败回退 OpenCV）",
    )
    STREAM_BITRATE: str = Field(default="2M", description="默认推流码率")
    STREAM_BUFFER_SIZE: str = Field(default="2M", description="默认推流缓冲区大小")
    STREAM_RECONNECT_DELAY: float = Field(default=2.0, description="视频源断线重连间隔（秒）")

    # MinIO 对象存储
    MINIO_ENDPOINT: str = Field(
        default="10.1.3.21:9000",
        description="MinIO API 地址（host:port，不含协议）",
    )
    MINIO_CONSOLE_URL: str = Field(
        default="http://10.1.3.21:9001",
        description="MinIO Console 访问地址",
    )
    MINIO_ROOT_USER: str = Field(default="minioadmin", description="MinIO Access Key")
    MINIO_ROOT_PASSWORD: str = Field(default="minioadmin", description="MinIO Secret Key")
    MINIO_BUCKET_NAME: str = Field(default="yingjiting", description="默认存储桶名称")
    MINIO_SECURE: bool = Field(default=False, description="MinIO 是否使用 HTTPS")
    MINIO_PUBLIC_READ: bool = Field(
        default=True,
        description="是否将存储桶设为匿名可读（返回的直链才能被浏览器直接打开）",
    )
    MINIO_PUBLIC_BASE_URL: Optional[str] = Field(
        default=None,
        description=(
            "对象公网/内网直链前缀；省略时按 "
            "`http(s)://{MINIO_ENDPOINT}/{bucket}/{object}` 拼装"
        ),
    )

    # 告警回写
    COAL_ALARM_UPDATE_URL: str = Field(
        default="http://192.168.26.224:9710/coal/alarm/update",
        description="事件截取完成后回写告警图片/视频的接口地址",
    )
    COAL_ALARM_UPDATE_TIMEOUT: float = Field(
        default=10.0,
        ge=1.0,
        description="告警回写接口超时（秒）",
    )

    # 人数识别结果上传（countingRecog）
    COUNTING_RECOG_UPLOAD_URL: str = Field(
        default="http://192.168.26.224:9810/mine/countingRecog/upload",
        description="人数/闸机识别结果上传接口地址",
    )
    COUNTING_RECOG_UPLOAD_TIMEOUT: float = Field(
        default=10.0,
        ge=1.0,
        description="人数识别结果上传接口超时（秒）",
    )
    COUNTING_RECOG_MINE_CODE: str = Field(
        default="",
        description="默认煤矿编码（12位数字）；可被推流请求 mine_code 覆盖",
    )
    COUNTING_RECOG_CAMERA_CODE: str = Field(
        default="",
        description="默认摄像仪编码（20位）；可被推流请求 camera_code 覆盖",
    )

    # 本项目自用 RabbitMQ（Webhook 投递削峰）
    RABBITMQ_HOST: str = Field(default="10.1.3.21", description="本项目 RabbitMQ 主机")
    RABBITMQ_PORT: int = Field(default=5672, description="本项目 RabbitMQ AMQP 端口")
    RABBITMQ_USERNAME: str = Field(default="admin", description="本项目 RabbitMQ 用户名")
    RABBITMQ_PASSWORD: str = Field(default="admin123", description="本项目 RabbitMQ 密码")
    RABBITMQ_VHOST: str = Field(default="/", description="本项目 RabbitMQ 虚拟主机")
    RABBITMQ_TIMEOUT: float = Field(
        default=5.0,
        ge=1.0,
        description="本项目 RabbitMQ 连接超时（秒）",
    )
    RABBITMQ_WEBHOOK_ENABLED: bool = Field(
        default=True,
        description="是否经本项目 MQ 异步投递 Webhook（关闭则回退为进程内线程直推）",
    )
    RABBITMQ_WEBHOOK_EXCHANGE: str = Field(
        default="alerts.webhook",
        description="Webhook 投递交换机",
    )
    RABBITMQ_WEBHOOK_QUEUE: str = Field(
        default="alerts.webhook.deliver",
        description="Webhook 投递队列",
    )
    RABBITMQ_WEBHOOK_ROUTING_KEY: str = Field(
        default="webhook.deliver",
        description="Webhook 投递路由键",
    )
    RABBITMQ_WEBHOOK_QUEUE_MAX_LENGTH: int = Field(
        default=10000,
        ge=100,
        description="投递队列最大长度（超出丢弃最旧，削峰）",
    )
    RABBITMQ_WEBHOOK_MESSAGE_TTL_MS: int = Field(
        default=86_400_000,
        ge=1000,
        description="投递消息 TTL（毫秒），默认 24h",
    )
    RABBITMQ_WEBHOOK_PREFETCH: int = Field(
        default=4,
        ge=1,
        le=64,
        description="Webhook Worker 预取并发（未 ack 上限）",
    )
    RABBITMQ_WEBHOOK_MAX_RETRY: int = Field(
        default=3,
        ge=0,
        le=20,
        description="Webhook 投递失败最大重试次数",
    )
    WEBHOOK_TIMEOUT: float = Field(
        default=5.0,
        ge=1.0,
        description="第三方 Webhook 推送超时（秒）",
    )



    # 外部第三方平台 RabbitMQ（过线人数 enter_count / 报警数据推送）
    EXTERNAL_RABBITMQ_HOST: str = Field(
        default="172.16.201.80",
        description="外部平台 RabbitMQ 主机",
    )
    EXTERNAL_RABBITMQ_PORT: int = Field(
        default=5672,
        description="外部平台 RabbitMQ AMQP 端口",
    )
    EXTERNAL_RABBITMQ_USERNAME: str = Field(
        default="ruoyi",
        description="外部平台 RabbitMQ 用户名",
    )
    EXTERNAL_RABBITMQ_PASSWORD: str = Field(
        default="ruoyi123",
        description="外部平台 RabbitMQ 密码",
    )
    EXTERNAL_RABBITMQ_VHOST: str = Field(
        default="/",
        description="外部平台 RabbitMQ 虚拟主机",
    )
    EXTERNAL_RABBITMQ_EXCHANGE: str = Field(
        default="mine.data.push",
        description="外部平台井下人数推送交换机",
    )
    EXTERNAL_RABBITMQ_ROUTING_KEY: str = Field(
        default="mine.undergroundCount",
        description="外部平台井下人数推送路由键",
    )
    EXTERNAL_RABBITMQ_TIMEOUT: float = Field(
        default=5.0,
        ge=1.0,
        description="外部平台 RabbitMQ 连接/发布超时（秒）",
    )

    # MySQL（管理台：摄像头 / 算法 / 任务 / 报警）
    MYSQL_HOST: str = Field(default="10.1.3.21", description="MySQL 主机")
    MYSQL_PORT: int = Field(default=3306, description="MySQL 端口")
    MYSQL_USER: str = Field(default="root", description="MySQL 用户名")
    MYSQL_PASSWORD: str = Field(default="root", description="MySQL 密码")
    MYSQL_DB: str = Field(default="yingjiting", description="MySQL 数据库名")

    # Redis（管理台缓存）
    REDIS_HOST: str = Field(default="10.1.3.21", description="Redis 主机")
    REDIS_PORT: int = Field(default=6379, description="Redis 端口")
    REDIS_DB: int = Field(default=1, description="Redis 数据库编号")
    REDIS_PASSWORD: str = Field(default="", description="Redis 密码（可空）")
    REDIS_ENABLED: bool = Field(default=True, description="是否启用 Redis 缓存")

    # JWT 用户鉴权（管理台）
    JWT_SECRET_KEY: str = Field(
        default="yingjiting-change-me-in-production-please",
        description="JWT 签名密钥，生产环境务必改成随机长字符串",
    )
    JWT_ALGORITHM: str = Field(default="HS256", description="JWT 算法")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=720,
        ge=5,
        description="Access Token 有效期（分钟），默认 12 小时",
    )
    AUTH_DEFAULT_ADMIN_USERNAME: str = Field(
        default="admin", description="首次启动自动创建的管理员用户名"
    )
    AUTH_DEFAULT_ADMIN_PASSWORD: str = Field(
        default="Admin@123",
        description="首次启动自动创建的管理员密码（登录后请尽快修改）",
    )

    # 前端静态资源（Vite build 产物）
    FRONTEND_DIST_DIR: str = Field(
        default="frontend/dist",
        description="相对 code 根目录的前端 dist 路径；存在则由 FastAPI 托管",
    )

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    @property
    def api_base_url(self) -> str:
        return f"http://{self.REST_HOST}:{self.REST_PORT}{self.API_V1_STR}"

    @property
    def mysql_database_url(self) -> str:
        user = quote(self.MYSQL_USER, safe="")
        password = quote(self.MYSQL_PASSWORD, safe="")
        return (
            f"mysql+pymysql://{user}:{password}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
            f"?charset=utf8mb4"
        )

    @property
    def mysql_server_url(self) -> str:
        """不带库名的连接，用于建库。"""
        user = quote(self.MYSQL_USER, safe="")
        password = quote(self.MYSQL_PASSWORD, safe="")
        return (
            f"mysql+pymysql://{user}:{password}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/"
            f"?charset=utf8mb4"
        )

    @property
    def frontend_dist_path(self) -> Path:
        raw = Path(self.FRONTEND_DIST_DIR)
        if raw.is_absolute():
            return raw
        return _CODE_ROOT / raw

    @property
    def zlm_http_base_url(self) -> str:
        return f"http://{self.ZLM_HOST}:{self.ZLM_HTTP_PORT}"

    @property
    def minio_public_base_url(self) -> str:
        if self.MINIO_PUBLIC_BASE_URL:
            return self.MINIO_PUBLIC_BASE_URL.rstrip("/")
        scheme = "https" if self.MINIO_SECURE else "http"
        return f"{scheme}://{self.MINIO_ENDPOINT}"

    def build_minio_object_url(self, object_name: str, bucket: Optional[str] = None) -> str:
        """拼装 MinIO 对象 HTTP 访问地址（path-style）。"""
        bucket_name = bucket or self.MINIO_BUCKET_NAME
        key = quote(object_name.lstrip("/"), safe="/")
        return f"{self.minio_public_base_url}/{bucket_name}/{key}"

    def build_record_http_url(self, app: str, stream: str, rel_path: str) -> str:
        """ZLMediaKit 录像文件 HTTP 访问地址（www/record/{app}/{stream}/...）。"""
        rel = rel_path.lstrip("/").replace("\\", "/")
        return f"{self.zlm_http_base_url}/record/{app}/{stream}/{rel}"

    def build_flv_play_url(self, app: str, stream: str) -> str:
        """out_url 对应的 HTTP-FLV 播放地址，如 /{app}/{stream}.live.flv。"""
        return f"{self.zlm_http_base_url}/{app}/{stream}.live.flv"

    def build_flv_play_url_from_out_url(self, out_url: Optional[str]) -> Optional[str]:
        """
        从推流地址解析 app/stream 并拼装 FLV。

        例：rtmp://host:1935/scene_001/person_count_detector26
         → http://host:8080/scene_001/person_count_detector26.live.flv
        """
        if not out_url:
            return None

        parts = [p for p in urlparse(out_url.strip()).path.split("/") if p]
        if len(parts) < 2:
            return None
        return self.build_flv_play_url(parts[0], parts[1])

    def _push_scheme_host(self, output_format: str) -> str:
        if output_format == "rtsp":
            return f"rtsp://{self.ZLM_HOST}:{self.ZLM_RTSP_PORT}"
        return f"rtmp://{self.ZLM_HOST}:{self.ZLM_RTMP_PORT}"

    def build_zlm_pull_url(
        self,
        app: str,
        stream: str,
        *,
        output_format: str = "rtmp",
    ) -> str:
        """拼装 ZLM 上已存在流的拉流地址（供 AI 取流 / 在线检测）。"""
        app_s = (app or "").strip().strip("/")
        stream_s = (stream or "").strip().strip("/")
        if not app_s or not stream_s:
            raise ValueError("zlm_app / zlm_stream 不能为空")
        return f"{self._push_scheme_host(output_format)}/{app_s}/{stream_s}"

    def build_push_url(
        self,
        output_format: Optional[str] = None,
        scene_id: Optional[str] = None,
        skill_name: Optional[str] = None,
    ) -> str:
        """
        拼装推流地址：{scheme}://{host}:{port}/{scene_id}/{skill_name}
        """
        fmt = output_format or self.DEFAULT_OUTPUT_FORMAT
        scene = scene_id or self.DEFAULT_SCENE_ID
        skill = skill_name or self.DEFAULT_SKILL_NAME
        return f"{self._push_scheme_host(fmt)}/{scene}/{skill}"

    def build_rtmp_push_url(
        self,
        scene_id: Optional[str] = None,
        skill_name: Optional[str] = None,
    ) -> str:
        return self.build_push_url("rtmp", scene_id, skill_name)

    def build_rtsp_push_url(
        self,
        scene_id: Optional[str] = None,
        skill_name: Optional[str] = None,
    ) -> str:
        return self.build_push_url("rtsp", scene_id, skill_name)

    def resolve_test_video_path(self, base_dir: Optional[Path] = None) -> str:
        base = base_dir or _CODE_ROOT
        path = Path(self.DEFAULT_TEST_VIDEO)
        if path.is_absolute():
            return str(path)
        return str(base / path)

    def resolve_stream_start(self, data: dict) -> dict:
        """补全推流启动参数中的默认项（out_url、in_url 等）。"""
        resolved = dict(data)
        output_format = resolved.get("output_format") or self.DEFAULT_OUTPUT_FORMAT
        resolved["output_format"] = output_format

        if not resolved.get("out_url"):
            resolved["out_url"] = self.build_push_url(
                output_format=output_format,
                scene_id=resolved.get("scene_id"),
                skill_name=resolved.get("skill_name"),
            )

        return resolved


settings = Settings()
