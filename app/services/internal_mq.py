"""本项目内部 RabbitMQ：Webhook 投递队列（削峰填谷）"""
from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)

_topology_lock = threading.Lock()
_topology_ready = False


def _import_pika():
    try:
        import pika
    except ImportError as e:
        raise RuntimeError("未安装 pika，请执行 pip install pika") from e
    return pika


def connection_params():
    pika = _import_pika()
    credentials = pika.PlainCredentials(
        settings.RABBITMQ_USERNAME,
        settings.RABBITMQ_PASSWORD,
    )
    return pika.ConnectionParameters(
        host=settings.RABBITMQ_HOST,
        port=int(settings.RABBITMQ_PORT),
        virtual_host=settings.RABBITMQ_VHOST or "/",
        credentials=credentials,
        heartbeat=30,
        blocked_connection_timeout=float(settings.RABBITMQ_TIMEOUT),
        socket_timeout=float(settings.RABBITMQ_TIMEOUT),
    )


def ensure_webhook_topology(channel) -> None:
    """声明交换机、队列并绑定（带长度与 TTL 限制）。"""
    exchange = settings.RABBITMQ_WEBHOOK_EXCHANGE
    queue = settings.RABBITMQ_WEBHOOK_QUEUE
    routing_key = settings.RABBITMQ_WEBHOOK_ROUTING_KEY
    channel.exchange_declare(
        exchange=exchange, exchange_type="direct", durable=True
    )
    args = {
        "x-max-length": int(settings.RABBITMQ_WEBHOOK_QUEUE_MAX_LENGTH),
        "x-overflow": "drop-head",
        "x-message-ttl": int(settings.RABBITMQ_WEBHOOK_MESSAGE_TTL_MS),
    }
    channel.queue_declare(queue=queue, durable=True, arguments=args)
    channel.queue_bind(queue=queue, exchange=exchange, routing_key=routing_key)


def ensure_topology_once() -> bool:
    """进程内确保拓扑已创建；失败返回 False。"""
    global _topology_ready
    if _topology_ready:
        return True
    with _topology_lock:
        if _topology_ready:
            return True
        pika = _import_pika()
        connection = None
        try:
            connection = pika.BlockingConnection(connection_params())
            channel = connection.channel()
            ensure_webhook_topology(channel)
            _topology_ready = True
            logger.info(
                "内部 MQ Webhook 拓扑已就绪 exchange=%s queue=%s max_length=%s ttl_ms=%s",
                settings.RABBITMQ_WEBHOOK_EXCHANGE,
                settings.RABBITMQ_WEBHOOK_QUEUE,
                settings.RABBITMQ_WEBHOOK_QUEUE_MAX_LENGTH,
                settings.RABBITMQ_WEBHOOK_MESSAGE_TTL_MS,
            )
            return True
        except Exception:
            logger.exception(
                "内部 MQ 拓扑初始化失败 host=%s:%s",
                settings.RABBITMQ_HOST,
                settings.RABBITMQ_PORT,
            )
            return False
        finally:
            if connection is not None and connection.is_open:
                try:
                    connection.close()
                except Exception:
                    pass


def publish_json(
    payload: Dict[str, Any],
    *,
    routing_key: Optional[str] = None,
) -> bool:
    """向 Webhook 投递交换机发布一条 JSON 任务。"""
    pika = _import_pika()
    if not ensure_topology_once():
        return False
    rk = routing_key or settings.RABBITMQ_WEBHOOK_ROUTING_KEY
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    connection = None
    try:
        connection = pika.BlockingConnection(connection_params())
        channel = connection.channel()
        channel.confirm_delivery()
        channel.basic_publish(
            exchange=settings.RABBITMQ_WEBHOOK_EXCHANGE,
            routing_key=rk,
            body=body,
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
            ),
            mandatory=False,
        )
        return True
    except Exception:
        logger.exception("内部 MQ 发布失败 routing_key=%s", rk)
        return False
    finally:
        if connection is not None and connection.is_open:
            try:
                connection.close()
            except Exception:
                pass


class WebhookMqWorker:
    """消费内部投递队列，调用回调处理消息。"""

    def __init__(self, on_message: Callable[[Dict[str, Any]], bool]):
        self._on_message = on_message
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._connection = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        if not settings.RABBITMQ_WEBHOOK_ENABLED:
            logger.info("RABBITMQ_WEBHOOK_ENABLED=false，跳过 Webhook MQ Worker")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="webhook-mq-worker", daemon=True
        )
        self._thread.start()
        logger.info(
            "Webhook MQ Worker 已启动 queue=%s prefetch=%s",
            settings.RABBITMQ_WEBHOOK_QUEUE,
            settings.RABBITMQ_WEBHOOK_PREFETCH,
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        conn = self._connection
        if conn is not None:
            try:
                conn.add_callback_threadsafe(conn.close)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        logger.info("Webhook MQ Worker 已停止")

    def _run(self) -> None:
        pika = _import_pika()
        while not self._stop.is_set():
            connection = None
            try:
                if not ensure_topology_once():
                    self._stop.wait(3.0)
                    continue
                connection = pika.BlockingConnection(connection_params())
                self._connection = connection
                channel = connection.channel()
                ensure_webhook_topology(channel)
                channel.basic_qos(prefetch_count=int(settings.RABBITMQ_WEBHOOK_PREFETCH))

                def _callback(ch, method, properties, body):  # noqa: ANN001
                    ok = False
                    try:
                        data = json.loads(body.decode("utf-8") or "{}")
                        ok = bool(self._on_message(data))
                    except Exception:
                        logger.exception("Webhook MQ 消息处理异常")
                        ok = False
                    try:
                        if ok:
                            ch.basic_ack(delivery_tag=method.delivery_tag)
                        else:
                            # 失败由业务侧决定是否重入队；此处 nack 不 requeue，避免死循环
                            ch.basic_nack(
                                delivery_tag=method.delivery_tag, requeue=False
                            )
                    except Exception:
                        logger.exception("Webhook MQ ack/nack 失败")

                channel.basic_consume(
                    queue=settings.RABBITMQ_WEBHOOK_QUEUE,
                    on_message_callback=_callback,
                    auto_ack=False,
                )
                while not self._stop.is_set() and connection.is_open:
                    connection.process_data_events(time_limit=1.0)
            except Exception:
                if not self._stop.is_set():
                    logger.exception("Webhook MQ Worker 连接异常，3s 后重试")
                    self._stop.wait(3.0)
            finally:
                self._connection = None
                if connection is not None and getattr(connection, "is_open", False):
                    try:
                        connection.close()
                    except Exception:
                        pass


webhook_mq_worker: Optional[WebhookMqWorker] = None
