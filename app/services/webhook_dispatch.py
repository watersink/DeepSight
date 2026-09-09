"""第三方 Webhook 投递：落库后入内部 MQ，Worker 消费再 POST"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings
from app.db import session_scope
from app.db.models import AlertRecord, PartnerIntegration
from app.services import partner_service as partner_svc

logger = logging.getLogger(__name__)


def _as_str_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    return [str(value).strip()] if str(value).strip() else []


def partner_matches_alert(partner: PartnerIntegration, row: AlertRecord) -> bool:
    cats = _as_str_list(partner.subscribe_categories) or ["alert"]
    if (row.category or "alert") not in cats:
        return False

    type_filter = _as_str_list(partner.subscribe_recognition_types)
    if type_filter:
        row_types = set(_as_str_list(row.recognition_types))
        if not row_types.intersection(type_filter):
            return False

    scene_filter = _as_str_list(partner.subscribe_scene_ids)
    if scene_filter and (row.scene_id or "") not in scene_filter:
        return False

    return True


def build_webhook_payload(
    row: AlertRecord, *, event_type: str = "alert.created"
) -> Dict[str, Any]:
    occurred = row.created_at
    if occurred is not None and occurred.tzinfo is None:
        occurred_s = occurred.isoformat(sep="T", timespec="seconds") + "+08:00"
    elif occurred is not None:
        occurred_s = occurred.isoformat()
    else:
        occurred_s = datetime.now(timezone.utc).isoformat()

    return {
        "schema_version": "1",
        "event_type": event_type,
        "alert_id": row.alert_uid,
        "id": row.id,
        "category": row.category or "alert",
        "recognition_types": _as_str_list(row.recognition_types),
        "scene_id": row.scene_id or "",
        "skill_name": row.skill_name or "",
        "status": row.status or "new",
        "occurred_at": occurred_s,
        "has_image": bool(row.image_url),
        "has_video": bool(row.video_url),
        "href": f"/open/v1/alerts/{row.alert_uid}",
    }


def _sign(secret: str, timestamp: str, body: bytes) -> str:
    msg = timestamp.encode("utf-8") + b"." + body
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def deliver_webhook(
    partner: PartnerIntegration,
    payload: Dict[str, Any],
    *,
    timeout: Optional[float] = None,
) -> bool:
    url = (partner.webhook_url or "").strip()
    if not url:
        logger.debug("跳过 Webhook：未配置 URL partner=%s", partner.code)
        return False

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    ts = str(int(datetime.now(timezone.utc).timestamp()))
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "X-Timestamp": ts,
        "X-Partner-Code": partner.code,
        "User-Agent": "YingJiTing-Webhook/1.0",
    }
    secret = (partner.webhook_secret or "").strip()
    if secret:
        headers["X-Signature"] = _sign(secret, ts, body)

    to = float(timeout if timeout is not None else settings.WEBHOOK_TIMEOUT)
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=to) as resp:
            status_code = int(getattr(resp, "status", 200) or 200)
            if 200 <= status_code < 300:
                logger.info(
                    "Webhook 投递成功 partner=%s status=%s alert_id=%s event=%s",
                    partner.code,
                    status_code,
                    payload.get("alert_id"),
                    payload.get("event_type"),
                )
                return True
            logger.warning(
                "Webhook 投递非 2xx partner=%s status=%s",
                partner.code,
                status_code,
            )
            return False
    except urllib.error.HTTPError as e:
        logger.warning(
            "Webhook HTTP 错误 partner=%s status=%s body=%s",
            partner.code,
            e.code,
            (e.read() or b"")[:300],
        )
        return False
    except Exception:
        logger.exception(
            "Webhook 投递失败 partner=%s url=%s alert_id=%s",
            partner.code,
            url,
            payload.get("alert_id"),
        )
        return False


def _alert_to_dict(r: AlertRecord) -> Dict[str, Any]:
    d = dict(getattr(r, "__dict__", {}) or {})
    types = d.get("recognition_types")
    if isinstance(types, list):
        types = list(types)
    created = d.get("created_at")
    if isinstance(created, datetime):
        created_s = created.isoformat(sep="T", timespec="seconds")
    else:
        created_s = created
    return {
        "id": int(d.get("id") or 0),
        "alert_uid": str(d.get("alert_uid") or ""),
        "scene_id": str(d.get("scene_id") or ""),
        "skill_name": str(d.get("skill_name") or ""),
        "recognition_types": types,
        "message": str(d.get("message") or ""),
        "count": int(d.get("count") or 0),
        "enter_count": int(d.get("enter_count") or 0),
        "image_url": d.get("image_url"),
        "video_url": d.get("video_url"),
        "category": str(d.get("category") or "alert"),
        "status": str(d.get("status") or "new"),
        "created_at": created_s,
    }


def _dict_to_alert(data: Dict[str, Any]) -> AlertRecord:
    created = data.get("created_at")
    if isinstance(created, str) and created:
        try:
            created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if created_at.tzinfo is not None:
                created_at = created_at.replace(tzinfo=None)
        except ValueError:
            created_at = None
    elif isinstance(created, datetime):
        created_at = created
    else:
        created_at = None
    return AlertRecord(
        id=int(data.get("id") or 0),
        alert_uid=str(data.get("alert_uid") or ""),
        scene_id=str(data.get("scene_id") or ""),
        skill_name=str(data.get("skill_name") or ""),
        recognition_types=data.get("recognition_types"),
        message=str(data.get("message") or ""),
        count=int(data.get("count") or 0),
        enter_count=int(data.get("enter_count") or 0),
        image_url=data.get("image_url"),
        video_url=data.get("video_url"),
        category=str(data.get("category") or "alert"),
        status=str(data.get("status") or "new"),
        created_at=created_at,
    )


def _load_partner_snapshots() -> List[Dict[str, Any]]:
    with session_scope() as db:
        partners = partner_svc.list_enabled_partners(db)
        return [
            {
                "id": p.id,
                "code": p.code,
                "webhook_url": p.webhook_url,
                "webhook_secret": p.webhook_secret,
                "subscribe_categories": p.subscribe_categories,
                "subscribe_recognition_types": p.subscribe_recognition_types,
                "subscribe_scene_ids": p.subscribe_scene_ids,
            }
            for p in partners
            if (p.webhook_url or "").strip()
        ]


def _pseudo_partner(snap: Dict[str, Any]) -> PartnerIntegration:
    return PartnerIntegration(
        id=int(snap["id"]),
        name="",
        code=str(snap["code"]),
        webhook_url=str(snap.get("webhook_url") or ""),
        webhook_secret=str(snap.get("webhook_secret") or ""),
        api_key="",
        enabled=True,
        subscribe_categories=snap.get("subscribe_categories"),
        subscribe_recognition_types=snap.get("subscribe_recognition_types"),
        subscribe_scene_ids=snap.get("subscribe_scene_ids"),
    )


def build_delivery_tasks(
    rows: Sequence[AlertRecord], *, event_type: str = "alert.created"
) -> List[Dict[str, Any]]:
    """按订阅过滤，生成「一家接入方一条」投递任务。"""
    try:
        snapshots = _load_partner_snapshots()
    except Exception:
        logger.exception("加载接入方失败")
        return []
    if not snapshots:
        return []

    tasks: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        for snap in snapshots:
            partner = _pseudo_partner(snap)
            if not partner_matches_alert(partner, row):
                continue
            tasks.append(
                {
                    "schema_version": "1",
                    "task_type": "webhook.deliver",
                    "event_type": event_type,
                    "partner_id": int(snap["id"]),
                    "partner_code": str(snap["code"]),
                    "payload": build_webhook_payload(row, event_type=event_type),
                    "retry": 0,
                    "enqueued_at": now,
                }
            )
    return tasks


def _direct_deliver_tasks(tasks: List[Dict[str, Any]]) -> None:
    """MQ 不可用时的进程内回退：后台线程直推。"""
    if not tasks:
        return

    def _run() -> None:
        try:
            id_map = {t["partner_id"]: t for t in tasks}
            with session_scope() as db:
                partners = {
                    p.id: p
                    for p in partner_svc.list_enabled_partners(db)
                    if p.id in id_map
                }
            for task in tasks:
                pid = int(task.get("partner_id") or 0)
                partner = partners.get(pid)
                if partner is None or not (partner.webhook_url or "").strip():
                    continue
                deliver_webhook(partner, task.get("payload") or {})
        except Exception:
            logger.exception("Webhook 直推回退失败")

    threading.Thread(target=_run, name="webhook-fallback", daemon=True).start()


def enqueue_webhook_tasks(tasks: List[Dict[str, Any]]) -> int:
    """写入内部 MQ；返回成功条数。"""
    if not tasks:
        return 0
    from app.services.internal_mq import publish_json

    return sum(1 for task in tasks if publish_json(task))


def schedule_alert_webhooks(
    rows: Sequence[AlertRecord], *, event_type: str = "alert.created"
) -> None:
    """
    落库后入口：生成投递任务写入内部 MQ（削峰）。
    MQ 关闭或发布失败时回退为进程内线程直推。
    """
    if not rows:
        return
    copies = [_dict_to_alert(_alert_to_dict(r)) for r in rows]
    tasks = build_delivery_tasks(copies, event_type=event_type)
    if not tasks:
        logger.debug("无匹配的 Webhook 接入方，跳过投递")
        return

    use_mq = bool(settings.RABBITMQ_WEBHOOK_ENABLED)
    if use_mq:
        from app.services.internal_mq import publish_json

        failed: List[Dict[str, Any]] = []
        published = 0
        for task in tasks:
            if publish_json(task):
                published += 1
            else:
                failed.append(task)
        if published:
            logger.info(
                "Webhook 任务已入队 count=%s event_type=%s",
                published,
                event_type,
            )
        if failed:
            logger.warning(
                "Webhook MQ 入队失败 %s 条，回退进程内直推",
                len(failed),
            )
            _direct_deliver_tasks(failed)
        return

    logger.info("Webhook MQ 已关闭，进程内直推 count=%s", len(tasks))
    _direct_deliver_tasks(tasks)


def process_webhook_mq_message(data: Dict[str, Any]) -> bool:
    """
    Worker 回调：处理一条投递任务。
    返回 True 表示 ack；False 表示 nack（不 requeue）。
    失败且未超重试次数时重新入队（retry+1）。
    """
    if not isinstance(data, dict):
        return True
    if data.get("task_type") not in (None, "webhook.deliver"):
        logger.warning("忽略未知 MQ 任务类型: %s", data.get("task_type"))
        return True

    partner_id = int(data.get("partner_id") or 0)
    payload = data.get("payload") or {}
    retry = int(data.get("retry") or 0)
    max_retry = int(settings.RABBITMQ_WEBHOOK_MAX_RETRY)

    if partner_id <= 0 or not payload:
        logger.warning("Webhook MQ 任务字段无效 partner_id=%s", partner_id)
        return True

    try:
        with session_scope() as db:
            partner = partner_svc.get_partner(db, partner_id)
            if partner is None or not partner.enabled:
                logger.info("接入方不存在或已禁用，丢弃任务 partner_id=%s", partner_id)
                return True
            if not (partner.webhook_url or "").strip():
                logger.info("接入方无 Webhook URL，丢弃任务 partner=%s", partner.code)
                return True
            # 在 session 内取所需字段，避免 detached
            snap = PartnerIntegration(
                id=partner.id,
                name=partner.name or "",
                code=partner.code,
                webhook_url=partner.webhook_url,
                webhook_secret=partner.webhook_secret or "",
                api_key="",
                enabled=True,
            )
    except Exception:
        logger.exception("加载接入方失败 partner_id=%s", partner_id)
        return _requeue_or_drop(data, retry, max_retry)

    ok = deliver_webhook(snap, payload)
    if ok:
        return True
    return _requeue_or_drop(data, retry, max_retry)


def _requeue_or_drop(data: Dict[str, Any], retry: int, max_retry: int) -> bool:
    if retry >= max_retry:
        logger.error(
            "Webhook 投递放弃 partner_id=%s alert_id=%s retry=%s",
            data.get("partner_id"),
            (data.get("payload") or {}).get("alert_id"),
            retry,
        )
        return True  # ack 丢掉，避免堵队列
    nxt = dict(data)
    nxt["retry"] = retry + 1
    from app.services.internal_mq import publish_json

    if publish_json(nxt):
        logger.warning(
            "Webhook 投递失败已重入队 partner_id=%s retry=%s",
            data.get("partner_id"),
            nxt["retry"],
        )
        return True  # 原消息 ack，新消息在队列
    logger.error("Webhook 重入队失败 partner_id=%s", data.get("partner_id"))
    return False


def start_webhook_mq_worker() -> None:
    from app.services import internal_mq as imq

    if imq.webhook_mq_worker and imq.webhook_mq_worker.running:
        return
    imq.webhook_mq_worker = imq.WebhookMqWorker(process_webhook_mq_message)
    imq.webhook_mq_worker.start()


def stop_webhook_mq_worker() -> None:
    from app.services import internal_mq as imq

    if imq.webhook_mq_worker:
        imq.webhook_mq_worker.stop()
        imq.webhook_mq_worker = None
