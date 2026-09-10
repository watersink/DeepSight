"""基于 APScheduler 的任务运行时间窗对账。

任务配置 schedule 示例：
{
  "enabled": true,
  "timezone": "Asia/Shanghai",
  "days": [1, 2, 3, 4, 5],   # ISO 周一=1 … 周日=7
  "start_time": "09:00",
  "end_time": "18:00"
}
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, time
from typing import Any, Dict, Optional, Set

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.db import session_scope
from app.db.models import TaskConfig
from app.services import mgmt_service as svc
from app.services.runtime_gateway import WorkerApiError, runtime_gateway

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None
_lock = threading.Lock()
DEFAULT_TZ = "Asia/Shanghai"
RECONCILE_SECONDS = 60


def _parse_hhmm(value: str) -> time:
    text = (value or "").strip()
    parts = text.split(":")
    if len(parts) < 2:
        raise ValueError(f"非法时间格式: {value!r}，期望 HH:MM")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"非法时间值: {value!r}")
    return time(hour=hour, minute=minute)


def _fmt_hhmm(value: str) -> str:
    t = _parse_hhmm(value)
    return f"{t.hour:02d}:{t.minute:02d}"


def normalize_schedule(raw: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """规范化并校验 schedule；空对象返回 None。"""
    if not raw:
        return None
    enabled = bool(raw.get("enabled"))
    tz_name = str(raw.get("timezone") or DEFAULT_TZ).strip() or DEFAULT_TZ
    try:
        pytz.timezone(tz_name)
    except Exception as exc:
        raise ValueError(f"无效时区: {tz_name}") from exc

    days_raw = raw.get("days")
    if days_raw is None:
        days = [1, 2, 3, 4, 5, 6, 7]
    else:
        days = sorted({int(d) for d in days_raw})
        if not days or any(d < 1 or d > 7 for d in days):
            raise ValueError("days 需为 1–7（周一–周日）的非空集合")

    start_s = _fmt_hhmm(str(raw.get("start_time") or "09:00"))
    end_s = _fmt_hhmm(str(raw.get("end_time") or "18:00"))

    return {
        "enabled": enabled,
        "timezone": tz_name,
        "days": days,
        "start_time": start_s,
        "end_time": end_s,
    }


def is_within_schedule(
    schedule: Optional[Dict[str, Any]],
    now: Optional[datetime] = None,
) -> bool:
    """当前是否落在运行时间窗内。schedule 未启用或为空 → False。"""
    if not schedule or not schedule.get("enabled"):
        return False
    tz_name = str(schedule.get("timezone") or DEFAULT_TZ)
    try:
        tz = pytz.timezone(tz_name)
    except Exception:
        tz = pytz.timezone(DEFAULT_TZ)

    local_now = now.astimezone(tz) if now and now.tzinfo else datetime.now(tz)
    days: Set[int] = {int(d) for d in (schedule.get("days") or [])}
    if local_now.isoweekday() not in days:
        return False

    start_t = _parse_hhmm(str(schedule.get("start_time") or "09:00"))
    end_t = _parse_hhmm(str(schedule.get("end_time") or "18:00"))
    cur = local_now.timetz().replace(tzinfo=None)

    if start_t == end_t:
        # 起止相同视为全天（在所选星期内）
        return True
    if start_t < end_t:
        return start_t <= cur < end_t
    # 跨午夜：22:00–06:00
    return cur >= start_t or cur < end_t


def schedule_wants_running(row: TaskConfig, now: Optional[datetime] = None) -> bool:
    """配置启用 + 时间窗启用 + 当前在窗内 → 应对账为运行中。"""
    if not row.enabled:
        return False
    return is_within_schedule(row.schedule, now=now)


def assert_manual_start_allowed(row: TaskConfig) -> None:
    sched = row.schedule or {}
    if not sched.get("enabled"):
        return
    if not is_within_schedule(sched):
        raise ValueError(
            "当前不在任务配置的运行时间段内，无法手动启动；"
            "可关闭「按时间段自动运行」或调整时间窗后再试"
        )


def _runtime_alive(row: TaskConfig) -> bool:
    tid = row.last_runtime_task_id
    if not tid:
        return False
    worker_id = getattr(row, "worker_id", None)
    try:
        data = runtime_gateway.get_task(tid, worker_id=worker_id)
    except WorkerApiError:
        logger.exception(
            "探测任务存活失败 runtime=%s worker=%s", tid, worker_id
        )
        return False
    if not data:
        return False
    status = data.get("status")
    if status not in {"running", "starting"}:
        return False
    # 远程 Worker 可能不回传 pid；状态为 running/starting 即视为存活
    if data.get("pid") is not None:
        return True
    node_local = str(data.get("worker_id") or worker_id or "") == "local"
    if node_local:
        return data.get("pid") is not None
    return True


def reconcile_scheduled_tasks() -> None:
    """扫描开启了时间窗的任务，自动启停运行时进程。"""
    try:
        with session_scope() as db:
            rows = (
                db.scalars(
                    select(TaskConfig).options(
                        joinedload(TaskConfig.camera),
                        joinedload(TaskConfig.algorithm_config),
                    )
                )
                .unique()
                .all()
            )
            for row in rows:
                sched = row.schedule or {}
                if not sched.get("enabled"):
                    continue
                want = schedule_wants_running(row)
                alive = _runtime_alive(row)
                try:
                    if want and not alive:
                        logger.info(
                            "时间窗对账：启动任务 id=%s name=%s",
                            row.id,
                            row.name,
                        )
                        svc.start_task_config(db, row)
                    elif not want and alive:
                        logger.info(
                            "时间窗对账：停止任务 id=%s name=%s",
                            row.id,
                            row.name,
                        )
                        svc.stop_task_config(db, row)
                except Exception:
                    logger.exception(
                        "时间窗对账失败 task_id=%s name=%s",
                        row.id,
                        row.name,
                    )
    except Exception:
        logger.exception("时间窗对账扫描失败")


def start_scheduler() -> None:
    global _scheduler
    with _lock:
        if _scheduler and _scheduler.running:
            return
        sched = BackgroundScheduler(timezone=DEFAULT_TZ)
        sched.add_job(
            reconcile_scheduled_tasks,
            trigger="interval",
            seconds=RECONCILE_SECONDS,
            id="task_schedule_reconcile",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        sched.start()
        _scheduler = sched
        logger.info(
            "APScheduler 已启动：每 %ss 对账任务运行时间窗",
            RECONCILE_SECONDS,
        )
        # 启动后立即对账一次
        threading.Thread(
            target=reconcile_scheduled_tasks,
            name="task-schedule-bootstrap",
            daemon=True,
        ).start()


def stop_scheduler() -> None:
    global _scheduler
    with _lock:
        if _scheduler is None:
            return
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            logger.exception("APScheduler 关闭异常")
        _scheduler = None
        logger.info("APScheduler 已停止")
