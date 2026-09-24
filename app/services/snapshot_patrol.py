"""周期截图巡检：对 snapshot 模式技能定时 ZLM getSnap 并识别。"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.db import session_scope
from app.db.models import TaskConfig
from app.plugins.skill_registry import create_skill, get_skill_run_mode
from app.services import camera_pose_core as pose_core
from app.services.mgmt_service import persist_classified_records, resolve_camera_snap_urls
from app.services.webhook_dispatch import schedule_alert_webhooks
from app.services.zlm_client import zlm_client

logger = logging.getLogger(__name__)

RUNTIME_PREFIX = "snap:"


class _PatrolJob:
    def __init__(self, task_id: int):
        self.task_id = task_id
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.runtime_id = f"{RUNTIME_PREFIX}{task_id}"
        self.last_error: str = ""
        self.last_status: str = "starting"
        self.last_check_at: Optional[str] = None


class SnapshotPatrolService:
    """API 进程内管理 snapshot 巡检线程。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[int, _PatrolJob] = {}

    def is_running(self, task_id: int) -> bool:
        with self._lock:
            job = self._jobs.get(task_id)
            return bool(
                job and job.thread and job.thread.is_alive() and not job.stop_event.is_set()
            )

    def runtime_id_for(self, task_id: int) -> str:
        return f"{RUNTIME_PREFIX}{task_id}"

    def get_status(self, task_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(task_id)
            if not job:
                return None
            alive = bool(
                job.thread and job.thread.is_alive() and not job.stop_event.is_set()
            )
            return {
                "task_id": job.runtime_id,
                "status": "running" if alive else "stopped",
                "pid": None,
                "worker_id": "local",
                "worker_name": "本机巡检",
                "last_error": job.last_error,
                "last_status": job.last_status,
                "last_check_at": job.last_check_at,
                "out_url": None,
            }

    def start(self, task_id: int) -> Dict[str, Any]:
        with self._lock:
            existing = self._jobs.get(task_id)
            if existing and existing.thread and existing.thread.is_alive():
                return {"task_id": existing.runtime_id, "status": "running", "worker_id": "local"}

            job = _PatrolJob(task_id)
            th = threading.Thread(
                target=self._run_loop,
                args=(job,),
                name=f"snap-patrol-{task_id}",
                daemon=True,
            )
            job.thread = th
            self._jobs[task_id] = job
            th.start()
            return {"task_id": job.runtime_id, "status": "running", "worker_id": "local"}

    def stop(self, task_id: int) -> None:
        with self._lock:
            job = self._jobs.pop(task_id, None)
        if not job:
            return
        job.stop_event.set()
        job.last_status = "stopping"
        if job.thread and job.thread.is_alive():
            job.thread.join(timeout=5.0)

    def stop_by_runtime_id(self, runtime_id: str) -> None:
        rid = (runtime_id or "").strip()
        if not rid.startswith(RUNTIME_PREFIX):
            raise ValueError("非截图巡检运行实例")
        task_id = int(rid[len(RUNTIME_PREFIX) :])
        self.stop(task_id)

    def _run_loop(self, job: _PatrolJob) -> None:
        logger.info("截图巡检已启动 task_id=%s", job.task_id)
        skill = None
        params_fp = ""
        interval = 5.0
        while not job.stop_event.is_set():
            started = time.time()
            try:
                skill, params_fp, interval = self._tick(job, skill, params_fp)
                job.last_status = "ok"
                job.last_error = ""
            except Exception as e:
                job.last_status = "error"
                job.last_error = str(e)
                logger.exception("截图巡检失败 task_id=%s", job.task_id)
            job.last_check_at = datetime.now().isoformat(timespec="seconds")
            elapsed = time.time() - started
            wait = max(0.5, float(interval) - elapsed)
            job.stop_event.wait(wait)
        logger.info("截图巡检已停止 task_id=%s", job.task_id)

    def _load_task_bundle(self, task_id: int) -> Dict[str, Any]:
        with session_scope() as db:
            row = db.scalars(
                select(TaskConfig)
                .options(
                    joinedload(TaskConfig.camera),
                    joinedload(TaskConfig.algorithm_config),
                )
                .where(TaskConfig.id == task_id)
            ).first()
            if row is None:
                raise ValueError("任务不存在")
            if not row.enabled:
                raise ValueError("任务已禁用")
            cam = row.camera
            algo = row.algorithm_config
            if not cam or not algo:
                raise ValueError("摄像头或算法配置缺失")
            skill_name = algo.skill_name
            if get_skill_run_mode(skill_name) != "snapshot":
                raise ValueError(f"技能 {skill_name} 不是 snapshot 模式")
            params = dict(algo.extra_params or {})
            ref_url = str(params.get("reference_image_url") or "").strip()
            if not ref_url:
                raise ValueError("未配置校准模板图片地址")
            candidates = resolve_camera_snap_urls(db, cam)
            return {
                "skill_name": skill_name,
                "params": params,
                "params_fp": str(sorted(params.items())),
                "interval": max(1.0, float(params.get("check_interval_sec", 5) or 5)),
                "ref_url": ref_url,
                "candidates": candidates,
                "scene_id": row.scene_id,
                "camera_id": cam.id,
                "camera_code": str(getattr(cam, "camera_code", None) or "").strip(),
                "mine_code": str(getattr(cam, "mine_code", None) or "").strip(),
                "analysis_type": str(getattr(cam, "analysis_type", None) or "").strip(),
                "alert_image_enabled": bool(getattr(row, "alert_image_enabled", True)),
            }

    def _tick(
        self, job: _PatrolJob, skill, params_fp: str
    ) -> Tuple[Any, str, float]:
        bundle = self._load_task_bundle(job.task_id)
        if skill is None or params_fp != bundle["params_fp"]:
            skill = create_skill(bundle["skill_name"], {"params": bundle["params"]})
            params_fp = bundle["params_fp"]
        elif str(getattr(skill, "reference_image_url", "") or "") != bundle["ref_url"]:
            skill.reload_reference(bundle["ref_url"])

        jpeg = zlm_client.get_snap_prefer_zlm(
            bundle["candidates"], timeout_sec=15, expire_sec=30
        )
        frame = pose_core.decode_jpeg_bgr(jpeg)
        result = skill.detect_frame(frame)

        if (
            result.get("has_camera_shift")
            or result.get("has_camera_tilt")
            or result.get("has_dark_alarm")
            or result.get("has_overexp_alarm")
        ):
            self._emit_alert(
                result,
                jpeg=jpeg if bundle["alert_image_enabled"] else b"",
                scene_id=bundle["scene_id"],
                camera_id=bundle["camera_id"],
                camera_code=str(bundle.get("camera_code") or ""),
                mine_code=str(bundle.get("mine_code") or ""),
                analysis_type=str(bundle.get("analysis_type") or ""),
                alert_image_enabled=bundle["alert_image_enabled"],
            )
        return skill, params_fp, bundle["interval"]

    def _emit_alert(
        self,
        result: Dict[str, Any],
        *,
        jpeg: bytes,
        scene_id: str,
        camera_id: int,
        camera_code: str = "",
        mine_code: str = "",
        analysis_type: str = "",
        alert_image_enabled: bool,
    ) -> None:
        skill_name = str(result.get("skill_name") or "camera_shift_detector")
        is_tilt = bool(result.get("has_camera_tilt")) or skill_name == "camera_tilt_detector"
        is_dark = bool(result.get("has_dark_alarm"))
        is_overexp = bool(result.get("has_overexp_alarm"))
        if is_dark:
            prefix, default_type, label = "guoan", "dark", "画面过暗"
            event_type = "dark_alarm"
        elif is_overexp:
            prefix, default_type, label = "guobao", "overexp", "画面过曝"
            event_type = "overexp_alarm"
        elif is_tilt:
            prefix, default_type, label = "camera_tilt", "09", "角度偏离"
            event_type = "camera_tilt_alert"
        else:
            prefix, default_type, label = "camera_shift", "08", "位置挪移"
            event_type = "camera_shift_alert"

        pic_url = None
        raw_frame = None
        if alert_image_enabled and jpeg:
            try:
                from app.services.minio_client import minio_storage
                import numpy as np

                name = (
                    f"{prefix}/{scene_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    f"_{uuid.uuid4().hex[:8]}.jpg"
                )
                pic_url = minio_storage.upload_bytes(jpeg, name, "image/jpeg")
                # 解码供视频质量异常转发 imagesBase64
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                raw_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                logger.exception("%s告警截图上传失败 scene=%s", label, scene_id)

        now = datetime.now()
        shift_px = result.get("shift_px")
        rotation_deg = result.get("rotation_deg")
        perspective_score = result.get("perspective_score")
        if is_dark or is_overexp:
            message = (
                f"{label} scene={scene_id} status={result.get('status')} "
                f"msg={result.get('message')}"
            )
        elif is_tilt:
            message = (
                f"摄像头角度偏离 scene={scene_id} "
                f"rotation_deg={rotation_deg} perspective={perspective_score} "
                f"ssim={result.get('ssim')} inlier={result.get('inlier_ratio')} "
                f"msg={result.get('message')}"
            )
        else:
            message = (
                f"摄像头位置挪移 scene={scene_id} shift_px={shift_px} "
                f"ssim={result.get('ssim')} inlier={result.get('inlier_ratio')} "
                f"msg={result.get('message')}"
            )

        event = {
            "alert_id": f"{scene_id}_{int(now.timestamp())}_{uuid.uuid4().hex[:6]}",
            "type": event_type,
            "category": "alert",
            "scene_id": scene_id,
            "skill_name": skill_name,
            "recognition_types": result.get("recognition_types") or [default_type],
            "message": message,
            "count": 0,
            "enter_count": 0,
            "active_alerts": result.get("active_alerts") or [],
            "alert_definitions": result.get("alert_definitions") or [],
            "has_camera_shift": bool(result.get("has_camera_shift")),
            "has_camera_tilt": bool(result.get("has_camera_tilt")),
            "has_dark_alarm": bool(result.get("has_dark_alarm")),
            "has_overexp_alarm": bool(result.get("has_overexp_alarm")),
            "shift_px": shift_px,
            "rotation_deg": rotation_deg,
            "perspective_score": perspective_score,
            "camera_id": camera_id,
            "camera_code": camera_code,
            "mine_code": mine_code,
            "analysis_type": analysis_type,
            "time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "pic_url": pic_url,
            "image_minio_url": pic_url,
            "alert_image_enabled": alert_image_enabled,
            "alert_video_enabled": False,
            "timestamp": now.isoformat(),
        }
        rows = persist_classified_records(event) or []
        if rows:
            schedule_alert_webhooks(rows, event_type="alert.created")
            logger.info(
                "%s告警已落库 scene=%s shift_px=%s rotation_deg=%s",
                label,
                scene_id,
                shift_px,
                rotation_deg,
            )
        try:
            from app.services.stream_alert import forward_video_anomaly_alerts

            forward_video_anomaly_alerts(event, raw_frame, scene_id=scene_id)
        except Exception:
            logger.exception("视频质量异常转发失败 scene=%s", scene_id)


snapshot_patrol = SnapshotPatrolService()
