"""推流任务默认告警处理"""
import base64
import json
import logging
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pytz

from app.core.config import settings
from app.services.minio_client import MinioClientError, minio_storage

logger = logging.getLogger(__name__)

_alert_queue = None
_MINE_CODE_RE = re.compile(r"^\d{12}$")
# HTTP 上传仅允许的识别类型（04/05 绕行，06/07 闸机反向翻越，08 胶轮车上下车）
_HTTP_UPLOAD_ANALYSIS_TYPES = frozenset({"04", "05", "06", "07", "08"})
_BOARDING_SKILL_NAMES = frozenset(
    {"boarding_detector", "non_fixed_parking_boarding_detector"}
)

# 识别类型中文名（仅用于日志）
_ANALYSIS_TYPE_LABELS = {
    "04": "非常规通道入井",
    "05": "非常规通道出井",
    "06": "入井闸机出闸",
    "07": "出井闸机入闸",
    "08": "无轨胶轮车上车点上下车人数",
}
# 各 scene 上次已成功推送 MQ 的 enter_count，用于判断是否变化
_last_mq_enter_count_by_scene: Dict[str, int] = {}

# ---------------------------------------------------------------------------
# 视频质量异常转发（煤安平台 ANALYSIS_TYPE=03）
# ---------------------------------------------------------------------------

# 告警信号 -> 平台 analysisCase 映射（煤安 videoAnomaly）。
# 0001=画面遮挡 0002=摄像仪挪动 0003=画面过暗 0004=画面过曝
# 0005=图像模糊 0006=画面冻结 0007=视频丢失 0008=画面抖动 0009=分辨率异常
# 注意：本项目内部 08/09 是挪移/角度识别类型，对应平台 0002（摄像仪挪动），
# 与平台 0008/0009（抖动/分辨率）语义不同，不可直译。
VIDEO_ANOMALY_SIGNAL_CASE: Dict[str, Tuple[str, str]] = {
    "occlusion": ("0001", "画面遮挡"),
    "has_zhedang_alarm": ("0001", "画面遮挡"),
    "zhedang": ("0001", "画面遮挡"),
    "dark": ("0003", "画面过暗"),
    "has_dark_alarm": ("0003", "画面过暗"),
    "overexp": ("0004", "画面过曝"),
    "has_overexp_alarm": ("0004", "画面过曝"),
    "blur": ("0005", "图像模糊"),
    "has_blur_alarm": ("0005", "图像模糊"),
    "has_camera_shift": ("0002", "摄像仪挪动"),
    "has_camera_tilt": ("0002", "摄像仪挪动"),
    "recognition_type:08": ("0002", "摄像仪挪动（位置偏移）"),
    "recognition_type:09": ("0002", "摄像仪挪动（角度偏离）"),
}

# 同一 (cameraCode, analysisCase) 的转发冷却秒数（可配），防止告警循环内重复刷平台
def _video_anomaly_cooldown() -> float:
    try:
        return max(
            float(getattr(settings, "COAL_VIDEO_ANOMALY_FORWARD_COOLDOWN", 60.0) or 0.0),
            0.0,
        )
    except (TypeError, ValueError):
        return 60.0


# 各 (cameraCode, analysisCase) 上次成功转发时间戳
_last_video_anomaly_forward_at: Dict[Tuple[str, str], float] = {}

# ---------------------------------------------------------------------------
# 重要运输设备实时人数转发（煤安平台 POST /mine/ai/importPersonCount，文档 9）
# ---------------------------------------------------------------------------

# 监控点位编码：参照附录A.1 矿井位置编码，14 位数字
_POSITION_CODE_RE = re.compile(r"^\d{14}$")
# 出入井方向
_DIRECTION_IN = "01"
_DIRECTION_OUT = "02"
# 摄像仪 analysis_type（01=人员计数入井，02=人员计数出井）-> direction 映射
_ANALYSIS_TYPE_TO_DIRECTION = {"01": _DIRECTION_IN, "02": _DIRECTION_OUT}
# 需要转发到 importPersonCount 的技能（画面人数 / 重要运输设备实时人数）
# 仅纳入 count 语义为「真实人数」的技能；
# 异常检测类（guoan/guobao/mohu/camera_shift/camera_tilt）的 count 是
# 「是否告警」的 0/1 标志，不是人数，必须排除，否则会把告警当成 1 人上报。
_PERSON_COUNT_SKILLS = frozenset(
    {
        "person_presence_detector26",   # 画面人数检测
        "person_count_detector26",      # 人流量/过线计数（count=画面人数）
        "person_count_detector",        # 旧版人数检测
        "boarding_detector",            # 无轨胶轮车上下车人数
        "non_fixed_parking_boarding_detector",  # 非固定停车上下车人数
    }
)
# 技能 -> 人数取值字段（按优先级）
_SKILL_COUNT_FIELDS: Dict[str, Tuple[str, ...]] = {
    "person_presence_detector26": ("count",),
    "person_count_detector26": ("count", "flow_metrics.current_person_count"),
    "person_count_detector": ("count", "flow_metrics.current_person_count"),
    "boarding_detector": ("count", "flow_metrics.current_boarding"),
    "non_fixed_parking_boarding_detector": ("count", "enter_count"),
}
# 保留旧名以兼容既有引用
_PRESENCE_SKILL_NAMES = _PERSON_COUNT_SKILLS

# ---------------------------------------------------------------------------
# 摄像仪状态转发（煤安平台 POST /mine/ai/cameraStatus，文档 5）
# ---------------------------------------------------------------------------

_CAMERA_STATUS_OFFLINE = "0"
_CAMERA_STATUS_ONLINE = "1"
# 实时转发节流：同一 cameraCode 的状态在此秒数内不重复推送
_CAMERA_STATUS_FORWARD_MIN_INTERVAL = 60.0
# 各 cameraCode 上次成功转发状态 (time, status)
_last_camera_status_forward: Dict[str, Tuple[float, str]] = {}
# 5 分钟定时全量推送的后台线程
_camera_status_timer_started = False

# ---------------------------------------------------------------------------
# 井下人数不符转发（煤安平台 POST /mine/ai/undergroundCount，文档 8）
# ---------------------------------------------------------------------------

# 参与井下人数上报的技能（count 语义为真实人数的技能）
_UNDERGROUND_SKILLS = _PERSON_COUNT_SKILLS
# 各 cameraCode 的最新人数（用于求井下总人数）
_UNDERGROUND_CAMERA_COUNTS: Dict[str, int] = {}
# 上次成功上报的井下总人数（仅总人数变化时上报）
_UNDERGROUND_LAST_TOTAL: Optional[int] = None


def configure_alert_queue(alert_queue) -> None:
    """子进程入口注入跨进程告警队列（供 SSE 推送）。"""
    global _alert_queue
    _alert_queue = alert_queue


def _upload_alert_image(frame: Any, object_name: str) -> Optional[str]:
    """将告警帧编码为 JPEG 并上传 MinIO，返回可访问 URL。"""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            logger.warning("告警图片 JPEG 编码失败 object=%s", object_name)
            return None
        return minio_storage.upload_bytes(
            buf.tobytes(),
            object_name=object_name,
            content_type="image/jpeg",
        )
    except MinioClientError:
        logger.exception("告警图片上传 MinIO 失败 object=%s", object_name)
        return None
    except Exception:
        logger.exception("告警图片处理失败 object=%s", object_name)
        return None


def build_person_count_alert(
    data: Dict[str, Any],
    raw_frame,
    scene_id: str,
) -> Dict[str, Any]:
    tz = pytz.timezone("Asia/Shanghai")
    now = datetime.now(tz)
    ts = now.strftime("%Y%m%d-%H:%M:%S.%f")
    alert_time = now.strftime("%Y-%m-%d %H:%M:%S")
    pic_name = f"person_count/{scene_id}_{ts}.jpg"
    want_image = bool(data.get("alert_image_enabled", True))
    pic_url = _upload_alert_image(raw_frame, pic_name) if want_image else None

    timing_ms = data.get("timing_ms") if isinstance(data.get("timing_ms"), dict) else {}
    count = int(data.get("count", 0) or 0)
    enter_count = int(data.get("enter_count", 0) or 0)
    gate_direction = str(data.get("gate_direction") or "").strip().upper() or None
    mine_code = str(data.get("mine_code") or "").strip()
    camera_code = str(data.get("camera_code") or "").strip()
    analysis_type = str(data.get("analysis_type") or "").strip()
    has_enter_count_change = bool(data.get("has_enter_count_change"))
    has_bypass_violation = bool(data.get("has_bypass_violation"))
    has_count_exit_violation = bool(data.get("has_count_exit_violation"))
    enter_events = data.get("enter_events") if isinstance(data.get("enter_events"), list) else []
    bypass_events = data.get("bypass_events") if isinstance(data.get("bypass_events"), list) else []
    count_exit_events = (
        data.get("count_exit_events") if isinstance(data.get("count_exit_events"), list) else []
    )
    active_alerts = data.get("active_alerts") if isinstance(data.get("active_alerts"), list) else []
    alert_definitions = (
        data.get("alert_definitions") if isinstance(data.get("alert_definitions"), list) else []
    )
    recognition_types = (
        data.get("recognition_types") if isinstance(data.get("recognition_types"), list) else []
    )
    if not recognition_types:
        recognition_types = sorted(
            {
                str(e.get("recognition_type"))
                for e in (enter_events + bypass_events + count_exit_events + active_alerts)
                if isinstance(e, dict) and e.get("recognition_type")
            }
        )
    process_time_ms = float(data.get("process_time_ms", 0.0))
    detect_time_ms = float(data.get("detect_time_ms", 0.0))
    draw_time_ms = float(data.get("draw_time_ms", 0.0))
    shape = list(raw_frame.shape) if raw_frame is not None and hasattr(raw_frame, "shape") else None

    timing_suffix = ""
    if timing_ms:
        timing_suffix = " " + " ".join(
            f"{key}={float(value):.2f}" for key, value in timing_ms.items()
        )

    violation_types = sorted(
        {
            str(e.get("violation_type"))
            for e in (enter_events + bypass_events + count_exit_events)
            if isinstance(e, dict) and e.get("violation_type")
        }
    )
    violation_suffix = f" violations={violation_types}" if violation_types else ""
    recognition_suffix = (
        f" recognition_types={recognition_types}" if recognition_types else ""
    )
    gate_suffix = f" gate_direction={gate_direction}" if gate_direction else ""

    alert_desc_parts = []
    for alert in active_alerts:
        if not isinstance(alert, dict):
            continue
        level = alert.get("level")
        code = str(alert.get("recognition_type") or "").strip()
        desc = str(alert.get("description") or "").strip()
        if desc:
            prefix = f"[L{level}/{code}]" if code else f"[L{level}]"
            alert_desc_parts.append(f"{prefix} {desc}" if level is not None else desc)
    alert_desc_suffix = (
        " alert_descriptions=[" + "; ".join(alert_desc_parts) + "]"
        if alert_desc_parts
        else ""
    )

    pic_suffix = (
        f" pic_url={pic_url}"
        if pic_url
        else (" pic=disabled" if not want_image else f" pic={pic_name}(upload_failed)")
    )

    message = (
        f"触发人数告警 time={alert_time} scene={scene_id} count={count} enter_count={enter_count}"
        f"{gate_suffix} "
        f"enter_count_change={has_enter_count_change} "
        f"count_exit_violation={has_count_exit_violation} "
        f"bypass_violation={has_bypass_violation}"
        f"{violation_suffix}{recognition_suffix}{alert_desc_suffix}"
        f"{pic_suffix} shape={tuple(shape) if shape else None} "
        f"process_time_ms={process_time_ms:.2f} detect_time_ms={detect_time_ms:.2f} "
        f"draw_time_ms={draw_time_ms:.2f}{timing_suffix}"
    )

    return {
        "type": "person_count_alert",
        "level": "INFO",
        "alert_id": f"{scene_id}_{ts}",
        "scene_id": scene_id,
        "skill_name": str(data.get("skill_name") or ""),
        "time": alert_time,
        "count": count,
        "enter_count": enter_count,
        "gate_direction": gate_direction,
        "mine_code": mine_code,
        "camera_code": camera_code,
        "analysis_type": analysis_type,
        "has_enter_count_change": has_enter_count_change,
        "has_bypass_violation": has_bypass_violation,
        "has_count_exit_violation": has_count_exit_violation,
        "has_dark_alarm": bool(data.get("has_dark_alarm")),
        "has_overexp_alarm": bool(data.get("has_overexp_alarm")),
        "has_blur_alarm": bool(data.get("has_blur_alarm")),
        "has_zhedang_alarm": bool(data.get("has_zhedang_alarm")),
        "has_camera_shift": bool(data.get("has_camera_shift")),
        "has_camera_tilt": bool(data.get("has_camera_tilt")),
        "recognition_types": recognition_types,
        "enter_events": enter_events,
        "bypass_events": bypass_events,
        "count_exit_events": count_exit_events,
        "active_alerts": active_alerts,
        "alert_definitions": alert_definitions,
        "pic": pic_name if want_image else "",
        "pic_url": pic_url,
        "image_minio_url": pic_url,
        "alert_image_enabled": want_image,
        "alert_video_enabled": bool(data.get("alert_video_enabled", False)),
        "alert_video_duration_sec": int(data.get("alert_video_duration_sec") or 10),
        "shape": shape,
        "process_time_ms": process_time_ms,
        "detect_time_ms": detect_time_ms,
        "draw_time_ms": draw_time_ms,
        "timing_ms": {str(k): float(v) for k, v in timing_ms.items()} if timing_ms else {},
        "message": message,
        "timestamp": now.isoformat(),
    }


def _publish_alert(event: Dict[str, Any]) -> None:
    q = _alert_queue
    if q is None:
        return
    try:
        q.put_nowait(event)
    except queue.Full:
        logger.warning("告警推送队列已满，丢弃本次事件 scene=%s", event.get("scene_id"))
    except Exception:
        logger.exception("写入告警推送队列失败")


def _lookup_camera_by_scene(scene_id: str) -> Optional[Dict[str, str]]:
    """按 scene_id 从任务配置反查摄像头的 camera_code / mine_code / analysis_type。"""
    scene = str(scene_id or "").strip()
    if not scene:
        return None
    try:
        from sqlalchemy import select
        from sqlalchemy.orm import joinedload

        from app.db import SessionLocal
        from app.db.models import TaskConfig

        with SessionLocal() as db:
            row = db.scalars(
                select(TaskConfig)
                .options(joinedload(TaskConfig.camera))
                .where(TaskConfig.scene_id == scene)
                .order_by(TaskConfig.id.desc())
            ).first()
            cam = getattr(row, "camera", None) if row is not None else None
            if cam is None:
                return None
            return {
                "camera_code": str(getattr(cam, "camera_code", None) or "").strip(),
                "mine_code": str(getattr(cam, "mine_code", None) or "").strip(),
                "analysis_type": str(getattr(cam, "analysis_type", None) or "").strip(),
            }
    except Exception:
        logger.exception("按 scene 反查摄像头失败 scene=%s", scene)
        return None


def _resolve_mine_camera_codes(
    event: Dict[str, Any],
) -> Tuple[str, str, str]:
    """解析 mineCode / cameraCode / analysisType。

    优先事件字段（任务启动时从摄像头配置注入），不足时按 scene_id 查库补齐，
    最后再回退全局默认。
    """
    mine_code = str(event.get("mine_code") or "").strip()
    camera_code = str(event.get("camera_code") or "").strip()
    analysis_type = str(event.get("analysis_type") or "").strip()

    if not camera_code or not mine_code or not analysis_type:
        looked = _lookup_camera_by_scene(str(event.get("scene_id") or ""))
        if looked:
            camera_code = camera_code or looked.get("camera_code", "")
            mine_code = mine_code or looked.get("mine_code", "")
            analysis_type = analysis_type or looked.get("analysis_type", "")

    if not mine_code:
        mine_code = str(settings.COUNTING_RECOG_MINE_CODE or "").strip()
    if not camera_code:
        camera_code = str(settings.COUNTING_RECOG_CAMERA_CODE or "").strip()
    return mine_code, camera_code, analysis_type


def _count_current_frame_violators(event: Dict[str, Any]) -> Dict[str, int]:
    """
    统计当前帧各 recognition_type 的违规人数。

    仅依据本帧 bypass_events / count_exit_events / enter_events，按 track_id 去重；
    不使用画面总人数 count，也不使用累计 enter_count。
    对应 04/05（绕行）、06/07（闸机反向翻越）、08（胶轮车上下车）的当次人数。
    """
    track_ids_by_type: Dict[str, set] = {}
    for key in ("bypass_events", "count_exit_events", "enter_events"):
        items = event.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            code = str(item.get("recognition_type") or "").strip()
            if not code:
                continue
            track_id = item.get("track_id")
            # 无 track_id 时用事件对象占位，仍计为 1 人
            identity = track_id if track_id is not None else id(item)
            track_ids_by_type.setdefault(code, set()).add(identity)
    return {code: len(ids) for code, ids in track_ids_by_type.items()}


def _build_counting_recog_payloads(
    event: Dict[str, Any],
    raw_frame: Any = None,
) -> List[Dict[str, Any]]:
    mine_code, camera_code, _analysis_type = _resolve_mine_camera_codes(event)
    if not mine_code or not _MINE_CODE_RE.match(mine_code):
        logger.warning(
            "跳过识别结果上传：煤矿编码无效 mine_code=%r scene=%s",
            mine_code,
            event.get("scene_id"),
        )
        return []
    if not camera_code:
        logger.warning(
            "跳过识别结果上传：摄像仪编码为空 scene=%s",
            event.get("scene_id"),
        )
        return []
    if len(camera_code) != 20:
        logger.warning(
            "摄像仪编码长度不是20位 camera_code=%r len=%s scene=%s，仍将上传",
            camera_code,
            len(camera_code),
            event.get("scene_id"),
        )

    data_time = str(event.get("time") or "").strip()
    if not data_time:
        tz = pytz.timezone("Asia/Shanghai")
        data_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    recognition_types = event.get("recognition_types") or []
    if not isinstance(recognition_types, list) or not recognition_types:
        logger.warning(
            "跳过识别结果上传：recognition_types 为空 scene=%s",
            event.get("scene_id"),
        )
        return []

    # 证据图片：本次告警帧 -> JPEG Base64（同帧所有识别类型共用同一张）
    images_base64: List[str] = []
    image_b64 = _frame_to_jpeg_base64(raw_frame)
    if image_b64:
        images_base64 = [image_b64]
    else:
        logger.info(
            "识别结果上传：无可用告警帧，仅推送数据 scene=%s", event.get("scene_id")
        )

    # 当前帧各类型违规人数（非累计）；04/05/06/07/08 走 HTTP 上传
    violator_counts = _count_current_frame_violators(event)
    payloads: List[Dict[str, Any]] = []
    for analysis_type in recognition_types:
        code = str(analysis_type or "").strip()
        if code not in _HTTP_UPLOAD_ANALYSIS_TYPES:
            continue
        # analysisCase：当前画面该类型违规人数，如 1 人→0001、同帧 2 人→0002
        person_count = int(violator_counts.get(code, 0) or 0)
        if person_count <= 0:
            continue
        payloads.append(
            {
                "mineCode": mine_code,
                "cameraCode": camera_code,
                "analysisType": code,
                "analysisCase": f"{person_count:04d}",
                "dataTime": data_time,
                "imagesBase64": images_base64,
            }
        )
    return payloads


# ===========================================================================
# 【对接文档 §7】人员计数识别数据-推送
#   接口：POST /mine/ai/countingRecog
#   识别类型 analysisType：04 非常规通道入井 / 05 非常规通道出井 /
#     06 入井闸机出闸 / 07 出井闸机入闸 / 08 无轨胶轮车上车点上下车人数识别
#   识别结果 analysisCase：4 位补零人数（0001=1人，依此类推）
#   请求体为 JSON 数组，一次可推送多条，含证据图片 imagesBase64（可选）
#   平台落库后经 SSE（事件名 countingRecog）推给前端
#   实现：app/services/coal_counting_recog_push.py
#   调用位置：default_alert_handler()
#   人数来源：_count_current_frame_violators()（bypass_events /
#     count_exit_events / enter_events，按 track_id 去重）
# ===========================================================================
def push_counting_recog_upload(
    event: Dict[str, Any],
    raw_frame: Any = None,
) -> List[Dict[str, Any]]:
    """将本次识别结果转发到煤安平台 ``POST /mine/ai/countingRecog``（文档 7）。

    - 请求体为 **JSON 数组**，一次把同帧所有识别类型（04/05/06/07/08）作为多条一起提交；
    - 证据图片取本次告警帧（Base64），为空则 ``localFiles`` 为 ``[]``；
    - 落库后平台会经 SSE（事件名 ``countingRecog``）推给前端；
    - 失败只记日志，不影响原有告警流程。

    返回构造好的 payload 列表（供调用方打印字段），未产生记录时返回空列表。
    """
    payloads = _build_counting_recog_payloads(event, raw_frame)
    if not payloads:
        return []

    # 识别类型中文名，便于日志排查
    labels = [
        _ANALYSIS_TYPE_LABELS.get(str(p.get("analysisType") or ""), "")
        for p in payloads
    ]
    logger.info(
        "识别结果转发 /mine/ai/countingRecog：scene=%s cameraCode=%s mineCode=%s "
        "dataTime=%s types=%s cases=%s 图片=%s",
        event.get("scene_id"),
        payloads[0].get("cameraCode"),
        payloads[0].get("mineCode"),
        payloads[0].get("dataTime"),
        [p.get("analysisType") for p in payloads],
        [p.get("analysisCase") for p in payloads],
        [len(p.get("imagesBase64") or []) for p in payloads],
    )
    if any(labels):
        logger.info(
            "识别结果含义：%s",
            "；".join(
                f"{p.get('analysisType')}={label}（{p.get('analysisCase')} 人）"
                for p, label in zip(payloads, labels)
                if label
            ),
        )

    try:
        from app.services.coal_counting_recog_push import push_counting_recog_records

        result = push_counting_recog_records(payloads)
    except Exception:
        logger.exception(
            "识别结果转发失败 scene=%s types=%s",
            event.get("scene_id"),
            [p.get("analysisType") for p in payloads],
        )
        return payloads

    if result and result.get("code") == 200:
        logger.info(
            "识别结果转发成功 scene=%s pushed=%s uploaded=%s",
            event.get("scene_id"),
            result.get("pushed"),
            result.get("uploaded"),
        )
    return payloads


def _build_enter_count_mq_payload(event: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    构建 MQ 过线人数消息。

    analysisCase 取累计 enter_count（4 位补零），与 HTTP 上传的当帧违规人数不同。
    """
    mine_code, camera_code, _analysis_type = _resolve_mine_camera_codes(event)
    if not mine_code or not _MINE_CODE_RE.match(mine_code):
        logger.warning(
            "跳过 MQ 推送：煤矿编码无效 mine_code=%r scene=%s",
            mine_code,
            event.get("scene_id"),
        )
        return None
    if not camera_code:
        logger.warning(
            "跳过 MQ 推送：摄像仪编码为空 scene=%s",
            event.get("scene_id"),
        )
        return None

    data_time = str(event.get("time") or "").strip()
    if not data_time:
        tz = pytz.timezone("Asia/Shanghai")
        data_time = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S")

    enter_count = int(event.get("enter_count", 0) or 0)
    if enter_count < 0:
        enter_count = 0

    return {
        "mineCode": mine_code,
        "cameraCode": camera_code,
        # MQ 的 analysisCase 表示累计过线进入人数 enter_count
        "analysisCase": f"{enter_count:04d}",
        "dataTime": data_time,
    }


def push_enter_count_to_mq(event: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """将 enter_count 推送到 RabbitMQ；仅当 enter_count 相对上次推送值有变化时发送。"""
    scene_id = str(event.get("scene_id") or "")
    current = int(event.get("enter_count", 0) or 0)
    if current < 0:
        current = 0
    prev = _last_mq_enter_count_by_scene.get(scene_id)
    if prev is None:
        # 首次记录基准值；仅当本帧确有 enter_count 变化时才推送
        if not bool(event.get("has_enter_count_change")):
            _last_mq_enter_count_by_scene[scene_id] = current
            logger.info(
                "跳过 MQ 推送：首次记录 enter_count 且本帧无变化 scene=%s enter_count=%s",
                scene_id,
                current,
            )
            return None
    elif prev == current:
        logger.info(
            "跳过 MQ 推送：enter_count 未变化 scene=%s enter_count=%s",
            scene_id,
            current,
        )
        return None

    payload = _build_enter_count_mq_payload(event)
    if not payload:
        return None

    exchange = str(settings.EXTERNAL_RABBITMQ_EXCHANGE or "").strip()
    routing_key = str(settings.EXTERNAL_RABBITMQ_ROUTING_KEY or "").strip()
    if not exchange or not routing_key:
        logger.warning(
            "跳过 MQ 推送：EXTERNAL_RABBITMQ_EXCHANGE/EXTERNAL_RABBITMQ_ROUTING_KEY 未配置"
        )
        return None

    try:
        import pika
    except ImportError:
        logger.error("跳过 MQ 推送：未安装 pika，请执行 pip install pika")
        return None

    connection = None
    try:
        credentials = pika.PlainCredentials(
            settings.EXTERNAL_RABBITMQ_USERNAME,
            settings.EXTERNAL_RABBITMQ_PASSWORD,
        )
        params = pika.ConnectionParameters(
            host=settings.EXTERNAL_RABBITMQ_HOST,
            port=int(settings.EXTERNAL_RABBITMQ_PORT),
            virtual_host=settings.EXTERNAL_RABBITMQ_VHOST or "/",
            credentials=credentials,
            socket_timeout=float(settings.EXTERNAL_RABBITMQ_TIMEOUT),
            blocked_connection_timeout=float(settings.EXTERNAL_RABBITMQ_TIMEOUT),
        )
        connection = pika.BlockingConnection(params)
        channel = connection.channel()
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # 交换机/绑定由服务端预先配置，此处仅按 routing_key 发布
        channel.basic_publish(
            exchange=exchange,
            routing_key=routing_key,
            body=body,
            properties=pika.BasicProperties(
                content_type="application/json",
                delivery_mode=2,
            ),
        )
        # 推送成功后再记录，失败时可在下次告警重试
        _last_mq_enter_count_by_scene[scene_id] = current
        logger.info(
            "MQ 推送成功 exchange=%s routing_key=%s mineCode=%s cameraCode=%s "
            "analysisCase=%s dataTime=%s prev_enter_count=%s",
            exchange,
            routing_key,
            payload.get("mineCode"),
            payload.get("cameraCode"),
            payload.get("analysisCase"),
            payload.get("dataTime"),
            prev,
        )
        return payload
    except Exception:
        logger.exception(
            "MQ 推送失败 exchange=%s routing_key=%s host=%s:%s",
            exchange,
            routing_key,
            settings.EXTERNAL_RABBITMQ_HOST,
            settings.EXTERNAL_RABBITMQ_PORT,
        )
        return None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _frame_to_jpeg_base64(frame: Any) -> Optional[str]:
    """把告警帧编码为 JPEG Base64（带 data URL 前缀）。失败返回 None。"""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    try:
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        logger.exception("视频质量异常转发：告警帧 JPEG 编码失败")
        return None


def _detect_video_anomaly_cases(event: Dict[str, Any]) -> List[Tuple[str, str]]:
    """从告警事件里提取视频质量异常 (analysisCase, 中文名)，按 case 去重排序。"""
    hits: Dict[str, str] = {}

    # ① 布尔信号
    for flag in (
        "has_dark_alarm",
        "has_overexp_alarm",
        "has_blur_alarm",
        "has_zhedang_alarm",
        "has_camera_shift",
        "has_camera_tilt",
    ):
        if event.get(flag):
            mapped = VIDEO_ANOMALY_SIGNAL_CASE.get(flag)
            if mapped:
                case, label = mapped
                hits.setdefault(case, label)

    skill_name = str(event.get("skill_name") or "").strip()
    skip_shift_codes = skill_name in _BOARDING_SKILL_NAMES

    # ② recognition_types（如 "dark" / "overexp" / "08" / "09"）
    types = event.get("recognition_types")
    if isinstance(types, list):
        for t in types:
            key = str(t or "").strip()
            if not key:
                continue
            if skip_shift_codes and key in {"08", "09"}:
                continue
            if key in VIDEO_ANOMALY_SIGNAL_CASE:
                case, label = VIDEO_ANOMALY_SIGNAL_CASE[key]
            elif f"recognition_type:{key}" in VIDEO_ANOMALY_SIGNAL_CASE:
                case, label = VIDEO_ANOMALY_SIGNAL_CASE[f"recognition_type:{key}"]
            else:
                continue
            hits.setdefault(case, label)

    # ③ active_alerts 上的 analysis_case / recognition_type
    alerts = event.get("active_alerts")
    if isinstance(alerts, list):
        for item in alerts:
            if not isinstance(item, dict):
                continue
            ac = str(item.get("analysis_case") or "").strip()
            if ac and len(ac) == 4 and ac.isdigit():
                hits.setdefault(
                    ac,
                    str(item.get("description") or item.get("violation_type") or ac),
                )
                continue
            rt = str(item.get("recognition_type") or "").strip()
            if not rt:
                continue
            if skip_shift_codes and rt in {"08", "09"}:
                continue
            if rt in VIDEO_ANOMALY_SIGNAL_CASE:
                case, label = VIDEO_ANOMALY_SIGNAL_CASE[rt]
            elif f"recognition_type:{rt}" in VIDEO_ANOMALY_SIGNAL_CASE:
                case, label = VIDEO_ANOMALY_SIGNAL_CASE[f"recognition_type:{rt}"]
            else:
                continue
            hits.setdefault(case, label)

    return sorted(hits.items())


# ===========================================================================
# 【对接文档 §6】视频质量异常数据-推送
#   接口：POST /mine/ai/videoAnomaly     分析类型 analysisType 固定 03
#   识别结果 analysisCase：0001 画面遮挡 / 0002 摄像仪挪动 / 0003 画面过暗 /
#     0004 画面过曝 / 0005 图像模糊 / 0006 画面冻结 / 0007 视频丢失 /
#     0008 画面抖动 / 0009 分辨率异常
#   实现：app/services/coal_video_anomaly_push.py
#   调用位置：default_alert_handler()；巡检路径见 snapshot_patrol._emit_alert()
#   触发信号 -> analysisCase 映射见下方 VIDEO_ANOMALY_SIGNAL_CASE
# ===========================================================================
def forward_video_anomaly_alerts(
    event: Dict[str, Any],
    raw_frame: Any = None,
    *,
    scene_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """把告警里的视频质量异常转发到煤安平台 ``POST /mine/ai/videoAnomaly``。

    - ``analysisType`` 优先取摄像头配置（通常为 ``03``），缺省固定 ``03``；
    - ``cameraCode`` / ``mineCode`` 来自摄像头配置（事件字段或按 scene 反查）；
    - ``analysisCase`` 由告警信号映射（见 VIDEO_ANOMALY_SIGNAL_CASE）；
    - 证据图片取本次告警帧（与告警图片同一帧），为空则不传 ``imagesBase64``；
    - 一次告警可命中多个异常类型，合并为一个 JSON 数组提交；
    - 有冷却时间，避免告警循环内重复刷平台；
    - 全部失败只记日志，不影响原有告警流程。
    """
    scene = str(scene_id or event.get("scene_id") or "").strip()

    if not bool(getattr(settings, "COAL_VIDEO_ANOMALY_FORWARD_ENABLED", True)):
        return None

    hits = _detect_video_anomaly_cases(event)
    if not hits:
        return None

    mine_code, camera_code, analysis_type = _resolve_mine_camera_codes(event)
    if not camera_code:
        logger.warning(
            "视频质量异常转发：cameraCode 为空，跳过 scene=%s cases=%s "
            "（请确认任务关联摄像头已配置摄像仪编码，或重启任务以注入编码）",
            scene,
            [c for c, _ in hits],
        )
        return None

    # 视频质量异常接口识别类型固定 03；若摄像头配置了 analysis_type 则优先采用
    analysis_type = (analysis_type or "03").strip() or "03"
    if analysis_type != "03":
        logger.info(
            "视频质量异常转发：摄像头 analysis_type=%s，仍按接口要求使用 03 "
            "scene=%s cameraCode=%s",
            analysis_type,
            scene,
            camera_code,
        )
        analysis_type = "03"

    data_time = str(event.get("time") or "").strip() or datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # 冷却过滤：同一摄像仪同一异常类型在冷却期内不重复转发
    now = time.time()
    cooldown = _video_anomaly_cooldown()
    pending: List[Tuple[str, str]] = []
    for case, label in hits:
        last = _last_video_anomaly_forward_at.get((camera_code, case))
        if cooldown > 0 and last is not None and (now - last) < cooldown:
            logger.info(
                "视频质量异常转发：冷却中跳过 scene=%s cameraCode=%s case=%s(%s) 距上次 %.0fs",
                scene,
                camera_code,
                case,
                label,
                now - last,
            )
            continue
        pending.append((case, label))
    if not pending:
        return None

    # 证据图片：优先告警帧；若无帧则尝试事件里已上传的 pic_url 不再二次编码
    images_base64: List[str] = []
    image_b64 = _frame_to_jpeg_base64(raw_frame)
    if image_b64:
        images_base64 = [image_b64]
    else:
        logger.warning(
            "视频质量异常转发：无可用告警帧，仅推送数据 scene=%s cameraCode=%s",
            scene,
            camera_code,
        )

    records: List[Dict[str, Any]] = [
        {
            "cameraCode": camera_code,
            "analysisType": analysis_type,
            "analysisCase": case,
            "dataTime": data_time,
            "imagesBase64": images_base64,
            "mineCode": mine_code or None,
        }
        for case, _ in pending
    ]

    logger.info(
        "视频质量异常转发：准备推送 scene=%s cameraCode=%s mineCode=%s "
        "analysisType=%s dataTime=%s cases=%s images=%s",
        scene,
        camera_code,
        mine_code,
        analysis_type,
        data_time,
        [c for c, _ in pending],
        len(images_base64),
    )

    try:
        from app.services.coal_video_anomaly_push import push_video_anomaly_records

        result = push_video_anomaly_records(records)
    except Exception:
        logger.exception(
            "视频质量异常转发失败 scene=%s cameraCode=%s cases=%s",
            scene,
            camera_code,
            [c for c, _ in pending],
        )
        return None

    # 仅成功（平台 code=200）才记录冷却时间；失败允许下次告警重试
    if result and result.get("code") == 200:
        for case, _ in pending:
            _last_video_anomaly_forward_at[(camera_code, case)] = now

    return result


def _resolve_position_info(
    event: Dict[str, Any],
    camera_code: str,
    scene: str,
) -> Tuple[str, str, str]:
    """解析 importPersonCount 需要的 (positionName, positionCode, direction)。

    取值优先级：
    - positionName：事件字段 > 配置 COAL_IMPORT_PERSON_COUNT_POSITION_NAME
                    > 摄像仪 position_desc
    - positionCode：事件字段 > 配置 COAL_IMPORT_PERSON_COUNT_POSITION_CODE
    - direction：   事件字段 > 配置 COAL_IMPORT_PERSON_COUNT_DIRECTION
                    > 摄像仪 analysis_type 映射（01->01 入井，02->02 出井）

    即：**显式配置优先于摄像仪自动取值**（配置了就以配置为准），
    摄像仪字段仅作为未配置时的兜底。

    本项目库里没有 14 位位置编码，positionCode 为空时必须显式配置，
    否则调用方应跳过推送（不猜、不编，避免把数据挂到错误的点位上）。
    """
    position_name = str(event.get("position_name") or "").strip()
    position_code = str(event.get("position_code") or "").strip()
    direction = str(event.get("direction") or "").strip()

    # 事件字段优先；缺失时用配置覆盖摄像仪自动取值
    if not position_code:
        position_code = str(
            getattr(settings, "COAL_IMPORT_PERSON_COUNT_POSITION_CODE", "") or ""
        ).strip()
    if not position_name:
        position_name = str(
            getattr(settings, "COAL_IMPORT_PERSON_COUNT_POSITION_NAME", "") or ""
        ).strip()
    if not direction:
        direction = str(
            getattr(settings, "COAL_IMPORT_PERSON_COUNT_DIRECTION", "") or ""
        ).strip()

    # 仍需兜底时才反查摄像仪（position_desc / analysis_type）
    if not position_name or not direction:
        looked = _lookup_camera_by_scene(scene) if scene else None
        if looked:
            if not position_name:
                position_name = str(looked.get("position_desc") or "").strip()
            if not direction:
                analysis_type = str(looked.get("analysis_type") or "").strip()
                direction = _ANALYSIS_TYPE_TO_DIRECTION.get(analysis_type, "")

    # 兜底：位置名缺失时用点位编码/摄像仪编码，保证必填字段有值
    if not position_name:
        position_name = position_code or camera_code

    return position_name, position_code, direction


# ===========================================================================
# 【对接文档 §9】重要运输设备实时人数-推送
#   接口：POST /mine/ai/importPersonCount
#   推送重要运输设备（架空乘人装置、罐笼、无轨胶轮车、电机车、单轨吊等）
#   监控点位的实时人数；按 positionCode 先删除后新增（全量替换），
#   同一监控点位可在一次请求内推送入井、出井两条（direction 不同、
#   cameraCode 不同）；平台落库后经 SSE（事件名 importPersonCount）
#   推给前端大屏，历史查询 GET /mine/importPersonCount/list
#   入参：positionName / positionCode(14位) / cameraCode / direction(01入井,02出井)
#     / personCount / mineCode(可选)
#   实现：app/services/coal_import_person_count_push.py
#   调用位置：default_alert_handler()
#   数据来源：画面人数检测技能 person_presence_detector26（personCount 取事件 count）
# ===========================================================================
def extract_person_count(event: Dict[str, Any]) -> Optional[int]:
    """从事件中提取「监控点位实时人数」。

    按技能取对应字段（priority 顺序），避免误用异常检测类技能里
    「是否告警」的 0/1 count：
    - person_presence_detector26 / person_count_detector*: ``count``
      （= 当前画面人数）、``flow_metrics.current_person_count``
    - boarding_detector / non_fixed_parking_boarding_detector:
      ``count``（= 当前车上人数）、``flow_metrics.current_boarding``、
      ``enter_count``

    返回 None 表示该事件没有可用的人数（不推送）。
    """
    skill_name = str(event.get("skill_name") or "").strip()
    fields = _SKILL_COUNT_FIELDS.get(skill_name, ("count",))
    for field in fields:
        value: Any = event
        for part in field.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def forward_import_person_count(
    event: Dict[str, Any],
    *,
    scene_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """把「监控点位实时人数」转发到煤安平台 ``POST /mine/ai/importPersonCount``（文档 9）。

    适用于所有 count 语义为「真实人数」的技能（见 ``_PERSON_COUNT_SKILLS``）：
    - ``personCount`` 由 ``extract_person_count()`` 按技能取对应字段；
    - ``positionName`` / ``positionCode`` / ``direction`` 由 ``_resolve_position_info`` 解析；
    - **positionCode 必须为 14 位数字**，否则跳过并记日志（缺必填字段不推送，
      避免平台按 positionCode 全量替换时把数据挂到错误点位上）；
    - 数据落库后平台经 SSE（事件名 ``importPersonCount``）推给前端；
    - 全部失败只记日志，不影响原有告警流程。

    注意：异常检测类技能（guoan/guobao/mohu/camera_shift/camera_tilt）
    的 count 是「是否告警」的 0/1 标志，不是人数，已排除，不在此转发。
    """
    scene = str(scene_id or event.get("scene_id") or "").strip()

    if not bool(
        getattr(settings, "COAL_IMPORT_PERSON_COUNT_FORWARD_ENABLED", True)
    ):
        return None

    skill_name = str(event.get("skill_name") or "").strip()
    if skill_name not in _PERSON_COUNT_SKILLS:
        # 非人数类技能不在此接口转发
        return None

    mine_code, camera_code, _analysis_type = _resolve_mine_camera_codes(event)
    if not camera_code:
        logger.warning(
            "重要运输设备人数转发：cameraCode 为空，跳过 scene=%s", scene
        )
        return None

    person_count = extract_person_count(event)
    if person_count is None:
        logger.warning(
            "重要运输设备人数转发：拿不到人数（skill=%s）scene=%s → 跳过",
            skill_name,
            scene,
        )
        return None

    position_name, position_code, direction = _resolve_position_info(
        event, camera_code, scene
    )
    if not _POSITION_CODE_RE.match(position_code):
        logger.warning(
            "重要运输设备人数转发：positionCode 缺失或非 14 位数字 "
            "（当前=%r）scene=%s cameraCode=%s → 跳过推送。"
            "请在 .env 配置 COAL_IMPORT_PERSON_COUNT_POSITION_CODE",
            position_code,
            scene,
            camera_code,
        )
        return None
    if direction not in (_DIRECTION_IN, _DIRECTION_OUT):
        logger.warning(
            "重要运输设备人数转发：direction 缺失或非法（当前=%r）"
            "scene=%s cameraCode=%s → 跳过推送。"
            "请配置 COAL_IMPORT_PERSON_COUNT_DIRECTION 或摄像仪 analysis_type",
            direction,
            scene,
            camera_code,
        )
        return None

    record: Dict[str, Any] = {
        "positionName": position_name,
        "positionCode": position_code,
        "cameraCode": camera_code,
        "direction": direction,
        "personCount": person_count,
        "mineCode": mine_code or None,
    }

    logger.info(
        "重要运输设备人数转发：准备推送 scene=%s positionName=%s positionCode=%s "
        "cameraCode=%s direction=%s personCount=%s mineCode=%s",
        scene,
        position_name,
        position_code,
        camera_code,
        direction,
        person_count,
        mine_code,
    )

    try:
        from app.services.coal_import_person_count_push import (
            push_import_person_count_records,
        )

        result = push_import_person_count_records([record])
    except Exception:
        logger.exception(
            "重要运输设备人数转发失败 scene=%s positionCode=%s", scene, position_code
        )
        return None

    if result and result.get("code") == 200:
        logger.info(
            "重要运输设备人数转发成功 scene=%s positionCode=%s personCount=%s",
            scene,
            position_code,
            person_count,
        )
    return result


def collect_and_forward_person_counts(
    event: Dict[str, Any],
    *,
    scene_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """收集本次事件里的「监控点位实时人数」并转发到 importPersonCount（文档 9）。

    与 ``forward_import_person_count`` 等价，但语义上强调「收集多个技能的人数」：
    从所有 count 语义为真实人数的技能中取出人数，组装成 **JSON 数组**
    一次提交（同一监控点位可含入井/出井两条，direction 不同、cameraCode 不同）。

    返回平台响应或 None（无可用人数 / 缺必填字段 / 未启用）。
    """
    result = forward_import_person_count(event, scene_id=scene_id)
    if result is None:
        skill_name = str(event.get("skill_name") or "").strip()
        if skill_name in _PERSON_COUNT_SKILLS:
            logger.info(
                "重要运输设备人数转发：本次无可用人数或缺少必填字段，未推送 "
                "scene=%s skill=%s count=%s",
                scene_id or event.get("scene_id"),
                skill_name,
                extract_person_count(event),
            )
    return result


def forward_underground_count(
    event: Dict[str, Any],
    *,
    scene_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """把「井下总人数」转发到煤安平台 ``POST /mine/ai/undergroundCount``（文档 8）。

    聚合语义：
    - 维护**各摄像仪的最新人数**（``_UNDERGROUND_CAMERA_COUNTS``），
      每次人数事件都刷新对应摄像仪的值；
    - ``analysisCase`` = **所有摄像仪人数之和**（井下总人数，4 位补零）；
    - ``cameraCode`` = **本次检测到人数变化的那台摄像仪**（哪台变化带哪台）；
    - **仅当总人数发生变化时才推送**（首次推送也发）；

    ``dataTime`` 取事件时间，缺省当前时间；``mineCode`` 由
    ``_resolve_mine_camera_codes()`` 解析。平台侧处理（本项目只负责推送）：
    写入 ``mine_underground_count``、调人员定位系统取 ``ps_person_card_count``、
    按冗余规则产生 ``analysis_type=12`` 报警、SSE（事件名 ``undergroundCount``）。
    失败只记日志，不影响原有告警流程。
    """
    global _UNDERGROUND_LAST_TOTAL
    scene = str(scene_id or event.get("scene_id") or "").strip()

    if not bool(getattr(settings, "COAL_UNDERGROUND_COUNT_FORWARD_ENABLED", True)):
        return None

    skill_name = str(event.get("skill_name") or "").strip()
    if skill_name not in _UNDERGROUND_SKILLS:
        # 非人数类技能不在此接口转发
        return None

    mine_code, camera_code, _analysis_type = _resolve_mine_camera_codes(event)
    if not camera_code:
        logger.warning("井下人数转发：cameraCode 为空，跳过 scene=%s", scene)
        return None

    person_count = extract_person_count(event)
    if person_count is None:
        logger.warning(
            "井下人数转发：拿不到人数（skill=%s）scene=%s → 跳过", skill_name, scene
        )
        return None

    # 刷新本摄像仪人数，并计算井下总人数（各摄像仪之和）
    prev_camera_count = _UNDERGROUND_CAMERA_COUNTS.get(camera_code)
    _UNDERGROUND_CAMERA_COUNTS[camera_code] = person_count
    total_count = sum(_UNDERGROUND_CAMERA_COUNTS.values())

    # 仅总人数发生变化时上报（首次也上报）
    prev_total = _UNDERGROUND_LAST_TOTAL
    if prev_total is not None and prev_total == total_count:
        logger.debug(
            "井下人数转发：总人数未变化（%s 人），跳过 cameraCode=%s（本机 %s 人）",
            total_count,
            camera_code,
            person_count,
        )
        return None

    data_time = str(event.get("time") or "").strip() or datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    logger.info(
        "井下人数转发：准备推送 scene=%s cameraCode=%s（变化来源）"
        "总人数=%s（上次=%s）本机人数=%s（上次=%s）参与摄像仪=%s mineCode=%s",
        scene,
        camera_code,
        total_count,
        prev_total,
        person_count,
        prev_camera_count,
        len(_UNDERGROUND_CAMERA_COUNTS),
        mine_code,
    )

    try:
        from app.services.coal_underground_count_push import (
            push_underground_count_records,
        )

        result = push_underground_count_records(
            [
                {
                    "cameraCode": camera_code,
                    "analysisCase": f"{total_count:04d}",
                    "dataTime": data_time,
                    "mineCode": mine_code or None,
                }
            ]
        )
    except Exception:
        logger.exception(
            "井下人数转发失败 scene=%s cameraCode=%s total=%s",
            scene,
            camera_code,
            total_count,
        )
        return None

    # 仅成功才记录，失败允许下次重试
    if result and result.get("code") == 200:
        _UNDERGROUND_LAST_TOTAL = total_count
        logger.info(
            "井下人数转发成功 scene=%s cameraCode=%s 总人数=%s（上次=%s）",
            scene,
            camera_code,
            total_count,
            prev_total,
        )
    return result


def forward_camera_status(
    event: Dict[str, Any],
    camera_status: Any = _CAMERA_STATUS_ONLINE,
    *,
    scene_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """把摄像仪在线/离线状态转发到煤安平台 ``POST /mine/ai/cameraStatus``（文档 5）。

    - ``cameraStatus``：``1``=在线，``0``=离线；
    - ``cameraCode`` / ``mineCode`` 由 ``_resolve_mine_camera_codes()`` 解析；
    - 同一 cameraCode 在 ``_CAMERA_STATUS_FORWARD_MIN_INTERVAL`` 秒内不重复推送
      （状态未变化且未超间隔时跳过），避免每条告警都刷平台；
    - 平台按 ``cameraCode`` 先删除后新增（全量替换），故单次请求内不重复；
    - 失败只记日志，不影响原有告警流程。
    """
    scene = str(scene_id or event.get("scene_id") or "").strip()

    if not bool(getattr(settings, "COAL_CAMERA_STATUS_PUSH_ENABLED", True)):
        return None

    mine_code, camera_code, _analysis_type = _resolve_mine_camera_codes(event)
    if not camera_code:
        logger.warning("摄像仪状态转发：cameraCode 为空，跳过 scene=%s", scene)
        return None

    from app.services.coal_camera_status_push import normalize_camera_status

    status = normalize_camera_status(camera_status)
    if status is None:
        logger.warning(
            "摄像仪状态转发：cameraStatus=%r 非法（应为 0/1）scene=%s cameraCode=%s",
            camera_status,
            scene,
            camera_code,
        )
        return None

    # 节流：同状态且在最小间隔内则跳过
    now = time.time()
    last = _last_camera_status_forward.get(camera_code)
    if last is not None:
        last_at, last_status = last
        if (
            last_status == status
            and (now - last_at) < _CAMERA_STATUS_FORWARD_MIN_INTERVAL
        ):
            logger.debug(
                "摄像仪状态转发：节流跳过 cameraCode=%s status=%s 距上次 %.0fs",
                camera_code,
                status,
                now - last_at,
            )
            return None

    data_time = str(event.get("time") or "").strip() or datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    logger.info(
        "摄像仪状态转发：准备推送 scene=%s cameraCode=%s cameraStatus=%s mineCode=%s",
        scene,
        camera_code,
        status,
        mine_code,
    )

    try:
        from app.services.coal_camera_status_push import push_camera_statuses

        result = push_camera_statuses(
            [
                {
                    "cameraCode": camera_code,
                    "cameraStatus": status,
                    "dataTime": data_time,
                    "mineCode": mine_code or None,
                }
            ]
        )
    except Exception:
        logger.exception(
            "摄像仪状态转发失败 scene=%s cameraCode=%s", scene, camera_code
        )
        return None

    # 仅成功才记录节流时间；失败允许下次重试
    if result and result.get("code") == 200:
        _last_camera_status_forward[camera_code] = (now, status)
        logger.info(
            "摄像仪状态转发成功 scene=%s cameraCode=%s status=%s",
            scene,
            camera_code,
            status,
        )
    return result


def push_all_camera_status_periodic() -> Optional[Dict[str, Any]]:
    """定时全量推送所有摄像仪在线状态（ZLM 判在线，文档 5）。

    供 5 分钟定时器调用；采集失败返回 None 且不推送。
    """
    try:
        from app.services.coal_camera_status_push import push_all_camera_statuses

        result = push_all_camera_statuses()
    except Exception:
        logger.exception("摄像仪状态定时推送异常")
        return None
    if result:
        logger.info(
            "摄像仪状态定时推送结果 online=%s offline=%s pushed=%s code=%s",
            result.get("online"),
            result.get("offline"),
            result.get("pushed"),
            result.get("code"),
        )
    return result


def _camera_status_periodic_loop(interval: float) -> None:
    """后台线程：每 interval 秒全量推送一次摄像仪状态。"""
    logger.info("摄像仪状态定时推送线程已启动：每 %.0f 秒一次", interval)
    while True:
        time.sleep(max(interval, 30.0))
        try:
            push_all_camera_status_periodic()
        except Exception:
            logger.exception("摄像仪状态定时推送循环异常")


def start_camera_status_timer(interval_seconds: Optional[float] = None) -> bool:
    """启动摄像仪状态定时推送（默认每 5 分钟）。

    优先注册到项目已有的 APScheduler（``task_scheduler``，主进程）；
    APScheduler 不可用时回退到本进程后台线程。幂等：重复调用只启动一次。

    注意：ingest 是多进程的，若每个子进程都启线程会重复推送，
    因此仅建议在主进程（``start_scheduler``）中调用本函数。
    """
    global _camera_status_timer_started
    if _camera_status_timer_started:
        return False
    if not bool(getattr(settings, "COAL_CAMERA_STATUS_PERIODIC_ENABLED", True)):
        logger.info("摄像仪状态定时推送未启用：COAL_CAMERA_STATUS_PERIODIC_ENABLED=false")
        return False

    if interval_seconds is None:
        try:
            interval_seconds = float(
                getattr(settings, "COAL_CAMERA_STATUS_PERIODIC_INTERVAL", 300.0) or 300.0
            )
        except (TypeError, ValueError):
            interval_seconds = 300.0
    interval_seconds = max(float(interval_seconds), 30.0)

    # 优先复用项目 APScheduler，避免多进程重复起线程
    try:
        from app.services.task_scheduler import get_scheduler

        sched = get_scheduler()
        if sched is not None and sched.running:
            sched.add_job(
                push_all_camera_status_periodic,
                "interval",
                seconds=interval_seconds,
                id="push_camera_status",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            _camera_status_timer_started = True
            logger.info(
                "摄像仪状态定时推送已注册到 APScheduler：每 %.0f 秒一次",
                interval_seconds,
            )
            return True
    except Exception:
        logger.exception("注册 APScheduler 摄像仪状态任务失败，回退后台线程")

    t = threading.Thread(
        target=_camera_status_periodic_loop,
        args=(interval_seconds,),
        name="camera-status-periodic",
        daemon=True,
    )
    t.start()
    _camera_status_timer_started = True
    logger.info("摄像仪状态定时推送线程已启动：每 %.0f 秒一次", interval_seconds)
    return True


def default_alert_handler(data, raw_frame, scene_id):
    event = build_person_count_alert(data, raw_frame, scene_id)
    if not event.get("skill_name"):
        event["skill_name"] = str(data.get("skill_name") or "")
    if data.get("has_person_count_change"):
        event["has_person_count_change"] = True
        # 画面人数变化时补充更直观的消息前缀
        if event.get("skill_name") == "person_presence_detector26":
            event["message"] = (
                f"画面人数变化 time={event.get('time')} scene={scene_id} "
                f"count={event.get('count')} skill=person_presence_detector26"
            )
    logger.info("%s", event["message"])

    # 管理台按类型拆分落库：01/02/08 上车人数→事件；04-07→报警
    persisted_rows = []
    try:
        from app.services.mgmt_service import persist_classified_records

        persisted_rows = persist_classified_records(event) or []
    except Exception:
        logger.exception("告警/事件落库调用失败 scene=%s", scene_id)

    # 标准 Webhook：按接入方订阅过滤后异步推送轻量通知
    if persisted_rows:
        try:
            from app.services.webhook_dispatch import schedule_alert_webhooks

            schedule_alert_webhooks(persisted_rows, event_type="alert.created")
        except Exception:
            logger.exception("Webhook 调度失败 scene=%s", scene_id)

    # 将识别结果作为 JSON 数组转发到煤安平台 /mine/ai/countingRecog（文档 7，含证据图片）
    payloads = push_counting_recog_upload(event, raw_frame)
    # 同帧可能有多种识别类型（如绕行 04 + 闸机翻越 06），会合并成一个数组一次提交，
    # 此处仅逐条打印字段，便于排查
    for payload in payloads:
        logger.info(
            "识别结果上传字段 mineCode=%s cameraCode=%s analysisType=%s "
            "analysisCase=%s dataTime=%s",
            payload.get("mineCode"),
            payload.get("cameraCode"),
            payload.get("analysisType"),
            payload.get("analysisCase"),
            payload.get("dataTime"),
        )

    # enter_count 相对上次推送值有变化时，才推送到 RabbitMQ
    mq_payload = push_enter_count_to_mq(event)
    if mq_payload:
        logger.info(
            "MQ 推送字段 mineCode=%s cameraCode=%s analysisCase=%s dataTime=%s",
            mq_payload.get("mineCode"),
            mq_payload.get("cameraCode"),
            mq_payload.get("analysisCase"),
            mq_payload.get("dataTime"),
        )

    # 视频质量异常（画面过暗/摄像仪挪动等）转发煤安平台 /mine/ai/videoAnomaly
    # 带本次告警帧作为证据图片；内部有冷却，失败不影响后续流程
    anomaly_result = forward_video_anomaly_alerts(event, raw_frame, scene_id=scene_id)
    if anomaly_result:
        logger.info(
            "视频质量异常转发结果 scene=%s pushed=%s uploaded=%s code=%s msg=%s",
            scene_id,
            anomaly_result.get("pushed"),
            anomaly_result.get("uploaded"),
            anomaly_result.get("code"),
            anomaly_result.get("msg"),
        )

    # 画面人数检测（person_presence_detector26）→ 重要运输设备实时人数
    # POST /mine/ai/importPersonCount；内部校验 positionCode/direction，
    # 缺必填字段时跳过，不影响其他推送
    # ── 监控点位实时人数收集与转发（对接文档 §9）────────────────────────
    # 从所有 count 语义为「真实人数」的技能（见 _PERSON_COUNT_SKILLS）收集人数，
    # 组装成 JSON 数组一次提交到 /mine/ai/importPersonCount；
    # personCount 由 extract_person_count() 按技能取对应字段，
    # positionName/positionCode/direction 由 _resolve_position_info() 解析，
    # 缺必填字段（positionCode 需 14 位、direction 需 01/02）则跳过并记日志。
    # 异常检测类技能（guoan/guobao/mohu/camera_shift/camera_tilt）的 count
    # 是告警标志而非人数，已在 _PERSON_COUNT_SKILLS 中排除。
    person_count_result = collect_and_forward_person_counts(event, scene_id=scene_id)
    if person_count_result:
        logger.info(
            "监控点位实时人数转发结果 scene=%s skill=%s personCount=%s "
            "pushed=%s code=%s msg=%s",
            scene_id,
            event.get("skill_name"),
            extract_person_count(event),
            person_count_result.get("pushed"),
            person_count_result.get("code"),
            person_count_result.get("msg"),
        )

    # ── 井下人数不符转发（对接文档 §8）────────────────────────────────
    # 该摄像仪人数发生变化时上报，cameraCode 带人数变化的那台摄像仪；
    # 平台侧据此与人员定位系统比对并产生 analysis_type=12 报警
    underground_result = forward_underground_count(event, scene_id=scene_id)
    if underground_result:
        logger.info(
            "井下人数转发结果 scene=%s cameraCode=%s personCount=%s code=%s msg=%s",
            scene_id,
            event.get("camera_code"),
            extract_person_count(event),
            underground_result.get("code"),
            underground_result.get("msg"),
        )

    # ── 摄像仪状态转发（对接文档 §5）────────────────────────────────────
    # 能产生告警说明该摄像仪在线，实时上报 cameraStatus=1
    # （同状态 60s 内节流；离线状态由 5 分钟定时全量任务按 ZLM 判定）
    camera_status_result = forward_camera_status(
        event, _CAMERA_STATUS_ONLINE, scene_id=scene_id
    )
    if camera_status_result:
        logger.info(
            "摄像仪状态转发结果 scene=%s cameraCode=%s code=%s msg=%s",
            scene_id,
            event.get("camera_code"),
            camera_status_result.get("code"),
            camera_status_result.get("msg"),
        )

    # 写入跨进程队列，供 SSE 订阅端推送告警
    _publish_alert(event)
    logger.info(
        "告警已写入 SSE 队列 scene=%s type=%s recognition_types=%s "
        "skill=%s count=%s enter_count=%s time=%s",
        event.get("scene_id"),
        event.get("type"),
        event.get("recognition_types"),
        event.get("skill_name"),
        event.get("count"),
        event.get("enter_count"),
        event.get("time"),
    )
