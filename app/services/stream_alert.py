"""推流任务默认告警处理"""
import base64
import json
import logging
import queue
import re
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
# HTTP 上传仅允许的识别类型（04/05 绕行，06/07 闸机反向翻越）
_HTTP_UPLOAD_ANALYSIS_TYPES = frozenset({"04", "05", "06", "07"})
# 各 scene 上次已成功推送 MQ 的 enter_count，用于判断是否变化
_last_mq_enter_count_by_scene: Dict[str, int] = {}

# ---------------------------------------------------------------------------
# 视频质量异常转发（煤安平台 ANALYSIS_TYPE=03）
# ---------------------------------------------------------------------------

# 告警信号 -> 平台 analysisCase 映射。
# 注意：这里的 "08"/"09" 是本项目内部对 camera_shift/camera_tilt 的识别类型编号，
# 与煤安平台 videoAnomaly 的 0008=画面抖动 / 0009=分辨率异常 语义不同，
# 因此不做直译，而是统一映射为 0002=摄像仪挪动。
VIDEO_ANOMALY_SIGNAL_CASE: Dict[str, Tuple[str, str]] = {
    "dark": (                            # guoan_detector_skill 输出的 recognition_type
        "0003",
        "画面过暗",
    ),
    "overexp": (                         # guobao_detector_skill 输出的 recognition_type
        "0004",
        "画面过曝",
    ),
    "has_dark_alarm": ("0003", "画面过暗"),
    "has_overexp_alarm": ("0004", "画面过曝"),
    "has_camera_shift": ("0002", "摄像仪挪动"),
    "has_camera_tilt": ("0002", "摄像仪挪动"),
    "recognition_type:08": ("0002", "摄像仪挪动（角度/位置偏移）"),
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
                for e in (enter_events + bypass_events + count_exit_events)
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
        "has_enter_count_change": has_enter_count_change,
        "has_bypass_violation": has_bypass_violation,
        "has_count_exit_violation": has_count_exit_violation,
        "has_dark_alarm": bool(data.get("has_dark_alarm")),
        "has_overexp_alarm": bool(data.get("has_overexp_alarm")),
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


def _resolve_mine_camera_codes(event: Dict[str, Any]) -> Tuple[str, str]:
    mine_code = str(event.get("mine_code") or "").strip() or str(
        settings.COUNTING_RECOG_MINE_CODE or ""
    ).strip()
    camera_code = str(event.get("camera_code") or "").strip() or str(
        settings.COUNTING_RECOG_CAMERA_CODE or ""
    ).strip()
    return mine_code, camera_code


def _count_current_frame_violators(event: Dict[str, Any]) -> Dict[str, int]:
    """
    统计当前帧各 recognition_type 的违规人数。

    仅依据本帧 bypass_events / count_exit_events，按 track_id 去重；
    不使用画面总人数 count，也不使用累计 enter_count。
    对应 04/05（绕行）、06/07（闸机反向翻越）的当次人数。
    """
    track_ids_by_type: Dict[str, set] = {}
    for key in ("bypass_events", "count_exit_events"):
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


def _build_counting_recog_payloads(event: Dict[str, Any]) -> List[Dict[str, str]]:
    mine_code, camera_code = _resolve_mine_camera_codes(event)
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

    # 当前帧各类型违规人数（非累计）；仅 04/05/06/07 走 HTTP 上传
    violator_counts = _count_current_frame_violators(event)
    payloads: List[Dict[str, str]] = []
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
            }
        )
    return payloads


def _post_counting_recog(payload: Dict[str, str]) -> None:
    url = str(settings.COUNTING_RECOG_UPLOAD_URL or "").strip()
    if not url:
        logger.warning("跳过识别结果上传：COUNTING_RECOG_UPLOAD_URL 未配置")
        return

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            req, timeout=settings.COUNTING_RECOG_UPLOAD_TIMEOUT
        ) as resp:
            resp_body = resp.read().decode("utf-8", errors="replace")
            logger.info(
                "识别结果上传成功 analysisType=%s analysisCase=%s status=%s body=%s",
                payload.get("analysisType"),
                payload.get("analysisCase"),
                getattr(resp, "status", None),
                resp_body[:200],
            )
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        logger.error(
            "识别结果上传 HTTP %s analysisType=%s body=%s",
            e.code,
            payload.get("analysisType"),
            detail[:300],
        )
    except urllib.error.URLError as e:
        logger.error(
            "无法连接识别结果上传接口 (%s): %s",
            url,
            e.reason,
        )
    except Exception:
        logger.exception(
            "识别结果上传异常 analysisType=%s",
            payload.get("analysisType"),
        )


def push_counting_recog_upload(event: Dict[str, Any]) -> List[Dict[str, str]]:
    """将本次告警按识别类型推送到 countingRecog/upload。"""
    payloads = _build_counting_recog_payloads(event)
    if not payloads:
        return []
    # 同帧可能有多种识别类型（如绕行 04 + 闸机翻越 06），每种 analysisType 各推送一次
    for payload in payloads:
        _post_counting_recog(payload)
    return payloads


def _build_enter_count_mq_payload(event: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    构建 MQ 过线人数消息。

    analysisCase 取累计 enter_count（4 位补零），与 HTTP 上传的当帧违规人数不同。
    """
    mine_code, camera_code = _resolve_mine_camera_codes(event)
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
    for flag in ("has_dark_alarm", "has_overexp_alarm", "has_camera_shift", "has_camera_tilt"):
        if event.get(flag):
            case, label = VIDEO_ANOMALY_SIGNAL_CASE[flag]
            hits.setdefault(case, label)

    # ② recognition_types（如 "dark"）
    types = event.get("recognition_types")
    if isinstance(types, list):
        for t in types:
            key = str(t or "").strip()
            if not key:
                continue
            if key in VIDEO_ANOMALY_SIGNAL_CASE:
                case, label = VIDEO_ANOMALY_SIGNAL_CASE[key]
            elif f"recognition_type:{key}" in VIDEO_ANOMALY_SIGNAL_CASE:
                case, label = VIDEO_ANOMALY_SIGNAL_CASE[f"recognition_type:{key}"]
            else:
                continue
            hits.setdefault(case, label)

    return sorted(hits.items())


def forward_video_anomaly_alerts(
    event: Dict[str, Any],
    raw_frame: Any = None,
    *,
    scene_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """把告警里的视频质量异常转发到煤安平台 ``POST /mine/ai/videoAnomaly``。

    - 识别类型固定 ``03``，``analysisCase`` 由告警信号映射（见 VIDEO_ANOMALY_SIGNAL_CASE）；
    - 证据图片取本次告警帧（与告警图片同一帧），为空则 ``localFiles`` 为 ``[]``；
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

    mine_code, camera_code = _resolve_mine_camera_codes(event)
    if not camera_code:
        logger.warning(
            "视频质量异常转发：cameraCode 为空，跳过 scene=%s cases=%s",
            scene,
            [c for c, _ in hits],
        )
        return None

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

    # 证据图片：本次告警帧 -> JPEG Base64（同一张图给本条记录所有异常类型）
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
            "analysisCase": case,
            "dataTime": data_time,
            "imagesBase64": images_base64,
            "mineCode": mine_code or None,
        }
        for case, _ in pending
    ]

    logger.info(
        "视频质量异常转发：准备推送 scene=%s cameraCode=%s mineCode=%s dataTime=%s "
        "cases=%s images=%s",
        scene,
        camera_code,
        mine_code,
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

    # 管理台按类型拆分落库：01/02/画面人数→事件；04-07→报警
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

    # 将识别结果 POST 到 countingRecog/upload 接口（仅 analysisType=04/05/06/07）
    payloads = push_counting_recog_upload(event)
    # 同帧可能有多种识别类型（如绕行 04 + 闸机翻越 06），recognition_types 为 ["04","06"]
    # 时会生成多条 payload，每种 analysisType 各一条，故循环分别上传与打印
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

    # 写入跨进程队列，供 SSE 订阅端推送告警
    _publish_alert(event)
    logger.info(
        "告警已写入 SSE 队列 scene=%s type=%s recognition_types=%s "
        "enter_count=%s time=%s",
        event.get("scene_id"),
        event.get("type"),
        event.get("recognition_types"),
        event.get("enter_count"),
        event.get("time"),
    )
