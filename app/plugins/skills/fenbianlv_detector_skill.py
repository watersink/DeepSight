"""
摄像头分辨率异常检测技能（有效画面 / 黑边 pillarbox / letterbox）

源自 LLM/camera_resolution；默认参数与 preview.py 当前配置一致。
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_CODE_ROOT = Path(__file__).resolve().parents[3]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cv2
import logging
import numpy as np

from app.skills.skill_base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

SKILL_NAME = "fenbianlv_detector_skill"
RECOGNITION_TYPE = "resolution_anomaly"
RECOGNITION_DESC = "分辨率异常"

_STATUS_COLOR = {
    "CALIBRATING": (180, 180, 180),
    "OK": (80, 200, 80),
    "CANDIDATE": (0, 200, 255),
    "ALARM": (0, 0, 230),
}

_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]

_REASON_ZH = {
    "warmup": "预热建基线",
    "ok": "正常",
    "invalid_frame": "无效帧",
    "black_bar_growth": "黑边变宽",
    "visible_area_shrink": "有效画面缩小",
    "frame_size_changed": "整帧尺寸变化",
}


# ---------------------------------------------------------------------------
# 分辨率 / 有效画面检测核心（LLM/camera_resolution）
# ---------------------------------------------------------------------------


def _frame_size(frame_bgr: np.ndarray) -> Tuple[int, int]:
    if frame_bgr is None or frame_bgr.size == 0:
        return 0, 0
    h, w = frame_bgr.shape[:2]
    return int(w), int(h)


def _size_label(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    return f"{width}x{height}"


def _content_label(x0: int, y0: int, x1: int, y1: int) -> str:
    w = max(0, x1 - x0)
    h = max(0, y1 - y0)
    return f"{w}x{h}"


def _detect_active_content_bbox(
    frame_bgr: np.ndarray,
    *,
    bar_luma_max: float = 22.0,
    bar_std_max: float = 12.0,
    min_bar_run: int = 6,
) -> Tuple[int, int, int, int, Dict[str, float]]:
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    col_mean = gray.mean(axis=0)
    col_std = gray.std(axis=0)
    row_mean = gray.mean(axis=1)
    row_std = gray.std(axis=1)

    def _is_bar(mean_v: float, std_v: float) -> bool:
        return mean_v <= bar_luma_max and std_v <= bar_std_max

    left = 0
    while left < w and _is_bar(col_mean[left], col_std[left]):
        left += 1

    right = w - 1
    while right >= left and _is_bar(col_mean[right], col_std[right]):
        right -= 1

    top = 0
    while top < h and _is_bar(row_mean[top], row_std[top]):
        top += 1

    bottom = h - 1
    while bottom >= top and _is_bar(row_mean[bottom], row_std[bottom]):
        bottom -= 1

    x0, y0 = left, top
    x1, y1 = right + 1, bottom + 1
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = 0, 0, w, h

    bar_left = float(x0)
    bar_right = float(w - x1)
    bar_top = float(y0)
    bar_bottom = float(h - y1)
    content_w = max(0, x1 - x0)
    content_h = max(0, y1 - y0)
    fill_ratio = (content_w * content_h) / max(1, w * h)

    meta: Dict[str, float] = {
        "bar_left_px": bar_left,
        "bar_right_px": bar_right,
        "bar_top_px": bar_top,
        "bar_bottom_px": bar_bottom,
        "content_width": float(content_w),
        "content_height": float(content_h),
        "fill_ratio": float(fill_ratio),
        "max_bar_px": float(max(bar_left, bar_right, bar_top, bar_bottom)),
    }
    if meta["max_bar_px"] < min_bar_run:
        x0, y0, x1, y1 = 0, 0, w, h
        meta.update(
            {
                "bar_left_px": 0.0,
                "bar_right_px": 0.0,
                "bar_top_px": 0.0,
                "bar_bottom_px": 0.0,
                "content_width": float(w),
                "content_height": float(h),
                "fill_ratio": 1.0,
                "max_bar_px": 0.0,
            }
        )
    return x0, y0, x1, y1, meta


class _ResolutionState(str, Enum):
    NORMAL = "normal"
    SUSPECT = "suspect"
    ANOMALY = "anomaly"


@dataclass
class _ContentBaseline:
    fill_ratio: float = 1.0
    bar_left: float = 0.0
    bar_right: float = 0.0
    bar_top: float = 0.0
    bar_bottom: float = 0.0
    content_width: float = 0.0
    content_height: float = 0.0
    frame_width: float = 0.0
    frame_height: float = 0.0


@dataclass
class _DetectorConfig:
    warmup_frames: int = 15
    suspect_frames_to_alarm: int = 3
    clear_frames_to_recover: int = 10
    bar_luma_max: float = 22.0
    bar_std_max: float = 12.0
    min_black_bar_px: int = 12
    fill_ratio_drop_min: float = 0.06
    bar_growth_px_min: float = 16.0
    bar_growth_ratio_min: float = 0.035
    check_frame_size_change: bool = False
    size_tolerance_px: int = 0


@dataclass
class ResolutionFrameResult:
    state: _ResolutionState
    is_anomaly: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "is_anomaly": self.is_anomaly,
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "metrics": {
                k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.metrics.items()
            },
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


class ResolutionDetector:
    def __init__(self, config: Optional[_DetectorConfig] = None) -> None:
        self.config = config or _DetectorConfig()
        self._baseline = _ContentBaseline()
        self._baseline_ready = False
        self._frame_index = 0
        self._suspect_streak = 0
        self._clear_streak = 0
        self._state = _ResolutionState.NORMAL
        self._warmup_meta: List[Dict[str, float]] = []

    def reset(self) -> None:
        self._baseline = _ContentBaseline()
        self._baseline_ready = False
        self._frame_index = 0
        self._suspect_streak = 0
        self._clear_streak = 0
        self._state = _ResolutionState.NORMAL
        self._warmup_meta.clear()

    def _analyze_frame(self, frame_bgr: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, float]]:
        cfg = self.config
        fw, fh = _frame_size(frame_bgr)
        x0, y0, x1, y1, meta = _detect_active_content_bbox(
            frame_bgr,
            bar_luma_max=cfg.bar_luma_max,
            bar_std_max=cfg.bar_std_max,
            min_bar_run=cfg.min_black_bar_px,
        )
        metrics: Dict[str, Any] = {
            "width": fw,
            "height": fh,
            "size_label": _size_label(fw, fh),
            "content_x0": x0,
            "content_y0": y0,
            "content_x1": x1,
            "content_y1": y1,
            "content_label": _content_label(x0, y0, x1, y1),
            **meta,
        }
        return metrics, meta

    def _finalize_baseline(self) -> None:
        if not self._warmup_meta:
            return
        arr = self._warmup_meta
        self._baseline.fill_ratio = float(np.median([m["fill_ratio"] for m in arr]))
        self._baseline.bar_left = float(np.median([m["bar_left_px"] for m in arr]))
        self._baseline.bar_right = float(np.median([m["bar_right_px"] for m in arr]))
        self._baseline.bar_top = float(np.median([m["bar_top_px"] for m in arr]))
        self._baseline.bar_bottom = float(np.median([m["bar_bottom_px"] for m in arr]))
        self._baseline.content_width = float(np.median([m["content_width"] for m in arr]))
        self._baseline.content_height = float(np.median([m["content_height"] for m in arr]))
        self._baseline.frame_width = float(np.median([m.get("frame_width", 0) for m in arr]))
        self._baseline.frame_height = float(np.median([m.get("frame_height", 0) for m in arr]))
        self._baseline_ready = True

    def _evaluate_vs_baseline(
        self, meta: Dict[str, float], fw: int, fh: int
    ) -> Tuple[bool, float, List[str], Dict[str, float]]:
        cfg = self.config
        scores: Dict[str, float] = {}
        reasons: List[str] = []
        b = self._baseline

        fill_drop = b.fill_ratio - meta["fill_ratio"]
        scores["baseline_fill_ratio"] = b.fill_ratio
        scores["fill_ratio_drop"] = fill_drop

        bar_growths = {
            "left": meta["bar_left_px"] - b.bar_left,
            "right": meta["bar_right_px"] - b.bar_right,
            "top": meta["bar_top_px"] - b.bar_top,
            "bottom": meta["bar_bottom_px"] - b.bar_bottom,
        }
        max_growth = max(bar_growths.values())
        scores["max_bar_growth_px"] = max_growth
        for side, g in bar_growths.items():
            scores[f"bar_growth_{side}"] = g

        frame_anomaly = False
        if cfg.check_frame_size_change and b.frame_width > 0:
            tol = cfg.size_tolerance_px
            if abs(fw - b.frame_width) > tol or abs(fh - b.frame_height) > tol:
                frame_anomaly = True
                reasons.append("frame_size_changed")

        bar_anomaly = False
        if max_growth >= cfg.bar_growth_px_min:
            bar_anomaly = True
            reasons.append("black_bar_growth")
        if fw > 0 and max_growth / fw >= cfg.bar_growth_ratio_min:
            bar_anomaly = True
            if "black_bar_growth" not in reasons:
                reasons.append("black_bar_growth")

        fill_anomaly = fill_drop >= cfg.fill_ratio_drop_min and meta["fill_ratio"] < b.fill_ratio - 1e-6
        if fill_anomaly:
            reasons.append("visible_area_shrink")

        is_suspect = bar_anomaly or fill_anomaly or frame_anomaly
        confidence = 0.0
        if is_suspect:
            conf_fill = (
                min(1.0, fill_drop / max(cfg.fill_ratio_drop_min, 1e-6)) if fill_anomaly else 0.0
            )
            conf_bar = min(1.0, max_growth / max(cfg.bar_growth_px_min, 1.0)) if bar_anomaly else 0.0
            confidence = max(0.55, min(1.0, max(conf_fill, conf_bar)))

        return is_suspect, confidence, reasons, scores

    def process(self, frame_bgr: np.ndarray) -> ResolutionFrameResult:
        cfg = self.config
        self._frame_index += 1
        metrics, meta = self._analyze_frame(frame_bgr)
        fw, fh = int(metrics["width"]), int(metrics["height"])
        meta["frame_width"] = float(fw)
        meta["frame_height"] = float(fh)

        if fw <= 0 or fh <= 0:
            return ResolutionFrameResult(
                state=self._state,
                is_anomaly=self._state == _ResolutionState.ANOMALY,
                confidence=0.0,
                reasons=["invalid_frame"],
                metrics=metrics,
                scores={"skipped": 1.0},
            )

        if self._frame_index <= cfg.warmup_frames:
            self._warmup_meta.append(meta)
            if self._frame_index == cfg.warmup_frames:
                self._finalize_baseline()
            metrics["baseline_fill_ratio"] = self._baseline.fill_ratio if self._baseline_ready else 1.0
            return ResolutionFrameResult(
                state=_ResolutionState.NORMAL,
                is_anomaly=False,
                confidence=0.0,
                reasons=["warmup"],
                metrics=metrics,
                scores={"warmup_frame": float(self._frame_index)},
            )

        if not self._baseline_ready:
            self._finalize_baseline()

        metrics["baseline_fill_ratio"] = self._baseline.fill_ratio
        metrics["baseline_content_label"] = _content_label(
            int(self._baseline.bar_left),
            int(self._baseline.bar_top),
            int(self._baseline.content_width + self._baseline.bar_left),
            int(self._baseline.content_height + self._baseline.bar_top),
        )

        frame_is_suspect, confidence, reasons, scores = self._evaluate_vs_baseline(meta, fw, fh)

        if frame_is_suspect:
            self._suspect_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._suspect_streak = max(0, self._suspect_streak - 1)

        if self._state != _ResolutionState.ANOMALY:
            if self._suspect_streak >= cfg.suspect_frames_to_alarm:
                self._state = _ResolutionState.ANOMALY
            elif self._suspect_streak >= 1:
                self._state = _ResolutionState.SUSPECT
            else:
                self._state = _ResolutionState.NORMAL

        if self._state == _ResolutionState.ANOMALY and self._clear_streak >= cfg.clear_frames_to_recover:
            self._state = _ResolutionState.NORMAL
            self._suspect_streak = 0

        is_anomaly = self._state == _ResolutionState.ANOMALY
        if not reasons:
            reasons = ["ok"]
        show_conf = is_anomaly or self._state == _ResolutionState.SUSPECT or frame_is_suspect
        return ResolutionFrameResult(
            state=self._state,
            is_anomaly=is_anomaly,
            confidence=confidence if show_conf else 0.0,
            reasons=reasons,
            metrics=metrics,
            scores=scores,
        )


def _map_skill_status(frame_result: ResolutionFrameResult) -> str:
    if "warmup" in frame_result.reasons:
        return "CALIBRATING"
    if frame_result.state == _ResolutionState.ANOMALY:
        return "ALARM"
    if frame_result.state == _ResolutionState.SUSPECT:
        return "CANDIDATE"
    return "OK"


def _status_message(status: str, frame_result: ResolutionFrameResult) -> str:
    m = frame_result.metrics
    cur = m.get("content_label") or m.get("size_label", "?")
    base = m.get("baseline_content_label") or cur
    fill = m.get("fill_ratio")
    fill_s = f"{fill:.0%}" if isinstance(fill, (int, float)) else "?"
    reasons = [_REASON_ZH.get(r, r) for r in frame_result.reasons if r != "ok"]
    tail = "，".join(reasons) if reasons else "有效画面正常"
    if status == "CALIBRATING":
        return f"分辨率检测预热中，帧 {m.get('size_label', '?')}"
    if status == "ALARM":
        return f"分辨率异常（黑边/可视区域缩小）：有效 {cur}，基准 {base}，填充 {fill_s}（{tail}）"
    if status == "CANDIDATE":
        return f"疑似有效画面缩小：有效 {cur}，基准 {base}（{tail}）"
    return f"有效画面正常：{cur}（填充 {fill_s}）"


class FenbianlvDetectorSkill(BaseSkill):
    """实时视频流分辨率（有效画面 / 黑边）异常检测。"""

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": SKILL_NAME,
        "name_zh": "分辨率异常检测",
        "version": "1.0",
        "cover_image": f"/skills/{SKILL_NAME}.png",
        "description": (
            "预热锁定正常有效画面区域，检测同分辨率下中途出现黑边（pillarbox/letterbox）"
            "或可视区域相对基准缩小。"
        ),
        "status": True,
        "run_mode": "stream",
        "required_models": [],
        "form_fields": [
            {
                "key": "confirm_count",
                "label": "连续确认帧数",
                "type": "number",
                "required": False,
                "default": 3,
            },
            {
                "key": "fill_ratio_drop_min",
                "label": "填充率下降阈值",
                "type": "number",
                "required": False,
                "default": 0.06,
                "hint": "相对预热基准的有效面积下降比例，如 0.06 表示 6%",
            },
            {
                "key": "bar_luma_max",
                "label": "黑边亮度上限",
                "type": "number",
                "required": False,
                "default": 22.0,
            },
            {
                "key": "cooldown_sec",
                "label": "告警冷却（秒）",
                "type": "number",
                "required": False,
                "default": 5,
            },
        ],
        "params": {
            "enable_default_sort_tracking": False,
            "enable_timing_log": False,
            "check_interval_sec": 2,
            "confirm_count": 3,
            "cooldown_sec": 5,
            "warmup_frames": 15,
            "clear_frames_to_recover": 10,
            "bar_luma_max": 22.0,
            "bar_std_max": 12.0,
            "min_black_bar_px": 12,
            "fill_ratio_drop_min": 0.06,
            "bar_growth_px_min": 16.0,
            "bar_growth_ratio_min": 0.035,
            "check_frame_size_change": False,
            "size_tolerance_px": 0,
        },
        "alert_definitions": [
            {
                "key": "resolution_alarm",
                "level": 2,
                "description": "有效画面相对基准缩小或出现黑边，并连续确认后触发。",
                "codes": {
                    "resolution_anomaly": {
                        "code": RECOGNITION_TYPE,
                        "description": RECOGNITION_DESC,
                        "analysis_case": "0009",
                    },
                },
            }
        ],
    }

    def _initialize(self) -> None:
        params = self.config.get("params", {}) if isinstance(self.config, dict) else {}
        confirm_count = max(1, int(params.get("confirm_count", 3) or 3))
        det_cfg = _DetectorConfig(
            warmup_frames=max(1, int(params.get("warmup_frames", 15) or 15)),
            suspect_frames_to_alarm=confirm_count,
            clear_frames_to_recover=max(1, int(params.get("clear_frames_to_recover", 10) or 10)),
            bar_luma_max=float(params.get("bar_luma_max", 22.0) or 22.0),
            bar_std_max=float(params.get("bar_std_max", 12.0) or 12.0),
            min_black_bar_px=max(1, int(params.get("min_black_bar_px", 12) or 12)),
            fill_ratio_drop_min=float(params.get("fill_ratio_drop_min", 0.06) or 0.06),
            bar_growth_px_min=float(params.get("bar_growth_px_min", 16.0) or 16.0),
            bar_growth_ratio_min=float(params.get("bar_growth_ratio_min", 0.035) or 0.035),
            check_frame_size_change=bool(params.get("check_frame_size_change", False)),
            size_tolerance_px=max(0, int(params.get("size_tolerance_px", 0) or 0)),
        )
        self._detector = ResolutionDetector(det_cfg)
        self.check_interval_sec = max(0.5, float(params.get("check_interval_sec", 2) or 2))
        self.confirm_count = confirm_count
        self.cooldown_sec = max(0.0, float(params.get("cooldown_sec", 5) or 5))
        self._repeat_sec = (
            self.cooldown_sec if self.cooldown_sec > 0 else max(1.0, self.check_interval_sec)
        )
        if self._repeat_sec >= 30:
            self._repeat_sec = max(5.0, self.check_interval_sec)

        self.alert_definitions: List[Dict[str, Any]] = list(
            self.config.get("alert_definitions") or []
        )
        self._last_process_ts: Optional[float] = None
        self._last_emit_ts: Optional[float] = None
        self._last_status: Optional[str] = None
        self._last_alert_ts = 0.0
        self._last_result: Dict[str, Any] = {}

        self.log(
            "info",
            f"初始化分辨率检测: interval={self.check_interval_sec}s confirm={confirm_count} "
            f"warmup={det_cfg.warmup_frames} fill_drop={det_cfg.fill_ratio_drop_min} "
            f"bar_luma={det_cfg.bar_luma_max}",
        )

    def get_required_models(self) -> List[str]:
        return []

    def detect_frame(self, current_bgr: np.ndarray) -> Dict[str, Any]:
        frame_result = self._detector.process(current_bgr)
        status = _map_skill_status(frame_result)
        message = _status_message(status, frame_result)

        now = time.time()
        prev_status = self._last_status
        has_status_change = prev_status is not None and status != prev_status
        alarm_ready = status == "ALARM" and frame_result.is_anomaly
        has_fenbianlv_alarm = bool(alarm_ready and (now - self._last_alert_ts >= self._repeat_sec))
        if has_fenbianlv_alarm:
            self._last_alert_ts = now
        if status != prev_status or has_fenbianlv_alarm:
            self.log(
                "info",
                f"分辨率检测: {status} emit={int(has_fenbianlv_alarm)} "
                f"conf={frame_result.confidence:.2f} {message}",
            )
            self._last_status = status

        result = self._build_result(
            frame_result,
            status=status,
            message=message,
            has_fenbianlv_alarm=has_fenbianlv_alarm,
            has_status_change=has_status_change,
        )
        result["run_mode"] = "stream"
        self._last_result = result
        return result

    def _build_result(
        self,
        frame_result: ResolutionFrameResult,
        *,
        status: str,
        message: str,
        has_fenbianlv_alarm: bool,
        has_status_change: bool,
    ) -> Dict[str, Any]:
        alert_def = self.alert_definitions[0] if self.alert_definitions else {}
        active_alerts: List[Dict[str, Any]] = []
        recognition_types: List[str] = []
        if has_fenbianlv_alarm:
            recognition_types = [RECOGNITION_TYPE]
            active_alerts.append(
                {
                    "key": alert_def.get("key", "resolution_alarm"),
                    "level": int(alert_def.get("level", 2) or 2),
                    "description": message,
                    "recognition_type": RECOGNITION_TYPE,
                    "analysis_case": "0009",
                    "violation_type": RECOGNITION_DESC,
                    "events": [
                        {
                            "status": status,
                            "message": message,
                            "confidence": frame_result.confidence,
                            "fill_ratio": frame_result.metrics.get("fill_ratio"),
                            "max_bar_growth_px": frame_result.scores.get("max_bar_growth_px"),
                        }
                    ],
                }
            )

        return {
            "detections": [],
            "count": 1 if has_fenbianlv_alarm else 0,
            "skill_name": self.config.get("name") or SKILL_NAME,
            "status": status,
            "message": message,
            "has_fenbianlv_alarm": has_fenbianlv_alarm,
            "has_fenbianlv_status_change": has_status_change,
            "is_resolution_anomaly": frame_result.is_anomaly,
            "has_person_count_change": False,
            "recognition_types": recognition_types,
            "alert_definitions": list(self.alert_definitions),
            "active_alerts": active_alerts,
            "resolution_state": frame_result.to_dict(),
            "safety_metrics": {
                "is_safe": status != "ALARM",
                "alert_info": {
                    "alert_triggered": has_fenbianlv_alarm,
                    "alert_name": "分辨率异常预警" if has_fenbianlv_alarm else "",
                    "alert_type": "分辨率检测",
                    "alert_description": message if has_fenbianlv_alarm else "",
                },
            },
            "flow_metrics": {
                "status": status,
                "confidence": frame_result.confidence,
                "fill_ratio": frame_result.metrics.get("fill_ratio"),
                "baseline_fill_ratio": frame_result.metrics.get("baseline_fill_ratio"),
            },
        }

    @staticmethod
    def _result_without_emit(result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result)
        out["has_fenbianlv_alarm"] = False
        out["recognition_types"] = []
        out["active_alerts"] = []
        safety = dict(out.get("safety_metrics") or {})
        alert_info = dict(safety.get("alert_info") or {})
        alert_info["alert_triggered"] = False
        safety["alert_info"] = alert_info
        out["safety_metrics"] = safety
        return out

    def process(
        self,
        input_data: Union[np.ndarray, str, Dict[str, Any], Any],
        fence_config: Dict = None,
    ) -> SkillResult:
        try:
            image = None
            if isinstance(input_data, np.ndarray):
                image = input_data.copy()
            elif isinstance(input_data, str):
                image = cv2.imread(input_data)
                if image is None:
                    return SkillResult.error_result(f"无法加载图像: {input_data}")
            elif isinstance(input_data, dict):
                image_data = input_data.get("image")
                if isinstance(image_data, np.ndarray):
                    image = image_data.copy()
                elif isinstance(image_data, str):
                    image = cv2.imread(image_data)
                else:
                    return SkillResult.error_result("输入字典中缺少有效 image")
            else:
                return SkillResult.error_result("不支持的输入数据类型")

            if image is None or image.size == 0:
                return SkillResult.error_result("无效的图像数据")

            data = self.detect_frame(image)
            now = time.time()
            can_emit = (
                self._last_emit_ts is None
                or now - self._last_emit_ts >= self.check_interval_sec
            )
            if not can_emit and data.get("has_fenbianlv_alarm"):
                data = self._result_without_emit(data)
            elif data.get("has_fenbianlv_alarm"):
                self._last_emit_ts = now
            self._last_process_ts = now
            return SkillResult.success_result(data)
        except Exception as e:
            logger.exception("分辨率检测技能处理失败: %s", e)
            return SkillResult.error_result(f"处理失败: {e}")

    def _draw_detections_on_frame(self, frame: np.ndarray, alert_data: Dict) -> np.ndarray:
        try:
            data = alert_data if isinstance(alert_data, dict) else {}
            if not data and isinstance(self._last_result, dict):
                data = self._last_result
            return _draw_fenbianlv_overlay(frame, data)
        except Exception as e:
            logger.error("绘制分辨率检测结果时出错: %s", e)
            return frame

    @staticmethod
    def _put_status_text(
        frame_bgr: np.ndarray,
        lines: List[str],
        xy: Tuple[int, int],
        color_bgr: Tuple[int, int, int],
        size: int = 28,
    ) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            y = xy[1]
            for line in lines:
                cv2.putText(
                    frame_bgr,
                    line[:80],
                    (xy[0], y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color_bgr,
                    2,
                )
                y += 26
            return frame_bgr

        font = None
        for path in _FONT_CANDIDATES:
            if path.is_file():
                for index in range(4):
                    try:
                        font = ImageFont.truetype(str(path), size=size, index=index)
                        break
                    except OSError:
                        continue
            if font is not None:
                break
        if font is None:
            font = ImageFont.load_default()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        x, y = xy
        for i, line in enumerate(lines):
            draw.text((x + 1, y + 1 + i * 32), line, font=font, fill=(0, 0, 0))
            draw.text(
                (x, y + i * 32),
                line,
                font=font,
                fill=(color_bgr[2], color_bgr[1], color_bgr[0]),
            )
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


_RES_STATE_ZH = {
    "normal": "正常",
    "suspect": "可疑",
    "anomaly": "异常",
}


def _draw_fenbianlv_overlay(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    status = str(result.get("status") or "OK")
    res = result.get("resolution_state") or {}
    inner = str(res.get("state") or "normal")
    display_status = status
    if status == "OK" and inner == "suspect":
        display_status = "CANDIDATE"
    elif status == "OK" and inner == "anomaly":
        display_status = "ALARM"

    conf = float(res.get("confidence") or 0)
    reasons = [_REASON_ZH.get(r, r) for r in (res.get("reasons") or []) if r != "ok"]
    if display_status == "OK" and (conf >= 0.5 or reasons):
        display_status = "CANDIDATE"
    if res.get("is_anomaly") or inner == "anomaly":
        display_status = "ALARM"

    color = _STATUS_COLOR.get(display_status, (200, 200, 200))
    thickness = 10 if display_status == "ALARM" else (7 if display_status == "CANDIDATE" else 4)
    vis = frame.copy()
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), color, thickness)

    m = res.get("metrics") or {}
    x0 = int(m.get("content_x0", 0))
    y0 = int(m.get("content_y0", 0))
    x1 = int(m.get("content_x1", vis.shape[1]))
    y1 = int(m.get("content_y1", vis.shape[0]))
    cv2.rectangle(vis, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (0, 255, 255), 2)

    inner_label = _RES_STATE_ZH.get(inner, inner)
    fill = m.get("fill_ratio")
    base_fill = m.get("baseline_fill_ratio")
    fill_s = f"{float(fill):.2f}" if fill is not None else "?"
    base_s = f"{float(base_fill):.2f}" if base_fill is not None else "-"
    lines = [
        f"分辨率检测：{display_status}（机内：{inner_label}）",
        f"平台报警：{'是' if result.get('has_fenbianlv_alarm') else '否'}",
        f"本帧置信度：{conf:.2f}",
        f"帧 {m.get('size_label', '?')}  有效 {m.get('content_label', '?')}",
        f"填充率 {fill_s}  基准 {base_s}",
    ]
    if reasons:
        lines.append(f"原因：{', '.join(reasons)}")

    return FenbianlvDetectorSkill._put_status_text(vis, lines, (16, 16), (255, 255, 255))


if __name__ == "__main__":
    img_dir = _CODE_ROOT / "test_images_videos"
    candidates = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"未找到测试图片: {img_dir}")
    img_name = str(candidates[0])
    skill = FenbianlvDetectorSkill(dict(FenbianlvDetectorSkill.DEFAULT_CONFIG))
    frame = cv2.imread(img_name)
    if frame is None:
        raise FileNotFoundError(img_name)
    out = skill.process(img_name)
    print(out.to_dict())
    if out.success and out.data:
        plotted = skill._draw_detections_on_frame(frame.copy(), out.data)
        out_path = str(Path(img_name).with_name(Path(img_name).stem + "_fenbianlv_ploted.jpg"))
        cv2.imwrite(out_path, plotted)
        print(f"结果已保存: {out_path}")
