"""
摄像头画面抖动检测技能（相位相关 + 滑动窗口抖动能量）

源自 LLM/camera_shake；默认阈值与 preview.py 当前配置一致。
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

_CODE_ROOT = Path(__file__).resolve().parents[3]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cv2
import logging
import numpy as np

from app.skills.skill_base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

SKILL_NAME = "doudong_detector_skill"
RECOGNITION_TYPE = "shake"
RECOGNITION_DESC = "画面抖动"

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
    "init": "初始化",
    "warmup": "预热建基线",
    "ok": "正常",
    "high_frequency_global_motion": "高频全局位移",
    "low_texture_skip": "纹理过低跳过",
    "low_correlation_skip": "相关峰值过低跳过",
    "large_motion_reset": "大位移重置",
}


# ---------------------------------------------------------------------------
# 抖动检测核心（LLM/camera_shake）
# ---------------------------------------------------------------------------


class _ShakeState(str, Enum):
    NORMAL = "normal"
    SUSPECT = "suspect"
    SHAKING = "shaking"


@dataclass
class _DetectorConfig:
    max_side: int = 640
    motion_window: int = 6
    baseline_alpha: float = 0.05
    delta_weight: float = 1.0
    jitter_threshold_abs: float = 0.55
    jitter_ratio_over_baseline: float = 1.6
    min_correlation_response: float = 0.15
    min_laplacian_var: float = 35.0
    large_motion_px: float = 18.0
    suspect_frames_to_alarm: int = 5
    clear_frames_to_recover: int = 12
    warmup_frames: int = 8


@dataclass
class ShakeFrameResult:
    state: _ShakeState
    is_shaking: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "is_shaking": self.is_shaking,
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "scores": {k: round(v, 4) for k, v in self.scores.items()},
        }


def _to_gray(frame_bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame_bgr = cv2.resize(
            frame_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def _laplacian_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _estimate_global_shift(prev_gray: np.ndarray, curr_gray: np.ndarray) -> Tuple[float, float, float]:
    a = prev_gray.astype(np.float32)
    b = curr_gray.astype(np.float32)
    h, w = a.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)
    shift, response = cv2.phaseCorrelate(a * win, b * win)
    return float(shift[0]), float(shift[1]), float(response)


def _shift_magnitude(dx: float, dy: float) -> float:
    return float(np.hypot(dx, dy))


class ShakeDetector:
    def __init__(self, config: Optional[_DetectorConfig] = None) -> None:
        self.config = config or _DetectorConfig()
        self._prev_gray: Optional[np.ndarray] = None
        self._shifts: Deque[Tuple[float, float]] = deque(
            maxlen=max(3, self.config.motion_window)
        )
        self._baseline_jitter: Optional[float] = None
        self._frame_index = 0
        self._suspect_streak = 0
        self._clear_streak = 0
        self._state = _ShakeState.NORMAL

    def reset(self) -> None:
        self._prev_gray = None
        self._shifts.clear()
        self._baseline_jitter = None
        self._frame_index = 0
        self._suspect_streak = 0
        self._clear_streak = 0
        self._state = _ShakeState.NORMAL

    def _window_jitter_score(self) -> Tuple[float, float, float]:
        cfg = self.config
        if len(self._shifts) < 2:
            return 0.0, 0.0, 0.0
        mags = [_shift_magnitude(dx, dy) for dx, dy in self._shifts]
        mean_mag = float(np.mean(mags))
        deltas: List[float] = []
        prev = self._shifts[0]
        for cur in list(self._shifts)[1:]:
            deltas.append(_shift_magnitude(cur[0] - prev[0], cur[1] - prev[1]))
            prev = cur
        mean_delta = float(np.mean(deltas)) if deltas else 0.0
        score = mean_mag + cfg.delta_weight * mean_delta
        return score, mean_mag, mean_delta

    def process(self, frame_bgr: np.ndarray) -> ShakeFrameResult:
        cfg = self.config
        gray = _to_gray(frame_bgr, max_side=cfg.max_side)
        lap_var = _laplacian_variance(gray)
        self._frame_index += 1

        metrics: Dict[str, float] = {"laplacian_var": lap_var}
        scores: Dict[str, float] = {}
        reasons: List[str] = []

        if self._prev_gray is None:
            self._prev_gray = gray
            return ShakeFrameResult(
                state=_ShakeState.NORMAL,
                is_shaking=False,
                confidence=0.0,
                reasons=["init"],
                metrics=metrics,
                scores={"warmup_frame": float(self._frame_index)},
            )

        dx, dy, response = _estimate_global_shift(self._prev_gray, gray)
        self._prev_gray = gray
        step_mag = _shift_magnitude(dx, dy)
        metrics.update(
            {
                "shift_dx": dx,
                "shift_dy": dy,
                "shift_mag": step_mag,
                "phase_response": response,
            }
        )

        if lap_var < cfg.min_laplacian_var:
            reasons.append("low_texture_skip")
            return ShakeFrameResult(
                state=self._state,
                is_shaking=self._state == _ShakeState.SHAKING,
                confidence=0.0,
                reasons=reasons,
                metrics=metrics,
                scores={"skipped": 1.0},
            )

        if response < cfg.min_correlation_response:
            reasons.append("low_correlation_skip")
            return ShakeFrameResult(
                state=self._state,
                is_shaking=self._state == _ShakeState.SHAKING,
                confidence=0.0,
                reasons=reasons,
                metrics=metrics,
                scores={"skipped": 1.0},
            )

        if step_mag >= cfg.large_motion_px:
            self._shifts.clear()
            reasons.append("large_motion_reset")
            if self._frame_index <= cfg.warmup_frames:
                return ShakeFrameResult(
                    state=_ShakeState.NORMAL,
                    is_shaking=False,
                    confidence=0.0,
                    reasons=reasons + ["warmup"],
                    metrics=metrics,
                    scores={"warmup_frame": float(self._frame_index)},
                )

        self._shifts.append((dx, dy))
        jitter_score, mean_mag, mean_delta = self._window_jitter_score()
        metrics.update(
            {
                "jitter_score": jitter_score,
                "mean_shift_mag": mean_mag,
                "mean_shift_delta": mean_delta,
            }
        )
        scores["jitter_score"] = jitter_score

        if self._frame_index <= cfg.warmup_frames:
            if jitter_score > 0:
                self._update_baseline(jitter_score)
            return ShakeFrameResult(
                state=_ShakeState.NORMAL,
                is_shaking=False,
                confidence=0.0,
                reasons=["warmup"],
                metrics=metrics,
                scores={"warmup_frame": float(self._frame_index), **scores},
            )

        baseline = self._baseline_jitter if self._baseline_jitter is not None else jitter_score
        scores["baseline_jitter"] = baseline
        ratio = jitter_score / baseline if baseline > 1e-6 else 1.0
        scores["jitter_ratio"] = ratio

        threshold = max(cfg.jitter_threshold_abs, baseline * cfg.jitter_ratio_over_baseline)
        scores["jitter_threshold"] = threshold
        frame_is_suspect = (
            jitter_score >= threshold and len(self._shifts) >= cfg.motion_window // 2
        )

        confidence = 0.0
        if frame_is_suspect:
            over = (jitter_score - threshold) / max(threshold, 1e-6)
            confidence = min(1.0, 0.55 + over * 0.5)
            reasons.append("high_frequency_global_motion")

        if frame_is_suspect:
            self._suspect_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._suspect_streak = max(0, self._suspect_streak - 1)

        if self._state != _ShakeState.SHAKING:
            if self._suspect_streak >= cfg.suspect_frames_to_alarm:
                self._state = _ShakeState.SHAKING
            elif self._suspect_streak >= max(2, cfg.suspect_frames_to_alarm // 3):
                self._state = _ShakeState.SUSPECT
            else:
                self._state = _ShakeState.NORMAL

        if self._state == _ShakeState.SHAKING and self._clear_streak >= cfg.clear_frames_to_recover:
            self._state = _ShakeState.NORMAL
            self._suspect_streak = 0

        if self._state == _ShakeState.NORMAL and not frame_is_suspect and jitter_score > 0:
            self._update_baseline(jitter_score)

        is_shaking = self._state == _ShakeState.SHAKING
        show_conf = is_shaking or self._state == _ShakeState.SUSPECT or frame_is_suspect
        if not reasons:
            reasons = ["ok"]
        return ShakeFrameResult(
            state=self._state,
            is_shaking=is_shaking,
            confidence=confidence if show_conf else 0.0,
            reasons=reasons,
            metrics=metrics,
            scores=scores,
        )

    def _update_baseline(self, jitter_score: float) -> None:
        alpha = self.config.baseline_alpha
        if self._baseline_jitter is None:
            self._baseline_jitter = jitter_score
            return
        self._baseline_jitter = (1.0 - alpha) * self._baseline_jitter + alpha * jitter_score


def _map_skill_status(frame_result: ShakeFrameResult) -> str:
    if "warmup" in frame_result.reasons or "init" in frame_result.reasons:
        return "CALIBRATING"
    if frame_result.state == _ShakeState.SHAKING:
        return "ALARM"
    if frame_result.state == _ShakeState.SUSPECT:
        return "CANDIDATE"
    return "OK"


def _status_message(status: str, frame_result: ShakeFrameResult) -> str:
    reasons = [_REASON_ZH.get(r, r) for r in frame_result.reasons if r != "ok"]
    tail = "，".join(reasons) if reasons else "画面稳定"
    js = frame_result.metrics.get("jitter_score", 0)
    if status == "CALIBRATING":
        return f"抖动检测预热中（{tail}）"
    if status == "ALARM":
        return f"检测到画面抖动，抖动分 {js:.2f}：{tail}"
    if status == "CANDIDATE":
        return f"疑似画面抖动，抖动分 {js:.2f}：{tail}"
    return f"画面稳定：{tail}"


class DoudongDetectorSkill(BaseSkill):
    """实时视频流画面抖动检测。"""

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": SKILL_NAME,
        "name_zh": "画面抖动检测",
        "version": "1.0",
        "cover_image": f"/skills/{SKILL_NAME}.png",
        "description": (
            "基于相邻帧相位相关的全局位移与滑动窗口抖动能量，"
            "检测固定机位画面异常抖动（含轻微抖动，阈值可配）。"
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
                "default": 5,
            },
            {
                "key": "jitter_threshold_abs",
                "label": "抖动分绝对阈值",
                "type": "number",
                "required": False,
                "default": 0.55,
                "hint": "越小越敏感，轻微抖动常用 0.35~0.55",
            },
            {
                "key": "jitter_ratio_over_baseline",
                "label": "相对基线倍数",
                "type": "number",
                "required": False,
                "default": 1.6,
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
            "confirm_count": 5,
            "cooldown_sec": 5,
            "max_side": 640,
            "motion_window": 6,
            "delta_weight": 1.0,
            "jitter_threshold_abs": 0.55,
            "jitter_ratio_over_baseline": 1.6,
            "baseline_alpha": 0.05,
            "large_motion_px": 18.0,
            "min_correlation_response": 0.15,
            "min_laplacian_var": 35.0,
            "warmup_frames": 8,
            "clear_frames_to_recover": 12,
        },
        "alert_definitions": [
            {
                "key": "shake_alarm",
                "level": 2,
                "description": "画面持续异常抖动并连续确认后触发。",
                "codes": {
                    "shake": {
                        "code": RECOGNITION_TYPE,
                        "description": RECOGNITION_DESC,
                        "analysis_case": "0008",
                    },
                },
            }
        ],
    }

    def _initialize(self) -> None:
        params = self.config.get("params", {}) if isinstance(self.config, dict) else {}
        confirm_count = max(1, int(params.get("confirm_count", 5) or 5))
        det_cfg = _DetectorConfig(
            max_side=int(params.get("max_side", 640) or 640),
            motion_window=max(3, int(params.get("motion_window", 6) or 6)),
            baseline_alpha=float(params.get("baseline_alpha", 0.05) or 0.05),
            delta_weight=float(params.get("delta_weight", 1.0) or 1.0),
            jitter_threshold_abs=float(params.get("jitter_threshold_abs", 0.55) or 0.55),
            jitter_ratio_over_baseline=float(
                params.get("jitter_ratio_over_baseline", 1.6) or 1.6
            ),
            min_correlation_response=float(
                params.get("min_correlation_response", 0.15) or 0.15
            ),
            min_laplacian_var=float(params.get("min_laplacian_var", 35.0) or 35.0),
            large_motion_px=float(params.get("large_motion_px", 18.0) or 18.0),
            suspect_frames_to_alarm=confirm_count,
            clear_frames_to_recover=max(
                1, int(params.get("clear_frames_to_recover", 12) or 12)
            ),
            warmup_frames=max(1, int(params.get("warmup_frames", 8) or 8)),
        )
        self._detector = ShakeDetector(det_cfg)
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
            f"初始化抖动检测: interval={self.check_interval_sec}s confirm={confirm_count} "
            f"jitter_abs={det_cfg.jitter_threshold_abs} "
            f"jitter_ratio={det_cfg.jitter_ratio_over_baseline} "
            f"warmup={det_cfg.warmup_frames}",
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
        alarm_ready = status == "ALARM" and frame_result.is_shaking
        has_doudong_alarm = bool(alarm_ready and (now - self._last_alert_ts >= self._repeat_sec))
        if has_doudong_alarm:
            self._last_alert_ts = now
        if status != prev_status or has_doudong_alarm:
            self.log(
                "info",
                f"抖动检测: {status} emit={int(has_doudong_alarm)} "
                f"conf={frame_result.confidence:.2f} {message}",
            )
            self._last_status = status

        result = self._build_result(
            frame_result,
            status=status,
            message=message,
            has_doudong_alarm=has_doudong_alarm,
            has_status_change=has_status_change,
        )
        result["run_mode"] = "stream"
        self._last_result = result
        return result

    def _build_result(
        self,
        frame_result: ShakeFrameResult,
        *,
        status: str,
        message: str,
        has_doudong_alarm: bool,
        has_status_change: bool,
    ) -> Dict[str, Any]:
        alert_def = self.alert_definitions[0] if self.alert_definitions else {}
        active_alerts: List[Dict[str, Any]] = []
        recognition_types: List[str] = []
        if has_doudong_alarm:
            recognition_types = [RECOGNITION_TYPE]
            active_alerts.append(
                {
                    "key": alert_def.get("key", "shake_alarm"),
                    "level": int(alert_def.get("level", 2) or 2),
                    "description": message,
                    "recognition_type": RECOGNITION_TYPE,
                    "analysis_case": "0008",
                    "violation_type": RECOGNITION_DESC,
                    "events": [
                        {
                            "status": status,
                            "message": message,
                            "confidence": frame_result.confidence,
                            "jitter_score": frame_result.metrics.get("jitter_score"),
                            "jitter_ratio": frame_result.scores.get("jitter_ratio"),
                        }
                    ],
                }
            )

        return {
            "detections": [],
            "count": 1 if has_doudong_alarm else 0,
            "skill_name": self.config.get("name") or SKILL_NAME,
            "status": status,
            "message": message,
            "has_doudong_alarm": has_doudong_alarm,
            "has_doudong_status_change": has_status_change,
            "has_person_count_change": False,
            "recognition_types": recognition_types,
            "alert_definitions": list(self.alert_definitions),
            "active_alerts": active_alerts,
            "shake_state": frame_result.to_dict(),
            "safety_metrics": {
                "is_safe": status != "ALARM",
                "alert_info": {
                    "alert_triggered": has_doudong_alarm,
                    "alert_name": "画面抖动预警" if has_doudong_alarm else "",
                    "alert_type": "抖动检测",
                    "alert_description": message if has_doudong_alarm else "",
                },
            },
            "flow_metrics": {
                "status": status,
                "confidence": frame_result.confidence,
                "jitter_score": frame_result.metrics.get("jitter_score"),
                "jitter_ratio": frame_result.scores.get("jitter_ratio"),
            },
        }

    @staticmethod
    def _result_without_emit(result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result)
        out["has_doudong_alarm"] = False
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
            if not can_emit and data.get("has_doudong_alarm"):
                data = self._result_without_emit(data)
            elif data.get("has_doudong_alarm"):
                self._last_emit_ts = now
            self._last_process_ts = now
            return SkillResult.success_result(data)
        except Exception as e:
            logger.exception("抖动检测技能处理失败: %s", e)
            return SkillResult.error_result(f"处理失败: {e}")

    def _draw_detections_on_frame(self, frame: np.ndarray, alert_data: Dict) -> np.ndarray:
        try:
            data = alert_data if isinstance(alert_data, dict) else {}
            if not data and isinstance(self._last_result, dict):
                data = self._last_result
            return _draw_doudong_overlay(frame, data)
        except Exception as e:
            logger.error("绘制抖动检测结果时出错: %s", e)
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


_SHAKE_STATE_ZH = {
    "normal": "正常",
    "suspect": "可疑",
    "shaking": "抖动",
}


def _draw_doudong_overlay(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    status = str(result.get("status") or "OK")
    shake = result.get("shake_state") or {}
    shake_state = str(shake.get("state") or "normal")
    display_status = status
    if status == "OK" and shake_state == "suspect":
        display_status = "CANDIDATE"
    elif status == "OK" and shake_state == "shaking":
        display_status = "ALARM"

    conf = float(shake.get("confidence") or 0)
    reasons = [_REASON_ZH.get(r, r) for r in (shake.get("reasons") or []) if r != "ok"]
    if display_status == "OK" and (conf >= 0.5 or reasons):
        display_status = "CANDIDATE"
    if shake.get("is_shaking") or shake_state == "shaking":
        display_status = "ALARM"

    color = _STATUS_COLOR.get(display_status, (200, 200, 200))
    thickness = 10 if display_status == "ALARM" else (7 if display_status == "CANDIDATE" else 4)
    vis = frame.copy()
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), color, thickness)

    occ_label = _SHAKE_STATE_ZH.get(shake_state, shake_state)
    lines = [
        f"抖动检测：{display_status}（机内：{occ_label}）",
        f"平台报警：{'是' if result.get('has_doudong_alarm') else '否'}",
        f"本帧置信度：{conf:.2f}",
        f"抖动分：{float(shake.get('metrics', {}).get('jitter_score', 0) or 0):.3f}",
    ]
    scores = shake.get("scores") or {}
    if scores.get("jitter_ratio") is not None:
        lines.append(
            f"相对基线 {float(scores.get('jitter_ratio', 0)):.2f}  "
            f"阈值 {float(scores.get('jitter_threshold', 0)):.2f}"
        )
    if reasons:
        lines.append(f"原因：{', '.join(reasons)}")

    return DoudongDetectorSkill._put_status_text(vis, lines, (16, 16), (255, 255, 255))


if __name__ == "__main__":
    img_dir = _CODE_ROOT / "test_images_videos"
    candidates = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"未找到测试图片: {img_dir}")
    img_name = str(candidates[0])
    skill = DoudongDetectorSkill(dict(DoudongDetectorSkill.DEFAULT_CONFIG))
    frame = cv2.imread(img_name)
    if frame is None:
        raise FileNotFoundError(img_name)
    out = skill.process(img_name)
    print(out.to_dict())
    if out.success and out.data:
        plotted = skill._draw_detections_on_frame(frame.copy(), out.data)
        out_path = str(Path(img_name).with_name(Path(img_name).stem + "_doudong_ploted.jpg"))
        cv2.imwrite(out_path, plotted)
        print(f"结果已保存: {out_path}")
