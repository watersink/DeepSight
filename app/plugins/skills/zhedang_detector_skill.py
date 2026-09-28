"""
摄像头画面遮挡检测技能（自适应基线 + 纹理/边缘 collapse）

不依赖 Triton / YOLO；相对滑动干净基线检测梯度、边缘与分块纹理骤降。
可选上传一张正常画面作为基线种子；否则启动后 warmup 帧自动建基线。
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

from app.services import camera_pose_core as pose_core
from app.skills.skill_base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

SKILL_NAME = "zhedang_detector_skill"
RECOGNITION_TYPE = "occlusion"
RECOGNITION_DESC = "画面遮挡"

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
    "texture_or_edge_collapse": "纹理/边缘骤降",
    "uniform_full_frame_occlusion": "全屏均匀遮挡",
    "partial_block_occlusion": "局部块遮挡",
    "brightness_change_texture_preserved": "亮度变化(非遮挡)",
}


# ---------------------------------------------------------------------------
# 遮挡检测核心（源自 LLM/camera_occlusion，内联便于 DeepSight 部署）
# ---------------------------------------------------------------------------


class _OcclusionState(str, Enum):
    NORMAL = "normal"
    SUSPECT = "suspect"
    OCCLUDED = "occluded"


@dataclass
class _DetectorConfig:
    max_side: int = 640
    baseline_alpha: float = 0.02
    texture_collapse_ratio: float = 0.45
    edge_collapse_ratio: float = 0.40
    laplacian_collapse_ratio: float = 0.50
    uniform_occlusion_cv_max: float = 0.35
    uniform_gradient_max: float = 8.0
    partial_block_ratio_min: float = 0.55
    partial_texture_collapse_ratio: float = 0.55
    brightness_only_luma_delta: float = 35.0
    brightness_only_min_gradient_ratio: float = 0.70
    suspect_frames_to_alarm: int = 3
    clear_frames_to_recover: int = 15
    warmup_frames: int = 15
    # 仅当当前帧足够「清晰」时才更新基线，避免遮挡把基线拖低后比值回升
    baseline_healthy_ratio: float = 0.88


@dataclass
class OcclusionFrameResult:
    state: _OcclusionState
    is_occluded: bool
    confidence: float
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "is_occluded": self.is_occluded,
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


def _gradient_energy(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.mean(mag))


def _edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return float(np.count_nonzero(edges) / edges.size)


def _laplacian_variance(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def _gray_entropy(gray: np.ndarray, bins: int = 256) -> float:
    hist = cv2.calcHist([gray], [0], None, [bins], [0, bins])
    hist = hist.ravel().astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _spatial_uniformity(gray: np.ndarray, grid: int = 8) -> float:
    h, w = gray.shape
    gh = max(1, h // grid)
    gw = max(1, w // grid)
    energies: List[float] = []
    for r in range(grid):
        for c in range(grid):
            y0, y1 = r * gh, min((r + 1) * gh, h)
            x0, x1 = c * gw, min((c + 1) * gw, w)
            patch = gray[y0:y1, x0:x1]
            if patch.size:
                energies.append(_gradient_energy(patch))
    if len(energies) < 2:
        return 0.0
    arr = np.asarray(energies, dtype=np.float64)
    mean = float(arr.mean())
    if mean < 1e-6:
        return 0.0
    return float(arr.std() / mean)


def _low_texture_block_ratio(gray: np.ndarray, grid: int = 8, quantile: float = 0.25) -> float:
    h, w = gray.shape
    gh = max(1, h // grid)
    gw = max(1, w // grid)
    energies: List[float] = []
    for r in range(grid):
        for c in range(grid):
            y0, y1 = r * gh, min((r + 1) * gh, h)
            x0, x1 = c * gw, min((c + 1) * gw, w)
            patch = gray[y0:y1, x0:x1]
            if patch.size:
                energies.append(_gradient_energy(patch))
    if not energies:
        return 0.0
    arr = np.asarray(energies, dtype=np.float64)
    thr = float(np.quantile(arr, quantile))
    return float(np.mean(arr <= max(thr, 1e-6)))


def _extract_frame_metrics(frame_bgr: np.ndarray, max_side: int) -> Dict[str, float]:
    gray = _to_gray(frame_bgr, max_side=max_side)
    return {
        "gradient_energy": _gradient_energy(gray),
        "edge_density": _edge_density(gray),
        "laplacian_var": _laplacian_variance(gray),
        "entropy": _gray_entropy(gray),
        "spatial_uniformity_cv": _spatial_uniformity(gray),
        "low_texture_block_ratio": _low_texture_block_ratio(gray),
        "mean_luma": float(np.mean(gray)),
        "std_luma": float(np.std(gray)),
    }


class OcclusionDetector:
    """单路流遮挡检测器（每任务一个实例）。"""

    def __init__(self, config: Optional[_DetectorConfig] = None) -> None:
        self.config = config or _DetectorConfig()
        self._baseline: Optional[Dict[str, float]] = None
        self._frame_index = 0
        self._suspect_streak = 0
        self._clear_streak = 0
        self._state = _OcclusionState.NORMAL

    def reset(self) -> None:
        self._baseline = None
        self._frame_index = 0
        self._suspect_streak = 0
        self._clear_streak = 0
        self._state = _OcclusionState.NORMAL

    def seed_baseline(self, frame_bgr: np.ndarray) -> None:
        self._baseline = _extract_frame_metrics(frame_bgr, max_side=self.config.max_side)

    def process(self, frame_bgr: np.ndarray) -> OcclusionFrameResult:
        cfg = self.config
        metrics = _extract_frame_metrics(frame_bgr, max_side=cfg.max_side)
        self._frame_index += 1

        if self._frame_index <= cfg.warmup_frames:
            self._update_baseline_warmup(metrics)
            return OcclusionFrameResult(
                state=_OcclusionState.NORMAL,
                is_occluded=False,
                confidence=0.0,
                reasons=["warmup"],
                metrics=metrics,
                scores={"warmup_frame": float(self._frame_index)},
            )

        confidence, reasons, scores = self._evaluate_frame(metrics)
        frame_is_suspect = confidence >= 0.55 and "brightness_change_texture_preserved" not in reasons

        if frame_is_suspect:
            self._suspect_streak += 1
            self._clear_streak = 0
        else:
            self._clear_streak += 1
            self._suspect_streak = max(0, self._suspect_streak - 1)

        if self._state != _OcclusionState.OCCLUDED:
            if self._suspect_streak >= cfg.suspect_frames_to_alarm:
                self._state = _OcclusionState.OCCLUDED
            elif self._suspect_streak >= 1:
                self._state = _OcclusionState.SUSPECT
            else:
                self._state = _OcclusionState.NORMAL

        if self._state == _OcclusionState.OCCLUDED and self._clear_streak >= cfg.clear_frames_to_recover:
            self._state = _OcclusionState.NORMAL
            self._suspect_streak = 0

        if self._state == _OcclusionState.NORMAL and not frame_is_suspect:
            self._update_baseline_if_healthy(metrics)

        is_occluded = self._state == _OcclusionState.OCCLUDED
        # 本帧可疑时也保留置信度，便于推流叠加层与 ALARM 未确认前排查
        show_conf = (
            is_occluded
            or self._state == _OcclusionState.SUSPECT
            or frame_is_suspect
        )
        return OcclusionFrameResult(
            state=self._state,
            is_occluded=is_occluded,
            confidence=confidence if show_conf else 0.0,
            reasons=reasons if reasons else (["ok"] if not is_occluded else reasons),
            metrics=metrics,
            scores=scores,
        )

    def _update_baseline_warmup(self, metrics: Dict[str, float]) -> None:
        """预热：对梯度/边缘/拉普拉斯取峰值，避免一上来就是遮挡画面时基线过低。"""
        if self._baseline is None:
            self._baseline = dict(metrics)
            return
        for key in ("gradient_energy", "edge_density", "laplacian_var"):
            self._baseline[key] = max(float(self._baseline[key]), float(metrics[key]))
        alpha = self.config.baseline_alpha
        for key, value in metrics.items():
            if key in ("gradient_energy", "edge_density", "laplacian_var"):
                continue
            self._baseline[key] = (1.0 - alpha) * self._baseline[key] + alpha * value

    def _baseline_is_healthy(self, metrics: Dict[str, float]) -> bool:
        if self._baseline is None:
            return True
        ratio = self.config.baseline_healthy_ratio
        return (
            metrics["gradient_energy"] >= self._baseline["gradient_energy"] * ratio
            and metrics["edge_density"] >= self._baseline["edge_density"] * ratio
            and metrics["laplacian_var"] >= self._baseline["laplacian_var"] * ratio
        )

    def _update_baseline_if_healthy(self, metrics: Dict[str, float]) -> None:
        if not self._baseline_is_healthy(metrics):
            return
        alpha = self.config.baseline_alpha
        if self._baseline is None:
            self._baseline = dict(metrics)
            return
        for key, value in metrics.items():
            self._baseline[key] = (1.0 - alpha) * self._baseline[key] + alpha * value

    @staticmethod
    def _ratio(current: float, baseline: float) -> float:
        if baseline <= 1e-9:
            return 1.0
        return current / baseline

    def _evaluate_frame(
        self, metrics: Dict[str, float]
    ) -> Tuple[float, List[str], Dict[str, float]]:
        cfg = self.config
        reasons: List[str] = []
        scores: Dict[str, float] = {}
        if self._baseline is None:
            return 0.0, reasons, scores

        grad_ratio = self._ratio(metrics["gradient_energy"], self._baseline["gradient_energy"])
        edge_ratio = self._ratio(metrics["edge_density"], self._baseline["edge_density"])
        lap_ratio = self._ratio(metrics["laplacian_var"], self._baseline["laplacian_var"])
        scores["gradient_ratio"] = grad_ratio
        scores["edge_ratio"] = edge_ratio
        scores["laplacian_ratio"] = lap_ratio

        luma_delta = abs(metrics["mean_luma"] - self._baseline["mean_luma"])
        scores["luma_delta"] = luma_delta

        if (
            luma_delta >= cfg.brightness_only_luma_delta
            and grad_ratio >= cfg.brightness_only_min_gradient_ratio
            and edge_ratio >= cfg.brightness_only_min_gradient_ratio
        ):
            scores["brightness_only_gate"] = 1.0
            return 0.0, ["brightness_change_texture_preserved"], scores

        texture_collapse = (
            grad_ratio <= (1.0 - cfg.texture_collapse_ratio)
            or edge_ratio <= (1.0 - cfg.edge_collapse_ratio)
            or lap_ratio <= (1.0 - cfg.laplacian_collapse_ratio)
        )
        uniform_cover = (
            metrics["gradient_energy"] <= cfg.uniform_gradient_max
            and metrics["spatial_uniformity_cv"] <= cfg.uniform_occlusion_cv_max
        )
        partial_cover = (
            metrics["low_texture_block_ratio"] >= cfg.partial_block_ratio_min
            and grad_ratio <= (1.0 - cfg.partial_texture_collapse_ratio)
        )

        confidence = 0.0
        if texture_collapse:
            collapse_severity = max(1.0 - grad_ratio, 1.0 - edge_ratio, 1.0 - lap_ratio)
            confidence = max(confidence, min(1.0, collapse_severity * 1.2))
            reasons.append("texture_or_edge_collapse")
        if uniform_cover:
            confidence = max(confidence, 0.85)
            reasons.append("uniform_full_frame_occlusion")
        if partial_cover:
            confidence = max(confidence, 0.75)
            reasons.append("partial_block_occlusion")

        scores["confidence_raw"] = confidence
        return confidence, reasons, scores


def _map_skill_status(frame_result: OcclusionFrameResult) -> str:
    if "warmup" in frame_result.reasons:
        return "CALIBRATING"
    if frame_result.state == _OcclusionState.OCCLUDED:
        return "ALARM"
    if frame_result.state == _OcclusionState.SUSPECT:
        return "CANDIDATE"
    return "OK"


def _status_message(status: str, frame_result: OcclusionFrameResult) -> str:
    reasons = [_REASON_ZH.get(r, r) for r in frame_result.reasons if r != "ok"]
    tail = "，".join(reasons) if reasons else "画面正常"
    if status == "CALIBRATING":
        return f"遮挡检测预热中（{tail}）"
    if status == "ALARM":
        return f"检测到画面遮挡：{tail}"
    if status == "CANDIDATE":
        return f"疑似遮挡：{tail}"
    return f"画面正常：{tail}"


# ---------------------------------------------------------------------------
# DeepSight 技能
# ---------------------------------------------------------------------------


class ZhedangDetectorSkill(BaseSkill):
    """实时视频流遮挡检测。"""

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": SKILL_NAME,
        "name_zh": "画面遮挡检测",
        "version": "1.0",
        "cover_image": f"/skills/{SKILL_NAME}.png",
        "description": (
            "基于滑动干净基线与边缘/梯度纹理 collapse，检测镜头被遮挡、"
            "全屏均匀覆盖或局部块遮挡；亮度突变但纹理仍在时不误报。"
        ),
        "status": True,
        "run_mode": "stream",
        "required_models": [],
        "form_fields": [
            {
                "key": "reference_image_url",
                "label": "可选：正常画面种子图",
                "type": "text",
                "required": False,
                "default": "",
                "hint": "不填则启动后按 warmup 帧自动建基线；填则首帧用该清晰图初始化",
            },
            {
                "key": "confirm_count",
                "label": "连续确认次数",
                "type": "number",
                "required": False,
                "default": 3,
                "hint": "每次检测间隔内连续多少次可疑才报遮挡",
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
            "reference_image_url": "",
            "check_interval_sec": 2,
            "confirm_count": 2,
            "cooldown_sec": 5,
            "max_side": 640,
            "baseline_alpha": 0.02,
            "texture_collapse_ratio": 0.45,
            "edge_collapse_ratio": 0.40,
            "laplacian_collapse_ratio": 0.50,
            "warmup_frames": 15,
            "baseline_healthy_ratio": 0.88,
            "clear_frames_to_recover": 15,
            "brightness_only_luma_delta": 35.0,
        },
        "alert_definitions": [
            {
                "key": "occlusion_alarm",
                "level": 2,
                "description": "相对本点位正常基线判定画面被遮挡，并连续确认后触发。",
                "codes": {
                    "occlusion": {
                        "code": RECOGNITION_TYPE,
                        "description": RECOGNITION_DESC,
                        "analysis_case": "0001",
                    },
                },
            }
        ],
    }

    def _initialize(self) -> None:
        params = self.config.get("params", {}) if isinstance(self.config, dict) else {}
        confirm_count = max(1, int(params.get("confirm_count", 3) or 3))
        det_cfg = _DetectorConfig(
            max_side=max(320, int(params.get("max_side", 640) or 640)),
            baseline_alpha=float(params.get("baseline_alpha", 0.02) or 0.02),
            texture_collapse_ratio=float(params.get("texture_collapse_ratio", 0.45) or 0.45),
            edge_collapse_ratio=float(params.get("edge_collapse_ratio", 0.40) or 0.40),
            laplacian_collapse_ratio=float(params.get("laplacian_collapse_ratio", 0.50) or 0.50),
            brightness_only_luma_delta=float(
                params.get("brightness_only_luma_delta", 35.0) or 35.0
            ),
            suspect_frames_to_alarm=confirm_count,
            clear_frames_to_recover=max(1, int(params.get("clear_frames_to_recover", 15) or 15)),
            warmup_frames=max(1, int(params.get("warmup_frames", 15) or 15)),
        )
        det_cfg.baseline_healthy_ratio = float(
            params.get("baseline_healthy_ratio", 0.88) or 0.88
        )
        self._detector = OcclusionDetector(det_cfg)
        self.reference_image_url = str(params.get("reference_image_url") or "").strip()
        self.check_interval_sec = max(0.5, float(params.get("check_interval_sec", 2) or 2))
        self.confirm_count = confirm_count
        self.cooldown_sec = max(0.0, float(params.get("cooldown_sec", 5) or 5))
        self._repeat_sec = self.cooldown_sec if self.cooldown_sec > 0 else max(1.0, self.check_interval_sec)
        if self._repeat_sec >= 30:
            self._repeat_sec = max(5.0, self.check_interval_sec)

        self.alert_definitions: List[Dict[str, Any]] = list(
            self.config.get("alert_definitions") or []
        )
        self._last_process_ts: Optional[float] = None
        self._last_infer_ts: Optional[float] = None
        self._last_emit_ts: Optional[float] = None
        self._last_status: Optional[str] = None
        self._last_alert_ts = 0.0
        self._last_result: Dict[str, Any] = {}

        if self.reference_image_url:
            try:
                self.reload_reference(self.reference_image_url)
            except Exception as e:
                self.log("warning", f"遮挡检测种子图加载失败: {e}")

        self.log(
            "info",
            f"初始化遮挡检测: interval={self.check_interval_sec}s "
            f"confirm={self.confirm_count} warmup={det_cfg.warmup_frames} "
            f"cooldown={self.cooldown_sec}s",
        )

    def get_required_models(self) -> List[str]:
        return []

    def reload_reference(self, url: str) -> None:
        bgr = pose_core.load_image_bgr(url)
        self._detector.seed_baseline(bgr)
        self._detector.reset()
        self._detector.seed_baseline(bgr)
        self._detector._frame_index = self._detector.config.warmup_frames
        self.reference_image_url = url
        self._last_status = None
        self._last_alert_ts = 0.0
        self.log("info", "已用种子图初始化遮挡检测基线")

    def detect_frame(self, current_bgr: np.ndarray) -> Dict[str, Any]:
        frame_result = self._detector.process(current_bgr)
        status = _map_skill_status(frame_result)
        message = _status_message(status, frame_result)

        now = time.time()
        prev_status = self._last_status
        has_status_change = prev_status is not None and status != prev_status
        alarm_ready = status == "ALARM" and frame_result.is_occluded
        has_zhedang_alarm = bool(alarm_ready and (now - self._last_alert_ts >= self._repeat_sec))
        if has_zhedang_alarm:
            self._last_alert_ts = now
        if status != prev_status or has_zhedang_alarm:
            self.log(
                "info",
                f"遮挡检测: {status} emit={int(has_zhedang_alarm)} "
                f"conf={frame_result.confidence:.2f} {message}",
            )
            self._last_status = status

        result = self._build_result(
            frame_result,
            status=status,
            message=message,
            has_zhedang_alarm=has_zhedang_alarm,
            has_status_change=has_status_change,
        )
        result["run_mode"] = "stream"
        self._last_result = result
        return result

    def _build_result(
        self,
        frame_result: OcclusionFrameResult,
        *,
        status: str,
        message: str,
        has_zhedang_alarm: bool,
        has_status_change: bool,
    ) -> Dict[str, Any]:
        alert_def = self.alert_definitions[0] if self.alert_definitions else {}
        active_alerts: List[Dict[str, Any]] = []
        recognition_types: List[str] = []
        if has_zhedang_alarm:
            recognition_types = [RECOGNITION_TYPE]
            active_alerts.append(
                {
                    "key": alert_def.get("key", "occlusion_alarm"),
                    "level": int(alert_def.get("level", 2) or 2),
                    "description": message,
                    "recognition_type": RECOGNITION_TYPE,
                    "analysis_case": "0001",
                    "violation_type": RECOGNITION_DESC,
                    "events": [
                        {
                            "status": status,
                            "message": message,
                            "confidence": frame_result.confidence,
                            "reasons": frame_result.reasons,
                            "gradient_ratio": frame_result.scores.get("gradient_ratio"),
                            "edge_ratio": frame_result.scores.get("edge_ratio"),
                        }
                    ],
                }
            )

        return {
            "detections": [],
            "count": 1 if has_zhedang_alarm else 0,
            "skill_name": self.config.get("name") or SKILL_NAME,
            "status": status,
            "message": message,
            "has_zhedang_alarm": has_zhedang_alarm,
            "has_zhedang_status_change": has_status_change,
            "has_person_count_change": False,
            "recognition_types": recognition_types,
            "alert_definitions": list(self.alert_definitions),
            "active_alerts": active_alerts,
            "occlusion_state": frame_result.to_dict(),
            "safety_metrics": {
                "is_safe": status != "ALARM",
                "alert_info": {
                    "alert_triggered": has_zhedang_alarm,
                    "alert_name": "画面遮挡预警" if has_zhedang_alarm else "",
                    "alert_type": "遮挡检测",
                    "alert_description": message if has_zhedang_alarm else "",
                },
            },
            "flow_metrics": {
                "status": status,
                "confidence": frame_result.confidence,
                "gradient_ratio": frame_result.scores.get("gradient_ratio"),
                "edge_ratio": frame_result.scores.get("edge_ratio"),
            },
        }

    @staticmethod
    def _result_without_emit(result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result)
        out["has_zhedang_alarm"] = False
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

            # 每帧都跑检测（与 LLM preview 一致）；仅平台落库/推送按间隔节流
            data = self.detect_frame(image)
            now = time.time()
            can_emit = (
                self._last_emit_ts is None
                or now - self._last_emit_ts >= self.check_interval_sec
            )
            if not can_emit and data.get("has_zhedang_alarm"):
                data = self._result_without_emit(data)
            elif data.get("has_zhedang_alarm"):
                self._last_emit_ts = now
            self._last_process_ts = now
            self._last_infer_ts = now
            return SkillResult.success_result(data)
        except Exception as e:
            logger.exception("遮挡检测技能处理失败: %s", e)
            return SkillResult.error_result(f"处理失败: {e}")

    def _draw_detections_on_frame(self, frame: np.ndarray, alert_data: Dict) -> np.ndarray:
        try:
            data = alert_data if isinstance(alert_data, dict) else {}
            if not data and isinstance(self._last_result, dict):
                data = self._last_result
            return _draw_zhedang_overlay(frame, data)
        except Exception as e:
            logger.error("绘制遮挡检测结果时出错: %s", e)
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


_OCC_STATE_ZH = {
    "normal": "正常",
    "suspect": "可疑",
    "occluded": "已确认遮挡",
}


def _draw_zhedang_overlay(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    status = str(result.get("status") or "OK")
    occ = result.get("occlusion_state") or {}
    occ_state = str(occ.get("state") or "normal")
    # 叠加层优先展示状态机 occ_state，避免「OK + 纹理骤降」误解
    display_status = status
    if status == "OK" and occ_state in {"suspect", "occluded"}:
        display_status = "CANDIDATE" if occ_state == "suspect" else "ALARM"

    conf = float(occ.get("confidence") or 0)
    reasons = [_REASON_ZH.get(r, r) for r in (occ.get("reasons") or []) if r != "ok"]
    if display_status == "OK" and (conf >= 0.5 or reasons):
        display_status = "CANDIDATE"
    if occ.get("is_occluded") or occ_state == "occluded":
        display_status = "ALARM"

    color = _STATUS_COLOR.get(display_status, (200, 200, 200))
    thickness = 10 if display_status == "ALARM" else (7 if display_status == "CANDIDATE" else 4)
    vis = frame.copy()
    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), color, thickness)

    occ_label = _OCC_STATE_ZH.get(occ_state, occ_state)
    lines = [
        f"遮挡检测：{display_status}（机内：{occ_label}）",
        f"平台报警：{'是' if result.get('has_zhedang_alarm') else '否'}",
        f"本帧置信度：{float(occ.get('confidence') or 0):.2f}",
        f"原因：{', '.join(reasons) if reasons else '-'}",
    ]
    scores = occ.get("scores") or {}
    if scores.get("gradient_ratio") is not None:
        lines.append(
            f"梯度比 {float(scores.get('gradient_ratio', 0)):.2f}  "
            f"边缘比 {float(scores.get('edge_ratio', 0)):.2f}"
        )

    return ZhedangDetectorSkill._put_status_text(vis, lines, (16, 16), (255, 255, 255))


if __name__ == "__main__":
    img_dir = _CODE_ROOT / "test_images_videos"
    candidates = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"未找到测试图片: {img_dir}")
    img_name = str(candidates[0])
    skill = ZhedangDetectorSkill(dict(ZhedangDetectorSkill.DEFAULT_CONFIG))
    frame = cv2.imread(img_name)
    if frame is None:
        raise FileNotFoundError(img_name)
    out = skill.process(img_name)
    print(out.to_dict())
    if out.success and out.data:
        plotted = skill._draw_detections_on_frame(frame.copy(), out.data)
        out_path = str(Path(img_name).with_name(Path(img_name).stem + "_zhedang_ploted.jpg"))
        cv2.imwrite(out_path, plotted)
        print(f"结果已保存: {out_path}")
