"""
画面过曝检测技能（点位基准）

规则：
  - 不依赖 Triton / YOLO，按当前帧亮度特征与点位基准比较
  - 启动后可用前若干帧自动标定基准，也可加载预存 baseline JSON
  - 可选电子围栏：用围栏外接矩形作为有效监控区 ROI
  - 判定为过曝候选（CANDIDATE）后，连续若干秒仍过曝才升为告警
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# 支持直接运行本文件: python app/plugins/skills/guobao_detector_skill.py
_CODE_ROOT = Path(__file__).resolve().parents[3]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cv2
import logging
import numpy as np

from app.services import camera_pose_core as pose_core
from app.skills.skill_base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

SKILL_NAME = "guobao_detector_skill"

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

# 摄影中灰，用于把对数平均亮度换成曝光档位 EV
_MIDDLE_GREY = 0.18
_LOG_EPS = 1e-4


@dataclass
class FeatureConfig:
    bright_pixel_threshold: int = 225
    clip_threshold: int = 250
    roi: Tuple[float, float, float, float] = (0.25, 0.25, 0.5, 0.5)


@dataclass
class AlarmConfig:
    median_rise_ratio: float = 0.35
    min_median_rise_abs: float = 15.0
    bright_ratio_rise: float = 0.20
    clip_ratio_rise: float = 0.12
    block_rise_ratio: float = 0.30
    block_brighten_fraction: float = 0.40
    ignore_block_mean: float = 220.0
    consecutive_seconds: float = 6.0
    exposure_rise_ratio: float = 0.35
    well_exposed_drop: float = 0.20


@dataclass
class BrightnessFeatures:
    width: int
    height: int
    mean: float
    median: float
    p90: float
    bright_ratio: float
    clip_ratio: float
    std: float
    roi_mean: float
    roi_median: float
    roi_p90: float
    roi_bright_ratio: float
    roi_clip_ratio: float
    roi_contrast: float
    exposure_level: Optional[float] = None
    roi_exposure_level: Optional[float] = None
    exposure_ev: Optional[float] = None
    roi_exposure_ev: Optional[float] = None
    well_exposed_ratio: Optional[float] = None
    roi_well_exposed_ratio: Optional[float] = None
    center_weighted_mean: Optional[float] = None
    block_means_3x3: List[List[float]] = field(default_factory=list)
    block_bright_3x3: List[List[float]] = field(default_factory=list)
    block_means_8x8: List[List[float]] = field(default_factory=list)
    block_bright_8x8: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BrightnessFeatures":
        def opt(name: str) -> Optional[float]:
            if name not in data or data[name] is None:
                return None
            return float(data[name])

        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            mean=float(data["mean"]),
            median=float(data["median"]),
            p90=float(data["p90"]),
            bright_ratio=float(data["bright_ratio"]),
            clip_ratio=float(data["clip_ratio"]),
            std=float(data["std"]),
            roi_mean=float(data["roi_mean"]),
            roi_median=float(data["roi_median"]),
            roi_p90=float(data.get("roi_p90", data["p90"])),
            roi_bright_ratio=float(data["roi_bright_ratio"]),
            roi_clip_ratio=float(data.get("roi_clip_ratio", data["clip_ratio"])),
            roi_contrast=float(data["roi_contrast"]),
            exposure_level=opt("exposure_level"),
            roi_exposure_level=opt("roi_exposure_level"),
            exposure_ev=opt("exposure_ev"),
            roi_exposure_ev=opt("roi_exposure_ev"),
            well_exposed_ratio=opt("well_exposed_ratio"),
            roi_well_exposed_ratio=opt("roi_well_exposed_ratio"),
            center_weighted_mean=opt("center_weighted_mean"),
            block_means_3x3=data.get("block_means_3x3") or [],
            block_bright_3x3=data.get("block_bright_3x3") or [],
            block_means_8x8=data.get("block_means_8x8") or [],
            block_bright_8x8=data.get("block_bright_8x8") or [],
        )


@dataclass
class CompareResult:
    is_candidate: bool
    reasons: List[str] = field(default_factory=list)
    median_rise_ratio: float = 0.0
    median_rise_abs: float = 0.0
    bright_ratio_rise: float = 0.0
    clip_ratio_rise: float = 0.0
    block_brighten_fraction: float = 0.0
    valid_blocks: int = 0
    brightened_blocks: int = 0
    roi_median: float = 0.0
    baseline_roi_median: float = 0.0
    roi_bright_ratio: float = 0.0
    baseline_roi_bright_ratio: float = 0.0
    roi_clip_ratio: float = 0.0
    baseline_roi_clip_ratio: float = 0.0
    exposure_level: Optional[float] = None
    baseline_exposure_level: Optional[float] = None
    exposure_ev: Optional[float] = None
    baseline_exposure_ev: Optional[float] = None
    exposure_rise_ratio: float = 0.0
    well_exposed_ratio: Optional[float] = None
    baseline_well_exposed_ratio: Optional[float] = None
    well_exposed_drop: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectState:
    status: str
    alarm: bool
    candidate: bool
    hold_seconds: float
    compare: CompareResult
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "alarm": self.alarm,
            "candidate": self.candidate,
            "hold_seconds": self.hold_seconds,
            "compare": self.compare.to_dict(),
            "message": self.message,
        }


class TemporalGate:
    """连续若干秒都判定为过曝候选后才升为告警。"""

    def __init__(self, consecutive_seconds: float) -> None:
        self.consecutive_seconds = consecutive_seconds
        self._hold = 0.0
        self._alarm = False

    def update(self, is_candidate: bool, dt: float) -> Tuple[bool, float]:
        if is_candidate:
            self._hold += dt
            if self._hold >= self.consecutive_seconds:
                self._alarm = True
        else:
            self._hold = 0.0
            self._alarm = False
        return self._alarm, self._hold

    def reset(self) -> None:
        self._hold = 0.0
        self._alarm = False


def _geometric_mean_y(y: np.ndarray) -> float:
    yf = np.clip(y.astype(np.float64), 0.0, 255.0)
    return float(np.exp(np.mean(np.log(yf + _LOG_EPS))))


def _log_average_norm(y: np.ndarray) -> float:
    yn = np.clip(y.astype(np.float64), 0.0, 255.0) / 255.0
    return float(np.exp(np.mean(np.log(yn + _LOG_EPS))))


def _exposure_ev(log_avg_norm: float, middle_grey: float = _MIDDLE_GREY) -> float:
    return float(np.log2(max(log_avg_norm, _LOG_EPS) / middle_grey))


def _well_exposed_ratio(y: np.ndarray, low: float = 40.0, high: float = 220.0) -> float:
    return float(((y >= low) & (y <= high)).mean())


def _center_weighted_mean(y: np.ndarray) -> float:
    h, w = y.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    sigma = max(1.0, 0.35 * min(h, w))
    weight = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma))
    return float((y.astype(np.float64) * weight).sum() / (weight.sum() + _LOG_EPS))


def _predict_exposure(y: np.ndarray, clip_thr: float = 250.0) -> Dict[str, float]:
    log_avg = _log_average_norm(y)
    return {
        "exposure_level": _geometric_mean_y(y),
        "exposure_ev": _exposure_ev(log_avg),
        "well_exposed_ratio": _well_exposed_ratio(y),
        "center_weighted_mean": _center_weighted_mean(y),
        "highlight_clip_ratio": float((y >= clip_thr).mean()),
    }


def _to_y(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def roi_rect_px(
    frame_shape: Tuple[int, ...],
    roi: Tuple[float, float, float, float],
) -> Tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x, top, rw, rh = roi
    x1 = max(0, min(w - 1, int(x * w)))
    y1 = max(0, min(h - 1, int(top * h)))
    x2 = max(x1 + 1, min(w, int((x + rw) * w)))
    y2 = max(y1 + 1, min(h, int((top + rh) * h)))
    return x1, y1, x2, y2


def _crop_roi(y: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = roi_rect_px(y.shape, roi)
    return y[y1:y2, x1:x2]


def _block_stats(y: np.ndarray, grid: int, bright_thr: float) -> Tuple[np.ndarray, np.ndarray]:
    h, w = y.shape
    bh, bw = max(1, h // grid), max(1, w // grid)
    means = np.zeros((grid, grid), dtype=np.float32)
    bright = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        y1, y2 = i * bh, h if i == grid - 1 else (i + 1) * bh
        for j in range(grid):
            x1, x2 = j * bw, w if j == grid - 1 else (j + 1) * bw
            block = y[y1:y2, x1:x2]
            means[i, j] = float(block.mean())
            bright[i, j] = float((block >= bright_thr).mean())
    return means, bright


def _as_nested(arr: np.ndarray) -> List[List[float]]:
    return [[round(float(v), 3) for v in row] for row in arr]


def extract_features(frame_bgr: np.ndarray, cfg: FeatureConfig) -> BrightnessFeatures:
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("空帧，无法提取亮度特征")
    y = _to_y(frame_bgr)
    h, w = y.shape
    roi = _crop_roi(y, cfg.roi)
    bright_thr = cfg.bright_pixel_threshold
    clip_thr = cfg.clip_threshold
    means3, bright3 = _block_stats(y, 3, bright_thr)
    means8, bright8 = _block_stats(y, 8, bright_thr)
    full_exp = _predict_exposure(y, clip_thr=clip_thr)
    roi_exp = _predict_exposure(roi, clip_thr=clip_thr)
    return BrightnessFeatures(
        width=int(w),
        height=int(h),
        mean=float(y.mean()),
        median=float(np.median(y)),
        p90=float(np.percentile(y, 90)),
        bright_ratio=float((y >= bright_thr).mean()),
        clip_ratio=float((y >= clip_thr).mean()),
        std=float(y.std()),
        roi_mean=float(roi.mean()),
        roi_median=float(np.median(roi)),
        roi_p90=float(np.percentile(roi, 90)),
        roi_bright_ratio=float((roi >= bright_thr).mean()),
        roi_clip_ratio=float((roi >= clip_thr).mean()),
        roi_contrast=float(roi.std()),
        exposure_level=full_exp["exposure_level"],
        roi_exposure_level=roi_exp["exposure_level"],
        exposure_ev=full_exp["exposure_ev"],
        roi_exposure_ev=roi_exp["exposure_ev"],
        well_exposed_ratio=full_exp["well_exposed_ratio"],
        roi_well_exposed_ratio=roi_exp["well_exposed_ratio"],
        center_weighted_mean=full_exp["center_weighted_mean"],
        block_means_3x3=_as_nested(means3),
        block_bright_3x3=_as_nested(bright3),
        block_means_8x8=_as_nested(means8),
        block_bright_8x8=_as_nested(bright8),
    )


def median_features(items: List[BrightnessFeatures]) -> BrightnessFeatures:
    if not items:
        raise ValueError("没有可用于标定的特征")
    if len(items) == 1:
        return items[0]

    def med_scalar(name: str) -> float:
        return float(np.median([getattr(x, name) for x in items]))

    def med_grid(name: str) -> List[List[float]]:
        stack = np.array([getattr(x, name) for x in items], dtype=np.float32)
        return _as_nested(np.median(stack, axis=0))

    first = items[0]
    return BrightnessFeatures(
        width=first.width,
        height=first.height,
        mean=med_scalar("mean"),
        median=med_scalar("median"),
        p90=med_scalar("p90"),
        bright_ratio=med_scalar("bright_ratio"),
        clip_ratio=med_scalar("clip_ratio"),
        std=med_scalar("std"),
        roi_mean=med_scalar("roi_mean"),
        roi_median=med_scalar("roi_median"),
        roi_p90=med_scalar("roi_p90"),
        roi_bright_ratio=med_scalar("roi_bright_ratio"),
        roi_clip_ratio=med_scalar("roi_clip_ratio"),
        roi_contrast=med_scalar("roi_contrast"),
        exposure_level=med_scalar("exposure_level"),
        roi_exposure_level=med_scalar("roi_exposure_level"),
        exposure_ev=med_scalar("exposure_ev"),
        roi_exposure_ev=med_scalar("roi_exposure_ev"),
        well_exposed_ratio=med_scalar("well_exposed_ratio"),
        roi_well_exposed_ratio=med_scalar("roi_well_exposed_ratio"),
        center_weighted_mean=med_scalar("center_weighted_mean"),
        block_means_3x3=med_grid("block_means_3x3"),
        block_bright_3x3=med_grid("block_bright_3x3"),
        block_means_8x8=med_grid("block_means_8x8"),
        block_bright_8x8=med_grid("block_bright_8x8"),
    )


def _as_array(grid: List[List[float]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.float32)


def compare_to_baseline(
    current: BrightnessFeatures,
    baseline: BrightnessFeatures,
    alarm: AlarmConfig,
) -> CompareResult:
    base_med = max(float(baseline.roi_median), 1.0)
    median_rise_abs = float(current.roi_median - baseline.roi_median)
    median_rise_ratio = median_rise_abs / base_med
    bright_rise_roi = float(current.roi_bright_ratio - baseline.roi_bright_ratio)
    bright_rise_global = float(current.bright_ratio - baseline.bright_ratio)
    bright_rise = max(bright_rise_roi, bright_rise_global)
    clip_rise_roi = float(current.roi_clip_ratio - baseline.roi_clip_ratio)
    clip_rise_global = float(current.clip_ratio - baseline.clip_ratio)
    clip_rise = max(clip_rise_roi, clip_rise_global)

    base_blocks = _as_array(baseline.block_means_8x8)
    cur_blocks = _as_array(current.block_means_8x8)
    if cur_blocks.shape != base_blocks.shape:
        valid = 0
        brightened = 0
        bright_frac = 0.0
    else:
        valid_mask = base_blocks <= alarm.ignore_block_mean
        rises = (cur_blocks - base_blocks) / np.maximum(base_blocks, 1.0)
        brightened_mask = valid_mask & (rises >= alarm.block_rise_ratio)
        valid = int(valid_mask.sum())
        brightened = int(brightened_mask.sum())
        bright_frac = float(brightened / valid) if valid else 0.0

    reasons: List[str] = []
    median_hit = (
        median_rise_ratio >= alarm.median_rise_ratio
        and median_rise_abs >= alarm.min_median_rise_abs
    )
    if median_hit:
        reasons.append(
            f"ROI中位数上升 {median_rise_ratio:.0%} "
            f"({baseline.roi_median:.1f}→{current.roi_median:.1f})"
        )
    if bright_rise >= alarm.bright_ratio_rise:
        reasons.append(
            f"亮像素比升高 {bright_rise:.0%} "
            f"(ROI {baseline.roi_bright_ratio:.1%}→{current.roi_bright_ratio:.1%})"
        )
    if clip_rise >= alarm.clip_ratio_rise:
        reasons.append(
            f"高光裁切升高 {clip_rise:.0%} "
            f"(ROI {baseline.roi_clip_ratio:.1%}→{current.roi_clip_ratio:.1%})"
        )
    if bright_frac >= alarm.block_brighten_fraction and valid > 0:
        reasons.append(f"分块变亮 {brightened}/{valid} ({bright_frac:.0%})")

    exp_rise = 0.0
    well_drop = 0.0
    cur_exp = current.roi_exposure_level
    base_exp = baseline.roi_exposure_level
    cur_exp_full = current.exposure_level
    base_exp_full = baseline.exposure_level
    getting_brighter = median_rise_abs > 0 or bright_rise > 0 or clip_rise > 0
    if (
        cur_exp is not None
        and base_exp is not None
        and base_exp > 1.0
        and cur_exp_full is not None
        and base_exp_full is not None
        and base_exp_full > 1.0
    ):
        roi_exp_rise = (cur_exp - base_exp) / base_exp
        full_exp_rise = (cur_exp_full - base_exp_full) / base_exp_full
        exp_rise = max(roi_exp_rise, full_exp_rise)
        if exp_rise >= alarm.exposure_rise_ratio:
            reasons.append(
                f"平均曝光水平上升 {exp_rise:.0%} "
                f"(几何均值 {base_exp:.1f}→{cur_exp:.1f}, "
                f"EV {(baseline.roi_exposure_ev or 0):.1f}→{(current.roi_exposure_ev or 0):.1f})"
            )
            getting_brighter = True

    cur_well = current.roi_well_exposed_ratio
    base_well = baseline.roi_well_exposed_ratio
    cur_well_full = current.well_exposed_ratio
    base_well_full = baseline.well_exposed_ratio
    if (
        cur_well is not None
        and base_well is not None
        and cur_well_full is not None
        and base_well_full is not None
    ):
        well_drop = max(base_well - cur_well, base_well_full - cur_well_full)
        if well_drop >= alarm.well_exposed_drop and getting_brighter:
            reasons.append(
                f"正常曝光像素下降 {well_drop:.0%} "
                f"(ROI {base_well:.1%}→{cur_well:.1%})"
            )

    return CompareResult(
        is_candidate=bool(reasons),
        reasons=reasons,
        median_rise_ratio=median_rise_ratio,
        median_rise_abs=median_rise_abs,
        bright_ratio_rise=bright_rise,
        clip_ratio_rise=clip_rise,
        block_brighten_fraction=bright_frac,
        valid_blocks=valid,
        brightened_blocks=brightened,
        roi_median=float(current.roi_median),
        baseline_roi_median=float(baseline.roi_median),
        roi_bright_ratio=float(current.roi_bright_ratio),
        baseline_roi_bright_ratio=float(baseline.roi_bright_ratio),
        roi_clip_ratio=float(current.roi_clip_ratio),
        baseline_roi_clip_ratio=float(baseline.roi_clip_ratio),
        exposure_level=current.roi_exposure_level,
        baseline_exposure_level=baseline.roi_exposure_level,
        exposure_ev=current.roi_exposure_ev,
        baseline_exposure_ev=baseline.roi_exposure_ev,
        exposure_rise_ratio=exp_rise,
        well_exposed_ratio=current.roi_well_exposed_ratio,
        baseline_well_exposed_ratio=baseline.roi_well_exposed_ratio,
        well_exposed_drop=well_drop,
    )


class OverexpDetector:
    def __init__(self, baseline: BrightnessFeatures, alarm: AlarmConfig) -> None:
        self.baseline = baseline
        self.alarm = alarm
        self.gate = TemporalGate(alarm.consecutive_seconds)

    def update(self, current: BrightnessFeatures, dt: float) -> DetectState:
        compare = compare_to_baseline(current, self.baseline, self.alarm)
        alarm, hold = self.gate.update(compare.is_candidate, dt)
        if alarm:
            status = "ALARM"
            message = "过曝告警: " + "; ".join(compare.reasons)
        elif compare.is_candidate:
            status = "CANDIDATE"
            remain = max(0.0, self.alarm.consecutive_seconds - hold)
            message = f"过曝候选，还需持续 {remain:.1f}s: " + "; ".join(compare.reasons)
        else:
            status = "OK"
            message = "曝光正常（相对本点位基准）"
        return DetectState(
            status=status,
            alarm=alarm,
            candidate=compare.is_candidate,
            hold_seconds=hold,
            compare=compare,
            message=message,
        )


def _parse_roi(raw: Any, default: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return default
    try:
        vals = [max(0.0, min(1.0, float(v))) for v in raw[:4]]
    except (TypeError, ValueError):
        return default
    x, y, w, h = vals
    if w <= 0 or h <= 0:
        return default
    return (x, y, min(w, 1.0 - x), min(h, 1.0 - y))


class GuobaoDetectorSkill(BaseSkill):
    """
    画面过曝检测技能（实时视频流）

    校准模板由外部上传/填写；在实时视频中按间隔比较亮度特征。
    """

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": SKILL_NAME,
        "name_zh": "画面过曝检测",
        "version": "1.0",
        "cover_image": f"/skills/{SKILL_NAME}.png",
        "description": (
            "基于上传的校准模板照片与实时视频帧，比较 ROI 中位数、亮像素比、"
            "高光裁切、分块变亮与曝光水平；过曝嫌疑持续达到设定秒数后告警。"
        ),
        "status": True,
        "run_mode": "stream",
        "required_models": [],
        "form_fields": [
            {
                "key": "reference_image_url",
                "label": "校准模板图片地址",
                "type": "text",
                "required": True,
                "default": "",
                "hint": "正常曝光画面的模板图 URL 或服务器本地路径，也可在任务配置中上传",
            },
            {
                "key": "confirm_count",
                "label": "连续确认次数",
                "type": "number",
                "required": False,
                "default": 3,
            },
            {
                "key": "cooldown_sec",
                "label": "告警冷却（秒）",
                "type": "number",
                "required": False,
                "default": 60,
            },
        ],
        "params": {
            "enable_default_sort_tracking": False,
            "enable_timing_log": False,
            "reference_image_url": "",
            "check_interval_sec": 2,
            "confirm_count": 3,
            "cooldown_sec": 60,
            "bright_pixel_threshold": 225,
            "clip_threshold": 250,
            "roi": [0.25, 0.25, 0.5, 0.5],
            "median_rise_ratio": 0.35,
            "min_median_rise_abs": 15.0,
            "bright_ratio_rise": 0.20,
            "clip_ratio_rise": 0.12,
            "block_rise_ratio": 0.30,
            "block_brighten_fraction": 0.40,
            "ignore_block_mean": 220.0,
            "consecutive_seconds": 6.0,
            "exposure_rise_ratio": 0.35,
            "well_exposed_drop": 0.20,
            "calibrate_frames": 15,
            "sample_fps": 2.0,
            "baseline_path": "",
            "baseline": {},
            "single_frame_as_final": False,
        },
        "alert_definitions": [
            {
                "key": "overexp_alarm",
                "level": 1,
                "description": "相对本点位基准判定画面过曝，并持续达到设定秒数后触发。",
                "codes": {
                    "overexp": {"code": "overexp", "description": "画面过曝"},
                },
            }
        ],
    }

    def _initialize(self) -> None:
        params = self.config.get("params", {}) if isinstance(self.config, dict) else {}

        self.feature_cfg = FeatureConfig(
            bright_pixel_threshold=int(params.get("bright_pixel_threshold", 225)),
            clip_threshold=int(params.get("clip_threshold", 250)),
            roi=_parse_roi(params.get("roi"), (0.25, 0.25, 0.5, 0.5)),
        )
        self.alarm_cfg = AlarmConfig(
            median_rise_ratio=float(params.get("median_rise_ratio", 0.35)),
            min_median_rise_abs=float(params.get("min_median_rise_abs", 15.0)),
            bright_ratio_rise=float(params.get("bright_ratio_rise", 0.20)),
            clip_ratio_rise=float(params.get("clip_ratio_rise", 0.12)),
            block_rise_ratio=float(params.get("block_rise_ratio", 0.30)),
            block_brighten_fraction=float(params.get("block_brighten_fraction", 0.40)),
            ignore_block_mean=float(params.get("ignore_block_mean", 220.0)),
            consecutive_seconds=float(params.get("consecutive_seconds", 6.0)),
            exposure_rise_ratio=float(params.get("exposure_rise_ratio", 0.35)),
            well_exposed_drop=float(params.get("well_exposed_drop", 0.20)),
        )
        self.reference_image_url = str(params.get("reference_image_url") or "").strip()
        self.check_interval_sec = max(1.0, float(params.get("check_interval_sec", 2) or 2))
        self.confirm_count = max(1, int(params.get("confirm_count", 3) or 3))
        self.cooldown_sec = max(0.0, float(params.get("cooldown_sec", 60) or 60))
        self.alarm_cfg.consecutive_seconds = float(self.confirm_count) * self.check_interval_sec
        self.calibrate_frames = max(1, int(params.get("calibrate_frames", 15) or 15))
        self.sample_fps = float(params.get("sample_fps", 2.0) or 2.0)
        self.default_dt = self.check_interval_sec
        self.single_frame_as_final = bool(params.get("single_frame_as_final", False))

        self._calibrate_buffer: List[BrightnessFeatures] = []
        self._detector: Optional[OverexpDetector] = None
        self._reference_bgr: Optional[np.ndarray] = None
        self._last_process_ts: Optional[float] = None
        self._last_status: Optional[str] = None
        self._last_alert_ts = 0.0
        self._last_result: Dict[str, Any] = {}
        self._last_roi = self.feature_cfg.roi
        self.alert_definitions: List[Dict[str, Any]] = list(
            self.config.get("alert_definitions") or []
        )

        if self.reference_image_url:
            try:
                self.reload_reference(self.reference_image_url)
            except Exception as e:
                self.log("warning", f"校准模板预加载失败: {e}")
        else:
            baseline = self._load_initial_baseline(params)
            if baseline is not None:
                self._detector = OverexpDetector(baseline, self.alarm_cfg)
                self.log(
                    "info",
                    f"初始化过曝检测技能: 已加载点位基准 ROI中位数={baseline.roi_median:.1f}",
                )

        self.log(
            "info",
            f"初始化过曝检测: interval={self.check_interval_sec}s "
            f"confirm={self.confirm_count} cooldown={self.cooldown_sec}s "
            f"roi={self.feature_cfg.roi}",
        )

    def get_required_models(self) -> List[str]:
        return []

    def reload_reference(self, url: str) -> None:
        bgr = pose_core.load_image_bgr(url)
        features = extract_features(bgr, self.feature_cfg)
        self._reference_bgr = bgr
        self._detector = OverexpDetector(features, self.alarm_cfg)
        self.reference_image_url = url
        self._last_status = None
        self._last_alert_ts = 0.0
        self.log(
            "info",
            f"已加载过曝校准模板: ROI中位数={features.roi_median:.1f}, "
            f"亮像素比={features.bright_ratio:.1%}, 高光裁切={features.clip_ratio:.1%}",
        )

    def ensure_reference(self) -> OverexpDetector:
        if self._detector is None:
            if not self.reference_image_url:
                raise ValueError("未配置校准模板图片地址")
            self.reload_reference(self.reference_image_url)
        assert self._detector is not None
        return self._detector

    def detect_frame(
        self, current_bgr: np.ndarray, *, dt: Optional[float] = None
    ) -> Dict[str, Any]:
        """对单张 BGR 帧做过曝检测（实时视频 / 离线测试入口）。"""
        self.ensure_reference()
        roi = self._resolve_roi(current_bgr, None, None)
        self.feature_cfg.roi = roi
        self._last_roi = roi
        current = extract_features(current_bgr, self.feature_cfg)
        step = float(dt) if dt is not None and dt > 0 else self.check_interval_sec
        state = self._detector.update(current, dt=step)

        now = time.time()
        prev_status = self._last_status
        has_status_change = prev_status is not None and state.status != prev_status
        rising_alarm = bool(state.alarm and prev_status != "ALARM")
        has_overexp_alarm = bool(
            rising_alarm and (now - self._last_alert_ts >= self.cooldown_sec)
        )
        if has_overexp_alarm:
            self._last_alert_ts = now
        if state.status != prev_status:
            self.log("info", f"过曝检测: {state.status}  {state.message}")
            self._last_status = state.status

        result = self._build_result(state, current, has_overexp_alarm, has_status_change)
        result["run_mode"] = "stream"
        self._last_result = result
        return result

    def _load_initial_baseline(self, params: Dict[str, Any]) -> Optional[BrightnessFeatures]:
        raw = params.get("baseline")
        if isinstance(raw, dict) and raw.get("features"):
            try:
                features = BrightnessFeatures.from_dict(raw["features"])
                roi = _parse_roi(raw.get("roi"), self.feature_cfg.roi)
                self.feature_cfg.roi = roi
                self._last_roi = roi
                if raw.get("bright_pixel_threshold") is not None:
                    self.feature_cfg.bright_pixel_threshold = int(raw["bright_pixel_threshold"])
                if raw.get("clip_threshold") is not None:
                    self.feature_cfg.clip_threshold = int(raw["clip_threshold"])
                return features
            except Exception as e:
                self.log("warning", f"params.baseline 解析失败，改为自动标定: {e}")
        elif isinstance(raw, dict) and raw.get("roi_median") is not None:
            try:
                return BrightnessFeatures.from_dict(raw)
            except Exception as e:
                self.log("warning", f"params.baseline 特征解析失败，改为自动标定: {e}")

        path_text = str(params.get("baseline_path") or "").strip()
        if not path_text:
            return None
        path = Path(path_text)
        if not path.is_absolute():
            path = (_CODE_ROOT / path).resolve()
        if not path.exists():
            self.log("warning", f"基准文件不存在: {path}，改为自动标定")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("features"), dict):
                roi = _parse_roi(data.get("roi"), self.feature_cfg.roi)
                self.feature_cfg.roi = roi
                self._last_roi = roi
                if data.get("bright_pixel_threshold") is not None:
                    self.feature_cfg.bright_pixel_threshold = int(data["bright_pixel_threshold"])
                if data.get("clip_threshold") is not None:
                    self.feature_cfg.clip_threshold = int(data["clip_threshold"])
                return BrightnessFeatures.from_dict(data["features"])
            return BrightnessFeatures.from_dict(data)
        except Exception as e:
            self.log("warning", f"读取基准文件失败: {path}, {e}")
            return None

    def _roi_from_fence(
        self,
        fence_config: Optional[Dict[str, Any]],
        image_size: Tuple[int, int],
    ) -> Optional[Tuple[float, float, float, float]]:
        if not self.is_fence_config_valid(fence_config):
            return None
        width, height = image_size
        if width <= 0 or height <= 0:
            return None
        xs: List[float] = []
        ys: List[float] = []
        for region in self._fence_regions(fence_config):
            for x, y in region.get("points") or []:
                xs.append(float(x))
                ys.append(float(y))
        if not xs:
            return None
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        w, h = max(0.01, x2 - x1), max(0.01, y2 - y1)
        return (x1, y1, w, h)

    def _resolve_roi(
        self,
        image: np.ndarray,
        fence_config: Optional[Dict[str, Any]],
        input_roi: Any,
    ) -> Tuple[float, float, float, float]:
        height, width = image.shape[:2]
        if input_roi is not None:
            return _parse_roi(input_roi, self.feature_cfg.roi)
        fence_roi = self._roi_from_fence(fence_config, (width, height))
        if fence_roi is not None:
            return fence_roi
        return self.feature_cfg.roi

    def _load_image(
        self,
        input_data: Union[np.ndarray, str, Dict[str, Any], Any],
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Any, Optional[str]]:
        image = None
        fence_config = None
        input_roi = None

        if isinstance(input_data, np.ndarray):
            image = input_data.copy()
        elif isinstance(input_data, str):
            image = cv2.imread(input_data)
            if image is None:
                return None, None, None, f"无法加载图像: {input_data}"
        elif isinstance(input_data, dict):
            if "image" not in input_data:
                return None, None, None, "输入字典中缺少 'image' 字段"
            image_data = input_data["image"]
            if isinstance(image_data, np.ndarray):
                image = image_data.copy()
            elif isinstance(image_data, str):
                image = cv2.imread(image_data)
                if image is None:
                    return None, None, None, f"无法加载图像: {image_data}"
            else:
                return None, None, None, "不支持的图像数据类型"
            fence_config = input_data.get("fence_config")
            input_roi = input_data.get("roi")
            baseline = input_data.get("baseline")
            if self._detector is None and isinstance(baseline, dict):
                loaded = self._load_initial_baseline({"baseline": baseline})
                if loaded is not None:
                    self._detector = OverexpDetector(loaded, self.alarm_cfg)
        else:
            return None, None, None, "不支持的输入数据类型"

        if image is None or image.size == 0:
            return None, None, None, "无效的图像数据"
        return image, fence_config, input_roi, None

    def _compute_dt(self, is_single_image: bool) -> float:
        now = time.monotonic()
        prev = self._last_process_ts
        self._last_process_ts = now
        if is_single_image and self.single_frame_as_final:
            return float(self.alarm_cfg.consecutive_seconds)
        if prev is None:
            return self.default_dt
        dt = now - prev
        if dt <= 0:
            return self.default_dt
        return min(max(dt, 0.01), 5.0)

    def _finish_calibration(self) -> Optional[BrightnessFeatures]:
        if not self._calibrate_buffer:
            return None
        features = median_features(self._calibrate_buffer)
        self._detector = OverexpDetector(features, self.alarm_cfg)
        self._calibrate_buffer = []
        self.log(
            "info",
            f"过曝检测自动标定完成: ROI中位数={features.roi_median:.1f}, "
            f"亮像素比={features.bright_ratio:.1%}, "
            f"曝光水平={features.roi_exposure_level:.1f}",
        )
        return features

    def _calibrating_state(self, current: BrightnessFeatures) -> DetectState:
        remain = max(0, self.calibrate_frames - len(self._calibrate_buffer))
        return DetectState(
            status="CALIBRATING",
            alarm=False,
            candidate=False,
            hold_seconds=0.0,
            compare=CompareResult(
                is_candidate=False,
                roi_median=current.roi_median,
                roi_bright_ratio=current.roi_bright_ratio,
                roi_clip_ratio=current.roi_clip_ratio,
            ),
            message=f"正在标定点位基准 {len(self._calibrate_buffer)}/{self.calibrate_frames}，剩余 {remain} 帧",
        )

    def process(
        self,
        input_data: Union[np.ndarray, str, Dict[str, Any], Any],
        fence_config: Dict = None,
    ) -> SkillResult:
        try:
            image, input_fence_config, input_roi, error_message = self._load_image(input_data)
            if error_message:
                return SkillResult.error_result(error_message)
            if input_fence_config is not None:
                fence_config = input_fence_config
            if input_roi is not None or fence_config is not None:
                self.feature_cfg.roi = self._resolve_roi(image, fence_config, input_roi)
                self._last_roi = self.feature_cfg.roi

            now = time.time()
            if (
                self._last_result
                and self._last_process_ts is not None
                and now - self._last_process_ts < self.check_interval_sec
            ):
                return SkillResult.success_result(self._last_result)

            dt = self.check_interval_sec
            if self._last_process_ts is not None:
                dt = max(0.01, now - self._last_process_ts)
            data = self.detect_frame(image, dt=dt)
            self._last_process_ts = now
            return SkillResult.success_result(data)
        except Exception as e:
            logger.exception(f"过曝检测技能处理失败: {str(e)}")
            return SkillResult.error_result(f"处理失败: {str(e)}")

    def _build_result(
        self,
        state: DetectState,
        current: BrightnessFeatures,
        has_overexp_alarm: bool,
        has_status_change: bool,
    ) -> Dict[str, Any]:
        x1, y1, x2, y2 = roi_rect_px(
            (current.height, current.width),
            self.feature_cfg.roi,
        )
        alert_def = self.alert_definitions[0] if self.alert_definitions else {}
        active_alerts: List[Dict[str, Any]] = []
        recognition_types: List[str] = []
        if has_overexp_alarm:
            recognition_types = ["overexp"]
            active_alerts.append(
                {
                    "key": alert_def.get("key", "overexp_alarm"),
                    "level": int(alert_def.get("level", 1) or 1),
                    "description": state.message,
                    "recognition_type": "overexp",
                    "violation_type": "画面过曝",
                    "events": [
                        {
                            "status": state.status,
                            "message": state.message,
                            "roi_median": state.compare.roi_median,
                            "baseline_roi_median": state.compare.baseline_roi_median,
                        }
                    ],
                }
            )

        return {
            "detections": [],
            "count": 1 if state.alarm else 0,
            "skill_name": self.config.get("name") or SKILL_NAME,
            "status": state.status,
            "message": state.message,
            "has_overexp_alarm": has_overexp_alarm,
            "has_overexp_status_change": has_status_change,
            "has_person_count_change": False,
            "recognition_types": recognition_types,
            "alert_definitions": list(self.alert_definitions),
            "active_alerts": active_alerts,
            "roi": list(self.feature_cfg.roi),
            "roi_box": [x1, y1, x2, y2],
            "features": {
                "roi_median": current.roi_median,
                "roi_bright_ratio": current.roi_bright_ratio,
                "roi_clip_ratio": current.roi_clip_ratio,
                "roi_exposure_level": current.roi_exposure_level,
                "roi_exposure_ev": current.roi_exposure_ev,
                "roi_well_exposed_ratio": current.roi_well_exposed_ratio,
            },
            "detect_state": state.to_dict(),
            "safety_metrics": {
                "is_safe": not state.alarm,
                "alert_info": {
                    "alert_triggered": has_overexp_alarm,
                    "alert_name": "画面过曝预警" if has_overexp_alarm else "",
                    "alert_type": "过曝检测",
                    "alert_description": state.message if has_overexp_alarm else "",
                },
            },
            "flow_metrics": {
                "status": state.status,
                "hold_seconds": state.hold_seconds,
                "roi_median": state.compare.roi_median,
                "baseline_roi_median": state.compare.baseline_roi_median,
            },
        }

    def _draw_detections_on_frame(self, frame: np.ndarray, alert_data: Dict) -> np.ndarray:
        """在帧上绘制 ROI 与过曝状态。"""
        try:
            vis = frame
            state = alert_data.get("detect_state") or {}
            status = str(state.get("status") or alert_data.get("status") or "OK")
            color = _STATUS_COLOR.get(status, (200, 200, 200))
            roi_box = alert_data.get("roi_box")
            if isinstance(roi_box, (list, tuple)) and len(roi_box) >= 4:
                x1, y1, x2, y2 = [int(v) for v in roi_box[:4]]
            else:
                roi = _parse_roi(alert_data.get("roi"), self._last_roi)
                x1, y1, x2, y2 = roi_rect_px(vis.shape, roi)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"视频暗曝状态：{status}"
            return self._put_status_text(vis, text, (16, 12), color)
        except Exception as e:
            logger.error(f"绘制过曝检测结果时出错: {str(e)}")
            return frame

    @staticmethod
    def _put_status_text(
        frame_bgr: np.ndarray,
        text: str,
        xy: Tuple[int, int],
        color_bgr: Tuple[int, int, int],
        size: int = 32,
    ) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            cv2.putText(
                frame_bgr, text, xy,
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 2,
            )
            return frame_bgr

        font = None
        for path in _FONT_CANDIDATES:
            if path.exists():
                font = ImageFont.truetype(str(path), size=size)
                break
        if font is None:
            font = ImageFont.load_default()

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        x, y = xy
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        draw.text((x, y), text, font=font, fill=color_rgb)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    img_dir = _CODE_ROOT / "test_images_videos"
    candidates = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"未找到测试图片: {img_dir}")
    img_name = str(candidates[0])
    cfg = dict(GuobaoDetectorSkill.DEFAULT_CONFIG)
    cfg["params"] = dict(cfg.get("params") or {})
    cfg["params"]["reference_image_url"] = img_name
    skill = GuobaoDetectorSkill(cfg)
    frame = cv2.imread(img_name)
    if frame is None:
        raise FileNotFoundError(f"无法加载测试图片: {img_name}")
    result = skill.process(img_name)
    print(result.to_dict())
    if result.success and result.data:
        ploted_img = skill._draw_detections_on_frame(frame.copy(), result.to_dict()["data"])
        out_path = str(Path(img_name).with_name(Path(img_name).stem + "_guobao_ploted.jpg"))
        cv2.imwrite(out_path, ploted_img)
        print(f"结果已保存: {out_path}")
    else:
        print(f"检测失败: {result.error_message}")
