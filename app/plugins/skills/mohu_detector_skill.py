"""
画面模糊检测技能（点位基准）

规则：
  - 不依赖 Triton / YOLO，按当前帧锐度特征与点位清晰基准比较
  - 校准模板由外部上传；实时视频中比较 ROI Laplacian / Tenengrad / 分块变糊
  - 过暗、过曝时抑制模糊误报
  - 连续确认后告警，持续模糊按冷却周期重复落库
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
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

SKILL_NAME = "mohu_detector_skill"

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


@dataclass
class FeatureConfig:
    dark_pixel_threshold: int = 30
    clip_threshold: int = 250
    roi: Tuple[float, float, float, float] = (0.25, 0.25, 0.5, 0.5)


@dataclass
class AlarmConfig:
    laplacian_drop_ratio: float = 0.45
    min_laplacian_drop_abs: float = 20.0
    tenengrad_drop_ratio: float = 0.40
    min_tenengrad_drop_abs: float = 80.0
    block_drop_ratio: float = 0.50
    block_blur_fraction: float = 0.40
    min_valid_blocks: int = 8
    ignore_block_laplacian: float = 40.0
    ignore_block_mean_dark: float = 30.0
    ignore_block_mean_clip: float = 230.0
    suppress_roi_median: float = 25.0
    suppress_clip_ratio: float = 0.25
    consecutive_seconds: float = 6.0


@dataclass
class SharpnessFeatures:
    width: int
    height: int
    mean: float
    median: float
    std: float
    clip_ratio: float
    dark_ratio: float
    laplacian_var: float
    tenengrad: float
    roi_mean: float
    roi_median: float
    roi_std: float
    roi_clip_ratio: float
    roi_dark_ratio: float
    roi_laplacian_var: float
    roi_tenengrad: float
    block_means_8x8: List[List[float]] = field(default_factory=list)
    block_laplacian_8x8: List[List[float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharpnessFeatures":
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            mean=float(data["mean"]),
            median=float(data["median"]),
            std=float(data["std"]),
            clip_ratio=float(data["clip_ratio"]),
            dark_ratio=float(data["dark_ratio"]),
            laplacian_var=float(data["laplacian_var"]),
            tenengrad=float(data["tenengrad"]),
            roi_mean=float(data["roi_mean"]),
            roi_median=float(data["roi_median"]),
            roi_std=float(data["roi_std"]),
            roi_clip_ratio=float(data["roi_clip_ratio"]),
            roi_dark_ratio=float(data["roi_dark_ratio"]),
            roi_laplacian_var=float(data["roi_laplacian_var"]),
            roi_tenengrad=float(data["roi_tenengrad"]),
            block_means_8x8=data.get("block_means_8x8") or [],
            block_laplacian_8x8=data.get("block_laplacian_8x8") or [],
        )


@dataclass
class CompareResult:
    is_candidate: bool
    reasons: List[str] = field(default_factory=list)
    suppressed: bool = False
    suppress_reason: str = ""
    laplacian_drop_ratio: float = 0.0
    laplacian_drop_abs: float = 0.0
    tenengrad_drop_ratio: float = 0.0
    tenengrad_drop_abs: float = 0.0
    block_blur_fraction: float = 0.0
    valid_blocks: int = 0
    blurred_blocks: int = 0
    roi_laplacian: float = 0.0
    baseline_roi_laplacian: float = 0.0
    roi_tenengrad: float = 0.0
    baseline_roi_tenengrad: float = 0.0
    roi_median: float = 0.0
    baseline_roi_median: float = 0.0
    roi_clip_ratio: float = 0.0
    baseline_roi_clip_ratio: float = 0.0

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
    """连续若干秒都判定为模糊候选后才升为告警。"""

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


def _to_y(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


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


def laplacian_var(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(cv2.Laplacian(y, cv2.CV_64F).var())


def tenengrad(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    gx = cv2.Sobel(y, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_64F, 0, 1, ksize=3)
    return float((gx * gx + gy * gy).mean())


def _block_stats(y: np.ndarray, grid: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    h, w = y.shape
    bh, bw = max(1, h // grid), max(1, w // grid)
    means = np.zeros((grid, grid), dtype=np.float32)
    laps = np.zeros((grid, grid), dtype=np.float32)
    for i in range(grid):
        y1, y2 = i * bh, h if i == grid - 1 else (i + 1) * bh
        for j in range(grid):
            x1, x2 = j * bw, w if j == grid - 1 else (j + 1) * bw
            block = y[y1:y2, x1:x2]
            means[i, j] = float(block.mean())
            laps[i, j] = laplacian_var(block)
    return means, laps


def _as_nested(arr: np.ndarray) -> List[List[float]]:
    return [[round(float(v), 3) for v in row] for row in arr]


def extract_features(frame_bgr: np.ndarray, cfg: FeatureConfig) -> SharpnessFeatures:
    if frame_bgr is None or frame_bgr.size == 0:
        raise ValueError("空帧，无法提取锐度特征")
    y = _to_y(frame_bgr)
    h, w = y.shape
    roi = _crop_roi(y, cfg.roi)
    dark_thr = cfg.dark_pixel_threshold
    clip_thr = cfg.clip_threshold
    yf = y.astype(np.float32)
    roif = roi.astype(np.float32)
    means8, laps8 = _block_stats(roi, 8)
    return SharpnessFeatures(
        width=int(w),
        height=int(h),
        mean=float(yf.mean()),
        median=float(np.median(yf)),
        std=float(yf.std()),
        clip_ratio=float((yf >= clip_thr).mean()),
        dark_ratio=float((yf <= dark_thr).mean()),
        laplacian_var=laplacian_var(y),
        tenengrad=tenengrad(y),
        roi_mean=float(roif.mean()),
        roi_median=float(np.median(roif)),
        roi_std=float(roif.std()),
        roi_clip_ratio=float((roif >= clip_thr).mean()),
        roi_dark_ratio=float((roif <= dark_thr).mean()),
        roi_laplacian_var=laplacian_var(roi),
        roi_tenengrad=tenengrad(roi),
        block_means_8x8=_as_nested(means8),
        block_laplacian_8x8=_as_nested(laps8),
    )


def median_features(items: List[SharpnessFeatures]) -> SharpnessFeatures:
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
    return SharpnessFeatures(
        width=first.width,
        height=first.height,
        mean=med_scalar("mean"),
        median=med_scalar("median"),
        std=med_scalar("std"),
        clip_ratio=med_scalar("clip_ratio"),
        dark_ratio=med_scalar("dark_ratio"),
        laplacian_var=med_scalar("laplacian_var"),
        tenengrad=med_scalar("tenengrad"),
        roi_mean=med_scalar("roi_mean"),
        roi_median=med_scalar("roi_median"),
        roi_std=med_scalar("roi_std"),
        roi_clip_ratio=med_scalar("roi_clip_ratio"),
        roi_dark_ratio=med_scalar("roi_dark_ratio"),
        roi_laplacian_var=med_scalar("roi_laplacian_var"),
        roi_tenengrad=med_scalar("roi_tenengrad"),
        block_means_8x8=med_grid("block_means_8x8"),
        block_laplacian_8x8=med_grid("block_laplacian_8x8"),
    )


def _as_array(grid: List[List[float]]) -> np.ndarray:
    return np.asarray(grid, dtype=np.float32)


def compare_to_baseline(
    current: SharpnessFeatures,
    baseline: SharpnessFeatures,
    alarm: AlarmConfig,
) -> CompareResult:
    lap_drop_abs = float(baseline.roi_laplacian_var - current.roi_laplacian_var)
    base_lap = max(float(baseline.roi_laplacian_var), 1.0)
    lap_drop_ratio = lap_drop_abs / base_lap

    ten_drop_abs = float(baseline.roi_tenengrad - current.roi_tenengrad)
    base_ten = max(float(baseline.roi_tenengrad), 1.0)
    ten_drop_ratio = ten_drop_abs / base_ten

    base_laps = _as_array(baseline.block_laplacian_8x8)
    cur_laps = _as_array(current.block_laplacian_8x8)
    base_means = _as_array(baseline.block_means_8x8)
    cur_means = _as_array(current.block_means_8x8)
    if cur_laps.shape != base_laps.shape or cur_means.shape != base_means.shape:
        blur_frac = 0.0
        valid = 0
        blurred = 0
    else:
        valid_mask = (
            (base_laps >= alarm.ignore_block_laplacian)
            & (base_means >= alarm.ignore_block_mean_dark)
            & (base_means <= alarm.ignore_block_mean_clip)
            & (cur_means >= alarm.ignore_block_mean_dark)
            & (cur_means <= alarm.ignore_block_mean_clip)
        )
        drops = (base_laps - cur_laps) / np.maximum(base_laps, 1.0)
        blurred_mask = valid_mask & (drops >= alarm.block_drop_ratio)
        valid = int(valid_mask.sum())
        blurred = int(blurred_mask.sum())
        blur_frac = float(blurred / valid) if valid else 0.0

    reasons: List[str] = []
    suppressed = False
    suppress_reason = ""
    if current.roi_median <= alarm.suppress_roi_median:
        suppressed = True
        suppress_reason = f"ROI过暗(中位数 {current.roi_median:.1f})，不报模糊"
    elif current.roi_clip_ratio >= alarm.suppress_clip_ratio:
        suppressed = True
        suppress_reason = f"ROI裁切过高({current.roi_clip_ratio:.0%})，不报模糊"
    else:
        lap_hit = (
            lap_drop_ratio >= alarm.laplacian_drop_ratio
            and lap_drop_abs >= alarm.min_laplacian_drop_abs
        )
        if lap_hit:
            reasons.append(
                f"ROI Laplacian下降 {lap_drop_ratio:.0%} "
                f"({baseline.roi_laplacian_var:.1f}→{current.roi_laplacian_var:.1f})"
            )
        ten_hit = (
            ten_drop_ratio >= alarm.tenengrad_drop_ratio
            and ten_drop_abs >= alarm.min_tenengrad_drop_abs
        )
        if ten_hit:
            reasons.append(
                f"ROI Tenengrad下降 {ten_drop_ratio:.0%} "
                f"({baseline.roi_tenengrad:.1f}→{current.roi_tenengrad:.1f})"
            )
        if blur_frac >= alarm.block_blur_fraction and valid >= alarm.min_valid_blocks:
            reasons.append(f"分块变糊 {blurred}/{valid} ({blur_frac:.0%})")

    return CompareResult(
        is_candidate=bool(reasons) and not suppressed,
        reasons=reasons,
        suppressed=suppressed,
        suppress_reason=suppress_reason,
        laplacian_drop_ratio=lap_drop_ratio,
        laplacian_drop_abs=lap_drop_abs,
        tenengrad_drop_ratio=ten_drop_ratio,
        tenengrad_drop_abs=ten_drop_abs,
        block_blur_fraction=blur_frac,
        valid_blocks=valid,
        blurred_blocks=blurred,
        roi_laplacian=float(current.roi_laplacian_var),
        baseline_roi_laplacian=float(baseline.roi_laplacian_var),
        roi_tenengrad=float(current.roi_tenengrad),
        baseline_roi_tenengrad=float(baseline.roi_tenengrad),
        roi_median=float(current.roi_median),
        baseline_roi_median=float(baseline.roi_median),
        roi_clip_ratio=float(current.roi_clip_ratio),
        baseline_roi_clip_ratio=float(baseline.roi_clip_ratio),
    )


class BlurDetector:
    def __init__(self, baseline: SharpnessFeatures, alarm: AlarmConfig) -> None:
        self.baseline = baseline
        self.alarm = alarm
        self.gate = TemporalGate(alarm.consecutive_seconds)

    def update(self, current: SharpnessFeatures, dt: float) -> DetectState:
        compare = compare_to_baseline(current, self.baseline, self.alarm)
        alarm, hold = self.gate.update(compare.is_candidate, dt)
        if alarm:
            status = "ALARM"
            message = "模糊告警: " + "; ".join(compare.reasons)
        elif compare.suppressed:
            status = "OK"
            message = compare.suppress_reason
        elif compare.is_candidate:
            status = "CANDIDATE"
            remain = max(0.0, self.alarm.consecutive_seconds - hold)
            message = f"模糊候选，还需持续 {remain:.1f}s: " + "; ".join(compare.reasons)
        else:
            status = "OK"
            message = "清晰正常（相对本点位基准）"
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


class MohuDetectorSkill(BaseSkill):
    """
    画面模糊检测技能（实时视频流）

    校准模板由外部上传/填写；在实时视频中按间隔比较锐度特征。
    """

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": SKILL_NAME,
        "name_zh": "画面模糊检测",
        "version": "1.0",
        "cover_image": f"/skills/{SKILL_NAME}.png",
        "description": (
            "基于上传的清晰校准模板与实时视频帧，比较 ROI Laplacian 方差、"
            "Tenengrad 与分块变糊；过暗/过曝时抑制误报，连续确认后告警。"
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
                "hint": "对焦正常的清晰画面模板图 URL 或服务器本地路径，也可在任务配置中上传",
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
                "default": 5,
                "hint": "画面持续模糊时，每隔多少秒再写一条报警",
            },
        ],
        "params": {
            "enable_default_sort_tracking": False,
            "enable_timing_log": False,
            "reference_image_url": "",
            "check_interval_sec": 2,
            "confirm_count": 3,
            "cooldown_sec": 5,
            "dark_pixel_threshold": 30,
            "clip_threshold": 250,
            "roi": [0.25, 0.25, 0.5, 0.5],
            "laplacian_drop_ratio": 0.45,
            "min_laplacian_drop_abs": 20.0,
            "tenengrad_drop_ratio": 0.40,
            "min_tenengrad_drop_abs": 80.0,
            "block_drop_ratio": 0.50,
            "block_blur_fraction": 0.40,
            "min_valid_blocks": 8,
            "ignore_block_laplacian": 40.0,
            "ignore_block_mean_dark": 30.0,
            "ignore_block_mean_clip": 230.0,
            "suppress_roi_median": 25.0,
            "suppress_clip_ratio": 0.25,
            "consecutive_seconds": 6.0,
            "calibrate_frames": 15,
            "sample_fps": 2.0,
            "baseline_path": "",
            "baseline": {},
            "single_frame_as_final": False,
        },
        "alert_definitions": [
            {
                "key": "blur_alarm",
                "level": 1,
                "description": "相对本点位清晰基准判定画面模糊，并持续达到设定秒数后触发。",
                "codes": {
                    "blur": {
                        "code": "blur",
                        "description": "图像模糊",
                        "analysis_case": "0005",
                    },
                },
            }
        ],
    }

    def _initialize(self) -> None:
        params = self.config.get("params", {}) if isinstance(self.config, dict) else {}

        self.feature_cfg = FeatureConfig(
            dark_pixel_threshold=int(params.get("dark_pixel_threshold", 30)),
            clip_threshold=int(params.get("clip_threshold", 250)),
            roi=_parse_roi(params.get("roi"), (0.25, 0.25, 0.5, 0.5)),
        )
        self.alarm_cfg = AlarmConfig(
            laplacian_drop_ratio=float(params.get("laplacian_drop_ratio", 0.45)),
            min_laplacian_drop_abs=float(params.get("min_laplacian_drop_abs", 20.0)),
            tenengrad_drop_ratio=float(params.get("tenengrad_drop_ratio", 0.40)),
            min_tenengrad_drop_abs=float(params.get("min_tenengrad_drop_abs", 80.0)),
            block_drop_ratio=float(params.get("block_drop_ratio", 0.50)),
            block_blur_fraction=float(params.get("block_blur_fraction", 0.40)),
            min_valid_blocks=int(params.get("min_valid_blocks", 8)),
            ignore_block_laplacian=float(params.get("ignore_block_laplacian", 40.0)),
            ignore_block_mean_dark=float(params.get("ignore_block_mean_dark", 30.0)),
            ignore_block_mean_clip=float(params.get("ignore_block_mean_clip", 230.0)),
            suppress_roi_median=float(params.get("suppress_roi_median", 25.0)),
            suppress_clip_ratio=float(params.get("suppress_clip_ratio", 0.25)),
            consecutive_seconds=float(params.get("consecutive_seconds", 6.0)),
        )
        self.reference_image_url = str(params.get("reference_image_url") or "").strip()
        self.check_interval_sec = max(1.0, float(params.get("check_interval_sec", 2) or 2))
        self.confirm_count = max(1, int(params.get("confirm_count", 3) or 3))
        self.cooldown_sec = max(0.0, float(params.get("cooldown_sec", 5) or 5))
        self._repeat_sec = self.cooldown_sec if self.cooldown_sec > 0 else max(1.0, self.check_interval_sec)
        if self._repeat_sec >= 30:
            self._repeat_sec = max(5.0, self.check_interval_sec)
        self.alarm_cfg.consecutive_seconds = float(self.confirm_count) * self.check_interval_sec
        self.calibrate_frames = max(1, int(params.get("calibrate_frames", 15) or 15))
        self.sample_fps = float(params.get("sample_fps", 2.0) or 2.0)
        self.default_dt = self.check_interval_sec
        self.single_frame_as_final = bool(params.get("single_frame_as_final", False))

        self._calibrate_buffer: List[SharpnessFeatures] = []
        self._detector: Optional[BlurDetector] = None
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
                self._detector = BlurDetector(baseline, self.alarm_cfg)
                self.log(
                    "info",
                    f"初始化模糊检测技能: ROI Laplacian={baseline.roi_laplacian_var:.1f} "
                    f"Tenengrad={baseline.roi_tenengrad:.1f}",
                )

        self.log(
            "info",
            f"初始化模糊检测: interval={self.check_interval_sec}s "
            f"confirm={self.confirm_count} cooldown={self.cooldown_sec}s "
            f"repeat={self._repeat_sec}s roi={self.feature_cfg.roi}",
        )

    def get_required_models(self) -> List[str]:
        return []

    def reload_reference(self, url: str) -> None:
        bgr = pose_core.load_image_bgr(url)
        features = extract_features(bgr, self.feature_cfg)
        self._reference_bgr = bgr
        self._detector = BlurDetector(features, self.alarm_cfg)
        self.reference_image_url = url
        self._last_status = None
        self._last_alert_ts = 0.0
        self.log(
            "info",
            f"已加载模糊校准模板: ROI Laplacian={features.roi_laplacian_var:.1f}, "
            f"Tenengrad={features.roi_tenengrad:.1f}",
        )

    def ensure_reference(self) -> BlurDetector:
        if self._detector is None:
            if not self.reference_image_url:
                raise ValueError("未配置校准模板图片地址")
            self.reload_reference(self.reference_image_url)
        assert self._detector is not None
        return self._detector

    def detect_frame(
        self, current_bgr: np.ndarray, *, dt: Optional[float] = None
    ) -> Dict[str, Any]:
        """对单张 BGR 帧做模糊检测（实时视频 / 离线测试入口）。"""
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
        has_blur_alarm = bool(
            state.alarm and (now - self._last_alert_ts >= self._repeat_sec)
        )
        if has_blur_alarm:
            self._last_alert_ts = now
        if state.status != prev_status or has_blur_alarm:
            self.log(
                "info",
                f"模糊检测: {state.status} emit={int(has_blur_alarm)} "
                f"hold={state.hold_seconds:.1f}s repeat={self._repeat_sec:.1f}s "
                f"{state.message}",
            )
            self._last_status = state.status

        result = self._build_result(state, current, has_blur_alarm, has_status_change)
        result["run_mode"] = "stream"
        self._last_result = result
        return result

    def _load_initial_baseline(self, params: Dict[str, Any]) -> Optional[SharpnessFeatures]:
        raw = params.get("baseline")
        if isinstance(raw, dict) and raw.get("features"):
            try:
                features = SharpnessFeatures.from_dict(raw["features"])
                roi = _parse_roi(raw.get("roi"), self.feature_cfg.roi)
                self.feature_cfg.roi = roi
                self._last_roi = roi
                if raw.get("dark_pixel_threshold") is not None:
                    self.feature_cfg.dark_pixel_threshold = int(raw["dark_pixel_threshold"])
                if raw.get("clip_threshold") is not None:
                    self.feature_cfg.clip_threshold = int(raw["clip_threshold"])
                return features
            except Exception as e:
                self.log("warning", f"params.baseline 解析失败: {e}")
        elif isinstance(raw, dict) and raw.get("roi_laplacian_var") is not None:
            try:
                return SharpnessFeatures.from_dict(raw)
            except Exception as e:
                self.log("warning", f"params.baseline 特征解析失败: {e}")

        path_text = str(params.get("baseline_path") or "").strip()
        if not path_text:
            return None
        path = Path(path_text)
        if not path.is_absolute():
            path = (_CODE_ROOT / path).resolve()
        if not path.exists():
            self.log("warning", f"基准文件不存在: {path}")
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("features"), dict):
                roi = _parse_roi(data.get("roi"), self.feature_cfg.roi)
                self.feature_cfg.roi = roi
                self._last_roi = roi
                if data.get("dark_pixel_threshold") is not None:
                    self.feature_cfg.dark_pixel_threshold = int(data["dark_pixel_threshold"])
                if data.get("clip_threshold") is not None:
                    self.feature_cfg.clip_threshold = int(data["clip_threshold"])
                return SharpnessFeatures.from_dict(data["features"])
            return SharpnessFeatures.from_dict(data)
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
                    self._detector = BlurDetector(loaded, self.alarm_cfg)
        else:
            return None, None, None, "不支持的输入数据类型"

        if image is None or image.size == 0:
            return None, None, None, "无效的图像数据"
        return image, fence_config, input_roi, None

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
                return SkillResult.success_result(
                    self._result_without_emit(self._last_result)
                )

            dt = self.check_interval_sec
            if self._last_process_ts is not None:
                dt = max(0.01, now - self._last_process_ts)
            data = self.detect_frame(image, dt=dt)
            self._last_process_ts = now
            return SkillResult.success_result(data)
        except Exception as e:
            logger.exception(f"模糊检测技能处理失败: {str(e)}")
            return SkillResult.error_result(f"处理失败: {str(e)}")

    @staticmethod
    def _result_without_emit(result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result)
        out["has_blur_alarm"] = False
        out["recognition_types"] = []
        out["active_alerts"] = []
        safety = dict(out.get("safety_metrics") or {})
        alert_info = dict(safety.get("alert_info") or {})
        alert_info["alert_triggered"] = False
        safety["alert_info"] = alert_info
        out["safety_metrics"] = safety
        return out

    def _build_result(
        self,
        state: DetectState,
        current: SharpnessFeatures,
        has_blur_alarm: bool,
        has_status_change: bool,
    ) -> Dict[str, Any]:
        x1, y1, x2, y2 = roi_rect_px(
            (current.height, current.width),
            self.feature_cfg.roi,
        )
        alert_def = self.alert_definitions[0] if self.alert_definitions else {}
        active_alerts: List[Dict[str, Any]] = []
        recognition_types: List[str] = []
        if has_blur_alarm:
            recognition_types = ["blur"]
            active_alerts.append(
                {
                    "key": alert_def.get("key", "blur_alarm"),
                    "level": int(alert_def.get("level", 1) or 1),
                    "description": state.message,
                    "recognition_type": "blur",
                    "analysis_case": "0005",
                    "violation_type": "图像模糊",
                    "events": [
                        {
                            "status": state.status,
                            "message": state.message,
                            "roi_laplacian": state.compare.roi_laplacian,
                            "baseline_roi_laplacian": state.compare.baseline_roi_laplacian,
                            "roi_tenengrad": state.compare.roi_tenengrad,
                            "baseline_roi_tenengrad": state.compare.baseline_roi_tenengrad,
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
            "has_blur_alarm": has_blur_alarm,
            "has_blur_status_change": has_status_change,
            "has_person_count_change": False,
            "recognition_types": recognition_types,
            "alert_definitions": list(self.alert_definitions),
            "active_alerts": active_alerts,
            "roi": list(self.feature_cfg.roi),
            "roi_box": [x1, y1, x2, y2],
            "features": {
                "roi_laplacian_var": current.roi_laplacian_var,
                "roi_tenengrad": current.roi_tenengrad,
                "roi_median": current.roi_median,
                "roi_clip_ratio": current.roi_clip_ratio,
            },
            "detect_state": state.to_dict(),
            "safety_metrics": {
                "is_safe": not state.alarm,
                "alert_info": {
                    "alert_triggered": has_blur_alarm,
                    "alert_name": "图像模糊预警" if has_blur_alarm else "",
                    "alert_type": "模糊检测",
                    "alert_description": state.message if has_blur_alarm else "",
                },
            },
            "flow_metrics": {
                "status": state.status,
                "hold_seconds": state.hold_seconds,
                "roi_laplacian": state.compare.roi_laplacian,
                "baseline_roi_laplacian": state.compare.baseline_roi_laplacian,
            },
        }

    def _draw_detections_on_frame(self, frame: np.ndarray, alert_data: Dict) -> np.ndarray:
        """在帧上绘制 ROI 与模糊状态。"""
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
            text = f"视频模糊状态：{status}"
            return self._put_status_text(vis, text, (16, 12), color)
        except Exception as e:
            logger.error(f"绘制模糊检测结果时出错: {str(e)}")
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
    cfg = dict(MohuDetectorSkill.DEFAULT_CONFIG)
    cfg["params"] = dict(cfg.get("params") or {})
    cfg["params"]["reference_image_url"] = img_name
    skill = MohuDetectorSkill(cfg)
    frame = cv2.imread(img_name)
    if frame is None:
        raise FileNotFoundError(f"无法加载测试图片: {img_name}")
    result = skill.process(img_name)
    print(result.to_dict())
    if result.success and result.data:
        ploted_img = skill._draw_detections_on_frame(frame.copy(), result.to_dict()["data"])
        out_path = str(Path(img_name).with_name(Path(img_name).stem + "_mohu_ploted.jpg"))
        cv2.imwrite(out_path, ploted_img)
        print(f"结果已保存: {out_path}")
    else:
        print(f"检测失败: {result.error_message}")
