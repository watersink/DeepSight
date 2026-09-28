"""
视频流丢失检测技能

规则与 shipingdiushi 的三层检测一致，接到实时视频帧上：
  - 无新帧：两次采样间隔过长
  - 黑屏 / 蓝屏 / 彩条 / 纯色 / 「信号丢失」灰底占位图
  - 花屏：大面积纯色高饱和色块
  - 冻结：新帧仍在到，但画面几乎不变
连续达到对应秒数后告警，恢复需连续正常一段时间。
煤安 analysisCase：视频丢失 0007；冻结单独记 0006。
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_CODE_ROOT = Path(__file__).resolve().parents[3]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cv2
import numpy as np

from app.skills.skill_base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

SKILL_NAME = "shipinliu_diushi_detector_skill"

_REASON_TEXT = {
    "no_frame": "无新帧",
    "black": "黑屏或无信号",
    "freeze": "画面冻结",
    "corrupt": "花屏或画面损坏",
}
_DETAIL_TEXT = {
    "dark": "黑屏",
    "blue_screen": "蓝屏",
    "color_bars": "彩条",
    "solid_color": "纯色无信号",
    "signal_lost": "无视频信号",
    "macroblock": "色块花屏",
}
_REASON_PRIORITY = ("no_frame", "corrupt", "black", "freeze")
_STATUS_COLOR = {
    "normal": (80, 190, 70),
    "suspect": (0, 180, 255),
    "lost": (50, 50, 230),
    "recovering": (220, 160, 40),
}
_STATUS_TEXT = {
    "normal": "正常",
    "suspect": "可疑",
    "lost": "视频丢失",
    "recovering": "恢复中",
}
_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\msyh.ttc"),
    Path(r"C:\Windows\Fonts\simhei.ttf"),
    Path(r"C:\Windows\Fonts\simsun.ttc"),
]


def _as_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2)
    if frame.ndim != 3 or frame.shape[2] not in (3, 4):
        raise ValueError("画面必须是灰度、BGR 或 BGRA")
    bgr = frame[:, :, :3]
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(bgr)


def _resize_nearest(image: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = image.shape[:2]
    ys = np.linspace(0, src_h - 1, height).astype(np.int32)
    xs = np.linspace(0, src_w - 1, width).astype(np.int32)
    return image[ys][:, xs]


def _bgr_to_gray(bgr: np.ndarray) -> np.ndarray:
    image = bgr.astype(np.float32)
    return 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]


def _is_color_bars(bgr: np.ndarray) -> bool:
    _height, width = bgr.shape[:2]
    bands = 8
    band_w = width // bands
    if band_w < 4:
        return False
    colors = []
    for index in range(bands):
        band = bgr[:, index * band_w : (index + 1) * band_w].astype(np.float32)
        if float(band.std(axis=(0, 1)).mean()) > 12.0:
            return False
        colors.append(band.mean(axis=(0, 1)))
    stacked = np.stack(colors)
    jumps = np.linalg.norm(stacked[1:] - stacked[:-1], axis=1)
    if int((jumps > 25.0).sum()) < 4:
        return False
    saturation = float((stacked.max(axis=1) - stacked.min(axis=1)).mean())
    return saturation > 20.0


def _uniform_detail(gray_small: np.ndarray, bgr_small: np.ndarray, params: Dict[str, float]) -> Optional[str]:
    mean = float(gray_small.mean())
    std = float(gray_small.std())
    if (mean <= params["very_dark_mean_max"] and std <= params["very_dark_std_max"]) or (
        mean <= params["black_mean_max"] and std <= params["black_std_max"]
    ):
        return "dark"
    if std > params["solid_std_max"]:
        return None
    blue, green, red = (float(bgr_small[:, :, channel].mean()) for channel in (0, 1, 2))
    if blue >= params["blue_min"] and blue >= red + params["blue_margin"] and blue >= green + params["blue_margin"]:
        return "blue_screen"
    return "solid_color"


def _is_signal_slate(bgr: np.ndarray, params: Dict[str, float]) -> bool:
    """灰底「信号丢失」占位图。中间字幕不会把整幅块亮度拉开。"""

    step = max(1, int(np.ceil(bgr.shape[1] / params["slate_max_width"])))
    image = bgr[::step, ::step]
    block = int(params["slate_block"])
    height, width = image.shape[:2]
    usable_h = height // block * block
    usable_w = width // block * block
    if usable_h < block * 2 or usable_w < block * 2:
        return False
    tiles = image[:usable_h, :usable_w].astype(np.float32)
    rows, cols = usable_h // block, usable_w // block
    tiles = tiles.reshape(rows, block, cols, block, 3).transpose(0, 2, 1, 3, 4)
    gray = 0.114 * tiles[..., 0] + 0.587 * tiles[..., 1] + 0.299 * tiles[..., 2]
    block_mean = gray.mean(axis=(2, 3))
    gray_image = 0.114 * image[..., 0].astype(np.float32) + 0.587 * image[..., 1] + 0.299 * image[..., 2]
    average = float(gray_image.mean())
    if not params["slate_mean_min"] <= average <= params["slate_mean_max"]:
        return False
    if float(gray_image.std()) > params["slate_global_std_max"]:
        return False
    if float(block_mean.std()) > params["slate_mean_std_max"]:
        return False
    channel_mean = image.reshape(-1, image.shape[2]).astype(np.float32).mean(axis=0)
    return float(channel_mean.max() - channel_mean.min()) <= params["slate_channel_gap_max"]


def _corrupt_ratio(bgr: np.ndarray, params: Dict[str, float]) -> float:
    step = max(1, int(np.ceil(bgr.shape[1] / params["corrupt_max_width"])))
    image = bgr[::step, ::step]
    block = int(params["corrupt_block"])
    height, width = image.shape[:2]
    usable_h = height // block * block
    usable_w = width // block * block
    if usable_h < block * 2 or usable_w < block * 2:
        return 0.0
    tiles = image[:usable_h, :usable_w].astype(np.float32)
    rows, cols = usable_h // block, usable_w // block
    tiles = tiles.reshape(rows, block, cols, block, 3).transpose(0, 2, 1, 3, 4)
    std = tiles.std(axis=(2, 3)).mean(axis=-1)
    means = tiles.mean(axis=(2, 3))
    channel_max = means.max(axis=-1)
    channel_min = means.min(axis=-1)
    solid = std <= params["corrupt_std_max"]
    saturated = (channel_max - channel_min) >= params["corrupt_saturation_min"]
    saturated &= channel_min <= params["corrupt_channel_min_max"]
    saturated &= channel_max >= params["corrupt_channel_max_min"]
    dist_h = np.linalg.norm(means[:, 1:, :] - means[:, :-1, :], axis=-1)
    dist_v = np.linalg.norm(means[1:, :, :] - means[:-1, :, :], axis=-1)
    jump = np.zeros((rows, cols), dtype=bool)
    jump[:, 1:] |= dist_h >= params["corrupt_jump_min"]
    jump[:, :-1] |= dist_h >= params["corrupt_jump_min"]
    jump[1:, :] |= dist_v >= params["corrupt_jump_min"]
    jump[:-1, :] |= dist_v >= params["corrupt_jump_min"]
    bad = solid & saturated & jump
    if int(bad.sum()) < int(params["corrupt_min_blocks"]):
        return 0.0
    if float(bad.any(axis=1).mean()) < params["corrupt_row_coverage_min"]:
        return 0.0
    if float(bad.any(axis=0).mean()) < params["corrupt_col_coverage_min"]:
        return 0.0
    if float(means[bad].std(axis=0).mean()) < params["corrupt_diversity_min"]:
        return 0.0
    return float(bad.mean())


def _frames_frozen(anchor: np.ndarray, current: np.ndarray, params: Dict[str, float]) -> bool:
    diff = np.abs(current.astype(np.float32) - anchor.astype(np.float32))
    mad = float(diff.mean())
    close_ratio = float((diff <= params["freeze_close_delta"]).mean())
    return mad <= params["freeze_mad_max"] and close_ratio >= params["freeze_close_ratio_min"]


def classify_loss_frame(frame: np.ndarray, params: Dict[str, float]) -> Tuple[str, str, np.ndarray]:
    """返回 (kind, detail, gray_small)。kind 为 ok / black / corrupt。"""

    bgr = _as_bgr(frame)
    if bgr.shape[0] < 2 or bgr.shape[1] < 2:
        raise ValueError("画面宽高至少为 2 像素")
    bgr_small = _resize_nearest(bgr, int(params["sample_width"]), int(params["sample_height"]))
    gray_small = _bgr_to_gray(bgr_small)
    if _is_color_bars(bgr):
        return "black", "color_bars", gray_small
    uniform = _uniform_detail(gray_small, bgr_small, params)
    if uniform is not None:
        return "black", uniform, gray_small
    if _is_signal_slate(bgr, params):
        return "black", "signal_lost", gray_small
    if _corrupt_ratio(bgr, params) >= params["corrupt_ratio_min"]:
        return "corrupt", "macroblock", gray_small
    return "ok", "", gray_small


def _reason_label(reason: Optional[str], detail: str) -> str:
    if not reason:
        return "画面有效"
    label = _REASON_TEXT.get(reason, reason)
    extra = _DETAIL_TEXT.get(detail, "")
    if extra and reason in {"black", "corrupt"}:
        label = f"{label}（{extra}）"
    return label


class ShipinliuDiushiDetectorSkill(BaseSkill):
    """实时视频流丢失检测。不依赖模型，也不需要校准模板。"""

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": SKILL_NAME,
        "name_zh": "视频流丢失检测",
        "version": "1.0",
        "cover_image": f"/skills/{SKILL_NAME}.png",
        "description": (
            "在实时视频中检测无新帧、黑屏、蓝屏、彩条、纯色、信号丢失占位图、花屏和画面冻结。"
            "达到对应持续秒数后告警，恢复需连续正常一段时间。"
            "视频丢失对应煤安 analysisCase=0007，冻结对应 0006。"
        ),
        "status": True,
        "run_mode": "stream",
        "required_models": [],
        "form_fields": [
            {
                "key": "check_interval_sec",
                "label": "采样间隔（秒）",
                "type": "number",
                "required": False,
                "default": 0.5,
            },
            {
                "key": "black_lost_sec",
                "label": "无信号确认（秒）",
                "type": "number",
                "required": False,
                "default": 2,
                "hint": "黑屏、蓝屏、彩条、纯色或信号丢失占位图持续这么久后告警",
            },
            {
                "key": "corrupt_lost_sec",
                "label": "花屏确认（秒）",
                "type": "number",
                "required": False,
                "default": 1.5,
            },
            {
                "key": "freeze_lost_sec",
                "label": "冻结确认（秒）",
                "type": "number",
                "required": False,
                "default": 8,
            },
            {
                "key": "frame_lost_sec",
                "label": "无新帧确认（秒）",
                "type": "number",
                "required": False,
                "default": 3,
            },
            {
                "key": "cooldown_sec",
                "label": "告警冷却（秒）",
                "type": "number",
                "required": False,
                "default": 10,
                "hint": "丢失持续期间，每隔多少秒再写一条报警",
            },
        ],
        "params": {
            "enable_default_sort_tracking": False,
            "enable_timing_log": False,
            "check_interval_sec": 0.5,
            "black_suspect_sec": 1.0,
            "black_lost_sec": 2.0,
            "corrupt_suspect_sec": 0.5,
            "corrupt_lost_sec": 1.5,
            "freeze_suspect_sec": 5.0,
            "freeze_lost_sec": 8.0,
            "frame_suspect_sec": 1.5,
            "frame_lost_sec": 3.0,
            "recover_sec": 2.0,
            "cooldown_sec": 10,
            "sample_width": 160,
            "sample_height": 90,
            "black_mean_max": 16.0,
            "black_std_max": 12.0,
            "very_dark_mean_max": 10.0,
            "very_dark_std_max": 22.0,
            "solid_std_max": 10.0,
            "blue_min": 80.0,
            "blue_margin": 40.0,
            "corrupt_block": 16,
            "corrupt_max_width": 640,
            "corrupt_std_max": 8.0,
            "corrupt_channel_min_max": 35.0,
            "corrupt_channel_max_min": 180.0,
            "corrupt_saturation_min": 120.0,
            "corrupt_jump_min": 50.0,
            "corrupt_ratio_min": 0.12,
            "corrupt_diversity_min": 12.0,
            "corrupt_min_blocks": 8,
            "corrupt_row_coverage_min": 0.5,
            "corrupt_col_coverage_min": 0.5,
            "slate_block": 32,
            "slate_max_width": 640,
            "slate_mean_min": 40.0,
            "slate_mean_max": 210.0,
            "slate_mean_std_max": 12.0,
            "slate_global_std_max": 24.0,
            "slate_channel_gap_max": 28.0,
            "freeze_mad_max": 1.0,
            "freeze_close_delta": 1.0,
            "freeze_close_ratio_min": 0.99,
        },
        "alert_definitions": [
            {
                "key": "video_lost",
                "level": 1,
                "description": "无新帧、黑屏、无信号占位图或花屏持续达到设定秒数。",
                "codes": {
                    "video_lost": {
                        "code": "video_lost",
                        "description": "视频丢失",
                        "analysis_case": "0007",
                    },
                },
            },
            {
                "key": "frame_freeze",
                "level": 2,
                "description": "画面冻结并持续达到设定秒数。",
                "codes": {
                    "freeze": {
                        "code": "freeze",
                        "description": "画面冻结",
                        "analysis_case": "0006",
                    },
                },
            },
        ],
    }

    def _initialize(self) -> None:
        params = self.config.get("params", {}) if isinstance(self.config, dict) else {}
        self.params = {key: params.get(key, default) for key, default in self.DEFAULT_CONFIG["params"].items()}
        self.check_interval_sec = max(0.2, float(self.params.get("check_interval_sec") or 0.5))
        self.cooldown_sec = max(0.0, float(self.params.get("cooldown_sec") or 0))
        self.recover_sec = max(0.0, float(self.params.get("recover_sec") or 0))
        self._suspect_after = {
            "no_frame": float(self.params["frame_suspect_sec"]),
            "black": float(self.params["black_suspect_sec"]),
            "corrupt": float(self.params["corrupt_suspect_sec"]),
            "freeze": float(self.params["freeze_suspect_sec"]),
        }
        self._lost_after = {
            "no_frame": float(self.params["frame_lost_sec"]),
            "black": float(self.params["black_lost_sec"]),
            "corrupt": float(self.params["corrupt_lost_sec"]),
            "freeze": float(self.params["freeze_lost_sec"]),
        }
        self.alert_definitions: List[Dict[str, Any]] = list(self.config.get("alert_definitions") or [])
        self._reset_runtime()
        self.log(
            "info",
            f"初始化视频流丢失检测: interval={self.check_interval_sec}s "
            f"black={self._lost_after['black']}s corrupt={self._lost_after['corrupt']}s "
            f"freeze={self._lost_after['freeze']}s frame={self._lost_after['no_frame']}s",
        )

    def get_required_models(self) -> List[str]:
        return []

    def _reset_runtime(self) -> None:
        self._last_sample_at: Optional[float] = None
        self._last_process_ts: Optional[float] = None
        self._last_alert_ts = 0.0
        self._last_result: Dict[str, Any] = {}
        self._state = "normal"
        self._reason: Optional[str] = None
        self._detail = ""
        self._black_since: Optional[float] = None
        self._corrupt_since: Optional[float] = None
        self._freeze_since: Optional[float] = None
        self._recover_since: Optional[float] = None
        self._anchor: Optional[np.ndarray] = None

    def detect_frame(
        self,
        current_bgr: np.ndarray,
        *,
        dt: Optional[float] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """对单张 BGR 帧做视频流丢失检测。now 仅用于离线回放。"""

        current = time.time() if now is None else float(now)
        if dt is not None and dt > 0 and self._last_sample_at is not None:
            gap = float(dt)
        elif self._last_sample_at is None:
            gap = 0.0
        else:
            gap = max(0.0, current - self._last_sample_at)

        kind, detail, gray_small = classify_loss_frame(current_bgr, self.params)
        self._update_content_timers(kind, gray_small, current)
        level, reason = self._ranked(current, gap)
        previous = self._state
        self._apply_state(level, reason, detail if reason in {"black", "corrupt"} else "", current)
        triggered = self._should_emit(previous, current)
        if triggered:
            self._last_alert_ts = current
        self._last_sample_at = current

        message = self._message()
        if self._state != previous or triggered:
            self.log("info", f"视频流丢失: {self._state} emit={int(triggered)} {message}")
        result = self._build_result(message, triggered, gap)
        self._last_result = result
        return result

    def _update_content_timers(self, kind: str, gray_small: np.ndarray, now: float) -> None:
        if kind == "black":
            if self._black_since is None:
                self._black_since = now
            self._corrupt_since = None
            self._freeze_since = None
            self._anchor = gray_small
            self._detail = ""
            return
        if kind == "corrupt":
            if self._corrupt_since is None:
                self._corrupt_since = now
            self._black_since = None
            self._freeze_since = None
            self._anchor = gray_small
            return
        self._black_since = None
        self._corrupt_since = None
        if self._anchor is None or not _frames_frozen(self._anchor, gray_small, self.params):
            self._anchor = gray_small
            self._freeze_since = None
            return
        if self._freeze_since is None:
            self._freeze_since = now

    def _ranked(self, now: float, gap: float) -> Tuple[str, Optional[str]]:
        elapsed: Dict[str, float] = {}
        if gap >= self._suspect_after["no_frame"]:
            elapsed["no_frame"] = gap
        if self._corrupt_since is not None:
            elapsed["corrupt"] = now - self._corrupt_since
        if self._black_since is not None:
            elapsed["black"] = now - self._black_since
        if self._freeze_since is not None:
            elapsed["freeze"] = now - self._freeze_since
        lost = [reason for reason, seconds in elapsed.items() if seconds + 1e-3 >= self._lost_after[reason]]
        suspect = [
            reason
            for reason, seconds in elapsed.items()
            if reason not in lost and seconds + 1e-3 >= self._suspect_after[reason]
        ]
        if lost:
            return "lost", self._pick(lost)
        if suspect:
            return "suspect", self._pick(suspect)
        return "none", None

    @staticmethod
    def _pick(reasons: List[str]) -> Optional[str]:
        for reason in _REASON_PRIORITY:
            if reason in reasons:
                return reason
        return None

    def _apply_state(self, level: str, reason: Optional[str], detail: str, now: float) -> None:
        if reason in {"black", "corrupt"} and detail:
            self._detail = detail
        elif level == "none":
            self._detail = ""
        if self._state in {"normal", "suspect"}:
            if level == "none":
                self._state = "normal"
                self._reason = None
                self._detail = ""
            elif level == "suspect":
                self._state = "suspect"
                self._reason = reason
            else:
                self._state = "lost"
                self._reason = reason
                self._recover_since = None
            return
        if self._state == "lost":
            if level == "lost":
                self._reason = reason
            else:
                self._state = "recovering"
                self._recover_since = now
            return
        if level == "lost":
            self._state = "lost"
            self._reason = reason
            self._recover_since = None
        elif level == "none" and self._recover_since is not None and now - self._recover_since + 1e-3 >= self.recover_sec:
            self._state = "normal"
            self._reason = None
            self._detail = ""
            self._recover_since = None
        elif level != "none":
            self._recover_since = now

    def _should_emit(self, previous: str, now: float) -> bool:
        if self._state != "lost":
            return False
        if previous != "lost":
            return True
        return now - self._last_alert_ts >= self.cooldown_sec

    def _message(self) -> str:
        label = _reason_label(self._reason, self._detail)
        if self._state == "normal":
            return "画面有效"
        if self._state == "suspect":
            return f"可疑：{label}"
        if self._state == "recovering":
            return f"故障条件已解除，等待恢复确认（此前为{label}）"
        return f"视频丢失：{label}"

    def _build_result(self, message: str, triggered: bool, gap: float) -> Dict[str, Any]:
        reason = self._reason
        is_freeze = reason == "freeze" and self._state == "lost"
        is_lost = self._state == "lost" and reason in {"no_frame", "black", "corrupt"}
        recognition = []
        active_alerts: List[Dict[str, Any]] = []
        if triggered and is_freeze:
            recognition = ["freeze"]
            active_alerts.append(self._alert("frame_freeze", "0006", "freeze", "画面冻结", message))
        elif triggered and is_lost:
            recognition = ["video_lost"]
            active_alerts.append(self._alert("video_lost", "0007", "video_lost", "视频丢失", message))
        return {
            "detections": [],
            "count": 1 if self._state == "lost" else 0,
            "skill_name": self.config.get("name") or SKILL_NAME,
            "run_mode": "stream",
            "status": self._state,
            "reason": reason or "",
            "detail": self._detail,
            "message": message,
            "frame_gap_sec": round(gap, 3),
            "has_video_lost_alarm": bool(triggered and is_lost),
            "has_freeze_alarm": bool(triggered and is_freeze),
            "recognition_types": recognition,
            "alert_definitions": list(self.alert_definitions),
            "active_alerts": active_alerts,
            "safety_metrics": {
                "is_safe": self._state != "lost",
                "alert_info": {
                    "alert_triggered": bool(triggered and self._state == "lost"),
                    "alert_name": "视频丢失预警" if is_lost else ("画面冻结预警" if is_freeze else ""),
                    "alert_type": "视频流丢失检测",
                    "alert_description": message if triggered else "",
                },
            },
        }

    def _alert(self, key: str, analysis_case: str, recognition_type: str, description: str, message: str) -> Dict[str, Any]:
        definition = next((item for item in self.alert_definitions if item.get("key") == key), {})
        return {
            "key": key,
            "level": int(definition.get("level", 1) or 1),
            "description": message or description,
            "recognition_type": recognition_type,
            "analysis_case": analysis_case,
            "violation_type": description,
        }

    def process(self, input_data: Any, context: Any = None, **kwargs) -> SkillResult:
        try:
            image, error_message = self._load_image(input_data)
            if error_message:
                return SkillResult.error_result(error_message)
            now = time.time()
            if (
                self._last_result
                and self._last_process_ts is not None
                and now - self._last_process_ts < self.check_interval_sec
            ):
                return SkillResult.success_result(self._result_without_emit(self._last_result))
            data = self.detect_frame(image, now=now)
            self._last_process_ts = now
            return SkillResult.success_result(data)
        except Exception as e:
            logger.exception("视频流丢失检测失败: %s", e)
            return SkillResult.error_result(f"处理失败: {e}")

    @staticmethod
    def _load_image(input_data: Any) -> Tuple[Optional[np.ndarray], Optional[str]]:
        image = None
        if isinstance(input_data, np.ndarray):
            image = input_data
        elif isinstance(input_data, str):
            image = cv2.imread(input_data)
            if image is None:
                return None, f"无法加载图像: {input_data}"
        elif isinstance(input_data, dict):
            image_data = input_data.get("image")
            if isinstance(image_data, np.ndarray):
                image = image_data
            elif isinstance(image_data, str):
                image = cv2.imread(image_data)
                if image is None:
                    return None, f"无法加载图像: {image_data}"
            else:
                return None, "输入字典中缺少图像"
        else:
            return None, "不支持的输入数据类型"
        if image is None or getattr(image, "size", 0) == 0:
            return None, "无效的图像数据"
        return image, None

    @staticmethod
    def _result_without_emit(result: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(result)
        out["has_video_lost_alarm"] = False
        out["has_freeze_alarm"] = False
        out["recognition_types"] = []
        out["active_alerts"] = []
        safety = dict(out.get("safety_metrics") or {})
        alert_info = dict(safety.get("alert_info") or {})
        alert_info["alert_triggered"] = False
        alert_info["alert_name"] = ""
        alert_info["alert_description"] = ""
        safety["alert_info"] = alert_info
        out["safety_metrics"] = safety
        return out

    def _draw_detections_on_frame(self, frame: np.ndarray, alert_data: Dict[str, Any]) -> np.ndarray:
        try:
            data = alert_data if isinstance(alert_data, dict) else {}
            if not data and isinstance(self._last_result, dict):
                data = self._last_result
            status = str(data.get("status") or "normal")
            color = _STATUS_COLOR.get(status, (200, 200, 200))
            title = _STATUS_TEXT.get(status, status)
            detail = _reason_label(data.get("reason") or None, str(data.get("detail") or ""))
            if status == "normal":
                detail = "画面有效"
            return self._put_status_text(frame, f"识别结果  {title}", (16, 12), color, detail)
        except Exception as e:
            logger.error("绘制视频流丢失结果失败: %s", e)
            return frame

    @staticmethod
    def _put_status_text(
        frame_bgr: np.ndarray,
        title: str,
        xy: Tuple[int, int],
        color_bgr: Tuple[int, int, int],
        detail: str,
        size: int = 32,
    ) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            cv2.putText(frame_bgr, title, xy, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_bgr, 2)
            return frame_bgr
        font = None
        small = None
        for path in _FONT_CANDIDATES:
            if path.exists():
                font = ImageFont.truetype(str(path), size=size)
                small = ImageFont.truetype(str(path), size=max(18, size - 8))
                break
        if font is None:
            font = ImageFont.load_default()
            small = font
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        x, y = xy
        color_rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            draw.text((x + dx, y + dy), title, font=font, fill=(0, 0, 0))
            draw.text((x + dx, y + size + dy), detail, font=small, fill=(0, 0, 0))
        draw.text((x, y), title, font=font, fill=color_rgb)
        draw.text((x, y + size), detail, font=small, fill=(255, 255, 255))
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    import json
    from copy import deepcopy

    sample = Path(r"d:\yingji\shipingzijian\video\shipindiushi.mp4")
    if not sample.is_file():
        raise SystemExit(f"未找到样例视频: {sample}")
    config = deepcopy(ShipinliuDiushiDetectorSkill.DEFAULT_CONFIG)
    skill = ShipinliuDiushiDetectorSkill(config)
    capture = cv2.VideoCapture(str(sample))
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    index = 0
    last_status = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % max(1, int(fps * skill.check_interval_sec)) != 0:
            index += 1
            continue
        result = skill.detect_frame(frame, now=index / fps)
        if result["status"] != last_status or result["has_video_lost_alarm"] or result["has_freeze_alarm"]:
            print(
                f"t={index / fps:6.1f}s {result['status']:11} "
                f"{result['reason'] or '-':8} {result['message']}"
            )
            last_status = result["status"]
        index += 1
    capture.release()
    print(json.dumps({"final": skill._last_result.get("status"), "message": skill._last_result.get("message")}, ensure_ascii=False))
