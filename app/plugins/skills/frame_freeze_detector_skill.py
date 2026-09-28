"""
画面冻结检测技能

运行模式 stream：接入实时视频流水线，按检测间隔对当前帧调用 detect_frame。
不依赖校准模板：相邻采样帧做相似度比较（像素差 / 直方图 / SSIM），
连续同帧达确认次数后告警。过暗/近黑屏抑制，避免与视频丢失混淆。

结构对齐 camera_shift_detector / camera_tilt_detector：
DEFAULT_CONFIG → _initialize → detect_frame → process → overlay → test_image/test_video
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 支持直接运行本文件: python app/plugins/skills/frame_freeze_detector_skill.py
_CODE_ROOT = Path(__file__).resolve().parents[3]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cv2
import numpy as np

from app.services.camera_pose_core import compute_ssim
from app.skills.skill_base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

RECOGNITION_CODE = "freeze"
RECOGNITION_DESC = "画面冻结"
ANALYSIS_CASE = "0006"  # 煤安 videoAnomaly：画面冻结


def _resize_max_side(gray: np.ndarray, max_side: int) -> np.ndarray:
    h, w = gray.shape[:2]
    side = max(h, w)
    if side <= max_side or max_side <= 0:
        return gray
    scale = float(max_side) / float(side)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)


def _crop_roi(
    gray: np.ndarray, roi: Tuple[float, float, float, float]
) -> np.ndarray:
    """roi = (x, y, w, h) 相对比例，中心 ROI 可避开 OSD 时间戳。"""
    h, w = gray.shape[:2]
    rx, ry, rw, rh = roi
    x0 = max(0, min(w - 1, int(round(rx * w))))
    y0 = max(0, min(h - 1, int(round(ry * h))))
    x1 = max(x0 + 1, min(w, int(round((rx + rw) * w))))
    y1 = max(y0 + 1, min(h, int(round((ry + rh) * h))))
    return gray[y0:y1, x0:x1]


def _preprocess_frame(
    bgr: np.ndarray,
    *,
    max_side: int,
    roi: Tuple[float, float, float, float],
) -> np.ndarray:
    if bgr.ndim == 3:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = bgr
    gray = _resize_max_side(gray, max_side)
    gray = _crop_roi(gray, roi)
    # 轻量平滑，抑制压缩噪声导致的假差分
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _hist_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ha = cv2.calcHist([a], [0], None, [64], [0, 256])
    hb = cv2.calcHist([b], [0], None, [64], [0, 256])
    cv2.normalize(ha, ha)
    cv2.normalize(hb, hb)
    corr = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
    return float(corr)


def _mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


class FrameFreezeDetectorSkill(BaseSkill):
    """相邻采样帧相似度：仅判定画面冻结。"""

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": "frame_freeze_detector",
        "name_zh": "画面冻结检测",
        "version": "1.0",
        "cover_image": "/skills/frame_freeze_detector.png",
        "description": (
            "按检测间隔比较相邻采样帧的像素差、直方图相关与 SSIM；"
            "连续同帧达到确认次数后判定画面冻结（煤安 analysisCase=0006）。"
            "过暗/近黑屏时抑制，避免与视频丢失混淆。无需校准模板。"
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
                "default": 1,
                "hint": "相邻采样帧间隔；过小易因相邻帧天然相似而误报",
            },
            {
                "key": "mean_abs_diff_threshold",
                "label": "像素差阈值",
                "type": "number",
                "required": False,
                "default": 3.0,
                "hint": "mean(|A-B|) 低于该值视为同帧（0–255）",
            },
            {
                "key": "hist_corr_threshold",
                "label": "直方图相关阈值",
                "type": "number",
                "required": False,
                "default": 0.99,
                "hint": "高于该值视为同帧",
            },
            {
                "key": "ssim_same_threshold",
                "label": "SSIM 同帧阈值",
                "type": "number",
                "required": False,
                "default": 0.985,
                "hint": "高于该值视为同帧",
            },
            {
                "key": "confirm_count",
                "label": "连续确认次数",
                "type": "number",
                "required": False,
                "default": 5,
                "hint": "连续同帧次数；约等于 确认次数×采样间隔 秒后告警",
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
            "check_interval_sec": 1.0,
            "mean_abs_diff_threshold": 3.0,
            "hist_corr_threshold": 0.99,
            "ssim_same_threshold": 0.985,
            "confirm_count": 5,
            "cooldown_sec": 60,
            "max_side": 480,
            # 中心 ROI，降低时间戳 OSD 对冻结判定的干扰
            "roi": [0.2, 0.2, 0.6, 0.6],
            "dark_mean_suppress": 12.0,
            "mismatch_tolerance": 0,
            "enable_timing_log": False,
        },
        "alert_definitions": [
            {
                "key": "frame_freeze",
                "level": 2,
                "description": "视频画面在设定时长内几乎无变化，疑似画面冻结。",
                "codes": {
                    "default": {
                        "code": RECOGNITION_CODE,
                        "description": RECOGNITION_DESC,
                        "analysis_case": ANALYSIS_CASE,
                    }
                },
            }
        ],
    }

    def _initialize(self) -> None:
        params = self.config.get("params", {}) if isinstance(self.config, dict) else {}
        self.check_interval_sec = max(
            0.2, float(params.get("check_interval_sec", 1.0) or 1.0)
        )
        self.mean_abs_diff_threshold = max(
            0.0, float(params.get("mean_abs_diff_threshold", 3.0) or 3.0)
        )
        self.hist_corr_threshold = float(
            params.get("hist_corr_threshold", 0.99) or 0.99
        )
        self.ssim_same_threshold = float(
            params.get("ssim_same_threshold", 0.985) or 0.985
        )
        self.confirm_count = max(1, int(params.get("confirm_count", 5) or 5))
        self.cooldown_sec = max(0.0, float(params.get("cooldown_sec", 60) or 60))
        self.max_side = max(64, int(params.get("max_side", 480) or 480))
        self.dark_mean_suppress = max(
            0.0, float(params.get("dark_mean_suppress", 12.0) or 12.0)
        )
        self.mismatch_tolerance = max(
            0, int(params.get("mismatch_tolerance", 0) or 0)
        )

        roi_raw = params.get("roi") or [0.2, 0.2, 0.6, 0.6]
        try:
            rx, ry, rw, rh = [float(x) for x in roi_raw[:4]]
            self.roi: Tuple[float, float, float, float] = (rx, ry, rw, rh)
        except Exception:
            self.roi = (0.2, 0.2, 0.6, 0.6)

        self.alert_definitions: List[Dict[str, Any]] = list(
            self.config.get("alert_definitions") or []
        )
        self._prev_gray: Optional[np.ndarray] = None
        self._freeze_streak = 0
        self._mismatch_streak = 0
        self._last_alert_ts = 0.0
        self._last_result: Dict[str, Any] = {}
        self._last_infer_ts = 0.0

        self.log(
            "info",
            f"初始化画面冻结检测: interval={self.check_interval_sec}s "
            f"diff<{self.mean_abs_diff_threshold} hist>{self.hist_corr_threshold} "
            f"ssim>{self.ssim_same_threshold} confirm={self.confirm_count} "
            f"cooldown={self.cooldown_sec}s",
        )

    def get_required_models(self) -> List[str]:
        return []

    def _is_same_frame(
        self, prev: np.ndarray, cur: np.ndarray
    ) -> Tuple[bool, Dict[str, float]]:
        if prev.shape != cur.shape:
            cur = cv2.resize(
                cur, (prev.shape[1], prev.shape[0]), interpolation=cv2.INTER_AREA
            )
        mad = _mean_abs_diff(prev, cur)
        hist = _hist_correlation(prev, cur)
        ssim = compute_ssim(prev, cur)
        metrics = {
            "mean_abs_diff": round(mad, 4),
            "hist_corr": round(hist, 5),
            "ssim": round(ssim, 5),
        }
        same = (
            mad <= self.mean_abs_diff_threshold
            and hist >= self.hist_corr_threshold
            and ssim >= self.ssim_same_threshold
        )
        return same, metrics

    def detect_frame(self, current_bgr: np.ndarray) -> Dict[str, Any]:
        """对单张 BGR 图做画面冻结检测。

        与上一采样帧比相似度 → streak / 冷却 / 告警。
        首帧仅建立基准，不告警。
        """
        cur_gray = _preprocess_frame(
            current_bgr, max_side=self.max_side, roi=self.roi
        )
        now = time.time()
        active_alerts: List[Dict[str, Any]] = []
        triggered = False
        metrics: Dict[str, float] = {
            "mean_abs_diff": 0.0,
            "hist_corr": 1.0,
            "ssim": 1.0,
        }
        cur_mean = float(np.mean(cur_gray))

        # 近黑屏：疑似视频丢失，不报冻结
        if cur_mean <= self.dark_mean_suppress:
            self._freeze_streak = 0
            self._mismatch_streak = 0
            self._prev_gray = cur_gray
            status = "suppress_dark"
            message = f"画面过暗 mean={cur_mean:.1f}，抑制冻结判定"
            result = self._build_result(
                status=status,
                message=message,
                metrics=metrics,
                triggered=False,
                active_alerts=[],
                frame_mean=cur_mean,
            )
            self._last_result = result
            return result

        if self._prev_gray is None:
            self._prev_gray = cur_gray
            status = "warmup"
            message = "已建立首帧基准，等待下一采样"
            result = self._build_result(
                status=status,
                message=message,
                metrics=metrics,
                triggered=False,
                active_alerts=[],
                frame_mean=cur_mean,
            )
            self._last_result = result
            return result

        same, metrics = self._is_same_frame(self._prev_gray, cur_gray)
        # 始终推进基准，使「冻结」对应连续采样内容不变
        self._prev_gray = cur_gray

        if same:
            self._mismatch_streak = 0
            self._freeze_streak += 1
            status = "freeze_pending"
            message = (
                f"同帧 streak={self._freeze_streak}/{self.confirm_count} "
                f"diff={metrics['mean_abs_diff']} "
                f"hist={metrics['hist_corr']} ssim={metrics['ssim']}"
            )
            if self._freeze_streak >= self.confirm_count:
                if now - self._last_alert_ts >= self.cooldown_sec:
                    triggered = True
                    self._last_alert_ts = now
                    self._freeze_streak = 0
                    status = "freeze"
                    message = (
                        f"画面冻结 diff={metrics['mean_abs_diff']} "
                        f"ssim={metrics['ssim']}"
                    )
                    active_alerts.append(
                        {
                            "key": "frame_freeze",
                            "level": 2,
                            "description": RECOGNITION_DESC,
                            "recognition_type": RECOGNITION_CODE,
                            "analysis_case": ANALYSIS_CASE,
                            **metrics,
                        }
                    )
                else:
                    status = "freeze_cooldown"
                    message = (
                        f"画面冻结(冷却中) remain="
                        f"{max(0.0, self.cooldown_sec - (now - self._last_alert_ts)):.1f}s"
                    )
        else:
            self._mismatch_streak += 1
            if self._mismatch_streak > self.mismatch_tolerance:
                self._freeze_streak = 0
                status = "normal"
                message = (
                    f"画面有变化 diff={metrics['mean_abs_diff']} "
                    f"hist={metrics['hist_corr']} ssim={metrics['ssim']}"
                )
            else:
                # 容许少量抖动帧，不立刻清零
                status = "freeze_pending" if self._freeze_streak > 0 else "normal"
                message = (
                    f"短暂差异(容错) mismatch={self._mismatch_streak}/"
                    f"{self.mismatch_tolerance} streak={self._freeze_streak}"
                )

        result = self._build_result(
            status=status,
            message=message,
            metrics=metrics,
            triggered=triggered,
            active_alerts=active_alerts,
            frame_mean=cur_mean,
        )
        self._last_result = result
        return result

    def _build_result(
        self,
        *,
        status: str,
        message: str,
        metrics: Dict[str, float],
        triggered: bool,
        active_alerts: List[Dict[str, Any]],
        frame_mean: float,
    ) -> Dict[str, Any]:
        return {
            "skill_name": self.config.get("name") or "frame_freeze_detector",
            "run_mode": "stream",
            "status": status,
            "message": message,
            "mean_abs_diff": metrics.get("mean_abs_diff"),
            "hist_corr": metrics.get("hist_corr"),
            "ssim": metrics.get("ssim"),
            "frame_mean": round(float(frame_mean), 2),
            "freeze_streak": self._freeze_streak,
            "confirm_count": self.confirm_count,
            "mean_abs_diff_threshold": self.mean_abs_diff_threshold,
            "hist_corr_threshold": self.hist_corr_threshold,
            "ssim_same_threshold": self.ssim_same_threshold,
            "has_freeze_alarm": triggered,
            "recognition_types": [RECOGNITION_CODE] if triggered else [],
            "alert_definitions": list(self.alert_definitions),
            "active_alerts": active_alerts,
            "count": 0,
            "enter_count": 0,
        }

    def process(self, input_data: Any, context: Any = None, **kwargs) -> SkillResult:
        """实时视频帧入口：按 check_interval_sec 节流后调用 detect_frame。"""
        try:
            if isinstance(input_data, dict):
                image = input_data.get("image")
            else:
                image = input_data
            if image is None:
                return SkillResult.error_result("缺少图像")
            now = time.time()
            if (
                self._last_result
                and now - self._last_infer_ts < self.check_interval_sec
            ):
                return SkillResult.success_result(self._last_result)
            data = self.detect_frame(image)
            self._last_infer_ts = now
            return SkillResult.success_result(data)
        except Exception as e:
            logger.exception("画面冻结检测失败")
            return SkillResult.error_result(str(e))

    def _draw_detections_on_frame(
        self, frame: np.ndarray, alert_data: Dict[str, Any]
    ) -> np.ndarray:
        """供实时视频流水线调用：把最近一次冻结检测结果画到画面上。"""
        try:
            data = alert_data if isinstance(alert_data, dict) else {}
            if not data and isinstance(self._last_result, dict):
                data = self._last_result
            return _draw_freeze_overlay(frame, data)
        except Exception as e:
            logger.error("绘制画面冻结检测结果失败: %s", e)
            return frame


def _ascii_only(text: Any, *, fallback: str = "") -> str:
    """OpenCV putText 无法可靠显示中文，仅保留可打印 ASCII。"""
    raw = str(text if text is not None else "")
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in raw)
    cleaned = " ".join(cleaned.split())
    return cleaned or fallback


def _draw_freeze_overlay(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """把画面冻结检测结果画到画面上（仅英文，避免 OpenCV 中文乱码）。"""
    status = _ascii_only(result.get("status"), fallback="unknown")
    if status in {"freeze", "fail"} or result.get("has_freeze_alarm"):
        color = (0, 0, 255)
    elif status in {"freeze_pending", "freeze_cooldown", "suppress_dark"}:
        color = (0, 165, 255)
    elif status in {"normal", "warmup"}:
        color = (0, 200, 0)
    else:
        color = (200, 200, 200)

    msg = _ascii_only(result.get("message"))
    lines = [
        f"status: {status}",
        f"diff: {result.get('mean_abs_diff')} / thr {result.get('mean_abs_diff_threshold')}",
        f"hist: {result.get('hist_corr')} / thr {result.get('hist_corr_threshold')}",
        f"ssim: {result.get('ssim')} / thr {result.get('ssim_same_threshold')}",
        f"streak: {result.get('freeze_streak')}/{result.get('confirm_count')}  "
        f"alert: {result.get('has_freeze_alarm')}",
        f"frame_mean: {result.get('frame_mean')}",
    ]
    if msg:
        lines.append(f"msg: {msg}")

    overlay = frame.copy()
    box_h = 28 + 22 * len(lines)
    cv2.rectangle(overlay, (8, 8), (min(frame.shape[1] - 8, 760), box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line[:95],
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 22
    return frame


def test_image() -> None:
    """本地双图测试：两张图相似度 → 是否同帧（confirm=1 时直接出结论）。

    用法:
      python app/plugins/skills/frame_freeze_detector_skill.py --a a.png --b b.png
      python app/plugins/skills/frame_freeze_detector_skill.py  # 默认用同一图测「冻结」
    """
    import argparse
    import json
    from copy import deepcopy

    default_img = str(_CODE_ROOT / "test_images_videos" / "mobangtu.png")

    parser = argparse.ArgumentParser(description="画面冻结检测（双图离线测试）")
    parser.add_argument(
        "--a",
        "--prev",
        dest="prev",
        default=default_img,
        help=f"上一帧图路径（默认: {default_img}）",
    )
    parser.add_argument(
        "--b",
        "--cur",
        "--current",
        dest="current",
        default=default_img,
        help=f"当前帧图路径（默认与 --a 相同，用于验证冻结）",
    )
    parser.add_argument(
        "--confirm-count",
        type=int,
        default=1,
        help="连续确认次数（离线默认 1）",
    )
    args = parser.parse_args()

    config = deepcopy(FrameFreezeDetectorSkill.DEFAULT_CONFIG)
    params = config.setdefault("params", {})
    params["confirm_count"] = max(1, int(args.confirm_count))
    params["cooldown_sec"] = 0

    skill = FrameFreezeDetectorSkill(config)
    prev_bgr = cv2.imread(str(args.prev), cv2.IMREAD_COLOR)
    cur_bgr = cv2.imread(str(args.current), cv2.IMREAD_COLOR)
    if prev_bgr is None:
        raise FileNotFoundError(f"无法读取上一帧: {args.prev}")
    if cur_bgr is None:
        raise FileNotFoundError(f"无法读取当前帧: {args.current}")

    # 先喂上一帧建立基准，再喂当前帧做判定
    skill.detect_frame(prev_bgr)
    result = skill.detect_frame(cur_bgr)

    print("=== 画面冻结检测结果 ===")
    print(f"上一帧: {args.prev}")
    print(f"当前帧: {args.current}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        f"结论: status={result.get('status')} "
        f"has_freeze_alarm={result.get('has_freeze_alarm')} "
        f"message={result.get('message')}"
    )
    sys.exit(0)


def test_video() -> None:
    """本机摄像头或本地视频文件测试画面冻结算法，结果叠加显示在画面上。

    用法:
      python app/plugins/skills/frame_freeze_detector_skill.py video
      python app/plugins/skills/frame_freeze_detector_skill.py video --camera 0
      python app/plugins/skills/frame_freeze_detector_skill.py video --source ./test_images_videos/freeze.mp4
      python app/plugins/skills/frame_freeze_detector_skill.py video --source ./test_images_videos/freeze.mp4 --loop

    按键: q 退出；f 强制把上一帧设为当前帧（模拟冻结）。
    """
    import argparse
    from copy import deepcopy

    default_source = str(_CODE_ROOT / "test_images_videos" / "person_gouzi.mp4")

    parser = argparse.ArgumentParser(
        description="画面冻结检测（本机摄像头 / 本地视频测试）"
    )
    parser.add_argument(
        "--source",
        "--video",
        dest="source",
        default="",
        help=(
            "本地视频文件路径；指定后优先读文件，不再打开摄像头。"
            f" 例: {default_source}"
        ),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="本机摄像头索引（默认 0；仅在未指定 --source 时生效）",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="本地视频播完后循环（默认播完退出）",
    )
    parser.add_argument(
        "--confirm-count",
        type=int,
        default=3,
        help="连续确认次数（实时预览默认 3）",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=500,
        help="检测间隔毫秒（默认 500）",
    )
    args = parser.parse_args()

    config = deepcopy(FrameFreezeDetectorSkill.DEFAULT_CONFIG)
    params = config.setdefault("params", {})
    params["confirm_count"] = max(1, int(args.confirm_count))
    params["cooldown_sec"] = 0
    params["check_interval_sec"] = max(0.2, float(args.interval_ms) / 1000.0)

    skill = FrameFreezeDetectorSkill(config)

    source = str(args.source or "").strip()
    if source:
        video_path = Path(source)
        if not video_path.is_absolute():
            video_path = (_CODE_ROOT / video_path).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"本地视频不存在: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开本地视频: {video_path}")
        source_desc = f"file={video_path}"
    else:
        api = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(args.camera), api)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(args.camera))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开本机摄像头 camera={args.camera}")
        source_desc = f"camera={args.camera}"

    window = "frame_freeze_detector"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    last_result: Dict[str, Any] = {"status": "init", "message": "waiting"}
    last_infer_ts = 0.0
    interval_sec = max(0.0, float(args.interval_ms) / 1000.0)
    hold_frame: Optional[np.ndarray] = None
    hold_until = 0.0

    print(
        f"实时测试已启动: {source_desc} interval={args.interval_ms}ms  "
        f"(q=退出, f=模拟冻结约 8 秒"
        f"{', loop' if source and args.loop else ''})"
    )
    try:
        while True:
            now = time.time()
            if hold_frame is not None and now < hold_until:
                frame = hold_frame
            else:
                hold_frame = None
                ok, frame = cap.read()
                if not ok or frame is None:
                    if source and args.loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            print("本地视频循环读取失败，退出")
                            break
                        print("本地视频已循环到开头")
                    else:
                        print("视频结束或读取失败，退出")
                        break

            if now - last_infer_ts >= interval_sec:
                try:
                    last_result = skill.detect_frame(frame)
                except Exception as e:
                    last_result = {
                        # status:
                        #   warmup          — 首帧建基准
                        #   normal          — 画面有变化
                        #   freeze_pending  — 同帧累计中
                        #   freeze          — 已确认冻结并告警
                        #   freeze_cooldown — 冻结但冷却中
                        #   suppress_dark   — 过暗抑制
                        "status": "fail",
                        "message": str(e),
                        "mean_abs_diff": None,
                        "hist_corr": None,
                        "ssim": None,
                        "freeze_streak": 0,
                        "has_freeze_alarm": False,
                    }
                last_infer_ts = now

            shown = _draw_freeze_overlay(frame, last_result)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("f"):
                hold_frame = frame.copy()
                hold_until = time.time() + 8.0
                print("已开始模拟冻结（重复当前帧约 8 秒）")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in {"video", "test_video", "--video"}:
        sys.argv.pop(1)
        test_video()
    else:
        test_image()
