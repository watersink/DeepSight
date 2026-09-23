"""
摄像头角度偏离检测技能

运行模式 stream：接入实时视频流水线，按检测间隔对当前帧调用 detect_frame。
与 camera_shift_detector 共用 camera_pose_core（模板 + 特征 + H）；
本技能只判转角 / 透视超阈，不报位置平移。
校准模板由外部提供（URL/路径）。
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 支持直接运行本文件: python app/plugins/skills/camera_tilt_detector_skill.py
_CODE_ROOT = Path(__file__).resolve().parents[3]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cv2
import numpy as np

from app.services import camera_pose_core as pose_core
from app.skills.skill_base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

RECOGNITION_CODE = "09"
RECOGNITION_DESC = "摄像头角度偏离"


class CameraTiltDetectorSkill(BaseSkill):
    """背景模板 + 特征单应：仅判定角度/透视偏离。"""

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": "camera_tilt_detector",
        "name_zh": "摄像头角度偏离检测",
        "version": "1.0",
        "cover_image": "/skills/camera_tilt_detector.png",
        "description": (
            "基于外部校准模板与实时视频帧，通过特征匹配与单应矩阵估计画面转角与透视变化，"
            "检测摄像头角度偏离。与位置挪移技能共用几何核心，可独立部署。"
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
                "hint": "外部提供的模板图 URL 或服务器本地路径（可与挪移技能共用）",
            },
            {
                "key": "tilt_threshold_deg",
                "label": "转角告警阈值（度）",
                "type": "number",
                "required": False,
                "default": 8,
                "hint": "四边方向角平均变化超过该值视为角度偏离",
            },
            {
                "key": "perspective_threshold",
                "label": "透视告警阈值",
                "type": "number",
                "required": False,
                "default": 0.12,
                "hint": "对边长度比变化之和超过该值视为俯仰/透视偏离",
            },
            {
                "key": "confirm_count",
                "label": "连续确认次数",
                "type": "number",
                "required": False,
                "default": 2,
            },
            {
                "key": "cooldown_sec",
                "label": "告警冷却（秒）",
                "type": "number",
                "required": False,
                "default": 5,
            },
            {
                "key": "ssim_skip_threshold",
                "label": "SSIM 跳过阈值",
                "type": "number",
                "required": False,
                "default": 0.92,
                "hint": "高于此值视为未偏离，跳过细检",
            },
        ],
        "params": {
            "enable_default_sort_tracking": False,
            "reference_image_url": "",
            "check_interval_sec": 5,
            "tilt_threshold_deg": 8,
            "perspective_threshold": 0.12,
            "confirm_count": 3,
            "cooldown_sec": 60,
            "ssim_skip_threshold": 0.92,
            "ssim_fail_threshold": 0.15,
            "min_match_count": 30,
            "min_inlier_ratio": 0.35,
            "max_side": 960,
            "enable_timing_log": False,
        },
        "alert_definitions": [
            {
                "key": "camera_tilt",
                "level": 2,
                "description": "摄像头相对校准模板发生明显角度/透视偏离。",
                "codes": {
                    "default": {
                        "code": RECOGNITION_CODE,
                        "description": RECOGNITION_DESC,
                    }
                },
            }
        ],
    }

    def _initialize(self) -> None:
        params = self.config.get("params", {}) if isinstance(self.config, dict) else {}
        self.reference_image_url = str(params.get("reference_image_url") or "").strip()
        self.check_interval_sec = max(1.0, float(params.get("check_interval_sec", 5) or 5))
        self.tilt_threshold_deg = max(
            0.1, float(params.get("tilt_threshold_deg", 8) or 8)
        )
        self.perspective_threshold = max(
            0.0, float(params.get("perspective_threshold", 0.12) or 0.12)
        )
        self.confirm_count = max(1, int(params.get("confirm_count", 3) or 3))
        self.cooldown_sec = max(0.0, float(params.get("cooldown_sec", 60) or 60))
        self.ssim_skip_threshold = float(params.get("ssim_skip_threshold", 0.92) or 0.92)
        self.ssim_fail_threshold = float(params.get("ssim_fail_threshold", 0.15) or 0.15)
        self.min_match_count = max(8, int(params.get("min_match_count", 30) or 30))
        self.min_inlier_ratio = float(params.get("min_inlier_ratio", 0.35) or 0.35)
        self.max_side = int(params.get("max_side", 960) or 960)

        self.alert_definitions: List[Dict[str, Any]] = list(
            self.config.get("alert_definitions") or []
        )
        self._reference_gray: Optional[np.ndarray] = None
        self._reference_bgr: Optional[np.ndarray] = None
        self._tilt_streak = 0
        self._last_alert_ts = 0.0
        self._last_result: Dict[str, Any] = {}
        self._last_infer_ts = 0.0

        if self.reference_image_url:
            try:
                self.reload_reference(self.reference_image_url)
            except Exception as e:
                self.log("warning", f"校准模板预加载失败: {e}")

        self.log(
            "info",
            f"初始化角度偏离检测: interval={self.check_interval_sec}s "
            f"tilt={self.tilt_threshold_deg}° persp={self.perspective_threshold} "
            f"confirm={self.confirm_count}",
        )

    def get_required_models(self) -> List[str]:
        return []

    def reload_reference(self, url: str) -> None:
        bgr = pose_core.load_image_bgr(url)
        gray, _ = pose_core.preprocess_pair(bgr, bgr, max_side=self.max_side)
        self._reference_bgr = bgr
        self._reference_gray = gray
        self.reference_image_url = url
        self._tilt_streak = 0

    def ensure_reference(self) -> np.ndarray:
        if self._reference_gray is None:
            if not self.reference_image_url:
                raise ValueError("未配置校准模板图片地址")
            self.reload_reference(self.reference_image_url)
        assert self._reference_gray is not None
        return self._reference_gray

    def _exceeds_tilt(self, pose: pose_core.PoseDetectResult) -> bool:
        return (
            pose.rotation_deg > self.tilt_threshold_deg
            or pose.perspective_score > self.perspective_threshold
        )

    def detect_frame(self, current_bgr: np.ndarray) -> Dict[str, Any]:
        """对单张 BGR 图做角度偏离检测。

        几何管线见 camera_pose_core.estimate_pose。
        本方法只看 rotation_deg / perspective_score → streak / 冷却 / 告警。
        """
        self.ensure_reference()
        if self._reference_bgr is None:
            raise ValueError("校准模板未加载")

        ref_gray, cur_gray = pose_core.preprocess_pair(
            self._reference_bgr, current_bgr, max_side=self.max_side
        )
        self._reference_gray = ref_gray

        pose = pose_core.estimate_pose(
            ref_gray,
            cur_gray,
            ssim_skip=self.ssim_skip_threshold,
            ssim_fail=self.ssim_fail_threshold,
            min_match_count=self.min_match_count,
            min_inlier_ratio=self.min_inlier_ratio,
        )

        now = time.time()
        active_alerts: List[Dict[str, Any]] = []
        triggered = False  # → has_camera_tilt；需同时满足下列条件才会为 True
        # 1) 几何估计成功：estimate_pose 返回 status=="normal"（非 skip/fail）。
        #    SSIM 过高会 skip，匹配不足/内点差会 fail，这两种都不告警。
        # 2) 角度或透视超阈（_exceeds_tilt）：
        #    rotation_deg > tilt_threshold_deg（默认 8°），或
        #    perspective_score > perspective_threshold（默认 0.12）。
        # 3) 连续确认够次数：超阈后 tilt_streak 累加，达到 confirm_count
        #    （线上默认 3；test_image/test_video 默认 1，单次超阈即可告警）。
        # 4) 不在冷却期：距上次告警已超过 cooldown_sec
        #    （线上默认 60s；测试里为 0）。
        # 任一不满足时 has_camera_tilt 仍为 False；
        # 可能是 tilt_pending（还在累计）或 tilt_cooldown（超阈但冷却中）。
        status = pose.status

        if pose.status in {"skip", "fail"}:
            self._tilt_streak = 0
        elif self._exceeds_tilt(pose):
            self._tilt_streak += 1
            status = "tilt_pending"
            if self._tilt_streak >= self.confirm_count:
                if now - self._last_alert_ts >= self.cooldown_sec:
                    triggered = True
                    self._last_alert_ts = now
                    self._tilt_streak = 0
                    status = "tilt"
                    active_alerts.append(
                        {
                            "key": "camera_tilt",
                            "level": 2,
                            "description": RECOGNITION_DESC,
                            "recognition_type": RECOGNITION_CODE,
                            "rotation_deg": round(pose.rotation_deg, 2),
                            "perspective_score": round(pose.perspective_score, 4),
                            "shift_px": round(pose.shift_px, 2),
                            "ssim": pose.ssim,
                            "inlier_ratio": round(pose.inlier_ratio, 3),
                        }
                    )
                else:
                    status = "tilt_cooldown"
        else:
            self._tilt_streak = 0
            status = "normal"

        result = {
            "skill_name": self.config.get("name") or "camera_tilt_detector",
            "run_mode": "stream",
            "status": status,
            "message": pose.message,
            "shift_px": round(float(pose.shift_px), 2),
            "rotation_deg": round(float(pose.rotation_deg), 2),
            "perspective_score": round(float(pose.perspective_score), 4),
            "ssim": pose.ssim,
            "match_count": pose.match_count,
            "inlier_ratio": round(float(pose.inlier_ratio), 3),
            "tilt_streak": self._tilt_streak,
            "tilt_threshold_deg": self.tilt_threshold_deg,
            "perspective_threshold": self.perspective_threshold,
            # has_camera_tilt=True 条件见上方 triggered 注释（几何成功 + 超阈 + 确认次数 + 非冷却）
            "has_camera_tilt": triggered,
            "recognition_types": [RECOGNITION_CODE] if triggered else [],
            "alert_definitions": list(self.alert_definitions),
            "active_alerts": active_alerts,
            "count": 0,
            "enter_count": 0,
        }
        self._last_result = result
        return result

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
            logger.exception("角度偏离检测失败")
            return SkillResult.error_result(str(e))

    def _draw_detections_on_frame(
        self, frame: np.ndarray, alert_data: Dict[str, Any]
    ) -> np.ndarray:
        """供实时视频流水线调用：把最近一次角度偏离检测结果画到画面上。"""
        try:
            data = alert_data if isinstance(alert_data, dict) else {}
            if not data and isinstance(self._last_result, dict):
                data = self._last_result
            return _draw_tilt_overlay(frame, data)
        except Exception as e:
            logger.error("绘制角度偏离检测结果失败: %s", e)
            return frame


def _ascii_only(text: Any, *, fallback: str = "") -> str:
    """OpenCV putText 无法可靠显示中文，仅保留可打印 ASCII。"""
    raw = str(text if text is not None else "")
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else " " for ch in raw)
    cleaned = " ".join(cleaned.split())
    return cleaned or fallback


def _draw_tilt_overlay(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """把角度偏离检测结果画到画面上（仅英文，避免 OpenCV 中文乱码）。"""
    status = _ascii_only(result.get("status"), fallback="unknown")
    if status in {"tilt", "fail"} or result.get("has_camera_tilt"):
        color = (0, 0, 255)
    elif status in {"tilt_pending", "tilt_cooldown"}:
        color = (0, 165, 255)
    elif status in {"normal", "skip"}:
        color = (0, 200, 0)
    else:
        color = (200, 200, 200)

    msg = _ascii_only(result.get("message"))
    lines = [
        f"status: {status}",
        f"rotation_deg: {result.get('rotation_deg')}",
        f"perspective: {result.get('perspective_score')}",
        f"shift_px: {result.get('shift_px')}",
        f"ssim: {result.get('ssim')}",
        f"match/inlier: {result.get('match_count')} / {result.get('inlier_ratio')}",
        f"streak: {result.get('tilt_streak')}  alert: {result.get('has_camera_tilt')}",
    ]
    if msg:
        lines.append(f"msg: {msg}")

    overlay = frame.copy()
    box_h = 28 + 22 * len(lines)
    cv2.rectangle(overlay, (8, 8), (min(frame.shape[1] - 8, 720), box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line[:90],
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
    """本地双图测试：模板图 + 测试图 → 输出角度偏离检测结果。

    默认:
      模板图 = test_images_videos/mobangtu.png
      测试图 = test_images_videos/camera-angle-deviation.png

    用法:
      python app/plugins/skills/camera_tilt_detector_skill.py
      python app/plugins/skills/camera_tilt_detector_skill.py --ref a.png --cur b.png
    """
    import argparse
    import json
    from copy import deepcopy

    default_ref = str(_CODE_ROOT / "test_images_videos" / "mobangtu.png")
    default_cur = str(_CODE_ROOT / "test_images_videos" / "camera-angle-deviation.png")

    parser = argparse.ArgumentParser(description="摄像头角度偏离检测（双图离线测试）")
    parser.add_argument(
        "--ref",
        "--reference",
        dest="reference",
        default=default_ref,
        help=f"校准模板图路径或 URL（默认: {default_ref}）",
    )
    parser.add_argument(
        "--cur",
        "--current",
        dest="current",
        default=default_cur,
        help=f"待检测图路径或 URL（默认: {default_cur}）",
    )
    parser.add_argument(
        "--tilt-threshold-deg",
        type=float,
        default=None,
        help="转角告警阈值（度），默认用技能配置",
    )
    parser.add_argument(
        "--perspective-threshold",
        type=float,
        default=None,
        help="透视告警阈值，默认用技能配置",
    )
    parser.add_argument(
        "--confirm-count",
        type=int,
        default=1,
        help="连续确认次数（离线单次测试默认 1，便于直接出最终结论）",
    )
    args = parser.parse_args()

    config = deepcopy(CameraTiltDetectorSkill.DEFAULT_CONFIG)
    params = config.setdefault("params", {})
    params["reference_image_url"] = args.reference
    params["confirm_count"] = max(1, int(args.confirm_count))
    params["cooldown_sec"] = 0
    if args.tilt_threshold_deg is not None:
        params["tilt_threshold_deg"] = float(args.tilt_threshold_deg)
    if args.perspective_threshold is not None:
        params["perspective_threshold"] = float(args.perspective_threshold)

    skill = CameraTiltDetectorSkill(config)
    current_bgr = pose_core.load_image_bgr(args.current)
    result = skill.detect_frame(current_bgr)

    print("=== 摄像头角度偏离检测结果 ===")
    print(f"模板图: {args.reference}")
    print(f"测试图: {args.current}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        f"结论: status={result.get('status')} "
        f"rotation_deg={result.get('rotation_deg')} "
        f"perspective_score={result.get('perspective_score')} "
        f"has_camera_tilt={result.get('has_camera_tilt')} "
        f"message={result.get('message')}"
    )
    sys.exit(0 if result.get("status") != "fail" else 1)


def test_video() -> None:
    """本机摄像头或本地视频文件测试角度偏离算法，结果叠加显示在画面上。

    默认用视频第一帧作为校准模板；可用 --ref 指定外部模板。

    用法:
      python app/plugins/skills/camera_tilt_detector_skill.py video
      python app/plugins/skills/camera_tilt_detector_skill.py video --camera 0
      python app/plugins/skills/camera_tilt_detector_skill.py video --source ./test.mp4
      python app/plugins/skills/camera_tilt_detector_skill.py video --source ./test.mp4 --loop

    按键: q 退出；s 把当前帧设为新模板。
    """
    import argparse
    from copy import deepcopy

    default_source = str(_CODE_ROOT / "test_images_videos" / "person_gouzi.mp4")

    parser = argparse.ArgumentParser(
        description="摄像头角度偏离检测（本机摄像头 / 本地视频测试）"
    )
    parser.add_argument(
        "--ref",
        "--reference",
        dest="reference",
        default="",
        help="校准模板图路径或 URL（默认空=使用视频第一帧）",
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
        "--tilt-threshold-deg",
        type=float,
        default=None,
        help="转角告警阈值（度），默认用技能配置",
    )
    parser.add_argument(
        "--perspective-threshold",
        type=float,
        default=None,
        help="透视告警阈值，默认用技能配置",
    )
    parser.add_argument(
        "--confirm-count",
        type=int,
        default=1,
        help="连续确认次数（实时预览默认 1）",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=200,
        help="检测间隔毫秒（默认 200，减轻本机算力压力）",
    )
    args = parser.parse_args()

    config = deepcopy(CameraTiltDetectorSkill.DEFAULT_CONFIG)
    params = config.setdefault("params", {})
    params["reference_image_url"] = str(args.reference or "").strip()
    params["confirm_count"] = max(1, int(args.confirm_count))
    params["cooldown_sec"] = 0
    if args.tilt_threshold_deg is not None:
        params["tilt_threshold_deg"] = float(args.tilt_threshold_deg)
    if args.perspective_threshold is not None:
        params["perspective_threshold"] = float(args.perspective_threshold)

    skill = CameraTiltDetectorSkill(config)

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
        # Windows 下 CAP_DSHOW 往往更稳
        api = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(args.camera), api)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(args.camera))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开本机摄像头 camera={args.camera}")
        source_desc = f"camera={args.camera}"

    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError("无法读取视频第一帧作为模板")

    if not skill.reference_image_url:
        gray, _ = pose_core.preprocess_pair(
            first_frame, first_frame, max_side=skill.max_side
        )
        skill._reference_bgr = first_frame.copy()
        skill._reference_gray = gray
        skill.reference_image_url = "<first_frame>"
        print("已用视频第一帧作为校准模板")
    else:
        print(f"使用外部校准模板: {skill.reference_image_url}")

    window = "camera_tilt_detector"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    last_result: Dict[str, Any] = {"status": "init", "message": "template ready"}
    last_infer_ts = 0.0
    interval_sec = max(0.0, float(args.interval_ms) / 1000.0)
    pending_frame: Optional[np.ndarray] = first_frame

    print(
        f"实时测试已启动: {source_desc} ref={skill.reference_image_url} "
        f"interval={args.interval_ms}ms  (q=退出, s=当前帧设为模板"
        f"{', loop' if source and args.loop else ''})"
    )
    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
            else:
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

            now = time.time()
            if now - last_infer_ts >= interval_sec:
                try:
                    last_result = skill.detect_frame(frame)
                except Exception as e:
                    last_result = {
                        # status 含义：
                        #   skip          — SSIM 很高，画面与模板高度一致，跳过细检（正常）
                        #   normal        — 几何估计成功，且转角/透视未超阈（正常）
                        #   fail          — 遮挡/失焦/匹配不足/内点差，或本处异常，无法可靠判决
                        #   tilt_pending  — 已超阈，但连续确认次数尚未达到 confirm_count
                        #   tilt          — 已超阈且确认够次、非冷却，触发角度偏离告警
                        #   tilt_cooldown — 已超阈且确认够次，但仍在告警冷却期内
                        "status": "fail",
                        "message": str(e),  # 失败原因说明
                        "rotation_deg": 0.0,  # 相对模板的转角（度）
                        "perspective_score": 0.0,  # 透视/俯仰偏离强度
                        "shift_px": 0.0,  # 相对模板的平移量（像素）
                        "ssim": None,  # 与模板的结构相似度
                        "match_count": 0,  # 特征匹配点数
                        "inlier_ratio": 0.0,  # 单应内点比例
                        "tilt_streak": 0,  # 连续超阈确认计数
                        "has_camera_tilt": False,  # 是否触发告警；True 需：几何 normal + 转角/透视超阈 + streak≥confirm_count + 非冷却
                    }
                last_infer_ts = now

            shown = _draw_tilt_overlay(frame, last_result)
            cv2.imshow(window, shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                # 用当前帧作为新模板，便于现场校准后立刻对比
                gray, _ = pose_core.preprocess_pair(frame, frame, max_side=skill.max_side)
                skill._reference_bgr = frame.copy()
                skill._reference_gray = gray
                skill._tilt_streak = 0
                skill.reference_image_url = "<live_frame>"
                print("已用当前帧更新校准模板")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # 默认双图测试；第一个参数为 video / test_video 时走本机摄像头
    if len(sys.argv) > 1 and sys.argv[1] in {"video", "test_video", "--video"}:
        sys.argv.pop(1)
        test_video()
    else:
        test_image()
