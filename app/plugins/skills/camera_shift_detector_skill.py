"""
摄像头位置挪移检测技能

运行模式 stream：接入实时视频流水线，按检测间隔对当前帧调用 detect_frame。
校准模板由外部提供（URL/路径），不做现场多帧融合校准。
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# 支持直接运行本文件: python app/plugins/skills/camera_shift_detector_skill.py
_CODE_ROOT = Path(__file__).resolve().parents[3]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import numpy as np

from app.services import camera_pose_core as pose_core
from app.skills.skill_base import BaseSkill, SkillResult

logger = logging.getLogger(__name__)

RECOGNITION_CODE = "08"
RECOGNITION_DESC = "摄像头位置挪移"


class CameraShiftDetectorSkill(BaseSkill):
    """背景模板 + 特征单应：仅判定位置挪移。"""

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": "camera_shift_detector",
        "name_zh": "摄像头位置挪移检测",
        "version": "1.0",
        "cover_image": "/skills/camera_shift_detector.png",
        "description": (
            "基于外部校准模板与实时视频帧，通过特征匹配与单应矩阵估计画面平移，"
            "检测摄像头位置挪移。"
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
                "hint": "外部提供的模板图 URL 或服务器本地路径",
            },
            {
                "key": "shift_threshold_px",
                "label": "平移告警阈值（像素）",
                "type": "number",
                "required": False,
                "default": 40,
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
            {
                "key": "ssim_skip_threshold",
                "label": "SSIM 跳过阈值",
                "type": "number",
                "required": False,
                "default": 0.92,
                "hint": "高于此值视为未挪动，跳过细检",
            },
        ],
        "params": {
            "enable_default_sort_tracking": False,
            "reference_image_url": "",
            "check_interval_sec": 5,
            "shift_threshold_px": 40,
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
                "key": "camera_shift",
                "level": 2,
                "description": "摄像头相对校准模板发生明显位置挪移。",
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
        self.shift_threshold_px = max(1.0, float(params.get("shift_threshold_px", 40) or 40))
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
        self._shift_streak = 0
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
            f"初始化位置挪移检测: interval={self.check_interval_sec}s "
            f"threshold={self.shift_threshold_px}px confirm={self.confirm_count}",
        )

    def get_required_models(self) -> List[str]:
        return []

    def reload_reference(self, url: str) -> None:
        bgr = pose_core.load_image_bgr(url)
        gray, _ = pose_core.preprocess_pair(bgr, bgr, max_side=self.max_side)
        self._reference_bgr = bgr
        self._reference_gray = gray
        self.reference_image_url = url
        self._shift_streak = 0

    def ensure_reference(self) -> np.ndarray:
        if self._reference_gray is None:
            if not self.reference_image_url:
                raise ValueError("未配置校准模板图片地址")
            self.reload_reference(self.reference_image_url)
        assert self._reference_gray is not None
        return self._reference_gray

    def detect_frame(self, current_bgr: np.ndarray) -> Dict[str, Any]:
        """对单张 BGR 图做挪移检测。

        几何管线见 camera_pose_core（L0 预处理 → L1–L4 由粗到细）。
        本方法负责 L4 业务判决：d 与 T_trans → streak / 冷却 / 告警。
        """
        self.ensure_reference()
        if self._reference_bgr is None:
            raise ValueError("校准模板未加载")

        # L0：灰度 + 统一尺度 → I0', It'
        ref_gray, cur_gray = pose_core.preprocess_pair(
            self._reference_bgr, current_bgr, max_side=self.max_side
        )
        self._reference_gray = ref_gray

        # L1–L4：任一级否决提前结束；normal 时带回 shift_px=d
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
        triggered = False
        status = pose.status

        # L4 业务：skip/fail 清零 streak；d>T_trans 累加；连续确认且非冷却 → 告警
        if pose.status in {"skip", "fail"}:
            self._shift_streak = 0
        elif pose.shift_px > self.shift_threshold_px:
            self._shift_streak += 1
            status = "shift_pending"
            if self._shift_streak >= self.confirm_count:
                if now - self._last_alert_ts >= self.cooldown_sec:
                    triggered = True
                    self._last_alert_ts = now
                    self._shift_streak = 0
                    status = "shift"
                    active_alerts.append(
                        {
                            "key": "camera_shift",
                            "level": 2,
                            "description": RECOGNITION_DESC,
                            "recognition_type": RECOGNITION_CODE,
                            "shift_px": round(pose.shift_px, 2),
                            "ssim": pose.ssim,
                            "inlier_ratio": round(pose.inlier_ratio, 3),
                        }
                    )
                else:
                    status = "shift_cooldown"
        else:
            self._shift_streak = 0
            status = "normal"

        result = {
            "skill_name": self.config.get("name") or "camera_shift_detector",
            "run_mode": "stream",
            "status": status,
            "message": pose.message,
            "shift_px": round(float(pose.shift_px), 2),
            "ssim": pose.ssim,
            "match_count": pose.match_count,
            "inlier_ratio": round(float(pose.inlier_ratio), 3),
            "shift_streak": self._shift_streak,
            "shift_threshold_px": self.shift_threshold_px,
            "has_camera_shift": triggered,
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
            logger.exception("位置挪移检测失败")
            return SkillResult.error_result(str(e))


def main() -> None:
    """本地双图测试：模板图 + 测试图 → 输出挪移检测结果。

    用法:
      python -m app.plugins.skills.camera_shift_detector_skill \\
          --ref template.jpg --cur test.jpg
    """
    import argparse
    import json
    import sys
    from copy import deepcopy

    parser = argparse.ArgumentParser(description="摄像头位置挪移检测（双图离线测试）")
    parser.add_argument(
        "--ref",
        "--reference",
        dest="reference",
        required=True,
        help="校准模板图路径或 URL",
    )
    parser.add_argument(
        "--cur",
        "--current",
        dest="current",
        required=True,
        help="待检测图路径或 URL",
    )
    parser.add_argument(
        "--shift-threshold-px",
        type=float,
        default=None,
        help="平移告警阈值（像素），默认用技能配置",
    )
    parser.add_argument(
        "--confirm-count",
        type=int,
        default=1,
        help="连续确认次数（离线单次测试默认 1，便于直接出最终结论）",
    )
    args = parser.parse_args()

    config = deepcopy(CameraShiftDetectorSkill.DEFAULT_CONFIG)
    params = config.setdefault("params", {})
    params["reference_image_url"] = args.reference
    params["confirm_count"] = max(1, int(args.confirm_count))
    params["cooldown_sec"] = 0
    if args.shift_threshold_px is not None:
        params["shift_threshold_px"] = float(args.shift_threshold_px)

    skill = CameraShiftDetectorSkill(config)
    current_bgr = pose_core.load_image_bgr(args.current)
    result = skill.detect_frame(current_bgr)

    print("=== 摄像头位置挪移检测结果 ===")
    print(f"模板图: {args.reference}")
    print(f"测试图: {args.current}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        f"结论: status={result.get('status')} "
        f"shift_px={result.get('shift_px')} "
        f"has_camera_shift={result.get('has_camera_shift')} "
        f"message={result.get('message')}"
    )
    sys.exit(0 if result.get("status") != "fail" else 1)


if __name__ == "__main__":
    main()
