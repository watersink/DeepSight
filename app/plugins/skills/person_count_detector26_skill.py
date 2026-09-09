"""
人数检测技能 (YOLO26)

规则：
  - 使用 yolo26_person 原始 COCO 检测模型，后处理中仅保留 person 类
  - 可选电子围栏过滤
  - 统计当前人数
  - count_line：中心轨迹穿线后，中心点 + 框四角中至少 3 个角进入 enter_point 侧，
    并连续确认 N 帧后 enter_count+1；反向离开到对侧不减计数，触发“翻越出去”告警
  - bypass_line：不做进出计数；同样按中心+至少3角确认后，进入/离开分别触发绕行告警
  - 短时丢检：用上一帧速度预测中心点续轨迹，避免过线中途漏计
  - 轨迹优先与 count_line 求交确定归属，命中后不再归属 bypass_line
"""

import sys
from pathlib import Path

# 支持直接运行本文件: python app/plugins/skills/person_count_detector26_skill.py
_CODE_ROOT = Path(__file__).resolve().parents[3]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Union, Optional, Set
from app.skills.skill_base import BaseSkill, SkillResult
from app.services.triton_client import triton_client
import logging

logger = logging.getLogger(__name__)

# COCO 80 类数据集中 person 的类别 id
COCO_PERSON_CLASS_ID = 0

# 过线判定：点落在线段延长线上时的容差（像素）
_LINE_SIDE_EPS = 1e-3
# 线段相交参数容差
_SEG_INTERSECT_EPS = 1e-6

_LINE_KIND_COUNT = "count"
_LINE_KIND_BYPASS = "bypass"

# 告警人体框颜色（BGR）：01/02 黄色；04/05/06/07 红色
_ALERT_BOX_COLOR_COUNT = (0, 255, 255)
_ALERT_BOX_COLOR_VIOLATION = (0, 0, 255)


class PersonCountDetector26Skill(BaseSkill):
    """
    简单人数检测技能 (YOLO26)

    使用 yolo26_person 原始检测模型，从 COCO 多类输出中筛选 person。
    支持用户画线 + 进入侧参考点的进入人数统计（enter_count），
    count_line 反向出线视为翻越告警；bypass_line 进出均告警且不计数。

    编码	含义	        触发条件
    01    人员计数（入）    正常过线计数 + gate_direction=IN（画面黄框）
    02    人员计数（出）    正常过线计数 + gate_direction=OUT（画面黄框）
    04    非常规通道入井    bypass 绕行进去（画面红框）
    05    非常规通道出井    bypass 绕行出去（画面红框）
    06    入井闸机出闸      闸机线反向翻越 + gate_direction=IN（画面红框）
    07    出井闸机入闸      闸机线反向翻越 + gate_direction=OUT（画面红框）
    """

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": "person_count_detector26",
        "name_zh": "人流量检测(YOLO26)",
        "version": "1.0",
        "cover_image": "/skills/person_count_detector26.png",
        "description": (
            "基于 yolo26_person 原始 COCO 模型的人数检测技能，后处理仅保留 person 类；"
            "支持 count_line 进入人数统计与翻越出线告警；支持 bypass_line 绕行进出告警"
        ),
        "status": True,
        "required_models": ["yolo26_person"],
        "params": {
            "classes": ["person"],
            "target_class_ids": [COCO_PERSON_CLASS_ID],
            "conf_thres": 0.5,
            "iou_thres": 0.45,
            "max_det": 300,
            "input_size": [640, 640],
            "enable_default_sort_tracking": True,
            "tracking_algorithm": "sort",
            "tracking_frame_rate": 30,
            "tracking_max_age": 30,
            "tracking_min_hits": 1,
            "tracking_iou_threshold": 0.3,
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "track_buffer": 30,
            "match_thresh": 0.8,
            "new_track_thresh": 0.6,
            "tracking_history_frames": 10,
            "enable_track_history": True,
            "model_version": "1",
            "enable_timing_log": False,
            # 过线进入人数初始值（仅统计进入 enter_point 一侧）
            "enter_count": 0,
            # 闸机方向：IN=入闸机口，OUT=出闸机口（影响画面参考点标注）
            "gate_direction": "IN",
            # 中心过线后，目标侧确认需连续满足的帧数
            "cross_confirm_frames": 2,
            # 四角中至少多少个角进入目标侧才算确认（配合中心点）
            "corner_confirm_min": 3,
            # 检测短暂丢失时，用速度预测中心点续轨迹的最大帧数
            "track_lost_keep_frames": 10,
            # 确认过程中允许连续不满足条件的容错帧数（避免断 1 帧就清零 streak）
            "cross_confirm_miss_tolerance": 1,
            # 煤矿编码（12位）/ 摄像仪编码（20位），用于识别结果上传接口
            "mine_code": "123456789012",
            "camera_code": "12345678901234567890",
            # 过线计数线（必填）：启动时必须由请求上传，默认空；空则报错
            # 例: {"lines":[{"line":[p1,p2],"enter_point":{...}}, ...]}
            # enter_point 决定“进入侧”：人员应朝该点所在一侧过线才会 +1
            "count_line": {},
            # 非计数绕行线（可选）：未上传或为空则不做绕行检测
            # 例: {"lines":[{"line":[p1,p2],"bypass_point":{...}}, ...]}
            "bypass_line": {},
        },
        "alert_definitions": [
            {
                "key": "enter_count",
                "level": 0,
                "description": "人员经闸机线进入 enter_point 一侧，人数增加。",
                # 识别类型：按 gate_direction 区分入/出
                "codes": {
                    "IN": {"code": "01", "description": "人员计数（入）"},
                    "OUT": {"code": "02", "description": "人员计数（出）"},
                },
            },
            {
                "key": "count_exit",
                "level": 1,
                "description": "人员经闸机线从进入侧翻越到对侧。",
                # 识别类型：入井闸机反向出闸 / 出井闸机反向入闸
                "codes": {
                    "IN": {"code": "06", "description": "入井闸机出闸"},
                    "OUT": {"code": "07", "description": "出井闸机入闸"},
                },
            },
            {
                "key": "bypass",
                "level": 2,
                "description": "人员穿过绕行线（非常规通道）。",
                # 识别类型：按绕行方向区分入井/出井（与 gate_direction 无关）
                "codes": {
                    "enter": {"code": "04", "description": "非常规通道入井"},
                    "exit": {"code": "05", "description": "非常规通道出井"},
                },
            },
        ]
    }

    def _initialize(self) -> None:
        params = self.config.get("params", {})

        self.classes = params.get("classes", ["person"])
        target_class_ids = params.get("target_class_ids", [COCO_PERSON_CLASS_ID])
        self.target_class_ids: Set[int] = {int(cid) for cid in target_class_ids}
        self.class_names = {COCO_PERSON_CLASS_ID: "person"}

        self.conf_thres = params.get("conf_thres", 0.5)
        self.iou_thres = params.get("iou_thres", 0.45)
        self.max_det = params.get("max_det", 300)
        self.input_width, self.input_height = params.get("input_size", [640, 640])

        self.required_models = self.config.get("required_models", ["yolo26_person"])
        self.model_name = self.required_models[0]
        self.model_version = str(params.get("model_version", "") or "").strip()

        self.enable_default_sort_tracking = params.get("enable_default_sort_tracking", True)
        self.tracking_algorithm = params.get("tracking_algorithm", "bytetrack")

        # 过线进入人数：仅统计进入 enter_point 一侧
        self.enter_count_initial: int = int(params.get("enter_count", 0) or 0)
        self.enter_count: int = self.enter_count_initial
        self.gate_direction: str = self._normalize_gate_direction(
            params.get("gate_direction", "IN")
        )
        self.mine_code: str = str(params.get("mine_code") or "").strip()
        self.camera_code: str = str(params.get("camera_code") or "").strip()
        # count_line 必填；bypass_line 可选（空则不做绕行逻辑）
        self.count_line_config: Optional[Dict[str, Any]] = self._normalize_optional_line_config(
            params.get("count_line")
        )
        if not self.count_line_config:
            raise ValueError(
                "count_line 不能为空：启动 person_count_detector26 时必须上传过线计数线配置"
            )
        self._active_count_line: Optional[Dict[str, Any]] = None
        self.bypass_line_config: Optional[Dict[str, Any]] = self._normalize_optional_line_config(
            params.get("bypass_line")
        )
        self._active_bypass_line: Optional[Dict[str, Any]] = None
        self._latest_bypass_events: List[Dict[str, Any]] = []
        self._latest_count_exit_events: List[Dict[str, Any]] = []
        self._latest_enter_events: List[Dict[str, Any]] = []
        self.alert_definitions: List[Dict[str, Any]] = list(
            self.config.get("alert_definitions") or []
        )
        self._alert_def_by_key: Dict[str, Dict[str, Any]] = {}
        for item in self.alert_definitions:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            if key:
                self._alert_def_by_key[key] = item
        # 兼容未配置 key 的旧结构
        if "enter_count" not in self._alert_def_by_key:
            for item in self.alert_definitions:
                if isinstance(item, dict) and int(item.get("level", -1) or -1) == 0:
                    self._alert_def_by_key["enter_count"] = item
                    break
        if "count_exit" not in self._alert_def_by_key and len(self.alert_definitions) >= 1:
            # 优先按 level=1，否则取第一个非 enter 定义
            for item in self.alert_definitions:
                if isinstance(item, dict) and int(item.get("level", -1) or -1) == 1:
                    self._alert_def_by_key["count_exit"] = item
                    break
            if "count_exit" not in self._alert_def_by_key:
                self._alert_def_by_key["count_exit"] = self.alert_definitions[0]
        if "bypass" not in self._alert_def_by_key and len(self.alert_definitions) >= 2:
            for item in self.alert_definitions:
                if isinstance(item, dict) and int(item.get("level", -1) or -1) == 2:
                    self._alert_def_by_key["bypass"] = item
                    break
            if "bypass" not in self._alert_def_by_key:
                self._alert_def_by_key["bypass"] = self.alert_definitions[-1]
        self.cross_confirm_frames: int = max(
            1, int(params.get("cross_confirm_frames", 2) or 2)
        )
        self.corner_confirm_min: int = min(
            4, max(1, int(params.get("corner_confirm_min", 3) or 3))
        )
        self.track_lost_keep_frames: int = max(
            0, int(params.get("track_lost_keep_frames", 10) or 0)
        )
        self.cross_confirm_miss_tolerance: int = max(
            0, int(params.get("cross_confirm_miss_tolerance", 1) or 0)
        )
        # track_id -> 上一帧检测框中心点（用于构造轨迹段求交）
        self._track_prev_center: Dict[Any, Tuple[float, float]] = {}
        # track_id -> 上一帧中心速度（像素/帧），用于短时丢检预测
        self._track_velocity: Dict[Any, Tuple[float, float]] = {}
        # track_id -> 连续丢检帧数
        self._track_lost_frames: Dict[Any, int] = {}
        # track_id -> 最近一次真实检测（预测续轨迹时复用）
        self._track_last_detection: Dict[Any, Dict[str, Any]] = {}
        # track_id -> 归属线类型（count 优先于 bypass，命中后粘性保留）
        self._track_line_affinity: Dict[Any, str] = {}
        # (track_id, kind, line_index) -> 已确认完整侧 (+ref / -ref)
        self._line_confirmed_side: Dict[Tuple[Any, str, int], int] = {}
        # (track_id, kind, line_index) -> 中心点已穿越的待确认方向 (+1 / -1)
        self._line_pending_direction: Dict[Tuple[Any, str, int], int] = {}
        # (track_id, kind, line_index) -> 中心点在待确认目标侧的连续停留帧数
        self._line_side_streak: Dict[Tuple[Any, str, int], int] = {}
        # (track_id, kind, line_index) -> 确认条件连续不满足的帧数（容错用）
        self._line_side_miss: Dict[Tuple[Any, str, int], int] = {}
        # track_id -> 上一帧中心相对各 count/bypass 线的侧别缓存不需要；用 prev_point 现算
        # 避免每帧重复打印 count_line 方向说明
        self._count_line_dir_log_key: Optional[str] = None

        self.log(
            "info",
            f"初始化 YOLO26 人数检测技能: model={self.model_name}"
            f"{f' v{self.model_version}' if self.model_version else ' (latest)'}, "
            f"target_classes={sorted(self.target_class_ids)}, "
            f"tracking={'off' if not self.enable_default_sort_tracking else self.tracking_algorithm}, "
            f"count_line={'on' if self.count_line_config else 'off'}, "
            f"gate_direction={self.gate_direction}, "
            f"bypass_line={'on' if self.bypass_line_config else 'off'}, "
            f"cross_confirm_frames={self.cross_confirm_frames}, "
            f"corner_confirm_min={self.corner_confirm_min}, "
            f"track_lost_keep_frames={self.track_lost_keep_frames}, "
            f"cross_confirm_miss_tolerance={self.cross_confirm_miss_tolerance}"
        )

    @staticmethod
    def _normalize_gate_direction(raw: Any) -> str:
        """闸机方向：仅允许 IN / OUT，默认 IN。"""
        value = str(raw or "IN").strip().upper()
        if value in {"IN", "OUT"}:
            return value
        return "IN"

    @staticmethod
    def _is_line_config_present(cfg: Any) -> bool:
        """判断线段配置是否有效（非空且含 lines/line）。"""
        if not isinstance(cfg, dict) or not cfg:
            return False
        lines = cfg.get("lines")
        if isinstance(lines, list) and len(lines) > 0:
            return True
        if cfg.get("line") is not None:
            return True
        return False

    @classmethod
    def _normalize_optional_line_config(cls, cfg: Any) -> Optional[Dict[str, Any]]:
        """空配置统一为 None，便于后续跳过相关逻辑。"""
        if not cls._is_line_config_present(cfg):
            return None
        return cfg if isinstance(cfg, dict) else None

    def get_required_models(self) -> List[str]:
        return self.required_models

    def _get_detection_point(self, detection: Dict) -> Optional[Tuple[float, float]]:
        """过线 / 绕行统计使用检测框中心点（与跟踪轨迹点一致）。"""
        bbox = detection.get("bbox", [])
        if len(bbox) < 4:
            return None

        x1, y1, x2, y2 = bbox
        return ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0)

    @staticmethod
    def _get_bbox_corners(detection: Dict) -> Optional[List[Tuple[float, float]]]:
        """检测框四角：(左上, 右上, 右下, 左下)。"""
        bbox = detection.get("bbox", [])
        if len(bbox) < 4:
            return None
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    @staticmethod
    def _parse_xy_point(raw: Any) -> Optional[Tuple[float, float]]:
        """解析点：{x,y} / [x,y] / (x,y) -> (x, y)。"""
        if isinstance(raw, dict):
            x, y = raw.get("x"), raw.get("y")
        elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
            x, y = raw[0], raw[1]
        else:
            return None
        try:
            return (float(x), float(y))
        except (TypeError, ValueError):
            return None

    def _resolve_count_line_config(
        self,
        input_data: Any,
        fence_config: Optional[Dict],
    ) -> Optional[Dict[str, Any]]:
        return self._resolve_line_config(input_data, fence_config, "count_line", self.count_line_config)

    def _resolve_bypass_line_config(
        self,
        input_data: Any,
        fence_config: Optional[Dict],
    ) -> Optional[Dict[str, Any]]:
        return self._resolve_line_config(input_data, fence_config, "bypass_line", self.bypass_line_config)

    @staticmethod
    def _resolve_line_config(
        input_data: Any,
        fence_config: Optional[Dict],
        key: str,
        default_config: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        解析线段配置，优先级：
          1) input_data[key]
          2) fence_config[key]
          3) params 默认值
        """
        if isinstance(input_data, dict):
            cfg = input_data.get(key)
            if PersonCountDetector26Skill._is_line_config_present(cfg):
                return cfg
        if isinstance(fence_config, dict):
            cfg = fence_config.get(key)
            if PersonCountDetector26Skill._is_line_config_present(cfg):
                return cfg
        if PersonCountDetector26Skill._is_line_config_present(default_config):
            return default_config
        return None

    def _parse_segment_endpoints(
        self,
        seg_raw: Any,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if not isinstance(seg_raw, (list, tuple)) or len(seg_raw) < 2:
            return None
        a = self._parse_xy_point(seg_raw[0])
        b = self._parse_xy_point(seg_raw[1])
        if a is None or b is None:
            return None
        if abs(a[0] - b[0]) < 1e-6 and abs(a[1] - b[1]) < 1e-6:
            return None
        return a, b

    def _parse_count_line_items(
        self,
        count_line: Dict[str, Any],
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """
        解析 count_line，返回 [(line_a, line_b, enter_point), ...]。

        期望结构（线段与 enter_point 一一对应）:
          {"lines": [{"line": [p1, p2], "enter_point": {...}}, ...]}
        """
        if not isinstance(count_line, dict):
            return []

        lines_raw = count_line.get("lines")
        if not isinstance(lines_raw, list):
            return []

        items: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = []
        for item in lines_raw:
            if not isinstance(item, dict):
                continue
            endpoints = self._parse_segment_endpoints(item.get("line"))
            enter_px = self._parse_xy_point(item.get("enter_point"))
            if endpoints is None or enter_px is None:
                continue
            items.append((endpoints[0], endpoints[1], enter_px))
        return items

    def _parse_bypass_line_items(
        self,
        bypass_line: Dict[str, Any],
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        """
        解析 bypass_line，返回 [(line_a, line_b, bypass_point), ...]。

        期望结构（线段与 bypass_point 一一对应）:
          {"lines": [{"line": [p1, p2], "bypass_point": {...}}, ...]}
        """
        if not isinstance(bypass_line, dict):
            return []

        lines_raw = bypass_line.get("lines")
        if not isinstance(lines_raw, list):
            return []

        items: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = []
        for item in lines_raw:
            if not isinstance(item, dict):
                continue
            endpoints = self._parse_segment_endpoints(item.get("line"))
            bypass_px = self._parse_xy_point(item.get("bypass_point"))
            if endpoints is None or bypass_px is None:
                continue
            items.append((endpoints[0], endpoints[1], bypass_px))
        return items

    @staticmethod
    def _line_config_scale(
        line_cfg: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> Tuple[float, float]:
        """
        若配置中带有标注截图分辨率，且与当前帧不一致，则按比例缩放像素坐标。
        getSnap 分辨率常与推流读帧不同，不缩放会导致线画在画面外。
        """
        width, height = image_size
        raw_w = line_cfg.get("image_width", line_cfg.get("width"))
        raw_h = line_cfg.get("image_height", line_cfg.get("height"))
        try:
            cfg_w = float(raw_w) if raw_w is not None else 0.0
            cfg_h = float(raw_h) if raw_h is not None else 0.0
        except (TypeError, ValueError):
            return 1.0, 1.0
        if cfg_w <= 1.0 or cfg_h <= 1.0 or width <= 0 or height <= 0:
            return 1.0, 1.0
        if abs(cfg_w - width) < 1.0 and abs(cfg_h - height) < 1.0:
            return 1.0, 1.0
        return width / cfg_w, height / cfg_h

    @staticmethod
    def _scale_xy(
        pt: Tuple[float, float],
        sx: float,
        sy: float,
    ) -> Tuple[float, float]:
        if sx == 1.0 and sy == 1.0:
            return pt
        return (pt[0] * sx, pt[1] * sy)

    def _normalize_count_line(
        self,
        count_line: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        """
        解析 count_line 像素坐标并计算每段线对应的进入侧符号。

        期望结构（像素，线段与点一一对应）:
          {
            "image_width": 1920, "image_height": 1080,  # 可选，标注截图尺寸
            "lines": [
              {"line": [{"x":432,"y":328}, {"x":723,"y":229}], "enter_point": {"x":513,"y":250}},
              ...
            ]
          }
        """
        width, height = image_size
        if width <= 0 or height <= 0:
            return None

        sx, sy = self._line_config_scale(count_line, image_size)
        if sx != 1.0 or sy != 1.0:
            self.log(
                "info",
                f"count_line 坐标按标注分辨率缩放 "
                f"cfg=({count_line.get('image_width')}x{count_line.get('image_height')}) "
                f"-> frame=({width}x{height}) scale=({sx:.4f},{sy:.4f})",
            )

        parsed_segments: List[Dict[str, Any]] = []
        for seg_idx, (line_a, line_b, enter_px) in enumerate(
            self._parse_count_line_items(count_line)
        ):
            line_a = self._scale_xy(line_a, sx, sy)
            line_b = self._scale_xy(line_b, sx, sy)
            enter_px = self._scale_xy(enter_px, sx, sy)
            enter_side = self._side_of_line(enter_px, line_a, line_b)
            if enter_side == 0:
                self.log("warning", "进入侧参考点落在计数线段上，跳过该线段")
                continue
            # 整段落在画面外则告警（常见于未写入 image_size 且分辨率不一致）
            xs = (line_a[0], line_b[0], enter_px[0])
            ys = (line_a[1], line_b[1], enter_px[1])
            if (
                max(xs) < 0
                or max(ys) < 0
                or min(xs) > width
                or min(ys) > height
            ):
                self.log(
                    "warning",
                    f"count_line[{seg_idx}] 坐标完全在画面外 "
                    f"frame={width}x{height}，请重新截图绘制并保存",
                )
            parsed_segments.append(
                {
                    "line_a": line_a,
                    "line_b": line_b,
                    "enter_point": enter_px,
                    "enter_side": enter_side,
                }
            )

        if not parsed_segments:
            self.log("warning", "count_line 无有效线段")
            return None

        # 方向说明只打一次（配置/分辨率变化时再打），不是过线告警
        log_key = "|".join(
            f"{i}:{s['enter_side']}:{s['enter_point']}:{s['line_a']}:{s['line_b']}"
            for i, s in enumerate(parsed_segments)
        )
        if log_key != self._count_line_dir_log_key:
            self._count_line_dir_log_key = log_key
            for seg_idx, s in enumerate(parsed_segments):
                ep, la, lb = s["enter_point"], s["line_a"], s["line_b"]
                self.log(
                    "info",
                    f"count_line[{seg_idx}] enter_side={s['enter_side']} "
                    f"enter_point=({ep[0]:.0f},{ep[1]:.0f}) "
                    f"line=({la[0]:.0f},{la[1]:.0f})->({lb[0]:.0f},{lb[1]:.0f})；"
                    f"人员须朝 enter_point 一侧过线才会 +1（反向只告警不减数）",
                )

        return {"segments": parsed_segments}

    def _normalize_bypass_line(
        self,
        bypass_line: Dict[str, Any],
        image_size: Tuple[int, int],
    ) -> Optional[Dict[str, Any]]:
        """
        解析 bypass_line 像素坐标。

        期望结构（像素，线段与点一一对应）:
          {
            "image_width": 1920, "image_height": 1080,  # 可选
            "lines": [
              {"line": [{"x":154,"y":421}, {"x":434,"y":337}], "bypass_point": {"x":280,"y":348}},
              ...
            ]
          }
        """
        width, height = image_size
        if width <= 0 or height <= 0:
            return None

        sx, sy = self._line_config_scale(bypass_line, image_size)
        if sx != 1.0 or sy != 1.0:
            self.log(
                "info",
                f"bypass_line 坐标按标注分辨率缩放 "
                f"cfg=({bypass_line.get('image_width')}x{bypass_line.get('image_height')}) "
                f"-> frame=({width}x{height}) scale=({sx:.4f},{sy:.4f})",
            )

        parsed_segments: List[Dict[str, Any]] = []
        for seg_idx, (line_a, line_b, bypass_px) in enumerate(
            self._parse_bypass_line_items(bypass_line)
        ):
            line_a = self._scale_xy(line_a, sx, sy)
            line_b = self._scale_xy(line_b, sx, sy)
            bypass_px = self._scale_xy(bypass_px, sx, sy)
            bypass_side = self._side_of_line(bypass_px, line_a, line_b)
            if bypass_side == 0:
                self.log("warning", "绕行侧参考点落在绕行线段上，跳过该线段")
                continue
            xs = (line_a[0], line_b[0], bypass_px[0])
            ys = (line_a[1], line_b[1], bypass_px[1])
            if (
                max(xs) < 0
                or max(ys) < 0
                or min(xs) > width
                or min(ys) > height
            ):
                self.log(
                    "warning",
                    f"bypass_line[{seg_idx}] 坐标完全在画面外 "
                    f"frame={width}x{height}，请重新截图绘制并保存",
                )
            parsed_segments.append(
                {
                    "line_a": line_a,
                    "line_b": line_b,
                    "bypass_point": bypass_px,
                    "bypass_side": bypass_side,
                }
            )

        if not parsed_segments:
            self.log("warning", "bypass_line 无有效线段")
            return None

        return {"segments": parsed_segments}

    @staticmethod
    def _side_of_line(
        point: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> int:
        """点相对有向线段 AB 的侧：+1 / -1 / 0(共线)。"""
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if abs(cross) <= _LINE_SIDE_EPS:
            return 0
        return 1 if cross > 0 else -1

    @staticmethod
    def _segments_intersect(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        q1: Tuple[float, float],
        q2: Tuple[float, float],
    ) -> bool:
        """判断轨迹段 p1->p2 与线段 q1->q2 是否相交（含端点）。"""
        ax, ay = p1
        bx, by = p2
        cx, cy = q1
        dx, dy = q2

        den = (ax - bx) * (cy - dy) - (ay - by) * (cx - dx)
        if abs(den) <= _SEG_INTERSECT_EPS:
            return False

        t = ((ax - cx) * (cy - dy) - (ay - cy) * (cx - dx)) / den
        u = -((ax - bx) * (ay - cy) - (ay - by) * (ax - cx)) / den
        return (
            -_SEG_INTERSECT_EPS <= t <= 1.0 + _SEG_INTERSECT_EPS
            and -_SEG_INTERSECT_EPS <= u <= 1.0 + _SEG_INTERSECT_EPS
        )

    def _crossing_direction(
        self,
        prev_point: Tuple[float, float],
        curr_point: Tuple[float, float],
        line_a: Tuple[float, float],
        line_b: Tuple[float, float],
        reference_side: int,
    ) -> Optional[int]:
        """
        中心点轨迹段与线段相交时的穿越方向：
          +1 进入 reference_side，-1 进入对侧，None 无法判定。
        """
        prev_side = self._side_of_line(prev_point, line_a, line_b)
        curr_side = self._side_of_line(curr_point, line_a, line_b)
        if curr_side == reference_side and prev_side != reference_side:
            return 1
        if curr_side == -reference_side and prev_side != -reference_side:
            return -1
        if prev_side == reference_side and curr_side != reference_side:
            return -1
        if prev_side == -reference_side and curr_side != -reference_side:
            return 1
        return None

    def _points_all_on_side(
        self,
        points: List[Tuple[float, float]],
        line_a: Tuple[float, float],
        line_b: Tuple[float, float],
        target_side: int,
    ) -> bool:
        """点集是否全部在目标侧（忽略落在线上的点，但不能出现对侧点）。"""
        if not points:
            return False
        saw_target = False
        for point in points:
            side = self._side_of_line(point, line_a, line_b)
            if side == 0:
                continue
            if side != target_side:
                return False
            saw_target = True
        return saw_target

    def _count_corners_on_side(
        self,
        corners: List[Tuple[float, float]],
        line_a: Tuple[float, float],
        line_b: Tuple[float, float],
        target_side: int,
    ) -> int:
        """统计四角中落在目标侧的数量（落在线上的角不计）。"""
        count = 0
        for point in corners:
            side = self._side_of_line(point, line_a, line_b)
            if side == target_side:
                count += 1
        return count

    def _is_center_and_corners_confirmed(
        self,
        center: Tuple[float, float],
        corners: List[Tuple[float, float]],
        line_a: Tuple[float, float],
        line_b: Tuple[float, float],
        target_side: int,
    ) -> bool:
        """
        过线确认条件：
          1) 中心点必须在目标侧
          2) 四角中至少 corner_confirm_min 个角在目标侧（默认 3）
        """
        if self._side_of_line(center, line_a, line_b) != target_side:
            return False
        if not corners:
            return False
        return (
            self._count_corners_on_side(corners, line_a, line_b, target_side)
            >= self.corner_confirm_min
        )

    def _is_fully_on_side(
        self,
        center: Tuple[float, float],
        corners: List[Tuple[float, float]],
        line_a: Tuple[float, float],
        line_b: Tuple[float, float],
        target_side: int,
    ) -> bool:
        """中心点与检测框四角全部在目标侧（保留兼容，过线确认已改用多数角）。"""
        return self._points_all_on_side(
            [center], line_a, line_b, target_side
        ) and self._points_all_on_side(
            corners, line_a, line_b, target_side
        )

    def _line_state_key(self, track_id: Any, kind: str, line_index: int) -> Tuple[Any, str, int]:
        return (track_id, kind, line_index)

    def _trajectory_hits_segments(
        self,
        prev_point: Tuple[float, float],
        curr_point: Tuple[float, float],
        segments: List[Dict[str, Any]],
        side_key: str,
    ) -> List[Tuple[int, int]]:
        """返回轨迹段与线段相交事件 [(line_index, direction), ...]。"""
        events: List[Tuple[int, int]] = []
        for line_index, segment in enumerate(segments):
            line_a = segment["line_a"]
            line_b = segment["line_b"]
            if not self._segments_intersect(prev_point, curr_point, line_a, line_b):
                continue
            direction = self._crossing_direction(
                prev_point,
                curr_point,
                line_a,
                line_b,
                segment[side_key],
            )
            if direction is None:
                continue
            events.append((line_index, direction))
        return events

    def _inject_pending_by_side_flip(
        self,
        track_id: Any,
        kind: str,
        segments: List[Dict[str, Any]],
        side_key: str,
        prev_point: Tuple[float, float],
        curr_point: Tuple[float, float],
    ) -> None:
        """
        丢检后中心点可能“跳过”线段，轨迹段与线不相交。
        若上一帧中心与当前中心分属线两侧，则补写 pending，保证后续能继续确认计数。
        """
        for line_index, segment in enumerate(segments):
            line_a = segment["line_a"]
            line_b = segment["line_b"]
            reference_side = segment[side_key]
            prev_side = self._side_of_line(prev_point, line_a, line_b)
            curr_side = self._side_of_line(curr_point, line_a, line_b)
            if prev_side == 0 or curr_side == 0 or prev_side == curr_side:
                continue
            state_key = self._line_state_key(track_id, kind, line_index)
            if prev_side == -reference_side and curr_side == reference_side:
                self._line_pending_direction[state_key] = 1
                self._line_side_miss.pop(state_key, None)
            elif prev_side == reference_side and curr_side == -reference_side:
                self._line_pending_direction[state_key] = -1
                self._line_side_miss.pop(state_key, None)

    def _resolve_center_crossing(
        self,
        track_id: Any,
        kind: str,
        segments: List[Dict[str, Any]],
        side_key: str,
        center: Tuple[float, float],
        corners: List[Tuple[float, float]],
        prev_center: Optional[Tuple[float, float]] = None,
    ) -> List[Tuple[int, int]]:
        """
        过线确认：
          - 中心轨迹穿线（或丢检后两侧翻转）后写入 pending
          - 中心点必须在目标侧，且四角中至少 corner_confirm_min 个角也在目标侧
          - 上述条件连续满足 cross_confirm_frames 帧才确认
          - 中间允许 cross_confirm_miss_tolerance 帧不满足，不断开确认流程
        返回 [(line_index, direction), ...]，direction: +1 进入参考侧，-1 进入对侧。
        """
        events: List[Tuple[int, int]] = []
        for line_index, segment in enumerate(segments):
            line_a = segment["line_a"]
            line_b = segment["line_b"]
            reference_side = segment[side_key]
            state_key = self._line_state_key(track_id, kind, line_index)
            confirmed = self._line_confirmed_side.get(state_key)
            pending = self._line_pending_direction.get(state_key)

            # 丢检跳跃：上一中心在对侧、当前已在本侧 → 补 pending
            if prev_center is not None and pending is None:
                prev_side = self._side_of_line(prev_center, line_a, line_b)
                curr_side = self._side_of_line(center, line_a, line_b)
                if (
                    prev_side == -reference_side
                    and curr_side == reference_side
                    and confirmed != reference_side
                ):
                    pending = 1
                    self._line_pending_direction[state_key] = 1
                elif (
                    prev_side == reference_side
                    and curr_side == -reference_side
                    and confirmed != -reference_side
                ):
                    pending = -1
                    self._line_pending_direction[state_key] = -1

            on_enter = self._is_center_and_corners_confirmed(
                center, corners, line_a, line_b, reference_side
            )
            on_exit = self._is_center_and_corners_confirmed(
                center, corners, line_a, line_b, -reference_side
            )

            if on_enter:
                self._line_side_miss.pop(state_key, None)
                if confirmed == reference_side:
                    self._line_pending_direction.pop(state_key, None)
                    self._line_side_streak.pop(state_key, None)
                    continue
                if pending == 1 or confirmed == -reference_side:
                    streak = int(self._line_side_streak.get(state_key, 0) or 0) + 1
                    self._line_side_streak[state_key] = streak
                    if streak >= self.cross_confirm_frames:
                        events.append((line_index, 1))
                        self._line_confirmed_side[state_key] = reference_side
                        self._line_pending_direction.pop(state_key, None)
                        self._line_side_streak.pop(state_key, None)
                else:
                    # 无穿越意图出现在 enter 侧：只记侧别，不计数（开场目标）
                    self._line_confirmed_side[state_key] = reference_side
                    self._line_side_streak.pop(state_key, None)
            elif on_exit:
                self._line_side_miss.pop(state_key, None)
                if confirmed == -reference_side:
                    self._line_pending_direction.pop(state_key, None)
                    self._line_side_streak.pop(state_key, None)
                    continue
                if pending == -1 or confirmed == reference_side:
                    streak = int(self._line_side_streak.get(state_key, 0) or 0) + 1
                    self._line_side_streak[state_key] = streak
                    if streak >= self.cross_confirm_frames:
                        events.append((line_index, -1))
                        self._line_confirmed_side[state_key] = -reference_side
                        self._line_pending_direction.pop(state_key, None)
                        self._line_side_streak.pop(state_key, None)
                else:
                    self._line_confirmed_side[state_key] = -reference_side
                    self._line_side_streak.pop(state_key, None)
            else:
                # 本帧未满足中心+角点：给少量容错，避免断 1 帧就清零
                if pending is not None or confirmed in (reference_side, -reference_side):
                    miss = int(self._line_side_miss.get(state_key, 0) or 0) + 1
                    self._line_side_miss[state_key] = miss
                    if miss > self.cross_confirm_miss_tolerance:
                        self._line_side_streak.pop(state_key, None)
                        self._line_side_miss.pop(state_key, None)
                else:
                    self._line_side_streak.pop(state_key, None)
                    self._line_side_miss.pop(state_key, None)
        return events

    def _process_track_crossing(
        self,
        *,
        track_id: Any,
        point: Tuple[float, float],
        detection: Dict[str, Any],
        count_segments: List[Dict[str, Any]],
        bypass_segments: List[Dict[str, Any]],
        is_predicted: bool = False,
    ) -> None:
        """处理单个 track 的过线/绕行（真实检测或短时预测均可）。"""
        prev_point = self._track_prev_center.get(track_id)
        if prev_point is not None and not is_predicted:
            self._track_velocity[track_id] = (
                float(point[0] - prev_point[0]),
                float(point[1] - prev_point[1]),
            )
        self._track_prev_center[track_id] = point
        if not is_predicted:
            self._track_lost_frames[track_id] = 0
            self._track_last_detection[track_id] = detection

        affinity = self._track_line_affinity.get(track_id)
        moved = (
            prev_point is not None
            and not (
                abs(prev_point[0] - point[0]) < _SEG_INTERSECT_EPS
                and abs(prev_point[1] - point[1]) < _SEG_INTERSECT_EPS
            )
        )

        if moved and prev_point is not None:
            count_hits = self._trajectory_hits_segments(
                prev_point, point, count_segments, "enter_side"
            )
            if count_hits:
                affinity = _LINE_KIND_COUNT
                for line_index, direction in count_hits:
                    self._line_pending_direction[
                        self._line_state_key(track_id, _LINE_KIND_COUNT, line_index)
                    ] = direction
                    # 新的穿线意图：重置 miss，保留/重建 streak 由确认逻辑处理
                    self._line_side_miss.pop(
                        self._line_state_key(track_id, _LINE_KIND_COUNT, line_index),
                        None,
                    )
                    self._line_side_streak.pop(
                        self._line_state_key(track_id, _LINE_KIND_COUNT, line_index),
                        None,
                    )
            else:
                # 丢检后轨迹可能跳过线段：用两侧翻转补 pending
                self._inject_pending_by_side_flip(
                    track_id,
                    _LINE_KIND_COUNT,
                    count_segments,
                    "enter_side",
                    prev_point,
                    point,
                )
                if any(
                    self._line_pending_direction.get(
                        self._line_state_key(track_id, _LINE_KIND_COUNT, i)
                    )
                    is not None
                    for i in range(len(count_segments))
                ):
                    affinity = _LINE_KIND_COUNT

            if affinity != _LINE_KIND_COUNT:
                bypass_hits = self._trajectory_hits_segments(
                    prev_point, point, bypass_segments, "bypass_side"
                )
                if bypass_hits:
                    affinity = _LINE_KIND_BYPASS
                    for line_index, direction in bypass_hits:
                        self._line_pending_direction[
                            self._line_state_key(
                                track_id, _LINE_KIND_BYPASS, line_index
                            )
                        ] = direction
                        self._line_side_miss.pop(
                            self._line_state_key(
                                track_id, _LINE_KIND_BYPASS, line_index
                            ),
                            None,
                        )
                        self._line_side_streak.pop(
                            self._line_state_key(
                                track_id, _LINE_KIND_BYPASS, line_index
                            ),
                            None,
                        )
                else:
                    self._inject_pending_by_side_flip(
                        track_id,
                        _LINE_KIND_BYPASS,
                        bypass_segments,
                        "bypass_side",
                        prev_point,
                        point,
                    )
                    if any(
                        self._line_pending_direction.get(
                            self._line_state_key(track_id, _LINE_KIND_BYPASS, i)
                        )
                        is not None
                        for i in range(len(bypass_segments))
                    ):
                        affinity = _LINE_KIND_BYPASS

        if affinity is not None:
            self._track_line_affinity[track_id] = affinity

        corners = self._get_bbox_corners(detection) or []
        if affinity == _LINE_KIND_COUNT:
            for line_index, direction in self._resolve_center_crossing(
                track_id,
                _LINE_KIND_COUNT,
                count_segments,
                "enter_side",
                point,
                corners,
                prev_center=prev_point,
            ):
                if direction > 0:
                    self.enter_count += 1
                    self._record_enter_event(
                        detection,
                        track_id=track_id,
                        line_index=line_index,
                        point=point,
                    )
                else:
                    self._record_line_event(
                        detection,
                        track_id=track_id,
                        violation_type="翻越出去",
                        line_kind=_LINE_KIND_COUNT,
                        line_index=line_index,
                        point=point,
                    )
        elif affinity == _LINE_KIND_BYPASS:
            for line_index, direction in self._resolve_center_crossing(
                track_id,
                _LINE_KIND_BYPASS,
                bypass_segments,
                "bypass_side",
                point,
                corners,
                prev_center=prev_point,
            ):
                violation_type = "绕行进去" if direction > 0 else "绕行出去"
                self._record_line_event(
                    detection,
                    track_id=track_id,
                    violation_type=violation_type,
                    line_kind=_LINE_KIND_BYPASS,
                    line_index=line_index,
                    point=point,
                )

    def _clear_track_state(self, track_id: Any) -> None:
        """清除单个 track 的过线相关状态。"""
        self._track_prev_center.pop(track_id, None)
        self._track_velocity.pop(track_id, None)
        self._track_lost_frames.pop(track_id, None)
        self._track_last_detection.pop(track_id, None)
        self._track_line_affinity.pop(track_id, None)
        stale_keys = [
            key for key in self._line_confirmed_side if key[0] == track_id
        ]
        for key in stale_keys:
            self._line_confirmed_side.pop(key, None)
            self._line_pending_direction.pop(key, None)
            self._line_side_streak.pop(key, None)
            self._line_side_miss.pop(key, None)
        pending_keys = [
            key for key in self._line_pending_direction if key[0] == track_id
        ]
        for key in pending_keys:
            self._line_pending_direction.pop(key, None)
            self._line_side_streak.pop(key, None)
            self._line_side_miss.pop(key, None)
        streak_keys = [
            key for key in self._line_side_streak if key[0] == track_id
        ]
        for key in streak_keys:
            self._line_side_streak.pop(key, None)
            self._line_side_miss.pop(key, None)
        miss_keys = [
            key for key in self._line_side_miss if key[0] == track_id
        ]
        for key in miss_keys:
            self._line_side_miss.pop(key, None)

    def _update_line_metrics(
        self,
        detections: List[Dict],
        count_line: Optional[Dict[str, Any]],
        bypass_line: Optional[Dict[str, Any]],
    ) -> None:
        """
        过线处理：
          1) 中心轨迹优先与 count_line 求交确定归属，命中则不再归属 bypass
          2) count_line：中心穿线后，中心在 enter 侧连续停留 N 帧 → enter_count+1；
             反向到对侧 → 不减计数，触发“翻越出去”告警
          3) bypass_line：不做计数；中心确认进入/离开分别告警
          4) 短时丢检：用速度预测中心点续轨迹，最多 track_lost_keep_frames 帧
        """
        self._latest_bypass_events = []
        self._latest_count_exit_events = []
        self._latest_enter_events = []
        if count_line is None and bypass_line is None:
            return

        if not self.enable_default_sort_tracking:
            if count_line is not None:
                self.log("warning", "已配置 count_line，但未开启跟踪，无法进行过线人数统计")
            if bypass_line is not None:
                self.log("warning", "已配置 bypass_line，但未开启跟踪，无法进行绕行违规检测")
            return

        active_ids: Set[Any] = set()
        count_segments = count_line.get("segments", []) if count_line else []
        bypass_segments = bypass_line.get("segments", []) if bypass_line else []

        for detection in detections:
            track_id = detection.get("track_id")
            if track_id is None:
                continue
            point = self._get_detection_point(detection)
            if point is None:
                continue
            active_ids.add(track_id)
            self._process_track_crossing(
                track_id=track_id,
                point=point,
                detection=detection,
                count_segments=count_segments,
                bypass_segments=bypass_segments,
                is_predicted=False,
            )

        # 短时丢检：预测中心续轨迹
        missing_ids = [
            tid for tid in list(self._track_prev_center.keys()) if tid not in active_ids
        ]
        for tid in missing_ids:
            lost = int(self._track_lost_frames.get(tid, 0) or 0) + 1
            if lost > self.track_lost_keep_frames:
                self._clear_track_state(tid)
                continue
            self._track_lost_frames[tid] = lost
            prev = self._track_prev_center.get(tid)
            if prev is None:
                self._clear_track_state(tid)
                continue
            vx, vy = self._track_velocity.get(tid, (0.0, 0.0))
            predicted = (float(prev[0] + vx), float(prev[1] + vy))
            last_det = dict(self._track_last_detection.get(tid) or {})
            last_det["track_id"] = tid
            # 按速度平移上次 bbox，便于后续画框/事件点一致
            bbox = last_det.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                last_det["bbox"] = [
                    float(bbox[0]) + vx,
                    float(bbox[1]) + vy,
                    float(bbox[2]) + vx,
                    float(bbox[3]) + vy,
                ]
            self._process_track_crossing(
                track_id=tid,
                point=predicted,
                detection=last_det,
                count_segments=count_segments,
                bypass_segments=bypass_segments,
                is_predicted=True,
            )

    def _get_alert_definition(
        self,
        alert_key: str,
        *,
        bypass_direction: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        按 key 取告警定义，并解析识别类型编码。

        - enter_count / count_exit：用 gate_direction（IN/OUT）选 codes
        - bypass：用 bypass_direction（enter/exit）选 codes
        """
        item = self._alert_def_by_key.get(alert_key) or {}
        codes = item.get("codes") if isinstance(item.get("codes"), dict) else {}

        if alert_key == "bypass":
            code_key = "enter" if bypass_direction == "enter" else "exit"
        else:
            code_key = self.gate_direction

        code_item = codes.get(code_key) if isinstance(codes.get(code_key), dict) else {}
        recognition_type = str(code_item.get("code") or "").strip()
        description = str(
            code_item.get("description") or item.get("description") or ""
        ).strip()

        return {
            "key": alert_key,
            "level": int(item.get("level", 0) or 0),
            "description": description,
            "recognition_type": recognition_type,
            "gate_direction": self.gate_direction,
            "code_key": code_key,
        }

    def _record_enter_event(
        self,
        detection: Dict,
        *,
        track_id: Any,
        line_index: int,
        point: Tuple[float, float],
    ) -> None:
        """记录 enter_count 增加事件，用于告警输出。"""
        alert_def = self._get_alert_definition("enter_count")
        detection["enter_count_changed"] = True
        detection["violation_type"] = "进入"
        detection["alert_level"] = alert_def["level"]
        detection["alert_description"] = alert_def["description"]
        detection["recognition_type"] = alert_def["recognition_type"]
        self._latest_enter_events.append(
            {
                "track_id": track_id,
                "violation_type": "进入",
                "line_kind": _LINE_KIND_COUNT,
                "line_index": line_index,
                "point": {"x": point[0], "y": point[1]},
                "enter_count": int(self.enter_count),
                "alert_key": "enter_count",
                "alert_level": alert_def["level"],
                "alert_description": alert_def["description"],
                "recognition_type": alert_def["recognition_type"],
                "gate_direction": self.gate_direction,
            }
        )

    def _record_line_event(
        self,
        detection: Dict,
        *,
        track_id: Any,
        violation_type: str,
        line_kind: str,
        line_index: int,
        point: Tuple[float, float],
    ) -> None:
        if line_kind == _LINE_KIND_BYPASS:
            bypass_direction = "enter" if violation_type == "绕行进去" else "exit"
            alert_key = "bypass"
            alert_def = self._get_alert_definition(
                alert_key, bypass_direction=bypass_direction
            )
        else:
            alert_key = "count_exit"
            alert_def = self._get_alert_definition(alert_key)

        detection["violation_type"] = violation_type
        detection["alert_level"] = alert_def["level"]
        detection["alert_description"] = alert_def["description"]
        detection["recognition_type"] = alert_def["recognition_type"]
        event = {
            "track_id": track_id,
            "violation_type": violation_type,
            "line_kind": line_kind,
            "line_index": line_index,
            "point": {"x": point[0], "y": point[1]},
            "alert_key": alert_key,
            "alert_level": alert_def["level"],
            "alert_description": alert_def["description"],
            "recognition_type": alert_def["recognition_type"],
            "gate_direction": self.gate_direction,
        }
        if line_kind == _LINE_KIND_BYPASS:
            detection["bypass_violation"] = True
            self._latest_bypass_events.append(event)
        else:
            detection["count_exit_violation"] = True
            self._latest_count_exit_events.append(event)

    def reset_enter_count(self) -> None:
        """重置过线进入人数计数到初始值。"""
        self.enter_count = self.enter_count_initial
        self._track_prev_center.clear()
        self._track_velocity.clear()
        self._track_lost_frames.clear()
        self._track_last_detection.clear()
        self._track_line_affinity.clear()
        self._line_confirmed_side.clear()
        self._line_pending_direction.clear()
        self._line_side_streak.clear()
        self._line_side_miss.clear()
        self._latest_bypass_events = []
        self._latest_count_exit_events = []
        self._latest_enter_events = []

    def reset_violation_events(self) -> None:
        """清空违规告警缓存，并重置过线状态。"""
        self._latest_bypass_events = []
        self._latest_count_exit_events = []
        self._latest_enter_events = []
        self._track_prev_center.clear()
        self._track_velocity.clear()
        self._track_lost_frames.clear()
        self._track_last_detection.clear()
        self._track_line_affinity.clear()
        self._line_confirmed_side.clear()
        self._line_pending_direction.clear()
        self._line_side_streak.clear()
        self._line_side_miss.clear()

    def _load_image(
        self,
        input_data: Union[np.ndarray, str, Dict[str, Any], Any]
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[str]]:
        image = None
        fence_config = None

        if isinstance(input_data, np.ndarray):
            image = input_data.copy()
        elif isinstance(input_data, str):
            image = cv2.imread(input_data)
            if image is None:
                return None, None, f"无法加载图像: {input_data}"
        elif isinstance(input_data, dict):
            if "image" not in input_data:
                return None, None, "输入字典中缺少 'image' 字段"

            image_data = input_data["image"]
            if isinstance(image_data, np.ndarray):
                image = image_data.copy()
            elif isinstance(image_data, str):
                image = cv2.imread(image_data)
                if image is None:
                    return None, None, f"无法加载图像: {image_data}"
            else:
                return None, None, "不支持的图像数据类型"

            fence_config = input_data.get("fence_config")
        else:
            return None, None, "不支持的输入数据类型"

        if image is None or image.size == 0:
            return None, None, "无效的图像数据"

        return image, fence_config, None

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, axis=0)

    def _is_end2end_output(self, predictions: np.ndarray) -> bool:
        """YOLO26 end2end: (max_det, 6)；原始模型: (4+nc, num_anchors)。"""
        if predictions.ndim == 3:
            _, dim1, dim2 = predictions.shape
            return dim1 > dim2
        if predictions.ndim == 2:
            return predictions.shape[0] > predictions.shape[1]
        return False

    def _clip_box(self, left: int, top: int, width_box: int, height_box: int,
                  img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        left = max(0, left)
        top = max(0, top)
        width_box = min(width_box, img_width - left)
        height_box = min(height_box, img_height - top)
        return left, top, width_box, height_box

    def _build_detection(self, left: int, top: int, width_box: int, height_box: int,
                         score: float, class_id: int) -> Dict[str, Any]:
        return {
            "bbox": [left, top, left + width_box, top + height_box],
            "confidence": float(score),
            "class_id": int(class_id),
            "class_name": self.class_names.get(class_id, "person"),
            "target_type": "person",
        }

    def _postprocess_end2end(
        self,
        predictions: np.ndarray,
        img_width: int,
        img_height: int,
    ) -> List[Dict[str, Any]]:
        """
        YOLO26 end2end 输出: (batch, max_det, 6) -> [x1, y1, x2, y2, conf, class_id]
        已内置 NMS，仅需置信度过滤并保留目标类别。
        """
        x_factor = img_width / self.input_width
        y_factor = img_height / self.input_height

        if predictions.ndim == 3:
            predictions = predictions[0]

        results: List[Dict[str, Any]] = []
        for row in predictions:
            conf = float(row[4])
            if conf < self.conf_thres:
                continue

            class_id = int(row[5])
            if class_id not in self.target_class_ids:
                continue

            x1 = int(float(row[0]) * x_factor)
            y1 = int(float(row[1]) * y_factor)
            x2 = int(float(row[2]) * x_factor)
            y2 = int(float(row[3]) * y_factor)
            left, top, width_box, height_box = self._clip_box(
                x1, y1, max(0, x2 - x1), max(0, y2 - y1), img_width, img_height
            )
            results.append(self._build_detection(left, top, width_box, height_box, conf, class_id))

            if len(results) >= self.max_det:
                break

        return results

    def _postprocess_grid(
        self,
        predictions: np.ndarray,
        img_width: int,
        img_height: int,
    ) -> List[Dict[str, Any]]:
        """
        YOLO26 原始模型输出: (batch, 4+nc, num_anchors)，通道优先布局。
        参考 ultralytics PostprocessDetect，对 xywh 框执行 NMS 后映射回原图。
        """
        if predictions.ndim == 3:
            predictions = predictions[0]

        num_channels, num_anchors = predictions.shape
        num_classes = num_channels - 4
        if num_classes <= 0:
            return []

        x_factor = img_width / self.input_width
        y_factor = img_height / self.input_height

        boxes: List[List[int]] = []
        scores: List[float] = []
        class_ids: List[int] = []

        for anchor_idx in range(num_anchors):
            class_scores = predictions[4:, anchor_idx]
            class_id = int(np.argmax(class_scores))
            max_score = float(class_scores[class_id])
            if max_score < self.conf_thres:
                continue
            if class_id not in self.target_class_ids:
                continue

            cx = float(predictions[0, anchor_idx])
            cy = float(predictions[1, anchor_idx])
            box_w = float(predictions[2, anchor_idx])
            box_h = float(predictions[3, anchor_idx])

            left = int((cx - 0.5 * box_w) * x_factor)
            top = int((cy - 0.5 * box_h) * y_factor)
            width_box = int(box_w * x_factor)
            height_box = int(box_h * y_factor)
            left, top, width_box, height_box = self._clip_box(
                left, top, width_box, height_box, img_width, img_height
            )

            boxes.append([left, top, width_box, height_box])
            scores.append(max_score)
            class_ids.append(class_id)

        if not boxes:
            return []

        nms_indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thres, self.iou_thres)
        if len(nms_indices) == 0:
            return []

        if isinstance(nms_indices, np.ndarray):
            nms_indices = nms_indices.flatten()

        results: List[Dict[str, Any]] = []
        for idx in nms_indices:
            if isinstance(idx, (list, tuple, np.ndarray)):
                idx = idx[0]
            idx = int(idx)
            box = boxes[idx]
            results.append(
                self._build_detection(box[0], box[1], box[2], box[3], scores[idx], class_ids[idx])
            )
            if len(results) >= self.max_det:
                break

        return results

    def postprocess(self, outputs: Dict[str, np.ndarray], original_img: np.ndarray) -> List[Dict[str, Any]]:
        img_height, img_width = original_img.shape[:2]
        predictions = outputs.get("output0")
        if predictions is None:
            return []

        predictions = np.asarray(predictions)
        if predictions.size == 0:
            return []

        if self._is_end2end_output(predictions):
            return self._postprocess_end2end(predictions, img_width, img_height)
        return self._postprocess_grid(predictions, img_width, img_height)


    def process(
        self,
        input_data: Union[np.ndarray, str, Dict[str, Any], Any],
        fence_config: Dict = None
    ) -> SkillResult:
        try:
            image, input_fence_config, error_message = self._load_image(input_data)
            if error_message:
                return SkillResult.error_result(error_message)

            if input_fence_config is not None:
                fence_config = input_fence_config

            height, width = image.shape[:2]

            # 先解析并激活线段（推理失败时仍可画线到推流画面）
            count_line_raw = self._resolve_count_line_config(input_data, fence_config)
            if not self._is_line_config_present(count_line_raw):
                return SkillResult.error_result(
                    "count_line 不能为空，请上传过线计数线配置"
                )
            bypass_line_raw = self._resolve_bypass_line_config(input_data, fence_config)
            parsed_count_line = self._normalize_count_line(count_line_raw, (width, height))
            if parsed_count_line is None:
                return SkillResult.error_result(
                    "count_line 无效：未解析到有效线段与 enter_point"
                )
            parsed_bypass_line = (
                self._normalize_bypass_line(bypass_line_raw, (width, height))
                if bypass_line_raw
                else None
            )
            self._active_count_line = parsed_count_line
            self._active_bypass_line = parsed_bypass_line

            ready, err_msg = self.ensure_models_ready()
            if not ready:
                return SkillResult.error_result(
                    f"模型未就绪: {err_msg}",
                    data={
                        "detections": [],
                        "count": 0,
                        "enter_count": int(self.enter_count),
                        "gate_direction": self.gate_direction,
                    },
                )

            timing: Dict[str, float] = {}

            t0 = self._timing_start()
            input_tensor = self.preprocess(image)
            timing["preprocess_ms"] = self._timing_elapsed_ms(t0)

            t0 = self._timing_start()
            outputs = triton_client.infer(
                self.model_name,
                {"images": input_tensor},
                model_version=self.model_version,
            )
            timing["infer_ms"] = self._timing_elapsed_ms(t0)
            if outputs is None:
                return SkillResult.error_result(
                    "人员检测推理失败",
                    data={
                        "detections": [],
                        "count": 0,
                        "enter_count": int(self.enter_count),
                        "gate_direction": self.gate_direction,
                    },
                )

            t0 = self._timing_start()
            detections = self.postprocess(outputs, image)
            timing["postprocess_ms"] = self._timing_elapsed_ms(t0)

            if self.enable_default_sort_tracking:
                t0 = self._timing_start()
                detections = self.add_tracking_ids(detections, image=image)
                timing["track_ms"] = self._timing_elapsed_ms(t0)

            # 过线进出人数 / 绕行违规：在围栏过滤前统计，避免刚越线的目标被滤掉漏计
            self._update_line_metrics(detections, parsed_count_line, parsed_bypass_line)

            if self.is_fence_config_valid(fence_config):
                detections = self.filter_detections_by_fence(detections, fence_config, (width, height))

            bypass_events = list(self._latest_bypass_events)
            count_exit_events = list(self._latest_count_exit_events)
            enter_events = list(self._latest_enter_events)
            active_alerts: List[Dict[str, Any]] = []

            def _append_active_alert(
                events: List[Dict[str, Any]],
                *,
                default_violation_type: str,
            ) -> None:
                if not events:
                    return
                # 按识别类型编码分组，避免入/出混在同一条 active_alert
                grouped: Dict[str, List[Dict[str, Any]]] = {}
                for event in events:
                    code = str(event.get("recognition_type") or "")
                    grouped.setdefault(code, []).append(event)
                for code, group in grouped.items():
                    first = group[0]
                    active_alerts.append(
                        {
                            "key": first.get("alert_key"),
                            "level": first.get("alert_level"),
                            "description": first.get("alert_description"),
                            "recognition_type": code,
                            "gate_direction": self.gate_direction,
                            "violation_type": first.get(
                                "violation_type", default_violation_type
                            ),
                            "events": group,
                        }
                    )

            _append_active_alert(enter_events, default_violation_type="进入")
            _append_active_alert(count_exit_events, default_violation_type="翻越出去")
            _append_active_alert(bypass_events, default_violation_type="绕行")

            recognition_types = sorted(
                {
                    str(e.get("recognition_type"))
                    for e in (enter_events + count_exit_events + bypass_events)
                    if e.get("recognition_type")
                }
            )
            result_data = {
                "detections": detections,
                "count": len(detections),
                "enter_count": int(self.enter_count),
                "gate_direction": self.gate_direction,
                "mine_code": self.mine_code,
                "camera_code": self.camera_code,
                "skill_name": self.config.get("name") or "person_count_detector26",
                "enter_events": enter_events,
                "bypass_events": bypass_events,
                "count_exit_events": count_exit_events,
                "has_enter_count_change": bool(enter_events),
                "has_bypass_violation": bool(bypass_events),
                "has_count_exit_violation": bool(count_exit_events),
                "recognition_types": recognition_types,
                "alert_definitions": list(self.alert_definitions),
                "active_alerts": active_alerts,
                "flow_metrics": {
                    "current_person_count": len(detections),
                    "enter_count": int(self.enter_count),
                    "gate_direction": self.gate_direction,
                },
                "violations": {
                    "enter": enter_events,
                    "count_exit": count_exit_events,
                    "bypass": bypass_events,
                },
            }
            self._attach_process_timing(result_data, timing)

            return SkillResult.success_result(result_data)

        except Exception as e:
            logger.exception(f"人数检测技能处理失败: {str(e)}")
            return SkillResult.error_result(f"处理失败: {str(e)}")




    def _draw_bypass_line(self, frame: np.ndarray) -> None:
        """绘制非计数绕行线与各段对应参考点。"""
        bypass_line = self._active_bypass_line
        if not bypass_line:
            return

        for segment in bypass_line.get("segments", []):
            a = segment["line_a"]
            b = segment["line_b"]
            bypass_pt = segment["bypass_point"]
            pt1 = (int(round(a[0])), int(round(a[1])))
            pt2 = (int(round(b[0])), int(round(b[1])))
            bp = (int(round(bypass_pt[0])), int(round(bypass_pt[1])))

            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            cv2.circle(frame, pt1, 4, (0, 0, 255), -1)
            cv2.circle(frame, pt2, 4, (0, 0, 255), -1)
            cv2.circle(frame, bp, 6, (0, 0, 255), -1)
            cv2.putText(
                frame, "bypass", (bp[0] + 8, bp[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )

    def _draw_count_line(self, frame: np.ndarray, enter_count: Optional[int] = None) -> None:
        """绘制计数线段、各段对应进入侧参考点与 enter_count。"""
        count_line = self._active_count_line
        if not count_line:
            return

        label = self.gate_direction  # IN / OUT
        for segment in count_line.get("segments", []):
            a = segment["line_a"]
            b = segment["line_b"]
            enter_pt = segment["enter_point"]
            pt1 = (int(round(a[0])), int(round(a[1])))
            pt2 = (int(round(b[0])), int(round(b[1])))
            ep = (int(round(enter_pt[0])), int(round(enter_pt[1])))

            cv2.line(frame, pt1, pt2, (0, 165, 255), 2)
            cv2.circle(frame, pt1, 4, (0, 165, 255), -1)
            cv2.circle(frame, pt2, 4, (0, 165, 255), -1)
            cv2.circle(frame, ep, 6, (0, 165, 255), -1)
            cv2.putText(
                frame, label, (ep[0] + 8, ep[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2
            )

        value = self.enter_count if enter_count is None else int(enter_count)
        cv2.putText(
            frame, f"{label}_count: {value}", (40, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
        )

    def _draw_detections_on_frame(self, frame: np.ndarray, alert_data: Dict) -> np.ndarray:
        """在帧上绘制检测框、跟踪轨迹与过线计数线"""
        try:
            detections = alert_data.get("detections", [])
            colors = [
                (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                (128, 0, 128), (255, 165, 0), (0, 128, 255), (128, 128, 128), (0, 0, 255),
            ]

            class_color_map: Dict[str, Tuple[int, int, int]] = {}
            track_color_map: Dict[Any, Tuple[int, int, int]] = {}
            color_index = 0

            for detection in detections:
                class_name = detection.get("class_name", "unknown")
                track_id = detection.get("track_id")

                if class_name not in class_color_map:
                    class_color_map[class_name] = colors[color_index % len(colors)]
                    color_index += 1

                if track_id is not None and track_id not in track_color_map:
                    track_color_map[track_id] = class_color_map[class_name]

            self.draw_track_trajectories(frame, detections, track_color_map=track_color_map)

            enter_count = alert_data.get("enter_count", self.enter_count)
            self._draw_count_line(frame, enter_count=enter_count)
            self._draw_bypass_line(frame)

            for detection in detections:
                bbox = detection.get("bbox", [])
                confidence = detection.get("confidence", 0.0)
                class_name = detection.get("class_name", "unknown")
                track_id = detection.get("track_id")
                # 04/05/06/07：绕行或闸机反向翻越 → 红框
                is_violation_alert = bool(
                    detection.get("bypass_violation")
                    or detection.get("count_exit_violation")
                )
                # 01/02：正常过线计数 → 黄框
                is_count_alert = bool(detection.get("enter_count_changed"))

                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox

                    color = class_color_map[class_name]
                    if is_violation_alert:
                        color = _ALERT_BOX_COLOR_VIOLATION
                    elif is_count_alert:
                        color = _ALERT_BOX_COLOR_COUNT
                    elif track_id is not None and track_id in track_color_map:
                        color = track_color_map[track_id]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    # 告警框与普通框均显示类别 + 置信度（如 person: 0.95）
                    label = f"{class_name}: {confidence:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )

                    cv2.rectangle(
                        frame, (int(x1), int(y1) - text_height - baseline - 5),
                        (int(x1) + text_width, int(y1)), color, -1
                    )
                    cv2.putText(
                        frame, label, (int(x1), int(y1) - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                    )

            return frame
        except Exception as e:
            logger.error(f"绘制检测框时出错: {str(e)}")
            return frame






if __name__ == "__main__":
    from copy import deepcopy

    # 本地试跑需显式提供 count_line（默认已改为必填空配置）
    demo_config = deepcopy(PersonCountDetector26Skill.DEFAULT_CONFIG)
    demo_config["params"]["count_line"] = {
        "lines": [
            {
                "line": [{"x": 432, "y": 328}, {"x": 723, "y": 229}],
                "enter_point": {"x": 513, "y": 250},
            },
        ],
    }
    skill = PersonCountDetector26Skill(demo_config)
    img_name = str(_CODE_ROOT / "test_images_videos" / "9-74-0004-000880.jpg")
    frame = cv2.imread(img_name)
    if frame is None:
        raise FileNotFoundError(f"无法加载测试图片: {img_name}")
    result = skill.process(frame)
    print(result.to_dict())

    ploted_img = skill._draw_detections_on_frame(frame.copy(), result.to_dict()["data"])
    out_path = str(Path(img_name).with_name(Path(img_name).stem + "_ploted.jpg"))
    cv2.imwrite(out_path, ploted_img)
    print(f"结果已保存: {out_path}")