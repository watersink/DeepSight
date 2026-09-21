"""
非固定停车上下车人数统计技能 (YOLO26)

在 boarding_detector 基础上：
  - 车辆停靠位置不固定时，仍根据检测到的人车自动估计车厢内区域（或可选手动框/电子围栏）
  - 按 track 脚点内外状态迁移累计计数：外→内为上车，内→外为下车
  - gate_direction=IN 仅统计上车；gate_direction=OUT 仅统计下车
  - 累计值 enter_count，识别类型编码与闸机计数一致（01 入 / 02 出）
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_CODE_ROOT = Path(__file__).resolve().parents[3]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cv2
import numpy as np

from app.plugins.skills.boarding_detector_skill import (
    BoardingDetectorSkill,
    PERSON_CLASS_NAMES,
)
import logging

logger = logging.getLogger(__name__)


class NonFixedParkingBoardingDetectorSkill(BoardingDetectorSkill):
    """
    非固定停车上下车人数统计 (YOLO26)

    依赖 yolo26_person 检测人员与人车；车厢内区域随当前帧人车框变化。
    """

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": "non_fixed_parking_boarding_detector",
        "name_zh": "非固定停车人数统计(YOLO26)",
        "version": "1.0",
        "cover_image": "/skills/non_fixed_parking_boarding_detector.png",
        "description": (
            "适用于巷道口固定相机、车辆驶入后尾对镜头停稳、人员从后门上下车的场景；"
            "车辆停稳后才累计计数；IN=上车，OUT=下车"
        ),
        "status": True,
        "required_models": ["yolo26_person"],
        "form_fields": [
            {
                "key": "gate_direction",
                "label": "统计方向",
                "type": "select",
                "required": True,
                "default": "IN",
                "options": [
                    {"value": "IN", "label": "IN（统计上车）"},
                    {"value": "OUT", "label": "OUT（统计下车）"},
                ],
            },
            {
                "key": "enter_count",
                "label": "计数初始值",
                "type": "number",
                "required": False,
                "default": 0,
            },
        ],
        "params": {
            **BoardingDetectorSkill.DEFAULT_CONFIG["params"],
            "conf_thres": 0.45,
            "vehicle_conf_thres": 0.22,
            "enable_helmet_filter": False,
            "roi_hold": 8,
            "enter_count": 0,
            "gate_direction": "IN",
            "zone_confirm_frames": 2,
            # 车框中心连续 N 帧几乎不动才认为停稳（20fps 下 10≈0.5s）
            "vehicle_stable_frames": 10,
            "vehicle_center_shift_ratio": 0.012,
            # 尾对停稳时，登车区向下延伸到车框外地面（后门台阶/等候区）
            "boarding_ground_extend_ratio": 0.55,
            "track_lost_keep_frames": 15,
            "person_min_height_ratio": 0.06,
            "mine_code": "123456789012",
            "camera_code": "12345678901234567890",
            # 推理仍检测人车以估计区域，推流画面只画 person
            "draw_vehicle_detections": False,
        },
        "alert_definitions": [
            {
                "key": "enter_count",
                "level": 0,
                "description": "人员脚点进入车厢内区域，累计上车人数 +1（gate_direction=IN）。",
                "codes": {
                    "IN": {"code": "01", "description": "非固定停车上车计数"},
                    "OUT": {"code": "02", "description": "非固定停车下车计数"},
                },
            },
        ],
    }

    def _initialize(self) -> None:
        super()._initialize()
        params = self.config.get("params", {})

        self.enter_count_initial: int = int(params.get("enter_count", 0) or 0)
        self.enter_count: int = self.enter_count_initial
        self.gate_direction: str = self._normalize_gate_direction(
            params.get("gate_direction", "IN")
        )
        self.zone_confirm_frames: int = max(
            1, int(params.get("zone_confirm_frames", 2) or 2)
        )
        self.vehicle_stable_frames: int = max(
            1, int(params.get("vehicle_stable_frames", 10) or 10)
        )
        self.vehicle_center_shift_ratio: float = float(
            params.get("vehicle_center_shift_ratio", 0.012) or 0.012
        )
        self.boarding_ground_extend_ratio: float = float(
            params.get("boarding_ground_extend_ratio", 0.55) or 0.55
        )
        self.track_lost_keep_frames: int = max(
            0, int(params.get("track_lost_keep_frames", 15) or 15)
        )
        self.person_min_height_ratio: float = float(
            params.get("person_min_height_ratio", 0.06) or 0.06
        )
        self.draw_vehicle_detections: bool = bool(
            params.get("draw_vehicle_detections", False)
        )
        self.mine_code: str = str(params.get("mine_code") or "").strip()
        self.camera_code: str = str(params.get("camera_code") or "").strip()

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

        self._latest_enter_events: List[Dict[str, Any]] = []
        self._track_zone_confirmed: Dict[Any, bool] = {}
        self._track_pending_inside: Dict[Any, bool] = {}
        self._track_zone_streak: Dict[Any, int] = {}
        self._track_zone_lost: Dict[Any, int] = {}
        self._vehicle_stable_streak: int = 0
        self._last_vehicle_center: Optional[Tuple[float, float]] = None
        self._vehicle_parked: bool = False
        self._last_good_vehicle_bbox: Optional[Tuple[int, int, int, int]] = None
        self.vehicle_conf_thres: float = float(
            params.get("vehicle_conf_thres", 0.22) or 0.22
        )

        self.log(
            "info",
            f"非固定停车上下车统计: gate_direction={self.gate_direction}, "
            f"enter_count_initial={self.enter_count_initial}, "
            f"zone_confirm_frames={self.zone_confirm_frames}",
        )

    @staticmethod
    def _normalize_gate_direction(raw: Any) -> str:
        value = str(raw or "IN").strip().upper()
        return value if value in {"IN", "OUT"} else "IN"

    def _get_alert_definition(self, alert_key: str) -> Dict[str, Any]:
        item = self._alert_def_by_key.get(alert_key) or {}
        codes = item.get("codes") if isinstance(item.get("codes"), dict) else {}
        code_item = (
            codes.get(self.gate_direction)
            if isinstance(codes.get(self.gate_direction), dict)
            else {}
        )
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
        }

    def _clear_track_zone_state(self, track_id: Any) -> None:
        self._track_zone_confirmed.pop(track_id, None)
        self._track_pending_inside.pop(track_id, None)
        self._track_zone_streak.pop(track_id, None)
        self._track_zone_lost.pop(track_id, None)

    @staticmethod
    def _union_boxes(
        box_a: Optional[Tuple[int, int, int, int]],
        box_b: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tuple[int, int, int, int]]:
        """合并两个 AABB，用于「车厢内 + 车门」统一计数区。"""
        boxes = [b for b in (box_a, box_b) if b is not None and len(b) >= 4]
        if not boxes:
            return None
        x1 = min(int(b[0]) for b in boxes)
        y1 = min(int(b[1]) for b in boxes)
        x2 = max(int(b[2]) for b in boxes)
        y2 = max(int(b[3]) for b in boxes)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _rear_door_threshold_box(
        self,
        vehicle_box: Tuple[int, int, int, int],
        frame_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:
        """尾对镜头：车底门槛带（踩台阶/进门槛），不含远处地面等候区。"""
        w, h = frame_size
        x1, y1, x2, y2 = vehicle_box
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        pad = self.boarding_ground_extend_ratio
        return self._clamp_box(
            (
                x1 + 0.06 * bw,
                y2 - int(0.16 * bh),
                x2 - 0.06 * bw,
                min(h - 1, y2 + int(max(0.08, pad * 0.22) * bh)),
            ),
            w,
            h,
        )

    def _foot_in_boarding_zone(
        self,
        foot: Optional[Tuple[float, float]],
        *,
        count_box: Optional[Tuple[int, int, int, int]],
        vehicle_box: Optional[Tuple[int, int, int, int]],
        inside_box: Optional[Tuple[int, int, int, int]],
        door_box: Optional[Tuple[int, int, int, int]],
        frame_size: Tuple[int, int],
        use_rear_threshold: bool,
    ) -> bool:
        if foot is None:
            return False
        cabin = self._union_boxes(inside_box, door_box) or count_box
        if cabin and self._point_in_box(foot, cabin):
            return True
        if use_rear_threshold and vehicle_box and len(vehicle_box) >= 4:
            return self._point_in_box(
                foot, self._rear_door_threshold_box(vehicle_box, frame_size)
            )
        return self._point_in_box(foot, count_box)

    def _boarding_visual_box(
        self,
        count_box: Optional[Tuple[int, int, int, int]],
        vehicle_box: Optional[Tuple[int, int, int, int]],
        frame_size: Tuple[int, int],
        use_rear_threshold: bool,
    ) -> Optional[Tuple[int, int, int, int]]:
        box = count_box
        if use_rear_threshold and vehicle_box:
            box = self._union_boxes(
                box, self._rear_door_threshold_box(vehicle_box, frame_size)
            )
        return box

    @staticmethod
    def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = box
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)

    def _vehicle_score(self, frame: np.ndarray, det: Dict[str, Any]) -> float:
        """巷道口尾对白车：放宽面积/白色占比，避免 pick 失败导致 zone:none。"""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.get("bbox") or [0, 0, 0, 0]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        area_ratio = (bw * bh) / float(w * h)
        if area_ratio < 0.025 or area_ratio > 0.72:
            return -1.0
        cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
        loc = 1.0 - (abs(cx - 0.52) + abs(cy - 0.50))
        if x1 < 0.03 * w:
            loc -= 0.35
        wr = self._white_ratio(frame, det["bbox"])
        if wr < 0.05:
            return -1.0
        conf = float(det.get("confidence", 0.0))
        return 0.25 * conf + 0.50 * wr + 0.25 * loc

    def _pick_vehicle(
        self, vehicles: List[Dict], frame: np.ndarray
    ) -> Optional[Dict[str, Any]]:
        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for det in vehicles:
            if float(det.get("confidence", 0.0)) < self.vehicle_conf_thres:
                continue
            score = self._vehicle_score(frame, det)
            if score > 0:
                ranked.append((score, det))
        if ranked:
            ranked.sort(key=lambda x: x[0], reverse=True)
            picked = ranked[0][1]
            self._last_good_vehicle_bbox = tuple(
                int(v) for v in picked["bbox"][:4]
            )
            return picked
        if self._last_good_vehicle_bbox and (
            self._vehicle_parked or self._vehicle_stable_streak >= 3
        ):
            return {
                "bbox": list(self._last_good_vehicle_bbox),
                "confidence": 0.01,
                "class_name": "truck",
                "class_id": 7,
                "target_type": "vehicle",
            }
        return None

    def postprocess(self, outputs: Dict[str, np.ndarray], original_img: np.ndarray):
        """人车分开阈值：车类略降置信度，减少「有人无 zone」。"""
        results = super().postprocess(outputs, original_img)
        th = self.vehicle_conf_thres
        filtered: List[Dict[str, Any]] = []
        for det in results:
            name = det.get("class_name")
            if name in PERSON_CLASS_NAMES:
                if float(det.get("confidence", 0.0)) >= self.conf_thres:
                    filtered.append(det)
            else:
                if float(det.get("confidence", 0.0)) >= th:
                    filtered.append(det)
        return filtered

    def _update_vehicle_parked(
        self,
        vehicle_box: Optional[Tuple[int, int, int, int]],
        frame_size: Tuple[int, int],
    ) -> bool:
        """车框中心连续多帧位移很小 → 认为已停稳（适合 test.mp4：侧向驶入后再尾对停住）。"""
        if vehicle_box is None:
            self._vehicle_stable_streak = 0
            self._last_vehicle_center = None
            self._vehicle_parked = False
            return False
        width, _height = frame_size
        cx, cy = self._box_center(vehicle_box)
        max_shift = max(4.0, float(width) * self.vehicle_center_shift_ratio)
        if self._last_vehicle_center is not None:
            lx, ly = self._last_vehicle_center
            if abs(cx - lx) <= max_shift and abs(cy - ly) <= max_shift:
                self._vehicle_stable_streak += 1
            else:
                self._vehicle_stable_streak = 0
        self._last_vehicle_center = (cx, cy)
        self._vehicle_parked = self._vehicle_stable_streak >= self.vehicle_stable_frames
        return self._vehicle_parked

    def _estimate_zones_from_vehicle(self, vehicle, frame_shape, door_expand: float):
        """
        尾对镜头（宽>高）：用后视切分（后门+厢内）。
        侧向驶入（高>=宽）：单一临时区，且停稳前不做累计。
        """
        h, w = frame_shape
        x1, y1, x2, y2 = vehicle
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        if bw / float(bh) >= 1.08:
            return super()._estimate_zones_from_vehicle(vehicle, frame_shape, door_expand)
        if bh / float(bw) >= 1.05:
            boarding = self._clamp_box(
                (
                    x1 + 0.10 * bw,
                    y1 + 0.20 * bh,
                    x2 - 0.08 * bw,
                    min(h - 1, y2 + int(door_expand * bh * 0.35)),
                ),
                w,
                h,
            )
            return boarding, boarding
        return super()._estimate_zones_from_vehicle(vehicle, frame_shape, door_expand)

    def _filter_person_detections(
        self,
        persons: List[Dict[str, Any]],
        *,
        frame_height: int,
        vehicle_box: Optional[Tuple[int, int, int, int]],
        door_box: Optional[Tuple[int, int, int, int]],
    ) -> List[Dict[str, Any]]:
        """去掉车身误检等小框，避免无关 person 框。"""
        min_h = max(8.0, float(frame_height) * self.person_min_height_ratio)
        door_top = int(door_box[1]) if door_box and len(door_box) >= 4 else None
        kept: List[Dict[str, Any]] = []
        for person in persons:
            bbox = person.get("bbox") or []
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox
            bh = float(y2 - y1)
            if bh < min_h:
                continue
            foot = self._get_detection_point(person)
            if (
                vehicle_box
                and door_top is not None
                and foot is not None
                and self._point_in_box(foot, vehicle_box)
                and float(foot[1]) < float(door_top)
                and bh < min_h * 1.35
            ):
                # 脚点仍在车身上部（如 R5 侧板），多为误检
                continue
            kept.append(person)
        return kept

    def _record_enter_event(
        self,
        detection: Dict[str, Any],
        *,
        track_id: Any,
        transition: str,
        point: Tuple[float, float],
    ) -> None:
        alert_def = self._get_alert_definition("enter_count")
        detection["enter_count_changed"] = True
        detection["violation_type"] = transition
        detection["alert_level"] = alert_def["level"]
        detection["alert_description"] = alert_def["description"]
        detection["recognition_type"] = alert_def["recognition_type"]
        self._latest_enter_events.append(
            {
                "track_id": track_id,
                "violation_type": transition,
                "transition": transition,
                "point": {"x": point[0], "y": point[1]},
                "enter_count": int(self.enter_count),
                "alert_key": "enter_count",
                "alert_level": alert_def["level"],
                "alert_description": alert_def["description"],
                "recognition_type": alert_def["recognition_type"],
                "gate_direction": self.gate_direction,
            }
        )

    def _update_zone_transitions(
        self,
        persons: List[Dict[str, Any]],
        *,
        count_box: Optional[Tuple[int, int, int, int]],
        vehicle_box: Optional[Tuple[int, int, int, int]],
        inside_box: Optional[Tuple[int, int, int, int]],
        door_box: Optional[Tuple[int, int, int, int]],
        frame_size: Tuple[int, int],
        use_rear_threshold: bool,
    ) -> None:
        """脚点内外状态稳定迁移后，按 gate_direction 累计 enter_count。"""
        self._latest_enter_events = []
        if count_box is None and not (use_rear_threshold and vehicle_box):
            return
        if not self.enable_default_sort_tracking:
            self.log(
                "warning",
                "未开启目标跟踪，无法进行非固定停车上下车累计计数",
            )
            return

        active_ids: Set[Any] = set()
        for person in persons:
            track_id = person.get("track_id")
            if track_id is None:
                continue
            foot = self._get_detection_point(person)
            if foot is None:
                continue
            active_ids.add(track_id)
            self._track_zone_lost[track_id] = 0
            inside = self._foot_in_boarding_zone(
                foot,
                count_box=count_box,
                vehicle_box=vehicle_box,
                inside_box=inside_box,
                door_box=door_box,
                frame_size=frame_size,
                use_rear_threshold=use_rear_threshold,
            )
            confirmed = self._track_zone_confirmed.get(track_id)

            if confirmed is None:
                # 首次出现一律从「车外」计状态，避免已在门口/门内时永远不产生外→内迁移
                self._track_zone_confirmed[track_id] = False
                self._track_pending_inside.pop(track_id, None)
                self._track_zone_streak.pop(track_id, None)
                if not inside:
                    continue

            if inside == confirmed:
                self._track_pending_inside.pop(track_id, None)
                self._track_zone_streak.pop(track_id, None)
                continue

            pending = self._track_pending_inside.get(track_id)
            if pending != inside:
                self._track_pending_inside[track_id] = inside
                self._track_zone_streak[track_id] = 1
            else:
                streak = int(self._track_zone_streak.get(track_id, 0) or 0) + 1
                self._track_zone_streak[track_id] = streak
                if streak >= self.zone_confirm_frames:
                    boarding = (not confirmed) and inside
                    alighting = confirmed and (not inside)
                    should_count = (
                        self.gate_direction == "IN" and boarding
                    ) or (self.gate_direction == "OUT" and alighting)
                    if should_count:
                        self.enter_count += 1
                        transition = "上车" if boarding else "下车"
                        self._record_enter_event(
                            person,
                            track_id=track_id,
                            transition=transition,
                            point=foot,
                        )
                    self._track_zone_confirmed[track_id] = inside
                    self._track_pending_inside.pop(track_id, None)
                    self._track_zone_streak.pop(track_id, None)

        for track_id in list(self._track_zone_confirmed.keys()):
            if track_id in active_ids:
                continue
            lost = int(self._track_zone_lost.get(track_id, 0) or 0) + 1
            self._track_zone_lost[track_id] = lost
            if lost > self.track_lost_keep_frames:
                self._clear_track_zone_state(track_id)

    def _count_inside_cabin(
        self,
        persons: List[Dict],
        vehicles: List[Dict],
        image: np.ndarray,
        fence_config: Optional[Dict],
        manual_inside,
    ) -> Tuple[List[Dict], int, Any, Any, Any, str]:
        result = super()._count_inside_cabin(
            persons, vehicles, image, fence_config, manual_inside
        )
        _boarding, _current, inside_box, door_box, vehicle_box, zone_source = result
        # 已停稳时允许短时 hold，避免车检闪失导致 zone:none、无法计数
        if zone_source == "hold":
            if self._vehicle_parked and inside_box is not None:
                zone_source = "vehicle"
            else:
                inside_box = None
                door_box = None
                vehicle_box = None
                zone_source = "none"
        height = image.shape[0]
        persons_filtered = self._filter_person_detections(
            persons,
            frame_height=height,
            vehicle_box=vehicle_box,
            door_box=door_box,
        )
        for person in persons:
            person.pop("boarding", None)
        persons.clear()
        persons.extend(persons_filtered)
        count_box = self._union_boxes(inside_box, door_box)
        if count_box is None and inside_box is not None:
            count_box = inside_box

        width, height = image.shape[1], image.shape[0]
        has_fixed_zone = bool(
            (manual_inside and len(manual_inside) >= 4)
            or self.is_fence_config_valid(fence_config)
        )
        parked = self._update_vehicle_parked(vehicle_box, (width, height))
        use_rear_threshold = (
            zone_source == "vehicle"
            and vehicle_box is not None
            and not has_fixed_zone
        )
        if (
            count_box is not None
            and zone_source == "vehicle"
            and not has_fixed_zone
            and not parked
        ):
            zone_source = "vehicle_moving"

        allow_count = zone_source != "vehicle_moving" and (
            count_box is not None or use_rear_threshold
        )
        frame_size = (width, height)
        visual_box = self._boarding_visual_box(
            count_box, vehicle_box, frame_size, use_rear_threshold and allow_count
        )

        boarding_persons = []
        for person in persons_filtered:
            foot = self._get_detection_point(person)
            if not allow_count:
                boarded = False
            else:
                boarded = self._foot_in_boarding_zone(
                    foot,
                    count_box=count_box,
                    vehicle_box=vehicle_box,
                    inside_box=inside_box,
                    door_box=door_box,
                    frame_size=frame_size,
                    use_rear_threshold=use_rear_threshold,
                )
            person["boarding"] = boarded
            if boarded:
                boarding_persons.append(person)

        current_boarding = len(
            {p.get("track_id", i) for i, p in enumerate(boarding_persons)}
        )
        if allow_count:
            self._update_zone_transitions(
                persons_filtered,
                count_box=count_box,
                vehicle_box=vehicle_box,
                inside_box=inside_box,
                door_box=door_box,
                frame_size=frame_size,
                use_rear_threshold=use_rear_threshold,
            )
        else:
            self._latest_enter_events = []
        # zones.inside 返回可视化合并区
        return (
            boarding_persons,
            current_boarding,
            visual_box or count_box,
            door_box,
            vehicle_box,
            zone_source,
        )

    def process(self, input_data, fence_config=None):
        result = super().process(input_data, fence_config)
        if not result.success or not result.data:
            return result
        data = result.data
        persons = list(data.get("persons") or [])
        vehicles = list(data.get("vehicles") or [])
        if self.draw_vehicle_detections:
            data["detections"] = persons + vehicles
        else:
            data["detections"] = persons
        data["count"] = int(data.get("count", 0) or 0)
        data["enter_count"] = int(self.enter_count)
        return result

    def _build_result(
        self,
        detections: List[Dict],
        persons: List[Dict],
        vehicles: List[Dict],
        boarding_persons: List[Dict],
        current_boarding: int,
        inside_box,
        door_box,
        vehicle_box,
        zone_source: str,
    ) -> Dict[str, Any]:
        data = super()._build_result(
            detections,
            persons,
            vehicles,
            boarding_persons,
            current_boarding,
            inside_box,
            door_box,
            vehicle_box,
            zone_source,
        )

        enter_events = list(self._latest_enter_events)
        recognition_types = sorted(
            {
                str(e.get("recognition_type"))
                for e in enter_events
                if e.get("recognition_type")
            }
        )
        active_alerts: List[Dict[str, Any]] = []
        if enter_events:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for event in enter_events:
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
                        "violation_type": first.get("violation_type", "进入"),
                        "events": group,
                    }
                )

        data["skill_name"] = (
            self.config.get("name") or "non_fixed_parking_boarding_detector"
        )
        data["enter_count"] = int(self.enter_count)
        data["gate_direction"] = self.gate_direction
        data["mine_code"] = self.mine_code
        data["camera_code"] = self.camera_code
        data["enter_events"] = enter_events
        data["has_enter_count_change"] = bool(enter_events)
        data["has_person_count_change"] = False
        data["recognition_types"] = recognition_types
        data["alert_definitions"] = list(self.alert_definitions)
        data["active_alerts"] = active_alerts
        data["flow_metrics"] = {
            **(data.get("flow_metrics") or {}),
            "enter_count": int(self.enter_count),
            "gate_direction": self.gate_direction,
            "current_boarding": current_boarding,
            "vehicle_parked": bool(self._vehicle_parked),
        }
        zones = data.get("zones") if isinstance(data.get("zones"), dict) else {}
        boarding = zones.get("inside")
        data["zones"] = {
            **zones,
            "boarding": boarding,
            "door": None,
            "vehicle": None,
        }
        return data

    def _draw_detections_on_frame(self, frame: np.ndarray, alert_data: Dict) -> np.ndarray:
        """仅绘制人员框 + 单一登车计数区（不叠画整车/车门/缓存 hold 框）。"""
        try:
            zones = alert_data.get("zones") or {}
            zone_source = str(zones.get("source") or "none")
            boarding = zones.get("boarding") or zones.get("inside")
            draw_zone = zone_source in {"vehicle", "manual", "fence"} and boarding
            if zone_source == "vehicle_moving":
                cv2.putText(
                    frame,
                    "vehicle moving, wait for park...",
                    (40, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 255),
                    2,
                )

            if draw_zone and len(boarding) >= 4:
                cv2.rectangle(
                    frame,
                    (int(boarding[0]), int(boarding[1])),
                    (int(boarding[2]), int(boarding[3])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "boarding zone",
                    (int(boarding[0]), max(int(boarding[1]) - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

            persons = alert_data.get("persons") or []
            detections = alert_data.get("detections") or persons
            detections = [
                d for d in detections if d.get("class_name") in PERSON_CLASS_NAMES
            ]
            boarded_track_ids = {
                p.get("track_id")
                for p in persons
                if p.get("track_id") is not None and p.get("boarding")
            }

            track_color_map: Dict[Any, Tuple[int, int, int]] = {}
            for detection in detections:
                track_id = detection.get("track_id")
                if track_id is None:
                    continue
                if track_id in boarded_track_ids or detection.get("boarding"):
                    track_color_map[track_id] = (0, 0, 255)
                else:
                    track_color_map[track_id] = (0, 255, 0)

            self.draw_track_trajectories(frame, detections, track_color_map=track_color_map)

            count = int(alert_data.get("count", 0))
            frame_people = len(persons)
            direction = str(alert_data.get("gate_direction") or self.gate_direction)
            mode = "上车累计" if direction == "IN" else "下车累计"
            enter_count = int(alert_data.get("enter_count", self.enter_count) or 0)

            cv2.putText(
                frame,
                f"onboard: {count}  persons: {frame_people}  zone: {zone_source}",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"{mode}: {enter_count}  dir={direction}",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
            )

            for detection in detections:
                bbox = detection.get("bbox", [])
                if len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox
                confidence = detection.get("confidence", 0.0)
                track_id = detection.get("track_id")
                boarded = bool(detection.get("boarding")) or track_id in boarded_track_ids
                color = (0, 0, 255) if boarded else (0, 255, 0)
                if track_id is not None and track_id in track_color_map:
                    color = track_color_map[track_id]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = (
                    f"ID:{track_id} person: {confidence:.2f}"
                    if track_id is not None
                    else f"person: {confidence:.2f}"
                )
                if boarded:
                    label = f"{label} ON"
                cv2.putText(
                    frame,
                    label,
                    (int(x1), max(int(y1) - 6, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        except Exception as e:
            logger.error("绘制非固定停车上下车画面失败: %s", e)
        return frame


if __name__ == "__main__":
    skill = NonFixedParkingBoardingDetectorSkill(
        NonFixedParkingBoardingDetectorSkill.DEFAULT_CONFIG
    )
    img_dir = _CODE_ROOT / "test_images_videos"
    candidates = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"未找到测试图片: {img_dir}")
    img_name = str(candidates[0])
    frame = cv2.imread(img_name)
    if frame is None:
        raise FileNotFoundError(f"无法加载测试图片: {img_name}")
    result = skill.process(frame)
    print(result.to_dict())

    if result.success and result.data:
        plotted = skill._draw_detections_on_frame(frame.copy(), result.to_dict()["data"])
        out_path = str(
            Path(img_name).with_name(
                Path(img_name).stem + "_non_fixed_parking_ploted.jpg"
            )
        )
        cv2.imwrite(out_path, plotted)
        print(f"结果已保存: {out_path}")
    else:
        print(f"检测失败: {result.error_message}")
