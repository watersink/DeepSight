"""
矿井人车后门上车人数统计技能 (YOLO26)

规则：
  - 使用 yolo26_person 原始 COCO 检测模型，后处理保留 person 与人车类
  - 仅当人员脚点进入车厢内区域后，才计入「当前上车」
  - 车厢内区域优先使用电子围栏；未画围栏时根据检测到的人车自动估计
"""

import sys
from pathlib import Path

# 支持直接运行本文件: python app/plugins/skills/boarding_detector_skill.py
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


class AlertThreshold:
    """预警阈值枚举"""
    boardingCount = 1  # 当前上车人数阈值


# COCO 80 类：人员 + 人车（用于无围栏时估计车厢内框）
COCO_PERSON_CLASS_ID = 0
COCO_CLASS_NAMES = {
    0: "person",
    2: "car",
    5: "bus",
    6: "train",
    7: "truck",
}
DEFAULT_TARGET_CLASS_IDS = [0, 2, 5, 6, 7]
PERSON_CLASS_NAMES = {"person"}
VEHICLE_CLASS_NAMES = {"car", "bus", "train", "truck"}

WHITE_HSV_LOW = np.array([0, 0, 145], dtype=np.uint8)
WHITE_HSV_HIGH = np.array([180, 55, 255], dtype=np.uint8)
ORANGE_HSV_LOW = np.array([3, 70, 80], dtype=np.uint8)
ORANGE_HSV_HIGH = np.array([28, 255, 255], dtype=np.uint8)


class BoardingDetectorSkill(BaseSkill):
    """
    上车人数统计技能 (YOLO26)

    基于 yolo26_person，检测人员与人车；人员脚点进入车厢内才计入当前上车人数。
    电子围栏优先作为车厢内区域；未画围栏时根据人车自动估计车厢内框。
    """

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": "boarding_detector",
        "name_zh": "固定停车人数统计(YOLO26)",
        "version": "1.0",
        "cover_image": "/skills/boarding_detector.png",
        "description": (
            "基于 yolo26_person 的矿井人车后门上车人数统计："
            "人员脚点进入车厢内区域才计数；未画围栏时根据人车自动估计车厢内框"
        ),
        "status": True,
        "required_models": ["yolo26_person"],
        "params": {
            "classes": ["person", "car", "bus", "train", "truck"],
            "target_class_ids": DEFAULT_TARGET_CLASS_IDS,
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "max_det": 300,
            "input_size": [640, 640],
            "enable_default_sort_tracking": True,
            "tracking_algorithm": "sort",
            "tracking_frame_rate": 30,
            "tracking_max_age": 90,
            "tracking_min_hits": 1,
            "tracking_iou_threshold": 0.3,
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.1,
            "track_buffer": 90,
            "match_thresh": 0.8,
            "new_track_thresh": 0.6,
            "tracking_history_frames": 10,
            "enable_track_history": True,
            "model_version": "1",
            "enable_timing_log": False,
            "enable_helmet_filter": True,
            "helmet_min_ratio": 0.04,
            "roi_hold": 45,
            "door_expand": 0.28,
            "boarding_count": AlertThreshold.boardingCount,
        },
        "alert_definitions": [
            {
                "key": "boarding_count",
                "level": 1,
                "description": (
                    f"当检测到 {AlertThreshold.boardingCount} 人及以上脚点进入车厢内区域时触发。"
                    "建议将电子围栏画在车门内侧车厢区域。"
                ),
                "codes": {
                    "BOARDING": {
                        "code": "08",
                        "description": "无轨胶轮车上车点上下车人数",
                    },
                },
            }
        ],
    }

    def _initialize(self) -> None:
        params = self.config.get("params", {})

        self.classes = params.get("classes", ["person", "car", "bus", "train", "truck"])
        target_class_ids = params.get("target_class_ids", DEFAULT_TARGET_CLASS_IDS)
        self.target_class_ids: Set[int] = {int(cid) for cid in target_class_ids}
        self.class_names = {
            int(cid): COCO_CLASS_NAMES.get(int(cid), str(cid))
            for cid in self.target_class_ids
        }
        self.class_names.setdefault(COCO_PERSON_CLASS_ID, "person")

        self.conf_thres = params.get("conf_thres", 0.25)
        self.iou_thres = params.get("iou_thres", 0.45)
        self.max_det = params.get("max_det", 300)
        self.input_width, self.input_height = params.get("input_size", [640, 640])

        self.required_models = self.config.get("required_models", ["yolo26_person"])
        self.model_name = self.required_models[0]
        self.model_version = str(params.get("model_version", "") or "").strip()

        self.enable_default_sort_tracking = params.get("enable_default_sort_tracking", True)
        self.tracking_algorithm = params.get("tracking_algorithm", "bytetrack")

        self.enable_helmet_filter = params.get("enable_helmet_filter", True)
        self.helmet_min_ratio = float(params.get("helmet_min_ratio", 0.04))
        self.roi_hold = int(params.get("roi_hold", 45))
        self.door_expand = float(params.get("door_expand", 0.28))
        self.boarding_count = int(params.get("boarding_count", AlertThreshold.boardingCount))

        self.last_inside = None
        self.last_door = None
        self.last_vehicle = None
        self.zone_miss = 0
        self._last_logged_boarding_count: Optional[int] = None
        self.alert_definitions: List[Dict[str, Any]] = list(
            self.config.get("alert_definitions") or []
        )

        self.log(
            "info",
            f"初始化 YOLO26 上车人数统计技能: model={self.model_name}"
            f"{f' v{self.model_version}' if self.model_version else ' (latest)'}, "
            f"target_classes={sorted(self.target_class_ids)}, "
            f"boarding_count={self.boarding_count}, "
            f"tracking={'off' if not self.enable_default_sort_tracking else self.tracking_algorithm}"
        )

    def get_required_models(self) -> List[str]:
        return self.required_models

    def _get_alert_definition(self) -> Dict[str, Any]:
        """解析上车人数识别类型：08=无轨胶轮车上车点上下车人数。"""
        item = self.alert_definitions[0] if self.alert_definitions else {}
        codes = item.get("codes") if isinstance(item.get("codes"), dict) else {}
        code_item = {}
        if isinstance(codes.get("BOARDING"), dict):
            code_item = codes["BOARDING"]
        elif codes:
            first = next(iter(codes.values()))
            if isinstance(first, dict):
                code_item = first
        return {
            "key": str(item.get("key") or "boarding_count"),
            "level": int(item.get("level", 1) or 1),
            "description": str(
                code_item.get("description") or item.get("description") or ""
            ).strip(),
            "recognition_type": str(code_item.get("code") or "08").strip() or "08",
        }

    def _get_detection_point(self, detection: Dict) -> Optional[Tuple[float, float]]:
        """围栏/车厢判定使用脚底中心点更稳定。"""
        bbox = detection.get("bbox", [])
        if len(bbox) < 4:
            return None
        x1, _, x2, y2 = bbox
        return ((x1 + x2) / 2.0, float(y2))

    def _load_image(
        self,
        input_data: Union[np.ndarray, str, Dict[str, Any], Any]
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[Any], Optional[str]]:
        image = None
        fence_config = None
        manual_inside = None

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
            manual_inside = input_data.get("inside_box") or input_data.get("roi")
        else:
            return None, None, None, "不支持的输入数据类型"

        if image is None or image.size == 0:
            return None, None, None, "无效的图像数据"

        return image, fence_config, manual_inside, None

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

    def _clip_box(
        self,
        left: int,
        top: int,
        width_box: int,
        height_box: int,
        img_width: int,
        img_height: int,
    ) -> Tuple[int, int, int, int]:
        left = max(0, left)
        top = max(0, top)
        width_box = min(width_box, img_width - left)
        height_box = min(height_box, img_height - top)
        return left, top, width_box, height_box

    def _build_detection(
        self,
        left: int,
        top: int,
        width_box: int,
        height_box: int,
        score: float,
        class_id: int,
    ) -> Dict[str, Any]:
        class_name = self.class_names.get(class_id, "unknown")
        target_type = "person" if class_name in PERSON_CLASS_NAMES else "vehicle"
        return {
            "bbox": [left, top, left + width_box, top + height_box],
            "confidence": float(score),
            "class_id": int(class_id),
            "class_name": class_name,
            "target_type": target_type,
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

            x1 = float(row[0])
            y1 = float(row[1])
            x2 = float(row[2])
            y2 = float(row[3])
            if x2 <= x1 or y2 <= y1:
                cx, cy, box_w, box_h = x1, y1, x2, y2
                x1, y1 = cx - box_w / 2.0, cy - box_h / 2.0
                x2, y2 = cx + box_w / 2.0, cy + box_h / 2.0

            left = int(x1 * x_factor)
            top = int(y1 * y_factor)
            width_box = int((x2 - x1) * x_factor)
            height_box = int((y2 - y1) * y_factor)
            left, top, width_box, height_box = self._clip_box(
                left, top, width_box, height_box, img_width, img_height
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
        对人/车分别做 NMS 后映射回原图，避免车门处人车框互相抑制。
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

        results: List[Dict[str, Any]] = []
        for class_id in set(class_ids):
            cls_indices = [i for i, cid in enumerate(class_ids) if cid == class_id]
            cls_boxes = [boxes[i] for i in cls_indices]
            cls_scores = [scores[i] for i in cls_indices]
            nms_indices = cv2.dnn.NMSBoxes(cls_boxes, cls_scores, self.conf_thres, self.iou_thres)
            if nms_indices is None or len(nms_indices) == 0:
                continue

            if isinstance(nms_indices, np.ndarray):
                nms_indices = nms_indices.flatten()

            for idx in nms_indices:
                if isinstance(idx, (list, tuple, np.ndarray)):
                    idx = idx[0]
                idx = int(idx)
                original_idx = cls_indices[idx]
                box = boxes[original_idx]
                results.append(
                    self._build_detection(
                        box[0], box[1], box[2], box[3],
                        scores[original_idx], class_ids[original_idx],
                    )
                )
                if len(results) >= self.max_det:
                    return results

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
            image, input_fence_config, manual_inside, error_message = self._load_image(input_data)
            if error_message:
                return SkillResult.error_result(error_message)

            if input_fence_config is not None:
                fence_config = input_fence_config

            ready, err_msg = self.ensure_models_ready()
            if not ready:
                return SkillResult.error_result(f"模型未就绪: {err_msg}")

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
                return SkillResult.error_result("人员检测推理失败")

            t0 = self._timing_start()
            detections = self.postprocess(outputs, image)
            timing["postprocess_ms"] = self._timing_elapsed_ms(t0)

            if self.enable_default_sort_tracking:
                t0 = self._timing_start()
                detections = self.add_tracking_ids(detections, image=image)
                timing["track_ms"] = self._timing_elapsed_ms(t0)

            persons = [d for d in detections if d.get("class_name") in PERSON_CLASS_NAMES]
            vehicles = [d for d in detections if d.get("class_name") in VEHICLE_CLASS_NAMES]

            if self.enable_helmet_filter:
                helmeted = [p for p in persons if self._has_orange_helmet(image, p)]
                if helmeted:
                    persons = helmeted

            t0 = self._timing_start()
            boarding_persons, current_boarding, inside_box, door_box, vehicle_box, zone_source = (
                self._count_inside_cabin(persons, vehicles, image, fence_config, manual_inside)
            )
            timing["boarding_ms"] = self._timing_elapsed_ms(t0)

            result_data = self._build_result(
                detections, persons, vehicles, boarding_persons,
                current_boarding, inside_box, door_box, vehicle_box, zone_source,
            )
            self._attach_process_timing(result_data, timing)
            return SkillResult.success_result(result_data)

        except Exception as e:
            logger.exception(f"上车人数统计技能处理失败: {str(e)}")
            return SkillResult.error_result(f"处理失败: {str(e)}")

    def _count_inside_cabin(
        self,
        persons: List[Dict],
        vehicles: List[Dict],
        image: np.ndarray,
        fence_config: Optional[Dict],
        manual_inside,
    ) -> Tuple[List[Dict], int, Any, Any, Any, str]:
        """脚点进入车厢内才计数：手动框 > 电子围栏 > 人车估计 > 短时保持。"""
        height, width = image.shape[:2]
        image_size = (width, height)
        inside_box = None
        door_box = None
        vehicle_box = None
        zone_source = "none"

        if manual_inside and len(manual_inside) >= 4:
            inside_box = self._clamp_box(manual_inside, width, height)
            zone_source = "manual"
            self.last_inside = inside_box
            self.zone_miss = 0
        elif self.is_fence_config_valid(fence_config):
            boarding_persons = self.filter_detections_by_fence(persons, fence_config, image_size)
            zone_source = "fence"
            inside_box = self._fence_aabb(fence_config, image_size)
            self.last_inside = inside_box
            self.zone_miss = 0
            counts = self._count_boarding(persons, set(id(p) for p in boarding_persons))
            for p in persons:
                p["boarding"] = id(p) in counts["boarding_obj_ids"]
            return (
                boarding_persons,
                counts["current_boarding"],
                inside_box,
                door_box,
                vehicle_box,
                zone_source,
            )
        elif fence_config:
            self.log(
                "info",
                f"围栏配置无效，改用人车自动估计车厢内框: regions={len(self._fence_regions(fence_config))}"
            )

        if zone_source != "manual":
            vehicle = self._pick_vehicle(vehicles, image)
            if vehicle is not None:
                vehicle_box = self._clamp_box(vehicle["bbox"], width, height)
                inside_box, door_box = self._estimate_zones_from_vehicle(
                    vehicle_box, (height, width), self.door_expand
                )
                self.last_inside = inside_box
                self.last_door = door_box
                self.last_vehicle = vehicle_box
                self.zone_miss = 0
                zone_source = "vehicle"
            else:
                self.zone_miss += 1
                if self.last_inside is not None and self.zone_miss <= self.roi_hold:
                    inside_box = self.last_inside
                    door_box = self.last_door
                    vehicle_box = self.last_vehicle
                    zone_source = "hold"
                else:
                    inside_box = None

        boarding_persons = []
        for p in persons:
            foot = self._get_detection_point(p)
            boarded = self._point_in_box(foot, inside_box)
            p["boarding"] = boarded
            if boarded:
                boarding_persons.append(p)

        current_boarding = len({p.get("track_id", i) for i, p in enumerate(boarding_persons)})
        return boarding_persons, current_boarding, inside_box, door_box, vehicle_box, zone_source

    def analyze_safety(self, current_boarding: int, frame_people: int, zone_source: str) -> Dict[str, Any]:
        is_safe = current_boarding == 0
        alert_triggered = False
        alert_name = ""
        alert_type = ""
        alert_description = ""

        if current_boarding >= self.boarding_count:
            alert_triggered = True
            alert_name = "上车人数预警"
            alert_type = "区域人数预警"
            alert_description = (
                f"检测到{current_boarding}人脚点已进入车厢内区域（画面人员{frame_people}人，"
                f"区域来源={zone_source}），请核对上车情况。"
            )

        return {
            "current_boarding": current_boarding,
            "frame_people": frame_people,
            "zone_source": zone_source,
            "is_safe": is_safe,
            "alert_info": {
                "alert_triggered": alert_triggered,
                "alert_name": alert_name,
                "alert_type": alert_type,
                "alert_description": alert_description,
            },
        }

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
        prev_count = self._last_logged_boarding_count
        has_person_count_change = prev_count is not None and current_boarding != prev_count
        if current_boarding != prev_count:
            self.log(
                "info",
                f"上车人数分析：当前上车={current_boarding}，画面人数={len(persons)}，"
                f"是否触发预警={current_boarding >= self.boarding_count}，区域来源={zone_source}"
            )
            self._last_logged_boarding_count = current_boarding

        safety_metrics = self.analyze_safety(current_boarding, len(persons), zone_source)
        alert_def = self._get_alert_definition()
        recognition_type = str(alert_def.get("recognition_type") or "08").strip() or "08"
        enter_events: List[Dict[str, Any]] = []
        active_alerts: List[Dict[str, Any]] = []
        recognition_types: List[str] = []
        if has_person_count_change:
            recognition_types = [recognition_type]
            for i, person in enumerate(boarding_persons):
                enter_events.append(
                    {
                        "track_id": person.get("track_id", i),
                        "violation_type": "上车",
                        "alert_key": alert_def.get("key", "boarding_count"),
                        "alert_level": alert_def.get("level"),
                        "alert_description": alert_def.get("description"),
                        "recognition_type": recognition_type,
                        "current_boarding": current_boarding,
                    }
                )
            active_alerts.append(
                {
                    "key": alert_def.get("key", "boarding_count"),
                    "level": alert_def.get("level"),
                    "description": alert_def.get("description"),
                    "recognition_type": recognition_type,
                    "violation_type": "上车",
                    "events": enter_events,
                }
            )
        return {
            "detections": detections,
            "persons": persons,
            "vehicles": vehicles,
            "boarding_detections": boarding_persons,
            "count": current_boarding,
            "zones": {
                "inside": list(inside_box) if inside_box else None,
                "door": list(door_box) if door_box else None,
                "vehicle": list(vehicle_box) if vehicle_box else None,
                "source": zone_source,
            },
            "safety_metrics": safety_metrics,
            "skill_name": self.config.get("name") or "boarding_detector",
            "has_person_count_change": has_person_count_change,
            "has_enter_count_change": bool(enter_events),
            "enter_events": enter_events,
            "recognition_types": recognition_types,
            "alert_definitions": list(self.alert_definitions),
            "active_alerts": active_alerts,
            "flow_metrics": {
                "current_boarding": current_boarding,
                "current_person_count": len(persons),
                "zone_source": zone_source,
            },
        }

    def _count_boarding(self, persons: List[Dict], boarding_obj_ids: set) -> Dict[str, Any]:
        boarding_ids = set()
        boarding_obj = set()
        for i, p in enumerate(persons):
            if id(p) in boarding_obj_ids:
                boarding_ids.add(p.get("track_id", i))
                boarding_obj.add(id(p))
        return {
            "current_boarding": len(boarding_ids),
            "boarding_obj_ids": boarding_obj,
        }

    @staticmethod
    def _clamp_box(box, w: int, h: int) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        x1, x2 = max(0, min(x1, x2)), min(w - 1, max(x1, x2))
        y1, y2 = max(0, min(y1, y2)), min(h - 1, max(y1, y2))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _point_in_box(pt: Optional[Tuple[float, float]], box: Optional[Tuple[int, int, int, int]]) -> bool:
        if pt is None or box is None:
            return False
        x, y = pt
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def _fence_aabb(self, fence_config: Dict, image_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        width, height = image_size
        xs, ys = [], []
        for region in self._fence_regions(fence_config):
            for x, y in region.get("points") or []:
                xs.append(x * width)
                ys.append(y * height)
        if not xs:
            return None
        return self._clamp_box((min(xs), min(ys), max(xs), max(ys)), width, height)

    def _has_orange_helmet(self, frame: np.ndarray, person: Dict) -> bool:
        bbox = person.get("bbox") or []
        if len(bbox) < 4:
            return False
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        helmet_h = max(1.0, (y2 - y1) * 0.32)
        hx1, hy1, hx2, hy2 = self._clamp_box((x1, y1, x2, y1 + helmet_h), w, h)
        crop = frame[hy1:hy2, hx1:hx2]
        if crop.size == 0:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ORANGE_HSV_LOW, ORANGE_HSV_HIGH)
        return float(np.count_nonzero(mask)) / float(mask.size) >= self.helmet_min_ratio

    def _white_ratio(self, frame: np.ndarray, box) -> float:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._clamp_box(box, w, h)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, WHITE_HSV_LOW, WHITE_HSV_HIGH)
        return float(np.count_nonzero(mask)) / float(mask.size)

    def _vehicle_score(self, frame: np.ndarray, det: Dict) -> float:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = det.get("bbox") or [0, 0, 0, 0]
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        area_ratio = (bw * bh) / float(w * h)
        if area_ratio < 0.04 or area_ratio > 0.62:
            return -1.0
        cx, cy = (x1 + x2) / 2.0 / w, (y1 + y2) / 2.0 / h
        loc = 1.0 - (abs(cx - 0.55) + abs(cy - 0.48))
        if x1 < 0.05 * w:
            loc -= 0.55
        wr = self._white_ratio(frame, det["bbox"])
        if wr < 0.08:
            return -1.0
        return 0.20 * float(det.get("confidence", 0.0)) + 0.55 * wr + 0.25 * loc

    def _pick_vehicle(self, vehicles: List[Dict], frame: np.ndarray) -> Optional[Dict]:
        ranked = []
        for det in vehicles:
            score = self._vehicle_score(frame, det)
            if score > 0:
                ranked.append((score, det))
        if not ranked:
            return None
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked[0][1]

    def _estimate_zones_from_vehicle(self, vehicle, frame_shape, door_expand: float):
        h, w = frame_shape
        x1, y1, x2, y2 = vehicle
        bw, bh = x2 - x1, y2 - y1
        inside = self._clamp_box(
            (x1 + 0.26 * bw, y1 + 0.08 * bh, x2 - 0.26 * bw, y1 + 0.55 * bh),
            w,
            h,
        )
        door = self._clamp_box(
            (x1 + 0.12 * bw, y1 + 0.55 * bh, x2 - 0.12 * bw, y2 + door_expand * bh),
            w,
            h,
        )
        return inside, door

    def _draw_detections_on_frame(self, frame: np.ndarray, alert_data: Dict) -> np.ndarray:
        """在帧上绘制车厢区域、检测框、跟踪轨迹与当前上车人数。"""
        try:
            zones = alert_data.get("zones") or {}
            inside = zones.get("inside")
            door = zones.get("door")
            vehicle = zones.get("vehicle")

            if vehicle and len(vehicle) >= 4:
                cv2.rectangle(
                    frame, (int(vehicle[0]), int(vehicle[1])),
                    (int(vehicle[2]), int(vehicle[3])), (255, 255, 255), 2,
                )
            if door and len(door) >= 4:
                cv2.rectangle(
                    frame, (int(door[0]), int(door[1])),
                    (int(door[2]), int(door[3])), (0, 255, 255), 2,
                )
            if inside and len(inside) >= 4:
                cv2.rectangle(
                    frame, (int(inside[0]), int(inside[1])),
                    (int(inside[2]), int(inside[3])), (0, 255, 0), 2,
                )

            detections = alert_data.get("detections", [])
            persons = alert_data.get("persons") or []
            boarding_ids = {
                id(p) for p in (alert_data.get("boarding_detections") or [])
            }
            boarded_track_ids = {
                p.get("track_id") for p in persons
                if p.get("track_id") is not None and (
                    p.get("boarding") or id(p) in boarding_ids
                )
            }

            track_color_map: Dict[Any, Tuple[int, int, int]] = {}
            for detection in detections:
                track_id = detection.get("track_id")
                if track_id is None:
                    continue
                if track_id in boarded_track_ids or detection.get("boarding"):
                    track_color_map[track_id] = (0, 0, 255)
                elif detection.get("class_name") in PERSON_CLASS_NAMES:
                    track_color_map[track_id] = (0, 255, 0)
                else:
                    track_color_map[track_id] = (255, 200, 0)

            self.draw_track_trajectories(frame, detections, track_color_map=track_color_map)

            count = int(alert_data.get("count", 0))
            frame_people = len(persons) if persons else int(
                (alert_data.get("flow_metrics") or {}).get("current_person_count", 0)
            )
            zone_source = zones.get("source") or "none"
            cv2.putText(
                frame, f"boarding: {count}  persons: {frame_people}  zone: {zone_source}",
                (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
            )

            for detection in detections:
                bbox = detection.get("bbox", [])
                if len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = bbox
                class_name = detection.get("class_name", "unknown")
                confidence = detection.get("confidence", 0.0)
                track_id = detection.get("track_id")
                boarded = bool(detection.get("boarding")) or track_id in boarded_track_ids

                if boarded:
                    color = (0, 0, 255)
                elif class_name in PERSON_CLASS_NAMES:
                    color = (0, 255, 0)
                else:
                    color = (255, 200, 0)
                if track_id is not None and track_id in track_color_map:
                    color = track_color_map[track_id]

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                if track_id is not None:
                    label = f"ID:{track_id} {class_name}: {confidence:.2f}"
                else:
                    label = f"{class_name}: {confidence:.2f}"
                if boarded:
                    label = f"{label} IN"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    frame, (int(x1), int(y1) - text_height - baseline - 5),
                    (int(x1) + text_width, int(y1)), color, -1,
                )
                cv2.putText(
                    frame, label, (int(x1), int(y1) - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2,
                )

            return frame
        except Exception as e:
            logger.error(f"绘制检测框时出错: {str(e)}")
            return frame


if __name__ == "__main__":
    skill = BoardingDetectorSkill(BoardingDetectorSkill.DEFAULT_CONFIG)
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
        ploted_img = skill._draw_detections_on_frame(frame.copy(), result.to_dict()["data"])
        out_path = str(Path(img_name).with_name(Path(img_name).stem + "_boarding_ploted.jpg"))
        cv2.imwrite(out_path, ploted_img)
        print(f"结果已保存: {out_path}")
    else:
        print(f"检测失败: {result.error_message}")
