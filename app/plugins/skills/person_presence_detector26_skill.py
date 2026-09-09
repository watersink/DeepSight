"""
画面人数检测技能 (YOLO26)

规则：
  - 使用 yolo26_person 原始 COCO 检测模型，后处理中仅保留 person 类
  - 可选电子围栏过滤
  - 仅统计当前画面人数（不做过线 / 绕行计数与告警）
"""

import sys
from pathlib import Path

# 支持直接运行本文件: python app/plugins/skills/person_presence_detector26_skill.py
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


class PersonPresenceDetector26Skill(BaseSkill):
    """
    画面人数检测技能 (YOLO26)

    基于 yolo26_person，统计当前帧（可选围栏内）的人员数量。
    不包含 count_line / bypass_line 过线逻辑。
    """

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": "person_presence_detector26",
        "name_zh": "画面人数检测(YOLO26)",
        "version": "1.0",
        "cover_image": "/skills/person_presence_detector26.png",
        "description": (
            "基于 yolo26_person 的画面人数检测技能，后处理仅保留 person 类；"
            "仅统计当前画面人数，支持可选电子围栏过滤"
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
        },
        "alert_definitions": [],
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
        # 上一帧已打印的人数，用于人数变化时打日志，避免每帧刷屏
        self._last_logged_person_count: Optional[int] = None

        self.log(
            "info",
            f"初始化 YOLO26 画面人数检测技能: model={self.model_name}"
            f"{f' v{self.model_version}' if self.model_version else ' (latest)'}, "
            f"target_classes={sorted(self.target_class_ids)}, "
            f"tracking={'off' if not self.enable_default_sort_tracking else self.tracking_algorithm}"
        )

    def get_required_models(self) -> List[str]:
        return self.required_models

    def _get_detection_point(self, detection: Dict) -> Optional[Tuple[float, float]]:
        """围栏人数统计使用脚底中心点更稳定。"""
        bbox = detection.get("bbox", [])
        if len(bbox) < 4:
            return None
        x1, _, x2, y2 = bbox
        return ((x1 + x2) / 2.0, float(y2))

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
        对 xywh 框执行 NMS 后映射回原图。
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

            if self.is_fence_config_valid(fence_config):
                height, width = image.shape[:2]
                detections = self.filter_detections_by_fence(
                    detections, fence_config, (width, height)
                )

            count = len(detections)
            prev_count = self._last_logged_person_count
            has_person_count_change = prev_count is not None and count != prev_count
            if count != prev_count:
                if count > 0:
                    self.log("info", f"当前画面检测到人数: {count}")
                elif prev_count is not None:
                    self.log("info", "当前画面无人")
                self._last_logged_person_count = count

            result_data = {
                "detections": detections,
                "count": count,
                "skill_name": self.config.get("name") or "person_presence_detector26",
                "has_person_count_change": has_person_count_change,
                "recognition_types": [],
                "flow_metrics": {
                    "current_person_count": count,
                },
            }
            self._attach_process_timing(result_data, timing)

            return SkillResult.success_result(result_data)

        except Exception as e:
            logger.exception(f"画面人数检测技能处理失败: {str(e)}")
            return SkillResult.error_result(f"处理失败: {str(e)}")

    def _draw_detections_on_frame(self, frame: np.ndarray, alert_data: Dict) -> np.ndarray:
        """在帧上绘制检测框、跟踪轨迹与当前人数。"""
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

            count = int(alert_data.get("count", len(detections)))
            cv2.putText(
                frame, f"person_count: {count}", (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
            )

            for detection in detections:
                bbox = detection.get("bbox", [])
                confidence = detection.get("confidence", 0.0)
                class_name = detection.get("class_name", "unknown")
                track_id = detection.get("track_id")

                if len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox

                    color = class_color_map[class_name]
                    if track_id is not None and track_id in track_color_map:
                        color = track_color_map[track_id]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

                    if track_id is not None:
                        label = f"ID:{track_id} {class_name}: {confidence:.2f}"
                    else:
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
    skill = PersonPresenceDetector26Skill(PersonPresenceDetector26Skill.DEFAULT_CONFIG)
    img_name = str(_CODE_ROOT / "test_images_videos" / "9-74-0004-000880.jpg")
    frame = cv2.imread(img_name)
    if frame is None:
        raise FileNotFoundError(f"无法加载测试图片: {img_name}")
    result = skill.process(frame)
    print(result.to_dict())

    ploted_img = skill._draw_detections_on_frame(frame.copy(), result.to_dict()["data"])
    out_path = str(Path(img_name).with_name(Path(img_name).stem + "_presence_ploted.jpg"))
    cv2.imwrite(out_path, ploted_img)
    print(f"结果已保存: {out_path}")
