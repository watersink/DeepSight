"""
人数检测技能

规则：
  - 使用 yolo11_person 检测人员
  - 可选电子围栏过滤
  - 统计当前人数
  - 超过人数上限时触发告警
"""

import sys
from pathlib import Path

# 支持直接运行本文件: python app/plugins/skills/person_count_detector_skill.py
_CODE_ROOT = Path(__file__).resolve().parents[3]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Union, Optional
from app.skills.skill_base import BaseSkill, SkillResult
from app.services.triton_client import triton_client
import logging

logger = logging.getLogger(__name__)


class PersonCountDetectorSkill(BaseSkill):
    """
    简单人数检测技能

    使用 yolo11_person 检测当前画面人数。
    """

    DEFAULT_CONFIG = {
        "type": "detection",
        "name": "person_count_detector",
        "name_zh": "人流量检测",
        "version": "1.0",
        "cover_image": "/skills/person_count_detector.png",
        "description": "基于 yolo11_person 的简单人数检测技能，支持人数统计",
        "status": True,
        "required_models": ["yolo11_person"],
        "params": {
            "classes": ["person"],
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
            "person_limit": 6,
            "model_version": "9",
            "enable_timing_log": False,
        },
        "alert_definitions": [
            {
                "level": 1,
                "description": "当检测区域内人数超过设定上限时触发告警。"
            }
        ]
    }

    def _initialize(self) -> None:
        params = self.config.get("params", {})

        self.classes = params.get("classes", ["person"])
        self.class_names = {i: name for i, name in enumerate(self.classes)}

        self.conf_thres = params.get("conf_thres", 0.5)
        self.iou_thres = params.get("iou_thres", 0.45)
        self.max_det = params.get("max_det", 300)
        self.input_width, self.input_height = params.get("input_size", [640, 640])

        self.required_models = self.config.get("required_models", ["yolo11_person"])
        self.model_name = self.required_models[0]
        self.model_version = str(params.get("model_version", "") or "").strip()

        self.enable_default_sort_tracking = params.get("enable_default_sort_tracking", True)
        self.tracking_algorithm = params.get("tracking_algorithm", "bytetrack")

        self.log(
            "info",
            f"初始化人数检测技能: model={self.model_name}"
            f"{f' v{self.model_version}' if self.model_version else ' (latest)'}, "
            f"tracking={'off' if not self.enable_default_sort_tracking else self.tracking_algorithm}"
        )

    def get_required_models(self) -> List[str]:
        return self.required_models

    def _get_detection_point(self, detection: Dict) -> Optional[Tuple[float, float]]:
        """
        围栏人数统计使用脚底中心点更稳定。
        """
        bbox = detection.get("bbox", [])
        if len(bbox) < 4:
            return None

        x1, _, x2, y2 = bbox
        return ((x1 + x2) / 2.0, float(y2))

    def _load_image(
        self,
        input_data: Union[np.ndarray, str, Dict[str, Any], Any]
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]], Optional[int], Optional[str]]:
        image = None
        fence_config = None
        custom_limit = None

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
            if "person_limit" in input_data:
                custom_limit = int(input_data["person_limit"])
        else:
            return None, None, None, "不支持的输入数据类型"

        if image is None or image.size == 0:
            return None, None, None, "无效的图像数据"

        return image, fence_config, custom_limit, None

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, axis=0)

    def postprocess(self, outputs: Dict[str, np.ndarray], original_img: np.ndarray) -> List[Dict[str, Any]]:
        height, width = original_img.shape[:2]
        detections = outputs.get("output0")
        if detections is None:
            return []

        detections = np.squeeze(detections, axis=0)
        detections = np.transpose(detections, (1, 0))

        boxes: List[List[int]] = []
        scores: List[float] = []
        class_ids: List[int] = []

        x_factor = width / self.input_width
        y_factor = height / self.input_height

        for i in range(detections.shape[0]):
            classes_scores = detections[i][4:]
            max_score = float(np.amax(classes_scores))
            if max_score < self.conf_thres:
                continue

            class_id = int(np.argmax(classes_scores))
            if class_id not in self.class_names:
                continue

            x, y, w, h = detections[i][:4]
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width_box = int(w * x_factor)
            height_box = int(h * y_factor)

            left = max(0, left)
            top = max(0, top)
            width_box = min(width_box, width - left)
            height_box = min(height_box, height - top)

            boxes.append([left, top, width_box, height_box])
            scores.append(max_score)
            class_ids.append(class_id)

        results: List[Dict[str, Any]] = []
        if not boxes:
            return results

        unique_classes = set(class_ids)
        for cls_id in unique_classes:
            cls_indices = [i for i, cid in enumerate(class_ids) if cid == cls_id]
            cls_boxes = [boxes[i] for i in cls_indices]
            cls_scores = [scores[i] for i in cls_indices]
            nms_indices = cv2.dnn.NMSBoxes(cls_boxes, cls_scores, self.conf_thres, self.iou_thres)

            if len(nms_indices) == 0:
                continue

            if isinstance(nms_indices, np.ndarray):
                nms_indices = nms_indices.flatten()

            for idx in nms_indices:
                if isinstance(idx, (list, tuple, np.ndarray)):
                    idx = idx[0]
                idx = int(idx)

                original_idx = cls_indices[idx]
                box = boxes[original_idx]

                results.append({
                    "bbox": [box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    "confidence": float(scores[original_idx]),
                    "class_id": int(class_ids[original_idx]),
                    "class_name": self.class_names.get(class_ids[original_idx], "person"),
                    "target_type": "person"
                })

        return results


    def process(
        self,
        input_data: Union[np.ndarray, str, Dict[str, Any], Any],
        fence_config: Dict = None
    ) -> SkillResult:
        try:
            image, input_fence_config, custom_limit, error_message = self._load_image(input_data)
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
                detections = self.filter_detections_by_fence(detections, fence_config, (width, height))

            result_data = {
                "detections": detections,
                "count": len(detections),
                "flow_metrics": {
                    "current_person_count": len(detections)
                }
            }
            self._attach_process_timing(result_data, timing)

            return SkillResult.success_result(result_data)

        except Exception as e:
            logger.exception(f"人数检测技能处理失败: {str(e)}")
            return SkillResult.error_result(f"处理失败: {str(e)}")




    def _draw_detections_on_frame(self, frame: np.ndarray, alert_data: Dict) -> np.ndarray:
        """在帧上绘制检测框与跟踪轨迹"""
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
    skill = PersonCountDetectorSkill(PersonCountDetectorSkill.DEFAULT_CONFIG)
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