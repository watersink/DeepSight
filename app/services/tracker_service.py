"""
目标跟踪服务 - 支持 SORT / ByteTrack / BoT-SORT / OC-SORT / FastTracker / Deep OC-SORT / TrackTrack
按类别分离多跟踪器，避免跨类别错误关联
"""
import logging
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from app.services.trackers.detection_results import DetectionResults
from app.services.trackers.sort import Sort
from app.services.trackers.tracker_args import build_tracker_args

logger = logging.getLogger(__name__)

TRACKING_ALGORITHMS = (
    "sort",
    "bytetrack",
    "botsort",
    "ocsort",
    "fasttrack",
    "deepocsort",
    "tracktrack",
)
_ALGORITHM_ALIASES = {
    "sort": "sort",
    "SORT": "sort",
    "bytetrack": "bytetrack",
    "byte_track": "bytetrack",
    "byte": "bytetrack",
    "ByteTrack": "bytetrack",
    "botsort": "botsort",
    "bot_sort": "botsort",
    "BoT-SORT": "botsort",
    "BOTSORT": "botsort",
    "ocsort": "ocsort",
    "oc_sort": "ocsort",
    "OCSORT": "ocsort",
    "fasttrack": "fasttrack",
    "fast_tracker": "fasttrack",
    "FASTTracker": "fasttrack",
    "deepocsort": "deepocsort",
    "deep_oc_sort": "deepocsort",
    "DeepOCSORT": "deepocsort",
    "tracktrack": "tracktrack",
    "track_track": "tracktrack",
    "TRACKTRACK": "tracktrack",
}


def normalize_tracking_algorithm(value: Any, default: str = "sort") -> str:
    """将配置值规范化为支持的跟踪算法名称"""
    if value is None:
        return default
    normalized = _ALGORITHM_ALIASES.get(str(value).strip(), str(value).strip().lower())
    if normalized not in TRACKING_ALGORITHMS:
        logger.warning(
            "未知跟踪算法 '%s'，回退为 '%s'，可选: %s",
            value, default, ", ".join(TRACKING_ALGORITHMS),
        )
        return default
    return normalized


class TrackerService:
    """目标跟踪服务 - 支持按类别分离的多跟踪器"""

    def __init__(
        self,
        algorithm: str = "sort",
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
        frame_rate: int = 30,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        new_track_thresh: float = 0.6,
        fuse_score: bool = True,
        tracking_history_frames: int = 10,
        enable_track_history: bool = True,
    ):
        """
        Args:
            algorithm: 跟踪算法，可选 sort / bytetrack / botsort / ocsort / fasttrack / deepocsort / tracktrack
            max_age: [SORT] 轨迹连续无匹配时最多保留的帧数，超出则删除该轨迹
            min_hits: [SORT] 轨迹开始对外输出前，至少需要连续匹配到的帧数
            iou_threshold: [SORT] 检测框与卡尔曼预测框进行 IoU 关联时的最低阈值
            frame_rate: [ByteTrack/BoT-SORT] 视频帧率，用于换算轨迹最大丢失时长
            track_high_thresh: [ByteTrack/BoT-SORT] 高分检测框置信度阈值，用于第一阶段关联
            track_low_thresh: [ByteTrack/BoT-SORT] 低分检测框置信度下限，用于第二阶段关联
            track_buffer: [ByteTrack/BoT-SORT] 轨迹缓冲帧数，目标丢失后仍保留轨迹的最大帧数
            match_thresh: [ByteTrack/BoT-SORT] 第一阶段轨迹与检测框 IoU 匹配的阈值
            new_track_thresh: [ByteTrack/BoT-SORT] 未匹配检测框创建新轨迹所需的最低置信度
            fuse_score: [ByteTrack/BoT-SORT] 是否将检测置信度融合进匹配代价，提高高置信度框优先级
            tracking_history_frames: 每个 track_id 保留的历史中心点帧数（如 10、20）
            enable_track_history: 是否记录并输出历史轨迹中心点
        """
        self.algorithm = normalize_tracking_algorithm(algorithm)
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_rate = frame_rate
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.new_track_thresh = new_track_thresh
        self.fuse_score = fuse_score
        self.tracking_history_frames = max(0, int(tracking_history_frames))
        self.enable_track_history = enable_track_history

        self.trackers: Dict[str, Any] = {}
        self._frame_index = 0
        self._track_histories: Dict[int, Deque[Tuple[float, float]]] = {}
        self._track_last_seen: Dict[int, int] = {}

        logger.info(
            "初始化多类别跟踪器: algorithm=%s, max_age=%s, min_hits=%s, iou_threshold=%s",
            self.algorithm, max_age, min_hits, iou_threshold,
        )

    @classmethod
    def from_params(cls, params: Optional[Dict[str, Any]] = None) -> "TrackerService":
        """从技能 params 字典创建跟踪服务"""
        params = params or {}
        return cls(
            algorithm=params.get("tracking_algorithm", "sort"),
            max_age=int(params.get("tracking_max_age", params.get("max_age", 30))),
            min_hits=int(params.get("tracking_min_hits", params.get("min_hits", 1))),
            iou_threshold=float(
                params.get("tracking_iou_threshold", params.get("iou_threshold", 0.3))
            ),
            frame_rate=int(params.get("tracking_frame_rate", params.get("frame_rate", 30))),
            track_high_thresh=float(params.get("track_high_thresh", 0.5)),
            track_low_thresh=float(params.get("track_low_thresh", 0.1)),
            track_buffer=int(params.get("track_buffer", 30)),
            match_thresh=float(params.get("match_thresh", 0.8)),
            new_track_thresh=float(params.get("new_track_thresh", 0.6)),
            fuse_score=bool(params.get("fuse_score", True)),
            tracking_history_frames=int(params.get("tracking_history_frames", 20)),
            enable_track_history=bool(params.get("enable_track_history", True)),
        )

    def update(
        self,
        detections: List[Dict],
        image: Any = None,
    ) -> List[Dict]:
        """
        更新跟踪器并为检测结果分配 track_id

        Args:
            detections: 检测结果列表（bbox/confidence/class_name 等）
            image: 当前帧图像（BoT-SORT 的 GMC 模块可选使用）

        Returns:
            带 track_id、track_history 的检测结果列表
        """
        try:
            if self.algorithm == "sort":
                tracked = self._update_sort(detections)
            else:
                tracked = self._update_ultralytics_tracker(detections, image=image)
            return self._record_and_attach_histories(tracked)
        except Exception as e:
            logger.error("多类别跟踪器更新失败 [%s]: %s", self.algorithm, e)
            return detections

    def _update_sort(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            for tracker in self.trackers.values():
                tracker.update(np.empty((0, 5)))
            return []

        detections_by_class = self._group_by_class(detections)
        all_tracked: List[Dict] = []

        for class_name, class_detections in detections_by_class.items():
            tracker = self._get_or_create_sort_tracker(class_name)
            dets_np = self._detections_to_sort_array(class_detections)
            tracked_objects = tracker.update(dets_np)
            all_tracked.extend(
                self._associate_sort_tracks(tracked_objects, class_detections, class_name)
            )

        active_classes = set(detections_by_class.keys())
        for class_name, tracker in self.trackers.items():
            if class_name not in active_classes:
                tracker.update(np.empty((0, 5)))

        return all_tracked

    def _update_ultralytics_tracker(
        self,
        detections: List[Dict],
        image: Any = None,
    ) -> List[Dict]:
        if not detections:
            for tracker in self.trackers.values():
                tracker.update(DetectionResults.empty(), img=image)
            return []

        detections_by_class = self._group_by_class(detections)
        all_tracked: List[Dict] = []

        for class_name, class_detections in detections_by_class.items():
            tracker = self._get_or_create_ultralytics_tracker(class_name)
            results = DetectionResults.from_detections(class_detections)
            tracked_array = tracker.update(results, img=image)
            all_tracked.extend(
                self._associate_ultralytics_tracks(tracked_array, class_detections, class_name)
            )

        active_classes = set(detections_by_class.keys())
        for class_name, tracker in self.trackers.items():
            if class_name not in active_classes:
                tracker.update(DetectionResults.empty(), img=image)

        return all_tracked

    def _get_or_create_sort_tracker(self, class_name: str) -> Sort:
        if class_name not in self.trackers:
            self.trackers[class_name] = Sort(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_threshold=self.iou_threshold,
            )
        return self.trackers[class_name]

    def _get_or_create_ultralytics_tracker(self, class_name: str):
        if class_name not in self.trackers:
            params = self._tracker_params_dict()
            args = build_tracker_args(params, self.algorithm)
            tracker = self._create_ultralytics_tracker(args)
            self.trackers[class_name] = tracker
        return self.trackers[class_name]

    def _tracker_params_dict(self) -> Dict[str, Any]:
        return {
            "tracking_frame_rate": self.frame_rate,
            "track_high_thresh": self.track_high_thresh,
            "track_low_thresh": self.track_low_thresh,
            "track_buffer": self.track_buffer,
            "match_thresh": self.match_thresh,
            "new_track_thresh": self.new_track_thresh,
            "fuse_score": self.fuse_score,
        }

    @staticmethod
    def _create_ultralytics_tracker(args):
        algo = args.tracker_type
        if algo == "botsort":
            from app.services.trackers.bot_sort import BOTSORT
            return BOTSORT(args)
        if algo == "ocsort":
            from app.services.trackers.oc_sort import OCSORT
            return OCSORT(args)
        if algo == "fasttrack":
            from app.services.trackers.fast_tracker import FASTTracker
            return FASTTracker(args)
        if algo == "deepocsort":
            from app.services.trackers.deep_oc_sort import DeepOCSORT
            return DeepOCSORT(args)
        if algo == "tracktrack":
            from app.services.trackers.track_tracker import TRACKTRACK
            return TRACKTRACK(args)
        from app.services.trackers.byte_tracker import BYTETracker
        return BYTETracker(args)

    @staticmethod
    def _empty_ultralytics_results() -> DetectionResults:
        return DetectionResults.empty()

    @staticmethod
    def _group_by_class(detections: List[Dict]) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for detection in detections:
            class_name = detection.get("class_name", "unknown")
            grouped.setdefault(class_name, []).append(detection)
        return grouped

    @staticmethod
    def _detections_to_sort_array(detections: List[Dict]) -> np.ndarray:
        dets = []
        for detection in detections:
            bbox = detection.get("bbox", [])
            confidence = detection.get("confidence", 0.0)
            if len(bbox) >= 4:
                dets.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence])
        return np.array(dets) if dets else np.empty((0, 5))

    def _associate_sort_tracks(
        self,
        tracked_objects: np.ndarray,
        detections: List[Dict],
        class_name: str,
    ) -> List[Dict]:
        tracked_detections: List[Dict] = []
        used_detection_indices = set()

        for tracked_obj in tracked_objects:
            track_bbox = tracked_obj[:4].tolist()
            sort_track_id = int(tracked_obj[4])
            global_track_id = self._get_global_track_id(class_name, sort_track_id)

            best_detection = None
            best_iou = 0.0
            best_idx = -1

            for idx, detection in enumerate(detections):
                if idx in used_detection_indices:
                    continue
                bbox = detection.get("bbox", [])
                if len(bbox) >= 4:
                    iou = self._calculate_iou(track_bbox, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_detection = detection
                        best_idx = idx

            if best_detection and best_iou > 0.05:
                tracked_detection = best_detection.copy()
                tracked_detection["track_id"] = global_track_id
                tracked_detection["class_track_id"] = sort_track_id
                tracked_detections.append(tracked_detection)
                used_detection_indices.add(best_idx)

        return tracked_detections

    def _associate_ultralytics_tracks(
        self,
        tracked_array: np.ndarray,
        detections: List[Dict],
        class_name: str,
    ) -> List[Dict]:
        if tracked_array is None or len(tracked_array) == 0:
            return []

        tracked_detections: List[Dict] = []
        used_detection_indices = set()

        for tracked_row in tracked_array:
            if len(tracked_row) < 5:
                continue
            track_bbox = tracked_row[:4].tolist()
            local_track_id = int(tracked_row[4])
            global_track_id = self._get_global_track_id(class_name, local_track_id)

            best_detection = None
            best_iou = 0.0
            best_idx = -1

            for idx, detection in enumerate(detections):
                if idx in used_detection_indices:
                    continue
                bbox = detection.get("bbox", [])
                if len(bbox) >= 4:
                    iou = self._calculate_iou(track_bbox, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_detection = detection
                        best_idx = idx

            if best_detection and best_iou > 0.05:
                tracked_detection = best_detection.copy()
                tracked_detection["track_id"] = global_track_id
                tracked_detection["class_track_id"] = local_track_id
                tracked_detections.append(tracked_detection)
                used_detection_indices.add(best_idx)

        return tracked_detections

    @staticmethod
    def _get_global_track_id(class_name: str, local_track_id: int) -> int:
        class_hash = abs(hash(class_name)) % 1000
        return class_hash * 10000 + local_track_id

    @staticmethod
    def _calculate_iou(box1: List[float], box2: List[float]) -> float:
        try:
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            if x2 <= x1 or y2 <= y1:
                return 0.0

            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _bbox_center(bbox: List[float]) -> Optional[Tuple[float, float]]:
        if len(bbox) < 4:
            return None
        x1, y1, x2, y2 = bbox[:4]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _history_idle_limit(self) -> int:
        if self.algorithm == "sort":
            return max(self.max_age, 1)
        return max(self.track_buffer, 1)

    def _record_and_attach_histories(self, tracked_detections: List[Dict]) -> List[Dict]:
        """记录每个 track_id 的历史中心点，并附加到检测结果"""
        self._frame_index += 1

        if not self.enable_track_history or self.tracking_history_frames <= 0:
            self._cleanup_stale_histories(set())
            return tracked_detections

        active_track_ids = set()
        result: List[Dict] = []

        for detection in tracked_detections:
            track_id = detection.get("track_id")
            if track_id is None:
                result.append(detection)
                continue

            active_track_ids.add(track_id)
            center = self._bbox_center(detection.get("bbox", []))
            if center is None:
                result.append(detection)
                continue

            history = self._track_histories.get(track_id)
            if history is None:
                history = deque(maxlen=self.tracking_history_frames)
                self._track_histories[track_id] = history

            history.append(center)
            self._track_last_seen[track_id] = self._frame_index

            tracked_detection = detection.copy()
            tracked_detection["track_history"] = list(history)
            result.append(tracked_detection)

        self._cleanup_stale_histories(active_track_ids)
        return result

    def _cleanup_stale_histories(self, active_track_ids: set) -> None:
        idle_limit = self._history_idle_limit()
        stale_ids = [
            track_id
            for track_id, last_seen in self._track_last_seen.items()
            if track_id not in active_track_ids
            and self._frame_index - last_seen > idle_limit
        ]
        for track_id in stale_ids:
            self._track_histories.pop(track_id, None)
            self._track_last_seen.pop(track_id, None)

    def get_track_histories(self) -> Dict[int, List[Tuple[float, float]]]:
        """获取当前所有 track_id 的历史中心点"""
        return {
            track_id: list(history)
            for track_id, history in self._track_histories.items()
        }

    def reset(self) -> None:
        self.trackers = {}
        self._frame_index = 0
        self._track_histories.clear()
        self._track_last_seen.clear()
        logger.info("所有跟踪器状态已重置 [algorithm=%s]", self.algorithm)

    def reset_class(self, class_name: str) -> None:
        if class_name in self.trackers:
            del self.trackers[class_name]
            logger.info("类别 '%s' 的跟踪器已重置", class_name)

    def get_tracker_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "algorithm": self.algorithm,
            "total_classes": len(self.trackers),
            "classes": {},
        }

        if self.algorithm == "sort":
            info.update({
                "max_age": self.max_age,
                "min_hits": self.min_hits,
                "iou_threshold": self.iou_threshold,
            })
            for class_name, tracker in self.trackers.items():
                info["classes"][class_name] = {
                    "frame_count": getattr(tracker, "frame_count", 0),
                    "active_tracks": len(getattr(tracker, "trackers", [])),
                }
        else:
            info.update({
                "frame_rate": self.frame_rate,
                "track_high_thresh": self.track_high_thresh,
                "track_low_thresh": self.track_low_thresh,
                "track_buffer": self.track_buffer,
                "match_thresh": self.match_thresh,
                "new_track_thresh": self.new_track_thresh,
            })
            for class_name, tracker in self.trackers.items():
                info["classes"][class_name] = {
                    "frame_id": getattr(tracker, "frame_id", 0),
                    "active_tracks": len(getattr(tracker, "tracked_stracks", [])),
                }

        return info
