"""将技能检测结果包装为 ultralytics Results-like 对象，供各跟踪器复用"""
from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np


class DetectionResults:
    """轻量检测容器，兼容 parse_bboxes / BYTETracker.update 所需接口"""

    __slots__ = ("conf", "cls", "xywh", "_xyxy")

    def __init__(
        self,
        conf: np.ndarray,
        cls: np.ndarray,
        xywh: np.ndarray,
    ):
        self.conf = np.asarray(conf, dtype=np.float32)
        self.cls = np.asarray(cls, dtype=np.float32)
        self.xywh = np.asarray(xywh, dtype=np.float32)
        self._xyxy = self._compute_xyxy(self.xywh)

    @property
    def xyxy(self) -> np.ndarray:
        return self._xyxy

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: Union[np.ndarray, slice]) -> "DetectionResults":
        if isinstance(mask, slice):
            idx = mask
            return DetectionResults(self.conf[idx], self.cls[idx], self.xywh[idx])
        mask = np.asarray(mask, dtype=bool)
        return DetectionResults(self.conf[mask], self.cls[mask], self.xywh[mask])

    @staticmethod
    def _compute_xyxy(xywh: np.ndarray) -> np.ndarray:
        if len(xywh) == 0:
            return np.zeros((0, 4), dtype=np.float32)
        out = xywh.copy()
        out[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
        out[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
        out[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
        out[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DetectionResults":
        conf = data.get("conf", [])
        cls_arr = data.get("cls", [])
        xywh = data.get("xywh", [])
        if len(conf) == 0:
            return cls.empty()
        return cls(
            np.asarray(conf, dtype=np.float32),
            np.asarray(cls_arr, dtype=np.float32),
            np.asarray(xywh, dtype=np.float32),
        )

    @classmethod
    def from_detections(cls, detections: list) -> "DetectionResults":
        xywh_list = []
        conf_list = []
        cls_list = []
        for detection in detections:
            bbox = detection.get("bbox", [])
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox[:4]
            w = x2 - x1
            h = y2 - y1
            xywh_list.append([x1 + w / 2, y1 + h / 2, w, h])
            conf_list.append(float(detection.get("confidence", 0.0)))
            cls_list.append(float(detection.get("class_id", 0)))
        if not xywh_list:
            return cls.empty()
        return cls(
            np.asarray(conf_list, dtype=np.float32),
            np.asarray(cls_list, dtype=np.float32),
            np.asarray(xywh_list, dtype=np.float32),
        )

    @classmethod
    def empty(cls) -> "DetectionResults":
        return cls(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.zeros((0, 4), dtype=np.float32),
        )
