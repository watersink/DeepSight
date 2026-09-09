"""ReID 辅助函数（默认关闭 ReID，仅保留特征平滑逻辑）"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def build_encoder(with_reid: bool, model: str | None) -> Optional[Callable]:
    """构建 ReID 编码器；当前工程默认不启用 ReID 模型推理"""
    if not with_reid:
        return None
    if model == "auto":

        def _auto_encoder(feats, _dets):
            if isinstance(feats, np.ndarray):
                return [f for f in feats]
            return [f.cpu().numpy() for f in feats]

        return _auto_encoder
    return None


def smooth_feature(feat: np.ndarray, smooth: np.ndarray | None, alpha: float):
    norm = np.linalg.norm(feat)
    if norm < 1e-12:
        return None, smooth
    feat = feat / norm
    if smooth is None:
        return feat, feat.copy()
    smooth = alpha * smooth + (1 - alpha) * feat
    return feat, smooth / np.linalg.norm(smooth)
