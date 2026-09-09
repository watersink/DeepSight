"""跟踪器参数构建（对齐 ultralytics cfg/trackers/*.yaml 默认值）"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional


def build_tracker_args(params: Optional[Dict[str, Any]] = None, algorithm: str = "bytetrack") -> SimpleNamespace:
    """从技能 params / TrackerService 配置构建跟踪器 args"""
    params = params or {}
    algo = str(algorithm or "bytetrack").lower()

    frame_rate = int(params.get("tracking_frame_rate", params.get("frame_rate", 30)))
    track_buffer = int(params.get("track_buffer", 30))

    args = SimpleNamespace(
        tracker_type=algo,
        track_high_thresh=float(params.get("track_high_thresh", 0.5)),
        track_low_thresh=float(params.get("track_low_thresh", 0.1)),
        new_track_thresh=float(params.get("new_track_thresh", 0.6)),
        track_buffer=track_buffer,
        match_thresh=float(params.get("match_thresh", 0.8)),
        fuse_score=bool(params.get("fuse_score", True)),
        frame_rate=frame_rate,
        max_frames_lost=max(1, int(frame_rate / 30.0 * track_buffer)),
        # BoT-SORT
        gmc_method=str(params.get("gmc_method", "sparseOptFlow")),
        proximity_thresh=float(params.get("proximity_thresh", 0.5)),
        appearance_thresh=float(params.get("appearance_thresh", 0.8)),
        with_reid=bool(params.get("with_reid", False)),
        model=str(params.get("reid_model", params.get("model", "auto"))),
        # OC-SORT
        delta_t=int(params.get("delta_t", 3)),
        inertia=float(params.get("inertia", 0.2)),
        use_byte=bool(params.get("use_byte", False)),
        # FastTracker
        reset_velocity_offset_occ=int(params.get("reset_velocity_offset_occ", 5)),
        reset_pos_offset_occ=int(params.get("reset_pos_offset_occ", 3)),
        enlarge_bbox_occ=float(params.get("enlarge_bbox_occ", 1.1)),
        dampen_motion_occ=float(params.get("dampen_motion_occ", 0.5)),
        active_occ_to_lost_thresh=int(params.get("active_occ_to_lost_thresh", 10)),
        occ_cover_thresh=float(params.get("occ_cover_thresh", 0.7)),
        occ_reappear_window=int(params.get("occ_reappear_window", 40)),
        init_iou_suppress=float(params.get("init_iou_suppress", 0.7)),
        # TrackTrack
        lost_match_thr=float(params.get("lost_match_thr", 0.0)),
        penalty_p=float(params.get("penalty_p", 0.2)),
        penalty_q=float(params.get("penalty_q", 0.4)),
        reduce_step=float(params.get("reduce_step", 0.05)),
        iou_weight=float(params.get("iou_weight", 0.5)),
        reid_weight=float(params.get("reid_weight", 0.5)),
        conf_weight=float(params.get("conf_weight", 0.1)),
        angle_weight=float(params.get("angle_weight", 0.05)),
        tai_thr=float(params.get("tai_thr", 0.55)),
        min_track_len=int(params.get("min_track_len", 3)),
        # Deep OC-SORT / TrackTrack
        alpha_fixed_emb=float(params.get("alpha_fixed_emb", 0.95)),
        det_thresh=float(params.get("det_thresh", 0.25)),
        min_box_area=float(params.get("min_box_area", 10)),
        aspect_thresh=float(params.get("aspect_thresh", 1.6)),
    )
    return args
