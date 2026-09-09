# Ultralytics YOLO tracker (adapted for app.services.trackers)

from __future__ import annotations

import logging
from collections import deque
from typing import Any

import numpy as np

from .basetrack import TrackState
from .bot_sort import BOTrack
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH
from .utils.metrics import bbox_ioa
from .utils.reid import build_encoder, smooth_feature
from .utils.stracks import joint_stracks, merge_track_pools, multi_gmc, parse_bboxes

LOGGER = logging.getLogger(__name__)

_CORNER_DX_IDX = np.array([0, 0, 2, 2])
_CORNER_DY_IDX = np.array([1, 3, 1, 3])


def attach_raw_preds_hook(predictor) -> None:
    """Optional YOLO predictor hook; not used in TrackerService."""
    return


def compute_dets_del(predictor) -> list | None:
    """Return loose-NMS recovered detections; unavailable outside YOLO predict pipeline."""
    return None


def _hmiou_distance(tracks_a: list, tracks_b: list) -> tuple[np.ndarray, np.ndarray]:
    n, m = len(tracks_a), len(tracks_b)
    if n == 0 or m == 0:
        return np.zeros((n, m), dtype=np.float32), np.ones((n, m), dtype=np.float32)
    boxes_a = np.ascontiguousarray([track.xyxy for track in tracks_a], dtype=np.float32)
    boxes_b = np.ascontiguousarray([track.xyxy for track in tracks_b], dtype=np.float32)
    iou_sim = bbox_ioa(boxes_a, boxes_b, iou=True)
    h_over = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    h_union = np.maximum(boxes_a[:, 3:4], boxes_b[:, 3:4].T) - np.minimum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    h_iou = np.clip(h_over / (h_union + 1e-9), 0, 1)
    return iou_sim, 1.0 - h_iou * iou_sim


def _angle_distance(tracks: list, dets: list, frame_id: int, delta_t: int = 3) -> np.ndarray:
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float32)
    track_boxes = np.stack([track.get_history_box(frame_id, delta_t) for track in tracks])
    det_boxes = np.stack([det.xyxy for det in dets])
    deltas = det_boxes[None] - track_boxes[:, None]
    dx = deltas[:, :, _CORNER_DX_IDX]
    dy = deltas[:, :, _CORNER_DY_IDX]
    norms = np.sqrt(dx * dx + dy * dy) + 1e-5
    dx /= norms
    dy /= norms
    track_velocities = np.stack([track.velocity for track in tracks])
    dot = track_velocities[:, None, :, 0] * dx + track_velocities[:, None, :, 1] * dy
    dist = np.abs(np.arccos(np.clip(dot, -1, 1))).mean(axis=-1) / np.pi
    return dist * np.array([det.score for det in dets])[None]


def _confidence_distance(tracks: list, dets: list) -> np.ndarray:
    if len(tracks) == 0 or len(dets) == 0:
        return np.ones((len(tracks), len(dets)), dtype=np.float32)
    track_prev_scores = np.array([track.prev_score for track in tracks])
    track_curr_scores = np.array([track.score for track in tracks])
    track_proj_scores = track_curr_scores + (track_curr_scores - track_prev_scores)
    det_scores = np.array([det.score for det in dets])
    return np.abs(track_proj_scores[:, None] - det_scores[None])


def _iterative_associate(cost: np.ndarray, match_thr: float, reduce_step: float = 0.05) -> tuple[list]:
    matches = []
    cost = cost.copy()
    while cost.shape[0] > 0 and cost.shape[1] > 0:
        nearest_det = np.argmin(cost, axis=1)
        nearest_track = np.argmin(cost, axis=0)
        new_matches = [
            [track_idx, nearest_det[track_idx]]
            for track_idx in range(cost.shape[0])
            if nearest_track[nearest_det[track_idx]] == track_idx
            and cost[track_idx, nearest_det[track_idx]] < match_thr
        ]
        if not new_matches:
            break
        matches.extend(new_matches)
        for track_idx, det_idx in new_matches:
            cost[track_idx, :] = np.inf
            cost[:, det_idx] = np.inf
        match_thr -= reduce_step
    matched_tracks = {track_idx for track_idx, _ in matches}
    matched_dets = {det_idx for _, det_idx in matches}
    unmatched_tracks = [i for i in range(cost.shape[0]) if i not in matched_tracks]
    unmatched_dets = [i for i in range(cost.shape[1]) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


def _track_aware_nms(tracks: list, dets: list, tai_thr: float, new_track_thresh: float) -> list[bool]:
    if not dets:
        return []
    scores = np.array([det.score for det in dets])
    allow = scores > new_track_thresh
    n_tracks, n_dets = len(tracks), len(dets)
    if n_tracks + n_dets < 2:
        return allow.tolist()
    boxes = np.ascontiguousarray([obj.xyxy for obj in tracks + dets], dtype=np.float32)
    iou = bbox_ioa(boxes, boxes, iou=True)

    if n_tracks:
        allow &= iou[n_tracks:, :n_tracks].max(axis=1) <= tai_thr

    det_iou = iou[n_tracks:, n_tracks:]
    order = scores.argsort()[::-1]
    for i in order:
        if not allow[i]:
            continue
        suppress = det_iou[i] > tai_thr
        suppress[i] = False
        allow[suppress] = False

    return allow.tolist()


def _cosine_distance(tracks: list, dets: list) -> np.ndarray:
    if not tracks or not dets:
        return np.ones((len(tracks), len(dets)), dtype=np.float32)
    tfeat = [t.smooth_feat if t.smooth_feat is not None else t.curr_feat for t in tracks]
    dfeat = [d.curr_feat for d in dets]
    dim = next((f.shape[0] for f in (*tfeat, *dfeat) if f is not None), 128)
    zeros = np.zeros(dim, dtype=np.float32)
    T = np.stack([f if f is not None else zeros for f in tfeat])
    D = np.stack([f if f is not None else zeros for f in dfeat])
    valid = np.array([f is not None for f in tfeat])[:, None] & np.array([f is not None for f in dfeat])[None, :]
    return np.where(valid, np.clip(1 - T @ D.T, 0, 1), np.nan).astype(np.float32)


class TTSTrack(BOTrack):
    """Single-object track for TrackTrack."""

    min_track_len = 3
    _alpha = 0.95
    _delta_t = 3

    def __init__(self, xywh: np.ndarray, score: float, cls: Any, feat: np.ndarray | None = None):
        super().__init__(xywh, score, cls)
        self.prev_score = score
        self.velocity = np.zeros((4, 2), dtype=np.float32)
        self._history: deque[tuple[int, np.ndarray]] = deque(maxlen=self._delta_t + 1)
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat: np.ndarray) -> None:
        beta = self._alpha + (1 - self._alpha) * (1 - self.score)
        curr, smooth = smooth_feature(feat, self.smooth_feat, beta)
        if curr is not None:
            self.curr_feat, self.smooth_feat = curr, smooth

    def get_history_box(self, frame_id: int, dt: int) -> np.ndarray:
        target = frame_id - dt
        for fid, box in self._history:
            if fid == target:
                return box.copy()
        if self._history:
            return self._history[-1][1].copy()
        return self.xyxy.copy()

    def activate(self, kalman_filter: KalmanFilterXYWH, frame_id: int) -> None:
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = kalman_filter.initiate(self.convert_coords(self._tlwh))
        self._history.append((frame_id, self.xyxy.copy()))
        self.tracklet_len = 0
        self.state = TrackState.New
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = self.start_frame = frame_id

    def re_activate(self, new_track, frame_id: int, new_id: bool = False) -> None:
        self.prev_score = self.score
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh), confidence=new_track.score
        )
        self._history.append((frame_id, self.xyxy.copy()))
        self.score = new_track.score
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.cls, self.angle, self.idx = new_track.cls, new_track.angle, new_track.idx

    def update(self, new_track, frame_id: int) -> None:
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.prev_score = self.score
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh), confidence=new_track.score
        )
        self._history.append((frame_id, new_track.xyxy.copy()))

        velocity = np.zeros((4, 2), dtype=np.float32)
        curr_box = new_track.xyxy
        for dt in range(1, self._delta_t + 1):
            delta = curr_box - self.get_history_box(frame_id, dt)
            dx, dy = delta[_CORNER_DX_IDX], delta[_CORNER_DY_IDX]
            norm = np.sqrt(dx * dx + dy * dy) + 1e-5
            velocity += np.stack([dx / norm, dy / norm], axis=-1) / dt
        self.velocity = velocity / self._delta_t

        self.score = new_track.score
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        if self.state == TrackState.Tracked or self.tracklet_len >= self.min_track_len:
            self.state = TrackState.Tracked
            self.is_activated = True
        self.cls, self.angle, self.idx = new_track.cls, new_track.angle, new_track.idx


class TRACKTRACK:
    """Multi-object tracker implementing Track-Perspective Association."""

    def __init__(self, args=None, frame_rate: int = 30):
        if args is None or isinstance(args, int):
            from .tracker_args import build_tracker_args
            fr = int(args if isinstance(args, int) else frame_rate)
            args = build_tracker_args({"tracking_frame_rate": fr}, "tracktrack")
        self.tracked_stracks: list[TTSTrack] = []
        self.lost_stracks: list[TTSTrack] = []
        self.removed_stracks: list[TTSTrack] = []
        self.frame_id = 0
        self.args = args
        self.max_time_lost = getattr(args, "max_frames_lost", args.track_buffer)
        self.kalman_filter = KalmanFilterXYWH()

        self.match_thr = getattr(args, "match_thresh", 0.7)
        self.lost_match_thr = getattr(args, "lost_match_thr", 0.0)
        self.penalty_p = getattr(args, "penalty_p", 0.2)
        self.penalty_q = getattr(args, "penalty_q", 0.4)
        self.reduce_step = getattr(args, "reduce_step", 0.05)
        self.iou_weight = getattr(args, "iou_weight", 0.5)
        self.reid_weight = getattr(args, "reid_weight", 0.5)
        self.conf_weight = getattr(args, "conf_weight", 0.1)
        self.angle_weight = getattr(args, "angle_weight", 0.05)
        self.tai_thr = getattr(args, "tai_thr", 0.55)
        self.new_track_thresh = getattr(args, "new_track_thresh", 0.7)
        self.min_track_len = getattr(args, "min_track_len", 3)

        self.gmc = GMC(method=getattr(args, "gmc_method", "sparseOptFlow"))
        self.encoder = build_encoder(getattr(args, "with_reid", False), getattr(args, "model", "auto"))

    @classmethod
    def setup_predictor(cls, predictor):
        if getattr(predictor.args, "task", "detect") in {"detect", "obb"}:
            attach_raw_preds_hook(predictor)

    @classmethod
    def compute_frame_extras(cls, predictor):
        return compute_dets_del(predictor)

    def _cost_matrix(self, tracks: list[TTSTrack], dets: list[TTSTrack]) -> np.ndarray:
        iou_sim, hmiou_dist = _hmiou_distance(tracks, dets)
        if self.encoder is not None:
            cos = _cosine_distance(tracks, dets)
            cost = np.where(np.isnan(cos), hmiou_dist, self.iou_weight * hmiou_dist + self.reid_weight * cos)
        else:
            cost = hmiou_dist
        cost += self.conf_weight * _confidence_distance(tracks, dets)
        cost += self.angle_weight * _angle_distance(tracks, dets, self.frame_id)
        if iou_sim.size > 0:
            cost[iou_sim <= 0.10] = 1.0
        return np.clip(cost, 0, 1)

    def _apply_gmc(self, img: np.ndarray, detections: list, pools: list[list[TTSTrack]]) -> None:
        try:
            warp = self.gmc.apply(img, [det.xyxy for det in detections])
        except Exception as e:
            LOGGER.warning("GMC failed, falling back to identity: %s", e)
            warp = np.eye(2, 3)
        for pool in pools:
            multi_gmc(pool, warp)

    def update(self, results, img: np.ndarray | None = None, dets_del=None, **kwargs) -> np.ndarray:
        self.frame_id += 1
        activated, refind, lost, removed = [], [], [], []

        scores = results.conf
        boxes = parse_bboxes(results)
        high_mask = scores >= self.args.track_high_thresh
        low_mask = (scores > self.args.track_low_thresh) & (scores < self.args.track_high_thresh)

        def _new_track(box, score, cls, feat=None):
            track = TTSTrack(box, score, cls, feat) if feat is not None else TTSTrack(box, score, cls)
            track.min_track_len = self.min_track_len
            return track

        high_boxes, high_scores, high_cls = boxes[high_mask], scores[high_mask], results.cls[high_mask]
        feats = kwargs.get("feats")
        use_native = getattr(self.args, "model", "auto") == "auto"
        encoder_input = None
        if self.encoder is not None and len(high_boxes) > 0:
            if use_native:
                encoder_input = feats[high_mask] if (feats is not None and len(feats)) else None
            elif img is not None:
                encoder_input = img

        if encoder_input is not None:
            features = self.encoder(encoder_input, high_boxes)
            dets_high = [_new_track(b, s, c, f) for b, s, c, f in zip(high_boxes, high_scores, high_cls, features)]
        else:
            dets_high = [_new_track(b, s, c) for b, s, c in zip(high_boxes, high_scores, high_cls)]
        dets_low = [_new_track(b, s, c) for b, s, c in zip(boxes[low_mask], scores[low_mask], results.cls[low_mask])]

        dets_recovered: list[TTSTrack] = []
        if dets_del is not None:
            del_xywh, del_conf, del_cls = dets_del
            mask = del_conf > self.args.track_high_thresh
            if mask.any():
                del_boxes = np.concatenate([del_xywh[mask], -np.ones((mask.sum(), 1))], axis=-1)
                dets_recovered = [_new_track(b, s, c) for b, s, c in zip(del_boxes, del_conf[mask], del_cls[mask])]

        unconfirmed, tracked = [], []
        for track in self.tracked_stracks:
            (unconfirmed if not track.is_activated else tracked).append(track)
        pool = joint_stracks(tracked, self.lost_stracks)

        if img is not None:
            self._apply_gmc(img, dets_high, [pool, unconfirmed])
        TTSTrack.multi_predict(pool)

        all_dets = dets_high + dets_low + dets_recovered
        n_high, n_low = len(dets_high), len(dets_low)
        cost = self._cost_matrix(pool, all_dets)
        if cost.shape[1] > n_high:
            cost[:, n_high : n_high + n_low] += self.penalty_p
        if dets_recovered:
            cost[:, n_high + n_low :] += self.penalty_q
        cost = np.clip(cost, 0, 1)

        matches, unmatched_tracks, unmatched_dets = _iterative_associate(cost, self.match_thr, self.reduce_step)
        for track_idx, det_idx in matches:
            track, det = pool[track_idx], all_dets[det_idx]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)
        for track_idx in unmatched_tracks:
            track = pool[track_idx]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

        leftover = [all_dets[i] for i in unmatched_dets if i < n_high]
        if unconfirmed and leftover:
            uc_cost = self._cost_matrix(unconfirmed, leftover)
            uc_matches, uc_unmatched_tracks, uc_unmatched_dets = _iterative_associate(
                uc_cost, self.match_thr, self.reduce_step
            )
            for track_idx, det_idx in uc_matches:
                unconfirmed[track_idx].update(leftover[det_idx], self.frame_id)
                activated.append(unconfirmed[track_idx])
            for track_idx in uc_unmatched_tracks:
                unconfirmed[track_idx].mark_removed()
                removed.append(unconfirmed[track_idx])
            leftover = [leftover[i] for i in uc_unmatched_dets]
        else:
            for track in unconfirmed:
                track.mark_removed()
                removed.append(track)

        if self.lost_match_thr > 0 and leftover:
            unmatched_lost = [t for t in pool if t.state == TrackState.Lost and t not in lost]
            if unmatched_lost:
                lost_cost = self._cost_matrix(unmatched_lost, leftover)
                lost_matches, _, lost_unmatched = _iterative_associate(lost_cost, self.lost_match_thr, self.reduce_step)
                for track_idx, det_idx in lost_matches:
                    unmatched_lost[track_idx].re_activate(leftover[det_idx], self.frame_id, new_id=False)
                    refind.append(unmatched_lost[track_idx])
                leftover = [leftover[i] for i in lost_unmatched]

        active = [track for track in self.tracked_stracks if track.state == TrackState.Tracked] + activated
        for det, ok in zip(leftover, _track_aware_nms(active, leftover, self.tai_thr, self.new_track_thresh)):
            if ok:
                det.activate(self.kalman_filter, self.frame_id)
                activated.append(det)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        merge_track_pools(self, activated, refind, lost, removed)
        return np.asarray(
            [track.result for track in self.tracked_stracks if track.is_activated and track.frame_id == self.frame_id],
            dtype=np.float32,
        )

    def reset(self) -> None:
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = KalmanFilterXYWH()
        TTSTrack.reset_id()
        self.gmc.reset_params()
