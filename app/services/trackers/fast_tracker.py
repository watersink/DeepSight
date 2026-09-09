# Ultralytics YOLO tracker (adapted for app.services.trackers)

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .utils.metrics import bbox_ioa
from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.stracks import parse_bboxes


class FastSTrack(STrack):
    """Single-object track for FastTracker with occlusion-aware state."""

    def __init__(self, xywh: np.ndarray, score: float, cls: Any, history_len: int = 16):
        super().__init__(xywh, score, cls)
        self.mean_history: deque = deque(maxlen=history_len)
        self.not_matched = 0
        self.is_occluded = False
        self.occluded_len = 0
        self.last_occluded_frame = -1
        self.was_recently_occluded = False

    def _push_history(self):
        if self.mean is not None:
            self.mean_history.append((self.mean.copy(), self.covariance.copy()))

    def activate(self, kalman_filter, frame_id: int):
        super().activate(kalman_filter, frame_id)
        self._push_history()

    def re_activate(self, new_track, frame_id: int, new_id: bool = False):
        super().re_activate(new_track, frame_id, new_id=new_id)
        self.is_occluded = False
        self.occluded_len = 0
        self.not_matched = 0
        self.was_recently_occluded = False
        self.last_occluded_frame = -1
        self._push_history()

    def update(self, new_track, frame_id: int):
        super().update(new_track, frame_id)
        self._push_history()


class FASTTracker(BYTETracker):
    """Occlusion-aware ByteTrack-style multi-object tracker."""

    track_class = FastSTrack

    def __init__(self, args=None, frame_rate: int = 30):
        if args is None or isinstance(args, int):
            from .tracker_args import build_tracker_args
            fr = int(args if isinstance(args, int) else frame_rate)
            args = build_tracker_args({"tracking_frame_rate": fr}, "fasttrack")
        super().__init__(args)
        self.reset_velocity_offset_occ = int(getattr(args, "reset_velocity_offset_occ", 5))
        self.reset_pos_offset_occ = int(getattr(args, "reset_pos_offset_occ", 3))
        self.enlarge_bbox_occ = float(getattr(args, "enlarge_bbox_occ", 1.1))
        self.dampen_motion_occ = float(getattr(args, "dampen_motion_occ", 0.5))
        self.active_occ_to_lost_thresh = int(getattr(args, "active_occ_to_lost_thresh", 10))
        self.init_iou_suppress = float(getattr(args, "init_iou_suppress", 0.7))
        self.occ_cover_thresh = float(getattr(args, "occ_cover_thresh", 0.7))
        self.occ_reappear_window = int(getattr(args, "occ_reappear_window", 40))
        self._history_len = max(self.reset_velocity_offset_occ, self.reset_pos_offset_occ) + 4

    def init_track(self, results, img: np.ndarray | None = None) -> list[FastSTrack]:
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        return [
            FastSTrack(xywh, s, c, history_len=self._history_len)
            for (xywh, s, c) in zip(bboxes, results.conf, results.cls)
        ]

    def _apply_match(self, track: STrack, det: STrack, activated: list[STrack], refind: list[STrack]) -> None:
        super()._apply_match(track, det, activated, refind)
        track.is_occluded = False
        track.not_matched = 0
        track.occluded_len = 0

    def _second_association(
        self,
        strack_pool: list[STrack],
        u_track: list[int],
        detections_second: list[STrack],
        activated: list[STrack],
        refind: list[STrack],
        lost: list[STrack],
    ) -> None:
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if r_tracked_stracks and detections_second:
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            if self.args.fuse_score:
                dists = matching.fuse_score(dists, detections_second)
            matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
            self._apply_matches(matches, r_tracked_stracks, detections_second, activated, refind)
        else:
            u_track = list(range(len(r_tracked_stracks)))
        self._handle_occlusions(r_tracked_stracks, u_track, activated, lost)

    def _init_new_tracks(
        self,
        u_detection: list[int],
        detections: list[STrack],
        activated: list[STrack],
        refind: list[STrack] | None = None,
    ) -> None:
        active_boxes = [t.xyxy for t in activated if t.is_activated]
        if refind:
            active_boxes.extend(t.xyxy for t in refind if t.is_activated)
        active_boxes.extend(t.xyxy for t in self.tracked_stracks if t.state == TrackState.Tracked)
        suppress_on = self.init_iou_suppress < 1.0
        active_stack = (
            np.asarray(active_boxes, dtype=np.float32) if active_boxes else np.empty((0, 4), dtype=np.float32)
        )
        for inew in u_detection:
            det = detections[inew]
            if det.score < self.args.new_track_thresh:
                continue
            if suppress_on and len(active_stack):
                if bbox_ioa(det.xyxy[None, :], active_stack, iou=True).max() >= self.init_iou_suppress:
                    continue
            det.activate(self.kalman_filter, self.frame_id)
            activated.append(det)
            active_stack = np.concatenate([active_stack, det.xyxy[None, :]], axis=0)

    def _remove_stale_lost(self, removed: list[STrack]) -> None:
        for track in self.lost_stracks:
            recently_occluded = track.was_recently_occluded and (
                self.frame_id - track.last_occluded_frame <= self.occ_reappear_window
            )
            if not recently_occluded and (self.frame_id - track.end_frame > self.max_frames_lost):
                track.mark_removed()
                removed.append(track)

    def _format_output(self) -> np.ndarray:
        return np.asarray(
            [x.result for x in self.tracked_stracks if x.is_activated and x.frame_id == self.frame_id],
            dtype=np.float32,
        )

    def _handle_occlusions(self, r_tracked, u_track, activated_stracks, lost_stracks):
        if len(u_track) == 0:
            return

        active = [t for t in activated_stracks if t.is_activated and not t.is_occluded]
        if len(active):
            active_boxes = np.asarray([t.xyxy for t in active], dtype=np.float32)
            active_ids = np.asarray([t.track_id for t in active])
        else:
            active_boxes = np.empty((0, 4), dtype=np.float32)
            active_ids = np.empty((0,), dtype=np.int64)

        unmatched = [r_tracked[i] for i in u_track]
        unmatched_boxes = (
            np.asarray([t.xyxy for t in unmatched], dtype=np.float32)
            if unmatched
            else np.empty((0, 4), dtype=np.float32)
        )

        if active_boxes.size and unmatched_boxes.size:
            cov = bbox_ioa(active_boxes, unmatched_boxes)
            unm_ids = np.asarray([t.track_id for t in unmatched])
            same = active_ids[:, None] == unm_ids[None, :]
            cov[same] = 0.0
            max_cov = cov.max(axis=0)
        else:
            max_cov = np.zeros(len(unmatched), dtype=np.float32)

        for i, track in enumerate(unmatched):
            track.not_matched += 1

            if max_cov[i] > self.occ_cover_thresh and not track.is_occluded and track.state == TrackState.Tracked:
                track.is_occluded = True
                track.occluded_len = 1
                track.last_occluded_frame = self.frame_id
                track.was_recently_occluded = True

                hist = track.mean_history
                if track.mean is not None and hist:
                    if len(hist) >= self.reset_velocity_offset_occ:
                        prev_mean, _ = hist[-self.reset_velocity_offset_occ]
                        track.mean[4:8] = prev_mean[4:8]
                    if len(hist) >= self.reset_pos_offset_occ:
                        prev_mean, prev_cov = hist[-self.reset_pos_offset_occ]
                        track.mean[0:4] = prev_mean[0:4]
                        track.covariance = prev_cov.copy()
                    track.mean[3] *= self.enlarge_bbox_occ
                    track.mean[4:8] *= self.dampen_motion_occ
            elif track.is_occluded:
                track.occluded_len += 1

            if track.was_recently_occluded and (self.frame_id - track.last_occluded_frame > self.occ_reappear_window):
                track.was_recently_occluded = False

            if track.state != TrackState.Lost:
                if track.not_matched > 2 and (
                    not track.is_occluded or track.occluded_len > self.active_occ_to_lost_thresh
                ):
                    track.mark_lost()
                    lost_stracks.append(track)
