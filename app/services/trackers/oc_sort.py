# Ultralytics YOLO tracker (adapted for app.services.trackers)

from __future__ import annotations

from typing import Any

import numpy as np

from .utils.ops import xyxy2ltwh
from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.stracks import parse_bboxes


class OCSortTrack(STrack):
    """Track object for OC-SORT with observation-centric state management."""

    def __init__(self, xywh: np.ndarray, score: float, cls: Any, delta_t: int = 3):
        super().__init__(xywh, score, cls)
        self.last_observation = np.array([-1, -1, -1, -1], dtype=np.float32)
        self.observations: dict[int, np.ndarray] = {}
        self.velocity: np.ndarray | None = None
        self.delta_t = delta_t
        self._saved_mean: np.ndarray | None = None
        self._saved_covariance: np.ndarray | None = None

    def activate(self, kalman_filter, frame_id: int) -> None:
        super().activate(kalman_filter, frame_id)
        self.last_observation = self.xyxy.copy()
        self.observations[frame_id] = self.xyxy.copy()
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()

    def update(self, new_track: STrack, frame_id: int) -> None:
        obs = new_track.xyxy.copy()
        self.last_observation = obs
        self.observations[frame_id] = obs
        self._prune_observations()
        super().update(new_track, frame_id)
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()
        self.velocity = self._compute_velocity()

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        obs = new_track.xyxy.copy()
        self.last_observation = obs
        self.observations[frame_id] = obs
        super().re_activate(new_track, frame_id, new_id)
        self._saved_mean = self.mean.copy()
        self._saved_covariance = self.covariance.copy()
        self.velocity = self._compute_velocity()

    @staticmethod
    def _xyxy_center(xyxy: np.ndarray) -> np.ndarray:
        return np.array([(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2])

    def _prune_observations(self) -> None:
        max_keep = self.delta_t + 2
        if len(self.observations) <= max_keep:
            return
        sorted_frames = sorted(self.observations.keys())
        for frame in sorted_frames[:-max_keep]:
            del self.observations[frame]

    def _compute_velocity(self) -> np.ndarray | None:
        if len(self.observations) < 2:
            return None

        current_frame = max(self.observations.keys())
        current_center = self._xyxy_center(self.observations[current_frame])

        prev_obs = None
        for frame in sorted(self.observations.keys(), reverse=True):
            if frame < current_frame - self.delta_t + 1:
                prev_obs = self.observations[frame]
                break

        if prev_obs is None:
            earliest_frame = min(self.observations.keys())
            if earliest_frame == current_frame:
                return None
            prev_obs = self.observations[earliest_frame]

        direction = current_center - self._xyxy_center(prev_obs)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (direction / norm).astype(np.float32)

    def apply_oru(self, new_observation_xyxy: np.ndarray, current_frame_id: int) -> None:
        if self._saved_mean is None or not self.observations:
            return

        last_frame = max(self.observations.keys())
        gap = current_frame_id - last_frame
        if gap <= 1:
            return

        self.mean = self._saved_mean.copy()
        self.covariance = self._saved_covariance.copy()
        last_obs = self.observations[last_frame]

        for t in range(1, gap):
            alpha = t / gap
            virtual_xyxy = (1 - alpha) * last_obs + alpha * new_observation_xyxy
            virtual_xyah = self.tlwh_to_xyah(xyxy2ltwh(virtual_xyxy))
            self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
            self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, virtual_xyah)

        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)


class OCSORT(BYTETracker):
    """OC-SORT multi-object tracker with observation-centric association."""

    track_class = OCSortTrack

    def __init__(self, args: Any = None, frame_rate: int = 30):
        if args is None or isinstance(args, int):
            from .tracker_args import build_tracker_args
            fr = int(args if isinstance(args, int) else frame_rate)
            args = build_tracker_args({"tracking_frame_rate": fr}, "ocsort")
        super().__init__(args)
        self.delta_t = getattr(args, "delta_t", 3)
        self.inertia = getattr(args, "inertia", 0.2)
        self.use_byte = getattr(args, "use_byte", False)

    def init_track(self, results, img: np.ndarray | None = None) -> list[OCSortTrack]:
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        return [OCSortTrack(xywh, s, c, self.delta_t) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def _fuse_appearance(
        self,
        dists: np.ndarray,
        tracks: list[OCSortTrack],
        detections: list[OCSortTrack],
        iou_dists: np.ndarray | None = None,
    ) -> np.ndarray:
        return dists

    def get_dists(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray:
        iou_dists = matching.iou_distance(tracks, detections)
        dists = matching.fuse_score(iou_dists, detections) if self.args.fuse_score else iou_dists.copy()
        dists = dists + self.inertia * self._velocity_direction_cost(tracks, detections)
        return self._fuse_appearance(dists, tracks, detections, iou_dists=iou_dists)

    def _ocr_associate(
        self,
        tracks: list[OCSortTrack],
        dets: list[OCSortTrack],
        activated: list[OCSortTrack],
        refind: list[OCSortTrack],
    ) -> tuple[list[int], list[int]]:
        if not tracks or not dets:
            return list(range(len(tracks))), list(range(len(dets)))
        ocr_dists = self._ocr_distance(tracks, dets)
        if self.args.fuse_score:
            ocr_dists = matching.fuse_score(ocr_dists, dets)
        ocr_dists = self._fuse_appearance(ocr_dists, tracks, dets)
        matches, u_track, u_det = matching.linear_assignment(ocr_dists, thresh=self.args.match_thresh)
        for itracked, idet in matches:
            track, det = tracks[itracked], dets[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated.append(track)
            else:
                track.apply_oru(det.xyxy, self.frame_id)
                track.re_activate(det, self.frame_id, new_id=False)
                refind.append(track)
        return list(u_track), list(u_det)

    def _post_first_association(
        self,
        strack_pool: list[OCSortTrack],
        detections: list[OCSortTrack],
        u_track: list[int],
        u_detection: list[int],
        activated: list[OCSortTrack],
        refind: list[OCSortTrack],
    ) -> tuple[list[int], list[int]]:
        ocr_dets = [detections[i] for i in u_detection]
        if not ocr_dets:
            return u_track, u_detection

        tracked = [i for i in u_track if strack_pool[i].state == TrackState.Tracked]
        other = [i for i in u_track if strack_pool[i].state != TrackState.Tracked]

        u_t1, u_d1 = self._ocr_associate([strack_pool[i] for i in tracked], ocr_dets, activated, refind)
        remaining = [ocr_dets[j] for j in u_d1]
        u_t2, u_d2 = self._ocr_associate([strack_pool[i] for i in other], remaining, activated, refind)

        u_track = [tracked[i] for i in u_t1] + [other[i] for i in u_t2]
        u_detection = [u_detection[u_d1[j]] for j in u_d2]
        return u_track, u_detection

    def _second_association(
        self,
        strack_pool: list[OCSortTrack],
        u_track: list[int],
        detections_second: list[OCSortTrack],
        activated: list[OCSortTrack],
        refind: list[OCSortTrack],
        lost: list[OCSortTrack],
    ) -> None:
        if not self.use_byte:
            for i in u_track:
                track = strack_pool[i]
                if track.state == TrackState.Tracked:
                    track.mark_lost()
                    lost.append(track)
            return
        super()._second_association(strack_pool, u_track, detections_second, activated, refind, lost)

    def _velocity_direction_cost(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray:
        cost = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        if cost.size == 0:
            return cost

        det_centers = np.array([OCSortTrack._xyxy_center(det.xyxy) for det in detections], dtype=np.float32)

        for i, track in enumerate(tracks):
            if track.velocity is None or track.last_observation[0] < 0:
                continue
            track_center = OCSortTrack._xyxy_center(track.last_observation)
            directions = det_centers - track_center
            norms = np.linalg.norm(directions, axis=1)
            valid = norms > 1e-6
            if not valid.any():
                continue
            directions[valid] /= norms[valid, None]
            dots = np.clip(directions[valid] @ track.velocity, -1.0, 1.0)
            cost[i, valid] = np.arccos(dots) / np.pi

        return cost

    def _ocr_distance(self, tracks: list[OCSortTrack], detections: list[OCSortTrack]) -> np.ndarray:
        if tracks and tracks[0].angle is not None:
            atlbrs = [t.xywha for t in tracks]
            btlbrs = [d.xywha for d in detections]
        else:
            atlbrs = [t.last_observation if t.last_observation[0] >= 0 else t.xyxy for t in tracks]
            btlbrs = [d.xyxy for d in detections]
        return matching.iou_distance(atlbrs, btlbrs)
