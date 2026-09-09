# Ultralytics YOLO tracker (adapted for app.services.trackers)

import logging
from typing import Any

import numpy as np

from .basetrack import BaseTrack, TrackState
from .utils import matching
from .utils.kalman_filter import KalmanFilterXYAH
from .utils.ops import xywh2ltwh
from .utils.stracks import joint_stracks, merge_track_pools, multi_gmc, parse_bboxes

LOGGER = logging.getLogger(__name__)


class STrack(BaseTrack):
    shared_kalman = KalmanFilterXYAH()

    def __init__(self, xywh: np.ndarray, score: float, cls: Any):
        super().__init__()
        assert len(xywh) in {5, 6}, f"expected 5 or 6 values but got {len(xywh)}"
        self._tlwh = np.asarray(xywh2ltwh(xywh[:4]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.idx = xywh[-1]
        self.angle = xywh[4] if len(xywh) == 6 else None

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id: int):
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id: int, new_id: bool = False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def update(self, new_track, frame_id: int):
        self.frame_id = frame_id
        self.tracklet_len += 1
        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self.cls = new_track.cls
        self.angle = new_track.angle
        self.idx = new_track.idx

    def convert_coords(self, tlwh):
        return self.tlwh_to_xyah(tlwh)

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def xyxy(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @property
    def xywh(self):
        ret = np.asarray(self.tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    @property
    def xywha(self):
        if self.angle is None:
            return self.xywh
        return np.concatenate([self.xywh, self.angle[None]])

    @property
    def result(self):
        coords = self.xyxy if self.angle is None else self.xywha
        return [*coords.tolist(), self.track_id, self.score, self.cls, self.idx]


class BYTETracker:
    track_class = STrack

    def __init__(self, args=None, frame_rate: int = 30):
        if args is None or isinstance(args, int):
            from .tracker_args import build_tracker_args
            fr = int(args if isinstance(args, int) else frame_rate)
            args = build_tracker_args({"tracking_frame_rate": fr}, "bytetrack")
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.args = args
        self.max_frames_lost = getattr(args, "max_frames_lost", args.track_buffer)
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()

    def update(self, results, img=None, feats=None, **kwargs):
        self.frame_id += 1
        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []
        results_high, results_low, mask_high, mask_low = self._split_detections(results)
        detections = self.init_track(results_high, self._input_for(img, feats, mask_high))
        detections_second = self.init_track(results_low, self._input_for(img, feats, mask_low))
        unconfirmed, tracked_stracks = self._split_tracked()
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)
        self._pre_first_associate(strack_pool, unconfirmed, img, results_high)
        u_track, u_detection = self._first_association(strack_pool, detections, activated_stracks, refind_stracks)
        u_track, u_detection = self._post_first_association(
            strack_pool, detections, u_track, u_detection, activated_stracks, refind_stracks
        )
        self._second_association(strack_pool, u_track, detections_second, activated_stracks, refind_stracks, lost_stracks)
        u_detection, detections = self._unconfirmed_association(
            unconfirmed, u_detection, detections, activated_stracks, removed_stracks
        )
        self._init_new_tracks(u_detection, detections, activated_stracks, refind_stracks)
        self._remove_stale_lost(removed_stracks)
        merge_track_pools(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks)
        return self._format_output()

    def _split_detections(self, results):
        scores = results.conf
        remain_inds = scores >= self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_below_high = scores < self.args.track_high_thresh
        return results[remain_inds], results[inds_low & inds_below_high], remain_inds, inds_low & inds_below_high

    def _input_for(self, img, feats, mask):
        if feats is not None and len(feats):
            return feats[mask]
        return img

    def _split_tracked(self):
        unconfirmed, tracked = [], []
        for track in self.tracked_stracks:
            (unconfirmed if not track.is_activated else tracked).append(track)
        return unconfirmed, tracked

    def _pre_first_associate(self, strack_pool, unconfirmed, img, results_high):
        if hasattr(self, "gmc") and img is not None:
            try:
                warp = self.gmc.apply(img, results_high.xyxy)
            except Exception as e:
                LOGGER.warning("GMC failed, falling back to identity: %s", e)
                warp = np.eye(2, 3)
            multi_gmc(strack_pool, warp)
            multi_gmc(unconfirmed, warp)

    def _first_association(self, strack_pool, detections, activated, refind):
        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)
        self._apply_matches(matches, strack_pool, detections, activated, refind)
        return u_track, u_detection

    def _post_first_association(self, strack_pool, detections, u_track, u_detection, activated, refind):
        return u_track, u_detection

    def _apply_matches(self, matches, pool, detections, activated, refind):
        for itracked, idet in matches:
            self._apply_match(pool[itracked], detections[idet], activated, refind)

    def _apply_match(self, track, det, activated, refind):
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id)
            activated.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind.append(track)

    def _second_association(self, strack_pool, u_track, detections_second, activated, refind, lost):
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        if r_tracked_stracks and detections_second:
            dists = matching.iou_distance(r_tracked_stracks, detections_second)
            if self.args.fuse_score:
                dists = matching.fuse_score(dists, detections_second)
            matches, u_track, _ = matching.linear_assignment(dists, thresh=0.5)
            self._apply_matches(matches, r_tracked_stracks, detections_second, activated, refind)
        else:
            u_track = list(range(len(r_tracked_stracks)))
        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost.append(track)

    def _unconfirmed_association(self, unconfirmed, u_detection, detections, activated, removed):
        detections = [detections[i] for i in u_detection]
        if not unconfirmed:
            return list(range(len(detections))), detections
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed.append(track)
        return u_detection, detections

    def _init_new_tracks(self, u_detection, detections, activated, refind=None):
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated.append(track)

    def _remove_stale_lost(self, removed):
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_frames_lost:
                track.mark_removed()
                removed.append(track)

    def _format_output(self):
        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def get_kalmanfilter(self):
        return KalmanFilterXYAH()

    def init_track(self, results, img=None):
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        return [self.track_class(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks, detections):
        dists = matching.iou_distance(tracks, detections)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        return dists

    def multi_predict(self, tracks):
        STrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        STrack.reset_id()

    def reset(self):
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.kalman_filter = self.get_kalmanfilter()
        self.reset_id()
