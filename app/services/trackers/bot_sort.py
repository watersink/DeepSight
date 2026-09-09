from typing import Any

import numpy as np

from .basetrack import TrackState
from .byte_tracker import BYTETracker, STrack
from .utils import matching
from .utils.gmc import GMC
from .utils.kalman_filter import KalmanFilterXYWH
from .utils.reid import build_encoder, smooth_feature
from .utils.stracks import parse_bboxes


class BOTrack(STrack):
    shared_kalman = KalmanFilterXYWH()

    def __init__(self, xywh: np.ndarray, score: float, cls: int, feat=None):
        super().__init__(xywh, score, cls)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

    def update_features(self, feat):
        curr, smooth = smooth_feature(feat, self.smooth_feat, self.alpha)
        if curr is not None:
            self.curr_feat, self.smooth_feat = curr, smooth

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    def re_activate(self, new_track, frame_id: int, new_id: bool = False):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().re_activate(new_track, frame_id, new_id)

    def update(self, new_track, frame_id: int):
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        super().update(new_track, frame_id)

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @staticmethod
    def multi_predict(stracks):
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6] = 0
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = BOTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov

    def convert_coords(self, tlwh):
        return self.tlwh_to_xywh(tlwh)

    @staticmethod
    def tlwh_to_xywh(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret


class BOTSORT(BYTETracker):
    def __init__(self, args=None, frame_rate: int = 30):
        if args is None or isinstance(args, int):
            from .tracker_args import build_tracker_args
            fr = int(args if isinstance(args, int) else frame_rate)
            args = build_tracker_args({"tracking_frame_rate": fr}, "botsort")
        super().__init__(args)
        self.gmc = GMC(method=args.gmc_method)
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.encoder = build_encoder(args.with_reid, args.model)

    def get_kalmanfilter(self):
        return KalmanFilterXYWH()

    def init_track(self, results, img=None):
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder(img, bboxes)
            return [BOTrack(xywh, s, c, f) for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features_keep)]
        return [BOTrack(xywh, s, c) for (xywh, s, c) in zip(bboxes, results.conf, results.cls)]

    def get_dists(self, tracks, detections):
        dists = matching.iou_distance(tracks, detections)
        dists_mask = dists > (1 - self.proximity_thresh)
        if self.args.fuse_score:
            dists = matching.fuse_score(dists, detections)
        if self.args.with_reid and self.encoder is not None:
            emb_dists = matching.embedding_distance(tracks, detections) / 2.0
            emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
            emb_dists[dists_mask] = 1.0
            dists = np.minimum(dists, emb_dists)
        return dists

    def multi_predict(self, tracks):
        BOTrack.multi_predict(tracks)

    def reset(self):
        super().reset()
        self.gmc.reset_params()
