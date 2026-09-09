# Ultralytics YOLO tracker (adapted for app.services.trackers)

from __future__ import annotations

from typing import Any

import numpy as np

from .byte_tracker import STrack
from .oc_sort import OCSORT, OCSortTrack
from .utils import matching
from .utils.gmc import GMC
from .utils.reid import build_encoder, smooth_feature
from .utils.stracks import parse_bboxes


class DeepOCSortTrack(OCSortTrack):
    """Track object for Deep OC-SORT with appearance features."""

    def __init__(
        self,
        xywh: np.ndarray,
        score: float,
        cls: Any,
        delta_t: int = 3,
        feat: np.ndarray | None = None,
        alpha_fixed_emb: float = 0.95,
        det_thresh: float = 0.25,
    ):
        super().__init__(xywh, score, cls, delta_t)
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha_fixed_emb = alpha_fixed_emb
        self.det_thresh = det_thresh
        if feat is not None:
            self.update_features(feat, score)

    def update_features(self, feat: np.ndarray, score: float | None = None) -> None:
        if score is not None and score > self.det_thresh:
            trust = (score - self.det_thresh) / max(1 - self.det_thresh, 1e-9)
            alpha = self.alpha_fixed_emb + (1 - self.alpha_fixed_emb) * (1 - trust)
        else:
            alpha = 1.0
        curr, smooth = smooth_feature(feat, self.smooth_feat, alpha)
        if curr is not None:
            self.curr_feat, self.smooth_feat = curr, smooth

    def update(self, new_track: STrack, frame_id: int) -> None:
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score)
        super().update(new_track, frame_id)

    def re_activate(self, new_track: STrack, frame_id: int, new_id: bool = False) -> None:
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat, new_track.score)
        super().re_activate(new_track, frame_id, new_id)

    @staticmethod
    def multi_gmc(stracks: list["DeepOCSortTrack"], H: np.ndarray) -> None:
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])

        R = H[:2, :2]
        t = H[:2, 2]
        R8x8 = np.eye(8, dtype=np.float32)
        R8x8[:2, :2] = R
        R8x8[4:6, 4:6] = R

        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            mean = R8x8.dot(mean)
            mean[:2] += t
            cov = R8x8.dot(cov).dot(R8x8.transpose())
            stracks[i].mean = mean
            stracks[i].covariance = cov

            if stracks[i].last_observation[0] >= 0:
                obs = stracks[i].last_observation
                cx, cy = DeepOCSortTrack._xyxy_center(obs)
                w, h = obs[2] - obs[0], obs[3] - obs[1]
                new_c = R @ np.array([cx, cy]) + t
                stracks[i].last_observation = np.array(
                    [new_c[0] - w / 2, new_c[1] - h / 2, new_c[0] + w / 2, new_c[1] + h / 2],
                    dtype=np.float32,
                )


class DeepOCSORT(OCSORT):
    """Deep OC-SORT: OC-SORT enhanced with appearance features and GMC."""

    track_class = DeepOCSortTrack

    def __init__(self, args: Any = None, frame_rate: int = 30):
        if args is None or isinstance(args, int):
            from .tracker_args import build_tracker_args
            fr = int(args if isinstance(args, int) else frame_rate)
            args = build_tracker_args({"tracking_frame_rate": fr}, "deepocsort")
        super().__init__(args)
        self.gmc = GMC(method=getattr(args, "gmc_method", "sparseOptFlow"))
        self.proximity_thresh = getattr(args, "proximity_thresh", 0.5)
        self.appearance_thresh = getattr(args, "appearance_thresh", 0.75)
        self.alpha_fixed_emb = getattr(args, "alpha_fixed_emb", 0.95)
        self.encoder = build_encoder(getattr(args, "with_reid", False), getattr(args, "model", "auto"))

    def init_track(self, results, img: np.ndarray | None = None) -> list[DeepOCSortTrack]:
        if len(results) == 0:
            return []
        bboxes = parse_bboxes(results)

        if self.encoder is not None and img is not None:
            features = self.encoder(img, bboxes)
            return [
                DeepOCSortTrack(
                    xywh, s, c, self.delta_t, feat=f,
                    alpha_fixed_emb=self.alpha_fixed_emb,
                    det_thresh=self.args.track_high_thresh,
                )
                for (xywh, s, c, f) in zip(bboxes, results.conf, results.cls, features)
            ]
        return [
            DeepOCSortTrack(
                xywh, s, c, self.delta_t,
                alpha_fixed_emb=self.alpha_fixed_emb,
                det_thresh=self.args.track_high_thresh,
            )
            for (xywh, s, c) in zip(bboxes, results.conf, results.cls)
        ]

    def _pre_first_associate(
        self,
        strack_pool: list[DeepOCSortTrack],
        unconfirmed: list[DeepOCSortTrack],
        img: np.ndarray | None,
        results_high: Any,
    ) -> None:
        if img is None:
            return
        try:
            warp = self.gmc.apply(img, results_high.xyxy if len(results_high) else np.empty((0, 4)))
        except Exception:
            warp = np.eye(2, 3)
        DeepOCSortTrack.multi_gmc(strack_pool, warp)
        DeepOCSortTrack.multi_gmc(unconfirmed, warp)

    def _fuse_appearance(
        self,
        dists: np.ndarray,
        tracks: list[DeepOCSortTrack],
        detections: list[DeepOCSortTrack],
        iou_dists: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.encoder is None or not tracks or not detections:
            return dists
        emb_dists = matching.embedding_distance(tracks, detections) / 2.0
        emb_dists[emb_dists > (1 - self.appearance_thresh)] = 1.0
        if iou_dists is not None:
            emb_dists[iou_dists > (1 - self.proximity_thresh)] = 1.0
        return np.minimum(dists, emb_dists)

    def reset(self) -> None:
        super().reset()
        self.gmc.reset_params()
