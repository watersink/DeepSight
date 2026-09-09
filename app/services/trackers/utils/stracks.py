# Shared helpers for operating on lists of track objects across trackers.

from __future__ import annotations

import numpy as np

from ..basetrack import TrackState
from . import matching

__all__ = (
    "joint_stracks",
    "merge_track_pools",
    "multi_gmc",
    "parse_bboxes",
    "remove_duplicate_stracks",
    "sub_stracks",
)


def merge_track_pools(tracker, activated, refind, lost, removed, removed_buffer: int = 1000) -> None:
    tracker.tracked_stracks = [t for t in tracker.tracked_stracks if t.state == TrackState.Tracked]
    tracker.tracked_stracks = joint_stracks(tracker.tracked_stracks, activated)
    tracker.tracked_stracks = joint_stracks(tracker.tracked_stracks, refind)
    tracker.lost_stracks = sub_stracks(tracker.lost_stracks, tracker.tracked_stracks)
    tracker.lost_stracks.extend(lost)
    tracker.lost_stracks = sub_stracks(tracker.lost_stracks, tracker.removed_stracks)
    tracker.tracked_stracks, tracker.lost_stracks = remove_duplicate_stracks(
        tracker.tracked_stracks, tracker.lost_stracks
    )
    tracker.removed_stracks.extend(removed)
    if len(tracker.removed_stracks) > removed_buffer:
        tracker.removed_stracks = tracker.removed_stracks[-removed_buffer:]


def parse_bboxes(results) -> np.ndarray:
    bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
    return np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)


def joint_stracks(atracks, btracks):
    a_ids = {t.track_id for t in atracks}
    return atracks + [t for t in btracks if t.track_id not in a_ids]


def sub_stracks(atracks, btracks):
    btrack_ids = {t.track_id for t in btracks}
    return [t for t in atracks if t.track_id not in btrack_ids]


def remove_duplicate_stracks(atracks, btracks, dup_thresh: float = 0.15):
    pdist = matching.iou_distance(atracks, btracks)
    pairs = np.where(pdist < dup_thresh)
    dupa, dupb = [], []
    for p, q in zip(*pairs):
        timep = atracks[p].frame_id - atracks[p].start_frame
        timeq = btracks[q].frame_id - btracks[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    dupa_set, dupb_set = set(dupa), set(dupb)
    resa = [t for i, t in enumerate(atracks) if i not in dupa_set]
    resb = [t for i, t in enumerate(btracks) if i not in dupb_set]
    return resa, resb


def multi_gmc(stracks, H: np.ndarray) -> None:
    if not stracks:
        return
    multi_mean = np.asarray([st.mean.copy() for st in stracks])
    multi_covariance = np.asarray([st.covariance for st in stracks])

    R = H[:2, :2]
    R8x8 = np.kron(np.eye(4, dtype=np.float32), R)
    t = H[:2, 2]

    for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
        mean = R8x8.dot(mean)
        mean[:2] += t
        cov = R8x8.dot(cov).dot(R8x8.transpose())
        stracks[i].mean = mean
        stracks[i].covariance = cov
