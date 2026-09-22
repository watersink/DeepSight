"""摄像头位姿巡检公共算法：模板 I0 + 当前帧 It → 由粗到细估计平移。

设计目标
========
每一轮对 (I0, It) 做四级检测；**任一级否决就提前结束**，省算力。
本模块只产出「本轮几何结论」（skip / normal / fail + shift_px）；
连续确认、冷却、告警落库由调用方（如 camera_shift_detector）完成。

与 tilt 的边界
==============
- shift：只看平移量 d（四角 warp 平均位移）
- tilt：看旋转角 / 透视项（可共用同一次 getSnap 与 H，本文件可复用）
- 模板均来自外部；调度可共用一次截图，两技能各自读结果

一轮判定表
==========
阶段        条件                              结果
截图        getSnap / 解码失败                 跳过，不告警（调用方捕获）
L1 粗筛     SSIM 很高                          skip → 正常
L1 粗筛     SSIM 极低 + 边缘几乎没了           fail（遮挡/失焦，非挪移）
L2 匹配     点数 < N_min                       fail / 跳过
L3 H        inlier_ratio 过低                  fail / 跳过
L4 平移     解出可靠 d（是否告警由调用方比阈） normal + shift_px

时序例子（间隔 5s，确认 3 次，在 skill 层）：
  t=0    d=10  < 阈值 → streak=0
  t=5    d=90  > 阈值 → streak=1
  t=10   d=95  > 阈值 → streak=2
  t=15   d=92  > 阈值 → streak=3 → 告警
  冷却中即使超阈也不推
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PoseDetectResult:
    """单轮几何检测结果。

    status:
      - skip:   L1 粗筛判定「高度一致」，未跑特征（省算力）
      - normal: L2–L4 解出可靠 H；平移/转角由调用方各自比阈
      - fail:   解码外的否决（遮挡/失焦/匹配不足/内点差）
    """

    status: str  # skip | normal | fail
    message: str
    shift_px: float = 0.0
    rotation_deg: float = 0.0
    perspective_score: float = 0.0
    ssim: Optional[float] = None
    match_count: int = 0
    inlier_ratio: float = 0.0
    detail: Optional[Dict[str, Any]] = None


def _wrap_angle_deg(deg: float) -> float:
    """把角度差归一到 [-180, 180]。"""
    d = float(deg)
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return d


def metrics_from_homography(
    h_mat: np.ndarray, image_shape: Tuple[int, int]
) -> Dict[str, Any]:
    """从单应 H 提取平移 / 转角 / 透视强度（shift 与 tilt 共用）。

    - shift_px: 模板四角 warp 后平均位移
    - rotation_deg: 四条边方向角变化的绝对值平均
    - perspective_score: 对边长度比变化之和（越大越像俯仰/透视偏离）
    """
    h, w = int(image_shape[0]), int(image_shape[1])
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(
        -1, 1, 2
    )
    warped = cv2.perspectiveTransform(corners, h_mat)
    c0 = corners.reshape(-1, 2)
    c1 = warped.reshape(-1, 2)
    deltas = c1 - c0
    shift_px = float(np.mean(np.linalg.norm(deltas, axis=1)))

    idx_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    angle_diffs: List[float] = []
    len0: List[float] = []
    len1: List[float] = []
    for i, j in idx_pairs:
        e0 = c0[j] - c0[i]
        e1 = c1[j] - c1[i]
        a0 = float(np.degrees(np.arctan2(e0[1], e0[0])))
        a1 = float(np.degrees(np.arctan2(e1[1], e1[0])))
        angle_diffs.append(abs(_wrap_angle_deg(a1 - a0)))
        len0.append(float(np.linalg.norm(e0)) + 1e-6)
        len1.append(float(np.linalg.norm(e1)) + 1e-6)

    rotation_deg = float(np.mean(angle_diffs))
    # 对边：上/下、左/右 的长度比变化
    ratio0_tb = len0[0] / len0[2]
    ratio1_tb = len1[0] / len1[2]
    ratio0_lr = len0[1] / len0[3]
    ratio1_lr = len1[1] / len1[3]
    perspective_score = float(abs(ratio1_tb - ratio0_tb) + abs(ratio1_lr - ratio0_lr))

    return {
        "shift_px": shift_px,
        "rotation_deg": rotation_deg,
        "perspective_score": perspective_score,
        "corner_shifts": deltas.tolist(),
        "edge_angle_diffs": angle_diffs,
        "homography": h_mat.tolist(),
    }


# ---------------------------------------------------------------------------
# L0 输入：加载 / 解码（失败则本轮作废，不报挪移）
# ---------------------------------------------------------------------------


def load_image_bgr(source: str) -> np.ndarray:
    """加载外部校准模板 I0（本地路径或 http(s) URL）。"""
    raw = (source or "").strip()
    if not raw:
        raise ValueError("校准模板地址为空")
    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"}:
        req = Request(raw, headers={"User-Agent": "DeepSight-CameraShift/1.0"})
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(raw, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"无法加载校准模板: {raw[:160]}")
    return image


def decode_jpeg_bgr(data: bytes) -> np.ndarray:
    """解码 ZLM getSnap 返回的 JPEG → It(BGR)。

    失败抛异常：调用方应本轮作废（截图失败），**不**报位置挪移。
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("截图解码失败")
    return image


def _resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    side = max(h, w)
    if side <= max_side or max_side <= 0:
        return image
    scale = max_side / float(side)
    return cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


# ---------------------------------------------------------------------------
# L0 预处理（必做、很轻）：灰度 + 统一尺度 → I0', It'
# ---------------------------------------------------------------------------


def preprocess_pair(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    max_side: int = 960,
) -> Tuple[np.ndarray, np.ndarray]:
    """第 0 级：输入与预处理。

    - 转灰度
    - 最长边缩放到同一尺度（默认 960，亦可 640）
    - 分辨率不一致时把 It resize 到与 I0 同尺寸，便于后续比像素位移

    输出：对齐后的 (I0', It')，供 L1–L4 使用。
    """
    ref = reference
    cur = current
    if ref.ndim == 3:
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    if cur.ndim == 3:
        cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    ref = _resize_max_side(ref, max_side)
    # 当前图缩放到与模板同尺寸，便于四角位移用像素解释
    if cur.shape[:2] != ref.shape[:2]:
        cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    return ref, cur


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """简化 SSIM（单通道），用于 L1 粗筛。阈值宜偏松，避免光照变化误杀。"""
    a_f = a.astype(np.float64)
    b_f = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a_f, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b_f, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a_f * a_f, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b_f * b_f, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a_f * b_f, (11, 11), 1.5) - mu_ab
    num = (2 * mu_ab + c1) * (2 * sigma_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    ssim_map = num / (den + 1e-12)
    return float(np.clip(ssim_map.mean(), -1.0, 1.0))


def _edge_density(gray: np.ndarray) -> float:
    """边缘密度：遮挡/失焦时 |e0−et| 往往很大且 et 接近 0。"""
    edges = cv2.Canny(gray, 80, 160)
    return float(np.count_nonzero(edges)) / float(edges.size or 1)


# ---------------------------------------------------------------------------
# L1 → L4：由粗到细；任一级否决提前 return
# ---------------------------------------------------------------------------


def estimate_shift_px(
    reference_gray: np.ndarray,
    current_gray: np.ndarray,
    *,
    ssim_skip: float = 0.92,
    ssim_fail: float = 0.15,
    min_match_count: int = 30,
    min_inlier_ratio: float = 0.35,
    orb_features: int = 1200,
) -> PoseDetectResult:
    """兼容旧名：等价于 estimate_pose（同时给出平移与转角）。"""
    return estimate_pose(
        reference_gray,
        current_gray,
        ssim_skip=ssim_skip,
        ssim_fail=ssim_fail,
        min_match_count=min_match_count,
        min_inlier_ratio=min_inlier_ratio,
        orb_features=orb_features,
    )


def estimate_pose(
    reference_gray: np.ndarray,
    current_gray: np.ndarray,
    *,
    ssim_skip: float = 0.92,
    ssim_fail: float = 0.15,
    min_match_count: int = 30,
    min_inlier_ratio: float = 0.35,
    orb_features: int = 1200,
) -> PoseDetectResult:
    """对已对齐的 (I0', It') 做 L1–L4，估计平移 d 与转角/透视。

    前提：调用方已完成 L0（decode + preprocess_pair）。
    本函数不处理连续确认 / 冷却；shift / tilt 技能各自比阈维护 streak。
    """
    if reference_gray is None or current_gray is None:
        return PoseDetectResult("fail", "输入图像为空")

    # ----- L1 粗筛（便宜）：过滤明显正常 / 明显异常，大部分轮次到此结束 -----
    ssim = compute_ssim(reference_gray, current_gray)
    if ssim >= ssim_skip:
        return PoseDetectResult(
            "skip",
            f"画面高度一致 SSIM={ssim:.3f}",
            shift_px=0.0,
            rotation_deg=0.0,
            perspective_score=0.0,
            ssim=ssim,
        )

    e0 = _edge_density(reference_gray)
    e1 = _edge_density(current_gray)
    if ssim <= ssim_fail and e1 < max(0.005, e0 * 0.2):
        return PoseDetectResult(
            "fail",
            f"疑似遮挡或失焦 SSIM={ssim:.3f} edge={e1:.4f}",
            ssim=ssim,
            detail={"edge_ref": e0, "edge_cur": e1},
        )

    # ----- L2 特征匹配（中等代价）：ORB + Lowe ratio(0.75) -----
    orb = cv2.ORB_create(nfeatures=int(orb_features))
    kp0, des0 = orb.detectAndCompute(reference_gray, None)
    kp1, des1 = orb.detectAndCompute(current_gray, None)
    if des0 is None or des1 is None or len(kp0) < 8 or len(kp1) < 8:
        return PoseDetectResult("fail", "特征点不足", ssim=ssim)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = matcher.knnMatch(des0, des1, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < int(min_match_count):
        return PoseDetectResult(
            "fail",
            f"匹配点不足 {len(good)}<{min_match_count}",
            ssim=ssim,
            match_count=len(good),
        )

    # ----- L3 解单应矩阵 -----
    pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    h_mat, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
    if h_mat is None or mask is None:
        return PoseDetectResult(
            "fail",
            "单应矩阵求解失败",
            ssim=ssim,
            match_count=len(good),
        )

    inliers = int(mask.ravel().sum())
    inlier_ratio = inliers / float(len(good))
    if inlier_ratio < float(min_inlier_ratio):
        return PoseDetectResult(
            "fail",
            f"内点比例过低 {inlier_ratio:.2f}<{min_inlier_ratio}",
            ssim=ssim,
            match_count=len(good),
            inlier_ratio=inlier_ratio,
        )

    # ----- L4：平移 + 转角/透视（shift / tilt 技能各取所需）-----
    metrics = metrics_from_homography(h_mat, reference_gray.shape[:2])
    shift_px = float(metrics["shift_px"])
    rotation_deg = float(metrics["rotation_deg"])
    perspective_score = float(metrics["perspective_score"])

    return PoseDetectResult(
        "normal",
        (
            f"平移约 {shift_px:.1f}px / 转角约 {rotation_deg:.1f}° "
            f"/ 透视 {perspective_score:.3f}"
        ),
        shift_px=shift_px,
        rotation_deg=rotation_deg,
        perspective_score=perspective_score,
        ssim=ssim,
        match_count=len(good),
        inlier_ratio=inlier_ratio,
        detail=metrics,
    )
