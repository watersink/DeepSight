"""技能注册与工厂 - 按名称创建技能实例"""
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type

from app.plugins.skills.boarding_detector_skill import BoardingDetectorSkill
from app.plugins.skills.camera_shift_detector_skill import CameraShiftDetectorSkill
from app.plugins.skills.camera_tilt_detector_skill import CameraTiltDetectorSkill
from app.plugins.skills.guoan_detector_skill import GuoanDetectorSkill
from app.plugins.skills.non_fixed_parking_boarding_detector_skill import (
    NonFixedParkingBoardingDetectorSkill,
)
from app.plugins.skills.person_count_detector26_skill import PersonCountDetector26Skill
from app.plugins.skills.person_count_detector_skill import PersonCountDetectorSkill
from app.plugins.skills.person_presence_detector26_skill import PersonPresenceDetector26Skill
from app.skills.skill_base import BaseSkill


class SkillNotFoundError(ValueError):
    def __init__(self, skill_name: str, available: List[str]):
        self.skill_name = skill_name
        self.available = available
        super().__init__(
            f"未知技能: {skill_name}，可用技能: {', '.join(available)}"
        )


# skill_name -> 技能类（新增技能在此注册）
_SKILL_CLASSES: Dict[str, Type[BaseSkill]] = {
    PersonCountDetectorSkill.DEFAULT_CONFIG["name"]: PersonCountDetectorSkill,
    PersonCountDetector26Skill.DEFAULT_CONFIG["name"]: PersonCountDetector26Skill,
    PersonPresenceDetector26Skill.DEFAULT_CONFIG["name"]: PersonPresenceDetector26Skill,
    BoardingDetectorSkill.DEFAULT_CONFIG["name"]: BoardingDetectorSkill,
    NonFixedParkingBoardingDetectorSkill.DEFAULT_CONFIG["name"]: (
        NonFixedParkingBoardingDetectorSkill
    ),
    CameraShiftDetectorSkill.DEFAULT_CONFIG["name"]: CameraShiftDetectorSkill,
    CameraTiltDetectorSkill.DEFAULT_CONFIG["name"]: CameraTiltDetectorSkill,
    GuoanDetectorSkill.DEFAULT_CONFIG["name"]: GuoanDetectorSkill,
}


# 算法管理详情弹窗：展示 params 中的业务/检测关键项，省略内部跟踪微调
_PARAM_LABELS: Dict[str, str] = {
    "classes": "检测类别",
    "conf_thres": "置信度阈值",
    "iou_thres": "NMS IoU 阈值",
    "max_det": "最大检测数",
    "input_size": "模型输入尺寸",
    "enable_default_sort_tracking": "启用目标跟踪",
    "tracking_algorithm": "跟踪算法",
    "tracking_frame_rate": "跟踪帧率",
    "tracking_max_age": "跟踪最大丢失帧",
    "tracking_min_hits": "确认跟踪最小命中",
    "tracking_iou_threshold": "跟踪 IoU 阈值",
    "model_version": "模型版本",
    "person_limit": "人数告警上限",
    "enter_count": "计数初始值",
    "gate_direction": "闸机方向",
    "cross_confirm_frames": "过线确认帧数",
    "corner_confirm_min": "过线角点确认数",
    "track_lost_keep_frames": "丢失续轨迹最大帧数",
    "cross_confirm_miss_tolerance": "过线确认容错帧",
    "count_line": "过线计数线",
    "bypass_line": "绕行线",
    "boarding_count": "上车人数告警阈值",
    "enable_helmet_filter": "安全帽区域过滤",
    "helmet_min_ratio": "安全帽最小占比",
    "roi_hold": "区域保持帧数",
    "door_expand": "车门区域扩展比例",
    "reference_image_url": "校准模板图片地址",
    "check_interval_sec": "检测间隔（秒）",
    "shift_threshold_px": "平移告警阈值（像素）",
    "tilt_threshold_deg": "转角告警阈值（度）",
    "perspective_threshold": "透视告警阈值",
    "confirm_count": "连续确认次数",
    "cooldown_sec": "告警冷却（秒）",
    "ssim_skip_threshold": "SSIM 跳过阈值",
    "ssim_fail_threshold": "SSIM 失效阈值",
    "min_match_count": "最少匹配点数",
    "min_inlier_ratio": "最低内点比例",
    "max_side": "处理最长边",
    "roi": "检测 ROI",
    "dark_pixel_threshold": "暗像素阈值",
    "clip_threshold": "裁剪阈值",
    "median_drop_ratio": "中位数下降比例",
    "min_median_drop_abs": "中位数最小下降",
    "dark_ratio_rise": "暗像素比上升",
    "block_drop_ratio": "分块下降比例",
    "block_darken_fraction": "变暗分块占比",
    "ignore_block_mean": "忽略分块均值",
    "consecutive_seconds": "连续过暗秒数",
    "exposure_drop_ratio": "曝光下降比例",
    "well_exposed_drop": "正常曝光下降",
    "calibrate_frames": "自动标定帧数",
    "sample_fps": "采样帧率",
    "baseline_path": "基准文件路径",
}

_SKIP_PARAM_KEYS = {
    "target_class_ids",
    "track_high_thresh",
    "track_low_thresh",
    "track_buffer",
    "match_thresh",
    "new_track_thresh",
    "tracking_history_frames",
    "enable_track_history",
    "enable_timing_log",
    "mine_code",
    "camera_code",
    "baseline",
    "single_frame_as_final",
}

_LINE_PARAM_KEYS = {"count_line", "bypass_line"}

# 与 app.services.tracker_service.TRACKING_ALGORITHMS / app.services.trackers 保持一致
_TRACKING_ALGORITHMS = (
    "sort",
    "bytetrack",
    "botsort",
    "ocsort",
    "fasttrack",
    "deepocsort",
    "tracktrack",
)
_TRACKING_ALGORITHM_LABELS: Dict[str, str] = {
    "sort": "SORT",
    "bytetrack": "ByteTrack",
    "botsort": "BoT-SORT",
    "ocsort": "OC-SORT",
    "fasttrack": "FastTracker",
    "deepocsort": "Deep OC-SORT",
    "tracktrack": "TrackTrack",
}


def _merge_form_fields(
    base: List[Any],
    extra: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = [item for item in base if isinstance(item, dict)]
    seen = {str(item.get("key")) for item in merged if item.get("key")}
    for item in extra:
        key = str(item.get("key") or "")
        if not key or key in seen:
            continue
        merged.append(item)
        seen.add(key)
    return merged


def tracking_form_fields(params: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """技能支持跟踪时，追加任务配置页的启用开关与算法下拉。"""
    if not isinstance(params, dict) or "enable_default_sort_tracking" not in params:
        return []
    algo = str(params.get("tracking_algorithm") or "sort").strip().lower()
    if algo not in _TRACKING_ALGORITHMS:
        algo = "sort"
    return [
        {
            "key": "enable_default_sort_tracking",
            "label": "启用跟踪",
            "type": "bool",
            "required": False,
            "default": bool(params.get("enable_default_sort_tracking", True)),
            "hint": "关闭后不做目标跟踪",
        },
        {
            "key": "tracking_algorithm",
            "label": "跟踪算法",
            "type": "select",
            "required": False,
            "default": algo,
            "options": [
                {
                    "value": name,
                    "label": _TRACKING_ALGORITHM_LABELS.get(name, name),
                }
                for name in _TRACKING_ALGORITHMS
            ],
            "hint": "仅在启用跟踪后生效",
        },
    ]


def _format_param_value(key: str, value: Any) -> str:
    if key in _LINE_PARAM_KEYS:
        if not value:
            return "未配置（在任务配置中绘制）"
        return "已配置"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(x) for x in value) if value else "—"
    if value is None or value == "":
        return "—"
    return str(value)


def skill_param_items(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从 DEFAULT_CONFIG.params 提取算法管理页展示的关键参数。"""
    params = cfg.get("params") or {}
    if not isinstance(params, dict):
        return []
    items: List[Dict[str, Any]] = []
    for key, value in params.items():
        if key in _SKIP_PARAM_KEYS:
            continue
        items.append(
            {
                "key": key,
                "label": _PARAM_LABELS.get(key, key),
                "value": _format_param_value(key, value),
            }
        )
    return items


def get_skill_run_mode(skill_name: str) -> str:
    """stream=实时视频；snapshot=周期截图。缺省 stream。"""
    cls = _SKILL_CLASSES.get(skill_name)
    if cls is None:
        return "stream"
    mode = str((cls.DEFAULT_CONFIG or {}).get("run_mode") or "stream").strip().lower()
    return mode if mode in {"stream", "snapshot"} else "stream"


def is_snapshot_skill(skill_name: str) -> bool:
    return get_skill_run_mode(skill_name) == "snapshot"


def list_available_skills() -> List[Dict[str, Any]]:
    """返回可选择的技能列表（去重）"""
    seen = set()
    result = []
    for cls in _SKILL_CLASSES.values():
        key = id(cls)
        if key in seen:
            continue
        seen.add(key)
        cfg = cls.DEFAULT_CONFIG
        tracking_fields = tracking_form_fields(cfg.get("params"))
        if str(cfg.get("run_mode") or "stream").strip().lower() == "snapshot":
            tracking_fields = []
        result.append({
            "skill_name": cfg.get("name"),
            "name_zh": cfg.get("name_zh"),
            "description": cfg.get("description"),
            "type": cfg.get("type"),
            "run_mode": str(cfg.get("run_mode") or "stream"),
            "version": cfg.get("version"),
            "required_models": cfg.get("required_models", []),
            "cover_image": cfg.get("cover_image")
            or f"/skills/{cfg.get('name')}.png",
            # 任务配置表单按此声明渲染；未声明则不展示计数/画线等专用项
            "form_fields": _merge_form_fields(
                cfg.get("form_fields") or [],
                tracking_fields,
            ),
            "params": skill_param_items(cfg),
            "alert_definitions": cfg.get("alert_definitions") or [],
        })
    return result


def resolve_skill_class(skill_name: str) -> Type[BaseSkill]:
    cls = _SKILL_CLASSES.get(skill_name)
    if cls is None:
        raise SkillNotFoundError(skill_name, sorted(_SKILL_CLASSES.keys()))
    return cls


def merge_skill_config_overrides(
    skill_config: Optional[Dict[str, Any]] = None,
    *,
    enter_count: Optional[int] = None,
    gate_direction: Optional[str] = None,
    count_line: Optional[Dict[str, Any]] = None,
    bypass_line: Optional[Dict[str, Any]] = None,
    mine_code: Optional[str] = None,
    camera_code: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    将请求级可选参数合并进 skill_config.params。

    仅合并调用方显式传入的项；技能若不使用对应参数可忽略。
    """
    overrides: Dict[str, Any] = {}
    if enter_count is not None:
        overrides["enter_count"] = int(enter_count)
    if gate_direction is not None:
        direction = str(gate_direction).strip().upper()
        if direction not in {"IN", "OUT"}:
            raise ValueError(f"gate_direction 仅支持 IN/OUT，收到: {gate_direction}")
        overrides["gate_direction"] = direction
    if count_line is not None:
        overrides["count_line"] = count_line
    if bypass_line is not None:
        overrides["bypass_line"] = bypass_line
    if mine_code is not None:
        overrides["mine_code"] = str(mine_code).strip()
    if camera_code is not None:
        overrides["camera_code"] = str(camera_code).strip()

    if not overrides:
        return skill_config

    merged = deepcopy(skill_config) if skill_config else {}
    params = merged.setdefault("params", {})
    if not isinstance(params, dict):
        params = {}
        merged["params"] = params
    params.update(overrides)
    return merged


def create_skill(
    skill_name: str,
    skill_config: Optional[Dict[str, Any]] = None,
) -> BaseSkill:
    """
    根据技能名称创建实例。

    Args:
        skill_name: 技能标识，如 person_count_detector
        skill_config: 可选覆盖配置（会与 DEFAULT_CONFIG 合并）
    """
    cls = resolve_skill_class(skill_name)
    config = deepcopy(cls.DEFAULT_CONFIG)

    if skill_config:
        if "params" in skill_config and isinstance(skill_config["params"], dict):
            config.setdefault("params", {})
            config["params"].update(skill_config["params"])
            overrides = {k: v for k, v in skill_config.items() if k != "params"}
            config.update(overrides)
        else:
            config.update(skill_config)

    return cls(config)
