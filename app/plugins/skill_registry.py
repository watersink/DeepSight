"""技能注册与工厂 - 按名称创建技能实例"""
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type

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
}


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
        result.append({
            "skill_name": cfg.get("name"),
            "name_zh": cfg.get("name_zh"),
            "description": cfg.get("description"),
            "required_models": cfg.get("required_models", []),
            "cover_image": cfg.get("cover_image")
            or f"/skills/{cfg.get('name')}.png",
            # 任务配置表单按此声明渲染；未声明则不展示计数/画线等专用项
            "form_fields": cfg.get("form_fields") or [],
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
