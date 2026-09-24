"""识别类型报警等级：按类型键配置，入库时快照到告警记录。"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AlertLevelConfig

# 与事件/报警归类一致：01/02、画面人数进事件；04-07 进报警。
PRESENCE_TYPE_KEY = "presence"
PRESENCE_SKILL_NAMES = frozenset({"person_presence_detector26"})
DEFAULT_EVENT_LEVEL = 3
DEFAULT_ALERT_LEVEL = 2

DEFAULT_ALERT_LEVELS: List[Dict[str, Any]] = [
    {
        "type_key": "01",
        "name_zh": "人员计数（入）",
        "category": "event",
        "level": DEFAULT_EVENT_LEVEL,
        "sort_order": 10,
    },
    {
        "type_key": "02",
        "name_zh": "人员计数（出）",
        "category": "event",
        "level": DEFAULT_EVENT_LEVEL,
        "sort_order": 20,
    },
    {
        "type_key": PRESENCE_TYPE_KEY,
        "name_zh": "画面人数",
        "category": "event",
        "level": DEFAULT_EVENT_LEVEL,
        "sort_order": 30,
    },
    {
        "type_key": "04",
        "name_zh": "非常规通道入井",
        "category": "alert",
        "level": DEFAULT_ALERT_LEVEL,
        "sort_order": 40,
    },
    {
        "type_key": "05",
        "name_zh": "非常规通道出井",
        "category": "alert",
        "level": DEFAULT_ALERT_LEVEL,
        "sort_order": 50,
    },
    {
        "type_key": "06",
        "name_zh": "入井闸机出闸",
        "category": "alert",
        "level": DEFAULT_ALERT_LEVEL,
        "sort_order": 60,
    },
    {
        "type_key": "07",
        "name_zh": "出井闸机入闸",
        "category": "alert",
        "level": DEFAULT_ALERT_LEVEL,
        "sort_order": 70,
    },
    {
        "type_key": "08",
        "name_zh": "摄像头位置挪移",
        "category": "alert",
        "level": DEFAULT_ALERT_LEVEL,
        "sort_order": 80,
    },
    {
        "type_key": "09",
        "name_zh": "摄像头角度偏离",
        "category": "alert",
        "level": DEFAULT_ALERT_LEVEL,
        "sort_order": 90,
    },
    {
        "type_key": "dark",
        "name_zh": "画面过暗",
        "category": "alert",
        "level": DEFAULT_ALERT_LEVEL,
        "sort_order": 100,
    },
    {
        "type_key": "overexp",
        "name_zh": "画面过曝",
        "category": "alert",
        "level": DEFAULT_ALERT_LEVEL,
        "sort_order": 110,
    },
]


def ensure_alert_levels(db: Session) -> None:
    """补齐内置类型行。已存在的等级不覆盖，避免重启冲掉管理员修改。"""
    existing = {
        row.type_key
        for row in db.scalars(select(AlertLevelConfig)).all()
    }
    for item in DEFAULT_ALERT_LEVELS:
        if item["type_key"] in existing:
            continue
        db.add(
            AlertLevelConfig(
                type_key=item["type_key"],
                name_zh=item["name_zh"],
                category=item["category"],
                level=int(item["level"]),
                sort_order=int(item["sort_order"]),
            )
        )
    db.commit()


def list_alert_levels(db: Session) -> List[AlertLevelConfig]:
    ensure_alert_levels(db)
    return list(
        db.scalars(
            select(AlertLevelConfig).order_by(
                AlertLevelConfig.sort_order, AlertLevelConfig.type_key
            )
        ).all()
    )


def update_alert_level(db: Session, type_key: str, level: int) -> AlertLevelConfig:
    ensure_alert_levels(db)
    row = db.get(AlertLevelConfig, type_key)
    if row is None:
        raise ValueError("未知类型，需先由技能声明该类型键")
    if level not in (1, 2, 3):
        raise ValueError("等级只能是 1、2、3")
    row.level = level
    db.commit()
    db.refresh(row)
    return row


def _normalize_codes(recognition_types: Optional[List[Any]]) -> List[str]:
    return sorted(
        {
            str(item).strip()
            for item in (recognition_types or [])
            if str(item).strip()
        }
    )


def resolve_alarm_level(
    db: Session,
    *,
    recognition_types: Optional[List[Any]] = None,
    skill_name: str = "",
    category: str = "alert",
) -> int:
    """按类型键取当前配置。多编码取更严重的（数字更小）。未配置则按列表默认。"""
    fallback = DEFAULT_ALERT_LEVEL if category == "alert" else DEFAULT_EVENT_LEVEL
    codes = _normalize_codes(recognition_types)
    if not codes and skill_name in PRESENCE_SKILL_NAMES:
        codes = [PRESENCE_TYPE_KEY]
    if not codes:
        return fallback
    rows = list(
        db.scalars(
            select(AlertLevelConfig).where(AlertLevelConfig.type_key.in_(codes))
        ).all()
    )
    levels = [int(row.level) for row in rows if int(row.level or 0) in (1, 2, 3)]
    if not levels:
        return fallback
    return min(levels)
