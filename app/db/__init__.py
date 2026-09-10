"""SQLAlchemy 引擎与会话"""
from __future__ import annotations

import logging
import re
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.base import Base

logger = logging.getLogger(__name__)

_DB_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

engine = create_engine(
    settings.mysql_database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """供告警子进程等非 FastAPI 上下文使用。"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def ensure_mysql_database() -> None:
    """若目标库不存在则创建（utf8mb4）。"""
    db_name = (settings.MYSQL_DB or "").strip()
    if not db_name or not _DB_NAME_RE.fullmatch(db_name):
        raise ValueError(f"非法 MySQL 库名: {db_name!r}，仅允许字母数字下划线")

    server_engine = create_engine(
        settings.mysql_server_url,
        pool_pre_ping=True,
        isolation_level="AUTOCOMMIT",
    )
    try:
        with server_engine.connect() as conn:
            exists = conn.execute(
                text(
                    "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA "
                    "WHERE SCHEMA_NAME = :name"
                ),
                {"name": db_name},
            ).scalar()
            if exists:
                logger.info("MySQL 数据库已存在: %s", db_name)
                return
            conn.execute(
                text(
                    f"CREATE DATABASE `{db_name}` "
                    "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
            logger.info("已自动创建 MySQL 数据库: %s", db_name)
    finally:
        server_engine.dispose()


def init_db() -> None:
    """确保库存在，并创建缺失的表（演示期够用；生产可改 Alembic）。"""
    from app.db import models  # noqa: F401
    from app.db import session_scope
    from app.services.auth_service import seed_default_admin
    from app.services.platform_settings_service import ensure_platform_settings

    ensure_mysql_database()
    Base.metadata.create_all(bind=engine)
    _ensure_mgmt_alerts_columns()
    _ensure_camera_tree_schema()
    _ensure_camera_ingest_columns()
    _ensure_task_schedule_column()
    _ensure_task_output_option_columns()
    _ensure_task_worker_id_column()
    seed_default_admin()
    try:
        with session_scope() as db:
            ensure_platform_settings(db)
    except Exception:
        logger.exception("平台基础配置初始化失败")


def _ensure_camera_tree_schema() -> None:
    """补齐摄像头树：site_id，以及把旧 mine_code 迁到默认煤矿/地点下。"""
    db_name = (settings.MYSQL_DB or "").strip()
    if not db_name:
        return
    with engine.begin() as conn:
        cam_cols = {
            r[0]
            for r in conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = :db AND TABLE_NAME = 'mgmt_cameras'"
                ),
                {"db": db_name},
            ).fetchall()
        }
        if not cam_cols:
            return
        if "site_id" not in cam_cols:
            conn.execute(
                text(
                    "ALTER TABLE `mgmt_cameras` "
                    "ADD COLUMN `site_id` INT NULL, "
                    "ADD INDEX `ix_mgmt_cameras_site_id` (`site_id`)"
                )
            )
            logger.info("已为 mgmt_cameras 增加 site_id 列")
            try:
                conn.execute(
                    text(
                        "ALTER TABLE `mgmt_cameras` "
                        "ADD CONSTRAINT `fk_mgmt_cameras_site_id` "
                        "FOREIGN KEY (`site_id`) REFERENCES `mgmt_sites` (`id`) "
                        "ON DELETE SET NULL"
                    )
                )
            except Exception:
                logger.exception("添加 cameras.site_id 外键失败（可忽略）")

        # 将未挂树的摄像头挂到「按 mine_code 自动创建」的煤矿/未分区地点
        try:
            rows = conn.execute(
                text(
                    "SELECT id, mine_code FROM mgmt_cameras "
                    "WHERE site_id IS NULL"
                )
            ).fetchall()
            for cam_id, mine_code in rows:
                code = (mine_code or "").strip() or "000000000000"
                mine = conn.execute(
                    text("SELECT id FROM mgmt_mines WHERE code = :code LIMIT 1"),
                    {"code": code},
                ).first()
                if mine:
                    mine_id = mine[0]
                else:
                    conn.execute(
                        text(
                            "INSERT INTO mgmt_mines (name, code, enabled, remark) "
                            "VALUES (:name, :code, 1, :remark)"
                        ),
                        {
                            "name": f"煤矿-{code}",
                            "code": code,
                            "remark": "由历史摄像头 mine_code 自动创建",
                        },
                    )
                    mine_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
                site = conn.execute(
                    text(
                        "SELECT id FROM mgmt_sites "
                        "WHERE mine_id = :mid AND code = 'default' LIMIT 1"
                    ),
                    {"mid": mine_id},
                ).first()
                if site:
                    site_id = site[0]
                else:
                    conn.execute(
                        text(
                            "INSERT INTO mgmt_sites "
                            "(mine_id, name, code, enabled, remark) "
                            "VALUES (:mid, :name, 'default', 1, :remark)"
                        ),
                        {
                            "mid": mine_id,
                            "name": "默认地点",
                            "remark": "历史摄像头自动归类",
                        },
                    )
                    site_id = conn.execute(text("SELECT LAST_INSERT_ID()")).scalar()
                conn.execute(
                    text("UPDATE mgmt_cameras SET site_id = :sid WHERE id = :cid"),
                    {"sid": site_id, "cid": cam_id},
                )
        except Exception:
            logger.exception("摄像头树历史数据归类失败（可忽略）")


def _ensure_camera_ingest_columns() -> None:
    """补齐摄像头接入模式字段：ingest_mode / source_url / zlm_app / zlm_stream / proxy_key。"""
    db_name = (settings.MYSQL_DB or "").strip()
    if not db_name:
        return
    with engine.begin() as conn:
        cam_cols = {
            r[0]
            for r in conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = :db AND TABLE_NAME = 'mgmt_cameras'"
                ),
                {"db": db_name},
            ).fetchall()
        }
        if not cam_cols:
            return
        alters = []
        if "ingest_mode" not in cam_cols:
            alters.append(
                "ADD COLUMN `ingest_mode` VARCHAR(16) NOT NULL DEFAULT 'push'"
            )
        if "source_url" not in cam_cols:
            alters.append("ADD COLUMN `source_url` VARCHAR(512) NULL")
        if "zlm_app" not in cam_cols:
            alters.append(
                "ADD COLUMN `zlm_app` VARCHAR(64) NOT NULL DEFAULT ''"
            )
        if "zlm_stream" not in cam_cols:
            alters.append(
                "ADD COLUMN `zlm_stream` VARCHAR(128) NOT NULL DEFAULT ''"
            )
        if "proxy_key" not in cam_cols:
            alters.append("ADD COLUMN `proxy_key` VARCHAR(256) NULL")
        if alters:
            conn.execute(
                text(f"ALTER TABLE `mgmt_cameras` {', '.join(alters)}")
            )
            logger.info("已为 mgmt_cameras 增加接入模式相关列: %s", alters)

        # 历史数据：从 in_url 解析 app/stream 回填（推流接入）
        try:
            rows = conn.execute(
                text(
                    "SELECT id, in_url FROM mgmt_cameras "
                    "WHERE (zlm_app = '' OR zlm_app IS NULL) "
                    "AND in_url IS NOT NULL AND in_url <> ''"
                )
            ).fetchall()
            for cam_id, in_url in rows:
                from app.services.zlm_client import parse_stream_url

                parsed = parse_stream_url(str(in_url or ""))
                if not parsed:
                    continue
                conn.execute(
                    text(
                        "UPDATE mgmt_cameras "
                        "SET zlm_app = :app, zlm_stream = :stream, "
                        "ingest_mode = COALESCE(NULLIF(ingest_mode, ''), 'push') "
                        "WHERE id = :cid"
                    ),
                    {
                        "app": parsed["app"],
                        "stream": parsed["stream"],
                        "cid": cam_id,
                    },
                )
        except Exception:
            logger.exception("摄像头 zlm_app/zlm_stream 回填失败（可忽略）")


def _ensure_mgmt_alerts_columns() -> None:
    """已有库补齐 mgmt_alerts 新增列（create_all 不会改已有表结构）。"""
    db_name = (settings.MYSQL_DB or "").strip()
    if not db_name:
        return
    with engine.begin() as conn:
        cols = {
            r[0]
            for r in conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = :db AND TABLE_NAME = 'mgmt_alerts'"
                ),
                {"db": db_name},
            ).fetchall()
        }
        if "video_url" not in cols:
            conn.execute(
                text("ALTER TABLE `mgmt_alerts` ADD COLUMN `video_url` VARCHAR(512) NULL")
            )
            logger.info("已为 mgmt_alerts 增加 video_url 列")
        if "category" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE `mgmt_alerts` "
                    "ADD COLUMN `category` VARCHAR(16) NOT NULL DEFAULT 'alert'"
                )
            )
            conn.execute(
                text("CREATE INDEX `ix_mgmt_alerts_category` ON `mgmt_alerts` (`category`)")
            )
            logger.info("已为 mgmt_alerts 增加 category 列")
            try:
                # 历史数据粗分：画面人数技能、仅 01/02 → event
                conn.execute(
                    text(
                        "UPDATE `mgmt_alerts` SET `category` = 'event' "
                        "WHERE `skill_name` = 'person_presence_detector26'"
                    )
                )
                conn.execute(
                    text(
                        "UPDATE `mgmt_alerts` SET `category` = 'event' "
                        "WHERE `category` = 'alert' "
                        "AND `recognition_types` IS NOT NULL "
                        "AND JSON_CONTAINS(`recognition_types`, '\"01\"') + "
                        "JSON_CONTAINS(`recognition_types`, '\"02\"') > 0 "
                        "AND COALESCE(JSON_CONTAINS(`recognition_types`, '\"04\"'), 0) = 0 "
                        "AND COALESCE(JSON_CONTAINS(`recognition_types`, '\"05\"'), 0) = 0 "
                        "AND COALESCE(JSON_CONTAINS(`recognition_types`, '\"06\"'), 0) = 0 "
                        "AND COALESCE(JSON_CONTAINS(`recognition_types`, '\"07\"'), 0) = 0"
                    )
                )
            except Exception:
                logger.exception("mgmt_alerts.category 历史数据回填失败（可忽略）")


def _ensure_task_schedule_column() -> None:
    """已有库补齐任务运行时间段 schedule JSON 列。"""
    db_name = (settings.MYSQL_DB or "").strip()
    if not db_name:
        return
    with engine.begin() as conn:
        cols = {
            r[0]
            for r in conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = :db AND TABLE_NAME = 'mgmt_task_configs'"
                ),
                {"db": db_name},
            ).fetchall()
        }
        if not cols:
            return
        if "schedule" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE `mgmt_task_configs` "
                    "ADD COLUMN `schedule` JSON NULL"
                )
            )
            logger.info("已为 mgmt_task_configs 增加 schedule 列")


def _ensure_task_output_option_columns() -> None:
    """补齐报警图片/视频与画框推流开关列。"""
    db_name = (settings.MYSQL_DB or "").strip()
    if not db_name:
        return
    with engine.begin() as conn:
        cols = {
            r[0]
            for r in conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = :db AND TABLE_NAME = 'mgmt_task_configs'"
                ),
                {"db": db_name},
            ).fetchall()
        }
        if not cols:
            return
        additions = [
            (
                "alert_image_enabled",
                "ADD COLUMN `alert_image_enabled` TINYINT(1) NOT NULL DEFAULT 1",
            ),
            (
                "alert_video_enabled",
                "ADD COLUMN `alert_video_enabled` TINYINT(1) NOT NULL DEFAULT 0",
            ),
            (
                "push_annotated_stream",
                "ADD COLUMN `push_annotated_stream` TINYINT(1) NOT NULL DEFAULT 0",
            ),
        ]
        for name, ddl in additions:
            if name not in cols:
                conn.execute(text(f"ALTER TABLE `mgmt_task_configs` {ddl}"))
                logger.info("已为 mgmt_task_configs 增加 %s 列", name)


def _ensure_task_worker_id_column() -> None:
    """补齐任务绑定的 Worker 节点列。"""
    db_name = (settings.MYSQL_DB or "").strip()
    if not db_name:
        return
    with engine.begin() as conn:
        cols = {
            r[0]
            for r in conn.execute(
                text(
                    "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                    "WHERE TABLE_SCHEMA = :db AND TABLE_NAME = 'mgmt_task_configs'"
                ),
                {"db": db_name},
            ).fetchall()
        }
        if not cols:
            return
        if "worker_id" not in cols:
            conn.execute(
                text(
                    "ALTER TABLE `mgmt_task_configs` "
                    "ADD COLUMN `worker_id` VARCHAR(64) NOT NULL DEFAULT 'local'"
                )
            )
            logger.info("已为 mgmt_task_configs 增加 worker_id 列")
