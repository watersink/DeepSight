"""开放接口：第三方用 API Key 拉取告警 / 媒体 / 回写状态"""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from fastapi.responses import RedirectResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.mgmt_schemas import AlertListResponse, AlertOut, AlertStatusUpdate, PageMeta
from app.db import get_db
from app.db.models import AlertRecord, PartnerIntegration
from app.services import mgmt_service as mgmt_svc
from app.services import partner_service as partner_svc
from app.services.webhook_dispatch import _as_str_list, partner_matches_alert

router = APIRouter(prefix="/open/v1", tags=["开放接口"])


def get_partner_by_api_key(
    db: Annotated[Session, Depends(get_db)],
    x_api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
    api_key: Annotated[Optional[str], Query(description="兼容 query 传 key")] = None,
) -> PartnerIntegration:
    key = (x_api_key or api_key or "").strip()
    if not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 API Key（Header: X-API-Key）",
        )
    partner = partner_svc.get_partner_by_api_key(db, key)
    if partner is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key 无效或接入方已禁用",
        )
    return partner


def _resolve_alert(db: Session, alert_id: str) -> Optional[AlertRecord]:
    key = (alert_id or "").strip()
    if not key:
        return None
    if key.isdigit():
        return db.get(AlertRecord, int(key))
    return db.scalar(select(AlertRecord).where(AlertRecord.alert_uid == key))


@router.get("/health", summary="开放接口健康检查")
def open_health():
    return {"status": "ok", "service": "open-api"}


@router.get("/me", summary="当前接入方信息")
def open_me(partner: Annotated[PartnerIntegration, Depends(get_partner_by_api_key)]):
    return {
        "id": partner.id,
        "name": partner.name,
        "code": partner.code,
        "enabled": partner.enabled,
        "subscribe_categories": _as_str_list(partner.subscribe_categories) or ["alert"],
        "subscribe_recognition_types": _as_str_list(
            partner.subscribe_recognition_types
        ),
        "subscribe_scene_ids": _as_str_list(partner.subscribe_scene_ids),
        "webhook_configured": bool((partner.webhook_url or "").strip()),
    }


@router.get("/alerts", response_model=AlertListResponse, summary="告警列表（按订阅过滤）")
def open_list_alerts(
    db: Annotated[Session, Depends(get_db)],
    partner: Annotated[PartnerIntegration, Depends(get_partner_by_api_key)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    scene_id: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    recognition_type: Optional[str] = None,
    time_from: Optional[datetime] = None,
    time_to: Optional[datetime] = None,
    category: Optional[str] = None,
):
    cats = _as_str_list(partner.subscribe_categories) or ["alert"]
    if category and category not in cats:
        return AlertListResponse(
            items=[], meta=PageMeta(total=0, page=page, page_size=page_size)
        )
    query_category = category if category else (cats[0] if len(cats) == 1 else None)

    partner_types = _as_str_list(partner.subscribe_recognition_types)
    if recognition_type and partner_types and recognition_type not in partner_types:
        return AlertListResponse(
            items=[], meta=PageMeta(total=0, page=page, page_size=page_size)
        )
    type_q = recognition_type or (
        partner_types[0] if len(partner_types) == 1 else None
    )

    partner_scenes = _as_str_list(partner.subscribe_scene_ids)
    if scene_id and partner_scenes and scene_id not in partner_scenes:
        return AlertListResponse(
            items=[], meta=PageMeta(total=0, page=page, page_size=page_size)
        )
    scene_q = scene_id or (partner_scenes[0] if len(partner_scenes) == 1 else None)

    # 多页扫描后按订阅规则过滤（保证分页可用；订阅面很宽时建议对方带过滤条件）
    need = page * page_size
    matched: list[AlertRecord] = []
    scan_page = 1
    scan_size = min(100, max(page_size * 3, 30))
    while len(matched) < need and scan_page <= 30:
        items, meta = mgmt_svc.list_alerts(
            db,
            scan_page,
            scan_size,
            scene_q,
            status_filter,
            category=query_category,
            recognition_type=type_q,
            time_from=time_from,
            time_to=time_to,
        )
        if not items:
            break
        for row in items:
            if partner_matches_alert(partner, row):
                matched.append(row)
        total = int(meta.get("total") or 0)
        if scan_page * scan_size >= total:
            break
        scan_page += 1

    start = (page - 1) * page_size
    page_items = matched[start : start + page_size]
    return AlertListResponse(
        items=[AlertOut.model_validate(i) for i in page_items],
        meta=PageMeta(total=len(matched), page=page, page_size=page_size),
    )


@router.get(
    "/alerts/{alert_id}",
    response_model=AlertOut,
    summary="告警详情（alert_uid 或数字 id）",
)
def open_get_alert(
    alert_id: str,
    db: Annotated[Session, Depends(get_db)],
    partner: Annotated[PartnerIntegration, Depends(get_partner_by_api_key)],
):
    row = _resolve_alert(db, alert_id)
    if row is None or not partner_matches_alert(partner, row):
        raise HTTPException(status_code=404, detail="告警不存在或无权访问")
    return AlertOut.model_validate(row)


@router.get(
    "/alerts/{alert_id}/media/image",
    summary="告警图片（重定向到存储 URL）",
)
def open_alert_image(
    alert_id: str,
    db: Annotated[Session, Depends(get_db)],
    partner: Annotated[PartnerIntegration, Depends(get_partner_by_api_key)],
):
    row = _resolve_alert(db, alert_id)
    if row is None or not partner_matches_alert(partner, row):
        raise HTTPException(status_code=404, detail="告警不存在或无权访问")
    if not row.image_url:
        raise HTTPException(status_code=404, detail="暂无告警图片")
    return RedirectResponse(url=row.image_url)


@router.get(
    "/alerts/{alert_id}/media/video",
    summary="告警视频（重定向到存储 URL；未就绪返回 404）",
)
def open_alert_video(
    alert_id: str,
    db: Annotated[Session, Depends(get_db)],
    partner: Annotated[PartnerIntegration, Depends(get_partner_by_api_key)],
):
    row = _resolve_alert(db, alert_id)
    if row is None or not partner_matches_alert(partner, row):
        raise HTTPException(status_code=404, detail="告警不存在或无权访问")
    if not row.video_url:
        raise HTTPException(status_code=404, detail="视频尚未就绪，请稍后重试")
    return RedirectResponse(url=row.video_url)


@router.patch(
    "/alerts/{alert_id}/ack",
    response_model=AlertOut,
    summary="回写告警状态",
)
def open_ack_alert(
    alert_id: str,
    body: AlertStatusUpdate,
    db: Annotated[Session, Depends(get_db)],
    partner: Annotated[PartnerIntegration, Depends(get_partner_by_api_key)],
):
    row = _resolve_alert(db, alert_id)
    if row is None or not partner_matches_alert(partner, row):
        raise HTTPException(status_code=404, detail="告警不存在或无权访问")
    updated = mgmt_svc.update_alert_status(db, row, body.status)
    return AlertOut.model_validate(updated)
