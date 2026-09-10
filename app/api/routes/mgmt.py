"""管理台 API：摄像头 / 算法 / 任务 / 报警"""
import logging
from datetime import datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.mgmt_schemas import (
    AlertListResponse,
    AlertOut,
    AlertStatusUpdate,
    AlgorithmConfigCreate,
    AlgorithmConfigListResponse,
    AlgorithmConfigOut,
    AlgorithmConfigUpdate,
    CameraCreate,
    CameraListResponse,
    CameraOut,
    CameraTreeResponse,
    CameraUpdate,
    MineCreate,
    MineListResponse,
    MineOut,
    MineUpdate,
    PageMeta,
    PartnerIntegrationCreate,
    PartnerIntegrationListResponse,
    PartnerIntegrationOut,
    PartnerIntegrationUpdate,
    SiteCreate,
    SiteListResponse,
    SiteOut,
    SiteUpdate,
    TaskConfigCreate,
    TaskConfigListResponse,
    TaskConfigOut,
    TaskConfigUpdate,
    TritonModelListResponse,
)
from app.api.deps import get_current_user, require_admin
from app.db import get_db
from app.db.models import AlertRecord, User
from app.plugins.skill_registry import SkillNotFoundError
from app.services import mgmt_service as svc
from app.services import partner_service as partner_svc
from app.services.webhook_dispatch import (
    build_webhook_payload,
    deliver_webhook,
)
from app.services.zlm_client import ZLMClientError, zlm_client

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["管理台"],
    dependencies=[Depends(get_current_user)],
)


def _meta(page: int, page_size: int, total: int) -> PageMeta:
    return PageMeta(total=total, page=page, page_size=page_size)


@router.get("/models", response_model=TritonModelListResponse, summary="Triton 模型列表")
def list_models():
    """汇总各 Worker 节点本机 Triton 仓库中的模型，并标注部署机器。"""
    from app.services.runtime_gateway import runtime_gateway

    data = runtime_gateway.list_models_aggregated()
    return TritonModelListResponse(**data)


@router.get(
    "/zlm/snap",
    summary="ZLM 实时截图",
    description=(
        "代理 ZLMediaKit `/index/api/getSnap`，返回 JPEG。"
        "前端用摄像头 `in_url`（RTMP/RTSP）拉一张底图做线段标注。"
    ),
    responses={200: {"content": {"image/jpeg": {}}}},
)
def get_zlm_snap(
    db: Annotated[Session, Depends(get_db)],
    url: str = Query("", description="需要截图的流地址，如 rtmp://host/app/stream"),
    timeout_sec: int = Query(10, ge=1, le=60),
    expire_sec: int = Query(30, ge=1, le=600),
    camera_id: Optional[int] = Query(
        None, description="可选：传摄像头 ID 时优先用其 in_url"
    ),
):
    snap_url = (url or "").strip()
    if camera_id is not None:
        row = svc.get_camera(db, camera_id)
        if not row:
            raise HTTPException(status_code=404, detail="摄像头不存在")
        snap_url = (row.in_url or "").strip()
    if not snap_url:
        raise HTTPException(status_code=400, detail="url 或 camera_id 必填其一")
    try:
        data = zlm_client.get_snap(
            snap_url, timeout_sec=timeout_sec, expire_sec=expire_sec
        )
    except ZLMClientError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("getSnap 失败 url=%s", snap_url)
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=data, media_type="image/jpeg")


# ---------- mines / sites / camera tree ----------

@router.get("/camera-tree", response_model=CameraTreeResponse, summary="摄像头树")
def get_camera_tree(db: Annotated[Session, Depends(get_db)]):
    return CameraTreeResponse(**svc.get_camera_tree(db))


@router.get("/mines", response_model=MineListResponse, summary="煤矿列表")
def list_mines(db: Annotated[Session, Depends(get_db)]):
    return MineListResponse(items=[MineOut.model_validate(i) for i in svc.list_mines(db)])


@router.post("/mines", response_model=MineOut, status_code=status.HTTP_201_CREATED, summary="新增煤矿")
def create_mine(body: MineCreate, db: Annotated[Session, Depends(get_db)]):
    row = svc.create_mine(db, body.model_dump())
    items = {m["id"]: m for m in svc.list_mines(db)}
    return MineOut.model_validate(items[row.id])


@router.patch("/mines/{mine_id}", response_model=MineOut, summary="更新煤矿")
def update_mine(
    mine_id: int, body: MineUpdate, db: Annotated[Session, Depends(get_db)]
):
    row = svc.get_mine(db, mine_id)
    if not row:
        raise HTTPException(status_code=404, detail="煤矿不存在")
    svc.update_mine(db, row, body.model_dump(exclude_unset=True))
    items = {m["id"]: m for m in svc.list_mines(db)}
    return MineOut.model_validate(items[mine_id])


@router.delete("/mines/{mine_id}", status_code=status.HTTP_204_NO_CONTENT, summary="删除煤矿")
def delete_mine(mine_id: int, db: Annotated[Session, Depends(get_db)]):
    row = svc.get_mine(db, mine_id)
    if not row:
        raise HTTPException(status_code=404, detail="煤矿不存在")
    svc.delete_mine(db, row)


@router.get("/sites", response_model=SiteListResponse, summary="地点列表")
def list_sites(
    db: Annotated[Session, Depends(get_db)],
    mine_id: Optional[int] = None,
):
    return SiteListResponse(
        items=[SiteOut.model_validate(i) for i in svc.list_sites(db, mine_id)]
    )


@router.post("/sites", response_model=SiteOut, status_code=status.HTTP_201_CREATED, summary="新增地点")
def create_site(body: SiteCreate, db: Annotated[Session, Depends(get_db)]):
    try:
        row = svc.create_site(db, body.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    items = {s["id"]: s for s in svc.list_sites(db)}
    return SiteOut.model_validate(items[row.id])


@router.patch("/sites/{site_id}", response_model=SiteOut, summary="更新地点")
def update_site(
    site_id: int, body: SiteUpdate, db: Annotated[Session, Depends(get_db)]
):
    row = svc.get_site(db, site_id)
    if not row:
        raise HTTPException(status_code=404, detail="地点不存在")
    try:
        svc.update_site(db, row, body.model_dump(exclude_unset=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    items = {s["id"]: s for s in svc.list_sites(db)}
    return SiteOut.model_validate(items[site_id])


@router.delete("/sites/{site_id}", status_code=status.HTTP_204_NO_CONTENT, summary="删除地点")
def delete_site(site_id: int, db: Annotated[Session, Depends(get_db)]):
    row = svc.get_site(db, site_id)
    if not row:
        raise HTTPException(status_code=404, detail="地点不存在")
    svc.delete_site(db, row)


@router.get("/cameras", response_model=CameraListResponse, summary="摄像头列表")
def list_cameras(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    site_id: Optional[int] = None,
    mine_id: Optional[int] = None,
):
    items, meta = svc.list_cameras(db, page, page_size, site_id=site_id, mine_id=mine_id)
    return CameraListResponse(
        items=[CameraOut.model_validate(i) for i in items],
        meta=_meta(**meta),
    )


@router.get("/cameras/{camera_id}", response_model=CameraOut, summary="摄像头详情")
def get_camera(camera_id: int, db: Annotated[Session, Depends(get_db)]):
    row = svc.get_camera(db, camera_id)
    if not row:
        raise HTTPException(status_code=404, detail="摄像头不存在")
    return CameraOut.model_validate(svc.enrich_camera(row))


@router.patch("/cameras/{camera_id}", response_model=CameraOut, summary="更新摄像头")
def update_camera(
    camera_id: int, body: CameraUpdate, db: Annotated[Session, Depends(get_db)]
):
    row = svc.get_camera(db, camera_id)
    if not row:
        raise HTTPException(status_code=404, detail="摄像头不存在")
    data = body.model_dump(exclude_unset=True)
    try:
        row = svc.update_camera(db, row, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ZLMClientError as e:
        raise HTTPException(status_code=502, detail=f"ZLMediaKit 操作失败: {e}")
    return CameraOut.model_validate(svc.enrich_camera(row))


@router.post(
    "/cameras",
    response_model=CameraOut,
    status_code=status.HTTP_201_CREATED,
    summary="创建摄像头",
)
def create_camera(body: CameraCreate, db: Annotated[Session, Depends(get_db)]):
    try:
        row = svc.create_camera(db, body.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ZLMClientError as e:
        raise HTTPException(status_code=502, detail=f"ZLMediaKit 操作失败: {e}")
    return CameraOut.model_validate(svc.enrich_camera(row))


@router.delete("/cameras/{camera_id}", summary="删除摄像头")
def delete_camera(camera_id: int, db: Annotated[Session, Depends(get_db)]):
    row = svc.get_camera(db, camera_id)
    if not row:
        raise HTTPException(status_code=404, detail="摄像头不存在")
    try:
        svc.delete_camera(db, row)
    except Exception as e:
        raise HTTPException(status_code=409, detail=f"删除失败（可能被任务引用）: {e}")
    return {"ok": True}


# ---------- algorithms ----------

@router.get(
    "/algorithm-configs",
    response_model=AlgorithmConfigListResponse,
    summary="算法配置列表",
)
def list_algorithms(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    items, meta = svc.list_algorithms(db, page, page_size)
    return AlgorithmConfigListResponse(
        items=[AlgorithmConfigOut.model_validate(i) for i in items],
        meta=_meta(**meta),
    )


@router.post(
    "/algorithm-configs",
    response_model=AlgorithmConfigOut,
    status_code=status.HTTP_201_CREATED,
    summary="创建算法配置",
)
def create_algorithm(body: AlgorithmConfigCreate, db: Annotated[Session, Depends(get_db)]):
    try:
        row = svc.create_algorithm(db, body.model_dump())
    except SkillNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AlgorithmConfigOut.model_validate(row)


@router.get(
    "/algorithm-configs/{algo_id}",
    response_model=AlgorithmConfigOut,
    summary="算法配置详情",
)
def get_algorithm(algo_id: int, db: Annotated[Session, Depends(get_db)]):
    row = svc.get_algorithm(db, algo_id)
    if not row:
        raise HTTPException(status_code=404, detail="算法配置不存在")
    return AlgorithmConfigOut.model_validate(row)


@router.patch(
    "/algorithm-configs/{algo_id}",
    response_model=AlgorithmConfigOut,
    summary="更新算法配置",
)
def update_algorithm(
    algo_id: int, body: AlgorithmConfigUpdate, db: Annotated[Session, Depends(get_db)]
):
    row = svc.get_algorithm(db, algo_id)
    if not row:
        raise HTTPException(status_code=404, detail="算法配置不存在")
    try:
        row = svc.update_algorithm(db, row, body.model_dump(exclude_unset=True))
    except SkillNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AlgorithmConfigOut.model_validate(row)


@router.delete("/algorithm-configs/{algo_id}", summary="删除算法配置")
def delete_algorithm(algo_id: int, db: Annotated[Session, Depends(get_db)]):
    row = svc.get_algorithm(db, algo_id)
    if not row:
        raise HTTPException(status_code=404, detail="算法配置不存在")
    try:
        svc.delete_algorithm(db, row)
    except Exception as e:
        raise HTTPException(status_code=409, detail=f"删除失败（可能被任务引用）: {e}")
    return {"ok": True}


# ---------- task configs ----------

@router.get("/task-configs", response_model=TaskConfigListResponse, summary="任务配置列表")
def list_tasks(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    items, meta = svc.list_tasks(db, page, page_size)
    return TaskConfigListResponse(
        items=[TaskConfigOut.model_validate(i) for i in items],
        meta=_meta(**meta),
    )


@router.post(
    "/task-configs",
    response_model=TaskConfigOut,
    status_code=status.HTTP_201_CREATED,
    summary="创建任务配置",
)
def create_task(body: TaskConfigCreate, db: Annotated[Session, Depends(get_db)]):
    try:
        data = svc.create_task(db, body.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return TaskConfigOut.model_validate(data)


@router.get("/task-configs/{task_id}", response_model=TaskConfigOut, summary="任务配置详情")
def get_task(task_id: int, db: Annotated[Session, Depends(get_db)]):
    row = svc.get_task(db, task_id)
    if not row:
        raise HTTPException(status_code=404, detail="任务配置不存在")
    return TaskConfigOut.model_validate(svc._enrich_task(row))


@router.patch("/task-configs/{task_id}", response_model=TaskConfigOut, summary="更新任务配置")
def update_task(
    task_id: int, body: TaskConfigUpdate, db: Annotated[Session, Depends(get_db)]
):
    row = svc.get_task(db, task_id)
    if not row:
        raise HTTPException(status_code=404, detail="任务配置不存在")
    try:
        data = svc.update_task(db, row, body.model_dump(exclude_unset=True))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return TaskConfigOut.model_validate(data)


@router.delete("/task-configs/{task_id}", summary="删除任务配置")
def delete_task(task_id: int, db: Annotated[Session, Depends(get_db)]):
    row = svc.get_task(db, task_id)
    if not row:
        raise HTTPException(status_code=404, detail="任务配置不存在")
    svc.delete_task(db, row)
    return {"ok": True}


@router.post(
    "/task-configs/{task_id}/start",
    response_model=TaskConfigOut,
    summary="按任务配置启动识别推流",
)
def start_task(task_id: int, db: Annotated[Session, Depends(get_db)]):
    row = svc.get_task(db, task_id)
    if not row:
        raise HTTPException(status_code=404, detail="任务配置不存在")
    try:
        data = svc.start_task_config(db, row)
    except SkillNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        logger.exception("按任务配置启动失败")
        raise HTTPException(status_code=500, detail=str(e))
    return TaskConfigOut.model_validate(data)


@router.post(
    "/task-configs/{task_id}/stop",
    response_model=TaskConfigOut,
    summary="停止任务配置对应的运行实例",
)
def stop_task(task_id: int, db: Annotated[Session, Depends(get_db)]):
    row = svc.get_task(db, task_id)
    if not row:
        raise HTTPException(status_code=404, detail="任务配置不存在")
    try:
        data = svc.stop_task_config(db, row)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return TaskConfigOut.model_validate(data)


# ---------- alerts ----------

@router.get("/alerts", response_model=AlertListResponse, summary="报警列表")
def list_alerts(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    scene_id: Optional[str] = Query(None, description="场景 ID，模糊匹配"),
    status_filter: Optional[str] = Query(None, alias="status"),
    skill_name: Optional[str] = Query(None, description="技能名，模糊匹配"),
    recognition_type: Optional[str] = Query(
        None, description="识别类型码，如 04/05/06/07"
    ),
    time_from: Optional[datetime] = Query(None, description="起始时间（含）"),
    time_to: Optional[datetime] = Query(None, description="结束时间（含）"),
    category: Optional[str] = Query(
        "alert",
        description="alert=违规报警；event=正常事件；不传则默认 alert",
    ),
):
    items, meta = svc.list_alerts(
        db,
        page,
        page_size,
        scene_id,
        status_filter,
        category=category,
        skill_name=skill_name,
        recognition_type=recognition_type,
        time_from=time_from,
        time_to=time_to,
    )
    return AlertListResponse(
        items=[AlertOut.model_validate(i) for i in items],
        meta=_meta(**meta),
    )


@router.get("/events", response_model=AlertListResponse, summary="事件列表")
def list_events(
    db: Annotated[Session, Depends(get_db)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    scene_id: Optional[str] = Query(None, description="场景 ID，模糊匹配"),
    status_filter: Optional[str] = Query(None, alias="status"),
    skill_name: Optional[str] = Query(None, description="技能名，模糊匹配"),
    recognition_type: Optional[str] = Query(
        None, description="识别类型码，如 01/02"
    ),
    time_from: Optional[datetime] = Query(None, description="起始时间（含）"),
    time_to: Optional[datetime] = Query(None, description="结束时间（含）"),
):
    """正常识别事件：过线计数 01/02、画面人数等（不含 04-07 违规报警）。"""
    items, meta = svc.list_alerts(
        db,
        page,
        page_size,
        scene_id,
        status_filter,
        category="event",
        skill_name=skill_name,
        recognition_type=recognition_type,
        time_from=time_from,
        time_to=time_to,
    )
    return AlertListResponse(
        items=[AlertOut.model_validate(i) for i in items],
        meta=_meta(**meta),
    )


@router.get("/alerts/{alert_id}", response_model=AlertOut, summary="报警详情")
def get_alert(alert_id: int, db: Annotated[Session, Depends(get_db)]):
    row = svc.get_alert(db, alert_id)
    if not row:
        raise HTTPException(status_code=404, detail="报警不存在")
    return AlertOut.model_validate(row)


@router.patch("/alerts/{alert_id}", response_model=AlertOut, summary="更新报警状态")
def patch_alert(
    alert_id: int, body: AlertStatusUpdate, db: Annotated[Session, Depends(get_db)]
):
    row = svc.get_alert(db, alert_id)
    if not row:
        raise HTTPException(status_code=404, detail="报警不存在")
    return AlertOut.model_validate(svc.update_alert_status(db, row, body.status))


@router.patch("/events/{event_id}", response_model=AlertOut, summary="更新事件状态")
def patch_event(
    event_id: int, body: AlertStatusUpdate, db: Annotated[Session, Depends(get_db)]
):
    row = svc.get_alert(db, event_id)
    if not row:
        raise HTTPException(status_code=404, detail="事件不存在")
    return AlertOut.model_validate(svc.update_alert_status(db, row, body.status))


# ---------- 第三方接入 ----------


@router.get(
    "/partners",
    response_model=PartnerIntegrationListResponse,
    summary="第三方接入列表",
)
def list_partners(
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    items, meta = partner_svc.list_partners(db, page, page_size)
    return PartnerIntegrationListResponse(
        items=[PartnerIntegrationOut.model_validate(i) for i in items],
        meta=_meta(**meta),
    )


@router.post(
    "/partners",
    response_model=PartnerIntegrationOut,
    status_code=status.HTTP_201_CREATED,
    summary="新增第三方接入",
)
def create_partner(
    body: PartnerIntegrationCreate,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    try:
        row = partner_svc.create_partner(db, body.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return PartnerIntegrationOut.model_validate(row)


@router.get(
    "/partners/{partner_id}",
    response_model=PartnerIntegrationOut,
    summary="第三方接入详情",
)
def get_partner(
    partner_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    row = partner_svc.get_partner(db, partner_id)
    if not row:
        raise HTTPException(status_code=404, detail="接入配置不存在")
    return PartnerIntegrationOut.model_validate(row)


@router.patch(
    "/partners/{partner_id}",
    response_model=PartnerIntegrationOut,
    summary="更新第三方接入",
)
def update_partner(
    partner_id: int,
    body: PartnerIntegrationUpdate,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    row = partner_svc.get_partner(db, partner_id)
    if not row:
        raise HTTPException(status_code=404, detail="接入配置不存在")
    try:
        row = partner_svc.update_partner(
            db, row, body.model_dump(exclude_unset=True)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return PartnerIntegrationOut.model_validate(row)


@router.delete(
    "/partners/{partner_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除第三方接入",
)
def delete_partner(
    partner_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    row = partner_svc.get_partner(db, partner_id)
    if not row:
        raise HTTPException(status_code=404, detail="接入配置不存在")
    partner_svc.delete_partner(db, row)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/partners/{partner_id}/test-webhook",
    summary="向接入方发送测试 Webhook",
)
def test_partner_webhook(
    partner_id: int,
    db: Annotated[Session, Depends(get_db)],
    _: Annotated[User, Depends(require_admin)],
):
    row = partner_svc.get_partner(db, partner_id)
    if not row:
        raise HTTPException(status_code=404, detail="接入配置不存在")
    if not (row.webhook_url or "").strip():
        raise HTTPException(status_code=400, detail="未配置 Webhook URL")
    sample = AlertRecord(
        id=0,
        alert_uid="test_webhook_ping",
        scene_id="test_scene",
        skill_name="test",
        recognition_types=["04"],
        message="webhook test",
        count=0,
        enter_count=0,
        image_url=None,
        video_url=None,
        category="alert",
        payload={"test": True},
        status="new",
        created_at=datetime.now(),
    )
    payload = build_webhook_payload(sample, event_type="alert.test")
    ok = deliver_webhook(row, payload)
    if not ok:
        raise HTTPException(status_code=502, detail="Webhook 投递失败，请检查 URL 与网络")
    return {"ok": True, "payload": payload}
