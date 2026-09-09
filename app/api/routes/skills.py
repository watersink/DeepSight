"""技能查询 API"""
from fastapi import APIRouter, Depends

from app.api.deps import get_current_user
from app.api.schemas import SkillInfoResponse, SkillListResponse
from app.plugins.skill_registry import list_available_skills

router = APIRouter(
    prefix="/skills",
    tags=["技能"],
    dependencies=[Depends(get_current_user)],
)


@router.get(
    "",
    response_model=SkillListResponse,
    summary="列出可用技能",
    description=(
        "返回当前已注册、可用于推流任务的 AI 检测技能列表。\n\n"
        "启动推流时 `skill_name` 须为本接口返回的 `skill_name` 之一。"
    ),
    responses={200: {"description": "技能列表"}},
)
def list_skills():
    items = [SkillInfoResponse(**item) for item in list_available_skills()]
    return SkillListResponse(skills=items)
