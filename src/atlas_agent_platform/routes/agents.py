from typing import Annotated

from fastapi import APIRouter, Depends

from atlas_agent_platform.agents.factory import get_agent_service
from atlas_agent_platform.agents.schemas import (
    AgentRunRequest,
    AgentRunResponse,
)
from atlas_agent_platform.agents.service import AgentService

router = APIRouter(
    prefix="/api/v1/agents",
    tags=["agents"],
)

@router.post("/run", response_model=AgentRunResponse)
async def run_agent(
    request: AgentRunRequest,
    service: Annotated[
        AgentService,
        Depends(get_agent_service),
    ],
) -> AgentRunResponse:
    return await service.run(request)