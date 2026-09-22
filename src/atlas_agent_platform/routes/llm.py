from typing import Annotated

from fastapi import APIRouter, Depends

from atlas_agent_platform.llm.factory import get_llm_provider
from atlas_agent_platform.llm.providers.base import LLMProvider
from atlas_agent_platform.llm.schemas import LLMRequest, LLMResponse
from atlas_agent_platform.llm.service import LLMGatewayService


def get_llm_gateway_service(
    provider: Annotated[
        LLMProvider,
        Depends(get_llm_provider),
    ],
) -> LLMGatewayService:
    return LLMGatewayService(provider)

router = APIRouter(
    prefix="/api/v1/llm",
    tags=["llm"],
)


@router.post("/generate", response_model=LLMResponse)
async def generate_llm_response(
    request: LLMRequest,
        service: Annotated[
        LLMGatewayService,
        Depends(get_llm_gateway_service),
    ],
) -> LLMResponse:
    return await service.generate(request)