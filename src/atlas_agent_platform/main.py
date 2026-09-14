from fastapi import FastAPI

from atlas_agent_platform.core.config import get_settings
from atlas_agent_platform.core.exception_handlers import (
    llm_provider_error_handler,
)
from atlas_agent_platform.llm.exceptions import LLMProviderError
from atlas_agent_platform.routes.health import router as health_router
from atlas_agent_platform.routes.llm import router as llm_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

app.add_exception_handler(
    LLMProviderError,
    llm_provider_error_handler,
)

app.include_router(health_router)
app.include_router(llm_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running"}