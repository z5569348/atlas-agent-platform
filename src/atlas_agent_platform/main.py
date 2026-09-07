from fastapi import FastAPI

from atlas_agent_platform.core.config import get_settings
from atlas_agent_platform.routes.health import router as health_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
)

app.include_router(health_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": f"{settings.app_name} is running"}