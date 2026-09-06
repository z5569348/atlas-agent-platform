from fastapi import FastAPI

from atlas_agent_platform.routes.health import router as health_router

app = FastAPI(
    title="Atlas Agent Platform",
    version="0.1.0",
)

app.include_router(health_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Atlas Agent Platform is running"}