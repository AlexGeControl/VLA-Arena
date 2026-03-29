from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the episode index at startup and tear down on shutdown."""
    from app.data.hdf5_reader import EpisodeIndex

    app.state.episode_index = EpisodeIndex(settings.data_dir)
    yield


app = FastAPI(
    title="OpenPI InterpreT API",
    description="REST API for Pi-Zero attention and embedding visualization",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

from app.routers import attention, embedding, episodes  # noqa: E402

app.include_router(episodes.router)
app.include_router(attention.router)
app.include_router(embedding.router)


@app.get("/api/health")
async def health():
    """Liveness probe."""
    return {"status": "ok"}
