# Task C1: FastAPI Scaffold

> Part of the [Backend Development](../README.md) epic.

## Goal

A runnable FastAPI application with CORS configuration, health check endpoint, project structure, and development server.

## Task

### Project Setup

Create the backend project at `openpi_interpret/backend/`:

```
backend/
  app/
    __init__.py
    main.py
    config.py
    routers/
      __init__.py
    data/
      __init__.py
  requirements.txt
  pyproject.toml
```

### Dependencies (`requirements.txt`)

```
fastapi>=0.115
uvicorn[standard]>=0.30
h5py>=3.11
numpy>=1.26
Pillow>=10.0
pydantic>=2.0
```

### Application Entry Point (`main.py`)

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: validate data directory exists
    yield
    # Shutdown: cleanup

app = FastAPI(
    title="OpenPI InterpreT API",
    description="REST API for Pi-Zero attention and embedding visualization",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["GET"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health():
    return {"status": "ok"}
```

### Configuration (`config.py`)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    data_dir: str = "../data"
    host: str = "0.0.0.0"
    port: int = 8080

    class Config:
        env_prefix = "INTERPRET_"

settings = Settings()
```

### Run Command

```bash
cd openpi_interpret/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

## Acceptance Criteria

- [ ] `uvicorn app.main:app --reload` starts without errors
- [ ] `GET /api/health` returns `{"status": "ok"}`
- [ ] CORS allows requests from `http://localhost:5173`
- [ ] OpenAPI docs available at `/docs`
- [ ] Project structure matches the spec
