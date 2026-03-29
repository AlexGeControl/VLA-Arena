# C1 — FastAPI Scaffold

## Goal

Bootstrap the **Track C** Python package: a minimal FastAPI application with correct CORS for LAN testing, centralized configuration, a health check, and a `requirements.txt` that excludes ML libraries.

## Project Structure

Create under `openpi_interpret/backend/`:

```
backend/
  app/
    __init__.py
    main.py
    config.py
    data/
      __init__.py
    routers/
      __init__.py
  requirements.txt
  tests/
    __init__.py
```

(Additional modules are added in C2–C5; C1 only needs the skeleton and health route.)

## `config.py`

- Use **`pydantic-settings`** `BaseSettings` (or `SettingsConfigDict`) to load:
  - **`data_dir`**: path to the directory containing `*.h5` files (env var e.g. `DATA_DIR`, default sensible for local dev).
- Expose a cached `get_settings()` or module-level `settings` instance for use in `main.py` and dependencies (C2).

## `main.py`

- Construct `FastAPI()` with optional **`lifespan`** context manager:
  - On startup: log or validate that `data_dir` exists (warn if missing; C2 will attach `EpisodeIndex`).
  - On shutdown: no heavy cleanup required for read-only service.
- Add **`CORSMiddleware`**:
  - `allow_origins=["*"]` — **required** so browsers on other machines on the LAN can call the API during development and demos.
  - `allow_credentials=False` when using wildcard origins (browser rules).
  - Include `allow_methods` and `allow_headers` as needed (e.g. `["*"]` or explicit GET-only set).
- Mount a **`GET /api/health`** route returning `{"status": "ok"}` (JSON).
- Include placeholder router modules if desired (empty routers OK until C3–C5).

## `requirements.txt`

Pin compatible versions, for example:

- `fastapi>=0.115`
- `uvicorn[standard]>=0.30`
- `h5py>=3.13`
- `numpy`
- `Pillow`
- `pydantic>=2`
- `pydantic-settings`

**Do not** add `jax`, `torch`, `scikit-learn`, or `scipy`.

## CORS Note

Using **`allow_origins=["*"]`** is intentional for LAN testing (phone/tablet/another laptop hitting the dev server). Tighten origins for production deployments.

## Run Locally

Document in code comments or team README (optional): `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000` from `backend/` with `PYTHONPATH` or package layout as chosen.

## Acceptance Criteria

1. `GET /api/health` returns HTTP 200 and body `{"status":"ok"}`.
2. CORS middleware is registered with `allow_origins=["*"]`.
3. `config.py` loads `data_dir` via pydantic-settings from environment with a documented variable name.
4. `requirements.txt` lists only the agreed stack; **no** sklearn/scipy/JAX/torch.
5. App starts with `uvicorn` without import errors.
