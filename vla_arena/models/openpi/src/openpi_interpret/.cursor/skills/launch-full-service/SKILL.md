---
name: launch-full-service
description: >-
  Launch the complete OpenPI InterpreT service (backend + frontend) for
  LAN access. Use when asked to start the full service, launch everything,
  or make the visualization available to teammates.
---

# Launch Full Service

## Quick Launch

```bash
# 1. Kill any existing processes
fuser -k 8080/tcp 5173/tcp 2>/dev/null; sleep 1

# 2. Get LAN IP
LAN_IP=$(hostname -I | awk '{print $1}')

# 3. Start backend
cd /home/yaoge/Workspace/11-977-Spring-2026--Group-2-VLN/vla-arena/baselines/openpi/VLA-Arena/vla_arena/models/openpi/src/openpi_interpret/backend
INTERPRET_DATA_DIR=../data conda run -n openpi-vla-arena \
  uvicorn app.main:app --host 0.0.0.0 --port 8080 &
sleep 3

# 4. Verify backend
curl -s http://localhost:8080/api/health

# 5. Start frontend
cd ../frontend
VITE_API_BASE="http://${LAN_IP}:8080/api" \
  conda run -n openpi-vla-arena npx vite --host 0.0.0.0 --port 5173 &
sleep 3

echo "Frontend: http://${LAN_IP}:5173"
echo "Backend:  http://${LAN_IP}:8080"
echo "API Docs: http://${LAN_IP}:8080/docs"
```

## Shutdown

```bash
fuser -k 8080/tcp 5173/tcp
```

## Prerequisites

- HDF5 data files in `src/openpi_interpret/data/` (run extraction first)
- conda env `openpi-vla-arena` with all deps (fastapi, uvicorn, h5py, nodejs)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| "Address already in use" | `fuser -k 8080/tcp` then retry |
| Frontend shows "Error: Failed to fetch" | Check backend is running: `curl localhost:8080/api/health` |
| Camera images missing | Verify HDF5 has per-timestep cameras: `h5ls data/ep_000000.h5/timestep_000/cameras` |
| Can't access from other devices | Ensure `VITE_API_BASE` uses the LAN IP, not `localhost` |
