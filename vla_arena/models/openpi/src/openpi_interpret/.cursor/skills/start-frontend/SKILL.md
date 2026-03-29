---
name: start-frontend
description: >-
  Start the React frontend dev server for OpenPI InterpreT. Serves the
  visualization SPA on port 5173. Use when asked to start the frontend,
  web UI, or visualization interface.
---

# Start Frontend

## Prerequisites

- conda env `openpi-vla-arena` with Node.js 20 installed
- Backend running on port 8080 (or wherever `VITE_API_BASE` points)

## Start (Development)

```bash
cd /home/yaoge/Workspace/11-977-Spring-2026--Group-2-VLN/vla-arena/baselines/openpi/VLA-Arena/vla_arena/models/openpi/src/openpi_interpret/frontend

# For LAN access (replace IP with your machine's LAN IP)
VITE_API_BASE="http://192.168.3.57:8080/api" \
  conda run -n openpi-vla-arena npx vite --host 0.0.0.0 --port 5173
```

## Start (Production Build)

```bash
conda run -n openpi-vla-arena npm run build
# Serve dist/ with any static file server
```

## Stop

```bash
fuser -k 5173/tcp
```

## Get LAN IP

```bash
hostname -I | awk '{print $1}'
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_API_BASE` | `http://localhost:8080/api` | Backend API URL (must be reachable from browser) |
