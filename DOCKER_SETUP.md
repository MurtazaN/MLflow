# Docker Setup & Run Guide

Run the MLflow Wine Quality pipeline using Docker Compose.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

## Environment File Setup

Create a `.env` file in the project root (not tracked by git):

```bash
# .env
MLFLOW_VERSION=3.10.1
MLFLOW_SERVER_ALLOWED_HOSTS=mlflow-server,mlflow-server:5000,localhost,localhost:5000,0.0.0.0
```

| Variable | Purpose |
|----------|---------|
| `MLFLOW_VERSION` | MLflow version for the tracking server Docker image (must match `requirements.txt`) |
| `MLFLOW_SERVER_ALLOWED_HOSTS` | Hosts allowed to connect to the MLflow tracking server |

## Services

| Service | Port | Description |
|---------|------|-------------|
| `mlflow-server` | 5000 | MLflow tracking server (official image + SQLite) |
| `pipeline` | — | Trains RF + XGBoost, registers models, then exits |
| `model-server` | 5001 | Serves the production model via MLflow |
| `backend-api` | 5002 | FastAPI inference service (`src/fastapi-app.py`) |

## Quick Start

```bash
# Build and run the full pipeline
docker compose up --build
```

Startup order (handled automatically by `depends_on`):

1. **mlflow-server** starts → healthcheck passes
2. **pipeline** runs → trains models, registers to MLflow, exits
3. **model-server** starts → serves the production model
4. **backend-api** starts → FastAPI app on http://localhost:5002

## Accessing the Services

| URL | What you see |
|-----|-------------|
| http://localhost:5000 | MLflow UI — experiments, runs, model registry |
| http://localhost:5002 | FastAPI browser UI — wine quality predictions |
| http://localhost:5002/docs | FastAPI auto-generated Swagger docs |

### Testing the API

```bash
# Browser UI
open http://localhost:5002

# JSON API
curl -X POST http://localhost:5002/predict
```

## Commands

| Command | What it does |
|---------|-------------|
| `docker compose up --build` | Build images + run everything (first time / dependency changes) |
| `docker compose up` | Run everything (no rebuild) |
| `docker compose up -d` | Run in background (detached) |
| `docker compose logs -f pipeline` | Follow logs for a specific service |
| `docker compose logs` | View all logs |
| `docker compose down` | Stop and remove all containers |
| `docker compose down -v` | Stop + remove containers AND volumes (full cleanup) |

## Code Changes

**No rebuild needed for code changes** — `src/` is volume-mounted. Just restart:

```bash
docker compose down
docker compose up
```

**Rebuild only when dependencies change** (requirements.txt):

```bash
docker compose up --build
```

## Run Individual Services

```bash
# Only start MLflow UI
docker compose up mlflow-server

# Run pipeline only (starts mlflow-server automatically)
docker compose up pipeline

# Start model server + FastAPI after pipeline has run
docker compose up model-server backend-api
```

## Cleanup

```bash
# Stop everything
docker compose down

# Full cleanup (containers + volumes + local artifacts)
docker compose down -v
rm -rf mlartifacts logs
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Port already in use | `kill $(lsof -t -i:5000)` or `docker compose down` first |
| mlflow-server unhealthy | Check `docker compose logs mlflow-server` — may need longer start_period |
| Pipeline can't reach MLflow | Verify MLFLOW_SERVER_ALLOWED_HOSTS in `.env` includes the service name |
| Model server fails | Pipeline may not have finished — check `docker compose logs pipeline` |
| Version mismatch errors | Ensure `MLFLOW_VERSION` in `.env` matches version in `requirements.txt` |
| `.env` not found | Create it — see [Environment File Setup](#environment-file-setup) above |
| Stale state | `docker compose down -v && docker compose up --build` |
