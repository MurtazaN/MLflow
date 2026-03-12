# Docker Setup & Run Guide

Run the MLflow Wine Quality pipeline using Docker Compose.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

## Quick Start

```bash
# Build and run the full pipeline
docker compose up --build
```

This starts 4 services in order:
1. **mlflow-server** — MLflow UI on http://localhost:5000
2. **pipeline** — Trains RF + XGBoost, registers models
3. **model-server** — Serves the production model on port 5001
4. **inference** — Sends a test sample and prints the prediction

## Commands

| Command | What it does |
|---------|-------------|
| `docker compose up --build` | Build images + run everything (first time or after dependency changes) |
| `docker compose up` | Run everything (no rebuild — uses cached images) |
| `docker compose up -d` | Run in background (detached mode) |
| `docker compose logs -f pipeline` | Follow logs for a specific service |
| `docker compose logs` | View all logs |
| `docker compose down` | Stop and remove all containers |
| `docker compose down -v` | Stop + remove containers AND volumes (full cleanup) |

## After Running

- **MLflow UI** — http://localhost:5000 (stays running until `docker compose down`)
- **Model Server** — http://localhost:5001/invocations (POST only)
- **Inference** — Check output with `docker compose logs inference`

## Code Changes

**No rebuild needed for code changes** — `src/` is volume-mounted into the containers, so your local edits are reflected immediately. Just run:

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

# Run pipeline only (starts mlflow-server automatically via depends_on)
docker compose up pipeline

# Run inference against an already-running model server
docker compose up inference
```

## Cleanup

```bash
# Stop everything
docker compose down

# Stop + delete all data (MLflow DB, artifacts, logs)
docker compose down -v
rm -rf mlartifacts logs
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Port 5000 already in use | `kill $(lsof -t -i:5000)` or `docker compose down` first |
| Port 5001 already in use | `kill $(lsof -t -i:5001)` or `docker compose down` first |
| Pipeline can't reach MLflow | Check `docker compose logs mlflow-server` — healthcheck may be failing |
| Model server fails to start | Pipeline may not have finished — check `docker compose logs pipeline` |
| Stale containers | `docker compose down -v && docker compose up --build` |
