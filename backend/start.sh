#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
uv run alembic upgrade head
exec uv run uvicorn cs2_predictor.api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
