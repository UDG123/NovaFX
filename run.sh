#!/usr/bin/env bash
set -euo pipefail

# Load environment variables from .env
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "[NovaFX] Loaded .env"
else
    echo "[NovaFX] WARNING: .env not found — copy .env.example to .env"
    exit 1
fi

cleanup() {
    echo ""
    echo "[NovaFX] Shutting down..."
    kill 0 2>/dev/null
    wait 2>/dev/null
    echo "[NovaFX] All services stopped."
}
trap cleanup EXIT INT TERM

echo "[NovaFX] Starting Signal Bot on port 8000..."
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
PID_SIGNAL=$!

echo "[NovaFX] Starting Backtester on port 8001..."
uvicorn backtester.main:app --reload --host 0.0.0.0 --port 8001 &
PID_BACKTEST=$!

echo "[NovaFX] Both services running:"
echo "  Signal Bot   → http://localhost:8000"
echo "  Backtester   → http://localhost:8001"
echo ""
echo "Press Ctrl+C to stop all services."

wait
