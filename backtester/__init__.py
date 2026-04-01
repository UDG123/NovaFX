from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backtester.app.core.state import BacktesterState
from backtester.app.data.results_store import get_recent_results

app = FastAPI(
    title="NovaFX Backtester",
    description="Backtesting engine for NovaFX strategies",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {
        "service": "NovaFX Backtester",
        "version": "0.1.0",
        "status": "ready",
    }


@app.get("/health")
async def health():
    state = BacktesterState.get()

    if not state.scheduler_running():
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": "scheduler not running",
                "service": "NovaFX Backtester",
            },
        )

    next_run = state.get_next_run_time()
    return {
        "status": "healthy",
        "service": "NovaFX Backtester",
        "uptime_seconds": state.uptime_seconds(),
        "last_backtest_cycle": state.last_cycle_time.isoformat() if state.last_cycle_time else None,
        "scheduler_next_run": next_run.isoformat() if next_run else None,
    }


@app.get("/history")
async def history(limit: int = 10):
    results = get_recent_results(min(limit, 100))
    return {
        "count": len(results),
        "results": [r.model_dump() for r in results],
    }
