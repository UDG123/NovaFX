from fastapi import FastAPI

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
    return {"status": "healthy"}
