# NovaFX — Claude Project Context

## Project Overview
NovaFX is a live trading signal generation system deployed on Railway.
Primary output: formatted trade signals sent to a Telegram channel for subscribers.

## Asset Coverage
- Forex: EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CAD, USD/CHF, NZD/USD + crosses (EUR/GBP, EUR/JPY, GBP/JPY)
- Crypto: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, XRP/USDT
- Stocks: AAPL, MSFT, NVDA, TSLA, SPY, QQQ
- Commodities: XAU/USD (Gold), XAG/USD (Silver)
- Indices: US30, US500 (SPX500), NAS100

## Tech Stack
- Framework: FastAPI (Python 3.11+)
- Scheduler: APScheduler (15-min signal scan jobs)
- HTTP Client: httpx (async API calls)
- Data Science: pandas, ta (technical analysis)
- Live Data: TwelveData API (Grow55 plan) — LIVE ONLY, never for backtesting
- Backtest Data: CCXT/Binance (crypto), Dukascopy (forex), yfinance (equities)
- Config: pydantic-settings (.env / Railway env vars)
- Deployment: Railway (auto-deploy from GitHub main branch)
- Signal Output: Telegram Bot API (httpx direct calls)

## Repo Structure
app/                    — NovaFX Signal Bot (main Railway service)
  config.py             — Settings via pydantic-settings
  main.py               — FastAPI app + lifespan scheduler
  data/fetcher.py       — TwelveData OHLCV fetcher + SQLite cache
  models/signals.py     — IncomingSignal + ProcessedSignal models
  routes/webhook.py     — /health and /webhook endpoints
  services/
    signal_engine.py    — Strategy runner + confluence filter
    signal_processor.py — SL/TP/TP1/TP2/TP3 calculator
    telegram.py         — Signal formatter + Telegram sender
    bot_commands.py     — /status command poller
    bot_state.py        — Singleton state tracker
backtester/             — Nova Backtester (companion system, not deployed)

## Hard Architecture Rules
RULE 1: TwelveData is EXCLUSIVELY for live signal generation. Never for backtesting.
RULE 2: Backtest data pipeline is FULLY DECOUPLED from live trading.
RULE 3: All credentials in Railway env vars. Never hardcode API keys.
RULE 4: Nova Backtester (backtester/) is a companion system — separate deployment.
RULE 5: Signal engine only emits when MIN_CONFLUENCE = 2 strategies agree.

## Live Strategies (signal_engine.py)
1. EMA 9/21 Cross — crossover detection on close prices
2. RSI 14 Reversal — oversold (<30) BUY, overbought (>70) SELL
3. MACD Cross — MACD line crossing signal line
4. Bollinger Band Reversion — price at upper/lower band extremes
Minimum 2 strategies must agree (confluence) before a signal is emitted.

## Risk Management Rules
MAX RISK: 2% of account per trade (DEFAULT_RISK_PCT env var)
MIN R:R: 1:2 (TP1 is always 2× the SL distance)
TP LEVELS: TP1 = 1× reward, TP2 = 2× reward, TP3 = 3× reward
MARKET SL/TP DEFAULTS (signal_processor.py MARKET_CONFIG):
  Forex:      SL=0.3%, TP1=0.6%, TP2=1.2%, TP3=1.8%
  Crypto:     SL=1.5%, TP1=3.0%, TP2=6.0%, TP3=9.0%
  Indices:    SL=0.5%, TP1=1.0%, TP2=2.0%, TP3=3.0%
  Commodities:SL=0.8%, TP1=1.6%, TP2=3.2%, TP3=4.8%

## Signal Format Template (telegram.py)
⚡ NOVAFX SIGNAL

🔷 [PAIR]  📈 BUY  or  📉 SELL

📍 Entry: [price]
🔴 Stop Loss: [price]  ([diff])

✅ TP1: [price]  ([diff])
✅ TP2: [price]  ([diff])
✅ TP3: [price]  ([diff])

⚖️ R:R → 1:X  |  Risk: $X
📊 Timeframe: [TF]
🧠 Strategy: [strategy name]
📅 [timestamp]
────────────────
⚠️ Risk max 1-2% per trade. Not financial advice.

## Environment Variables (Railway)
TELEGRAM_BOT_TOKEN    — Telegram bot token from @BotFather
TELEGRAM_CHAT_ID      — Channel or chat ID to send signals to
TWELVEDATA_API_KEY    — TwelveData Grow55 API key
WEBHOOK_SECRET        — Secret for TradingView webhook authentication
DEFAULT_RISK_PCT      — Risk per trade as % of account (default: 1.0)
ACCOUNT_BALANCE       — Account size for position sizing (default: 10000)
SIGNAL_ENGINE_ENABLED — Enable/disable automated scanning (default: true)
SIGNAL_ENGINE_INTERVAL_MINUTES — Scan frequency in minutes (default: 15)
PORT                  — Server port (default: 8000, Railway sets this)

## Deployment (Railway)
- Procfile: web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
- Health check: GET /health (returns 200 healthy / 503 unhealthy)
- railway.toml: nixpacks builder, restart on failure, max 3 retries
- Auto-deploys on push to main branch

## Code Standards
- Python 3.11+ type hints on all functions
- Async/await for all I/O (httpx AsyncClient)
- pydantic models for all data structures
- pydantic-settings for all config (never os.environ directly)
- logging module only — no print() in production
- All monetary values as float, rounded to 6 decimal places max
- SQL queries use parameterized statements (SQLite cache)
