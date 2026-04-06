# NovaFX — Multi-Asset Trading Signal Aggregation System

Live trading signal generation system deployed on Railway. Scans crypto, forex, and stocks across multiple data sources with automatic failover, confluence filtering, and Telegram delivery.

## Architecture

```
                        ┌─────────────────────────────────┐
                        │         DATA SOURCES            │
                        ├─────────┬──────────┬────────────┤
                        │ Binance │ Twelve   │ Alpaca     │
                        │ Kraken  │ Data     │ Finnhub    │
                        │ OKX     │ FMP      │ Alpha      │
                        │ KuCoin  │ CCompare │ Vantage    │
                        └────┬────┴────┬─────┴─────┬──────┘
                             │         │           │
                    ┌────────▼──┐ ┌────▼─────┐ ┌───▼────────┐
                    │  Crypto   │ │  Forex   │ │   Stock    │
                    │  Signals  │ │  Signals │ │  Signals   │
                    │ Freqtrade │ │ Scanner  │ │  Scanner   │
                    │ +Fallback │ │          │ │ +WS Standby│
                    └─────┬─────┘ └────┬─────┘ └─────┬──────┘
                          │            │              │
                          │   POST /signals/ingest    │
                          ▼            ▼              ▼
                    ┌─────────────────────────────────────┐
                    │          DISPATCHER (FastAPI)        │
                    │  Redis Streams · Confluence Engine   │
                    │  Webhook normalization (TV/Freqtrade)│
                    └──────────────┬──────────────────────┘
                                   │
                          Redis pub/sub: telegram:signals
                                   │
                    ┌──────────────▼──────────────────────┐
                    │         TELEGRAM BOT                 │
                    │  Desk routing · Health alerts        │
                    │  /status · /health · /help           │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       TELEGRAM CHANNELS              │
                    │  Crypto · Forex · Stocks · Portfolio │
                    └─────────────────────────────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `dispatcher` | 8000 | Central hub — signal ingestion, confluence, pub/sub |
| `crypto-signals` | 8080 | Freqtrade dry-run + CCXT fallback scanner |
| `forex-signals` | — | Four-tier forex scanner (TD/Finnhub/FMP/AV) |
| `stock-signals` | — | Alpaca + three-tier fallback, market hours gating |
| `telegram-bot` | — | Redis subscriber, Telegram delivery, bot commands |
| `novafx-app` | 8001 | Original signal bot (webhook, signal engine, DB) |
| `redis` | 6379 | Streams, pub/sub, candle cache |

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/UDG123/NovaFX.git
cd NovaFX
cp .env.example .env
# Edit .env with your API keys

# 2. Start all services
docker compose up --build

# 3. Test dispatcher health
curl http://localhost:8000/health

# 4. Send a test signal
curl -X POST http://localhost:8000/webhook/tradingview \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "action": "buy",
    "price": 185.50,
    "confidence": 0.8,
    "timeframe": "1h",
    "strategy": "test"
  }'

# 5. Check recent signals
curl http://localhost:8000/signals/recent/stocks

# 6. Check confluence
curl http://localhost:8000/signals/confluence/AAPL
```

## Railway Deployment

1. Push repo to GitHub
2. Create new Railway project
3. Add **Redis** plugin (Railway dashboard > New > Database > Redis)
4. Deploy each service:
   - New Service > GitHub Repo > Set **Root Directory** to `services/dispatcher`, `services/crypto-signals`, etc.
   - Set build command / Dockerfile path per service
5. Set environment variables for each service in Railway dashboard
6. Wire private networking:
   - `DISPATCHER_URL=http://dispatcher.railway.internal:8000`
   - `REDIS_URL` from Railway Redis plugin (auto-injected)
7. Generate public domain for dispatcher (Settings > Networking > Generate Domain)
8. Set `TELEGRAM_BOT_TOKEN` and channel IDs

## API Endpoints

### Dispatcher

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service health check |
| `POST` | `/signals/ingest` | Ingest a canonical Signal |
| `POST` | `/webhook/freqtrade` | Normalize Freqtrade webhook |
| `POST` | `/webhook/tradingview` | Normalize TradingView alert |
| `GET` | `/signals/recent/{asset_class}` | Recent signals from stream |
| `GET` | `/signals/confluence/{symbol}` | Manual confluence check |

### NovaFX App

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | App health + API usage |
| `POST` | `/webhook` | TradingView webhook (legacy) |
| `GET` | `/signals/stats` | Weekly signal statistics |

## Signal Schema

```json
{
  "signal_id": "uuid4",
  "timestamp": "2026-04-06T15:30:00Z",
  "source": "freqtrade-NovaFXCryptoStrategy",
  "action": "buy",
  "symbol": "BTCUSDT",
  "asset_class": "crypto",
  "confidence": 0.82,
  "price": 84250.00,
  "stop_loss": 80037.50,
  "take_profit": [88462.50],
  "timeframe": "1h",
  "strategy": "RSI-EMA-Crypto",
  "metadata": {
    "rsi": 28.5,
    "ema_fast": 84100.0,
    "ema_slow": 83800.0,
    "data_source": "ccxt-multi",
    "data_confidence": "HIGH"
  }
}
```

## Confluence Engine

The dispatcher evaluates confluence using weighted voting:

1. Signals arrive from independent sources within a **5-minute window**
2. Deduplicated by source group (keeps latest per source)
3. Weighted vote: `weight = source_weight * signal_confidence`
4. Consensus requires **2+ unique sources** agreeing on direction
5. Minimum **60% weighted confidence** to emit

Source weights: Freqtrade=1.0, TwelveData=1.0, Alpaca=1.0, Finnhub=0.9, TradingView=0.8, CryptoCompare=0.8

## Data Source Failover

### Crypto
```
Binance (CCXT) → Kraken → OKX → KuCoin → GateIO
    ↓ (all CCXT fail)
TwelveData (BTC/USD mapping)
    ↓
CryptoCompare (CCCAGG aggregate)
    ↓
WebSocket cache (Binance + Kraken streams)
```

### Forex
```
TwelveData (Grow 55, primary)
    ↓
Finnhub (OANDA:EUR_USD format, 60/min)
    ↓
Financial Modeling Prep (EURUSD, 250/day)
    ↓
Alpha Vantage (FX_INTRADAY, 25/day)
```

### Stocks
```
Alpaca (IEX feed, paper trading)
    ↓
TwelveData (shared plan)
    ↓
Finnhub (stock/candle, 60/min)
    ↓
Alpha Vantage (TIME_SERIES_INTRADAY, 25/day)
```

## Free API Signups

| Provider | URL | Tier | Limit |
|----------|-----|------|-------|
| Finnhub | https://finnhub.io | Free | 60 calls/min |
| CryptoCompare | https://min-api.cryptocompare.com | Free | 100K calls/mo |
| Alpaca | https://alpaca.markets | Paper | Unlimited |
| FMP | https://financialmodelingprep.com | Free | 250 calls/day |
| Alpha Vantage | https://www.alphavantage.co | Free | 25 calls/day |

All require email signup only — no KYC, no credit card.

## Cost Estimate

| Item | Monthly Cost |
|------|-------------|
| Railway (6 services) | $5-20 |
| Railway Redis | $5-10 |
| TwelveData Grow 55 | Existing plan |
| All other APIs | Free tier |
| **Total** | **~$10-30/mo** |

## Project Structure

```
NovaFX/
├── app/                          # Original NovaFX signal bot
│   ├── config.py
│   ├── main.py
│   ├── data/fetcher.py
│   ├── db/
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── signal_store.py
│   │   └── trade_monitor.py
│   ├── models/signals.py
│   ├── routes/
│   │   ├── webhook.py
│   │   └── stats.py
│   └── services/
│       ├── signal_engine.py
│       ├── signal_processor.py
│       ├── telegram.py
│       ├── regime.py
│       ├── htf_bias.py
│       ├── outcome_engine.py
│       ├── price_monitor.py
│       ├── pnl_calculator.py
│       ├── api_tracker.py
│       ├── bot_commands.py
│       └── bot_state.py
├── shared/                       # Shared resilience module
│   ├── config.py
│   ├── models.py
│   └── resilience.py
├── services/
│   ├── dispatcher/               # Central signal hub
│   │   ├── main.py
│   │   ├── confluence.py
│   │   ├── Dockerfile
│   │   └── railway.toml
│   ├── crypto-signals/           # Freqtrade + CCXT fallback
│   │   ├── config.json
│   │   ├── strategies/NovaFXCryptoStrategy.py
│   │   ├── fallback_scanner.py
│   │   ├── ws_monitor.py
│   │   ├── entrypoint.sh
│   │   ├── Dockerfile
│   │   └── railway.toml
│   ├── forex-signals/            # Four-tier forex scanner
│   │   ├── main.py
│   │   ├── sources.py
│   │   ├── signals.py
│   │   ├── Dockerfile
│   │   └── railway.toml
│   ├── stock-signals/            # Alpaca + three-tier fallback
│   │   ├── main.py
│   │   ├── sources.py
│   │   ├── signals.py
│   │   ├── ws_standby.py
│   │   ├── Dockerfile
│   │   └── railway.toml
│   └── telegram-bot/             # Redis subscriber + Telegram
│       ├── main.py
│       ├── Dockerfile
│       └── railway.toml
├── backtester/                   # Nova Backtester (companion)
├── docker-compose.yml
├── Dockerfile
├── Procfile
├── railway.toml
├── requirements.txt
├── .env.example
├── .gitignore
├── CLAUDE.md
└── README.md
```
