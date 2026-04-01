# NovaFX Security Model

## Authentication & Authorization

### Webhook Endpoint (`POST /webhook`)

Incoming TradingView (or any external) signals are authenticated via a shared
secret passed in the request body as the `secret` field.

- The secret is compared against `WEBHOOK_SECRET` using `hmac.compare_digest()`
  (constant-time comparison) to prevent timing-based attacks.
- If `WEBHOOK_SECRET` is set and the request secret is missing or wrong, the
  endpoint returns `401 Unauthorized`.
- The `secret` field is marked `exclude=True` on the Pydantic model so it is
  never included in any serialized response.

### Telegram Bot Commands (`/status`)

The `/status` command poller validates the sender's `chat.id` against
`TELEGRAM_CHAT_ID`. Messages from unauthorized chats are ignored and logged
as warnings. If `TELEGRAM_CHAT_ID` is empty, all chats are permitted (not
recommended for production).

### Telegram Alerts (outbound)

Outbound alerts are sent to the `TELEGRAM_CHAT_ID` specified in environment
variables. The bot token is only used in direct API calls and is never logged.

## Secrets Management

| Secret                | Env Var               | Usage                                |
|-----------------------|-----------------------|--------------------------------------|
| Telegram Bot Token    | `TELEGRAM_BOT_TOKEN`  | Telegram Bot API authentication      |
| Telegram Chat ID      | `TELEGRAM_CHAT_ID`    | Restricts who receives alerts/status |
| Webhook Secret        | `WEBHOOK_SECRET`      | Authenticates incoming webhooks      |
| TwelveData API Key    | `TWELVEDATA_API_KEY`  | Market data API authentication       |

### Secret Handling Rules

1. **No secrets in logs.** All `logger.error()` / `logger.warning()` calls that
   could surface httpx exceptions have been sanitized to log only the operation
   name and status code, never the full URL or exception repr (which may contain
   tokens or API keys as URL query parameters).

2. **No secrets in responses.** The `IncomingSignal.secret` field uses
   `Field(exclude=True)` so `model_dump()` and JSON serialization never include
   it.

3. **No secrets in git.** `.env` is listed in `.gitignore` and has been removed
   from git tracking. Only `.env.example` (with placeholder values) is committed.

4. **Environment-only configuration.** All secrets are read from environment
   variables or a `.env` file via `pydantic-settings`. No secrets are hardcoded.

## Input Validation

All external input passes through Pydantic models with strict field constraints:

| Field              | Validation                                       |
|--------------------|--------------------------------------------------|
| `symbol`           | 1-20 chars, alphanumeric (may contain `/` or `-`)|
| `price`            | Must be > 0                                      |
| `sl` / `tp`        | Must be > 0 when provided                        |
| `timeframe`        | Must match pattern `^\d+[mhd]$`                  |
| `source`           | 1-50 chars                                       |
| `indicator`        | Max 200 chars                                    |
| `confluence_count` | Must be >= 0 when provided                       |
| `action`           | Literal `"BUY"` or `"SELL"` only                 |

ProcessedSignal fields (`entry_price`, `stop_loss`, `take_profit`,
`risk_reward`, `position_size`, `risk_amount`) are all constrained to
non-negative values.

## Data Storage

- SQLite cache (`ohlcv_cache.db`) stores only OHLCV market data. No secrets or
  user data are written to disk. The `*.db` pattern is in `.gitignore`.
- All SQL queries use parameterized statements to prevent injection.

## Rate Limiting

- TwelveData API calls go through an async token-bucket rate limiter
  (60 requests/minute) to avoid triggering upstream 429 responses.
- HTTP 429 responses are handled with exponential backoff retries.

## Reporting Vulnerabilities

If you discover a security issue, please open a private report on the
repository rather than a public issue.
