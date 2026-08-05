# market-data-mirror

Flat-file OHLCV mirror. A GitHub Action pulls market data once a day and commits
plain CSVs, so that any environment which can reach `raw.githubusercontent.com`
can read market data with a single `pd.read_csv` — no API keys, no rate limits,
no fetch grind.

Built because the Claude cloud sandbox has allowlist egress: Python reaches
`api.github.com` and `raw.githubusercontent.com` and nothing else. Every
market-data host is blocked.

## Read it

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/<user>/market-data-mirror/main/data"
tsla = pd.read_csv(f"{BASE}/TSLA_1d.csv", parse_dates=["date"])
```

## Contract

```
data/<SYM>_1d.csv          full daily history
data/<SYM>_4h.csv          recent 4-hour candles
data/<SYM>_1w_kraken.csv   weekly — crypto only, and only when it reaches
                           further back than the daily file does
```

All: `date,open,high,low,close,volume`, UTC, ascending, no index column.
`1d`/`1w` dates are `YYYY-MM-DD`; `4h` dates are `YYYY-MM-DDTHH:MM`.

This is exactly the contract `chart-analysis.py` expects, so mirror CSVs drop
straight into its `--csv-dir` with no code change. Monthly is derived from daily
by that script; weekly is too, *unless* the `_1w_kraken` file goes back further.

The odd filename is deliberate — it is the name the analysis engine already
looks for. Kraken's OHLC endpoint is capped at 720 candles no matter what you
ask, which makes it useless for daily (2 years) and ideal for weekly (~13.8
years): BTC weekly reaches 2013-10, two years past even Coinbase's daily.

**A candle that has not closed is never written.** For crypto that means the
current UTC day is excluded. For stocks the daily bar is released at the closing
bell (20:00 UTC in EDT, 21:00 in EST), not at midnight, so a same-day analysis
after the close does get that day — the cutoff is 21:00 UTC, safe year-round.
This is what makes an unchanged day produce a byte-identical file and no commit;
two back-to-back runs were verified byte-identical across all 20 files.

## Symbols

| | |
|---|---|
| Stocks (yfinance) | TSLA, PLTR, SPY, QQQ, SOFI, AMD, MSTR |
| Indices (yfinance) | DXY (`DX-Y.NYB`), SPX (`^GSPC`), NDX (`^NDX`) |
| Commodities (yfinance futures) | GOLD (`GC=F`), SILVER (`SI=F`), COPPER (`HG=F`), BRENT (`BZ=F`) |
| Crypto (ccxt) | BTC, ETH, SOL, HYPE, ONDO, MON, XPL — every candidate venue is tried and the one with the **longest** daily history wins |

Commodities are quoted from Yahoo's continuous **front-month futures**, not spot
ETFs: the futures tape is deeper and cleaner (`GC=F` runs from 2000, against
GLD's 2004 / SLV's 2006 / CPER's 2011 / BNO's 2010), it has no expense-ratio
drift, and Yahoo rolls the contract itself so the filename stem never changes.

The stem *is* the contract — `chart-analysis.py` asks for `GOLD_1d.csv`,
`NDX_1d.csv` and so on by name, so the left-hand column above must stay exactly
as written. Add a symbol by editing the `STOCKS` / `CRYPTO` dicts at the top of
`fetch.py`; nothing else in the pipeline needs to know.

Longest-wins matters more than it sounds: hardcoding Binance would start BTC at
2017-08 instead of Coinbase's 2015-07 and quietly drop 759 sessions, including
the 2015 bottom. It is deterministic, so the winning venue does not flip between
runs. Current winners: BTC → Coinbase Exchange (4021 daily bars from 2015-07-20),
SOL → Binance (2172 from 2020-08-11), HYPE → Kraken (Binance has no HYPE/USDT).
The 4H frame may come from a different venue than the daily, because the venue
with the deepest history does not always offer a 4-hour granularity.

4H for stocks is best-effort: Yahoo caps intraday history at 60 days and has no
native 4-hour interval, so it is resampled from 1H. Crypto 4H comes straight
from the exchange.

## Schedule

Daily at **08:17 UTC** (off-peak minute — GitHub jitters and drops on-the-hour
cron). Manual runs via **Actions → fetch → Run workflow**, optionally with a
comma-separated `symbols` subset. On-demand from a session: `POST` to
`workflow_dispatch` on `api.github.com` with a PAT carrying `actions:write`.

## Failure behaviour

Symbols are fetched independently — one broken vendor never costs the others.
The run writes `status.json`, **commits whatever succeeded**, and only then
fails and fires a Telegram alert. Stale data is therefore always the last known
good snapshot, never a truncated one.

Secrets used: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Without them the run
still works and just logs a warning instead of alerting.

## What does not belong here

Positions, theses, journals, the Twitter JSONL archive. Public market data only.
