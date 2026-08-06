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
runs. Current winners (2026-08-05): BTC → Coinbase Exchange (4034 bars from
2015-07-20), ETH → Coinbase Exchange (3503 from 2017-01-01), SOL → Coinbase
Exchange (1875 from 2021-06-17), HYPE / ONDO / MON → Kraken, XPL → KuCoin.
The 4H frame may come from a different venue than the daily, because the venue
with the deepest history does not always offer a 4-hour granularity.

**Binance never wins here.** GitHub's runners sit in a region Binance blocks, so
every `binance:` candidate returns `451 Service unavailable from a restricted
location` and is skipped. The candidates stay in the list because they cost one
failed request and would start winning again the day that changes — but do not
expect Binance depth from this mirror. Bybit answers `403` from CloudFront for
the same reason.

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

Alerts are deduplicated on the set of failing symbols: `status.json` records
which set the last alert covered (`alerted_failed`), and Telegram is pinged only
when that set changes — a new symbol breaks, or everything recovers. A vendor
outage lasting a week is one message, not seven. The run itself still fails
every day, so the red X in Actions stays honest.

Secrets used: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. Without them the run
still works and just logs a warning instead of alerting.

## What does not belong here

Positions, theses, journals, the Twitter JSONL archive. Public market data only.

## tools/chart-analysis.py (read-only αντίγραφο)

Το analysis script του trading-cockpit, δημοσιευμένο εδώ ώστε ένα fresh
session ΧΩΡΙΣ πρόσβαση στο private repo να μπορεί να τρέξει την ανάλυση
(«PAT-free analyze»): δεδομένα από `data/`, script από εδώ.

- **Canonical: το private trading-cockpit** (`services/chart-scan/`) — εκεί
  γίνονται reviews/tests/pins. Αυτό εδώ είναι one-way αντίγραφο.
- **Sync rule:** κάθε commit του cockpit που αλλάζει το chart-analysis.py
  οφείλει στο ίδιο batch να αντιγράψει το αρχείο κι εδώ. Το heartbeat του
  cockpit συγκρίνει καθημερινά τα sha256 και ειδοποιεί σε drift — αν το
  δεις να αποκλίνει, ισχύει το cockpit.
- Καμία προσωπική πληροφορία: μόνο ο αλγόριθμος (EMA/S/R/verdicts/pairs).
  Τα χειροκίνητα levels (`--levels`) μένουν στο private repo — χωρίς το
  αρχείο, το section παραλείπεται σιωπηλά.

```
pip install pandas numpy
python3 tools/chart-analysis.py analyze SOL --csv-dir data --out /tmp/out --no-charts
```
