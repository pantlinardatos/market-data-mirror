#!/usr/bin/env python3
"""
market-data-mirror — fetch.py

Pulls OHLCV for a fixed symbol list and writes CSVs that match the
chart-analysis.py contract exactly:

    data/<SYM>_1d.csv    date,open,high,low,close,volume   (UTC, ascending)
    data/<SYM>_4h.csv    same header, date = YYYY-MM-DDTHH:MM

Design rules (see Vault System/system-redesign-2026-07.md, Κίνηση 2):
  * per-symbol isolation — one broken symbol never kills the run
  * the still-forming candle is always dropped (yesterday's close is the
    contract; keeps output stable so commit-only-on-change works)
  * deterministic rounding, so an unchanged day produces a byte-identical file
  * a run that had any failure writes status.json and exits 1 AFTER the good
    data has already been written, so the workflow commits first, alerts second
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

COLS = ["date", "open", "high", "low", "close", "volume"]

# ---------------------------------------------------------------- symbols ---
# name = the filename stem AND what the analysis engine is asked for.
# yahoo/ccxt fields are the vendor-side tickers.
STOCKS = {
    "TSLA": "TSLA",
    "PLTR": "PLTR",
    "SPY":  "SPY",
    "QQQ":  "QQQ",
    "SOFI": "SOFI",
    "AMD":  "AMD",
    "MSTR": "MSTR",
    "DXY":  "DX-Y.NYB",   # ICE dollar index on Yahoo
    "SPX":  "^GSPC",      # Yahoo's S&P 500 cash index (the plan writes ^SPX)
    "NDX":  "^NDX",       # Nasdaq-100 cash index

    # Commodities via Yahoo's continuous front-month futures, NOT spot ETFs.
    # The futures series is the deeper and more stable feed: GC=F goes back to
    # 2000 with an unbroken daily tape, while the ETF proxies (GLD 2004, SLV
    # 2006, CPER 2011, BNO 2010) start later, carry expense-ratio drift, and
    # track the metal rather than quote it. Yahoo rolls the front month itself,
    # so the stem stays constant and the CSV keeps appending across expiries.
    "GOLD":   "GC=F",     # COMEX gold front-month
    "SILVER": "SI=F",     # COMEX silver front-month
    "COPPER": "HG=F",     # COMEX copper front-month
    "BRENT":  "BZ=F",     # ICE Brent crude front-month
}

# ccxt candidates per symbol. Every venue that answers is tried and the one
# with the LONGEST daily history wins — venues differ by years (Binance BTC
# starts 2017-08, Coinbase 2015-07) and hardcoding one silently truncates the
# record. Longest-wins is deterministic, so the winner does not flip run to run.
CRYPTO = {
    "BTC":  ["coinbaseexchange:BTC/USD", "coinbase:BTC/USD",
             "binance:BTC/USDT", "kraken:BTC/USD"],
    "SOL":  ["coinbaseexchange:SOL/USD", "coinbase:SOL/USD",
             "binance:SOL/USDT", "kraken:SOL/USD"],
    "HYPE": ["binance:HYPE/USDT", "coinbaseexchange:HYPE/USD",
             "coinbase:HYPE/USD", "kraken:HYPE/USD"],
    "ETH":  ["coinbaseexchange:ETH/USD", "coinbase:ETH/USD",
             "binance:ETH/USDT", "kraken:ETH/USD"],
    "ONDO": ["coinbaseexchange:ONDO/USD", "coinbase:ONDO/USD",
             "binance:ONDO/USDT", "kraken:ONDO/USD"],
    # MON/XPL are recent listings — the majors may not carry them yet, so the
    # candidate list is widened to the venues that list new tokens first.
    # Per-symbol isolation means a token nobody lists just fails on its own.
    "MON":  ["binance:MON/USDT", "bybit:MON/USDT", "kucoin:MON/USDT",
             "gateio:MON/USDT", "kraken:MON/USD", "coinbaseexchange:MON/USD"],
    "XPL":  ["binance:XPL/USDT", "bybit:XPL/USDT", "kucoin:XPL/USDT",
             "gateio:XPL/USDT", "kraken:XPL/USD", "coinbaseexchange:XPL/USD"],
}

# Kraken's OHLC endpoint is capped at 720 candles whatever you ask for. That is
# useless for daily (2 years) but ideal for weekly (~13.8 years) — deeper than
# any venue's daily. chart-analysis.py looks for exactly this filename and
# prefers it over daily-derived weekly when it reaches further back.
WEEKLY_VENUES = ["kraken:{q}/USD", "coinbaseexchange:{q}/USD"]

FOURH_LIMIT = 1000        # ~165 days — plenty for EMA200 on 4H, bounded size


# ------------------------------------------------------------------ utils ---
def _fmt(df: pd.DataFrame, intraday: bool) -> pd.DataFrame:
    """Normalise to the CSV contract: UTC ascending, fixed cols, stable rounding."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    out = pd.DataFrame({
        "date": df.index.strftime("%Y-%m-%dT%H:%M" if intraday else "%Y-%m-%d"),
        "open": df["open"].round(6),
        "high": df["high"].round(6),
        "low": df["low"].round(6),
        "close": df["close"].round(6),
        "volume": df["volume"].round(8),
    })
    return out[COLS].dropna(subset=["open", "high", "low", "close"])


def _drop_forming_session(df: pd.DataFrame) -> pd.DataFrame:
    """Daily bars for an exchange with a session (stocks), not a 24h tape.

    A US session's daily bar is final once the bell rings — 20:00 UTC in EDT,
    21:00 in EST — even though the UTC day has hours left. Treating it as
    'still forming' until midnight throws away same-day closes for any run
    after the close, which is exactly when someone asks for an analysis.
    21:00 is the safe year-round cutoff; at the 08:17 cron it changes nothing.
    """
    if df.empty:
        return df
    now = pd.Timestamp(datetime.now(timezone.utc))
    today = now.normalize()
    if df.index[-1].normalize() >= today and now.hour < 21:
        return df[df.index.normalize() < today]
    return df


def _drop_forming(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Drop the candle that has not closed yet (24h tape: crypto, intraday)."""
    if df.empty:
        return df
    now = pd.Timestamp(datetime.now(timezone.utc))
    step = {"1d": pd.Timedelta(days=1),
            "4h": pd.Timedelta(hours=4),
            "1w": pd.Timedelta(weeks=1)}[tf]
    return df[df.index + step <= now]


def _write(sym: str, tf: str, df: pd.DataFrame) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{sym}_{tf}.csv")
    df.to_csv(path, index=False, lineterminator="\n")
    return f"{sym}_{tf}: {len(df)} rows {df['date'].iloc[0]} → {df['date'].iloc[-1]}"


# ------------------------------------------------------------------ stocks --
def fetch_stock(name: str, ticker: str) -> list:
    import yfinance as yf
    notes = []

    daily = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
    if daily.empty:
        raise RuntimeError(f"yfinance returned no daily rows for {ticker}")
    daily.columns = [c.lower() for c in daily.columns]
    daily.index = pd.to_datetime(daily.index, utc=True)
    daily = _drop_forming_session(daily)
    notes.append(_write(name, "1d", _fmt(daily, intraday=False)))

    # 4H is best-effort: Yahoo caps intraday history, and there is no native 4h.
    try:
        h1 = yf.Ticker(ticker).history(period="60d", interval="1h", auto_adjust=False)
        if not h1.empty:
            h1.columns = [c.lower() for c in h1.columns]
            h1.index = pd.to_datetime(h1.index, utc=True)
            h4 = h1.resample("4h", label="left", closed="left").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}).dropna(subset=["open"])
            h4 = _drop_forming(h4, "4h")
            if not h4.empty:
                notes.append(_write(name, "4h", _fmt(h4, intraday=True)))
    except Exception as exc:                                    # noqa: BLE001
        notes.append(f"{name}_4h: skipped ({type(exc).__name__}: {exc})")

    return notes


# ------------------------------------------------------------------ crypto --
def _ohlcv(ex, pair: str, timeframe: str, limit: int, since=None) -> pd.DataFrame:
    rows = ex.fetch_ohlcv(pair, timeframe=timeframe, since=since, limit=limit)
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
    df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
    return df[["open", "high", "low", "close", "volume"]]


def _exchange(ex_id: str):
    import ccxt
    ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
    ex.load_markets()
    return ex


PAGE = 300             # Coinbase caps the response near this; Binance is happy
MAX_PAGES = 200        # Coinbase answers ~100 rows/page ⇒ 11 years needs ~40

# An empty page does NOT mean "end of data" — venues honour the requested
# window and return nothing for a range that predates the listing. Giving up on
# the first empty page is how you silently truncate BTC to Binance's 2017
# instead of Coinbase's 2015. Blind window-sliding overshoots into the future
# (which Coinbase rejects outright), so probe a short ladder of plausible
# listing years instead and page forward from the first one that answers.
START_LADDER = ["2010-01-01", "2013-01-01", "2015-01-01", "2017-01-01",
                "2019-01-01", "2021-01-01", "2023-01-01", "2025-01-01"]


def _paged_daily(ex, pair: str) -> pd.DataFrame:
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    since, seed = None, pd.DataFrame()
    for probe in START_LADDER:
        cand = ex.parse8601(probe + "T00:00:00Z")
        if cand >= now_ms:
            break
        seed = _ohlcv(ex, pair, "1d", PAGE, cand)
        if not seed.empty:
            since = cand
            break
    if since is None:
        return pd.DataFrame()

    frames = [seed]
    since = int(seed.index[-1].timestamp() * 1000) + 86_400_000
    for _ in range(MAX_PAGES):
        if since >= now_ms:
            break
        chunk = _ohlcv(ex, pair, "1d", PAGE, since)
        if chunk.empty:
            break
        frames.append(chunk)
        nxt = int(chunk.index[-1].timestamp() * 1000) + 86_400_000
        if nxt <= since:
            break
        since = nxt

    return pd.concat(frames)


def fetch_crypto(name: str, candidates: list) -> list:
    errors, best = [], None                   # best = (rows, ex_id, pair, ex, df)

    for cand in candidates:
        ex_id, pair = cand.split(":", 1)
        try:
            ex = _exchange(ex_id)
            if pair not in ex.markets:
                errors.append(f"{ex_id}: no {pair}")
                continue
            daily = _drop_forming(_paged_daily(ex, pair), "1d")
            if daily.empty:
                errors.append(f"{ex_id}: no daily candles")
                continue
            if best is None or len(daily) > best[0]:
                best = (len(daily), ex_id, pair, ex, daily)
        except Exception as exc:                                # noqa: BLE001
            errors.append(f"{ex_id}: {type(exc).__name__}: {exc}")

    if best is None:
        raise RuntimeError(f"all sources failed for {name}: {'; '.join(errors)}")

    _, ex_id, pair, ex, daily = best
    notes = [f"[{ex_id}] " + _write(name, "1d", _fmt(daily, intraday=False))]

    # 4H: prefer the venue that won the daily, but it may not offer a 4h
    # granularity at all (Coinbase Exchange goes 1h → 6h). Fall back to the
    # other candidates — a different venue costs a few basis points on a
    # short-horizon frame, which is immaterial next to having no frame.
    h4_order = [f"{ex_id}:{pair}"] + [c for c in candidates
                                      if not c.startswith(ex_id + ":")]
    for cand in h4_order:
        h_id, h_pair = cand.split(":", 1)
        try:
            hex_ = ex if h_id == ex_id else _exchange(h_id)
            if h_pair not in hex_.markets or "4h" not in (hex_.timeframes or {}):
                continue
            h4 = _drop_forming(_ohlcv(hex_, h_pair, "4h", FOURH_LIMIT), "4h")
            if not h4.empty:
                notes.append(f"[{h_id}] " + _write(name, "4h",
                                                   _fmt(h4, intraday=True)))
                break
        except Exception as exc:                                # noqa: BLE001
            errors.append(f"{h_id} 4h: {type(exc).__name__}: {exc}")

    # Weekly, best-effort: only worth writing if it reaches further back than
    # the daily file, because that is the only case chart-analysis.py uses it.
    for tmpl in WEEKLY_VENUES:
        w_id, w_pair = tmpl.format(q=name).split(":", 1)
        try:
            wex = _exchange(w_id)
            if w_pair not in wex.markets:
                continue
            wk = _drop_forming(_ohlcv(wex, w_pair, "1w", 720), "1w")
            if not wk.empty and wk.index[0] < daily.index[0]:
                notes.append(f"[{w_id}] " + _write(
                    name, "1w_kraken", _fmt(wk, intraday=False)))
            break
        except Exception:                                       # noqa: BLE001
            continue

    if errors:
        notes.append(f"(skipped: {'; '.join(errors)})")
    return notes


# -------------------------------------------------------------------- main ---
def main() -> int:
    ap = argparse.ArgumentParser(description="fetch OHLCV into data/*.csv")
    ap.add_argument("--symbols", default="",
                    help="comma-separated subset; empty = all")
    args = ap.parse_args()

    wanted = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    jobs = []
    for name, ticker in STOCKS.items():
        if not wanted or name in wanted:
            jobs.append((name, lambda n=name, t=ticker: fetch_stock(n, t)))
    for name, cands in CRYPTO.items():
        if not wanted or name in wanted:
            jobs.append((name, lambda n=name, c=cands: fetch_crypto(n, c)))

    unknown = [s for s in wanted
               if s not in STOCKS and s not in CRYPTO]
    ok, failed = {}, {}
    for name, fn in jobs:
        try:
            ok[name] = fn()
            for line in ok[name]:
                print(f"OK   {line}", flush=True)
        except Exception as exc:                                # noqa: BLE001
            failed[name] = f"{type(exc).__name__}: {exc}"
            print(f"FAIL {name}: {failed[name]}", file=sys.stderr, flush=True)
            traceback.print_exc()

    for s in unknown:
        failed[s] = "unknown symbol (not in STOCKS or CRYPTO)"
        print(f"FAIL {s}: unknown symbol", file=sys.stderr, flush=True)

    status = {
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ok": sorted(ok),
        "failed": failed,
    }
    with open(os.path.join(os.path.dirname(DATA_DIR), "status.json"), "w",
              encoding="utf-8") as fh:
        json.dump(status, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    print(f"\n{len(ok)} ok, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
