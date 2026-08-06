#!/usr/bin/env python3
"""
Trading Copilot — chart-analysis.py (v2, 2026-08-06 — v1 2026-07-23 + scan hooks)

Data-first chart analysis: OHLCV CSVs -> charts (4H/D/W/M) with EMA 21/50/200
+ algorithmic S/R levels + multi-timeframe text verdict.

KISS: one file, three subcommands. No dashboard, no DB, no ML.

  plan       print the WebFetch URLs a session must fetch for a symbol
  normalize  raw Kraken/Coinbase JSON dir -> normalized CSVs
  analyze    CSVs -> PNG charts + markdown report

CSV contract (produced by `normalize` or by the fetch agent):
  <SYM>_1d.csv          full daily history   date,open,high,low,close,volume
  <SYM>_4h.csv          ~720 4H candles      (optional; crypto)
  <SYM>_1w_kraken.csv   full weekly history  (optional; extends the daily tail)

Execution environment: Anthropic cloud sandbox (Cowork session). Direct Python
egress to market-data hosts is BLOCKED — fetching happens via the session's
WebFetch tool (see `plan`). Script itself never touches the network.

Not an advisor: output is analysis of price vs EMAs + levels. No trade calls.

v2 additions (migration στο trading-cockpit, 6/8/2026 — S/R λογική ΑΘΙΚΤΗ):
  --json PATH    structured output για το morning-scan (verdicts, levels, pairs)
  --as-of DATE   cutoff — ανάλυση όπως θα φαινόταν εκείνη τη μέρα (ιστορικά verdicts)
  --no-charts    χωρίς PNGs (scan/CI mode — δεν χρειάζεται matplotlib)
  --all-levels   default output = top 3 levels ανά πλευρά· με το flag όλα όσα
                 υπολογίζει το (αμετάβλητο) sr_levels
  pairs rule     relative-strength ratios σε D/W/M (crypto→/BTC, stock→/SPX+/NDX,
                 commodity→/Gold) από τις mirror daily σειρές — verdict ανά TF
"""

import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config ----

EMAS = (21, 50, 200)

# symbol -> (kraken pair, coinbase product or None)
SYMBOLS = {
    "BTC":  ("XBTUSD",  "BTC-USD"),
    "ETH":  ("ETHUSD",  "ETH-USD"),
    "SOL":  ("SOLUSD",  "SOL-USD"),
    "HYPE": ("HYPEUSD", None),
    "XRP":  ("XRPUSD",  "XRP-USD"),
    "ADA":  ("ADAUSD",  "ADA-USD"),
    "LINK": ("LINKUSD", "LINK-USD"),
    "DOGE": ("XDGUSD",  "DOGE-USD"),
    "ONDO": ("ONDOUSD", "ONDO-USD"),
}

# asset class → pairs rule (mirror-symbols-and-analysis-rules.md + v2 §Γ):
# crypto → ratio vs BTC · stock → vs SPX & NDX · metal/commodity → vs Gold,
# υπολογισμένα σε D/W/M από τις mirror daily σειρές (καμία νέα αποθήκευση).
ASSET_CLASS = {
    "crypto": {"BTC", "ETH", "SOL", "HYPE", "XRP", "ADA", "LINK", "DOGE",
               "ONDO", "MON", "XPL"},
    "stock": {"TSLA", "PLTR", "SOFI", "AMD", "MSTR"},
    "commodity": {"GOLD", "SILVER", "COPPER", "BRENT"},
}
PAIR_BENCH = {"crypto": ["BTC"], "stock": ["SPX", "NDX"], "commodity": ["GOLD"]}

# per-timeframe: pivot window k, candles shown, S/R lookback, log scale
TF_CFG = {
    "4H": dict(k=6, show=360, look=0,   log=False),
    "D":  dict(k=5, show=365, look=600, log=False),
    "W":  dict(k=4, show=208, look=260, log=True),
    "M":  dict(k=3, show=999, look=120, log=True),
}

# colors (TradingView-familiar; dataviz-checked: distinct hues, text stays ink)
C_UP, C_DN = "#26a69a", "#ef5350"
C_EMA = {21: "#2962ff", 50: "#f57c00", 200: "#7b1fa2"}
C_SUP, C_RES = "#00897b", "#d32f2f"
C_GRID, C_INK, C_MUT = "#e6e8ea", "#1d2126", "#6b7280"


# ------------------------------------------------------------------ plan ----

def cmd_plan(sym: str) -> None:
    kr, cb = SYMBOLS.get(sym, (f"{sym}USD", f"{sym}-USD"))
    uniq = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    print(f"# Fetch plan for {sym} — call via WebFetch, ALWAYS unique &nocache=")
    print(f"# Responses >~30KB are persisted to a file by the tool -> copy into raw/")
    print(f"https://api.kraken.com/0/public/OHLC?pair={kr}&interval=240&nocache={uniq}a   # 4H (720)")
    print(f"https://api.kraken.com/0/public/OHLC?pair={kr}&interval=1440&nocache={uniq}b  # D (720 = 2y)")
    print(f"https://api.kraken.com/0/public/OHLC?pair={kr}&interval=10080&nocache={uniq}c # W (full)")
    if cb:
        print(f"# Full daily history — Coinbase chunks of ~290 days, newest format "
              f"[time,low,high,open,close,vol]:")
        print(f"https://api.exchange.coinbase.com/products/{cb}/candles"
              f"?granularity=86400&start=<ISO>&end=<ISO>&nocache={uniq}<i>")
    else:
        print(f"# {sym}: not on Coinbase — Kraken 720-day window is the ceiling.")


# ------------------------------------------------------------- normalize ----

def _read_json(path: str):
    txt = open(path).read()
    s, e = txt.find("{"), txt.rfind("}")
    if s == -1:  # coinbase = bare array
        s, e = txt.find("["), txt.rfind("]")
    return json.loads(txt[s:e + 1])


def cmd_normalize(raw_dir: str, out_dir: str) -> None:
    """raw/<SYM>_<tf>_<source>[_chunkNN].json -> csv/<SYM>_<tf>[_kraken].csv"""
    os.makedirs(out_dir, exist_ok=True)
    groups = {}
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.json"))):
        base = os.path.basename(p)[:-5]
        parts = base.split("_")           # SYM, tf, source, [chunkNN]
        sym, tf, source = parts[0], parts[1], parts[2]
        groups.setdefault((sym, tf, source), []).append(p)

    for (sym, tf, source), paths in groups.items():
        rows = []
        for p in paths:
            d = _read_json(p)
            if source == "coinbase":
                for t, lo, hi, op, cl, vol in d:
                    rows.append((t, op, hi, lo, cl, vol))
            else:  # kraken
                pair = next(iter(d["result"]))
                if pair == "last":
                    raise ValueError(f"{p}: empty kraken result")
                for t, op, hi, lo, cl, _vwap, vol, _n in d["result"][pair]:
                    rows.append((int(t), float(op), float(hi), float(lo),
                                 float(cl), float(vol)))
        df = (pd.DataFrame(rows, columns=["t", "open", "high", "low", "close",
                                          "volume"])
              .astype(float).drop_duplicates("t").sort_values("t"))
        ts = pd.to_datetime(df["t"], unit="s", utc=True)
        df["date"] = ts.dt.strftime("%Y-%m-%dT%H:%M" if tf == "4h"
                                    else "%Y-%m-%d")
        suffix = "_kraken" if (source == "kraken" and tf == "1w") else ""
        out = os.path.join(out_dir, f"{sym}_{tf}{suffix}.csv")
        df[["date", "open", "high", "low", "close", "volume"]].to_csv(
            out, index=False)
        print(f"wrote {out}: {len(df)} rows {df['date'].iloc[0]} → "
              f"{df['date'].iloc[-1]}")


# --------------------------------------------------------------- analyze ----

def cut_asof(df: pd.DataFrame, as_of: str):
    """as-of mode: κράτα μόνο candles με date <= as_of (πριν από κάθε resample,
    ώστε W/M να δείχνουν ό,τι θα έδειχναν εκείνη τη μέρα)."""
    if not as_of:
        return df
    cut = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    if getattr(df.index, "tz", None) is not None:
        cut = cut.tz_localize(df.index.tz)
    return df[df.index < cut]


def _finite(x):
    """JSON safety: μη-πεπερασμένο/μη-αριθμός → None (ποτέ bare NaN)."""
    try:
        return float(x) if math.isfinite(float(x)) else None
    except (TypeError, ValueError):
        return None


def load_frames(csv_dir: str, sym: str, as_of: str = None) -> dict:
    """Return {tf: df} with DatetimeIndex, cols open/high/low/close/volume."""
    def rd(path):
        df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        return df.sort_index()

    frames = {}
    p1d = os.path.join(csv_dir, f"{sym}_1d.csv")
    if not os.path.exists(p1d):
        sys.exit(f"missing {p1d} — run the fetch flow first (see `plan`)")
    daily = cut_asof(rd(p1d), as_of)
    if daily.empty:
        sys.exit(f"{sym}: no candles at or before --as-of {as_of}")
    frames["D"] = daily

    p4h = os.path.join(csv_dir, f"{sym}_4h.csv")
    if os.path.exists(p4h):
        f4 = cut_asof(rd(p4h), as_of)
        if not f4.empty:
            frames["4H"] = f4

    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    pw = os.path.join(csv_dir, f"{sym}_1w_kraken.csv")
    if os.path.exists(pw):
        wk = cut_asof(rd(pw), as_of)
        if as_of:
            # το weekly bar καλύπτει ΟΛΗ την εβδομάδα — bar με label πριν το
            # as_of μπορεί να περιέχει μελλοντικές μέρες. Κράτα μόνο εβδομάδες
            # που έχουν κλείσει ως το as_of (ίδια σημασιολογία με το live:
            # το mirror γράφει μόνο κλεισμένα candles).
            end = pd.Timestamp(as_of) + pd.Timedelta(days=1)
            if getattr(wk.index, "tz", None) is not None:
                end = end.tz_localize(wk.index.tz)
            wk = wk[wk.index + pd.Timedelta(days=7) <= end]
        # Kraken weekly = full history; prefer it if longer than daily-derived
        wk_d = daily.resample("W-MON", label="left", closed="left").agg(agg).dropna()
        frames["W"] = wk if len(wk) > len(wk_d) else wk_d
    else:
        frames["W"] = daily.resample("W-MON", label="left",
                                     closed="left").agg(agg).dropna()
    frames["M"] = daily.resample("MS").agg(agg).dropna()
    return frames


def add_emas(df: pd.DataFrame) -> pd.DataFrame:
    for n in EMAS:
        if len(df) >= n:
            df[f"ema{n}"] = df["close"].ewm(span=n, adjust=False).mean()
    return df


def atr(df: pd.DataFrame, n: int = 14) -> float:
    hl = df["high"] - df["low"]
    hc = (df["high"] - df["close"].shift()).abs()
    lc = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return float(tr.rolling(n).mean().iloc[-1])


def sr_levels(df: pd.DataFrame, k: int, lookback: int, max_each: int = 4):
    """Swing-pivot detection + ATR-radius clustering. Simple & explainable:
    a pivot high/low is a local extreme over ±k candles; pivots within
    0.75*ATR(14) merge into one level; each pivot is weighted by recency
    (exp decay — a level from years ago must be VERY heavy to outrank a
    recent one); score = sum of weights."""
    win = df.iloc[-lookback:] if lookback else df
    if len(win) < 2 * k + 5:
        return [], []
    highs, lows = win["high"].values, win["low"].values
    idx = win.index
    n = len(win)
    piv = []  # (price, position 0..1)
    for i in range(k, n - k):
        if highs[i] == highs[i - k:i + k + 1].max():
            piv.append((highs[i], i / n))
        if lows[i] == lows[i - k:i + k + 1].min():
            piv.append((lows[i], i / n))
    if not piv:
        return [], []
    radius = 0.75 * atr(win)
    piv.sort()
    clusters = []  # [prices], [recencies]
    for price, rec in piv:
        if clusters and price - np.mean(clusters[-1][0]) <= radius:
            clusters[-1][0].append(price)
            clusters[-1][1].append(rec)
        else:
            clusters.append(([price], [rec]))
    levels = []
    for prices, recs in clusters:
        weights = [np.exp(-4.0 * (1.0 - r)) for r in recs]  # now=1, old→~0.02
        score = float(np.sum(weights))
        levels.append((float(np.mean(prices)), len(prices), round(score, 2)))
    last = float(df["close"].iloc[-1])
    sup = sorted([l for l in levels if l[0] < last],
                 key=lambda l: -l[2])[:max_each]
    res = sorted([l for l in levels if l[0] >= last],
                 key=lambda l: -l[2])[:max_each]
    return (sorted(sup, key=lambda l: -l[0]),      # nearest support first
            sorted(res, key=lambda l: l[0]))       # nearest resistance first


def verdict(df: pd.DataFrame) -> tuple:
    """Explainable score: price vs EMAs + EMA stack + EMA21 slope."""
    last = df["close"].iloc[-1]
    checks, score, mx = [], 0, 0
    for n in EMAS:
        col = f"ema{n}"
        if col in df:
            ok = last > df[col].iloc[-1]
            score += ok
            mx += 1
            checks.append(f"{'✅' if ok else '❌'} τιμή {'>' if ok else '<'} EMA{n}")
    for a, b in ((21, 50), (50, 200)):
        ca, cb = f"ema{a}", f"ema{b}"
        if ca in df and cb in df:
            ok = df[ca].iloc[-1] > df[cb].iloc[-1]
            score += ok
            mx += 1
            checks.append(f"{'✅' if ok else '❌'} EMA{a} {'>' if ok else '<'} EMA{b}")
    if "ema21" in df and len(df) >= 26:
        ok = df["ema21"].iloc[-1] > df["ema21"].iloc[-6]
        score += ok
        mx += 1
        checks.append(f"{'✅' if ok else '❌'} EMA21 slope {'↑' if ok else '↓'} (5 candles)")
    if mx == 0:
        return "N/A", checks
    r = score / mx
    v = ("BULLISH" if r >= 0.8 else "lean bullish" if r >= 0.6 else
         "NEUTRAL" if r >= 0.4 else "lean bearish" if r >= 0.2 else "BEARISH")
    thin = " ⚠️λίγο ιστορικό" if len(df) < 60 else ""
    return f"{v} ({score}/{mx}){thin}", checks


def pair_verdicts(csv_dir: str, sym: str, daily: pd.DataFrame, as_of: str):
    """Relative-strength pairs (v2 §Γ): ratio των daily σειρών από το mirror,
    ίδια EMA/trend ανάγνωση ανά TF (D/W/M) → {"SOL/BTC": {"D": verdict, ...}}.
    Δεύτερη τιμή: benchmarks που λείπουν/είναι πολύ λεπτά από το csv_dir."""
    cls = next((c for c, syms in ASSET_CLASS.items() if sym in syms), None)
    pairs, missing = {}, []
    for bench in PAIR_BENCH.get(cls, []):
        if bench == sym:
            continue
        p = os.path.join(csv_dir, f"{bench}_1d.csv")
        if not os.path.exists(p):
            missing.append(bench)
            continue
        bdf = pd.read_csv(p, parse_dates=["date"]).set_index("date").sort_index()
        ratio = (daily["close"] / cut_asof(bdf, as_of)["close"]).dropna()
        if len(ratio) < 30:   # δεν βγαίνει ούτε EMA21 — άχρηστο verdict
            missing.append(bench)
            continue
        tf_series = {
            "D": ratio,
            "W": ratio.resample("W-MON", label="left", closed="left")
                      .last().dropna(),
            "M": ratio.resample("MS").last().dropna(),
        }
        pairs[f"{sym}/{bench}"] = {
            tf: verdict(add_emas(s.to_frame("close")))[0]
            for tf, s in tf_series.items()}
    return pairs, missing


def load_manual_levels(path: str, sym: str):
    """Χειροκίνητα levels του Π. από το config/levels.csv (levels route).
    Σχήμα: symbol,price,kind,source,date,note — kind: support|resistance.
    Επιστρέφει (levels, skipped): λίστα dicts για το σύμβολο + πλήθος
    γραμμών που δεν πέρασαν τα deterministic checks (ποτέ guess)."""
    levels, skipped, seen = [], 0, set()
    if not path or not os.path.exists(path):
        return levels, skipped
    import csv as _csv
    import re as _re

    def _txt(v):
        # review 6/8: τα free-text πεδία μπαίνουν σε report/markdown —
        # όχι newlines/markdown δομή από το CSV, cap μήκους
        return _re.sub(r"\s+", " ", (v or "").strip())[:120]

    with open(path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            if (row.get("symbol") or "").strip().upper() != sym:
                continue
            kind = (row.get("kind") or "").strip().lower()
            raw = (row.get("price") or "").strip()
            # αυστηρό format (όχι float('nan'/'inf'/underscores) — review 6/8)
            price = float(raw) if _re.fullmatch(r"\d+(\.\d+)?", raw) else None
            if (price is None or not math.isfinite(price) or price <= 0
                    or kind not in ("support", "resistance")):
                skipped += 1
                continue
            if (price, kind) in seen:   # διπλή γραμμή = ένα level, όχι διπλά 📐
                continue
            seen.add((price, kind))
            levels.append({"price": price, "kind": kind,
                           "source": _txt(row.get("source")),
                           "date": _txt(row.get("date")),
                           "note": _txt(row.get("note"))})
    levels.sort(key=lambda m: (-m["price"], m["kind"]))
    return levels, skipped


def manual_confluence(level_price: float, jtf: dict, k: float = 0.5):
    """Confluence χειροκίνητου level με αλγοριθμικά: |Δ| ≤ k×ATR14 του TF.
    Επιστρέφει λίστα 'TF side price' strings (deterministic σειρά)."""
    hits = []
    for tf in ("M", "W", "D", "4H"):
        d = jtf.get(tf)
        if not d or not d.get("atr14"):
            continue
        tol = k * d["atr14"]
        for side, key in (("res", "resistance"), ("sup", "support")):
            for lv in d.get(key, []):
                p = lv.get("price")
                if p is not None and abs(p - level_price) <= tol:
                    hits.append(f"{tf} {side} {fmt(p)}")
    return hits


def fmt(p: float) -> str:
    return (f"{p:,.0f}" if p >= 1000 else f"{p:,.2f}" if p >= 1
            else f"{p:.4f}")


def plot_tf(sym, tf, df, sup, res, verd, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = TF_CFG[tf]
    d = df.iloc[-cfg["show"]:]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=110)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # candles (thin marks, no borders)
    w = 0.65
    up = d["close"].values >= d["open"].values
    ax.vlines(x, d["low"], d["high"],
              color=np.where(up, C_UP, C_DN), lw=0.7, zorder=2)
    ax.bar(x, (d["close"] - d["open"]).abs(),
           bottom=np.minimum(d["open"], d["close"]), width=w,
           color=np.where(up, C_UP, C_DN), zorder=3)

    for n in EMAS:
        col = f"ema{n}"
        if col in d and d[col].notna().any():
            ax.plot(x, d[col], color=C_EMA[n], lw=1.4, label=f"EMA {n}",
                    zorder=4)

    lo_vis, hi_vis = d["low"].min(), d["high"].max()
    for price, touches, _s in sup + res:
        if not (lo_vis * 0.9 <= price <= hi_vis * 1.1):
            continue
        c = C_SUP if price < d["close"].iloc[-1] else C_RES
        ax.axhline(price, color=c, lw=1.0, ls=(0, (5, 4)), alpha=0.85, zorder=1)
        ax.annotate(f" {fmt(price)} ({touches}x)", xy=(1.001, price),
                    xycoords=("axes fraction", "data"), fontsize=8,
                    color=c, va="center")

    step = max(len(d) // 8, 1)
    ax.set_xticks(x[::step])
    fmt_d = "%b %d %H:%M" if tf == "4H" else ("%Y-%m" if tf in ("W", "M")
                                              else "%b %d %y")
    ax.set_xticklabels([t.strftime(fmt_d) for t in d.index[::step]],
                       fontsize=8, color=C_MUT)
    if cfg["log"]:
        ax.set_yscale("log")
        from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
        lo, hi = np.log10(lo_vis * 0.95), np.log10(hi_vis * 1.05)
        ticks = np.geomspace(10 ** lo, 10 ** hi, 9)
        nice = [float(f"{t:.2g}") for t in ticks]
        ax.yaxis.set_major_locator(FixedLocator(sorted(set(nice))))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt(v)))
    ax.tick_params(colors=C_MUT, labelsize=8)
    ax.grid(axis="y", color=C_GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(C_GRID)

    last = d["close"].iloc[-1]
    ax.set_title(f"{sym} · {tf} · {fmt(last)} $ · {verd}",
                 loc="left", fontsize=11, color=C_INK, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=C_INK)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def cmd_analyze(sym: str, csv_dir: str, out_dir: str, as_of: str = None,
                json_path: str = None, no_charts: bool = False,
                all_levels: bool = False, levels_path: str = None) -> None:
    os.makedirs(out_dir, exist_ok=True)
    frames = load_frames(csv_dir, sym, as_of)
    daily = frames["D"]
    last = float(daily["close"].iloc[-1])
    ath = float(daily["high"].max())
    ath_d = daily["high"].idxmax().date()
    lo52 = float(daily["low"].iloc[-365:].min())
    hi52 = float(daily["high"].iloc[-365:].max())

    stamp = (f"as-of {as_of}" if as_of
             else f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    lines = [f"# {sym} — chart analysis · {stamp}",
             "",
             f"**Τιμή:** {fmt(last)} $ · **ATH:** {fmt(ath)} $ ({ath_d}) · "
             f"**52w:** {fmt(lo52)}–{fmt(hi52)} $",
             f"_Δεδομένα: daily {daily.index[0].date()} → {daily.index[-1].date()}"
             f" ({len(daily)} candles· το τελευταίο = μερικό/σημερινό)_",
             ""]
    verdicts, jtf = {}, {}
    for tf in [t for t in ("M", "W", "D", "4H") if t in frames]:
        df = add_emas(frames[tf].copy())
        cfg = TF_CFG[tf]
        sup, res = sr_levels(df, cfg["k"], lookback=cfg["look"])
        verd, checks = verdict(df)
        verdicts[tf] = verd
        # less-is-more default (απόφαση Π.): top 3 ανά πλευρά στο report/chart —
        # το sr_levels μένει ΑΘΙΚΤΟ, αυτό είναι καθαρά παρουσίαση. --all-levels
        # δείχνει όλα όσα υπολογίστηκαν· το JSON τα έχει πάντα όλα.
        d_sup = sup if all_levels else sup[:3]
        d_res = res if all_levels else res[:3]
        a14 = atr(df) if len(df) >= 15 else float("nan")
        jtf[tf] = {"verdict": verd, "checks": checks,
                   "atr14": _finite(a14),
                   "support": [{"price": _finite(p), "touches": t,
                                "score": _finite(s)} for p, t, s in sup],
                   "resistance": [{"price": _finite(p), "touches": t,
                                   "score": _finite(s)} for p, t, s in res],
                   "last_close": _finite(df["close"].iloc[-1]),
                   "candles": len(df)}
        if not no_charts:
            png = os.path.join(out_dir, f"{sym}_{tf}.png")
            plot_tf(sym, tf, df, d_sup, d_res, verd, png)

        lines += [f"## {tf} — {verd}", ""]
        lines += [f"- {c}" for c in checks]
        missing = [n for n in EMAS if f"ema{n}" not in df]
        if missing:
            lines += [f"- ⚪ EMA {'/'.join(map(str, missing))}: μη διαθέσιμο "
                      f"(ιστορικό {len(df)} candles)"]
        more_r = " _(top 3 — όλα με --all-levels)_" \
            if not all_levels and len(res) > 3 else ""
        more_s = " _(top 3 — όλα με --all-levels)_" \
            if not all_levels and len(sup) > 3 else ""
        lines += ["", "**Resistance (πλησιέστερο πρώτα):** " +
                  (" · ".join(f"{fmt(p)} ({t}x)" for p, t, _ in d_res) or "—") +
                  more_r]
        lines += ["**Support (πλησιέστερο πρώτα):** " +
                  (" · ".join(f"{fmt(p)} ({t}x)" for p, t, _ in d_sup) or "—") +
                  more_s, ""]

    pairs, pair_missing = pair_verdicts(csv_dir, sym, daily, as_of)
    if pairs or pair_missing:
        lines += ["## Pairs — relative strength (ratio mirror daily σειρών)", ""]
        for name, tfv in pairs.items():
            lines += [f"**{name}:** " +
                      " · ".join(f"{tf}: {v}" for tf, v in tfv.items())]
        for b in pair_missing:
            lines += [f"_⚪ pair vs {b}: δεν υπάρχει {b}_1d.csv στο csv-dir "
                      "(ή πολύ λίγα κοινά candles) — προστίθεται με το mirror._"]
        lines += [""]

    manual, manual_skipped = load_manual_levels(levels_path, sym)
    jml = []
    if manual or manual_skipped:
        lines += ["## Levels Π. (χειροκίνητα — config/levels.csv)", ""]
        for m in manual:
            hits = manual_confluence(m["price"], jtf)
            src = f" · πηγή: {m['source']}" if m["source"] else ""
            dt = f" ({m['date']})" if m["date"] else ""
            note = f" — {m['note']}" if m["note"] else ""
            conf = (" · ✔ confluence: " + ", ".join(hits)) if hits else ""
            lines += [f"- **{fmt(m['price'])}** {m['kind']}{src}{dt}{note}{conf}"]
            jml.append({**m, "price": _finite(m["price"]), "confluence": hits})
        if manual_skipped:
            lines += [f"_⚪ {manual_skipped} γραμμές του levels.csv δεν πέρασαν "
                      "τα checks (symbol/price/kind) — αγνοήθηκαν, δες αρχείο._"]
        lines += [""]

    order = [t for t in ("M", "W", "D", "4H") if t in verdicts]
    lines += ["## Σύνοψη", "",
              " · ".join(f"**{t}:** {verdicts[t]}" for t in order),
              "",
              "_Analysis tool, όχι advisor — κανένα trade signal. "
              "Τα levels είναι αλγοριθμικά (swing pivots + clustering)· "
              "σετάρονται χειροκίνητα στο TradingView._"]
    lines += ["",
              "<!-- 💡 Παρατήρηση: προαιρετικό section που προσθέτει το session "
              "ΜΕΤΑ το σταθερό core (κανόνας: services/chart-scan/README.md). "
              "Το script δεν γράφει εδώ. -->"]
    rep = os.path.join(out_dir, f"{sym}_report.md")
    open(rep, "w").write("\n".join(lines))
    print("\n".join(lines))
    if json_path:
        payload = {
            "symbol": sym,
            "as_of": as_of,
            "data_end": str(daily.index[-1].date()),
            "last_close": _finite(last),
            "ath": _finite(ath),
            "ath_date": str(ath_d),
            "range_52w": [_finite(lo52), _finite(hi52)],
            "timeframes": jtf,
            "pairs": pairs,
            "pairs_missing_bench": pair_missing,
            "manual_levels": jml,
            "manual_levels_skipped": manual_skipped,
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1,
                      allow_nan=False)
        print(f"[json] {json_path}", file=sys.stderr)
    files = rep if no_charts else f"{out_dir}/{sym}_*.png + {rep}"
    print(f"\n[files] {files}", file=sys.stderr)


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("symbol")
    p = sub.add_parser("normalize")
    p.add_argument("raw_dir")
    p.add_argument("csv_dir")
    p = sub.add_parser("analyze")
    p.add_argument("symbol")
    p.add_argument("--csv-dir", default="csv")
    p.add_argument("--out", default="out")
    p.add_argument("--as-of", metavar="YYYY-MM-DD",
                   help="cutoff: ανάλυση μόνο με candles μέχρι και αυτή τη μέρα")
    p.add_argument("--json", metavar="PATH",
                   help="γράψε structured output (verdicts/levels/pairs) εδώ")
    p.add_argument("--no-charts", action="store_true",
                   help="χωρίς PNGs — δεν απαιτεί matplotlib (scan mode)")
    p.add_argument("--all-levels", action="store_true",
                   help="όλα τα S/R levels αντί για το top-3 default")
    p.add_argument("--levels", metavar="CSV", default="config/levels.csv",
                   help="χειροκίνητα levels (levels route)· αν λείπει το "
                        "αρχείο, το section απλώς παραλείπεται")
    a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan(a.symbol.upper())
    elif a.cmd == "normalize":
        cmd_normalize(a.raw_dir, a.csv_dir)
    else:
        if a.as_of:
            try:
                datetime.strptime(a.as_of, "%Y-%m-%d")
            except ValueError:
                sys.exit(f"--as-of '{a.as_of}': περιμένω YYYY-MM-DD")
        cmd_analyze(a.symbol.upper(), a.csv_dir, a.out, as_of=a.as_of,
                    json_path=a.json, no_charts=a.no_charts,
                    all_levels=a.all_levels, levels_path=a.levels)


if __name__ == "__main__":
    main()
