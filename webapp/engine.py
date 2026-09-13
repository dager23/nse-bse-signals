"""Signal engine for the web UI.

Computes both portfolio views from live market data:

  MONTHLY  — rank-based baskets that rebalance on the last trading day of the
             month (champion R-S, most-bias-resistant P2, JT momentum with
             crash overlay Q9, volume-weighted momentum O17).
  DAILY    — event/oscillator sleeves that can open or close any day
             (R-W volume-spike + inside-day, R-T deep-oversold RSI).

All metrics quoted to the UI are the SURVIVORSHIP-CORRECTED (point-in-time)
numbers from reports/survivorship_corrected.md — not the inflated backfilled
ones.  Results are cached; refresh runs in a background thread.
"""
import os
import sys
import threading
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: E402

CACHE_TTL = 3 * 3600  # serve cached signals for 3h before auto-refetch

# sid, spec, label, blurb, PIT 15Y sharpe, PIT 15Y CAGR, PIT 15Y maxDD
MONTHLY = [
    ("R-S", "S|lowbeta*0.50+mom121*0.30+rev1m*0.20|top10", "Champion blend",
     "50% low-beta + 30% 12-month momentum + 20% one-month reversal. "
     "Best overall across four independent test domains.", 0.71, 0.19, -0.29),
    ("P2", "S|lowvol*0.50+rev1m*0.50|top5", "Low-vol + reversal",
     "Half low-volatility, half one-month reversal, concentrated in 5 names. "
     "The least survivorship-biased strategy found — best on honest data.",
     0.79, 0.21, -0.32),
    ("Q9", "jt121|crash8", "Momentum + crash guard",
     "Classic 12-1 momentum, but goes to cash when the Nifty drops more than "
     "8% in 10 days.", 0.44, 0.14, -0.30),
    ("O17", "lb63", "Volume-weighted momentum",
     "Momentum weighted by traded volume over 63 days.", -0.18, 0.05, -0.45),
]
DAILY = [
    ("R-W", "W|vspike&insideday|vix_low|rsi70", "Volume spike + inside day",
     "Enters on a 2x volume spike with an inside-day candle while VIX is calm; "
     "exits when RSI(14) clears 70.", 0.52, 0.17, -0.49),
    ("R-T", "T|rsi5|<15|>80|none|state", "Deep oversold RSI",
     "Buys when RSI(5) drops under 15, holds until RSI(5) tops 80.",
     0.46, 0.15, -0.40),
]

_lock = threading.Lock()
_state = {
    "status": "idle",        # idle | running | ok | error
    "message": "",
    "data": None,
    "fetched_at": None,
    "started_at": None,
}


def _fetch_ctx():
    import yfinance as yf
    from datetime import timedelta
    start = (datetime.now() - timedelta(days=4 * 365)).strftime("%Y-%m-%d")
    tickers = sorted(set(config.STOCK_UNIVERSE))
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                      group_by="column")
    o, h, l, c, v = (raw["Open"], raw["High"], raw["Low"], raw["Close"],
                     raw["Volume"])
    for df in (o, h, l, c, v):
        df.index = pd.to_datetime(df.index).tz_localize(None)
    bench = yf.download("^NSEI", start=start, auto_adjust=True, progress=False)["Close"]
    if isinstance(bench, pd.DataFrame):
        bench = bench.iloc[:, 0]
    bench.index = pd.to_datetime(bench.index).tz_localize(None)
    bench = bench.reindex(c.index).ffill()
    try:
        vix = yf.download("^INDIAVIX", start=start, auto_adjust=False,
                          progress=False)["Close"]
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
    except Exception:  # noqa: BLE001
        vix = None

    # Yahoo often appends a partial/empty bar for the in-progress session.
    # Signals must be computed on the last COMPLETE bar, so drop any trailing
    # rows where no stock has a close.
    good = c.notna().sum(axis=1) > 0
    if good.any():
        last_good = c.index[good][-1]
        o, h, l, c, v = (df.loc[:last_good] for df in (o, h, l, c, v))
        bench = bench.loc[:last_good]

    # tickers with no data in the last 5 sessions (delisted / renamed)
    stale = sorted(t.replace(".NS", "") for t in c.columns
                   if c[t].tail(5).notna().sum() == 0)
    return {"open": o, "high": h, "low": l, "close": c, "volume": v,
            "benchmark": bench, "vix": vix, "interval": "1d",
            "universe": "live", "stale_tickers": stale}


def _positions(ctx, sid, spec):
    if sid in ("R-S", "P2", "R-W", "R-T"):
        from strategies.cat_R_promoted import build_positions
        return build_positions(ctx, spec)
    if sid == "Q9":
        from strategies.cat_Q_overlay import bases, apply_overlay
        kind, pos = bases(ctx)["jt121"]
        newpos, _ = apply_overlay(ctx, pos, kind, "crash8")
        return newpos
    if sid == "O17":
        from utils.composite import monthly_topn
        c, v = ctx["close"], ctx["volume"]
        ret = c.pct_change(fill_method=None)
        vwm = (ret * v).rolling(63).sum() / v.rolling(63).sum()
        return monthly_topn(c, vwm, 10)
    raise ValueError(sid)


def _clean(t):
    return t.replace(".NS", "")


def _last_bday_of_month(ts):
    end = ts + pd.offsets.MonthEnd(0)
    while end.weekday() >= 5:
        end -= pd.Timedelta(days=1)
    return end


def _rebalance_info(asof):
    """Rebalance = last trading day of a month.

    The data always ends mid-month for the *current* month, so a month-end
    cannot be inferred from the calendar of observed bars (the newest bar
    would always look like one).  Estimate from the business calendar
    instead; NSE holidays can shift the true date by a day.
    """
    this_month_end = _last_bday_of_month(asof)
    if asof >= this_month_end:
        nxt = _last_bday_of_month(asof + pd.offsets.MonthBegin(1))
    else:
        nxt = this_month_end
    is_today = asof.normalize() == this_month_end.normalize()
    days = max(int(np.busday_count(asof.date(), nxt.date())), 0)
    return nxt, is_today, days


def _completed_month_ends(index, asof):
    """Last trading day of each FULLY ELAPSED month in the data.

    The current month's newest bar is not a rebalance point — treating it as
    one would re-rank the book every session instead of monthly.
    """
    s = pd.Series(index, index=index)
    ends = sorted(set(s.groupby([index.year, index.month]).transform("max")))
    return [d for d in ends if (d.year, d.month) != (asof.year, asof.month)]


def _sleeve_payload(ctx, entry, kind):
    sid, spec, label, blurb, sh, cagr, dd = entry
    c = ctx["close"]
    asof = c.index[-1]
    pos = _positions(ctx, sid, spec)
    pos = pos.reindex(index=c.index, columns=c.columns).fillna(0)

    if kind == "monthly":
        # holdings are those set at the last completed month-end, and the
        # comparison point is the rebalance before it
        ends = _completed_month_ends(c.index, asof)
        if ends:
            cur = pos.loc[ends[-1]]
            prev = pos.loc[ends[-2]] if len(ends) > 1 else cur * 0
        else:
            cur = pos.iloc[-1]
            prev = cur * 0
    else:
        cur = pos.iloc[-1]
        prev = pos.iloc[-2] if len(pos) > 1 else cur * 0
    held = sorted(cur[cur > 0].index)
    prev_held = set(prev[prev > 0].index)
    ret1 = (c.iloc[-1] / c.iloc[-2] - 1) if len(c) > 1 else c.iloc[-1] * 0
    holdings = [{
        "ticker": _clean(t),
        "price": None if pd.isna(c.iloc[-1][t]) else round(float(c.iloc[-1][t]), 2),
        "day_pct": None if pd.isna(ret1[t]) else round(float(ret1[t]) * 100, 2),
        "is_new": t not in prev_held,
    } for t in held]
    exits = sorted(_clean(t) for t in prev_held - set(held))
    if kind == "monthly":
        ends = _completed_month_ends(c.index, asof)
        basis = f"{ends[-1]:%d %b %Y}" if ends else f"{asof:%d %b %Y}"
    else:
        basis = f"{asof:%d %b %Y}"
    return {
        "sid": sid, "spec": spec, "label": label, "blurb": blurb, "kind": kind,
        "pit_sharpe": sh, "pit_cagr": cagr, "pit_dd": dd, "basis": basis,
        "holdings": holdings, "n": len(holdings), "exits": exits,
        "new_count": sum(1 for x in holdings if x["is_new"]),
    }


def compute():
    ctx = _fetch_ctx()
    c = ctx["close"]
    asof = c.index[-1]
    monthly = [_sleeve_payload(ctx, e, "monthly") for e in MONTHLY]
    daily = [_sleeve_payload(ctx, e, "daily") for e in DAILY]

    nxt, is_rebal_day, days_to_rebal = _rebalance_info(asof)

    # combined meta-portfolio: equal capital per sleeve, equal weight inside
    weights = {}
    live = monthly + daily
    n_sleeves = sum(1 for s in live if s["n"] > 0)
    for s in live:
        if not s["n"]:
            continue
        w_each = (1.0 / n_sleeves) / s["n"]
        for hold in s["holdings"]:
            weights[hold["ticker"]] = weights.get(hold["ticker"], 0.0) + w_each
    combined = sorted(({"ticker": k, "weight_pct": round(v * 100, 2),
                        "sleeves": sum(1 for s in live
                                       for h in s["holdings"] if h["ticker"] == k)}
                       for k, v in weights.items()),
                      key=lambda x: -x["weight_pct"])

    bench_1d = float(ctx["benchmark"].iloc[-1] / ctx["benchmark"].iloc[-2] - 1) * 100
    return {
        "asof": f"{asof:%Y-%m-%d}",
        "asof_pretty": f"{asof:%d %b %Y}",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_tickers": int(c.iloc[-1].notna().sum()),
        "nifty": round(float(ctx["benchmark"].iloc[-1]), 2),
        "nifty_day_pct": round(bench_1d, 2),
        "vix": None if ctx["vix"] is None or ctx["vix"].dropna().empty
        else round(float(ctx["vix"].dropna().iloc[-1]), 2),
        "monthly": monthly,
        "daily": daily,
        "combined": combined,
        "next_rebalance": f"{nxt:%d %b %Y}",
        "days_to_rebalance": days_to_rebal,
        "is_rebalance_day": bool(is_rebal_day),
        "stale_tickers": ctx.get("stale_tickers", []),
    }


# ------------------------------------------------------------------ threading
def _worker():
    try:
        data = compute()
        with _lock:
            _state.update({"status": "ok", "data": data, "message": "",
                           "fetched_at": time.time()})
    except Exception as e:  # noqa: BLE001
        with _lock:
            _state.update({"status": "error", "message": f"{type(e).__name__}: {e}"})


def start_refresh(force=False):
    with _lock:
        if _state["status"] == "running":
            return False
        fresh = (_state["data"] is not None and _state["fetched_at"]
                 and time.time() - _state["fetched_at"] < CACHE_TTL)
        if fresh and not force:
            return False
        _state.update({"status": "running", "message": "", "started_at": time.time()})
    threading.Thread(target=_worker, daemon=True).start()
    return True


def snapshot():
    with _lock:
        s = dict(_state)
    if s["status"] == "running" and s["started_at"]:
        s["elapsed"] = int(time.time() - s["started_at"])
    if s["fetched_at"]:
        s["age_min"] = int((time.time() - s["fetched_at"]) / 60)
    s.pop("started_at", None)
    return s
