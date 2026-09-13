"""Forward-test signal generator.

Each run (daily, after NSE close):
  1. Downloads the last ~4 years of OHLCV for the research universe + Nifty +
     India VIX (enough warm-up for every indicator used).
  2. Recomputes the champion and the 5 meta-portfolio sleeves EXACTLY as
     backtested (same spec interpreters) and extracts today's target holdings.
  3. Marks-to-market the holdings recorded on the previous run and appends
     realized returns to forward_test/performance.csv - a genuinely
     out-of-sample track record accumulates.

Outputs (forward_test/):
  holdings_YYYY-MM-DD.csv   today's target portfolio per sleeve
  log.csv                   run log (append)
  performance.csv           realized daily returns: meta, champion, Nifty
"""
import os
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config

FWD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "forward_test")
os.makedirs(FWD, exist_ok=True)

CHAMPION = ("R-S", "S|lowbeta*0.50+mom121*0.30+rev1m*0.20|top10")
# (sid, spec, rebalance_kind): monthly sleeves only change at month-end
SLEEVES = [
    CHAMPION + ("monthly",),
    ("R-W", "W|vspike&insideday|vix_low|rsi70", "daily"),
    ("Q9", "jt121|crash8", "monthly"),
    ("O17", "lb63", "monthly"),
    ("R-T", "T|rsi5|<15|>80|none|state", "daily"),
]


def completed_month_ends(index, asof):
    """Last trading day of each fully elapsed month.

    The newest bar of the CURRENT month is not a rebalance point; treating it
    as one would re-rank monthly sleeves every session (far more turnover than
    was ever backtested).
    """
    s = pd.Series(index, index=index)
    ends = sorted(set(s.groupby([index.year, index.month]).transform("max")))
    return [d for d in ends if (d.year, d.month) != (asof.year, asof.month)]


def fetch_live_ctx():
    import yfinance as yf
    start = (datetime.now() - timedelta(days=4 * 365)).strftime("%Y-%m-%d")
    tickers = sorted(set(config.STOCK_UNIVERSE))
    raw = yf.download(tickers, start=start, auto_adjust=True,
                      progress=False, group_by="column")
    o, h, l, c, v = (raw["Open"], raw["High"], raw["Low"], raw["Close"],
                     raw["Volume"])
    for df in (o, h, l, c, v):
        df.index = pd.to_datetime(df.index).tz_localize(None)
    bench = yf.download("^NSEI", start=start, auto_adjust=True, progress=False)
    bench = bench["Close"]
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
    # Yahoo appends a partial/empty bar for the in-progress session; signals
    # must be computed on the last COMPLETE bar.
    good = c.notna().sum(axis=1) > 0
    if good.any():
        last_good = c.index[good][-1]
        o, h, l, c, v = (df.loc[:last_good] for df in (o, h, l, c, v))
        bench = bench.loc[:last_good]
    return {"open": o, "high": h, "low": l, "close": c, "volume": v,
            "benchmark": bench, "vix": vix, "interval": "1d",
            "universe": "live"}


def sleeve_positions(ctx, sid, variant):
    if sid.startswith("R-"):
        from strategies.cat_R_promoted import build_positions
        return build_positions(ctx, variant)
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


def main():
    ctx = fetch_live_ctx()
    c = ctx["close"]
    asof = c.index[-1]
    print(f"data through {asof:%Y-%m-%d} | {c.iloc[-1].notna().sum()} tickers")

    month_ends = completed_month_ends(c.index, asof)
    holdings = {}
    for sid, variant, kind in SLEEVES:
        try:
            pos = sleeve_positions(ctx, sid, variant)
            pos = pos.reindex(index=c.index).fillna(0)
            if kind == "monthly" and month_ends:
                row = pos.loc[month_ends[-1]]   # set at last true rebalance
            else:
                row = pos.iloc[-1]
            held = sorted(row[row > 0].index)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {sid} failed: {e}")
            held = None
        holdings[f"{sid} {variant}"] = held

    # ---- mark-to-market previous run ----
    perf_path = os.path.join(FWD, "performance.csv")
    log_path = os.path.join(FWD, "log.csv")
    if os.path.exists(log_path):
        log = pd.read_csv(log_path)
        prev_dates = log[log.asof < f"{asof:%Y-%m-%d}"]
        if len(prev_dates):
            prev_asof = prev_dates.asof.max()
            prev = log[log.asof == prev_asof]
            ret = c.pct_change(fill_method=None)
            if prev_asof in [f"{d:%Y-%m-%d}" for d in c.index]:
                # return from prev close to latest close
                pd_idx = c.index[[f"{d:%Y-%m-%d}" for d in c.index].index(prev_asof)]
                window_ret = c.loc[asof] / c.loc[pd_idx] - 1
                bench_ret = ctx["benchmark"].loc[asof] / ctx["benchmark"].loc[pd_idx] - 1
                sleeve_rets = {}
                for _, lr in prev.iterrows():
                    tk = [] if pd.isna(lr.tickers) or lr.tickers == "" \
                        else lr.tickers.split(";")
                    sleeve_rets[lr.sleeve] = (window_ret[tk].mean()
                                              if tk else 0.0)
                champ_key = f"{CHAMPION[0]} {CHAMPION[1]}"
                meta_ret = np.mean(list(sleeve_rets.values()))
                row = {"from": prev_asof, "to": f"{asof:%Y-%m-%d}",
                       "meta_ret": meta_ret,
                       "champion_ret": sleeve_rets.get(champ_key, np.nan),
                       "nifty_ret": bench_ret}
                pd.DataFrame([row]).to_csv(
                    perf_path, mode="a",
                    header=not os.path.exists(perf_path), index=False)
                print(f"marked-to-market {prev_asof} -> {asof:%Y-%m-%d}: "
                      f"meta {meta_ret*100:+.2f}% | nifty {bench_ret*100:+.2f}%")

    # ---- write today's holdings ----
    rows = []
    for sleeve, held in holdings.items():
        rows.append({"asof": f"{asof:%Y-%m-%d}", "sleeve": sleeve,
                     "n": len(held) if held is not None else -1,
                     "tickers": ";".join(held) if held else ""})
    pd.DataFrame(rows).to_csv(log_path, mode="a",
                              header=not os.path.exists(log_path), index=False)
    snap = os.path.join(FWD, f"holdings_{asof:%Y-%m-%d}.csv")
    pd.DataFrame(rows).to_csv(snap, index=False)

    print("\n=== TARGET HOLDINGS ===")
    for sleeve, held in holdings.items():
        if held is None:
            print(f"{sleeve}: ERROR")
        elif not held:
            print(f"{sleeve}: (flat)")
        else:
            print(f"{sleeve}: {', '.join(t.replace('.NS','') for t in held)}")
    # cumulative forward-test summary
    if os.path.exists(perf_path):
        p = pd.read_csv(perf_path)
        cum_m = (1 + p.meta_ret).prod() - 1
        cum_n = (1 + p.nifty_ret).prod() - 1
        print(f"\nforward test to date ({len(p)} periods): "
              f"meta {cum_m*100:+.2f}% vs Nifty {cum_n*100:+.2f}%")


if __name__ == "__main__":
    main()
