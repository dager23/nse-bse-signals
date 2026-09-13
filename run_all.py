"""Master runner: executes every registered strategy variant across all
timeframes, appends metrics to results/master_leaderboard.csv (checkpointed,
resumable), and saves an equity plot for the best variant of each strategy.

Usage:  python run_all.py --cats A,B,C          # run selected categories
        python run_all.py --cats all            # run everything
"""
import argparse
import importlib
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import backtest_engine as be
from utils import data as dutil

MASTER = os.path.join(config.RESULTS_DIR, "master_leaderboard.csv")
ERRLOG = os.path.join(config.RESULTS_DIR, "error_log.txt")

CATEGORY_MODULES = {
    "A": "strategies.cat_A_trend",
    "B": "strategies.cat_B_meanrev",
    "C": "strategies.cat_C_momentum",
    "D": "strategies.cat_D_volume",
    "E": "strategies.cat_E_volatility",
    "F": "strategies.cat_F_patterns",
    "G": "strategies.cat_G_statistical",
    "H": "strategies.cat_H_ml",
    "I": "strategies.cat_I_factor",
    "J": "strategies.cat_J_seasonal",
    "K": "strategies.cat_K_event",
    "L": "strategies.cat_L_hybrid",
    "M": "strategies.cat_M_grids",
    "N": "strategies.cat_N_composite",
    "O": "strategies.cat_O_novel",
    "P": "strategies.cat_P_rankblend",
    "Q": "strategies.cat_Q_overlay",
    "R": "strategies.cat_R_promoted",
}

METRIC_COLS = ["total_return", "cagr", "volatility", "sharpe", "sortino", "calmar",
               "max_dd", "dd_duration", "ulcer", "recovery_factor", "exposure",
               "alpha", "beta", "info_ratio", "treynor", "n_bars", "total_trades",
               "win_rate", "avg_win", "avg_loss", "profit_factor", "expectancy",
               "avg_holding_days"]


def build_context(universe="nifty50", interval="1d"):
    o, h, l, c, v = dutil.get_ohlcv(universe, interval)
    bench, vix = dutil.load_benchmark(interval)
    bench = bench.reindex(c.index).ffill()
    return {"open": o, "high": h, "low": l, "close": c, "volume": v,
            "benchmark": bench, "vix": vix, "interval": interval,
            "universe": universe}


_CTX_CACHE = {}


def get_ctx(universe="nifty50", interval="1d"):
    key = (universe, interval)
    if key not in _CTX_CACHE:
        _CTX_CACHE[key] = build_context(universe, interval)
    return _CTX_CACHE[key]


def load_done():
    if os.path.exists(MASTER):
        df = pd.read_csv(MASTER)
        return set(zip(df.strategy_id, df.variant, df.timeframe, df.universe, df.interval)), df
    return set(), None


def run_variant(meta, positions, results_rows, best_tracker):
    """Backtest one variant across all timeframes."""
    ctx = get_ctx(meta.get("universe", "nifty50"), meta.get("interval", "1d"))
    interval = meta.get("interval", "1d")
    for tf, win in config.TIMEFRAMES.items():
        if interval != "1d" and tf in ("1Y",):
            pass  # weekly/monthly still fine on 1Y, just few bars
        pos_w = dutil.slice_window(positions, win["start"], win["end"])
        if len(pos_w) < 10:
            continue
        o = dutil.slice_window(ctx["open"], win["start"], win["end"])
        c = dutil.slice_window(ctx["close"], win["start"], win["end"])
        b = dutil.slice_window(ctx["benchmark"].to_frame(), win["start"], win["end"]).iloc[:, 0]
        try:
            res = be.run_backtest(pos_w, o, c, benchmark=b, interval=interval,
                                  allow_short=meta.get("allow_short", False),
                                  signals_are_weights=meta.get("signals_are_weights", False))
        except Exception:
            with open(ERRLOG, "a") as f:
                f.write(f"BACKTEST FAIL {meta['id']} {meta['variant']} {tf}\n{traceback.format_exc()}\n")
            continue
        if res is None:
            continue
        row = {"strategy_id": meta["id"], "name": meta["name"], "category": meta["category"],
               "variant": meta["variant"], "universe": meta.get("universe", "nifty50"),
               "interval": interval, "timeframe": tf}
        row.update({k: res[k] for k in METRIC_COLS})
        results_rows.append(row)
        # track best equity curve for plotting (prefer 10Y, fall back to longest)
        pref = {"10Y": 3, "15Y": 2, "5Y": 1}.get(tf, 0)
        key = meta["id"]
        score = (pref, res["sharpe"] if not np.isnan(res["sharpe"]) else -9)
        if key not in best_tracker or score > best_tracker[key][0]:
            best_tracker[key] = (score, meta["variant"], tf, res["_equity"],
                                 b.reindex(res["_equity"].index))


def flush_rows(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    header = not os.path.exists(MASTER)
    df.to_csv(MASTER, mode="a", header=header, index=False)
    rows.clear()


def plot_best(best_tracker):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    for sid, (score, variant, tf, eq, bench) in best_tracker.items():
        try:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            eq.plot(ax=ax, label=f"{sid} {variant}", lw=1.2)
            if bench is not None and bench.notna().any():
                (bench / bench.dropna().iloc[0]).plot(ax=ax, label="Nifty 50", lw=1.0, alpha=0.7)
            ax.set_title(f"{sid} — {variant} ({tf})")
            ax.legend()
            ax.set_ylabel("Growth of Rs 1")
            fig.tight_layout()
            fig.savefig(os.path.join(config.PLOTS_DIR, f"strategy_{sid}_equity.png"), dpi=90)
            plt.close(fig)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cats", default="all")
    args = ap.parse_args()
    cats = list(CATEGORY_MODULES) if args.cats == "all" else args.cats.split(",")

    done, _ = load_done()
    rows = []
    best = {}
    t0 = time.time()
    n_run = n_skip = n_fail = 0

    for cat in cats:
        mod_name = CATEGORY_MODULES[cat.strip().upper()]
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            with open(ERRLOG, "a") as f:
                f.write(f"IMPORT FAIL {mod_name}\n{traceback.format_exc()}\n")
            print(f"[FAIL] import {mod_name}")
            continue
        for meta, fn in mod.get_strategies():
            uni = meta.get("universe", "nifty50")
            iv = meta.get("interval", "1d")
            already = all((meta["id"], meta["variant"], tf, uni, iv) in done
                          for tf in config.TIMEFRAMES)
            if already:
                n_skip += 1
                continue
            try:
                ctx = get_ctx(uni, iv)
                positions = fn(ctx)
                run_variant(meta, positions, rows, best)
                n_run += 1
                print(f"[ok] {meta['id']} {meta['variant']} ({time.time() - t0:.0f}s)")
            except Exception:
                n_fail += 1
                with open(ERRLOG, "a") as f:
                    f.write(f"SIGNAL FAIL {meta['id']} {meta['variant']}\n{traceback.format_exc()}\n")
                print(f"[FAIL] {meta['id']} {meta['variant']}")
            if len(rows) > 40:
                flush_rows(rows)
    flush_rows(rows)
    plot_best(best)
    print(f"DONE cats={cats} ran={n_run} skipped={n_skip} failed={n_fail} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
