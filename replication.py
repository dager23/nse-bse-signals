"""Wave 3: replication probe — do the top robust families survive on data they
were NOT selected on?  Re-runs each family's best robust variant on
(a) weekly bars (Nifty 50) and (b) the daily midcap universe, and compares
10Y Sharpe against the daily/Nifty-50 baseline.

A family that keeps Sharpe > 0.4 in a replication domain earns a REPLICATES
mark; selection bias does not travel across bar size or universe.
"""
import os

import numpy as np
import pandas as pd

import config
import backtest_engine as be
from meta_portfolio import registry
from run_all import get_ctx
from utils import data as dutil

WIN = config.TIMEFRAMES["10Y"]
N_FAMILIES = 32


def run_one(meta, fn, universe, interval):
    ctx = get_ctx(universe, interval)
    pos = fn(ctx)
    pos_w = dutil.slice_window(pos, WIN["start"], WIN["end"])
    o = dutil.slice_window(ctx["open"], WIN["start"], WIN["end"])
    c = dutil.slice_window(ctx["close"], WIN["start"], WIN["end"])
    b = dutil.slice_window(ctx["benchmark"].to_frame(), WIN["start"], WIN["end"]).iloc[:, 0]
    res = be.run_backtest(pos_w, o, c, benchmark=b, interval=interval,
                          allow_short=meta.get("allow_short", False),
                          signals_are_weights=meta.get("signals_are_weights", False))
    if res is None:
        return np.nan, np.nan
    return res["sharpe"], res["cagr"]


def main():
    df = pd.read_csv(os.path.join(config.RESULTS_DIR, "master_leaderboard.csv"))
    df = df.drop_duplicates(subset=["strategy_id", "variant", "universe",
                                    "interval", "timeframe"], keep="last")
    df = df[(df.universe == "nifty50") & (df.interval == "1d")
            & ~df.variant.str.contains(r"\(static-snapshot\)")]
    piv = df.pivot_table(index=["strategy_id", "name", "variant"],
                         columns="timeframe", values="sharpe")
    rob = piv.dropna(subset=["5Y", "10Y", "15Y"])
    rob = rob[(rob[["5Y", "10Y", "15Y"]] > 0.5).all(axis=1)]
    rob = rob[~(rob.index.get_level_values(0) == "O5")]          # bias suspect
    rob = rob[~rob.index.get_level_values(0).str.startswith("H")]  # daily-panel ML
    # best variant per family by 15Y sharpe
    best = rob.sort_values("15Y", ascending=False).groupby(level=0).head(1)
    best = best.sort_values("15Y", ascending=False).head(N_FAMILIES)

    reg = registry()
    rows = []
    for (sid, name, var), r in best.iterrows():
        key = (sid, var)
        if key not in reg:
            continue
        meta, fn = reg[key]
        try:
            sh_w, cg_w = run_one(meta, fn, "nifty50", "1wk")
        except Exception:
            sh_w, cg_w = np.nan, np.nan
        try:
            sh_m, cg_m = run_one(meta, fn, "midcap", "1d")
        except Exception:
            sh_m, cg_m = np.nan, np.nan
        n_rep = sum(1 for s in (sh_w, sh_m) if not np.isnan(s) and s > 0.4)
        rows.append({"family": f"{sid} {name}", "variant": var,
                     "daily_10Y": r["10Y"], "weekly": sh_w, "midcap": sh_m,
                     "replicates": n_rep})
        print(f"{sid:5s} {name[:28]:30s} daily {r['10Y']:5.2f} | wk "
              f"{sh_w if not np.isnan(sh_w) else float('nan'):5.2f} | mid "
              f"{sh_m if not np.isnan(sh_m) else float('nan'):5.2f}")

    out = pd.DataFrame(rows)
    md = ["# Replication Probe (Wave 3)", "",
          "*Best robust variant of each top family, re-run on 10Y windows it was "
          "not selected on: weekly bars (Nifty 50) and the daily midcap universe. "
          "Sharpe > 0.4 in a domain = replicates there.*", "",
          "| family | variant | daily 10Y | weekly 10Y | midcap 10Y | replicates |",
          "|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        def f(x):
            return "-" if pd.isna(x) else f"{x:.2f}"
        mark = {0: "✗", 1: "◐ 1/2", 2: "✓ 2/2"}[int(r.replicates)]
        md.append(f"| {r.family} | {r.variant} | {f(r.daily_10Y)} | {f(r.weekly)} | "
                  f"{f(r.midcap)} | {mark} |")
    n2 = (out.replicates == 2).sum()
    n1 = (out.replicates == 1).sum()
    md += ["", f"**Summary:** of {len(out)} families, {n2} replicate in both "
           f"domains, {n1} in one, {len(out) - n2 - n1} in neither. Families that "
           "replicate everywhere are the study's most defensible findings."]
    with open(os.path.join(config.REPORTS_DIR, "replication.md"), "w",
              encoding="utf-8") as f_:
        f_.write("\n".join(md))
    out.to_csv(os.path.join(config.RESULTS_DIR, "replication.csv"), index=False)
    print(f"\n{n2} of {len(out)} families replicate in both domains")


if __name__ == "__main__":
    main()
