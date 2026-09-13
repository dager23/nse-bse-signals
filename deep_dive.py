"""Phase 5.3/6: deep dive on the top 10 unbiased strategies.

For each: metrics across every timeframe, parameter-sensitivity vs sibling
variants, and bull/bear regime split (Nifty above/below its 200DMA) using the
cached 10Y return streams produced by meta_portfolio.py.
"""
import os
import re

import numpy as np
import pandas as pd

import config
from run_all import get_ctx
from utils import data as dutil

RET_CACHE = os.path.join(config.RESULTS_DIR, "returns_cache")


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s)[:60]


def fmt_pct(x):
    return "-" if pd.isna(x) else f"{x*100:.1f}%"


def fmt2(x):
    return "-" if pd.isna(x) else f"{x:.2f}"


def main():
    master = pd.read_csv(os.path.join(config.RESULTS_DIR, "master_leaderboard.csv"))
    master = master.drop_duplicates(subset=["strategy_id", "variant", "universe",
                                            "interval", "timeframe"], keep="last")
    head = pd.read_csv(os.path.join(config.RESULTS_DIR, "headline_leaderboard.csv"))
    top10 = head[~head.biased].head(10)

    ctx = get_ctx()
    win = config.TIMEFRAMES["10Y"]
    bench = dutil.slice_window(ctx["benchmark"].to_frame(), win["start"], win["end"]).iloc[:, 0]
    bull_mask = (bench > bench.rolling(200, min_periods=50).mean())

    md = ["# Top 10 Deep Dive", "",
          "*Top 10 bias-free variants by composite score. Regime split uses Nifty vs "
          "its 200DMA over the 10Y window; sleeve returns are net of costs.*", ""]

    for i, (_, r) in enumerate(top10.iterrows(), 1):
        md += [f"## {i}. {r.strategy_id} — {r['name']} `[{r.variant}]`",
               "",
               f"Tier **{r.tier}**, composite score **{r.score:.2f}** "
               f"(headline timeframe {r.timeframe}).",
               ""]
        if r.strategy_id == "O5":
            md += ["> ⚠️ **Survivorship-bias suspect**: overweighting today's "
                   "least-liquid index members is where backfill bias concentrates; "
                   "excluded from the meta-portfolio. See robustness report.", ""]
        # all-timeframe table
        rows = master[(master.strategy_id == r.strategy_id) & (master.variant == r.variant)
                      & (master.universe == r.universe)].sort_values(
            "timeframe", key=lambda s: s.map({"1Y": 0, "3Y": 1, "5Y": 2, "10Y": 3, "15Y": 4}))
        md += ["| timeframe | CAGR | Sharpe | Sortino | MaxDD | trades | win rate | exposure |",
               "|---|---|---|---|---|---|---|---|"]
        for _, t in rows.iterrows():
            md.append(f"| {t.timeframe} | {fmt_pct(t.cagr)} | {fmt2(t.sharpe)} | "
                      f"{fmt2(t.sortino)} | {fmt_pct(t.max_dd)} | "
                      f"{'-' if pd.isna(t.total_trades) else int(t.total_trades)} | "
                      f"{fmt_pct(t.win_rate)} | {fmt_pct(t.exposure)} |")
        md.append("")
        # consistency check
        tf_sharpes = rows.set_index("timeframe")["sharpe"]
        n_pos = (tf_sharpes > 0.5).sum()
        md.append(f"**Robustness across timeframes:** Sharpe > 0.5 in {n_pos}/{len(tf_sharpes)} windows.")
        # parameter sensitivity: sibling variants of same strategy_id
        sibs = head[head.strategy_id == r.strategy_id]
        if len(sibs) > 1:
            md.append(f"**Parameter sensitivity:** {len(sibs)} variants tested; composite scores "
                      f"range {sibs.score.min():.2f}–{sibs.score.max():.2f} "
                      f"(spread {(sibs.score.max()-sibs.score.min()):.2f} — "
                      f"{'robust' if sibs.score.max()-sibs.score.min() < 0.12 else 'parameter-sensitive'}).")
        # regime split from cached returns
        path = os.path.join(RET_CACHE, f"{r.strategy_id}_{slug(r.variant)}.parquet")
        if os.path.exists(path):
            ret = pd.read_parquet(path).iloc[:, 0]
            bm = bull_mask.reindex(ret.index).fillna(False)
            r_bull = ret[bm]
            r_bear = ret[~bm]
            md.append(f"**Regime split (10Y):** bull (Nifty>200DMA, {bm.mean()*100:.0f}% of days) "
                      f"ann. return {fmt_pct(r_bull.mean()*252)}; "
                      f"bear ann. return {fmt_pct(r_bear.mean()*252)}.")
        md.append(f"\n![equity](../results/plots/strategy_{r.strategy_id}_equity.png)\n")

    with open(os.path.join(config.REPORTS_DIR, "top_10_deep_dive.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))
    print("top_10_deep_dive.md written for:",
          ", ".join(top10.strategy_id + " " + top10.variant))


if __name__ == "__main__":
    main()
