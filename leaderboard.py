"""Phase 5: aggregate master_leaderboard.csv into ranked leaderboard, tiers,
and cross-sectional analyses.  Writes reports/*.md.

Composite Score = 0.25*Sharpe + 0.20*CAGR + 0.20*(1-MaxDD) + 0.15*Sortino
                + 0.10*WinRate + 0.10*ProfitFactor      (each min-max normalized)

Normalization is done WITHIN each timeframe cohort so 1Y and 15Y rows are
comparable; the headline ranking uses each variant's longest timeframe with
>= 750 bars (10Y/15Y for daily strategies).
"""
import os

import numpy as np
import pandas as pd

import config

MASTER = os.path.join(config.RESULTS_DIR, "master_leaderboard.csv")
REPORTS = config.REPORTS_DIR

WEIGHTS = {"sharpe": 0.25, "cagr": 0.20, "dd_score": 0.20, "sortino": 0.15,
           "win_rate": 0.10, "profit_factor": 0.10}

STATIC_SNAPSHOT = "(static-snapshot)"


def load():
    df = pd.read_csv(MASTER)
    df = df.drop_duplicates(subset=["strategy_id", "variant", "universe",
                                    "interval", "timeframe"], keep="last")
    # clean
    df["profit_factor"] = df["profit_factor"].replace([np.inf, -np.inf], np.nan).clip(upper=5)
    df["sharpe"] = df["sharpe"].clip(-3, 6)
    df["sortino"] = df["sortino"].replace([np.inf, -np.inf], np.nan).clip(-3, 10)
    df["cagr"] = df["cagr"].clip(-1, 2)
    df["dd_score"] = 1 - df["max_dd"].abs().clip(0, 1)   # higher is better
    df["biased"] = df["variant"].str.contains(STATIC_SNAPSHOT, regex=False)
    df["label"] = df["strategy_id"] + " " + df["name"] + " [" + df["variant"] + "]"
    return df


def composite(df):
    df = df.copy()
    df["score"] = np.nan
    for tf, grp in df.groupby("timeframe"):
        sc = pd.Series(0.0, index=grp.index)
        for m, w in WEIGHTS.items():
            x = grp[m].astype(float)
            lo, hi = x.min(), x.max()
            norm = (x - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=x.index)
            sc += w * norm.fillna(0.3)   # missing metric -> below-median credit
        df.loc[grp.index, "score"] = sc
    return df


def headline(df):
    """One row per variant: longest timeframe with enough bars."""
    pref = {"15Y": 5, "10Y": 4, "5Y": 3, "3Y": 2, "1Y": 1}
    d = df[df.n_bars >= 200].copy()
    d["tf_rank"] = d.timeframe.map(pref)
    d = d.sort_values("tf_rank", ascending=False)
    head = d.groupby(["strategy_id", "variant", "universe", "interval"], as_index=False).first()
    head = head.sort_values("score", ascending=False).reset_index(drop=True)
    head["rank"] = np.arange(1, len(head) + 1)
    n = len(head)
    tiers = pd.Series("D", index=head.index)
    tiers[head.index < n * 0.60] = "C"
    tiers[head.index < n * 0.35] = "B"
    tiers[head.index < n * 0.15] = "A"
    tiers[head.index < n * 0.05] = "S"
    head["tier"] = tiers
    return head


def fmt_pct(x):
    return "-" if pd.isna(x) else f"{x*100:.1f}%"


def fmt2(x):
    return "-" if pd.isna(x) else f"{x:.2f}"


def table_md(d, cols=None):
    cols = cols or ["rank", "tier", "strategy_id", "name", "variant", "timeframe",
                    "cagr", "sharpe", "sortino", "max_dd", "win_rate",
                    "profit_factor", "total_trades", "exposure", "alpha", "score"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in d[cols].iterrows():
        cells = []
        for c_ in cols:
            v = r[c_]
            if c_ in ("cagr", "max_dd", "win_rate", "exposure", "alpha"):
                cells.append(fmt_pct(v))
            elif c_ in ("sharpe", "sortino", "profit_factor", "score"):
                cells.append(fmt2(v))
            elif c_ == "total_trades":
                cells.append("-" if pd.isna(v) else str(int(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    os.makedirs(REPORTS, exist_ok=True)
    df = composite(load())
    head = headline(df)
    head.to_csv(os.path.join(config.RESULTS_DIR, "headline_leaderboard.csv"), index=False)

    bench_row = df[(df.strategy_id == "I10") & (df.timeframe == head[head.strategy_id == "I10"].timeframe.iloc[0])] if (df.strategy_id == "I10").any() else None

    # ---------- master_leaderboard.md ----------
    md = ["# Master Leaderboard — NSE Strategy Research",
          "",
          f"*Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} | universe: Nifty 50 (+mid/small variants) | "
          f"costs {config.TRANSACTION_COST*100:.2f}% + slippage {config.SLIPPAGE*100:.2f}% per side | "
          "signals execute at next open*",
          "",
          f"**{len(head)} strategy variants** ranked by composite score on their longest "
          "available timeframe (10Y/15Y for most).",
          "",
          "> **Bias flags**: variants marked `(static-snapshot)` use TODAY'S fundamentals "
          "across history (lookahead) — shown for completeness, excluded from tiers S/A "
          "eligibility commentary. Proxy-labelled event strategies approximate unavailable "
          "data feeds.",
          "",
          "## Full ranking",
          "",
          table_md(head)]
    with open(os.path.join(REPORTS, "master_leaderboard.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ---------- tier_analysis.md ----------
    md = ["# Tier Analysis", ""]
    for tier in ["S", "A", "B", "C", "D"]:
        sub = head[head.tier == tier]
        md += [f"## {tier} Tier ({len(sub)} variants)", ""]
        if len(sub):
            md.append(table_md(sub, ["rank", "strategy_id", "name", "variant", "timeframe",
                                     "cagr", "sharpe", "max_dd", "score"]))
        md.append("")
    with open(os.path.join(REPORTS, "tier_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    # ---------- category_analysis.md ----------
    md = ["# Category Analysis", ""]
    cat = head.groupby("category").agg(
        n=("score", "size"), mean_score=("score", "mean"),
        best_score=("score", "max"), mean_sharpe=("sharpe", "mean"),
        mean_cagr=("cagr", "mean"), mean_dd=("max_dd", "mean")).sort_values(
        "mean_score", ascending=False).reset_index()
    md += ["## By category (headline timeframe)", "",
           "| category | n | mean score | best score | mean Sharpe | mean CAGR | mean MaxDD |",
           "|---|---|---|---|---|---|---|"]
    for _, r in cat.iterrows():
        md.append(f"| {r.category} | {int(r.n)} | {r.mean_score:.2f} | {r.best_score:.2f} | "
                  f"{r.mean_sharpe:.2f} | {fmt_pct(r.mean_cagr)} | {fmt_pct(r.mean_dd)} |")
    # timeframe analysis
    tf = df.groupby("timeframe").agg(mean_sharpe=("sharpe", "mean"),
                                     med_cagr=("cagr", "median"),
                                     mean_dd=("max_dd", "mean")).reindex(
        ["1Y", "3Y", "5Y", "10Y", "15Y"]).reset_index()
    md += ["", "## By timeframe (all variants)", "",
           "| timeframe | mean Sharpe | median CAGR | mean MaxDD |", "|---|---|---|---|"]
    for _, r in tf.iterrows():
        md.append(f"| {r.timeframe} | {fmt2(r.mean_sharpe)} | {fmt_pct(r.med_cagr)} | {fmt_pct(r.mean_dd)} |")
    # universe analysis
    uni = df[df.timeframe.isin(["5Y", "10Y"])].groupby("universe").agg(
        mean_sharpe=("sharpe", "mean"), med_cagr=("cagr", "median")).reset_index()
    md += ["", "## By universe (5Y/10Y rows)", "",
           "| universe | mean Sharpe | median CAGR |", "|---|---|---|"]
    for _, r in uni.iterrows():
        md.append(f"| {r.universe} | {fmt2(r.mean_sharpe)} | {fmt_pct(r.med_cagr)} |")
    with open(os.path.join(REPORTS, "category_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"headline variants: {len(head)}")
    print(head.head(15)[["rank", "tier", "label", "timeframe", "cagr", "sharpe",
                         "max_dd", "score"]].to_string(index=False))
    return head


if __name__ == "__main__":
    main()
