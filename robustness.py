"""Robustness & multiple-testing analysis over the full strategy zoo.

Outputs reports/robustness_multiple_testing.md:
  1. Expected max Sharpe under the null given N tested variants (mining bar).
  2. The ROBUST SET: unbiased variants with Sharpe > 0.5 in 5Y AND 10Y AND
     15Y windows and positive 10Y alpha — survivors of every regime tested.
  3. Per-family parameter sensitivity (score dispersion within strategy id).
  4. Wave comparison: hand-built (A-L) vs generated (M-Q).
"""
import os

import numpy as np
import pandas as pd

import config

REPORTS = config.REPORTS_DIR


def fmt_pct(x):
    return "-" if pd.isna(x) else f"{x*100:.1f}%"


def main():
    df = pd.read_csv(os.path.join(config.RESULTS_DIR, "master_leaderboard.csv"))
    df = df.drop_duplicates(subset=["strategy_id", "variant", "universe",
                                    "interval", "timeframe"], keep="last")
    df["biased"] = df["variant"].str.contains(r"\(static-snapshot\)", regex=True)
    head = pd.read_csv(os.path.join(config.RESULTS_DIR, "headline_leaderboard.csv"))

    n_variants = df.groupby(["strategy_id", "variant"]).ngroups
    # include mass-screened configs in the total tested count
    import glob
    n_screened = 0
    for p in glob.glob(os.path.join(config.RESULTS_DIR, "screen", "*.parquet")):
        n_screened += len(pd.read_parquet(p, columns=["family"]))
    n_total = n_variants + n_screened
    # mining bar: expected max |Sharpe| of N iid noise strategies over T years
    T = 15
    bar15 = np.sqrt(2 * np.log(max(n_total, 2))) / np.sqrt(T)
    bar10 = np.sqrt(2 * np.log(max(n_total, 2))) / np.sqrt(10)

    # robust set
    piv = df[~df.biased].pivot_table(index=["strategy_id", "name", "variant"],
                                     columns="timeframe", values="sharpe")
    alpha10 = df[~df.biased].pivot_table(index=["strategy_id", "name", "variant"],
                                         columns="timeframe", values="alpha")["10Y"]
    cagr15 = df[~df.biased].pivot_table(index=["strategy_id", "name", "variant"],
                                        columns="timeframe", values="cagr").get("15Y")
    dd15 = df[~df.biased].pivot_table(index=["strategy_id", "name", "variant"],
                                      columns="timeframe", values="max_dd").get("15Y")
    need = ["5Y", "10Y", "15Y"]
    have = piv.dropna(subset=[c for c in need if c in piv.columns])
    robust = have[(have[need] > 0.5).all(axis=1) & (alpha10.reindex(have.index) > 0)]
    robust = robust.assign(alpha10=alpha10.reindex(robust.index),
                           cagr15=cagr15.reindex(robust.index),
                           dd15=dd15.reindex(robust.index))
    robust = robust.sort_values("15Y", ascending=False)

    # exceptional set: clears the mining bar in BOTH 10Y and 15Y
    exceptional = robust[(robust["15Y"] > bar15) & (robust["10Y"] > bar10)]

    # family sensitivity
    fam = head.groupby(["strategy_id", "name"]).agg(
        n=("score", "size"), best=("score", "max"), worst=("score", "min"),
        spread=("score", lambda s: s.max() - s.min())).reset_index()
    fragile = fam[(fam.n >= 4) & (fam.spread > 0.25)].sort_values("spread", ascending=False)

    # wave comparison
    head["wave"] = np.where(head.category.str.match(r"^[M-Q]-"), "generated (M-Q)",
                            "hand-built (A-L)")
    wave = head.groupby("wave").agg(n=("score", "size"), mean_score=("score", "mean"),
                                    best=("score", "max"),
                                    mean_sharpe=("sharpe", "mean")).reset_index()

    md = ["# Robustness & Multiple-Testing Analysis", "",
          f"**{n_variants} variants** ran through the full multi-timeframe engine; "
          f"**{n_screened:,} additional configurations** were evaluated by the wave-4 "
          f"mass screener — **{n_total:,} techniques tested in total**. Mining this "
          "many configurations guarantees impressive-looking winners by chance alone.",
          "",
          "## The mining bar", "",
          f"Expected MAX Sharpe of {n_total:,} pure-noise strategies: "
          f"**~{bar15:.2f}** over 15Y, **~{bar10:.2f}** over 10Y "
          "(sqrt(2 ln N / T) approximation).",
          "",
          "Any single variant's Sharpe below these bars is *indistinguishable from "
          "luck* once selection is accounted for. Confidence should come from (a) "
          "consistency across independent windows, (b) economic rationale, (c) whole "
          "*families* working rather than lone parameter picks.", "",
          f"## Robust set — Sharpe > 0.5 in 5Y, 10Y AND 15Y + positive 10Y alpha "
          f"({len(robust)} of {len(have)} eligible variants)", "",
          "| strategy | variant | 5Y | 10Y | 15Y | 15Y CAGR | 15Y MaxDD | 10Y alpha |",
          "|---|---|---|---|---|---|---|---|"]
    for (sid, name, var), r in robust.head(40).iterrows():
        md.append(f"| {sid} {name} | {var} | {r['5Y']:.2f} | {r['10Y']:.2f} | "
                  f"{r['15Y']:.2f} | {fmt_pct(r.cagr15)} | {fmt_pct(r.dd15)} | "
                  f"{fmt_pct(r.alpha10)} |")
    md += ["",
           f"**Exceptional set** (clears the mining bar in both long windows): "
           f"{len(exceptional)} variants:", ""]
    for (sid, name, var), r in exceptional.iterrows():
        md.append(f"- {sid} {name} [{var}] — 10Y {r['10Y']:.2f} / 15Y {r['15Y']:.2f}")
    md += ["", "## Parameter-fragile families (score spread > 0.25 across variants)", "",
           "| family | n variants | best | worst | spread |", "|---|---|---|---|---|"]
    for _, r in fragile.head(20).iterrows():
        md.append(f"| {r.strategy_id} {r['name']} | {int(r.n)} | {r.best:.2f} | "
                  f"{r.worst:.2f} | {r.spread:.2f} |")
    md += ["", "## Bias-suspect standouts", "",
           "- **O5 Amihud illiquid tilt** (15Y Sharpe ~1.4-1.5) is the single "
           "strongest result in the zoo — and the most suspect. Overweighting "
           "today's *least-liquid* Nifty 50 names is precisely where survivorship "
           "bias concentrates: those are the stocks that grew into the index. "
           "Excluded from the meta-portfolio; treat as an artifact until tested on "
           "point-in-time constituents.",
           "", "## Hand-built vs generated", "",
           "| wave | n | mean score | best score | mean Sharpe |", "|---|---|---|---|---|"]
    for _, r in wave.iterrows():
        md.append(f"| {r.wave} | {int(r.n)} | {r.mean_score:.2f} | {r.best:.2f} | "
                  f"{r.mean_sharpe:.2f} |")
    with open(os.path.join(REPORTS, "robustness_multiple_testing.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"variants={n_variants} bar15={bar15:.2f} bar10={bar10:.2f} "
          f"robust={len(robust)} exceptional={len(exceptional)}")


if __name__ == "__main__":
    main()
