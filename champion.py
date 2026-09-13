"""Final stage: select THE champion algorithm.

Candidate filter (all must hold):
  1. Bias-clean (no static-snapshot, no O5 illiquid tilt).
  2. Robust: Sharpe > 0.5 in 5Y AND 10Y AND 15Y full-engine windows.
  3. Replicates: Sharpe > 0.4 on BOTH weekly bars and the midcap universe.

Champion = highest mean Sharpe across the four independent views
(10Y daily, 15Y daily, weekly, midcap).  Compared head-to-head against the
meta-portfolio, which remains the recommended deployable.
"""
import os

import numpy as np
import pandas as pd

import config


def main():
    df = pd.read_csv(os.path.join(config.RESULTS_DIR, "master_leaderboard.csv"))
    df = df.drop_duplicates(subset=["strategy_id", "variant", "universe",
                                    "interval", "timeframe"], keep="last")
    df = df[(df.universe == "nifty50") & (df.interval == "1d")]
    df = df[~df.variant.str.contains(r"\(static-snapshot\)")]
    df = df[~((df.strategy_id == "O5") & df.variant.str.contains("illiquid"))]

    piv_sh = df.pivot_table(index=["strategy_id", "name", "variant"],
                            columns="timeframe", values="sharpe")
    piv_cg = df.pivot_table(index=["strategy_id", "name", "variant"],
                            columns="timeframe", values="cagr")
    piv_dd = df.pivot_table(index=["strategy_id", "name", "variant"],
                            columns="timeframe", values="max_dd")
    rob = piv_sh.dropna(subset=["5Y", "10Y", "15Y"])
    rob = rob[(rob[["5Y", "10Y", "15Y"]] > 0.5).all(axis=1)]

    rep = pd.read_csv(os.path.join(config.RESULTS_DIR, "replication.csv"))
    rep["sid"] = rep.family.str.split(" ").str[0]
    rep_ok = rep[(rep.weekly > 0.4) & (rep.midcap > 0.4)]

    rows = []
    for (sid, name, var), r in rob.iterrows():
        m = rep_ok[(rep_ok.sid == sid) & (rep_ok.variant == var)]
        if m.empty:
            continue
        wk, mid = m.iloc[0].weekly, m.iloc[0].midcap
        mean_sh = np.mean([r["10Y"], r["15Y"], wk, mid])
        rows.append({"sid": sid, "name": name, "variant": var,
                     "sh_10y": r["10Y"], "sh_15y": r["15Y"], "sh_wk": wk,
                     "sh_mid": mid, "mean_sharpe": mean_sh,
                     "cagr_15y": piv_cg.loc[(sid, name, var), "15Y"],
                     "dd_15y": piv_dd.loc[(sid, name, var), "15Y"]})
    cand = pd.DataFrame(rows).sort_values("mean_sharpe", ascending=False)
    cand.to_csv(os.path.join(config.RESULTS_DIR, "champion_candidates.csv"), index=False)

    champ = cand.iloc[0]
    md = ["# Champion Algorithm Selection", "",
          "Candidates must be bias-clean, robust in every full-engine window "
          "(5Y/10Y/15Y Sharpe > 0.5), and replicate on weekly bars AND midcaps "
          "(Sharpe > 0.4). Ranked by mean Sharpe across the four independent views.",
          "",
          f"**{len(cand)} candidates survived all filters.**", "",
          "## The champion", "",
          f"### {champ.sid} {champ['name']} `[{champ.variant}]`", "",
          f"- Mean Sharpe across views: **{champ.mean_sharpe:.2f}**",
          f"- 15Y: CAGR {champ.cagr_15y*100:.1f}%, Sharpe {champ.sh_15y:.2f}, "
          f"MaxDD {champ.dd_15y*100:.1f}%",
          f"- 10Y Sharpe {champ.sh_10y:.2f} | weekly {champ.sh_wk:.2f} | "
          f"midcap {champ.sh_mid:.2f}", "",
          "## Top 15 candidates", "",
          "| rank | strategy | variant | 10Y | 15Y | weekly | midcap | mean | 15Y CAGR | 15Y DD |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, (_, r) in enumerate(cand.head(15).iterrows(), 1):
        md.append(f"| {i} | {r.sid} {r['name']} | {r.variant} | {r.sh_10y:.2f} | "
                  f"{r.sh_15y:.2f} | {r.sh_wk:.2f} | {r.sh_mid:.2f} | "
                  f"**{r.mean_sharpe:.2f}** | {r.cagr_15y*100:.1f}% | "
                  f"{r.dd_15y*100:.1f}% |")
    md += ["", "## Champion vs meta-portfolio", "",
           "The single-strategy champion is the best *individual* rule set, but the "
           "**meta-portfolio remains the recommended deployable**: five decorrelated "
           "sleeves cannot all be a mining artifact at once, and its drawdown "
           "profile is structurally better than any single sleeve's. Use the "
           "champion as the core sleeve; use the meta-portfolio as the portfolio.",
           "", "*(See meta_portfolio.md for the combined result.)*"]
    with open(os.path.join(config.REPORTS_DIR, "champion.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"{len(cand)} candidates; champion: {champ.sid} {champ['name']} "
          f"[{champ.variant}] mean Sharpe {champ.mean_sharpe:.2f}")


if __name__ == "__main__":
    main()
