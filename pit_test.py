"""Survivorship-corrected re-test: run the 31 champion candidates on the
point-in-time (PIT) Nifty 50 universe.

Universe = union of all 2010-2025 members (current + ex-constituents with
available data).  Strategies see full price history (as a real manager
would), but positions are masked to zero whenever a stock was NOT an index
member — so the tradeable set at every date is the actual index of that day.

Outputs reports/survivorship_corrected.md + results/pit_results.csv.
"""
import os

import numpy as np
import pandas as pd

import config
import backtest_engine as be
from meta_portfolio import registry
from utils import data as dutil
from utils.membership import membership_matrix

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def build_pit_ctx():
    # base (current members) frames
    o, h, l, c, v = dutil.get_ohlcv("nifty50", "1d")
    base = {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}
    # ex-members
    extra_path = os.path.join(config.DATA_DIR, "pit_extra_1d.parquet")
    panel = pd.read_parquet(extra_path)
    tickers = sorted(set(panel.columns.get_level_values(0)))
    merged = {}
    for f in FIELDS:
        extra = pd.concat({t: panel[t][f] for t in tickers}, axis=1)
        extra = extra.reindex(base[f].index)
        merged[f] = pd.concat([base[f], extra], axis=1)
        merged[f] = merged[f].loc[:, ~merged[f].columns.duplicated()]
        merged[f] = merged[f].sort_index(axis=1)
    bench, vix = dutil.load_benchmark("1d")
    bench = bench.reindex(merged["Close"].index).ffill()
    return {"open": merged["Open"], "high": merged["High"], "low": merged["Low"],
            "close": merged["Close"], "volume": merged["Volume"],
            "benchmark": bench, "vix": vix, "interval": "1d",
            "universe": "nifty50_pit"}


def run_window(pos, ctx, start, end):
    sl = lambda df: df.loc[start:end]
    res = be.run_backtest(sl(pos), sl(ctx["open"]), sl(ctx["close"]),
                          benchmark=ctx["benchmark"].loc[start:end], interval="1d")
    if res is None:
        return {}
    return {k: res[k] for k in ["sharpe", "cagr", "max_dd", "alpha"]}


def main():
    ctx = build_pit_ctx()
    c = ctx["close"]
    M = membership_matrix(c.index, c.columns)

    # coverage stat: member-days with a price
    memb_days = int(M.sum().sum())
    have = int((M & c.notna()).sum().sum())
    coverage = have / memb_days
    n_members_daily = (M & c.notna()).sum(axis=1)

    reg = registry()
    cand = pd.read_csv(os.path.join(config.RESULTS_DIR, "champion_candidates.csv"))
    master = pd.read_csv(os.path.join(config.RESULTS_DIR, "master_leaderboard.csv"))
    master = master.drop_duplicates(subset=["strategy_id", "variant", "universe",
                                            "interval", "timeframe"], keep="last")

    w10 = config.TIMEFRAMES["10Y"]
    w15 = config.TIMEFRAMES["15Y"]
    rows = []
    for _, r in cand.iterrows():
        key = (r.sid, r.variant)
        if key not in reg:
            continue
        meta, fn = reg[key]
        if meta.get("signals_are_weights"):
            weights_flag = True
        else:
            weights_flag = False
        try:
            pos = fn(ctx)
        except Exception as e:  # noqa: BLE001
            print(f"skip {key}: {e}")
            continue
        pos = pos.reindex(index=c.index, columns=c.columns).fillna(0)
        pos = pos.where(M, 0.0)
        m10 = run_window_weights(pos, ctx, w10, weights_flag)
        m15 = run_window_weights(pos, ctx, w15, weights_flag)
        bf = master[(master.strategy_id == r.sid) & (master.variant == r.variant)
                    & (master.universe == "nifty50") & (master.interval == "1d")]
        bf15 = bf[bf.timeframe == "15Y"]["sharpe"]
        bf10 = bf[bf.timeframe == "10Y"]["sharpe"]
        rows.append({
            "sid": r.sid, "name": r["name"], "variant": r.variant,
            "bf_sh_10y": float(bf10.iloc[0]) if len(bf10) else np.nan,
            "pit_sh_10y": m10.get("sharpe", np.nan),
            "bf_sh_15y": float(bf15.iloc[0]) if len(bf15) else np.nan,
            "pit_sh_15y": m15.get("sharpe", np.nan),
            "pit_cagr_15y": m15.get("cagr", np.nan),
            "pit_dd_15y": m15.get("max_dd", np.nan),
            "pit_alpha_10y": m10.get("alpha", np.nan),
        })
        print(f"{r.sid:5s} {str(r.variant)[:45]:47s} "
              f"15Y {rows[-1]['bf_sh_15y']:.2f} -> {rows[-1]['pit_sh_15y']:.2f}")

    out = pd.DataFrame(rows)
    out["delta_15y"] = out.pit_sh_15y - out.bf_sh_15y
    out = out.sort_values("pit_sh_15y", ascending=False)
    out.to_csv(os.path.join(config.RESULTS_DIR, "pit_results.csv"), index=False)

    # equal-weight yardstick on PIT universe
    ew_pos = pd.DataFrame(1.0, index=c.index, columns=c.columns).where(M, 0.0)
    me = pd.Series(c.index, index=c.index).groupby(
        [c.index.year, c.index.month]).transform("max") == c.index
    ew = ew_pos.where(pd.DataFrame(np.tile(me.to_numpy()[:, None],
                                           (1, c.shape[1])), index=c.index,
                                   columns=c.columns), np.nan).ffill().fillna(0)
    ew15 = run_window(ew, ctx, w15["start"], w15["end"])

    md = ["# Survivorship-Corrected Re-Test (Point-in-Time Universe)", "",
          f"*Universe = the actual Nifty 50 of each date (membership reconstructed "
          f"from public reconstitution logs; {len(c.columns)} ever-member tickers, "
          f"price coverage {coverage*100:.1f}% of member-days; median members with "
          f"data per day: {int(n_members_daily.median())}). Strategies use full "
          "price history but may only HOLD current members.*", "",
          f"**Equal-weight monthly baseline on PIT universe (15Y): Sharpe "
          f"{ew15.get('sharpe', float('nan')):.2f}, CAGR "
          f"{ew15.get('cagr', float('nan'))*100:.1f}%** — compare 0.88 / 21.4% on "
          "the backfilled universe. The gap is the measured survivorship premium.",
          "",
          "| strategy | variant | 15Y backfilled | 15Y PIT | Δ | 10Y PIT | "
          "PIT CAGR 15Y | PIT DD 15Y |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in out.iterrows():
        md.append(f"| {r.sid} {r['name']} | {str(r.variant)[:40]} | "
                  f"{r.bf_sh_15y:.2f} | **{r.pit_sh_15y:.2f}** | "
                  f"{r.delta_15y:+.2f} | {r.pit_sh_10y:.2f} | "
                  f"{r.pit_cagr_15y*100:.1f}% | {r.pit_dd_15y*100:.1f}% |")
    surv = out[(out.pit_sh_15y > 0.5) & (out.pit_sh_10y > 0.5)]
    md += ["", f"**{len(surv)} of {len(out)} candidates keep Sharpe > 0.5 in both "
           "PIT windows.** These are the study's final defensible set.", "",
           "Membership-table caveats: a handful of 2013-18 re-entry dates are "
           "approximate (VEDL, BANKBARODA, NMDC, HINDPETRO, INDUSTOWER, YESBANK); "
           "worst-case error is one slot of 50 for limited stretches. Tickers "
           "with no recoverable data are treated as never-members (bias direction: "
           "slightly favorable, noted)."]
    with open(os.path.join(config.REPORTS_DIR, "survivorship_corrected.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\ncoverage {coverage*100:.1f}% | EW PIT 15Y Sharpe "
          f"{ew15.get('sharpe', float('nan')):.2f} | survivors {len(surv)}/{len(out)}")


def run_window_weights(pos, ctx, win, weights_flag):
    sl = lambda df: df.loc[win["start"]:win["end"]]
    res = be.run_backtest(sl(pos), sl(ctx["open"]), sl(ctx["close"]),
                          benchmark=ctx["benchmark"].loc[win["start"]:win["end"]],
                          interval="1d", signals_are_weights=weights_flag)
    if res is None:
        return {}
    return {k: res[k] for k in ["sharpe", "cagr", "max_dd", "alpha"]}


if __name__ == "__main__":
    main()
