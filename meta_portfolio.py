"""Phase 5.4: build a meta-portfolio from the top uncorrelated strategies.

Selection: walk the headline leaderboard top-down (bias-flagged variants
excluded), re-generate each variant's 10Y daily return stream, and keep it if
its |correlation| with every already-selected stream is < 0.6.  Stop at 8.

Combination: equal capital across sleeves (rebalanced daily) — no
optimization, no lookahead.  Also reports an inverse-vol (trailing 63d)
variant for reference.
"""
import os
import re

import numpy as np
import pandas as pd

import config
import backtest_engine as be
from run_all import CATEGORY_MODULES, get_ctx
from utils import data as dutil

RET_CACHE = os.path.join(config.RESULTS_DIR, "returns_cache")
os.makedirs(RET_CACHE, exist_ok=True)

WINDOW = config.TIMEFRAMES["10Y"]
MAX_SLEEVES = 8
CORR_LIMIT = 0.6
TOP_POOL = 120
MAX_PER_FAMILY = 1


def slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s)[:60]


def registry():
    import importlib
    reg = {}
    for cat, mod_name in CATEGORY_MODULES.items():
        try:
            mod = importlib.import_module(mod_name)
            for meta, fn in mod.get_strategies():
                reg[(meta["id"], meta["variant"])] = (meta, fn)
        except Exception as e:  # noqa: BLE001
            print(f"registry skip {mod_name}: {e}")
    return reg


def returns_for(meta, fn):
    path = os.path.join(RET_CACHE, f"{meta['id']}_{slug(meta['variant'])}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path).iloc[:, 0]
    ctx = get_ctx(meta.get("universe", "nifty50"), meta.get("interval", "1d"))
    positions = fn(ctx)
    pos_w = dutil.slice_window(positions, WINDOW["start"], WINDOW["end"])
    o = dutil.slice_window(ctx["open"], WINDOW["start"], WINDOW["end"])
    c = dutil.slice_window(ctx["close"], WINDOW["start"], WINDOW["end"])
    b = dutil.slice_window(ctx["benchmark"].to_frame(), WINDOW["start"], WINDOW["end"]).iloc[:, 0]
    res = be.run_backtest(pos_w, o, c, benchmark=b, interval=meta.get("interval", "1d"),
                          allow_short=meta.get("allow_short", False),
                          signals_are_weights=meta.get("signals_are_weights", False))
    if res is None:
        return None
    r = res["_returns"]
    r.to_frame("ret").to_parquet(path)
    return r


def metrics(r, bench=None, ann=252):
    eq = (1 + r).cumprod()
    yrs = len(r) / ann
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(ann)
    sharpe = (r.mean() * ann - config.RISK_FREE_RATE) / vol if vol > 0 else np.nan
    dn = r[r < 0].std() * np.sqrt(ann)
    sortino = (r.mean() * ann - config.RISK_FREE_RATE) / dn if dn and dn > 0 else np.nan
    peak = eq.cummax()
    mdd = (eq / peak - 1).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    out = {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino,
           "MaxDD": mdd, "Calmar": calmar, "Total": eq.iloc[-1] - 1}
    if bench is not None:
        br = bench.pct_change(fill_method=None).reindex(r.index).fillna(0)
        cov = np.cov(r, br)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
        out["Beta"] = beta
        out["Alpha"] = (r.mean() - beta * br.mean()) * ann
    return out, eq


def main():
    head = pd.read_csv(os.path.join(config.RESULTS_DIR, "headline_leaderboard.csv"))
    cand_path = os.path.join(config.RESULTS_DIR, "champion_candidates.csv")
    if os.path.exists(cand_path):
        # preferred pool: robust + replicated candidates, ranked by mean Sharpe
        cand = pd.read_csv(cand_path).sort_values("mean_sharpe", ascending=False)
        pool = cand.rename(columns={"sid": "strategy_id"}).merge(
            head[["strategy_id", "variant", "rank", "tier", "score"]],
            on=["strategy_id", "variant"], how="left")
        pool["rank"] = pool["rank"].fillna(9999)
        pool["tier"] = pool["tier"].fillna("-")
        pool["score"] = pool["score"].fillna(pool["mean_sharpe"])
    else:
        pool = head[~head.biased]
        pool = pool[~((pool.strategy_id == "O5") & pool.variant.str.contains("illiquid"))]
    pool = pool.head(TOP_POOL)
    reg = registry()

    selected, streams = [], {}
    fam_count = {}
    for corr_limit in (CORR_LIMIT, 0.75):  # relaxed 2nd pass if < 5 sleeves
        for _, row in pool.iterrows():
            key = (row.strategy_id, row.variant)
            label = f"{row.strategy_id} {row['name']} [{row.variant}]"
            if key not in reg or label in streams:
                continue
            if fam_count.get(row.strategy_id, 0) >= MAX_PER_FAMILY:
                continue
            meta, fn = reg[key]
            try:
                r = returns_for(meta, fn)
            except Exception as e:  # noqa: BLE001
                print(f"skip {key}: {e}")
                continue
            if r is None or len(r) < 750 or r.std() == 0:
                continue
            if any(abs(r.corr(s)) >= corr_limit for s in streams.values()):
                continue
            selected.append((label, row))
            streams[label] = r
            fam_count[row.strategy_id] = fam_count.get(row.strategy_id, 0) + 1
            print(f"selected #{len(selected)} (|corr|<{corr_limit}): {label}")
            if len(streams) >= MAX_SLEEVES:
                break
        if len(streams) >= 5:
            break

    R = pd.DataFrame(streams).fillna(0)
    corr_m = R.corr()

    eqw = R.mean(axis=1)
    iv = (1 / R.rolling(63).std()).replace([np.inf, -np.inf], np.nan)
    iv = iv.div(iv.sum(axis=1), axis=0).shift(1)
    ivw = (R * iv).sum(axis=1).fillna(R.mean(axis=1))

    ctx = get_ctx()
    bench = dutil.slice_window(ctx["benchmark"].to_frame(), WINDOW["start"],
                               WINDOW["end"]).iloc[:, 0]
    bh = bench.pct_change(fill_method=None).reindex(R.index).fillna(0)

    rows = {}
    rows["Meta (equal weight)"], eq_meta = metrics(eqw, bench)
    rows["Meta (inverse vol)"], eq_iv = metrics(ivw, bench)
    rows["Nifty 50 buy & hold"], eq_bh = metrics(bh, bench)
    for label in streams:
        rows[f"Sleeve: {label}"], _ = metrics(streams[label], bench)

    # ---- report ----
    md = ["# Meta-Portfolio (Top Uncorrelated Strategies)", "",
          f"*10Y window {WINDOW['start']} → {WINDOW['end']} | {len(streams)} sleeves, "
          f"pairwise |corr| < {CORR_LIMIT} (relaxed to 0.75 beyond the first pass), "
          "equal capital daily*", "",
          "## Sleeves", ""]
    for i, (label, row) in enumerate(selected, 1):
        md.append(f"{i}. **{label}** — headline rank {int(row['rank'])}, tier {row.tier}, "
                  f"score {row.score:.2f}")
    md += ["", "## Correlation matrix", "", "| |" + "|".join(
        f"S{i+1}" for i in range(len(corr_m))) + "|",
        "|---|" + "---|" * len(corr_m)]
    for i, (name, r_) in enumerate(corr_m.iterrows()):
        md.append(f"| S{i+1} |" + "|".join(f"{v:.2f}" for v in r_) + "|")
    md += ["", "## Performance", "",
           "| portfolio | CAGR | Vol | Sharpe | Sortino | MaxDD | Calmar | Beta | Alpha |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, m in rows.items():
        md.append(f"| {name} | {m['CAGR']*100:.1f}% | {m['Vol']*100:.1f}% | "
                  f"{m['Sharpe']:.2f} | {m['Sortino']:.2f} | {m['MaxDD']*100:.1f}% | "
                  f"{m.get('Calmar', np.nan):.2f} | {m.get('Beta', np.nan):.2f} | "
                  f"{m.get('Alpha', np.nan)*100:.1f}% |")
    with open(os.path.join(config.REPORTS_DIR, "meta_portfolio.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(md))

    # ---- plot ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    eq_meta.plot(ax=ax, label="Meta (equal weight)", lw=1.6)
    eq_iv.plot(ax=ax, label="Meta (inverse vol)", lw=1.2)
    eq_bh.plot(ax=ax, label="Nifty 50 B&H", lw=1.2, alpha=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("Growth of Rs 1 (log)")
    ax.legend()
    ax.set_title("Meta-portfolio vs Nifty 50 — 10Y")
    fig.tight_layout()
    fig.savefig(os.path.join(config.PLOTS_DIR, "meta_portfolio.png"), dpi=110)

    # persist streams for deep-dive regime analysis
    R.to_parquet(os.path.join(config.RESULTS_DIR, "meta_sleeve_returns.parquet"))
    print("\nMeta portfolio written. Equal-weight metrics:")
    for k, v in rows["Meta (equal weight)"].items():
        print(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
