# Executive Summary — NSE Strategy Research Program (Waves 1–4)

*Generated 2026-07-14. **107,677 techniques tested in total**: 1,500 variants
through the full 5-timeframe engine (categories A–R) + 106,572 configurations
through the vectorized mass screener (families V/T/U/S/W — every one logged
with its score). Universe: Nifty 50 (+ weekly-bar and midcap replication
domains). Costs 0.10% + 0.05% slippage per side, next-open execution,
walk-forward fitting, no lookahead in any signal.*

## The champion algorithm

**`R-S: rank-blend of 0.50·low-beta + 0.30·momentum(12-1) + 0.20·1-month
reversal, hold top 10, monthly rebalance`**

| view | Sharpe |
|---|---|
| Daily Nifty 50, 10Y | 0.97 |
| Daily Nifty 50, 15Y | 1.23 (CAGR 26.5%, MaxDD −30.7%) |
| Weekly bars (not selected on) | 0.57 |
| Midcap universe (not selected on) | **1.25** |

It won on every filter: bias-clean, Sharpe > 0.5 in all full-engine windows,
and it *replicates* — strongest of all 31 surviving candidates (mean Sharpe
1.00 across the four views). Economically it is betting-against-beta +
momentum + short-term reversal: three of the best-documented cross-sectional
anomalies, found independently by the mass screen. Runner-ups:
min-variance (I12, 0.93), momentum+crash-overlay (Q9, 0.92), low-beta+momentum
(O12, 0.90) — see [champion.md](champion.md).

## The recommended deployable: meta-portfolio

Five decorrelated sleeves drawn ONLY from the robust+replicated candidate
pool (champion blend, volume-spike/inside-day composition, JT momentum with
crash de-risk, volume-weighted momentum, deep-oversold RSI reversion):

| | CAGR (10Y) | Sharpe | MaxDD | Beta | Alpha |
|---|---|---|---|---|---|
| **Meta (inverse-vol)** | **24.0%** | **1.11** | **−29.1%** | 0.75 | **13.1%/yr** |
| Meta (equal weight) | 24.0% | 1.05 | −31.5% | 0.83 | 12.2%/yr |
| Nifty 50 buy & hold | 11.9% | 0.37 | −38.4% | 1.00 | — |

## What 107k+ tests actually taught us

1. **The mining bar is now 1.24 (15Y) / 1.52 (10Y)** — the expected maximum
   Sharpe of 107,677 noise strategies. **Zero variants clear it.** At this
   search intensity, *no* single backtest number is self-evidently skill; only
   replication across domains separates signal from selection. 723 variants
   are "robust" (Sharpe > 0.5 in 5Y/10Y/15Y); 31 also replicate on weekly bars
   and midcaps. Those 31 are the research output.
2. **Factor portfolios beat trading rules — again, and at scale.** In the 106k
   screen, the S family (factor blends) produced the entire top decile;
   low-beta was the single most valuable ingredient (present in 9 of the top
   10 blends). Composed daily entry/exit rules (W/U/T families) screened well
   but replicated weakly — most of their edge was selection.
3. **Structure that survived everywhere**: monthly cross-sectional ranking,
   crash de-risk overlays, RSI-70-style profit-taking exits, and trend gates
   used as *de-risking* rather than entry filters.
4. **Structure that failed everywhere**: daily ML stock-picking, seasonality
   stand-alone, candlestick patterns, tight time-stops, most entry gates.

## Wave 5: survivorship-corrected re-test + live forward test

The point-in-time universe (membership reconstructed from public
reconstitution logs, 35 ex-constituents' price histories recovered, ~5
irrecoverable) finally *measured* the survivorship premium: equal-weight
buy-and-hold falls from Sharpe 0.88 to **0.45** on the true index.
**Only 8 of the 31 candidates survive** (Sharpe > 0.5 in both PIT windows) —
momentum-heavy strategies decayed most (JT 12-1: 0.90 → 0.32), which is
exactly what survivorship theory predicts. The champion **held up best among
factor strategies: 1.23 → 0.71**, its decay matching the universe-wide
premium — the *relative* edge is real. The single most bias-resistant
strategy: **P2 low-vol + 1-month-reversal (top 5)** at PIT Sharpe **0.79/0.75**
with only −0.12 decay. Details: [survivorship_corrected.md](survivorship_corrected.md).

**The forward test is live**: `signals_live.py` recomputes the champion and
all 5 meta sleeves daily (scheduled weekdays 18:30 IST), logs target holdings,
and marks the prior day's book to market into `forward_test/performance.csv` —
a genuinely out-of-sample track record is now accumulating.

## Honest caveats (unchanged and non-negotiable)

- **Survivorship bias inflates all absolute numbers** (today's constituents
  backfilled 15Y; equal-weight buy-and-hold "earned" 21% CAGR). Relative
  results, family-level conclusions, and replication are the trustworthy
  output. Point-in-time constituent data is the single most valuable upgrade.
- Promoted (R-family) full-engine metrics are selection-biased by
  construction — their weekly/midcap replication is the meaningful evidence.
- The Amihud illiquidity result (former #1) is flagged as a survivorship
  artifact and excluded from all recommendations.
- Yahoo daily data, estimated costs, no intraday fills, no taxes.

## Deliverables

[master_leaderboard.md](master_leaderboard.md) (1,500 full-engine variants) ·
[champion.md](champion.md) · [robustness_multiple_testing.md](robustness_multiple_testing.md) ·
[replication.md](replication.md) · [meta_portfolio.md](meta_portfolio.md) ·
[tier_analysis.md](tier_analysis.md) · [category_analysis.md](category_analysis.md) ·
[top_10_deep_dive.md](top_10_deep_dive.md) · screen scores in `results/screen/`
(106,572 rows) · 18 strategy modules + screener + full pipeline, reproducible
end-to-end: `screener.py → promote.py → run_all.py → leaderboard.py →
robustness.py → replication.py → champion.py → meta_portfolio.py`.
