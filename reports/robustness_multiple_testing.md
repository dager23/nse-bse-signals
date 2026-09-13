# Robustness & Multiple-Testing Analysis

**1501 variants** ran through the full multi-timeframe engine; **106,572 additional configurations** were evaluated by the wave-4 mass screener — **108,073 techniques tested in total**. Mining this many configurations guarantees impressive-looking winners by chance alone.

## The mining bar

Expected MAX Sharpe of 108,073 pure-noise strategies: **~1.24** over 15Y, **~1.52** over 10Y (sqrt(2 ln N / T) approximation).

Any single variant's Sharpe below these bars is *indistinguishable from luck* once selection is accounted for. Confidence should come from (a) consistency across independent windows, (b) economic rationale, (c) whole *families* working rather than lone parameter picks.

## Robust set — Sharpe > 0.5 in 5Y, 10Y AND 15Y + positive 10Y alpha (723 of 1445 eligible variants)

| strategy | variant | 5Y | 10Y | 15Y | 15Y CAGR | 15Y MaxDD | 10Y alpha |
|---|---|---|---|---|---|---|---|
| O5 Amihud Illiquidity | illiquid lb63 | 1.75 | 1.28 | 1.47 | 36.6% | -38.9% | 18.8% |
| O5 Amihud Illiquidity | illiquid lb126 | 1.77 | 1.25 | 1.42 | 35.2% | -37.9% | 18.3% |
| R-S Promoted:S | S|lowbeta*0.50+mom121*0.30+rev1m*0.20|top10 | 1.45 | 0.97 | 1.23 | 26.5% | -30.7% | 12.5% |
| R-S Promoted:S | S|liq*0.06+lowbeta*0.47+mom121*0.26+rev1m*0.21|top10 | 1.64 | 1.00 | 1.23 | 26.4% | -29.3% | 12.8% |
| R-S Promoted:S | S|lowbeta*0.70+mom121*0.30|top15 | 1.41 | 0.88 | 1.21 | 24.9% | -29.0% | 10.3% |
| R-S Promoted:S | S|lowbeta*0.50+mom121*0.30+rev1m*0.20|top15 | 1.58 | 0.95 | 1.20 | 25.1% | -30.2% | 11.1% |
| R-S Promoted:S | S|lowbeta*0.37+mom121*0.05+mom6*0.07+rev1m*0.17+smom*0.34|top10 | 1.29 | 0.89 | 1.20 | 26.8% | -30.7% | 11.2% |
| R-S Promoted:S | S|lowbeta*0.50+mom121*0.50|top15 | 1.50 | 0.97 | 1.19 | 25.5% | -29.0% | 11.6% |
| R-S Promoted:S | S|lowbeta*0.50+rev1m*0.50|top5 | 1.64 | 1.09 | 1.19 | 28.2% | -28.6% | 16.2% |
| R-S Promoted:S | S|lowbeta*0.60+mom121*0.40|top15 | 1.34 | 0.88 | 1.19 | 24.8% | -29.0% | 10.2% |
| R-S Promoted:S | S|lowbeta*0.60+mom121*0.20+rev1m*0.20|top15 | 1.39 | 0.87 | 1.18 | 24.2% | -28.4% | 10.1% |
| R-S Promoted:S | S|hi52*0.22+lowbeta*0.32+lowvol*0.06+mom121*0.18+mom6*0.09+rev1m*0.13|top10 | 1.13 | 0.83 | 1.18 | 25.6% | -27.8% | 10.3% |
| R-S Promoted:S | S|liq*0.14+lowbeta*0.62+mom121*0.24|top15 | 1.33 | 0.84 | 1.18 | 24.2% | -29.0% | 9.7% |
| R-S Promoted:S | S|lowbeta*0.30+mom121*0.70|top5 | 1.17 | 0.81 | 1.17 | 30.7% | -37.9% | 12.5% |
| R-S Promoted:S | S|lowbeta*0.49+mom121*0.38+mom6*0.06+rev1m*0.07|top10 | 1.43 | 0.85 | 1.17 | 26.1% | -30.2% | 10.8% |
| R-S Promoted:S | S|lowbeta*0.60+rev1m*0.40|top5 | 1.52 | 0.93 | 1.17 | 26.9% | -27.2% | 13.5% |
| R-S Promoted:S | S|lowbeta*0.56+lowvol*0.06+mom121*0.12+onight*0.09+rev1m*0.07+smom*0.11|top15 | 1.23 | 0.85 | 1.16 | 23.8% | -28.0% | 9.8% |
| R-S Promoted:S | S|lowbeta*0.50+mom121*0.50|top10 | 1.37 | 0.88 | 1.16 | 25.9% | -30.2% | 11.3% |
| R-S Promoted:S | S|lowbeta*0.44+mom121*0.24+onight*0.10+smom*0.22|top5 | 1.28 | 0.87 | 1.16 | 28.9% | -23.8% | 13.9% |
| R-S Promoted:S | S|lowbeta*0.70+rev1m*0.30|top5 | 1.28 | 0.91 | 1.15 | 26.3% | -23.3% | 13.5% |
| R-S Promoted:S | S|hi52*0.10+lowbeta*0.52+lowvol*0.08+mom121*0.07+rev1m*0.23|top15 | 1.38 | 0.92 | 1.15 | 23.2% | -27.1% | 10.6% |
| R-S Promoted:S | S|hi52*0.06+lowbeta*0.34+mom121*0.30+mom6*0.06+rev1m*0.12+smom*0.13|top5 | 1.45 | 0.85 | 1.15 | 28.9% | -35.8% | 13.1% |
| R-S Promoted:S | S|lowbeta*0.44+lowvol*0.07+mom121*0.29+rev1m*0.11+smom*0.09|top15 | 1.37 | 0.86 | 1.15 | 24.1% | -29.9% | 9.9% |
| R-S Promoted:S | S|lowbeta*0.23+lowvol*0.19+mom121*0.50+onight*0.08|top10 | 1.20 | 0.81 | 1.14 | 26.1% | -29.0% | 9.8% |
| R-S Promoted:S | S|hi52*0.06+lowbeta*0.27+lowvol*0.18+mom121*0.27+mom3*0.09+rev1m*0.14|top10 | 1.44 | 0.93 | 1.14 | 24.5% | -27.1% | 11.5% |
| R-S Promoted:S | S|lowbeta*0.40+mom121*0.60|top5 | 1.16 | 0.79 | 1.14 | 29.2% | -30.2% | 12.1% |
| R-S Promoted:S | S|lowbeta*0.50+mom6*0.30+smom*0.20|top10 | 1.19 | 0.85 | 1.14 | 25.6% | -30.2% | 10.7% |
| R-S Promoted:S | S|lowbeta*0.32+lowvol*0.20+mom121*0.26+rev1m*0.21|top15 | 1.40 | 0.90 | 1.14 | 23.6% | -27.7% | 10.2% |
| R-S Promoted:S | S|hi52*0.06+lowbeta*0.43+lowvol*0.16+mom121*0.18+onight*0.08+rev1m*0.09|top10 | 1.33 | 0.80 | 1.14 | 23.9% | -26.4% | 9.8% |
| R-S Promoted:S | S|hi52*0.07+lowbeta*0.35+mom121*0.33+rev1m*0.18+smom*0.07|top15 | 1.34 | 0.90 | 1.14 | 24.9% | -30.1% | 10.5% |
| R-S Promoted:S | S|lowbeta*0.33+mom121*0.33+mom6*0.33|top5 | 1.34 | 0.90 | 1.14 | 29.5% | -30.1% | 14.3% |
| R-S Promoted:S | S|liq*0.17+lowbeta*0.59+mom121*0.24|top15 | 1.32 | 0.83 | 1.14 | 23.6% | -29.1% | 9.3% |
| R-S Promoted:S | S|lowbeta*0.52+lowvol*0.07+mom121*0.13+onight*0.07+rev1m*0.10+smom*0.11|top15 | 1.24 | 0.85 | 1.14 | 23.4% | -29.4% | 9.8% |
| R-S Promoted:S | S|lowbeta*0.67+mom121*0.06+mom6*0.06+onight*0.12+rev1m*0.08|top15 | 1.34 | 0.78 | 1.13 | 23.2% | -27.5% | 9.0% |
| R-S Promoted:S | S|lowbeta*0.68+mom3*0.10+rev1m*0.22|top15 | 1.37 | 0.83 | 1.13 | 23.0% | -27.1% | 9.5% |
| R-S Promoted:S | S|hi52*0.10+lowbeta*0.52+mom121*0.11+mom6*0.15+rev1m*0.11|top15 | 1.30 | 0.89 | 1.13 | 23.6% | -29.2% | 10.4% |
| R-S Promoted:S | S|lowbeta*0.50+rev1m*0.30+smom*0.20|top15 | 1.32 | 0.85 | 1.13 | 23.4% | -28.2% | 9.6% |
| R-S Promoted:S | S|lowbeta*0.60+rev1m*0.20+smom*0.20|top15 | 1.27 | 0.83 | 1.13 | 23.0% | -26.3% | 9.5% |
| R-S Promoted:S | S|lowbeta*0.46+lowvol*0.13+mom121*0.10+mom6*0.06+onight*0.09+rev1m*0.17|top15 | 1.32 | 0.86 | 1.13 | 23.1% | -28.1% | 9.8% |
| R-S Promoted:S | S|lowbeta*0.59+lowvol*0.11+mom121*0.22+smom*0.08|top15 | 1.21 | 0.80 | 1.13 | 23.3% | -28.6% | 9.2% |

**Exceptional set** (clears the mining bar in both long windows): 0 variants:


## Parameter-fragile families (score spread > 0.25 across variants)

| family | n variants | best | worst | spread |
|---|---|---|---|---|
| M13 Fixed-Hold Sweep | 12 | 0.71 | 0.26 | 0.44 |
| M6 ROC Grid | 18 | 0.74 | 0.31 | 0.44 |
| M1 MA Cross Grid | 40 | 0.76 | 0.35 | 0.41 |
| M3 RSI Zone Grid | 15 | 0.77 | 0.38 | 0.39 |
| M4 Donchian Grid | 16 | 0.72 | 0.36 | 0.36 |
| N3 Compose:rsi14lo | 36 | 0.77 | 0.43 | 0.34 |
| N5 Compose:bbtouch | 36 | 0.79 | 0.46 | 0.33 |
| M2 Price vs MA Grid | 12 | 0.67 | 0.37 | 0.30 |
| Q11 Overlay:lowvol | 9 | 0.79 | 0.50 | 0.29 |
| O15 NR Breakout | 4 | 0.58 | 0.30 | 0.28 |
| N2 Compose:donch20 | 36 | 0.74 | 0.47 | 0.28 |
| N1 Compose:emaX | 36 | 0.72 | 0.45 | 0.27 |
| O4 Efficiency Ratio | 6 | 0.64 | 0.37 | 0.27 |

## Bias-suspect standouts

- **O5 Amihud illiquid tilt** (15Y Sharpe ~1.4-1.5) is the single strongest result in the zoo — and the most suspect. Overweighting today's *least-liquid* Nifty 50 names is precisely where survivorship bias concentrates: those are the stocks that grew into the index. Excluded from the meta-portfolio; treat as an artifact until tested on point-in-time constituents.

## Hand-built vs generated

| wave | n | mean score | best score | mean Sharpe |
|---|---|---|---|---|
| generated (M-Q) | 904 | 0.62 | 0.89 | 0.36 |
| hand-built (A-L) | 596 | 0.71 | 0.86 | 0.66 |