# Meta-Portfolio (Top Uncorrelated Strategies)

*10Y window 2015-07-01 → 2025-07-01 | 5 sleeves, pairwise |corr| < 0.6 (relaxed to 0.75 beyond the first pass), equal capital daily*

## Sleeves

1. **R-S Promoted:S [S|lowbeta*0.50+mom121*0.30+rev1m*0.20|top10]** — headline rank 69, tier S, score 0.82
2. **R-W Promoted:W [W|vspike&insideday|vix_low|rsi70]** — headline rank 35, tier S, score 0.83
3. **Q9 Overlay:jt121 [jt121|crash8]** — headline rank 176, tier A, score 0.80
4. **O17 Volume-Weighted Momentum [lb63]** — headline rank 444, tier B, score 0.74
5. **R-T Promoted:T [T|rsi5|<15|>80|none|state]** — headline rank 345, tier B, score 0.76

## Correlation matrix

| |S1|S2|S3|S4|S5|
|---|---|---|---|---|---|
| S1 |1.00|0.59|0.63|0.73|0.71|
| S2 |0.59|1.00|0.46|0.58|0.66|
| S3 |0.63|0.46|1.00|0.73|0.61|
| S4 |0.73|0.58|0.73|1.00|0.73|
| S5 |0.71|0.66|0.61|0.73|1.00|

## Performance

| portfolio | CAGR | Vol | Sharpe | Sortino | MaxDD | Calmar | Beta | Alpha |
|---|---|---|---|---|---|---|---|---|
| Meta (equal weight) | 24.0% | 15.5% | 1.05 | 1.32 | -31.5% | 0.76 | 0.83 | 12.2% |
| Meta (inverse vol) | 24.0% | 14.6% | 1.11 | 1.42 | -29.1% | 0.83 | 0.75 | 13.1% |
| Nifty 50 buy & hold | 11.9% | 16.5% | 0.37 | 0.45 | -38.4% | 0.31 | 1.00 | 0.0% |
| Sleeve: R-S Promoted:S [S|lowbeta*0.50+mom121*0.30+rev1m*0.20|top10] | 22.3% | 15.3% | 0.97 | 1.26 | -30.7% | 0.73 | 0.70 | 12.5% |
| Sleeve: R-W Promoted:W [W|vspike&insideday|vix_low|rsi70] | 28.7% | 20.2% | 1.03 | 1.41 | -42.6% | 0.67 | 0.81 | 17.0% |
| Sleeve: Q9 Overlay:jt121 [jt121|crash8] | 24.1% | 18.3% | 0.92 | 1.19 | -25.0% | 0.97 | 0.74 | 14.0% |
| Sleeve: O17 Volume-Weighted Momentum [lb63] | 20.2% | 19.2% | 0.72 | 0.91 | -36.5% | 0.55 | 0.94 | 8.4% |
| Sleeve: R-T Promoted:T [T|rsi5|<15|>80|none|state] | 21.7% | 18.9% | 0.79 | 1.02 | -41.6% | 0.52 | 0.97 | 9.2% |