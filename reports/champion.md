# Champion Algorithm Selection

Candidates must be bias-clean, robust in every full-engine window (5Y/10Y/15Y Sharpe > 0.5), and replicate on weekly bars AND midcaps (Sharpe > 0.4). Ranked by mean Sharpe across the four independent views.

**31 candidates survived all filters.**

## The champion

### R-S Promoted:S `[S|lowbeta*0.50+mom121*0.30+rev1m*0.20|top10]`

- Mean Sharpe across views: **1.00**
- 15Y: CAGR 26.5%, Sharpe 1.23, MaxDD -30.7%
- 10Y Sharpe 0.97 | weekly 0.57 | midcap 1.25

## Top 15 candidates

| rank | strategy | variant | 10Y | 15Y | weekly | midcap | mean | 15Y CAGR | 15Y DD |
|---|---|---|---|---|---|---|---|---|---|
| 1 | R-S Promoted:S | S|lowbeta*0.50+mom121*0.30+rev1m*0.20|top10 | 0.97 | 1.23 | 0.57 | 1.25 | **1.00** | 26.5% | -30.7% |
| 2 | I12 Min Variance | shrink.3 cap10% | 0.80 | 1.04 | 0.73 | 1.17 | **0.93** | 21.3% | -30.1% |
| 3 | Q9 Overlay:jt121 | jt121|crash8 | 0.92 | 1.00 | 0.75 | 1.02 | **0.92** | 25.4% | -26.2% |
| 4 | O12 Low-Beta + Momentum | rank blend | 0.81 | 1.05 | 0.68 | 1.06 | **0.90** | 24.0% | -30.2% |
| 5 | G9 RMT Min-Variance | MP-cleaned | 0.73 | 0.96 | 0.73 | 1.14 | **0.89** | 20.2% | -30.4% |
| 6 | O8 Tail Ratio | high 126d | 0.76 | 0.88 | 0.88 | 0.96 | **0.87** | 23.0% | -34.2% |
| 7 | O17 Volume-Weighted Momentum | lb63 | 0.72 | 0.87 | 0.88 | 0.99 | **0.86** | 23.4% | -36.5% |
| 8 | I10 Equal Weight Monthly | EW rebal | 0.80 | 0.88 | 0.79 | 0.93 | **0.85** | 21.4% | -38.0% |
| 9 | R-V Promoted:V | V|sma150>ema200|none|inv | 0.87 | 0.96 | 0.75 | 0.79 | **0.84** | 22.9% | -36.1% |
| 10 | Q12 Overlay:sharpemom | sharpemom|crash8 | 0.79 | 0.82 | 0.81 | 0.94 | **0.84** | 20.5% | -27.1% |
| 11 | I11 Risk Parity | inv-vol | 0.78 | 0.87 | 0.76 | 0.94 | **0.84** | 20.6% | -36.5% |
| 12 | M12 XS Momentum Grid | lb252top15 | 0.80 | 0.95 | 0.73 | 0.85 | **0.83** | 24.1% | -37.3% |
| 13 | N10 Compose:gapdn | gapdn|adx20|rsi70 | 0.83 | 0.82 | 0.80 | 0.85 | **0.83** | 22.2% | -41.9% |
| 14 | P1 Factor:smom | smom top15 | 0.78 | 0.88 | 0.75 | 0.85 | **0.82** | 22.3% | -37.6% |
| 15 | P2 Pair:lowvol+rev | lowvol.5/rev.5 top5 | 0.83 | 0.91 | 0.64 | 0.89 | **0.82** | 22.1% | -29.9% |

## Champion vs meta-portfolio

The single-strategy champion is the best *individual* rule set, but the **meta-portfolio remains the recommended deployable**: five decorrelated sleeves cannot all be a mining artifact at once, and its drawdown profile is structurally better than any single sleeve's. Use the champion as the core sleeve; use the meta-portfolio as the portfolio.

*(See meta_portfolio.md for the combined result.)*