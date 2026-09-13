# Replication Probe (Wave 3)

*Best robust variant of each top family, re-run on 10Y windows it was not selected on: weekly bars (Nifty 50) and the daily midcap universe. Sharpe > 0.4 in a domain = replicates there.*

| family | variant | daily 10Y | weekly 10Y | midcap 10Y | replicates |
|---|---|---|---|---|---|
| R-S Promoted:S | S|lowbeta*0.50+mom121*0.30+rev1m*0.20|top10 | 0.97 | 0.57 | 1.25 | ✓ 2/2 |
| R-W Promoted:W | W|vspike&insideday|vix_low|rsi70 | 1.03 | 0.47 | 0.46 | ✓ 2/2 |
| R-U Promoted:U | U|insideday&vspike|none|rsi70 | 0.92 | 0.50 | 0.48 | ✓ 2/2 |
| O12 Low-Beta + Momentum | rank blend | 0.81 | 0.68 | 1.06 | ✓ 2/2 |
| I12 Min Variance | shrink.3 cap10% | 0.80 | 0.73 | 1.17 | ✓ 2/2 |
| Q9 Overlay:jt121 | jt121|crash8 | 0.92 | 0.75 | 1.02 | ✓ 2/2 |
| R-T Promoted:T | T|rsi5|<15|>80|none|state | 0.79 | 0.54 | 0.48 | ✓ 2/2 |
| G9 RMT Min-Variance | MP-cleaned | 0.73 | 0.73 | 1.14 | ✓ 2/2 |
| N5 Compose:bbtouch | bbtouch|quiet|rsi70 | 0.88 | 0.56 | 0.53 | ✓ 2/2 |
| R-V Promoted:V | V|sma150>ema200|none|inv | 0.87 | 0.75 | 0.79 | ✓ 2/2 |
| M12 XS Momentum Grid | lb252top15 | 0.80 | 0.73 | 0.85 | ✓ 2/2 |
| N4 Compose:rsi2lo | rsi2lo|mktup|rsi70 | 0.83 | 0.66 | 0.80 | ✓ 2/2 |
| P2 Pair:lowvol+rev | lowvol.5/rev.5 top5 | 0.83 | 0.64 | 0.89 | ✓ 2/2 |
| I1 FF Size+Value (price proxy) | small+3yrev | 0.84 | -0.17 | 1.02 | ◐ 1/2 |
| C7 Jegadeesh-Titman 12-1 | top10 | 0.73 | 0.75 | 0.84 | ✓ 2/2 |
| O8 Tail Ratio | high 126d | 0.76 | 0.88 | 0.96 | ✓ 2/2 |
| P1 Factor:smom | smom top15 | 0.78 | 0.75 | 0.85 | ✓ 2/2 |
| C8 Dual Momentum | top10/abs | 0.73 | 0.65 | 0.73 | ✓ 2/2 |
| I10 Equal Weight Monthly | EW rebal | 0.80 | 0.79 | 0.93 | ✓ 2/2 |
| I11 Risk Parity | inv-vol | 0.78 | 0.76 | 0.94 | ✓ 2/2 |
| O17 Volume-Weighted Momentum | lb63 | 0.72 | 0.88 | 0.99 | ✓ 2/2 |
| N7 Compose:roc21 | roc21|quiet|rsi70 | 0.83 | 0.63 | 0.76 | ✓ 2/2 |
| O7 Skewness Factor | pos lb126 | 0.66 | 0.55 | 0.94 | ✓ 2/2 |
| O1 Residual Momentum | lb252 top10 | 0.70 | 0.41 | 0.78 | ✓ 2/2 |
| O2 Overnight Strength | overnight lb126 | 0.84 | 0.56 | 0.73 | ✓ 2/2 |
| N6 Compose:macdX | macdX|breadth|rsi70 | 0.80 | 0.58 | 0.58 | ✓ 2/2 |
| N10 Compose:gapdn | gapdn|adx20|rsi70 | 0.83 | 0.80 | 0.85 | ✓ 2/2 |
| N3 Compose:rsi14lo | rsi14lo|breadth|rsi70 | 0.78 | 0.65 | 0.42 | ✓ 2/2 |
| Q12 Overlay:sharpemom | sharpemom|crash8 | 0.79 | 0.81 | 0.94 | ✓ 2/2 |
| P4 IC-Weighted Composite | IC36m top10 | 0.80 | 0.86 | 0.74 | ✓ 2/2 |
| I13 Max Sharpe | mom mu/shrink.3 | 0.69 | 0.78 | 0.85 | ✓ 2/2 |
| M1 MA Cross Grid | ema100/200 | 0.72 | 0.73 | 0.91 | ✓ 2/2 |

**Summary:** of 32 families, 31 replicate in both domains, 1 in one, 0 in neither. Families that replicate everywhere are the study's most defensible findings.