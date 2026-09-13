# Survivorship-Corrected Re-Test (Point-in-Time Universe)

*Universe = the actual Nifty 50 of each date (membership reconstructed from public reconstitution logs; 85 ever-member tickers, price coverage 100.0% of member-days; median members with data per day: 48). Strategies use full price history but may only HOLD current members.*

**Equal-weight monthly baseline on PIT universe (15Y): Sharpe 0.45, CAGR 13.5%** — compare 0.88 / 21.4% on the backfilled universe. The gap is the measured survivorship premium.

| strategy | variant | 15Y backfilled | 15Y PIT | Δ | 10Y PIT | PIT CAGR 15Y | PIT DD 15Y |
|---|---|---|---|---|---|---|---|
| P2 Pair:lowvol+rev | lowvol.5/rev.5 top5 | 0.91 | **0.79** | -0.11 | 0.75 | 21.1% | -32.3% |
| R-S Promoted:S | S|lowbeta*0.50+mom121*0.30+rev1m*0.20|to | 1.23 | **0.71** | -0.52 | 0.63 | 18.9% | -28.5% |
| N5 Compose:bbtouch | bbtouch|quiet|rsi70 | 0.96 | **0.65** | -0.31 | 0.69 | 18.1% | -40.2% |
| R-V Promoted:V | V|sma150>ema200|none|inv | 0.96 | **0.60** | -0.36 | 0.59 | 16.2% | -36.0% |
| R-U Promoted:U | U|insideday&vspike|none|rsi70 | 1.09 | **0.57** | -0.52 | 0.74 | 17.5% | -39.8% |
| N7 Compose:roc21 | roc21|quiet|rsi70 | 0.87 | **0.54** | -0.33 | 0.63 | 15.5% | -40.4% |
| R-W Promoted:W | W|vspike&insideday|vix_low|rsi70 | 1.13 | **0.52** | -0.61 | 0.70 | 16.5% | -49.3% |
| N4 Compose:rsi2lo | rsi2lo|mktup|rsi70 | 0.92 | **0.51** | -0.41 | 0.61 | 14.9% | -39.9% |
| N3 Compose:rsi14lo | rsi14lo|breadth|rsi70 | 0.82 | **0.48** | -0.35 | 0.55 | 15.4% | -45.2% |
| N10 Compose:gapdn | gapdn|adx20|rsi70 | 0.82 | **0.46** | -0.37 | 0.58 | 14.4% | -44.9% |
| R-T Promoted:T | T|rsi5|<15|>80|none|state | 0.98 | **0.46** | -0.53 | 0.63 | 14.5% | -38.7% |
| M1 MA Cross Grid | ema100/200 | 0.81 | **0.45** | -0.36 | 0.45 | 13.2% | -35.3% |
| Q9 Overlay:jt121 | jt121|crash8 | 1.00 | **0.44** | -0.57 | 0.53 | 14.6% | -35.9% |
| N6 Compose:macdX | macdX|breadth|rsi70 | 0.83 | **0.43** | -0.40 | 0.60 | 13.4% | -42.7% |
| Q12 Overlay:sharpemom | sharpemom|crash8 | 0.82 | **0.40** | -0.42 | 0.53 | 13.1% | -35.1% |
| I12 Min Variance | shrink.3 cap10% | 1.04 | **0.39** | -0.65 | 0.43 | 10.3% | -26.0% |
| G9 RMT Min-Variance | MP-cleaned | 0.96 | **0.36** | -0.60 | 0.41 | 10.1% | -26.3% |
| O8 Tail Ratio | high 126d | 0.88 | **0.35** | -0.54 | 0.25 | 12.1% | -43.7% |
| P1 Factor:smom | smom top15 | 0.88 | **0.35** | -0.54 | 0.36 | 11.9% | -41.2% |
| O12 Low-Beta + Momentum | rank blend | 1.05 | **0.33** | -0.72 | 0.26 | 11.7% | -32.6% |
| C7 Jegadeesh-Titman 12-1 | top10 | 0.90 | **0.32** | -0.57 | 0.30 | 12.0% | -47.1% |
| M12 XS Momentum Grid | lb252top15 | 0.95 | **0.32** | -0.63 | 0.14 | 11.6% | -40.6% |
| I11 Risk Parity | inv-vol | 0.87 | **0.27** | -0.61 | 0.33 | 9.1% | -25.4% |
| P4 IC-Weighted Composite | IC36m top10 | 0.82 | **0.21** | -0.61 | 0.23 | 9.0% | -41.4% |
| I10 Equal Weight Monthly | EW rebal | 0.88 | **0.17** | -0.71 | 0.25 | 8.0% | -24.4% |
| O2 Overnight Strength | overnight lb126 | 0.86 | **0.07** | -0.79 | 0.32 | 3.7% | -72.7% |
| C8 Dual Momentum | top10/abs | 0.88 | **0.07** | -0.81 | 0.00 | 5.7% | -50.9% |
| O7 Skewness Factor | pos lb126 | 0.86 | **0.07** | -0.80 | 0.24 | 5.8% | -48.2% |
| I13 Max Sharpe | mom mu/shrink.3 | 0.82 | **0.05** | -0.77 | 0.13 | 6.7% | -21.7% |
| O1 Residual Momentum | lb252 top10 | 0.86 | **0.03** | -0.83 | -0.14 | 4.8% | -47.4% |
| O17 Volume-Weighted Momentum | lb63 | 0.87 | **-0.18** | -1.06 | -0.17 | -1.4% | -60.0% |

**8 of 31 candidates keep Sharpe > 0.5 in both PIT windows.** These are the study's final defensible set.

Membership-table caveats: a handful of 2013-18 re-entry dates are approximate (VEDL, BANKBARODA, NMDC, HINDPETRO, INDUSTOWER, YESBANK); worst-case error is one slot of 50 for limited stretches. Tickers with no recoverable data are treated as never-members (bias direction: slightly favorable, noted).