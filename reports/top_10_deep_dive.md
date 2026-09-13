# Top 10 Deep Dive

*Top 10 bias-free variants by composite score. Regime split uses Nifty vs its 200DMA over the 10Y window; sleeve returns are net of costs.*

## 1. O5 — Amihud Illiquidity `[illiquid lb126]`

Tier **S**, composite score **0.90** (headline timeframe 15Y).

> ⚠️ **Survivorship-bias suspect**: overweighting today's least-liquid index members is where backfill bias concentrates; excluded from the meta-portfolio. See robustness report.

| timeframe | CAGR | Sharpe | Sortino | MaxDD | trades | win rate | exposure |
|---|---|---|---|---|---|---|---|
| 1Y | 17.2% | 0.69 | 1.03 | -15.9% | 17 | 76.5% | 99.6% |
| 3Y | 29.7% | 1.49 | 2.22 | -15.9% | 31 | 77.4% | 99.9% |
| 5Y | 40.5% | 1.77 | 2.40 | -15.9% | 54 | 83.3% | 99.9% |
| 10Y | 32.3% | 1.25 | 1.58 | -37.9% | 99 | 78.8% | 100.0% |
| 15Y | 35.2% | 1.42 | 1.88 | -37.9% | 131 | 80.9% | 98.8% |

**Robustness across timeframes:** Sharpe > 0.5 in 5/5 windows.
**Parameter sensitivity:** 4 variants tested; composite scores range 0.71–0.90 (spread 0.19 — parameter-sensitive).

![equity](../results/plots/strategy_O5_equity.png)

## 2. O5 — Amihud Illiquidity `[illiquid lb63]`

Tier **S**, composite score **0.89** (headline timeframe 15Y).

> ⚠️ **Survivorship-bias suspect**: overweighting today's least-liquid index members is where backfill bias concentrates; excluded from the meta-portfolio. See robustness report.

| timeframe | CAGR | Sharpe | Sortino | MaxDD | trades | win rate | exposure |
|---|---|---|---|---|---|---|---|
| 1Y | 12.8% | 0.42 | 0.65 | -16.1% | 24 | 66.7% | 99.6% |
| 3Y | 27.7% | 1.34 | 1.99 | -16.1% | 54 | 72.2% | 99.9% |
| 5Y | 40.1% | 1.75 | 2.42 | -16.1% | 83 | 77.1% | 99.9% |
| 10Y | 33.1% | 1.28 | 1.60 | -38.9% | 159 | 73.0% | 100.0% |
| 15Y | 36.6% | 1.47 | 1.93 | -38.9% | 205 | 73.2% | 100.0% |

**Robustness across timeframes:** Sharpe > 0.5 in 4/5 windows.
**Parameter sensitivity:** 4 variants tested; composite scores range 0.71–0.90 (spread 0.19 — parameter-sensitive).

![equity](../results/plots/strategy_O5_equity.png)

## 3. I12 — Min Variance `[shrink.3 cap10%]`

Tier **S**, composite score **0.82** (headline timeframe 15Y).

| timeframe | CAGR | Sharpe | Sortino | MaxDD | trades | win rate | exposure |
|---|---|---|---|---|---|---|---|
| 1Y | 7.3% | 0.11 | 0.17 | -17.8% | 51 | 64.7% | 99.6% |
| 3Y | 23.4% | 1.42 | 2.16 | -17.8% | 90 | 84.4% | 99.9% |
| 5Y | 23.4% | 1.24 | 1.73 | -17.8% | 150 | 78.0% | 99.9% |
| 10Y | 18.0% | 0.80 | 1.00 | -30.1% | 251 | 67.7% | 100.0% |
| 15Y | 21.3% | 1.04 | 1.35 | -30.1% | 337 | 70.0% | 100.0% |

**Robustness across timeframes:** Sharpe > 0.5 in 4/5 windows.
**Regime split (10Y):** bull (Nifty>200DMA, 74% of days) ann. return 26.0%; bear ann. return -7.4%.

![equity](../results/plots/strategy_I12_equity.png)

## 4. G9 — RMT Min-Variance `[MP-cleaned]`

Tier **S**, composite score **0.81** (headline timeframe 15Y).

| timeframe | CAGR | Sharpe | Sortino | MaxDD | trades | win rate | exposure |
|---|---|---|---|---|---|---|---|
| 1Y | 7.3% | 0.10 | 0.16 | -17.1% | 47 | 61.7% | 99.6% |
| 3Y | 22.8% | 1.38 | 2.11 | -17.1% | 90 | 76.7% | 99.9% |
| 5Y | 23.1% | 1.23 | 1.71 | -17.1% | 129 | 75.2% | 99.9% |
| 10Y | 16.9% | 0.73 | 0.91 | -30.4% | 208 | 67.3% | 100.0% |
| 15Y | 20.2% | 0.96 | 1.26 | -30.4% | 284 | 69.7% | 100.0% |

**Robustness across timeframes:** Sharpe > 0.5 in 4/5 windows.
**Regime split (10Y):** bull (Nifty>200DMA, 74% of days) ann. return 25.4%; bear ann. return -9.1%.

![equity](../results/plots/strategy_G9_equity.png)

## 5. Q9 — Overlay:jt121 `[jt121|crash8]`

Tier **S**, composite score **0.80** (headline timeframe 15Y).

| timeframe | CAGR | Sharpe | Sortino | MaxDD | trades | win rate | exposure |
|---|---|---|---|---|---|---|---|
| 1Y | -1.4% | -0.36 | -0.53 | -25.0% | 33 | 51.5% | 99.6% |
| 3Y | 27.5% | 1.11 | 1.37 | -25.0% | 85 | 70.6% | 99.9% |
| 5Y | 29.9% | 1.12 | 1.42 | -25.0% | 156 | 62.8% | 99.2% |
| 10Y | 24.1% | 0.92 | 1.19 | -25.0% | 321 | 58.9% | 97.8% |
| 15Y | 25.4% | 1.00 | 1.33 | -26.2% | 509 | 57.8% | 97.5% |

**Robustness across timeframes:** Sharpe > 0.5 in 4/5 windows.
**Parameter sensitivity:** 9 variants tested; composite scores range 0.68–0.80 (spread 0.12 — parameter-sensitive).
**Regime split (10Y):** bull (Nifty>200DMA, 74% of days) ann. return 36.6%; bear ann. return -15.5%.

![equity](../results/plots/strategy_Q9_equity.png)

## 6. O12 — Low-Beta + Momentum `[rank blend]`

Tier **S**, composite score **0.80** (headline timeframe 15Y).

| timeframe | CAGR | Sharpe | Sortino | MaxDD | trades | win rate | exposure |
|---|---|---|---|---|---|---|---|
| 1Y | 4.7% | -0.06 | -0.09 | -20.9% | 45 | 53.3% | 99.6% |
| 3Y | 25.2% | 1.31 | 1.91 | -20.9% | 105 | 57.1% | 99.9% |
| 5Y | 26.3% | 1.20 | 1.68 | -20.9% | 175 | 52.6% | 99.9% |
| 10Y | 20.1% | 0.81 | 1.03 | -30.2% | 315 | 51.1% | 100.0% |
| 15Y | 24.0% | 1.05 | 1.40 | -30.2% | 446 | 53.6% | 100.0% |

**Robustness across timeframes:** Sharpe > 0.5 in 4/5 windows.
**Parameter sensitivity:** 2 variants tested; composite scores range 0.72–0.80 (spread 0.08 — robust).
**Regime split (10Y):** bull (Nifty>200DMA, 74% of days) ann. return 31.4%; bear ann. return -14.8%.

![equity](../results/plots/strategy_O12_equity.png)

## 7. O10 — Drawdown Buying `[dd15%+mkt]`

Tier **S**, composite score **0.79** (headline timeframe 15Y).

| timeframe | CAGR | Sharpe | Sortino | MaxDD | trades | win rate | exposure |
|---|---|---|---|---|---|---|---|
| 1Y | 17.8% | 0.69 | 1.07 | -15.8% | 52 | 80.8% | 99.6% |
| 3Y | 27.7% | 1.23 | 1.79 | -15.8% | 125 | 89.6% | 99.9% |
| 5Y | 35.6% | 1.45 | 2.04 | -15.8% | 234 | 90.2% | 99.9% |
| 10Y | 22.1% | 0.79 | 1.02 | -41.8% | 385 | 85.5% | 100.0% |
| 15Y | 22.5% | 0.81 | 1.11 | -41.8% | 587 | 86.0% | 100.0% |

**Robustness across timeframes:** Sharpe > 0.5 in 5/5 windows.
**Parameter sensitivity:** 6 variants tested; composite scores range 0.68–0.79 (spread 0.11 — robust).
**Regime split (10Y):** bull (Nifty>200DMA, 74% of days) ann. return 32.0%; bear ann. return -7.4%.

![equity](../results/plots/strategy_O10_equity.png)

## 8. Q11 — Overlay:lowvol `[lowvol|volt15]`

Tier **S**, composite score **0.79** (headline timeframe 15Y).

| timeframe | CAGR | Sharpe | Sortino | MaxDD | trades | win rate | exposure |
|---|---|---|---|---|---|---|---|
| 1Y | 0.6% | -0.31 | -0.49 | -26.1% | 29 | 51.7% | 99.6% |
| 3Y | 23.5% | 1.09 | 1.71 | -26.1% | 63 | 66.7% | 99.9% |
| 5Y | 22.4% | 1.01 | 1.47 | -26.1% | 101 | 69.3% | 99.9% |
| 10Y | 15.2% | 0.58 | 0.77 | -26.8% | 166 | 60.8% | 100.0% |
| 15Y | 19.4% | 0.81 | 1.14 | -26.8% | 235 | 63.4% | 100.0% |

**Robustness across timeframes:** Sharpe > 0.5 in 4/5 windows.
**Parameter sensitivity:** 9 variants tested; composite scores range 0.50–0.79 (spread 0.29 — parameter-sensitive).
**Regime split (10Y):** bull (Nifty>200DMA, 74% of days) ann. return 26.9%; bear ann. return -18.2%.

![equity](../results/plots/strategy_Q11_equity.png)

## 9. N5 — Compose:bbtouch `[bbtouch|quiet|rsi70]`

Tier **S**, composite score **0.79** (headline timeframe 15Y).

| timeframe | CAGR | Sharpe | Sortino | MaxDD | trades | win rate | exposure |
|---|---|---|---|---|---|---|---|
| 1Y | 7.6% | 0.13 | 0.19 | -20.0% | 73 | 61.6% | 99.6% |
| 3Y | 25.9% | 1.26 | 1.87 | -20.0% | 211 | 78.2% | 99.9% |
| 5Y | 33.7% | 1.54 | 2.20 | -20.0% | 351 | 81.5% | 99.9% |
| 10Y | 22.8% | 0.88 | 1.09 | -40.4% | 645 | 78.0% | 100.0% |
| 15Y | 24.2% | 0.96 | 1.27 | -40.4% | 971 | 79.7% | 100.0% |

**Robustness across timeframes:** Sharpe > 0.5 in 4/5 windows.
**Parameter sensitivity:** 36 variants tested; composite scores range 0.46–0.79 (spread 0.33 — parameter-sensitive).
**Regime split (10Y):** bull (Nifty>200DMA, 74% of days) ann. return 34.1%; bear ann. return -12.7%.

![equity](../results/plots/strategy_N5_equity.png)

## 10. I1 — FF Size+Value (price proxy) `[small+3yrev]`

Tier **S**, composite score **0.79** (headline timeframe 15Y).

| timeframe | CAGR | Sharpe | Sortino | MaxDD | trades | win rate | exposure |
|---|---|---|---|---|---|---|---|
| 1Y | 16.1% | 0.68 | 1.06 | -12.5% | 32 | 75.0% | 99.6% |
| 3Y | 27.6% | 1.50 | 2.26 | -12.5% | 65 | 80.0% | 99.9% |
| 5Y | 32.6% | 1.54 | 2.11 | -13.1% | 97 | 80.4% | 99.9% |
| 10Y | 21.5% | 0.84 | 1.04 | -39.5% | 178 | 74.2% | 100.0% |
| 15Y | 21.7% | 0.90 | 1.10 | -39.5% | 237 | 76.8% | 89.3% |

**Robustness across timeframes:** Sharpe > 0.5 in 5/5 windows.
**Regime split (10Y):** bull (Nifty>200DMA, 74% of days) ann. return 27.9%; bear ann. return 0.9%.

![equity](../results/plots/strategy_I1_equity.png)
