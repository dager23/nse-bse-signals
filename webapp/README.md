# NSE Signals Dashboard

On-demand web UI for the strategy research in `../reports/`. Shows what each
surviving strategy holds right now.

## Run it

```bash
cd E:\projectX-pypi\stock_research
python webapp/app.py
```

Open http://127.0.0.1:5000. The first load fetches ~4 years of prices for the
universe and takes 1–2 minutes; after that it's instant. Results cache for 3
hours — hit **Refresh** to force a fresh pull.

## The three views

**Monthly portfolio** — the rank-based baskets. These rebalance on the last
trading day of the month and *do not change in between*; the banner shows the
next rebalance date. Sleeves: Champion blend (R-S), Low-vol + reversal (P2),
Momentum + crash guard (Q9), Volume-weighted momentum (O17).

**Today's signals** — the event-driven sleeves that re-evaluate every session.
`NEW` marks a position opened at the latest close. Sleeves: volume-spike +
inside-day (R-W), deep-oversold RSI (R-T).

**Combined** — all six sleeves merged into one weighted portfolio (equal
capital per sleeve, equal weight within). A stock several sleeves agree on
gets a bigger allocation. This is the diversified version.

## Honest numbers

Every performance figure in the UI is **survivorship-corrected** — measured on
point-in-time Nifty 50 membership, not on today's constituents backfilled.
That correction cost about 0.4 Sharpe across the board and is why these numbers
look more modest than the raw backtests in `reports/master_leaderboard.md`.

| sleeve | PIT Sharpe (15Y) | PIT CAGR | PIT max DD |
|---|---|---|---|
| P2 Low-vol + reversal | 0.79 | 21% | −32% |
| R-S Champion blend | 0.71 | 19% | −29% |
| R-W Volume spike + inside day | 0.52 | 17% | −49% |
| R-T Deep oversold RSI | 0.46 | 15% | −40% |
| Q9 Momentum + crash guard | 0.44 | 14% | −30% |
| O17 Volume-weighted momentum | −0.18 | 5% | −45% |

O17 is kept in the UI because it is a meta-portfolio sleeve, but it failed the
point-in-time test on its own — treat it as diversification, not as a
standalone strategy.

## Deploying

It's a plain Flask app. **Use a single worker** — signals are cached in
process memory, so multiple workers would each fetch their own copy.

```bash
pip install gunicorn
cd webapp && gunicorn -w 1 -b 0.0.0.0:8000 app:app
```

Any host that runs Python and can reach Yahoo Finance works (Render, Railway,
Fly.io, a VPS). Two things to know before putting it on the public internet:

- There's no auth. Add some if the URL will be reachable by others.
- Yahoo Finance rate-limits. The 3-hour cache keeps a single instance well
  under any limit, but don't run many instances against it.

## Maintenance

The universe lives in `../config.py` (`STOCK_UNIVERSE`). When an index
constituent changes or a ticker is renamed, update that list — the UI footer
flags any symbol that stopped returning data (currently TATAMOTORS, post
demerger). This is the one edit that does *not* invalidate the forward test;
changing strategy logic does.

## Not investment advice

These are rule outputs from a backtesting study, not recommendations. The
research found that at 107k configurations tested, no single strategy clears
the statistical bar for "provably better than luck" — the edge claim rests on
replication across independent domains, and it is modest. You can lose money
on any of this.
