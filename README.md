# NSE / BSE Strategy Signals

**Live site: https://dager23.github.io/nse-bse-signals/**

Daily rule-based stock baskets for Nifty 50 companies, tradeable on NSE or
BSE, produced by the strategies that survived a large backtesting study
(107k+ configurations, survivorship-corrected, replicated across independent
data). The site regenerates every weekday after market close, and on demand.

> **Not investment advice.** These are outputs of mechanical rules from a
> research project. The honest (point-in-time) backtest edge is modest —
> Sharpe ~0.5–0.8, drawdowns of 30–50% — and past results don't predict
> future returns. You can lose money on any of this.

## Getting a fresh recommendation on demand

GitHub Pages only serves static files, so the computation runs in GitHub
Actions and the page displays its latest result:

1. Open **Actions → "Generate signals & deploy site" → Run workflow**
   (the site's *Regenerate* button links straight there).
2. Wait ~2–3 minutes for it to finish; reload the site.

It also runs automatically at 19:00 IST every weekday.

## What the site shows

| view | what it is | changes |
|---|---|---|
| **Monthly portfolio** | Rank-based baskets: champion blend (low-beta + momentum + reversal), low-vol + reversal, momentum with crash guard, volume-weighted momentum | only on the last trading day of the month |
| **Today's signals** | Event sleeves: volume spike + inside day, deep-oversold RSI | any session |
| **Combined** | All six sleeves merged into one weighted portfolio, with a capital calculator | follows the above |

Toggle **NSE / BSE** to see prices from either exchange. Signals are computed
from NSE data (what the strategies were validated on); the BSE view shows the
same companies' BSE quotes so you can place orders there.

## Repository layout

```
build_site.py                 # computes signals -> _site/ (run by Actions)
site/index.html               # the static UI
.github/workflows/pages.yml   # schedule + manual trigger + Pages deploy
webapp/                       # same engine as a local Flask dashboard
signals_live.py               # local forward-test logger
strategies/ utils/            # strategy + indicator code (18 modules)
reports/                      # full research write-up — start with
                              #   reports/executive_summary.md
```

Run locally: `pip install -r requirements-site.txt && python build_site.py`,
then serve `_site/` (e.g. `python -m http.server -d _site`).

The research pipeline (`run_all.py`, `screener.py`, …) also needs
scipy/scikit-learn/statsmodels etc. and downloads its own data into `data/`,
which is not committed.

## Maintenance notes

- **TATAMOTORS** no longer resolves on Yahoo after the demerger and is
  excluded automatically (the site footer lists any such symbol). Update
  `config.STOCK_UNIVERSE` when index constituents change.
- GitHub disables scheduled workflows on public repos after 60 days without
  commits; re-enable from the Actions tab if the site stops updating.
