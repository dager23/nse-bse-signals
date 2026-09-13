"""Build the static GitHub Pages site.

Computes today's signals with the SAME engine as the local dashboard
(webapp/engine.py — sleeve specs unchanged), attaches BSE prices for the same
companies, and writes a self-contained site to _site/:

    _site/index.html     UI (copied from site/index.html)
    _site/signals.json   today's recommendations

Run by .github/workflows/pages.yml after market close and on manual dispatch.
Exits non-zero on a bad data day so the previous deployment stays live
instead of publishing an empty or broken page.
"""
import json
import os
import shutil
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, os.path.join(ROOT, "webapp")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

import engine  # noqa: E402

OUT = os.path.join(ROOT, "_site")
IST = timezone(timedelta(hours=5, minutes=30))
MIN_TICKERS = 40


def compute_with_retry(attempts=3):
    last = None
    for i in range(attempts):
        try:
            return engine.compute()
        except Exception as e:  # noqa: BLE001  (Yahoo is flaky from CI IPs)
            last = e
            print(f"[warn] compute attempt {i + 1} failed: {e}")
            time.sleep(20 * (i + 1))
    raise last


def bse_quotes(tickers):
    """Latest BSE close + day change for each NSE symbol (best effort)."""
    import yfinance as yf
    symbols = sorted({t + ".BO" for t in tickers})
    try:
        raw = yf.download(symbols, period="10d", auto_adjust=True,
                          progress=False, group_by="column")
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(symbols[0])
        close.index = pd.to_datetime(close.index).tz_localize(None)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] BSE quotes unavailable: {e}")
        return {}
    out = {}
    for sym in close.columns:
        s = close[sym].dropna()
        if s.empty:
            continue
        q = {"price": round(float(s.iloc[-1]), 2), "date": f"{s.index[-1]:%Y-%m-%d}"}
        if len(s) > 1:
            q["day_pct"] = round(float(s.iloc[-1] / s.iloc[-2] - 1) * 100, 2)
        out[sym[:-3]] = q
    return out


def main():
    data = compute_with_retry()

    sleeves = data["monthly"] + data["daily"]
    if data["n_tickers"] < MIN_TICKERS:
        sys.exit(f"ABORT: only {data['n_tickers']} tickers had data "
                 f"(need {MIN_TICKERS}); keeping previous deployment")
    if not any(s["n"] for s in sleeves):
        sys.exit("ABORT: every sleeve is empty; keeping previous deployment")

    held = {h["ticker"] for s in sleeves for h in s["holdings"]}
    bse = bse_quotes(held)
    for s in sleeves:
        for h in s["holdings"]:
            h["bse"] = bse.get(h["ticker"])
    for row in data["combined"]:
        row["bse"] = bse.get(row["ticker"])

    now = datetime.now(timezone.utc)
    data["generated_at"] = now.astimezone(IST).strftime("%d %b %Y, %H:%M IST")
    data["generated_at_utc"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    data["bse_coverage"] = f"{sum(1 for t in held if t in bse)}/{len(held)}"

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    shutil.copy(os.path.join(ROOT, "site", "index.html"), OUT)
    open(os.path.join(OUT, ".nojekyll"), "w").close()
    with open(os.path.join(OUT, "signals.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, default=str)

    print(f"built _site/ | data through {data['asof']} | {data['n_tickers']} NSE "
          f"tickers | BSE quotes {data['bse_coverage']} | "
          f"next rebalance {data['next_rebalance']}")
    for s in sleeves:
        print(f"  {s['sid']:4s} {s['label']:28s} "
              f"{', '.join(h['ticker'] for h in s['holdings']) or '(flat)'}")


if __name__ == "__main__":
    main()
