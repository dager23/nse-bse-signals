"""Fetch current fundamental snapshot + full dividend history for Nifty 50.

NOTE: yfinance exposes only *current* fundamentals, not point-in-time history.
Strategies built on the snapshot (I2, I5-I9) carry lookahead bias and are
labelled "(static-snapshot)" in the leaderboard.  Dividend history is genuine
point-in-time data.
"""
import json
import os
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

FIELDS = ["returnOnEquity", "debtToEquity", "trailingEps", "forwardEps",
          "priceToBook", "trailingPE", "profitMargins", "marketCap",
          "totalRevenue", "totalDebt", "totalCash", "sharesOutstanding",
          "currentRatio", "grossMargins", "operatingMargins", "earningsGrowth",
          "revenueGrowth", "bookValue", "enterpriseValue", "ebitda"]

out = {}
divs = {}
for t in config.STOCK_UNIVERSE:
    try:
        tk = yf.Ticker(t)
        info = tk.info
        out[t] = {f: info.get(f) for f in FIELDS}
        d = tk.dividends
        if d is not None and len(d):
            d.index = pd.to_datetime(d.index).tz_localize(None)
            divs[t] = d
        print(t, "ok")
        time.sleep(0.3)
    except Exception as e:  # noqa: BLE001
        print(t, "FAIL", e)

with open(os.path.join(config.DATA_DIR, "fundamentals_snapshot.json"), "w") as f:
    json.dump(out, f, indent=1)
if divs:
    pd.DataFrame(divs).to_parquet(os.path.join(config.DATA_DIR, "dividends.parquet"))
print("saved", len(out), "snapshots,", len(divs), "dividend histories")
