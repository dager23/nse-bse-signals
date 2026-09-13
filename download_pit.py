"""Download daily OHLCV for all ever-members of Nifty 50 (2010-2025) that are
not already in the nifty50 dataset, for the point-in-time universe."""
import os
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.membership import all_tickers

OUT = os.path.join(config.DATA_DIR, "pit_extra_1d.parquet")

existing = set(config.STOCK_UNIVERSE)
need = [t for t in all_tickers() if t not in existing]
print(f"{len(need)} extra tickers to fetch: {need}")

frames = {}
failed = []
for t in need:
    try:
        df = yf.download(t, start="2009-01-01", end="2025-07-02",
                         auto_adjust=True, progress=False)
        if df is None or len(df) < 100:
            failed.append(t)
            print(t, "INSUFFICIENT")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        frames[t] = df[["Open", "High", "Low", "Close", "Volume"]]
        print(t, "ok", len(df))
        time.sleep(0.25)
    except Exception as e:  # noqa: BLE001
        failed.append(t)
        print(t, "FAIL", e)

if frames:
    panel = pd.concat(frames, axis=1)  # columns: (ticker, field)
    panel.to_parquet(OUT)
print(f"saved {len(frames)} tickers; failed: {failed}")
