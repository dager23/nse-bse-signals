"""Data loading utilities: wide OHLCV panels per universe/interval."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_CACHE = {}

FIELDS = ["Open", "High", "Low", "Close", "Volume"]


def load_panel(label, interval="1d"):
    """Returns dict field -> wide DataFrame (dates x tickers)."""
    key = (label, interval)
    if key in _CACHE:
        return _CACHE[key]
    path = os.path.join(config.DATA_DIR, f"{label}_{interval}.parquet")
    raw = pd.read_parquet(path)
    out = {}
    for f in FIELDS:
        if f in raw.columns.get_level_values(0):
            df = raw[f].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.sort_index()
            # drop all-nan columns, forward-fill tiny gaps (max 5 bars) in prices
            df = df.dropna(axis=1, how="all")
            if f != "Volume":
                df = df.ffill(limit=5)
            out[f] = df
    _CACHE[key] = out
    return out


def load_universe(universe="nifty50", interval="1d"):
    return load_panel(universe, interval)


def load_benchmark(interval="1d"):
    panel = load_panel("index", interval)
    close = panel["Close"]
    bench = close[config.BENCHMARK].rename("benchmark")
    vix = close[config.INDIA_VIX].rename("vix") if config.INDIA_VIX in close.columns else None
    return bench, vix


def slice_window(df, start, end):
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]


def get_ohlcv(universe="nifty50", interval="1d"):
    p = load_universe(universe, interval)
    return p["Open"], p["High"], p["Low"], p["Close"], p["Volume"]
