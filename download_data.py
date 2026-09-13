"""Phase 1: download OHLCV history for all universes at 1d/1wk/1mo intervals.

Stores one parquet per (universe-batch, interval) as a wide panel with a
column MultiIndex (field, ticker).  auto_adjust=True so Close is
split/bonus/dividend adjusted; Open/High/Low are adjusted proportionally.
"""
import os
import sys
import time

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def download_batch(tickers, interval, label, max_retries=3):
    out_path = os.path.join(config.DATA_DIR, f"{label}_{interval}.parquet")
    if os.path.exists(out_path):
        print(f"[skip] {out_path} exists")
        return
    for attempt in range(max_retries):
        try:
            df = yf.download(
                tickers,
                start=config.DATA_START,
                end=config.DATA_END,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=True,
                group_by="column",
            )
            if df is None or df.empty:
                raise RuntimeError("empty frame")
            df.to_parquet(out_path)
            got = df["Close"].notna().any().sum() if isinstance(df.columns, pd.MultiIndex) else 1
            print(f"[ok] {label} {interval}: {df.shape[0]} rows, {got} tickers with data")
            return
        except Exception as e:  # noqa: BLE001
            print(f"[retry {attempt + 1}] {label} {interval}: {e}")
            time.sleep(5 * (attempt + 1))
    print(f"[FAIL] {label} {interval}")


def main():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    jobs = [
        (config.STOCK_UNIVERSE, "nifty50"),
        (config.MIDCAP_UNIVERSE, "midcap"),
        (config.SMALLCAP_UNIVERSE, "smallcap"),
        ([config.BENCHMARK, config.INDIA_VIX], "index"),
    ]
    for interval in config.DATA_INTERVALS:
        for tickers, label in jobs:
            download_batch(tickers, interval, label)
    # 1h data for the last ~2y (yfinance limit ~730 days) for intraday notes
    try:
        df = yf.download(config.STOCK_UNIVERSE[:10], period="720d", interval="1h",
                         auto_adjust=True, progress=False, group_by="column")
        if df is not None and not df.empty:
            df.to_parquet(os.path.join(config.DATA_DIR, "nifty50_top10_1h.parquet"))
            print(f"[ok] 1h sample: {df.shape}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 1h download failed: {e}")
    print("DOWNLOAD COMPLETE")


if __name__ == "__main__":
    main()
