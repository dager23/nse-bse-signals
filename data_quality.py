"""Phase 1.3: data quality report -> reports/data_quality.md"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils import data as dutil

lines = ["# Data Quality Report\n"]
excluded = {}
for label in ["nifty50", "midcap", "smallcap"]:
    panel = dutil.load_panel(label, "1d")
    close = panel["Close"]
    lines.append(f"\n## {label} (daily)\n")
    lines.append(f"- Date range: {close.index.min().date()} to {close.index.max().date()}")
    lines.append(f"- Trading days: {len(close)}")
    lines.append(f"- Tickers with data: {close.shape[1]}\n")
    lines.append("| Ticker | First date | Days | Missing % | Years | Flag |")
    lines.append("|---|---|---|---|---|---|")
    flags = []
    for tkr in close.columns:
        s = close[tkr].dropna()
        first = s.index.min()
        yrs = (s.index.max() - first).days / 365.25
        miss = 1 - len(s) / max(1, (close.index >= first).sum())
        flag = "EXCLUDE (<3y)" if yrs < 3 else ("short history" if yrs < 10 else "")
        if yrs < 3:
            flags.append(tkr)
        lines.append(f"| {tkr} | {first.date()} | {len(s)} | {miss:.1%} | {yrs:.1f} | {flag} |")
    excluded[label] = flags

requested = {"nifty50": set(config.STOCK_UNIVERSE), "midcap": set(config.MIDCAP_UNIVERSE),
             "smallcap": set(config.SMALLCAP_UNIVERSE)}
lines.append("\n## Tickers requested but not delivered\n")
for label in requested:
    got = set(dutil.load_panel(label, "1d")["Close"].columns)
    missing = sorted(requested[label] - got)
    lines.append(f"- {label}: {missing if missing else 'none'}")

lines.append("\n## Survivorship-bias note\n")
lines.append("Universe = mid-2025 Nifty 50 constituents applied retroactively; results on ")
lines.append("long windows carry survivorship bias (stocks that later joined the index look ")
lines.append("stronger historically). TATAMOTORS.NS was delisted after the Oct-2025 demerger ")
lines.append("and returns no Yahoo data; it is absent from all backtests.")

os.makedirs(config.REPORTS_DIR, exist_ok=True)
with open(os.path.join(config.REPORTS_DIR, "data_quality.md"), "w") as f:
    f.write("\n".join(lines))
print("Excluded (<3y):", excluded)
print("Report written.")
