"""Category K: Event-Driven & Alternative strategies (K1-K8).

Data honesty: NSE insider filings, delivery %, FII/DII flows, put-call ratio
and open interest have no free point-in-time history.  Where the true feed is
unavailable the strategy is implemented as a clearly-labelled PRICE/VOLUME
PROXY of the underlying idea; K5 uses USDINR as an FII-flow proxy (downloaded
on first use).  K8 (market breadth) is fully genuine.
"""
import os

import numpy as np
import pandas as pd

import config
from utils import indicators as ta


def _hold_n(entry, n):
    return entry.astype(float).rolling(n, min_periods=1).max().fillna(0)


def _basket(c, cond_series):
    return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
        c.notna(), 0.0).mul(cond_series.astype(float).reindex(c.index).fillna(0), axis=0)


def _usdinr(index):
    path = os.path.join(config.DATA_DIR, "usdinr_1d.parquet")
    if not os.path.exists(path):
        import yfinance as yf
        df = yf.download("INR=X", start="2009-01-01", auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        px = df["Close"]
        if isinstance(px, pd.DataFrame):
            px = px.iloc[:, 0]
        px.index = pd.to_datetime(px.index).tz_localize(None)
        px.to_frame("usdinr").to_parquet(path)
    px = pd.read_parquet(path)["usdinr"]
    return px.reindex(index).ffill()


def get_strategies():
    S = []

    # K1: Post-earnings-announcement drift (event proxy: big gap + volume spike)
    for jump, hold in [(0.05, 20), (0.04, 40)]:
        def fn(ctx, jump=jump, hold=hold):
            c, v = ctx["close"], ctx["volume"]
            ret = c.pct_change(fill_method=None)
            vspike = v > 3 * v.rolling(20).mean()
            event_up = (ret > jump) & vspike
            return _hold_n(event_up, hold)
        S.append(({"id": "K1", "name": "PEAD (event proxy)", "category": "K-Event",
                   "variant": f"+{int(jump*100)}%/3xVol h{hold}"}, fn))

    # K2: Insider-accumulation proxy (huge volume, flat price = block accumulation)
    def fn_k2(ctx):
        c, v = ctx["close"], ctx["volume"]
        ret = c.pct_change(fill_method=None)
        block = (v > 5 * v.rolling(60).mean()) & (ret.abs() < 0.015)
        return _hold_n(block, 40)
    S.append(({"id": "K2", "name": "Insider/Block Accumulation (proxy)", "category": "K-Event",
               "variant": "5xVol flat px h40"}, fn_k2))

    # K3: Index-inclusion proxy (stock enters top-50 by traded value)
    def fn_k3(ctx):
        c, v = ctx["close"], ctx["volume"]
        traded = (c * v).rolling(63).mean()
        rank = traded.rank(axis=1, ascending=False)
        entered = (rank <= 40) & (rank.shift(21) > 45)
        return _hold_n(entered, 60)
    S.append(({"id": "K3", "name": "Index Inclusion (proxy)", "category": "K-Event",
               "variant": "traded-value entry h60"}, fn_k3))

    # K4: High delivery-% proxy (high volume + narrow range = conviction buying)
    def fn_k4(ctx):
        o, h, l, c, v = ctx["open"], ctx["high"], ctx["low"], ctx["close"], ctx["volume"]
        rng = (h - l) / c
        narrow = rng < rng.rolling(60).quantile(0.3)
        highvol = v > 1.5 * v.rolling(20).mean()
        up = c > c.shift(1)
        return _hold_n(narrow & highvol & up, 10)
    S.append(({"id": "K4", "name": "Delivery Conviction (proxy)", "category": "K-Event",
               "variant": "narrow+1.5xVol h10"}, fn_k4))

    # K5: FII flow proxy via USDINR trend (INR strength ~ foreign inflows)
    def fn_k5(ctx):
        c = ctx["close"]
        inr = _usdinr(c.index)
        if inr is None:
            raise RuntimeError("USDINR download failed")
        inr_weak = inr > inr.rolling(50).mean()
        risk_on = ~inr_weak  # INR strengthening -> inflows
        return _basket(c, risk_on)
    S.append(({"id": "K5", "name": "FII Flow (USDINR proxy)", "category": "K-Event",
               "variant": "INR<50dma"}, fn_k5))

    # K6: Sentiment extreme via India VIX percentile (PCR proxy)
    def fn_k6(ctx):
        c = ctx["close"]
        vix = ctx["vix"]
        if vix is None:
            raise RuntimeError("India VIX unavailable")
        vix = vix.reindex(c.index).ffill()
        pct = vix.rolling(500, min_periods=250).rank(pct=True)
        fear = pct > 0.9
        entry = fear & (pct.shift(1) <= 0.9)
        sig = _hold_n(pd.DataFrame({0: entry}), 40)[0]
        return _basket(c, sig)
    S.append(({"id": "K6", "name": "Sentiment Extreme (VIX pctl)", "category": "K-Event",
               "variant": "VIX>p90 h40"}, fn_k6))

    # K7: Rising-OI analog (price up + sustained volume uptrend)
    def fn_k7(ctx):
        c, v = ctx["close"], ctx["volume"]
        vol_trend = v.rolling(20).mean() > v.rolling(60).mean()
        px_trend = c > ta.sma(c, 20)
        return ((vol_trend & px_trend)).astype(float)
    S.append(({"id": "K7", "name": "Rising OI Analog (proxy)", "category": "K-Event",
               "variant": "vol20>60 & px>20d"}, fn_k7))

    # K8: 200-DMA market breadth (genuine)
    def fn_k8a(ctx):
        c = ctx["close"]
        above = (c > ta.sma(c, 200)).sum(axis=1) / c.notna().sum(axis=1).clip(lower=1)
        return _basket(c, above > 0.5)
    S.append(({"id": "K8", "name": "200DMA Breadth", "category": "K-Event",
               "variant": ">50% above"}, fn_k8a))

    def fn_k8b(ctx):
        c = ctx["close"]
        above = (c > ta.sma(c, 200)).sum(axis=1) / c.notna().sum(axis=1).clip(lower=1)
        washout = (above > 0.2) & (above.shift(5) <= 0.2)  # thrust off washout low
        sig = _hold_n(pd.DataFrame({0: washout}), 120)[0]
        return _basket(c, sig)
    S.append(({"id": "K8", "name": "Breadth Washout Thrust", "category": "K-Event",
               "variant": "20% washout h120"}, fn_k8b))

    return S
