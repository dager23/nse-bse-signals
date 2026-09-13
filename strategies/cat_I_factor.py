"""Category I: Multi-Factor & Portfolio strategies (I1-I15).

Data honesty: yfinance provides only a CURRENT fundamental snapshot, not
point-in-time history.  Strategies marked "(static-snapshot)" apply today's
fundamentals across history and therefore carry lookahead bias — they are
included for completeness and flagged in every report.  Price/volume/dividend
based factors (I1 proxies, I3, I4, I10-I15) are bias-free.
"""
import json
import os

import numpy as np
import pandas as pd

import config
from utils import indicators as ta


def _load_snapshot():
    path = os.path.join(config.DATA_DIR, "fundamentals_snapshot.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return pd.DataFrame(json.load(f)).T.apply(pd.to_numeric, errors="coerce")


def _load_dividends():
    path = os.path.join(config.DATA_DIR, "dividends.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def _month_end_mask(c):
    return pd.Series(c.index, index=c.index).groupby(
        [c.index.year, c.index.month]).transform("max") == c.index


def _monthly_hold(c, select_fn):
    """select_fn(date) -> pd.Series of weights/flags on c.columns."""
    me = _month_end_mask(c)
    pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[me]:
        row = select_fn(d)
        if row is not None:
            pos.loc[d] = row
    return pos.ffill().fillna(0)


def _static_rank_hold(c, scores, top_n=15, trend_gate=True):
    """Hold top_n by static score; optional 200dma trend gate (else biased score
    would just be buy&hold of today's winners)."""
    sel = scores.dropna().nlargest(top_n).index
    pos = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    pos[list(sel.intersection(c.columns))] = 1.0
    if trend_gate:
        pos = pos.where(c > ta.sma(c, 200), 0.0)
    return pos.fillna(0)


def get_strategies():
    S = []
    snap = _load_snapshot()

    # I1: Fama-French style Size + Value tilt (price-based proxies, bias-free)
    #     size = trailing traded value (small = long leg), value = 3y reversal
    def fn_i1(ctx):
        c, v = ctx["close"], ctx["volume"]
        traded = (c * v).rolling(126).mean()
        longterm_rev = -(c.pct_change(756, fill_method=None))  # cheap = beaten down 3y
        def pick(d):
            t = traded.loc[:d].iloc[-1]
            val = longterm_rev.loc[:d].iloc[-1]
            if t.isna().all() or val.isna().all():
                return None
            small = t.rank() <= 25          # smaller half of Nifty 50
            cheap = val.rank(ascending=False) <= 25
            row = ((small & cheap)).astype(float)
            return row if row.sum() >= 3 else None
        return _monthly_hold(c, pick)
    S.append(({"id": "I1", "name": "FF Size+Value (price proxy)", "category": "I-Factor",
               "variant": "small+3yrev"}, fn_i1))

    # I2: Quality factor (static snapshot: ROE high, D/E low, margins stable)
    if snap is not None:
        def fn_i2(ctx):
            c = ctx["close"]
            score = (snap["returnOnEquity"].rank(pct=True)
                     - snap["debtToEquity"].rank(pct=True)
                     + snap["profitMargins"].rank(pct=True))
            return _static_rank_hold(c, score, 15)
        S.append(({"id": "I2", "name": "Quality Factor", "category": "I-Factor",
                   "variant": "(static-snapshot) ROE-DE+PM"}, fn_i2))

    # I3: Low volatility anomaly (bias-free)
    for q in [0.2, 0.3]:
        def fn(ctx, q=q):
            c = ctx["close"]
            vol = c.pct_change(fill_method=None).rolling(252).std()
            def pick(d):
                vv = vol.loc[:d].iloc[-1]
                if vv.isna().all():
                    return None
                return (vv.rank(pct=True) <= q).astype(float)
            return _monthly_hold(c, pick)
        S.append(({"id": "I3", "name": "Low Volatility", "category": "I-Factor",
                   "variant": f"bottom {int(q*100)}% vol"}, fn))

    # I4: High dividend yield (true point-in-time trailing 12m yield)
    def fn_i4(ctx):
        c = ctx["close"]
        divs = _load_dividends()
        if divs is None:
            raise RuntimeError("dividends.parquet missing")
        divs = divs.reindex(columns=c.columns)
        div_daily = divs.reindex(c.index).fillna(0.0)
        ttm = div_daily.rolling(252, min_periods=1).sum()
        yld = ttm / c
        def pick(d):
            yy = yld.loc[:d].iloc[-1]
            if yy.isna().all():
                return None
            return (yy.rank(ascending=False) <= 10).astype(float)
        return _monthly_hold(c, pick)
    S.append(({"id": "I4", "name": "High Dividend Yield", "category": "I-Factor",
               "variant": "top10 TTM"}, fn_i4))

    # I5: Earnings momentum (static snapshot: forward/trailing EPS growth)
    if snap is not None:
        def fn_i5(ctx):
            c = ctx["close"]
            growth = snap["forwardEps"] / snap["trailingEps"].replace(0, np.nan) - 1
            growth = growth.where(snap["trailingEps"] > 0)
            return _static_rank_hold(c, growth, 15)
        S.append(({"id": "I5", "name": "Earnings Momentum", "category": "I-Factor",
                   "variant": "(static-snapshot) fwd/ttm EPS"}, fn_i5))

    # I6: CANSLIM simplified (RS rank + 52w high + market trend [+ static EPS gate])
    def fn_i6(ctx):
        c = ctx["close"]
        bench = ctx["benchmark"]
        rs = c.pct_change(126, fill_method=None).rank(axis=1, pct=True)
        near_high = c > 0.9 * c.rolling(252).max()
        market_up = (bench > bench.rolling(200).mean()).reindex(c.index).fillna(False)
        sig = (rs > 0.7) & near_high
        if snap is not None:
            eps_ok = snap.reindex(c.columns)["earningsGrowth"] > 0.10
            sig = sig & pd.Series(eps_ok.fillna(False), index=c.columns)
        return sig.mul(market_up, axis=0).astype(float)
    S.append(({"id": "I6", "name": "CANSLIM (simplified)", "category": "I-Factor",
               "variant": "RS+52wH+M (EPS static)"}, fn_i6))

    # I7: Magic formula (static snapshot: earnings yield + ROA proxy)
    if snap is not None:
        def fn_i7(ctx):
            c = ctx["close"]
            ey = 1 / snap["trailingPE"].replace(0, np.nan)
            roc = snap["returnOnEquity"]
            score = ey.rank(pct=True) + roc.rank(pct=True)
            return _static_rank_hold(c, score, 15)
        S.append(({"id": "I7", "name": "Magic Formula", "category": "I-Factor",
                   "variant": "(static-snapshot) EY+ROE"}, fn_i7))

    # I8: Piotroski-style score (static snapshot subset of criteria)
    if snap is not None:
        def fn_i8(ctx):
            c = ctx["close"]
            f = ((snap["returnOnEquity"] > 0).astype(int)
                 + (snap["profitMargins"] > 0).astype(int)
                 + (snap["currentRatio"] > 1).astype(int)
                 + (snap["debtToEquity"] < 100).astype(int)
                 + (snap["grossMargins"] > snap["grossMargins"].median()).astype(int)
                 + (snap["operatingMargins"] > snap["operatingMargins"].median()).astype(int)
                 + (snap["revenueGrowth"] > 0).astype(int))
            return _static_rank_hold(c, f.astype(float), 15)
        S.append(({"id": "I8", "name": "Piotroski F (subset)", "category": "I-Factor",
                   "variant": "(static-snapshot) 7 checks"}, fn_i8))

    # I9: Altman Z filter (static snapshot approximation) + trend
    if snap is not None:
        def fn_i9(ctx):
            c = ctx["close"]
            ev = snap["enterpriseValue"]
            z_proxy = (snap["ebitda"] / ev.replace(0, np.nan)).rank(pct=True) \
                + (snap["marketCap"] / snap["totalDebt"].replace(0, np.nan)).rank(pct=True) \
                + snap["currentRatio"].rank(pct=True)
            safe = z_proxy.dropna().nlargest(30).index
            pos = pd.DataFrame(0.0, index=c.index, columns=c.columns)
            pos[list(safe.intersection(c.columns))] = 1.0
            return pos.where(c > ta.sma(c, 100), 0.0).fillna(0)
        S.append(({"id": "I9", "name": "Altman-Z Filter (approx)", "category": "I-Factor",
                   "variant": "(static-snapshot) safe30+trend"}, fn_i9))

    # I10: Equal-weight monthly rebalance (bias-free benchmark strategy)
    def fn_i10(ctx):
        c = ctx["close"]
        def pick(d):
            alive = c.loc[:d].iloc[-1].notna()
            w = alive.astype(float)
            return w / max(w.sum(), 1)
        return _monthly_hold(c, pick)
    S.append(({"id": "I10", "name": "Equal Weight Monthly", "category": "I-Factor",
               "variant": "EW rebal", "signals_are_weights": True}, fn_i10))

    # I11: Risk parity (inverse vol, monthly)
    def fn_i11(ctx):
        c = ctx["close"]
        vol = c.pct_change(fill_method=None).rolling(126).std()
        def pick(d):
            vv = vol.loc[:d].iloc[-1]
            iv = 1 / vv
            iv = iv.replace([np.inf, -np.inf], np.nan).fillna(0)
            s = iv.sum()
            return iv / s if s > 0 else None
        return _monthly_hold(c, pick)
    S.append(({"id": "I11", "name": "Risk Parity", "category": "I-Factor",
               "variant": "inv-vol", "signals_are_weights": True}, fn_i11))

    # I12: Minimum variance portfolio (shrunk covariance, long-only, monthly)
    def fn_i12(ctx):
        c = ctx["close"]
        ret = c.pct_change(fill_method=None)
        def pick(d):
            hist = ret.loc[:d].tail(250).dropna(axis=1)
            if hist.shape[0] < 250 or hist.shape[1] < 10:
                return None
            X = hist.to_numpy()
            cov = np.cov(X.T)
            shrink = 0.3
            cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
            inv = np.linalg.pinv(cov)
            ones = np.ones(cov.shape[0])
            w = inv @ ones / (ones @ inv @ ones)
            w = np.clip(w, 0, 0.1)
            if w.sum() <= 0:
                return None
            w = w / w.sum()
            row = pd.Series(0.0, index=c.columns)
            row[hist.columns] = w
            return row
        return _monthly_hold(c, pick)
    S.append(({"id": "I12", "name": "Min Variance", "category": "I-Factor",
               "variant": "shrink.3 cap10%", "signals_are_weights": True}, fn_i12))

    # I13: Max Sharpe (momentum expected returns, shrunk cov, long-only, monthly)
    def fn_i13(ctx):
        c = ctx["close"]
        ret = c.pct_change(fill_method=None)
        def pick(d):
            hist = ret.loc[:d].tail(250).dropna(axis=1)
            if hist.shape[0] < 250 or hist.shape[1] < 10:
                return None
            mu = hist.tail(126).mean().to_numpy() * 252
            X = hist.to_numpy()
            cov = np.cov(X.T) * 252
            shrink = 0.3
            cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
            try:
                w = np.linalg.pinv(cov) @ (mu - config.RISK_FREE_RATE)
            except Exception:
                return None
            w = np.clip(w, 0, 0.15)
            if w.sum() <= 0:
                return None
            w = w / w.sum()
            row = pd.Series(0.0, index=c.columns)
            row[hist.columns] = w
            return row
        return _monthly_hold(c, pick)
    S.append(({"id": "I13", "name": "Max Sharpe", "category": "I-Factor",
               "variant": "mom mu/shrink.3", "signals_are_weights": True}, fn_i13))

    # I14: Black-Litterman (traded-value prior + sector momentum view)
    def fn_i14(ctx):
        from utils.sectors import SECTOR_MAP
        c, v = ctx["close"], ctx["volume"]
        ret = c.pct_change(fill_method=None)
        traded = (c * v).rolling(126).mean()
        sec = pd.Series({t: SECTOR_MAP.get(t, "Other") for t in c.columns})
        tau = 0.05
        def pick(d):
            hist = ret.loc[:d].tail(250).dropna(axis=1)
            if hist.shape[0] < 250 or hist.shape[1] < 10:
                return None
            cols = hist.columns
            X = hist.to_numpy()
            cov = np.cov(X.T) * 252
            cov = 0.7 * cov + 0.3 * np.diag(np.diag(cov))
            mktw = traded.loc[:d].iloc[-1][cols]
            mktw = (mktw / mktw.sum()).fillna(0).to_numpy()
            delta = 2.5
            pi = delta * cov @ mktw                      # implied equilibrium returns
            # view: top momentum sector outperforms bottom by 5%/yr
            mom3 = ret[cols].loc[:d].tail(63).sum()
            sec_mom = mom3.groupby(sec[cols]).mean()
            if len(sec_mom) < 3:
                return None
            top_s, bot_s = sec_mom.idxmax(), sec_mom.idxmin()
            P = np.zeros((1, len(cols)))
            top_m = (sec[cols] == top_s).to_numpy()
            bot_m = (sec[cols] == bot_s).to_numpy()
            if top_m.sum() == 0 or bot_m.sum() == 0:
                return None
            P[0, top_m] = 1.0 / top_m.sum()
            P[0, bot_m] = -1.0 / bot_m.sum()
            Q = np.array([0.05])
            omega = P @ (tau * cov) @ P.T
            try:
                mid = np.linalg.pinv(P @ (tau * cov) @ P.T + omega)
                mu_bl = pi + (tau * cov) @ P.T @ mid @ (Q - P @ pi)
                w = np.linalg.pinv(delta * cov) @ mu_bl
            except Exception:
                return None
            w = np.clip(w, 0, 0.15)
            if w.sum() <= 0:
                return None
            w = w / w.sum()
            row = pd.Series(0.0, index=c.columns)
            row[cols] = w
            return row
        return _monthly_hold(c, pick)
    S.append(({"id": "I14", "name": "Black-Litterman", "category": "I-Factor",
               "variant": "sector view", "signals_are_weights": True}, fn_i14))

    # I15: Kelly-sized trend following (fractional Kelly on SMA50/200 sleeve)
    def fn_i15(ctx):
        c = ctx["close"]
        base = (ta.sma(c, 50) > ta.sma(c, 200)).astype(float)
        held = base.shift(1).fillna(0)
        n_active = (held != 0).sum(axis=1).clip(lower=1)
        w_eq = held.div(n_active, axis=0)
        strat_ret = (w_eq * c.pct_change(fill_method=None)).sum(axis=1)
        mu = strat_ret.rolling(252).mean() * 252
        var = strat_ret.rolling(252).var() * 252
        kelly = (mu / var.replace(0, np.nan)).clip(0, 2) * 0.5   # half-Kelly
        scale = kelly.fillna(0).clip(0, 1.0)
        return base.div(n_active, axis=0).mul(scale, axis=0)
    S.append(({"id": "I15", "name": "Kelly-Sized Trend", "category": "I-Factor",
               "variant": "half-K SMA50/200", "signals_are_weights": True}, fn_i15))

    return S
