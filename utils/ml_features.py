"""Shared ML feature panel for Category H (walk-forward safe).

Features use only information available at the close of day t; the label is
the sign of the day t+2 close over the day t+1 close (i.e., what a signal at
close t actually earns under next-open execution in the backtest engine).
"""
import numpy as np
import pandas as pd

from utils import indicators as ta
from utils.sectors import SECTOR_MAP

_CACHE = {}


def build_panel(ctx):
    key = (ctx["universe"], ctx["interval"])
    if key in _CACHE:
        return _CACHE[key]
    o, h, l, c, v = ctx["open"], ctx["high"], ctx["low"], ctx["close"], ctx["volume"]
    ret = c.pct_change(fill_method=None)

    feats = {
        "rsi14": ta.rsi(c, 14) / 100,
        "macd_hist": ta.macd(c)[2] / c,
        "bb_pctb": (c - ta.bollinger(c)[2]) / (ta.bollinger(c)[1] - ta.bollinger(c)[2]).replace(0, np.nan),
        "atr_pct": ta.atr(h, l, c, 14) / c,
        "vol_ratio": v / v.rolling(20).mean(),
        "ret1": ret,
        "ret5": c.pct_change(5, fill_method=None),
        "ret20": c.pct_change(20, fill_method=None),
        "vol20": ret.rolling(20).std(),
        "dist_hi52": c / c.rolling(252).max() - 1,
        "dist_lo52": c / c.rolling(252).min() - 1,
    }
    sec = pd.Series({t: SECTOR_MAP.get(t, "Other") for t in c.columns})
    sec_mom = ret.rolling(63).sum().T.groupby(sec).transform("mean").T
    feats["sector_mom"] = sec_mom

    stacked = {}
    for name, df in feats.items():
        stacked[name] = df.stack()
    X = pd.DataFrame(stacked)
    idx = X.index
    X["dow"] = idx.get_level_values(0).dayofweek / 4.0
    X["month"] = idx.get_level_values(0).month / 12.0

    # label: return earned by a close-t signal under next-open execution
    fwd = c.shift(-2) / c.shift(-1) - 1
    y = (fwd.stack() > 0).astype(int)
    yret = fwd.stack()

    common = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[common]
    y = y.loc[common]
    yret = yret.loc[common]
    _CACHE[key] = (X, y, yret)
    return _CACHE[key]


def walk_forward_positions(ctx, fit_predict, start_year=2012, prob_thr=0.55,
                           max_train=None, purge_days=5):
    """Generic walk-forward loop.  fit_predict(X_train, y_train, X_test) -> prob array."""
    X, y, _ = build_panel(ctx)
    c = ctx["close"]
    dates = X.index.get_level_values(0)
    pos = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    years = sorted(set(dates.year))
    for yy in [a for a in years if a >= start_year]:
        test_mask = dates.year == yy
        cutoff = pd.Timestamp(f"{yy}-01-01") - pd.Timedelta(days=purge_days)
        train_mask = dates < cutoff
        if train_mask.sum() < 5000 or test_mask.sum() == 0:
            continue
        Xtr, ytr = X[train_mask], y[train_mask]
        if max_train and len(Xtr) > max_train:
            sel = np.random.RandomState(42).choice(len(Xtr), max_train, replace=False)
            Xtr, ytr = Xtr.iloc[sel], ytr.iloc[sel]
        prob = fit_predict(Xtr, ytr, X[test_mask])
        if prob is None:
            continue
        sig = pd.Series(prob, index=X[test_mask].index)
        sig = (sig > prob_thr).astype(float)
        wide = sig.unstack()
        pos.loc[wide.index, wide.columns] = wide.fillna(0.0)
    return pos
