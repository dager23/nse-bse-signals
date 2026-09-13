"""Shared helpers for generated strategy families (categories M-Q)."""
import numpy as np
import pandas as pd


def state(entry, exit_, index, columns):
    """Long/flat state machine: enter on entry=True, flat on exit=True."""
    st = pd.DataFrame(np.nan, index=index, columns=columns)
    st[entry & ~exit_] = 1.0
    st[exit_] = 0.0
    return st.ffill().fillna(0)


def hold_n(entry, n):
    """Long for n bars after each entry (overlapping entries extend)."""
    return entry.astype(float).rolling(n, min_periods=1).max().fillna(0)


def monthly_topn(c, score, top_n=10, ascending=False):
    """Monthly-rebalanced equal basket of top_n by score (data up to rebal date)."""
    me = pd.Series(c.index, index=c.index).groupby(
        [c.index.year, c.index.month]).transform("max") == c.index
    pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
    for d in c.index[me]:
        s = score.loc[:d].iloc[-1]
        s = s.dropna()
        if len(s) < top_n:
            continue
        sel = s.nsmallest(top_n).index if ascending else s.nlargest(top_n).index
        row = pd.Series(0.0, index=c.columns)
        row[list(sel)] = 1.0
        pos.loc[d] = row
    return pos.ffill().fillna(0)


def atr_trail(entry, close, atr_df, mult):
    """Long positions with an ATR trailing stop (loop over time, vectorized cols)."""
    c = close.to_numpy(float)
    a = atr_df.to_numpy(float)
    e = entry.to_numpy(bool)
    T, N = c.shape
    pos = np.zeros((T, N))
    stop = np.full(N, np.nan)
    inpos = np.zeros(N, dtype=bool)
    for t in range(T):
        px, at = c[t], a[t]
        hit = inpos & ~np.isnan(px) & ~np.isnan(stop) & (px < stop)
        inpos = inpos & ~hit
        stop = np.where(hit, np.nan, stop)
        opens = e[t] & ~inpos & ~np.isnan(px) & ~np.isnan(at)
        inpos = inpos | opens
        stop = np.where(opens, px - mult * at, stop)
        trail = inpos & ~opens & ~np.isnan(px) & ~np.isnan(at)
        stop = np.where(trail, np.maximum(stop, px - mult * at), stop)
        pos[t] = inpos.astype(float)
    return pd.DataFrame(pos, index=close.index, columns=close.columns)


def cross_up(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1))


def cross_dn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1))
