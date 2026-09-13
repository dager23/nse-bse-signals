"""Vectorized technical indicators operating on wide DataFrames (dates x tickers).

Every function accepts and returns wide DataFrames so strategies can be computed
for the whole universe in one call.  Iterative indicators (Supertrend, PSAR,
KAMA, McGinley) loop over rows but stay vectorized across columns.
"""
import numpy as np
import pandas as pd


# ----------------------------------------------------------------- moving averages
def sma(df, n):
    return df.rolling(n, min_periods=n).mean()


def ema(df, n):
    return df.ewm(span=n, adjust=False, min_periods=n).mean()


def wma(df, n):
    w = np.arange(1, n + 1, dtype=float)
    return df.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def dema(df, n):
    e1 = ema(df, n)
    return 2 * e1 - ema(e1, n)


def tema(df, n):
    e1 = ema(df, n)
    e2 = ema(e1, n)
    e3 = ema(e2, n)
    return 3 * e1 - 3 * e2 + e3


def hma(df, n):
    half, root = max(2, n // 2), max(2, int(np.sqrt(n)))
    return wma(2 * wma(df, half) - wma(df, n), root)


def vwma(close, volume, n):
    return (close * volume).rolling(n).sum() / volume.rolling(n).sum()


def kama(close, er_n=10, fast=2, slow=30):
    """Kaufman adaptive MA (row loop, column vectorized)."""
    price = close.to_numpy(float)
    change = np.abs(price - np.roll(price, er_n, axis=0))
    vol = pd.DataFrame(np.abs(np.diff(price, axis=0, prepend=price[:1]))).rolling(er_n).sum().to_numpy()
    er = np.where(vol == 0, 0.0, change / np.where(vol == 0, 1, vol))
    er[:er_n] = 0.0
    sc = (er * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
    out = np.full_like(price, np.nan)
    started = np.zeros(price.shape[1], dtype=bool)
    prev = np.zeros(price.shape[1])
    for t in range(price.shape[0]):
        p = price[t]
        valid = ~np.isnan(p)
        new = valid & ~started
        prev = np.where(new, p, prev)
        started |= new
        upd = valid & started & ~new
        prev = np.where(upd, prev + sc[t] * (p - prev), prev)
        out[t] = np.where(started, prev, np.nan)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


def mcginley(close, n=14):
    price = close.to_numpy(float)
    out = np.full_like(price, np.nan)
    started = np.zeros(price.shape[1], dtype=bool)
    md = np.zeros(price.shape[1])
    for t in range(price.shape[0]):
        p = price[t]
        valid = ~np.isnan(p)
        new = valid & ~started
        md = np.where(new, p, md)
        started |= new
        upd = valid & started & ~new
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(md > 0, p / np.where(md == 0, 1, md), 1.0)
            md = np.where(upd, md + (p - md) / np.maximum(n * ratio ** 4, 1e-9), md)
        out[t] = np.where(started, md, np.nan)
    return pd.DataFrame(out, index=close.index, columns=close.columns)


# ----------------------------------------------------------------- volatility
def true_range(high, low, close):
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()]).groupby(level=0).max().reindex(close.index)


def atr(high, low, close, n=14):
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std()
    return mid, mid + k * sd, mid - k * sd, (4 * sd) / mid  # mid, upper, lower, bandwidth(2k)


def keltner(high, low, close, n=20, k=2.0, atr_n=10):
    mid = ema(close, n)
    a = atr(high, low, close, atr_n)
    return mid, mid + k * a, mid - k * a


def donchian(high, low, n=20):
    return high.rolling(n, min_periods=n).max(), low.rolling(n, min_periods=n).min()


# ----------------------------------------------------------------- oscillators
def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100).where(close.notna())


def stochastic(high, low, close, k_n=14, d_n=3, smooth=3):
    ll = low.rolling(k_n, min_periods=k_n).min()
    hh = high.rolling(k_n, min_periods=k_n).max()
    k_raw = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    k = k_raw.rolling(smooth).mean()
    return k, k.rolling(d_n).mean()


def stoch_rsi(close, rsi_n=14, stoch_n=14, k_n=3, d_n=3):
    r = rsi(close, rsi_n)
    lo = r.rolling(stoch_n).min()
    hi = r.rolling(stoch_n).max()
    sr = 100 * (r - lo) / (hi - lo).replace(0, np.nan)
    k = sr.rolling(k_n).mean()
    return k, k.rolling(d_n).mean()


def cci(high, low, close, n=20):
    tp = (high + low + close) / 3
    ma = tp.rolling(n).mean()
    md = (tp - ma).abs().rolling(n).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def williams_r(high, low, close, n=14):
    hh = high.rolling(n).max()
    ll = low.rolling(n).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


def macd(close, fast=12, slow=26, signal=9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def roc(close, n=12):
    return close.pct_change(n, fill_method=None) * 100


def tsi(close, long=25, short=13):
    m = close.diff()
    num = m.ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
    den = m.abs().ewm(span=long, adjust=False).mean().ewm(span=short, adjust=False).mean()
    return 100 * num / den.replace(0, np.nan)


def dpo(close, n=20):
    shift = n // 2 + 1
    return close.shift(shift) - close.rolling(n).mean().shift(shift)  # causal variant


def kst(close):
    r1 = roc(close, 10).rolling(10).mean()
    r2 = roc(close, 15).rolling(10).mean()
    r3 = roc(close, 20).rolling(10).mean()
    r4 = roc(close, 30).rolling(15).mean()
    line = r1 + 2 * r2 + 3 * r3 + 4 * r4
    return line, line.rolling(9).mean()


def coppock(close, wl=10, r1=14, r2=11):
    return (roc(close, r1) + roc(close, r2)).rolling(wl).apply(
        lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True)


# ----------------------------------------------------------------- trend/direction
def adx(high, low, close, n=14):
    up = high.diff()
    dn = -low.diff()
    plus_dm = up.where((up > dn) & (up > 0), 0.0)
    minus_dm = dn.where((dn > up) & (dn > 0), 0.0)
    tr_s = true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr_s.replace(0, np.nan)
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr_s.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), pdi, mdi


def aroon(high, low, n=25):
    up = high.rolling(n + 1).apply(lambda x: 100 * float(np.argmax(x)) / n, raw=True)
    dn = low.rolling(n + 1).apply(lambda x: 100 * float(np.argmin(x)) / n, raw=True)
    return up, dn


def supertrend(high, low, close, n=10, mult=3.0):
    """Returns direction df: +1 uptrend, -1 downtrend."""
    a = atr(high, low, close, n)
    hl2 = (high + low) / 2
    ub = (hl2 + mult * a).to_numpy(float)
    lb = (hl2 - mult * a).to_numpy(float)
    c = close.to_numpy(float)
    T, N = c.shape
    fub = np.copy(ub)
    flb = np.copy(lb)
    dirn = np.full((T, N), np.nan)
    d = np.ones(N)
    for t in range(1, T):
        fub[t] = np.where((ub[t] < fub[t - 1]) | (c[t - 1] > fub[t - 1]), ub[t], fub[t - 1])
        flb[t] = np.where((lb[t] > flb[t - 1]) | (c[t - 1] < flb[t - 1]), lb[t], flb[t - 1])
        d = np.where(c[t] > fub[t - 1], 1, np.where(c[t] < flb[t - 1], -1, d))
        dirn[t] = np.where(np.isnan(c[t]) | np.isnan(fub[t]), np.nan, d)
    return pd.DataFrame(dirn, index=close.index, columns=close.columns)


def psar(high, low, close, af0=0.02, af_step=0.02, af_max=0.2):
    """Returns direction df: +1 long, -1 short."""
    h = high.to_numpy(float)
    l = low.to_numpy(float)
    T, N = h.shape
    dirn = np.full((T, N), np.nan)
    up = np.ones(N, dtype=bool)
    af = np.full(N, af0)
    ep = np.where(np.isnan(h[0]), np.nan, h[0])
    sar = np.where(np.isnan(l[0]), np.nan, l[0])
    for t in range(1, T):
        sar = sar + af * (ep - sar)
        rev_dn = up & (l[t] < sar)
        rev_up = ~up & (h[t] > sar)
        sar = np.where(rev_dn, ep, np.where(rev_up, ep, sar))
        new_ep_up = up & ~rev_dn & (h[t] > ep)
        new_ep_dn = ~up & ~rev_up & (l[t] < ep)
        af = np.where(rev_dn | rev_up, af0, np.where(new_ep_up | new_ep_dn, np.minimum(af + af_step, af_max), af))
        ep = np.where(rev_dn, l[t], np.where(rev_up, h[t], np.where(new_ep_up, h[t], np.where(new_ep_dn, l[t], ep))))
        up = np.where(rev_dn, False, np.where(rev_up, True, up)).astype(bool)
        dirn[t] = np.where(np.isnan(h[t]), np.nan, np.where(up, 1.0, -1.0))
    return pd.DataFrame(dirn, index=close.index, columns=close.columns)


def ichimoku(high, low, close, tenkan_n=9, kijun_n=26, senkou_b_n=52):
    tenkan = (high.rolling(tenkan_n).max() + low.rolling(tenkan_n).min()) / 2
    kijun = (high.rolling(kijun_n).max() + low.rolling(kijun_n).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(kijun_n)
    span_b = ((high.rolling(senkou_b_n).max() + low.rolling(senkou_b_n).min()) / 2).shift(kijun_n)
    return tenkan, kijun, span_a, span_b


# ----------------------------------------------------------------- volume
def obv(close, volume):
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


def ad_line(high, low, close, volume):
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (clv.fillna(0) * volume).cumsum()


def cmf(high, low, close, volume, n=20):
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    return (clv.fillna(0) * volume).rolling(n).sum() / volume.rolling(n).sum()


def mfi(high, low, close, volume, n=14):
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def emv(high, low, volume, n=14, scale=1e8):
    dm = ((high + low) / 2).diff()
    box = (volume / scale) / (high - low).replace(0, np.nan)
    return (dm / box).rolling(n).mean()


def force_index(close, volume, n=13):
    return (close.diff() * volume).ewm(span=n, adjust=False).mean()


def klinger(high, low, close, volume, fast=34, slow=55, sig_n=13):
    trend = np.sign((high + low + close).diff()).fillna(0)
    dm = high - low
    vf = volume * trend * 100
    vf = vf * (dm / dm.rolling(2).mean().replace(0, np.nan)).fillna(1)
    ko = vf.ewm(span=fast, adjust=False).mean() - vf.ewm(span=slow, adjust=False).mean()
    return ko, ko.ewm(span=sig_n, adjust=False).mean()


def nvi(close, volume):
    ret = close.pct_change(fill_method=None).fillna(0)
    lowvol = (volume < volume.shift(1)).astype(float)
    return (1 + ret * lowvol).cumprod() * 1000


# ----------------------------------------------------------------- stats helpers
def zscore(df, n=20):
    return (df - df.rolling(n).mean()) / df.rolling(n).std().replace(0, np.nan)


def rolling_hurst(close, window=200, max_lag=20):
    """Rolling Hurst exponent via rescaled variance of lagged differences."""
    logp = np.log(close)
    lags = np.arange(2, max_lag)
    loglags = np.log(lags)

    def hurst_1d(x):
        if np.isnan(x).any():
            return np.nan
        tau = [np.std(x[lag:] - x[:-lag]) for lag in lags]
        tau = np.asarray(tau)
        if (tau <= 0).any():
            return np.nan
        return np.polyfit(loglags, np.log(tau), 1)[0]

    return logp.rolling(window).apply(hurst_1d, raw=True)


def linreg_channel(close, n=100, k=2.0):
    """Rolling linear regression endpoint value and residual std."""
    x = np.arange(n)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def endpoint(y):
        b = ((x - x_mean) * (y - y.mean())).sum() / x_var
        a = y.mean() - b * x_mean
        return a + b * (n - 1)

    def resid_std(y):
        b = ((x - x_mean) * (y - y.mean())).sum() / x_var
        a = y.mean() - b * x_mean
        return np.std(y - (a + b * x))

    ep = close.rolling(n).apply(endpoint, raw=True)
    sd = close.rolling(n).apply(resid_std, raw=True)
    return ep, ep + k * sd, ep - k * sd


def heikin_ashi(open_, high, low, close):
    ha_close = (open_ + high + low + close) / 4
    ha_open = ha_close.copy()
    o = open_.to_numpy(float)
    hc = ha_close.to_numpy(float)
    ho = np.full_like(o, np.nan)
    ho[0] = (o[0] + close.to_numpy(float)[0]) / 2
    for t in range(1, len(o)):
        prev = np.where(np.isnan(ho[t - 1]), o[t], ho[t - 1])
        ho[t] = (prev + np.where(np.isnan(hc[t - 1]), o[t], hc[t - 1])) / 2
    ha_open = pd.DataFrame(ho, index=open_.index, columns=open_.columns)
    return ha_open, ha_close
