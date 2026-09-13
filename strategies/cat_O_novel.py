"""Category O: original/novel strategy ideas beyond the classic canon.

33 idea families, ~104 variants: residual momentum, overnight-strength,
trend-quality R^2, efficiency ratio, Amihud liquidity, up/down volume
imbalance, skew/tail-ratio factors, vol-of-vol, drawdown buying, correlation
decoupling, beta timing, NR7, robust-z reversion, streaks, quiet
accumulation, autocorrelation auto-switching, OU half-life selection, VRP
timing, trend age, and more.  All causal, all vectorized.
"""
import numpy as np
import pandas as pd

from utils import indicators as ta
from utils.composite import state, hold_n, monthly_topn
from utils.sectors import SECTOR_MAP


def _robust_sd(df, n):
    q3 = df.rolling(n).quantile(0.75)
    q1 = df.rolling(n).quantile(0.25)
    return ((q3 - q1) / 1.349).replace(0, np.nan)


def _rolling_beta(ret, bret, n=252):
    bm = bret.rolling(n).mean()
    cov = ret.mul(bret, axis=0).rolling(n).mean() - ret.rolling(n).mean().mul(bm, axis=0)
    var = (bret ** 2).rolling(n).mean() - bm ** 2
    return cov.div(var.replace(0, np.nan), axis=0)


def _time_ramp(c):
    return pd.DataFrame(np.tile(np.arange(len(c))[:, None], (1, c.shape[1])),
                        index=c.index, columns=c.columns, dtype=float)


def get_strategies():
    S = []

    # O1: Residual (idiosyncratic) momentum
    for lb in [63, 126, 252]:
        def fn(ctx, lb=lb):
            c = ctx["close"]
            ret = c.pct_change(fill_method=None)
            bret = ctx["benchmark"].pct_change().reindex(c.index).fillna(0)
            beta = _rolling_beta(ret, bret, 252)
            resid = ret - beta.mul(bret, axis=0)
            return monthly_topn(c, resid.rolling(lb).sum(), 10)
        S.append(({"id": "O1", "name": "Residual Momentum", "category": "O-Novel",
                   "variant": f"lb{lb} top10"}, fn))

    # O2: Overnight-strength / intraday-strength factors
    for lb in [21, 63, 126]:
        for mode in ["overnight", "intraday"]:
            def fn(ctx, lb=lb, mode=mode):
                o, c = ctx["open"], ctx["close"]
                on = o / c.shift(1) - 1
                intr = c / o - 1
                sig = on.rolling(lb).sum() if mode == "overnight" else intr.rolling(lb).sum()
                return monthly_topn(c, sig, 10)
            S.append(({"id": "O2", "name": "Overnight Strength", "category": "O-Novel",
                       "variant": f"{mode} lb{lb}"}, fn))

    # O3: Trend-quality R^2 (rolling corr^2 of logprice with time)
    for n in [63, 126]:
        for thr in [0.6, 0.75, 0.85]:
            def fn(ctx, n=n, thr=thr):
                c = ctx["close"]
                logp = np.log(c)
                r = logp.rolling(n).corr(_time_ramp(c))
                r2 = r ** 2
                up = logp.diff(n) > 0
                return state((r2 > thr) & up, (r2 < thr * 0.7) | ~up,
                             c.index, c.columns)
            S.append(({"id": "O3", "name": "Trend Quality R2", "category": "O-Novel",
                       "variant": f"n{n} R2>{thr}"}, fn))

    # O4: Kaufman efficiency-ratio direction
    for n in [10, 20, 50]:
        for thr in [0.3, 0.5]:
            def fn(ctx, n=n, thr=thr):
                c = ctx["close"]
                change = (c - c.shift(n)).abs()
                path = c.diff().abs().rolling(n).sum()
                er = change / path.replace(0, np.nan)
                up = c > c.shift(n)
                return state((er > thr) & up, (er < thr / 2) | ~up, c.index, c.columns)
            S.append(({"id": "O4", "name": "Efficiency Ratio", "category": "O-Novel",
                       "variant": f"n{n} ER>{thr}"}, fn))

    # O5: Amihud illiquidity factor (both directions)
    for lb in [63, 126]:
        for mode in ["liquid", "illiquid"]:
            def fn(ctx, lb=lb, mode=mode):
                c, v = ctx["close"], ctx["volume"]
                ret = c.pct_change(fill_method=None)
                amihud = (ret.abs() / (c * v).replace(0, np.nan)).rolling(lb).mean()
                return monthly_topn(c, amihud, 10, ascending=(mode == "liquid"))
            S.append(({"id": "O5", "name": "Amihud Illiquidity", "category": "O-Novel",
                       "variant": f"{mode} lb{lb}"}, fn))

    # O6: Up/down volume imbalance
    for n in [20, 50]:
        for thr in [1.2, 1.5]:
            def fn(ctx, n=n, thr=thr):
                c, v = ctx["close"], ctx["volume"]
                up_v = v.where(c > c.shift(1), 0.0).rolling(n).sum()
                dn_v = v.where(c < c.shift(1), 0.0).rolling(n).sum()
                ratio = up_v / dn_v.replace(0, np.nan)
                return state(ratio > thr, ratio < 1.0, c.index, c.columns)
            S.append(({"id": "O6", "name": "Up/Down Volume", "category": "O-Novel",
                       "variant": f"n{n}>{thr}"}, fn))

    # O7: Return skewness factor
    for lb in [63, 126]:
        for mode in ["neg", "pos"]:
            def fn(ctx, lb=lb, mode=mode):
                c = ctx["close"]
                sk = c.pct_change(fill_method=None).rolling(lb).skew()
                return monthly_topn(c, sk, 10, ascending=(mode == "neg"))
            S.append(({"id": "O7", "name": "Skewness Factor", "category": "O-Novel",
                       "variant": f"{mode} lb{lb}"}, fn))

    # O8: Tail ratio factor
    for mode in ["high", "low"]:
        def fn(ctx, mode=mode):
            c = ctx["close"]
            ret = c.pct_change(fill_method=None)
            tail = ret.rolling(126).quantile(0.95) / \
                ret.rolling(126).quantile(0.05).abs().replace(0, np.nan)
            return monthly_topn(c, tail, 10, ascending=(mode == "low"))
        S.append(({"id": "O8", "name": "Tail Ratio", "category": "O-Novel",
                   "variant": f"{mode} 126d"}, fn))

    # O9: Vol-of-vol
    def fn_o9a(ctx):
        c = ctx["close"]
        vol = c.pct_change(fill_method=None).rolling(20).std()
        vov = vol.rolling(63).std() / vol.rolling(63).mean().replace(0, np.nan)
        return monthly_topn(c, vov, 10, ascending=True)
    S.append(({"id": "O9", "name": "Vol-of-Vol", "category": "O-Novel",
               "variant": "lowest top10"}, fn_o9a))
    for q in [0.3, 0.5]:
        def fn(ctx, q=q):
            c = ctx["close"]
            vol = c.pct_change(fill_method=None).rolling(20).std()
            vov = vol.rolling(63).std() / vol.rolling(63).mean().replace(0, np.nan)
            lowvov = vov < vov.rolling(252).quantile(q)
            up = c > ta.sma(c, 100)
            return (lowvov & up).astype(float)
        S.append(({"id": "O9", "name": "Vol-of-Vol Gate", "category": "O-Novel",
                   "variant": f"q{q}+trend"}, fn))

    # O10: Drawdown buying
    for x in [15, 25, 35]:
        for gate in [True, False]:
            def fn(ctx, x=x, gate=gate):
                c = ctx["close"]
                bench = ctx["benchmark"]
                dd = c / c.rolling(252).max() - 1
                entry = dd < -x / 100
                if gate:
                    bench_up = (bench > bench.rolling(200).mean()).reindex(c.index).fillna(False)
                    entry = entry.mul(bench_up, axis=0).astype(bool)
                exit_ = dd > -x / 200  # recovered half the drawdown
                return state(entry, exit_, c.index, c.columns)
            S.append(({"id": "O10", "name": "Drawdown Buying", "category": "O-Novel",
                       "variant": f"dd{x}%{'+mkt' if gate else ''}"}, fn))

    # O11: Correlation decoupling + momentum
    for thr in [0.1, 0.2]:
        def fn(ctx, thr=thr):
            c = ctx["close"]
            ret = c.pct_change(fill_method=None)
            bret = ctx["benchmark"].pct_change().reindex(c.index).fillna(0)
            corr = ret.rolling(63).corr(pd.DataFrame(
                np.tile(bret.to_numpy()[:, None], (1, c.shape[1])),
                index=c.index, columns=c.columns))
            base = corr.rolling(252).median()
            decoupled = corr < base - thr
            mom = c > c.shift(21)
            return (decoupled & mom).astype(float)
        S.append(({"id": "O11", "name": "Correlation Decoupling", "category": "O-Novel",
                   "variant": f"drop>{thr}"}, fn))

    # O12: Beta timing
    def fn_o12a(ctx):
        c = ctx["close"]
        ret = c.pct_change(fill_method=None)
        bret = ctx["benchmark"].pct_change().reindex(c.index).fillna(0)
        beta = _rolling_beta(ret, bret, 252)
        bench_up = (ctx["benchmark"] > ctx["benchmark"].rolling(200).mean()).reindex(c.index)
        me = pd.Series(c.index, index=c.index).groupby(
            [c.index.year, c.index.month]).transform("max") == c.index
        pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        for d in c.index[me]:
            b = beta.loc[:d].iloc[-1].dropna()
            if len(b) < 15:
                continue
            sel = b.nlargest(10).index if bench_up.loc[d] else b.nsmallest(10).index
            row = pd.Series(0.0, index=c.columns)
            row[list(sel)] = 1.0
            pos.loc[d] = row
        return pos.ffill().fillna(0)
    S.append(({"id": "O12", "name": "Beta Timing", "category": "O-Novel",
               "variant": "hi-beta bull/lo-beta bear"}, fn_o12a))

    def fn_o12b(ctx):
        c = ctx["close"]
        ret = c.pct_change(fill_method=None)
        bret = ctx["benchmark"].pct_change().reindex(c.index).fillna(0)
        beta = _rolling_beta(ret, bret, 252)
        mom = c.pct_change(126, fill_method=None)
        score = -beta.rank(axis=1, pct=True) + mom.rank(axis=1, pct=True)
        return monthly_topn(c, score, 10)
    S.append(({"id": "O12", "name": "Low-Beta + Momentum", "category": "O-Novel",
               "variant": "rank blend"}, fn_o12b))

    # O13: Sector-relative strength
    def fn_o13a(ctx):
        c = ctx["close"]
        sec = pd.Series({t: SECTOR_MAP.get(t, "Other") for t in c.columns})
        r63 = c.pct_change(63, fill_method=None)
        sec_mean = r63.T.groupby(sec).transform("mean").T
        return monthly_topn(c, r63 - sec_mean, 10)
    S.append(({"id": "O13", "name": "Sector-Relative Strength", "category": "O-Novel",
               "variant": "top10 vs sector"}, fn_o13a))

    def fn_o13b(ctx):
        c = ctx["close"]
        sec = pd.Series({t: SECTOR_MAP.get(t, "Other") for t in c.columns})
        r63 = c.pct_change(63, fill_method=None)
        me = pd.Series(c.index, index=c.index).groupby(
            [c.index.year, c.index.month]).transform("max") == c.index
        pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        for d in c.index[me]:
            r = r63.loc[:d].iloc[-1]
            rel = r - r.groupby(sec).transform("mean")
            row = pd.Series(0.0, index=c.columns)
            for s_, grp in rel.dropna().groupby(sec):
                row[grp.nlargest(2).index] = 1.0
            pos.loc[d] = row
        return pos.ffill().fillna(0)
    S.append(({"id": "O13", "name": "Sector-Relative Strength", "category": "O-Novel",
               "variant": "top2/sector"}, fn_o13b))

    # O14: Gap-down buying
    for g in [0.015, 0.025]:
        for gate in ["none", "mktup", "oversold"]:
            def fn(ctx, g=g, gate=gate):
                o, c = ctx["open"], ctx["close"]
                gap = o / c.shift(1) - 1
                entry = gap < -g
                if gate == "mktup":
                    bu = (ctx["benchmark"] > ctx["benchmark"].rolling(200).mean()
                          ).reindex(c.index).fillna(False)
                    entry = entry.mul(bu, axis=0).astype(bool)
                elif gate == "oversold":
                    entry = entry & (ta.rsi(c, 14) < 35)
                return hold_n(entry.fillna(False), 10)
            S.append(({"id": "O14", "name": "Gap-Down Buy", "category": "O-Novel",
                       "variant": f"gap{g*100:.1f}%|{gate}"}, fn))

    # O15: Narrow-range breakout (NR4/NR7)
    for nr in [4, 7]:
        for hp in [5, 10]:
            def fn(ctx, nr=nr, hp=hp):
                h, l, c = ctx["high"], ctx["low"], ctx["close"]
                rng = h - l
                is_nr = rng == rng.rolling(nr).min()
                brk = c > h.shift(1)
                return hold_n((is_nr.shift(1) & brk).fillna(False), hp)
            S.append(({"id": "O15", "name": "NR Breakout", "category": "O-Novel",
                       "variant": f"NR{nr} h{hp}"}, fn))

    # O16: Higher-high structure persistence
    for thr in [0.6, 0.7]:
        def fn(ctx, thr=thr):
            c, h = ctx["close"], ctx["high"]
            hh20 = h.rolling(20).max()
            rising = (hh20 > hh20.shift(1)).rolling(50).mean()
            return state(rising > thr, rising < thr - 0.15, c.index, c.columns)
        S.append(({"id": "O16", "name": "Higher-High Structure", "category": "O-Novel",
                   "variant": f"HH>{int(thr*100)}%"}, fn))

    # O17: Volume-weighted momentum
    for lb in [21, 63, 126]:
        def fn(ctx, lb=lb):
            c, v = ctx["close"], ctx["volume"]
            ret = c.pct_change(fill_method=None)
            vwm = (ret * v).rolling(lb).sum() / v.rolling(lb).sum()
            return monthly_topn(c, vwm, 10)
        S.append(({"id": "O17", "name": "Volume-Weighted Momentum", "category": "O-Novel",
                   "variant": f"lb{lb}"}, fn))

    # O18: Median + MAD channel breakout
    for n in [50, 100]:
        for k in [1.0, 2.0]:
            def fn(ctx, n=n, k=k):
                c = ctx["close"]
                med = c.rolling(n).median()
                sd = _robust_sd(c, n)
                return state(c > med + k * sd, c < med, c.index, c.columns)
            S.append(({"id": "O18", "name": "Median-MAD Channel", "category": "O-Novel",
                       "variant": f"n{n}k{k}"}, fn))

    # O19: Seasonally-adjusted momentum
    for lb in [126, 252]:
        def fn(ctx, lb=lb):
            c = ctx["close"]
            ret = c.pct_change(fill_method=None)
            mret = (1 + ret).groupby([c.index.year, c.index.month]).transform(
                lambda x: x)  # placeholder no-op to keep index
            # expanding mean of same-calendar-month daily returns, shifted 1yr
            month = pd.Series(c.index.month, index=c.index)
            seas = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
            for m in range(1, 13):
                mask = month == m
                seas.loc[mask] = ret[mask.to_numpy()].expanding().mean().shift(21).to_numpy()
            adj = ret.rolling(lb).mean() - seas
            return monthly_topn(c, adj, 10)
        S.append(({"id": "O19", "name": "Seasonal-Adj Momentum", "category": "O-Novel",
                   "variant": f"lb{lb}"}, fn))

    # O20: Robust z-score reversion
    for n in [20, 50]:
        for z in [2.0, 3.0]:
            def fn(ctx, n=n, z=z):
                c = ctx["close"]
                med = c.rolling(n).median()
                sd = _robust_sd(c, n)
                zs = (c - med) / sd
                return state(zs < -z, zs > 0, c.index, c.columns)
            S.append(({"id": "O20", "name": "Robust-Z Reversion", "category": "O-Novel",
                       "variant": f"n{n}z{z}"}, fn))

    # O21: Streak strategies
    for k in [3, 5]:
        def fn_r(ctx, k=k):
            c, v = ctx["close"], ctx["volume"]
            dn = (c < c.shift(1)).astype(int)
            streak = dn.rolling(k).sum() == k
            vs = v > 1.5 * v.rolling(20).mean()
            return hold_n((streak & vs).fillna(False), 10)
        S.append(({"id": "O21", "name": "Down-Streak Reversal", "category": "O-Novel",
                   "variant": f"{k}dn+vol h10"}, fn_r))
    for k in [5, 7]:
        def fn_c(ctx, k=k):
            c = ctx["close"]
            up = (c > c.shift(1)).astype(int)
            streak = up.rolling(k).sum() == k
            return hold_n(streak.fillna(False), 5)
        S.append(({"id": "O21", "name": "Up-Streak Continuation", "category": "O-Novel",
                   "variant": f"{k}up h5"}, fn_c))

    # O22: Quiet accumulation (flat price, rising OBV/CMF)
    for strict in [1, 2, 3]:
        def fn(ctx, strict=strict):
            h, l, c, v = ctx["high"], ctx["low"], ctx["close"], ctx["volume"]
            flat = c.pct_change(20, fill_method=None).abs() < 0.03
            obv_up = ta.obv(c, v).diff(20) > 0
            cond = flat & obv_up
            if strict >= 2:
                cond = cond & (ta.cmf(h, l, c, v, 20) > 0)
            if strict >= 3:
                cond = cond & (v.rolling(5).mean() > v.rolling(60).mean())
            return hold_n(cond.fillna(False), 21)
        S.append(({"id": "O22", "name": "Quiet Accumulation", "category": "O-Novel",
                   "variant": f"strict{strict}"}, fn))

    # O23: Autocorrelation auto-switch (trend rule vs reversion rule per stock)
    for thr in [0.0, 0.05, 0.10]:
        def fn(ctx, thr=thr):
            c = ctx["close"]
            ret = c.pct_change(fill_method=None)
            ac = ret.rolling(60).corr(ret.shift(1))
            trending = ac > thr
            reverting = ac < -thr
            mom_sig = (c > ta.ema(c, 20)).astype(float)
            rev_sig = hold_n((ta.rsi(c, 2) < 10).fillna(False), 5)
            out = mom_sig.where(trending, 0.0) + rev_sig.where(reverting, 0.0)
            return out.clip(0, 1)
        S.append(({"id": "O23", "name": "AC Auto-Switch", "category": "O-Novel",
                   "variant": f"|ac|>{thr}"}, fn))

    # O24: Close-location-value persistence
    for thr in [0.1, 0.2]:
        def fn(ctx, thr=thr):
            h, l, c = ctx["high"], ctx["low"], ctx["close"]
            clv = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
            m = clv.rolling(20).mean()
            return state(m > thr, m < 0, c.index, c.columns)
        S.append(({"id": "O24", "name": "CLV Persistence", "category": "O-Novel",
                   "variant": f"clv20>{thr}"}, fn))

    # O25: Momentum with crash protection
    for ddthr in [0.10, 0.15]:
        def fn(ctx, ddthr=ddthr):
            c = ctx["close"]
            bench = ctx["benchmark"]
            mom = c.pct_change(126, fill_method=None)
            pos = monthly_topn(c, mom, 10)
            bdd = (bench / bench.rolling(252).max() - 1).reindex(c.index).fillna(0)
            ok = (bdd > -ddthr).astype(float)
            return pos.mul(ok, axis=0)
        S.append(({"id": "O25", "name": "Momentum + Crash Guard", "category": "O-Novel",
                   "variant": f"exit@dd{int(ddthr*100)}%"}, fn))

    # O26: Sharpe momentum (return / vol rank)
    for lb in [63, 126, 252]:
        def fn(ctx, lb=lb):
            c = ctx["close"]
            ret = c.pct_change(fill_method=None)
            sh = ret.rolling(lb).mean() / ret.rolling(lb).std().replace(0, np.nan)
            return monthly_topn(c, sh, 10)
        S.append(({"id": "O26", "name": "Sharpe Momentum", "category": "O-Novel",
                   "variant": f"lb{lb}"}, fn))

    # O27: Trend age filter (only young golden crosses)
    for age in [63, 126]:
        def fn(ctx, age=age):
            c = ctx["close"]
            fast, slow = ta.sma(c, 50), ta.sma(c, 200)
            above = (fast > slow)
            grp = (above != above.shift(1)).cumsum()
            bars_in_state = above.groupby([grp[col] for col in [c.columns[0]]][0]).cumcount() \
                if False else None
            # vectorized: bars since cross via cumsum trick per column
            age_mat = above.astype(float).copy() * 0.0
            arr = above.to_numpy(bool)
            out = np.zeros_like(arr, dtype=float)
            cnt = np.zeros(arr.shape[1])
            for t in range(arr.shape[0]):
                cnt = np.where(arr[t], cnt + 1, 0)
                out[t] = cnt
            age_df = pd.DataFrame(out, index=c.index, columns=c.columns)
            return ((age_df > 0) & (age_df <= age)).astype(float)
        S.append(({"id": "O27", "name": "Young Trend Only", "category": "O-Novel",
                   "variant": f"age<{age}d"}, fn))

    # O28: Volatility risk premium timing (VIX vs realized)
    for q in [0.5, 0.7]:
        def fn(ctx, q=q):
            c = ctx["close"]
            vix = ctx["vix"]
            if vix is None:
                raise RuntimeError("no VIX")
            vix = vix.reindex(c.index).ffill()
            rv = ctx["benchmark"].pct_change().rolling(20).std().reindex(c.index) \
                * np.sqrt(252) * 100
            vrp = vix - rv
            rich = vrp > vrp.rolling(252).quantile(q)
            return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
                c.notna(), 0.0).mul(rich.fillna(False).astype(float), axis=0)
        S.append(({"id": "O28", "name": "VRP Timing", "category": "O-Novel",
                   "variant": f"VRP>q{int(q*100)}"}, fn))

    # O29: OU half-life gated reversion
    for hl_max in [10, 20]:
        def fn(ctx, hl_max=hl_max):
            c = ctx["close"]
            ret = c.pct_change(fill_method=None)
            phi = ret.rolling(126).corr(ret.shift(1))
            hl = -np.log(2) / np.log(phi.abs().clip(1e-6, 0.999))
            fast_rev = (phi < 0) & (hl < hl_max)
            z = ta.zscore(c, 20)
            entry = fast_rev & (z < -2)
            exit_ = z > 0
            return state(entry, exit_, c.index, c.columns)
        S.append(({"id": "O29", "name": "OU Half-Life Reversion", "category": "O-Novel",
                   "variant": f"HL<{hl_max}d"}, fn))

    # O30: Up-day fraction quality
    for thr in [0.60, 0.65]:
        def fn(ctx, thr=thr):
            c = ctx["close"]
            upfrac = (c > c.shift(1)).rolling(63).mean()
            vol = c.pct_change(fill_method=None).rolling(20).std()
            quiet = vol < vol.rolling(252).median()
            return state((upfrac > thr) & quiet, upfrac < 0.5, c.index, c.columns)
        S.append(({"id": "O30", "name": "Up-Day Fraction", "category": "O-Novel",
                   "variant": f">{int(thr*100)}%+quiet"}, fn))

    # O31: Relative volume trend + momentum
    for lb in [20, 60]:
        def fn(ctx, lb=lb):
            c, v = ctx["close"], ctx["volume"]
            rvol = v.rolling(lb).mean() / v.rolling(100).mean().replace(0, np.nan)
            mom = c.pct_change(63, fill_method=None)
            score = rvol.rank(axis=1, pct=True) + mom.rank(axis=1, pct=True)
            return monthly_topn(c, score.where(mom > 0), 10)
        S.append(({"id": "O31", "name": "Rising Volume + Momentum", "category": "O-Novel",
                   "variant": f"rvol{lb}"}, fn))

    # O33: Effortless rise (range contraction while price advances)
    for n in [20, 40]:
        def fn(ctx, n=n):
            h, l, c = ctx["high"], ctx["low"], ctx["close"]
            rng = ((h - l) / c).rolling(n).mean()
            contracting = rng < rng.shift(n)
            advancing = c > c.shift(n)
            return state(contracting & advancing, ~advancing, c.index, c.columns)
        S.append(({"id": "O33", "name": "Effortless Rise", "category": "O-Novel",
                   "variant": f"n{n}"}, fn))

    return S
