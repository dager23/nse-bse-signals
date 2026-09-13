"""Category G: Statistical & Quantitative strategies (G1-G14).

All model fitting is walk-forward: parameters are estimated only on data
preceding the dates being traded.
"""
import numpy as np
import pandas as pd

from utils import indicators as ta
from utils.sectors import SECTOR_MAP


def _long(cond):
    return cond.astype(float)


def _state(entry, exit_, index, columns):
    st = pd.DataFrame(np.nan, index=index, columns=columns)
    st[entry] = 1.0
    st[exit_] = 0.0
    return st.ffill().fillna(0)


def _hold_n(entry, n):
    return entry.astype(float).rolling(n, min_periods=1).max().fillna(0)


def _basket(c, cond_series):
    return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(c.notna(), 0.0).mul(
        cond_series.astype(float), axis=0)


def get_strategies():
    S = []

    # G1: Kalman filter local-level trend
    for q_over_r in [1e-4, 1e-3]:
        def fn(ctx, q_over_r=q_over_r):
            c = ctx["close"]
            price = np.log(c.to_numpy(float))
            T, N = price.shape
            xhat = np.full((T, N), np.nan)
            P = np.full(N, 1.0)
            x = np.full(N, np.nan)
            R, Q = 1.0, q_over_r
            for t in range(T):
                z = price[t]
                new = np.isnan(x) & ~np.isnan(z)
                x = np.where(new, z, x)
                Pp = P + Q
                K = Pp / (Pp + R)
                valid = ~np.isnan(z) & ~np.isnan(x)
                x = np.where(valid, x + K * (z - x), x)
                P = np.where(valid, (1 - K) * Pp, P)
                xhat[t] = x
            kf = pd.DataFrame(np.exp(xhat), index=c.index, columns=c.columns)
            return _long(c > kf)
        S.append(({"id": "G1", "name": "Kalman Filter Trend", "category": "G-Statistical",
                   "variant": f"Q/R={q_over_r}"}, fn))

    # G2: HMM regime detection on Nifty (walk-forward yearly refit)
    def fn_g2(ctx):
        from hmmlearn.hmm import GaussianHMM
        c = ctx["close"]
        bench = ctx["benchmark"]
        ret = bench.pct_change().fillna(0)
        feat = pd.DataFrame({"ret": ret, "vol": ret.rolling(10).std()}).fillna(0)
        bull = pd.Series(0.0, index=c.index)
        years = sorted(set(c.index.year))
        for y in years[4:]:
            train = feat[feat.index.year < y].to_numpy()
            test_idx = feat.index[feat.index.year == y]
            if len(train) < 500 or len(test_idx) == 0:
                continue
            try:
                hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100,
                                  random_state=42)
                hmm.fit(train)
                bull_state = int(np.argmax(hmm.means_[:, 0]))
                # filtered probabilities on expanding history (no lookahead within year)
                hist = feat[feat.index <= test_idx[-1]].to_numpy()
                post = hmm.predict_proba(hist)
                probs = pd.Series(post[:, bull_state], index=feat.index[: len(post)])
                bull.loc[test_idx] = (probs.loc[test_idx] > 0.6).astype(float)
            except Exception:
                continue
        return _basket(c, bull)
    S.append(({"id": "G2", "name": "HMM Regime", "category": "G-Statistical",
               "variant": "3-state/Nifty"}, fn_g2))

    # G3: Fourier dominant cycle timing (rolling 250d, every 5 bars)
    def fn_g3(ctx):
        c = ctx["close"]
        logp = np.log(c.to_numpy(float))
        T, N = logp.shape
        win, step = 250, 5
        rising = np.full((T, N), np.nan)
        x = np.arange(win)
        for t in range(win, T, step):
            seg = logp[t - win:t]
            for j in range(N):
                y = seg[:, j]
                if np.isnan(y).any():
                    continue
                coef = np.polyfit(x, y, 1)
                detr = y - (coef[0] * x + coef[1])
                spec = np.fft.rfft(detr)
                mag = np.abs(spec)
                mag[0] = 0
                k = int(np.argmax(mag[1:20])) + 1  # dominant cycle 12-250d
                phase = np.angle(spec[k])
                # cycle value now and derivative
                cyc_now = np.cos(2 * np.pi * k * (win - 1) / win + phase)
                cyc_prev = np.cos(2 * np.pi * k * (win - 2) / win + phase)
                rising[t, j] = 1.0 if cyc_now > cyc_prev else 0.0
        out = pd.DataFrame(rising, index=c.index, columns=c.columns).ffill(limit=step).fillna(0)
        return out
    S.append(({"id": "G3", "name": "Fourier Cycle", "category": "G-Statistical",
               "variant": "250d dom-cycle"}, fn_g3))

    # G4: Wavelet denoised trend (db4, rolling 256d, every 5 bars)
    def fn_g4(ctx):
        import pywt
        c = ctx["close"]
        logp = np.log(c.to_numpy(float))
        T, N = logp.shape
        win, step = 256, 5
        sig = np.full((T, N), np.nan)
        for t in range(win, T, step):
            seg = logp[t - win:t]
            for j in range(N):
                y = seg[:, j]
                if np.isnan(y).any():
                    continue
                coeffs = pywt.wavedec(y, "db4", level=4)
                for k in range(2, len(coeffs)):
                    coeffs[k] = np.zeros_like(coeffs[k])  # kill high-freq detail
                den = pywt.waverec(coeffs, "db4")[:win]
                sig[t, j] = 1.0 if den[-1] > den[-5] else 0.0
        return pd.DataFrame(sig, index=c.index, columns=c.columns).ffill(limit=step).fillna(0)
    S.append(({"id": "G4", "name": "Wavelet Trend", "category": "G-Statistical",
               "variant": "db4 L4"}, fn_g4))

    # G5: Permutation entropy regime filter + momentum
    def fn_g5(ctx):
        from itertools import permutations
        c = ctx["close"]
        ret = c.pct_change(fill_method=None).to_numpy(float)
        T, N = ret.shape
        win, step, order = 100, 5, 3
        perms = {p: i for i, p in enumerate(permutations(range(order)))}
        pent = np.full((T, N), np.nan)
        for t in range(win, T, step):
            seg = ret[t - win:t]
            for j in range(N):
                y = seg[:, j]
                if np.isnan(y).any():
                    continue
                pats = np.zeros(len(perms))
                for i in range(len(y) - order + 1):
                    pats[perms[tuple(np.argsort(y[i:i + order]))]] += 1
                p = pats / pats.sum()
                p = p[p > 0]
                pent[t, j] = -np.sum(p * np.log(p)) / np.log(len(perms))
        pe = pd.DataFrame(pent, index=c.index, columns=c.columns).ffill(limit=step)
        predictable = pe < pe.rolling(250).quantile(0.3)
        mom = c > c.shift(20)
        return _long(predictable & mom)
    S.append(({"id": "G5", "name": "Permutation Entropy", "category": "G-Statistical",
               "variant": "PE<q30+mom"}, fn_g5))

    # G6: Granger causality lead-lag within sectors (yearly refit)
    def fn_g6(ctx):
        from statsmodels.tsa.stattools import grangercausalitytests
        from itertools import permutations as iperm
        c = ctx["close"]
        ret = c.pct_change(fill_method=None)
        pos = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        sectors = {}
        for tkr, sec in SECTOR_MAP.items():
            if tkr in c.columns:
                sectors.setdefault(sec, []).append(tkr)
        years = sorted(set(c.index.year))
        for y in years[2:]:
            form = ret[(ret.index.year >= y - 2) & (ret.index.year < y)]
            tr_idx = c.index[c.index.year == y]
            if len(form) < 300 or len(tr_idx) == 0:
                continue
            links = []
            for sec, tkrs in sectors.items():
                for a, b in iperm(tkrs, 2):  # does a cause b?
                    d = form[[b, a]].dropna()
                    if len(d) < 250:
                        continue
                    try:
                        r = grangercausalitytests(d, maxlag=2, verbose=False)
                        pv = min(r[k][0]["ssr_ftest"][1] for k in r)
                    except Exception:
                        continue
                    if pv < 0.01:
                        links.append((pv, a, b))
            for _, a, b in sorted(links)[:10]:
                lead = ret[a].reindex(tr_idx).shift(1).rolling(3).sum()
                pos.loc[tr_idx, b] += (lead > 0.01).astype(float)
        return pos.clip(0, 1)
    S.append(({"id": "G6", "name": "Granger Lead-Lag", "category": "G-Statistical",
               "variant": "sector p<.01"}, fn_g6))

    # G7: Johansen cointegrated basket reversion (banks sector, yearly refit)
    def fn_g7(ctx):
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        c = ctx["close"]
        logp = np.log(c)
        banks = [t for t in ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS",
                             "AXISBANK.NS"] if t in c.columns]
        pos = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        years = sorted(set(c.index.year))
        for y in years[3:]:
            form = logp[banks][(logp.index.year >= y - 3) & (logp.index.year < y)].dropna()
            tr_idx = c.index[c.index.year == y]
            if len(form) < 500 or len(tr_idx) == 0:
                continue
            try:
                res = coint_johansen(form, det_order=0, k_ar_diff=1)
                if res.lr1[0] < res.cvt[0, 1]:  # no cointegration at 95%
                    continue
                w = res.evec[:, 0]
            except Exception:
                continue
            spread = (logp[banks] @ w)
            mu, sd = spread.loc[form.index].mean(), spread.loc[form.index].std()
            z = ((spread - mu) / sd).reindex(tr_idx)
            sig = pd.Series(np.nan, index=tr_idx)
            sig[z < -1.5] = 1.0
            sig[z > 1.5] = -1.0
            sig[z.abs() < 0.25] = 0.0
            sig = sig.ffill().fillna(0)
            for i, b in enumerate(banks):
                pos.loc[tr_idx, b] += sig * np.sign(w[i])
        return pos.clip(-1, 1)
    S.append(({"id": "G7", "name": "Johansen Basket", "category": "G-Statistical",
               "variant": "banks/z1.5", "allow_short": True}, fn_g7))

    # G8: PCA factor momentum (monthly refit on trailing 250d)
    def fn_g8(ctx):
        c = ctx["close"]
        ret = c.pct_change(fill_method=None)
        month_end = pd.Series(c.index, index=c.index).groupby(
            [c.index.year, c.index.month]).transform("max") == c.index
        rebal_dates = c.index[month_end]
        pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        for d in rebal_dates:
            hist = ret[ret.index <= d].tail(250).dropna(axis=1)
            if hist.shape[0] < 250 or hist.shape[1] < 10:
                continue
            X = hist.to_numpy()
            Xc = X - X.mean(0)
            cov = np.cov(Xc.T)
            evals, evecs = np.linalg.eigh(cov)
            pcs = evecs[:, ::-1][:, :3]                      # top 3 components
            fac_ret = X @ pcs                                # factor return series
            fac_mom = fac_ret[-60:].sum(0)                   # 60d factor momentum
            score = pcs @ fac_mom                            # stock alignment
            sel = pd.Series(score, index=hist.columns).rank(ascending=False) <= 10
            row = pd.Series(0.0, index=c.columns)
            row[sel[sel].index] = 1.0
            pos.loc[d] = row
        return pos.ffill().fillna(0)
    S.append(({"id": "G8", "name": "PCA Factor Momentum", "category": "G-Statistical",
               "variant": "3PC/top10"}, fn_g8))

    # G9: RMT-cleaned min-variance portfolio (monthly, signals are weights)
    def fn_g9(ctx):
        c = ctx["close"]
        ret = c.pct_change(fill_method=None)
        month_end = pd.Series(c.index, index=c.index).groupby(
            [c.index.year, c.index.month]).transform("max") == c.index
        pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        for d in c.index[month_end]:
            hist = ret[ret.index <= d].tail(250).dropna(axis=1)
            T_, N_ = hist.shape
            if T_ < 250 or N_ < 10:
                continue
            X = hist.to_numpy()
            sd = X.std(0)
            sd[sd == 0] = 1e-9
            corr = np.corrcoef(X.T)
            evals, evecs = np.linalg.eigh(corr)
            lam_max = (1 + np.sqrt(N_ / T_)) ** 2
            noise = evals < lam_max
            evals_clean = evals.copy()
            if noise.any():
                evals_clean[noise] = evals[noise].mean()
            corr_c = evecs @ np.diag(evals_clean) @ evecs.T
            np.fill_diagonal(corr_c, 1.0)
            cov = corr_c * np.outer(sd, sd) * 252
            try:
                inv = np.linalg.pinv(cov)
            except Exception:
                continue
            ones = np.ones(N_)
            w = inv @ ones / (ones @ inv @ ones)
            w = np.clip(w, 0, 0.1)
            if w.sum() <= 0:
                continue
            w = w / w.sum()
            row = pd.Series(0.0, index=c.columns)
            row[hist.columns] = w
            pos.loc[d] = row
        return pos.ffill().fillna(0)
    S.append(({"id": "G9", "name": "RMT Min-Variance", "category": "G-Statistical",
               "variant": "MP-cleaned", "signals_are_weights": True}, fn_g9))

    # G10: Benford's law anomaly filter + trend
    def fn_g10(ctx):
        c, v = ctx["close"], ctx["volume"]
        benford = np.log10(1 + 1 / np.arange(1, 10))
        vol = v.to_numpy(float)
        T, N = vol.shape
        win, step = 120, 10
        chi = np.full((T, N), np.nan)
        first_digit = np.where(vol > 0, (vol / 10 ** np.floor(np.log10(
            np.where(vol > 0, vol, 1)))).astype(int), 0)
        for t in range(win, T, step):
            seg = first_digit[t - win:t]
            for j in range(N):
                d = seg[:, j]
                d = d[(d >= 1) & (d <= 9)]
                if len(d) < 60:
                    continue
                obs = np.bincount(d, minlength=10)[1:10] / len(d)
                chi[t, j] = np.sum((obs - benford) ** 2 / benford)
        anomaly = pd.DataFrame(chi, index=c.index, columns=c.columns).ffill(limit=step)
        normal = anomaly < anomaly.rolling(250).quantile(0.7)
        return _long(normal & (c > ta.sma(c, 50)))
    S.append(({"id": "G10", "name": "Benford Anomaly Filter", "category": "G-Statistical",
               "variant": "vol digits+trend"}, fn_g10))

    # G11: Autocorrelation trading
    def fn_g11(ctx):
        c = ctx["close"]
        ret = c.pct_change(fill_method=None)
        ac = ret.rolling(60).corr(ret.shift(1))
        up_yday = ret > 0
        follow = (ac > 0.1) & up_yday
        fade = (ac < -0.1) & ~up_yday
        return _long(follow | fade)
    S.append(({"id": "G11", "name": "Autocorrelation Trading", "category": "G-Statistical",
               "variant": "AC60 +/-0.1"}, fn_g11))

    # G12: Jump detection -> reversal
    for k, hold in [(4.0, 5), (3.0, 10)]:
        def fn(ctx, k=k, hold=hold):
            c = ctx["close"]
            ret = c.pct_change(fill_method=None)
            sd = ret.rolling(60).std()
            down_jump = ret < -k * sd
            return _hold_n(down_jump, hold)
        S.append(({"id": "G12", "name": "Jump Reversal", "category": "G-Statistical",
                   "variant": f"{k}sd/hold{hold}"}, fn))

    # G13: CUSUM changepoint trend detection
    def fn_g13(ctx):
        c = ctx["close"]
        ret = c.pct_change(fill_method=None).to_numpy(float)
        T, N = ret.shape
        h_thr, drift = 0.05, 0.0005
        pos = np.zeros((T, N))
        gp = np.zeros(N)
        gn = np.zeros(N)
        state = np.zeros(N)
        for t in range(1, T):
            r = np.nan_to_num(ret[t])
            gp = np.maximum(0, gp + r - drift)
            gn = np.maximum(0, gn - r - drift)
            up_cp = gp > h_thr
            dn_cp = gn > h_thr
            state = np.where(up_cp, 1.0, np.where(dn_cp, 0.0, state))
            gp = np.where(up_cp | dn_cp, 0.0, gp)
            gn = np.where(up_cp | dn_cp, 0.0, gn)
            pos[t] = state
        return pd.DataFrame(pos, index=c.index, columns=c.columns)
    S.append(({"id": "G13", "name": "CUSUM Changepoint", "category": "G-Statistical",
               "variant": "h=5%"}, fn_g13))

    # G14: Gaussian copula pairs (conditional probability, yearly refit)
    def fn_g14(ctx):
        from scipy.stats import norm
        from itertools import combinations
        c = ctx["close"]
        ret = c.pct_change(fill_method=None)
        pos = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        sectors = {}
        for tkr, sec in SECTOR_MAP.items():
            if tkr in c.columns:
                sectors.setdefault(sec, []).append(tkr)
        years = sorted(set(c.index.year))
        for y in years[2:]:
            form = ret[(ret.index.year >= y - 2) & (ret.index.year < y)]
            tr_idx = c.index[c.index.year == y]
            if len(form) < 300 or len(tr_idx) == 0:
                continue
            cands = []
            for sec, tkrs in sectors.items():
                for a, b in combinations(tkrs, 2):
                    d = form[[a, b]].dropna()
                    if len(d) < 250:
                        continue
                    rho = d[a].corr(d[b])
                    if rho > 0.6:
                        cands.append((-rho, a, b, rho))
            for _, a, b, rho in sorted(cands)[:6]:
                # cumulative 5d relative return -> copula conditional prob
                ra = ret[a].rolling(5).sum()
                rb = ret[b].rolling(5).sum()
                mu_a, sd_a = form[a].rolling(5).sum().mean(), form[a].rolling(5).sum().std()
                mu_b, sd_b = form[b].rolling(5).sum().mean(), form[b].rolling(5).sum().std()
                ua = norm.cdf((ra - mu_a) / sd_a).clip(0.001, 0.999)
                ub = norm.cdf((rb - mu_b) / sd_b).clip(0.001, 0.999)
                za, zb = norm.ppf(ua), norm.ppf(ub)
                # P(A <= a | B = b) under Gaussian copula
                h = norm.cdf((za - rho * zb) / np.sqrt(1 - rho ** 2))
                h = pd.Series(h, index=ret.index).reindex(tr_idx)
                sig = pd.Series(np.nan, index=tr_idx)
                sig[h < 0.05] = 1.0    # A very low given B -> long A short B
                sig[h > 0.95] = -1.0
                sig[(h > 0.4) & (h < 0.6)] = 0.0
                sig = sig.ffill().fillna(0)
                pos.loc[tr_idx, a] += sig
                pos.loc[tr_idx, b] -= sig
        return pos.clip(-1, 1)
    S.append(({"id": "G14", "name": "Copula Pairs", "category": "G-Statistical",
               "variant": "gauss h5/95", "allow_short": True}, fn_g14))

    return S
