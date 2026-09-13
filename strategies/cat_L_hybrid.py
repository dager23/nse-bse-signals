"""Category L: Hybrid & Composite strategies (L1-L8)."""
import numpy as np
import pandas as pd

from utils import indicators as ta
from utils.sectors import SECTOR_MAP


def _basket(c, cond_series):
    return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
        c.notna(), 0.0).mul(cond_series.astype(float).reindex(c.index).fillna(0), axis=0)


def _state(entry, exit_, index, columns):
    st = pd.DataFrame(np.nan, index=index, columns=columns)
    st[entry] = 1.0
    st[exit_] = 0.0
    return st.ffill().fillna(0)


def get_strategies():
    S = []

    # L1: ADX trend-strength gate + MACD crossover
    def fn_l1(ctx):
        h, l, c = ctx["high"], ctx["low"], ctx["close"]
        adx, dip, dim = ta.adx(h, l, c, 14)
        macd_line, sig_line, _ = ta.macd(c)
        entry = (adx > 25) & (macd_line > sig_line) & \
                (macd_line.shift(1) <= sig_line.shift(1)) & (dip > dim)
        exit_ = macd_line < sig_line
        return _state(entry, exit_, c.index, c.columns)
    S.append(({"id": "L1", "name": "ADX + MACD Combo", "category": "L-Hybrid",
               "variant": "ADX25 gate"}, fn_l1))

    # L2: Volatility-compression mean reversion (low ATR pctl + RSI extreme)
    def fn_l2(ctx):
        h, l, c = ctx["high"], ctx["low"], ctx["close"]
        atr_pct = ta.atr(h, l, c, 14) / c
        compressed = atr_pct < atr_pct.rolling(252).quantile(0.3)
        r = ta.rsi(c, 14)
        entry = compressed & (r < 30)
        exit_ = r > 55
        return _state(entry, exit_, c.index, c.columns)
    S.append(({"id": "L2", "name": "LowVol + RSI Reversion", "category": "L-Hybrid",
               "variant": "ATR<q30 RSI30/55"}, fn_l2))

    # L3: Donchian breakout confirmed by volume spike
    def fn_l3(ctx):
        h, l, c, v = ctx["high"], ctx["low"], ctx["close"], ctx["volume"]
        upper = h.rolling(55).max().shift(1)
        lower = l.rolling(20).min().shift(1)
        vspike = v > 1.5 * v.rolling(20).mean()
        entry = (c > upper) & vspike
        exit_ = c < lower
        return _state(entry, exit_, c.index, c.columns)
    S.append(({"id": "L3", "name": "Donchian + Volume", "category": "L-Hybrid",
               "variant": "55/20 + 1.5xVol"}, fn_l3))

    # L4: Multi-timeframe: weekly trend, daily pullback entry
    def fn_l4(ctx):
        c = ctx["close"]
        wk = c.resample("W-FRI").last()
        wk_trend = (ta.ema(wk, 10) > ta.ema(wk, 30)).shift(1)  # completed weeks only
        wk_daily = wk_trend.reindex(c.index, method="ffill").fillna(False)
        r = ta.rsi(c, 5)
        entry = wk_daily & (r < 40)
        exit_ = (~wk_daily) | (r > 70)
        return _state(entry, exit_, c.index, c.columns)
    S.append(({"id": "L4", "name": "Multi-Timeframe", "category": "L-Hybrid",
               "variant": "wkEMA10/30 + RSI5<40"}, fn_l4))

    # L5: ML ensemble (logistic + XGB + LGBM average probability, walk-forward)
    def fn_l5(ctx):
        from utils.ml_features import walk_forward_positions
        def fit_predict(Xtr, ytr, Xte):
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from xgboost import XGBClassifier
            from lightgbm import LGBMClassifier
            probs = []
            m1 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500))
            m1.fit(Xtr, ytr)
            probs.append(m1.predict_proba(Xte)[:, 1])
            m2 = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                               subsample=0.8, tree_method="hist", eval_metric="logloss",
                               random_state=42, n_jobs=-1)
            m2.fit(Xtr, ytr)
            probs.append(m2.predict_proba(Xte)[:, 1])
            m3 = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                                subsample=0.8, random_state=42, n_jobs=-1, verbose=-1)
            m3.fit(Xtr, ytr)
            probs.append(m3.predict_proba(Xte)[:, 1])
            return np.mean(probs, axis=0)
        return walk_forward_positions(ctx, fit_predict, prob_thr=0.55)
    S.append(({"id": "L5", "name": "ML Ensemble", "category": "L-Hybrid",
               "variant": "LR+XGB+LGBM p>.55"}, fn_l5))

    # L6: Regime-adaptive (HMM bull -> momentum; sideways -> mean reversion; bear -> cash)
    def fn_l6(ctx):
        from hmmlearn.hmm import GaussianHMM
        c = ctx["close"]
        bench = ctx["benchmark"]
        ret = bench.pct_change().fillna(0)
        feat = pd.DataFrame({"ret": ret, "vol": ret.rolling(10).std()}).fillna(0)
        regime = pd.Series(np.nan, index=c.index)  # 1 bull, 0 side, -1 bear
        years = sorted(set(c.index.year))
        for y in years[4:]:
            train = feat[feat.index.year < y].to_numpy()
            test_idx = feat.index[feat.index.year == y]
            if len(train) < 500 or len(test_idx) == 0:
                continue
            try:
                hmm = GaussianHMM(n_components=3, covariance_type="diag",
                                  n_iter=100, random_state=42)
                hmm.fit(train)
                order = np.argsort(hmm.means_[:, 0])  # bear, side, bull
                hist = feat[feat.index <= test_idx[-1]].to_numpy()
                states = pd.Series(hmm.predict(hist), index=feat.index[: len(hist)])
                mapping = {order[0]: -1, order[1]: 0, order[2]: 1}
                regime.loc[test_idx] = states.loc[test_idx].map(mapping)
            except Exception:
                continue
        regime = regime.ffill().fillna(0)
        # momentum sleeve: top-10 6m momentum; reversion sleeve: RSI2 < 10
        mom_rank = c.pct_change(126, fill_method=None).rank(axis=1, ascending=False)
        mom_pos = (mom_rank <= 10).astype(float)
        rev_pos = (ta.rsi(c, 2) < 10).astype(float)
        bull = (regime == 1).astype(float)
        side = (regime == 0).astype(float)
        return mom_pos.mul(bull, axis=0) + rev_pos.mul(side, axis=0)
    S.append(({"id": "L6", "name": "Regime-Adaptive (HMM)", "category": "L-Hybrid",
               "variant": "bull-mom/side-rev"}, fn_l6))

    # L7: Sector rotation + stock momentum
    def fn_l7(ctx):
        c = ctx["close"]
        ret6 = c.pct_change(126, fill_method=None)
        ret3 = c.pct_change(63, fill_method=None)
        sec = pd.Series({t: SECTOR_MAP.get(t, "Other") for t in c.columns})
        me = pd.Series(c.index, index=c.index).groupby(
            [c.index.year, c.index.month]).transform("max") == c.index
        pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        for d in c.index[me]:
            r6 = ret6.loc[:d].iloc[-1]
            r3 = ret3.loc[:d].iloc[-1]
            if r6.isna().all():
                continue
            sec_mom = r6.groupby(sec).mean().dropna()
            if len(sec_mom) < 4:
                continue
            top_secs = set(sec_mom.nlargest(3).index)
            row = pd.Series(0.0, index=c.columns)
            for s_ in top_secs:
                members = [t for t in c.columns if sec[t] == s_ and not np.isnan(r3.get(t, np.nan))]
                best = pd.Series({t: r3[t] for t in members}).nlargest(2).index
                row[list(best)] = 1.0
            pos.loc[d] = row
        return pos.ffill().fillna(0)
    S.append(({"id": "L7", "name": "Sector Rotation + Momentum", "category": "L-Hybrid",
               "variant": "top3sec x top2stk"}, fn_l7))

    # L8: Risk-on/off exposure switch (VIX pctl + breadth + index trend)
    def fn_l8(ctx):
        c = ctx["close"]
        bench = ctx["benchmark"]
        vix = ctx["vix"]
        score = pd.Series(0.0, index=c.index)
        trend = (bench > bench.rolling(200).mean()).reindex(c.index).fillna(False)
        score += trend.astype(float)
        breadth = (c > ta.sma(c, 200)).sum(axis=1) / c.notna().sum(axis=1).clip(lower=1)
        score += (breadth > 0.5).astype(float)
        if vix is not None:
            vp = vix.reindex(c.index).ffill().rolling(500, min_periods=250).rank(pct=True)
            score += (vp < 0.7).fillna(0).astype(float)
        exposure = (score / 3.0).clip(0, 1)
        n_alive = c.notna().sum(axis=1).clip(lower=1)
        w = pd.DataFrame(1.0, index=c.index, columns=c.columns).where(c.notna(), 0.0)
        w = w.div(n_alive, axis=0).mul(exposure, axis=0)
        return w
    S.append(({"id": "L8", "name": "Risk-On/Off Switch", "category": "L-Hybrid",
               "variant": "VIX+breadth+trend", "signals_are_weights": True}, fn_l8))

    return S
