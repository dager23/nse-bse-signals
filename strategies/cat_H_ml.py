"""Category H: Machine Learning strategies (H1-H13).

All classifiers use walk-forward validation: expanding training window,
5-day purge gap, test on the following calendar year.  No test data ever
enters training.
"""
import numpy as np
import pandas as pd

from utils import indicators as ta
from utils.ml_features import build_panel, walk_forward_positions


def _scaled(model):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), model)


def _clf_fn(make_model, use_decision=False, max_train=None, prob_thr=0.55):
    def fn(ctx):
        def fit_predict(Xtr, ytr, Xte):
            m = make_model()
            m.fit(Xtr, ytr)
            if use_decision:
                d = m.decision_function(Xte)
                return 1 / (1 + np.exp(-d))  # squash margin to pseudo-prob
            return m.predict_proba(Xte)[:, 1]
        return walk_forward_positions(ctx, fit_predict, max_train=max_train,
                                      prob_thr=prob_thr)
    return fn


def get_strategies():
    S = []

    # H1: Logistic regression
    def mk_h1():
        from sklearn.linear_model import LogisticRegression
        return _scaled(LogisticRegression(C=1.0, max_iter=500))
    S.append(({"id": "H1", "name": "Logistic Regression", "category": "H-ML",
               "variant": "walkfwd p>.55"}, _clf_fn(mk_h1)))

    # H2: Random forest
    def mk_h2():
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=120, max_depth=8,
                                      min_samples_leaf=200, n_jobs=-1, random_state=42)
    S.append(({"id": "H2", "name": "Random Forest", "category": "H-ML",
               "variant": "120tree d8"}, _clf_fn(mk_h2, max_train=250_000)))

    # H3: XGBoost and LightGBM
    def mk_h3a():
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                             eval_metric="logloss", random_state=42, n_jobs=-1)
    S.append(({"id": "H3", "name": "XGBoost", "category": "H-ML",
               "variant": "200x4 lr.05"}, _clf_fn(mk_h3a)))

    def mk_h3b():
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              n_jobs=-1, verbose=-1)
    S.append(({"id": "H3", "name": "LightGBM", "category": "H-ML",
               "variant": "LGBM 300x31"}, _clf_fn(mk_h3b)))

    # H3 cross-sectional variant: daily top-quintile by XGB probability
    def fn_h3c(ctx):
        from xgboost import XGBClassifier
        X, y, _ = build_panel(ctx)
        c = ctx["close"]
        dates = X.index.get_level_values(0)
        pos = pd.DataFrame(0.0, index=c.index, columns=c.columns)
        for yy in sorted({d for d in dates.year if d >= 2012}):
            test_mask = dates.year == yy
            train_mask = dates < pd.Timestamp(f"{yy}-01-01") - pd.Timedelta(days=5)
            if train_mask.sum() < 5000 or test_mask.sum() == 0:
                continue
            m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              subsample=0.8, tree_method="hist",
                              eval_metric="logloss", random_state=42, n_jobs=-1)
            m.fit(X[train_mask], y[train_mask])
            prob = pd.Series(m.predict_proba(X[test_mask])[:, 1], index=X[test_mask].index)
            wide = prob.unstack()
            ranks = wide.rank(axis=1, ascending=False)
            sel = (ranks <= 10).astype(float)
            pos.loc[wide.index, wide.columns] = sel.fillna(0.0)
        return pos
    S.append(({"id": "H3", "name": "XGBoost", "category": "H-ML",
               "variant": "XS top10 daily"}, fn_h3c))

    # H4: Linear SVM (margin squashed)
    def mk_h4():
        from sklearn.svm import LinearSVC
        return _scaled(LinearSVC(C=0.5, dual=False, max_iter=3000))
    S.append(({"id": "H4", "name": "SVM (linear)", "category": "H-ML",
               "variant": "C=0.5"}, _clf_fn(mk_h4, use_decision=True, max_train=150_000)))

    # H5: k-NN
    def mk_h5():
        from sklearn.neighbors import KNeighborsClassifier
        return _scaled(KNeighborsClassifier(n_neighbors=100, n_jobs=-1))
    S.append(({"id": "H5", "name": "k-NN", "category": "H-ML",
               "variant": "k=100"}, _clf_fn(mk_h5, max_train=60_000)))

    # H6: Gaussian naive Bayes
    def mk_h6():
        from sklearn.naive_bayes import GaussianNB
        return _scaled(GaussianNB())
    S.append(({"id": "H6", "name": "Naive Bayes", "category": "H-ML",
               "variant": "gaussian"}, _clf_fn(mk_h6)))

    # H7: Pruned decision tree
    def mk_h7():
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=500,
                                      ccp_alpha=1e-5, random_state=42)
    S.append(({"id": "H7", "name": "Decision Tree", "category": "H-ML",
               "variant": "d5 pruned"}, _clf_fn(mk_h7)))

    # H8: AdaBoost
    def mk_h8():
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2),
                                  n_estimators=100, learning_rate=0.5, random_state=42)
    S.append(({"id": "H8", "name": "AdaBoost", "category": "H-ML",
               "variant": "100 stumps"}, _clf_fn(mk_h8, max_train=100_000)))

    # H9: Gaussian process regression on monthly index returns
    def fn_h9(ctx):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
        c = ctx["close"]
        bench = ctx["benchmark"]
        m_ret = bench.resample("ME").last().pct_change()
        lags = pd.DataFrame({f"lag{i}": m_ret.shift(i) for i in range(1, 7)})
        lags["vol6"] = m_ret.rolling(6).std()
        data = pd.concat([lags, m_ret.rename("y")], axis=1).dropna()
        pred = pd.Series(np.nan, index=data.index)
        for i in range(36, len(data)):
            Xtr = data.iloc[:i, :-1].to_numpy()
            ytr = data.iloc[:i]["y"].to_numpy()
            kern = ConstantKernel() * RBF() + WhiteKernel()
            try:
                g = GaussianProcessRegressor(kernel=kern, normalize_y=True,
                                             n_restarts_optimizer=0, random_state=42)
                g.fit(Xtr, ytr)
                pred.iloc[i] = g.predict(data.iloc[[i], :-1].to_numpy())[0]
            except Exception:
                continue
        # month-end signal applied to next month
        long_month = (pred > 0).astype(float)
        sig_daily = long_month.reindex(c.index, method="ffill").fillna(0)
        return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
            c.notna(), 0.0).mul(sig_daily, axis=0)
    S.append(({"id": "H9", "name": "Gaussian Process (monthly)", "category": "H-ML",
               "variant": "RBF idx"}, fn_h9))

    # H10: Isolation forest anomaly gate + trend
    def fn_h10(ctx):
        from sklearn.ensemble import IsolationForest
        c = ctx["close"]
        bench = ctx["benchmark"]
        ret = bench.pct_change()
        feat = pd.DataFrame({"ret": ret, "vol": ret.rolling(20).std(),
                             "rng": ret.rolling(5).max() - ret.rolling(5).min()}).dropna()
        score = pd.Series(np.nan, index=feat.index)
        for i in range(500, len(feat), 21):
            tr = feat.iloc[i - 500:i]
            m = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
            m.fit(tr)
            end = min(i + 21, len(feat))
            score.iloc[i:end] = m.decision_function(feat.iloc[i:end])
        normal = (score > 0).reindex(c.index).ffill().fillna(False)
        uptrend = (bench > bench.rolling(200).mean()).reindex(c.index).fillna(False)
        gate = (normal & uptrend).astype(float)
        return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
            c.notna(), 0.0).mul(gate, axis=0)
    S.append(({"id": "H10", "name": "Isolation Forest Gate", "category": "H-ML",
               "variant": "5% contam"}, fn_h10))

    # H11: DBSCAN regime clustering
    def fn_h11(ctx):
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler
        c = ctx["close"]
        bench = ctx["benchmark"]
        ret = bench.pct_change()
        feat = pd.DataFrame({"mom": ret.rolling(20).sum(),
                             "vol": ret.rolling(20).std()}).dropna()
        sig = pd.Series(0.0, index=feat.index)
        for i in range(500, len(feat), 21):
            tr = feat.iloc[i - 500:i]
            sc = StandardScaler().fit(tr)
            lab = DBSCAN(eps=0.4, min_samples=15).fit_predict(sc.transform(tr))
            fwd = ret.shift(-1).reindex(tr.index)  # within training only
            good = set()
            for k in set(lab):
                if k >= 0 and fwd[lab == k].mean() > 0:
                    good.add(k)
            cur = sc.transform(feat.iloc[i:min(i + 21, len(feat))])
            # assign to nearest training point's cluster
            from scipy.spatial import cKDTree
            tree = cKDTree(sc.transform(tr))
            _, nn = tree.query(cur)
            assigned = lab[nn]
            end = min(i + 21, len(feat))
            sig.iloc[i:end] = [1.0 if a in good else 0.0 for a in assigned]
        gate = sig.reindex(c.index).ffill().fillna(0)
        return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
            c.notna(), 0.0).mul(gate, axis=0)
    S.append(({"id": "H11", "name": "DBSCAN Regimes", "category": "H-ML",
               "variant": "eps.4"}, fn_h11))

    # H12: K-Means regime clustering
    def fn_h12(ctx):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        c = ctx["close"]
        bench = ctx["benchmark"]
        ret = bench.pct_change()
        feat = pd.DataFrame({"mom": ret.rolling(20).sum(),
                             "vol": ret.rolling(20).std()}).dropna()
        sig = pd.Series(0.0, index=feat.index)
        for i in range(500, len(feat), 21):
            tr = feat.iloc[i - 500:i]
            sc = StandardScaler().fit(tr)
            km = KMeans(n_clusters=3, n_init=5, random_state=42).fit(sc.transform(tr))
            fwd = ret.shift(-1).reindex(tr.index)
            lab = km.labels_
            good = {k for k in range(3) if fwd[lab == k].mean() > 0}
            end = min(i + 21, len(feat))
            assigned = km.predict(sc.transform(feat.iloc[i:end]))
            sig.iloc[i:end] = [1.0 if a in good else 0.0 for a in assigned]
        gate = sig.reindex(c.index).ffill().fillna(0)
        return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
            c.notna(), 0.0).mul(gate, axis=0)
    S.append(({"id": "H12", "name": "K-Means Regimes", "category": "H-ML",
               "variant": "k=3"}, fn_h12))

    # H13: Tabular Q-learning on index state
    def fn_h13(ctx):
        c = ctx["close"]
        bench = ctx["benchmark"]
        ret = bench.pct_change().fillna(0)
        mom = (bench > bench.rolling(50).mean()).astype(int)
        r = ta.rsi(bench.to_frame(), 14).iloc[:, 0]
        rsi_b = pd.cut(r, [0, 35, 65, 100], labels=False).fillna(1).astype(int)
        vol = ret.rolling(20).std()
        vol_b = (vol > vol.rolling(250).median()).astype(int)
        state = (mom * 6 + rsi_b * 2 + vol_b).astype(int)  # 12 states
        actions = 2  # 0 flat, 1 long
        gamma, alpha_lr, cost = 0.9, 0.1, 0.0015
        sig = pd.Series(0.0, index=bench.index)
        years = sorted(set(bench.index.year))
        for yy in [a for a in years if a >= 2013]:
            hist = bench.index[bench.index.year < yy]
            test = bench.index[bench.index.year == yy]
            if len(hist) < 750 or len(test) == 0:
                continue
            Q = np.zeros((12, actions))
            s_arr = state.loc[hist].to_numpy()
            r_arr = ret.shift(-1).loc[hist].fillna(0).to_numpy()
            for _ in range(20):  # epochs
                prev_a = 0
                for t in range(len(s_arr) - 1):
                    s, s2 = s_arr[t], s_arr[t + 1]
                    for a in range(actions):
                        rew = a * r_arr[t] - cost * abs(a - prev_a)
                        Q[s, a] += alpha_lr * (rew + gamma * Q[s2].max() - Q[s, a])
                    prev_a = int(Q[s].argmax())
            pol = Q.argmax(axis=1)
            sig.loc[test] = pol[state.loc[test].to_numpy()]
        gate = sig.reindex(c.index).fillna(0)
        return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
            c.notna(), 0.0).mul(gate, axis=0)
    S.append(({"id": "H13", "name": "Q-Learning (tabular)", "category": "H-ML",
               "variant": "12 states"}, fn_h13))

    return S
