"""Category P: cross-sectional factor-rank blends (monthly top-N baskets).

Seven causal price/volume factors are converted to daily cross-sectional
percentile ranks and blended with systematic weight sweeps: singles, all
pairs, selected triples, and a walk-forward IC-weighted composite.
~146 variants.
"""
import itertools

import numpy as np
import pandas as pd

from utils.composite import monthly_topn

_F = {}


def factors(ctx):
    key = (ctx["universe"], ctx["interval"])
    if key in _F:
        return _F[key]
    o, c, v = ctx["open"], ctx["close"], ctx["volume"]
    ret = c.pct_change(fill_method=None)
    raw = {
        "mom": c.shift(21) / c.shift(126) - 1,            # 12-1 style 6m ex last month
        "lowvol": -ret.rolling(252).std(),
        "rev": -c.pct_change(21, fill_method=None),
        "liq": (c * v).rolling(63).mean(),
        "onight": (o / c.shift(1) - 1).rolling(63).sum(),
        "smom": ret.rolling(126).mean() / ret.rolling(126).std().replace(0, np.nan),
        "hi52": c / c.rolling(252).max(),
    }
    ranks = {k: df.rank(axis=1, pct=True) for k, df in raw.items()}
    _F[key] = ranks
    return ranks


def blend_fn(names, weights, top_k):
    def fn(ctx):
        r = factors(ctx)
        score = sum(w * r[n] for n, w in zip(names, weights))
        return monthly_topn(ctx["close"], score, top_k)
    return fn


def get_strategies():
    S = []
    FN = ["mom", "lowvol", "rev", "liq", "onight", "smom", "hi52"]

    # P1: single factors x top_k
    for name in FN:
        for k in [5, 10, 15]:
            S.append(({"id": "P1", "name": f"Factor:{name}", "category": "P-RankBlend",
                       "variant": f"{name} top{k}"}, blend_fn([name], [1.0], k)))

    # P2: all pairs x weight splits (top10) + equal split top5
    for a, b in itertools.combinations(FN, 2):
        for wa, wb in [(0.5, 0.5), (0.7, 0.3), (0.3, 0.7)]:
            S.append(({"id": "P2", "name": f"Pair:{a}+{b}", "category": "P-RankBlend",
                       "variant": f"{a}{wa}/{b}{wb} top10"},
                      blend_fn([a, b], [wa, wb], 10)))
        S.append(({"id": "P2", "name": f"Pair:{a}+{b}", "category": "P-RankBlend",
                   "variant": f"{a}.5/{b}.5 top5"}, blend_fn([a, b], [0.5, 0.5], 5)))

    # P3: selected triples x weight sets
    TRIPLES = [("mom", "lowvol", "rev"), ("mom", "lowvol", "hi52"),
               ("mom", "liq", "lowvol"), ("mom", "onight", "lowvol"),
               ("smom", "lowvol", "rev"), ("mom", "lowvol", "onight")]
    WSETS = [(1 / 3, 1 / 3, 1 / 3), (0.5, 0.3, 0.2), (0.2, 0.3, 0.5), (0.6, 0.2, 0.2)]
    for tri in TRIPLES:
        for ws in WSETS:
            S.append(({"id": "P3", "name": f"Tri:{'+'.join(tri)}", "category": "P-RankBlend",
                       "variant": f"{'/'.join(f'{w:.2f}' for w in ws)} top10"},
                      blend_fn(list(tri), list(ws), 10)))
        for k in [5, 15]:
            S.append(({"id": "P3", "name": f"Tri:{'+'.join(tri)}", "category": "P-RankBlend",
                       "variant": f"eq top{k}"}, blend_fn(list(tri), [1 / 3] * 3, k)))

    # P4: all-factor equal composite + walk-forward IC-weighted composite
    for k in [5, 10, 15]:
        S.append(({"id": "P4", "name": "All-Factor Composite", "category": "P-RankBlend",
                   "variant": f"eq7 top{k}"}, blend_fn(FN, [1 / 7] * 7, k)))

    for ic_months in [24, 36]:
        def fn(ctx, ic_months=ic_months):
            c = ctx["close"]
            r = factors(ctx)
            me_mask = pd.Series(c.index, index=c.index).groupby(
                [c.index.year, c.index.month]).transform("max") == c.index
            me_dates = list(c.index[me_mask])
            fwd = {}
            for i, d in enumerate(me_dates[:-1]):
                nxt = me_dates[i + 1]
                fwd[d] = c.loc[nxt] / c.loc[d] - 1
            pos = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
            for i, d in enumerate(me_dates):
                past = me_dates[max(0, i - ic_months - 1):i - 1]  # completed months only
                if len(past) < 12:
                    continue
                wts = {}
                for name in r:
                    ics = []
                    for p in past:
                        f_ = r[name].loc[p]
                        y_ = fwd.get(p)
                        if y_ is None:
                            continue
                        m = f_.notna() & y_.notna()
                        if m.sum() > 10:
                            ics.append(f_[m].corr(y_[m], method="spearman"))
                    wts[name] = max(np.nanmean(ics), 0.0) if ics else 0.0
                tot = sum(wts.values())
                if tot <= 0:
                    continue
                score = sum(w / tot * r[n].loc[d] for n, w in wts.items())
                sel = score.dropna().nlargest(10).index
                row = pd.Series(0.0, index=c.columns)
                row[list(sel)] = 1.0
                pos.loc[d] = row
            return pos.ffill().fillna(0)
        S.append(({"id": "P4", "name": "IC-Weighted Composite", "category": "P-RankBlend",
                   "variant": f"IC{ic_months}m top10"}, fn))

    return S
