"""Category J: Calendar & Seasonal strategies (J1-J9).

Seasonal parameters (best weekday, best months) are selected walk-forward on
prior years only.  NSE holidays are inferred from gaps in the trading
calendar; Diwali/Muhurat dates 2010-2024 are hardcoded.
"""
import numpy as np
import pandas as pd


def _basket(c, cond_series):
    """Broadcast a date-level 0/1 series to all listed stocks."""
    return pd.DataFrame(1.0, index=c.index, columns=c.columns).where(
        c.notna(), 0.0).mul(cond_series.astype(float).reindex(c.index).fillna(0), axis=0)


DIWALI = ["2010-11-05", "2011-10-26", "2012-11-13", "2013-11-03", "2014-10-23",
          "2015-11-11", "2016-10-30", "2017-10-19", "2018-11-07", "2019-10-27",
          "2020-11-14", "2021-11-04", "2022-10-24", "2023-11-12", "2024-11-01"]


def get_strategies():
    S = []

    # J1: Day-of-week effect (walk-forward best weekday)
    def fn_j1(ctx):
        c = ctx["close"]
        bench = ctx["benchmark"]
        # return credited to a close-t signal = open(t+1) -> open(t+2) ~ next day ret
        fwd = bench.shift(-2) / bench.shift(-1) - 1
        sig = pd.Series(0.0, index=c.index)
        years = sorted(set(c.index.year))
        for y in years[3:]:
            hist = fwd[c.index.year < y]
            by_dow = hist.groupby(hist.index.dayofweek).mean()
            if by_dow.empty:
                continue
            best = int(by_dow.idxmax())
            mask = (c.index.year == y) & (c.index.dayofweek == best)
            sig[mask] = 1.0
        return _basket(c, sig)
    S.append(({"id": "J1", "name": "Day-of-Week Effect", "category": "J-Seasonal",
               "variant": "walkfwd best dow"}, fn_j1))

    # J2: Month-of-year — classic Sell-in-May and walk-forward best-6
    def fn_j2a(ctx):
        c = ctx["close"]
        nov_apr = pd.Series(c.index.month, index=c.index).isin([11, 12, 1, 2, 3, 4])
        return _basket(c, nov_apr)
    S.append(({"id": "J2", "name": "Sell in May", "category": "J-Seasonal",
               "variant": "long Nov-Apr"}, fn_j2a))

    def fn_j2b(ctx):
        c = ctx["close"]
        bench = ctx["benchmark"]
        ret = bench.pct_change()
        sig = pd.Series(0.0, index=c.index)
        years = sorted(set(c.index.year))
        for y in years[4:]:
            hist = ret[c.index.year < y]
            by_m = hist.groupby(hist.index.month).mean()
            if len(by_m) < 12:
                continue
            best6 = set(by_m.nlargest(6).index)
            mask = (c.index.year == y) & pd.Series(c.index.month, index=c.index).isin(best6)
            sig[mask.to_numpy()] = 1.0
        return _basket(c, sig)
    S.append(({"id": "J2", "name": "Best-6-Months", "category": "J-Seasonal",
               "variant": "walkfwd best 6mo"}, fn_j2b))

    # J3: Turn-of-month (long last 2 + first 3 trading days)
    def fn_j3(ctx):
        c = ctx["close"]
        idx = c.index
        ym = idx.year * 100 + idx.month
        pos_in_month = pd.Series(np.arange(len(idx)), index=idx).groupby(ym).cumcount()
        month_len = pd.Series(ym, index=idx).map(pd.Series(ym, index=idx).value_counts())
        first3 = pos_in_month < 3
        last2 = pos_in_month >= (month_len - 2)
        return _basket(c, (first3 | last2))
    S.append(({"id": "J3", "name": "Turn-of-Month", "category": "J-Seasonal",
               "variant": "last2+first3"}, fn_j3))

    # J4: Pre-holiday rally (holiday = weekday gap in trading calendar)
    def fn_j4(ctx):
        c = ctx["close"]
        idx = c.index
        next_day = pd.Series(idx[1:].append(idx[-1:]), index=idx)
        biz_gap = pd.Series([np.busday_count(a.date(), b.date())
                             for a, b in zip(idx, next_day)], index=idx)
        pre_holiday = biz_gap >= 2  # at least one weekday closed before next session
        return _basket(c, pre_holiday)
    S.append(({"id": "J4", "name": "Pre-Holiday Rally", "category": "J-Seasonal",
               "variant": "day before closure"}, fn_j4))

    # J5: Budget-day window (Jan 25 - Feb 3; pre-2017 budget was Feb 28: Feb 22 - Mar 3)
    def fn_j5(ctx):
        c = ctx["close"]
        idx = c.index
        sig = pd.Series(0.0, index=idx)
        for y in sorted(set(idx.year)):
            if y >= 2017:
                a, b = pd.Timestamp(y, 1, 25), pd.Timestamp(y, 2, 3)
            else:
                a, b = pd.Timestamp(y, 2, 22), pd.Timestamp(y, 3, 3)
            sig[(idx >= a) & (idx <= b)] = 1.0
        return _basket(c, sig)
    S.append(({"id": "J5", "name": "Budget-Day Window", "category": "J-Seasonal",
               "variant": "±5d around budget"}, fn_j5))

    # J6: F&O expiry week (long 3 sessions ending at last Thursday of month)
    def fn_j6(ctx):
        c = ctx["close"]
        idx = c.index
        is_thu = idx.dayofweek == 3
        ym = idx.year * 100 + idx.month
        last_thu = pd.Series(False, index=idx)
        thu_dates = pd.Series(idx[is_thu], index=idx[is_thu]).groupby(ym[is_thu]).max()
        last_thu[thu_dates.values] = True
        # expiry day and 2 sessions before
        sig = (last_thu | last_thu.shift(-1, fill_value=False)
               | last_thu.shift(-2, fill_value=False))
        return _basket(c, sig)
    S.append(({"id": "J6", "name": "Expiry-Week Effect", "category": "J-Seasonal",
               "variant": "3d into expiry"}, fn_j6))

    # J7: Muhurat/Diwali seasonality (5 sessions before Diwali to 10 after)
    def fn_j7(ctx):
        c = ctx["close"]
        idx = c.index
        sig = pd.Series(0.0, index=idx)
        pos = pd.Series(np.arange(len(idx)), index=idx)
        for d in pd.to_datetime(DIWALI):
            near = pos[idx >= d]
            if near.empty:
                continue
            k = near.iloc[0]
            sig.iloc[max(0, k - 5):min(len(idx), k + 10)] = 1.0
        return _basket(c, sig)
    S.append(({"id": "J7", "name": "Muhurat Window", "category": "J-Seasonal",
               "variant": "-5/+10 sessions"}, fn_j7))

    # J8: January effect on small caps
    def fn_j8(ctx):
        c = ctx["close"]
        jan = pd.Series(c.index.month, index=c.index) == 1
        return _basket(c, jan)
    S.append(({"id": "J8", "name": "January Effect (smallcap)", "category": "J-Seasonal",
               "variant": "long Jan", "universe": "smallcap"}, fn_j8))

    # J9: Quarter-end window dressing (last 5 sessions of quarter)
    def fn_j9(ctx):
        c = ctx["close"]
        idx = c.index
        q = idx.year * 10 + idx.quarter
        pos_in_q = pd.Series(np.arange(len(idx)), index=idx).groupby(q).cumcount()
        q_len = pd.Series(q, index=idx).map(pd.Series(q, index=idx).value_counts())
        last5 = pos_in_q >= (q_len - 5)
        return _basket(c, last5)
    S.append(({"id": "J9", "name": "Quarter-End Dressing", "category": "J-Seasonal",
               "variant": "last 5d of qtr"}, fn_j9))

    return S
