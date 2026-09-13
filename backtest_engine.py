"""Unified vectorized backtesting engine.

Execution model (anti-lookahead):
  * A signal/position decided at the close of bar t becomes effective at the
    OPEN of bar t+1 (positions are shifted by one bar internally).
  * On a change day the outgoing weight earns close(t-1)->open(t) and the
    incoming weight earns open(t)->close(t).
  * Transaction cost + slippage are charged on every unit of turnover.

Portfolio construction: equal weight across all names with an active signal
(long-only by default; long-short if allow_short).  Idle capital earns 0.
"""
import numpy as np
import pandas as pd

import config

ANN = {"1d": 252, "1wk": 52, "1mo": 12}


def _max_drawdown(equity):
    peak = equity.cummax()
    dd = equity / peak - 1
    mdd = dd.min()
    # drawdown duration: longest run below previous peak
    below = dd < 0
    if below.any():
        grp = (~below).cumsum()
        dur = below.groupby(grp).cumsum().max()
    else:
        dur = 0
    return mdd, dd, int(dur)


def extract_trades(positions, open_, close):
    """Trade log from a position-state matrix (already the *held* positions).

    Entry at open of first held bar, exit at open of bar after last held bar
    (approximated by last close if the series ends in-position).
    """
    trades = []
    pos = positions.fillna(0.0)
    sign = np.sign(pos)
    for tkr in pos.columns:
        s = sign[tkr].to_numpy()
        if not (s != 0).any():
            continue
        change = np.flatnonzero(np.diff(s, prepend=0) != 0)
        op = open_[tkr].to_numpy(float)
        cl = close[tkr].to_numpy(float)
        idx = pos.index
        run_start = None
        run_sign = 0
        for c in list(change) + [len(s)]:
            if run_start is not None:
                exit_i = c
                entry_px = op[run_start] if not np.isnan(op[run_start]) else cl[run_start]
                if exit_i < len(s):
                    exit_px = op[exit_i] if not np.isnan(op[exit_i]) else cl[exit_i - 1]
                    exit_date = idx[exit_i]
                else:
                    exit_px = cl[-1]
                    exit_date = idx[-1]
                if entry_px and exit_px and not np.isnan(entry_px) and not np.isnan(exit_px):
                    ret = run_sign * (exit_px / entry_px - 1) - 2 * (config.TRANSACTION_COST + config.SLIPPAGE)
                    trades.append((tkr, idx[run_start], exit_date, run_sign, ret,
                                   (exit_date - idx[run_start]).days))
                run_start = None
                run_sign = 0
            if c < len(s) and s[c] != 0:
                run_start = c
                run_sign = s[c]
    return pd.DataFrame(trades, columns=["stock", "entry", "exit", "side", "ret", "days"])


def run_backtest(signals, open_, close, benchmark=None, interval="1d",
                 allow_short=False, signals_are_weights=False, trade_log=False):
    """signals: desired position at close of each bar (dates x tickers), in {-1,0,1};
    if signals_are_weights, values are portfolio weights directly.
    Returns dict of metrics + equity curve series."""
    ann = ANN.get(interval, 252)
    sig = signals.reindex(index=close.index, columns=close.columns).astype(float)
    sig = sig.where(close.notna())
    if not allow_short:
        sig = sig.clip(lower=0)
    # held position during bar t = signal from close of t-1
    held = sig.shift(1).fillna(0.0)

    if signals_are_weights:
        w = held
    else:
        n_active = (held != 0).sum(axis=1).clip(lower=1)
        w = held.div(n_active, axis=0)

    cc = close.pct_change(fill_method=None)
    co = open_ / close.shift(1) - 1
    oc = close / open_ - 1
    w_prev = w.shift(1).fillna(0.0)
    changed = (w != w_prev)
    gross = (w * cc).where(~changed, w_prev * co.fillna(0) + w * oc.fillna(0))
    gross = gross.fillna(0.0)
    turnover = (w - w_prev).abs().sum(axis=1)
    cost = turnover * (config.TRANSACTION_COST + config.SLIPPAGE)
    port_ret = gross.sum(axis=1) - cost

    equity = (1 + port_ret).cumprod()
    n = len(port_ret)
    if n < 2 or equity.iloc[-1] <= 0:
        return None
    yrs = n / ann
    total_return = equity.iloc[-1] - 1
    cagr = equity.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan
    vol = port_ret.std() * np.sqrt(ann)
    sharpe = (port_ret.mean() * ann - config.RISK_FREE_RATE) / vol if vol > 0 else np.nan
    downside = port_ret[port_ret < 0].std() * np.sqrt(ann)
    sortino = (port_ret.mean() * ann - config.RISK_FREE_RATE) / downside if downside and downside > 0 else np.nan
    mdd, dd_curve, dd_dur = _max_drawdown(equity)
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    exposure = float((held != 0).any(axis=1).mean())
    ulcer = np.sqrt((dd_curve ** 2).mean()) * 100
    recovery = total_return / abs(mdd) if mdd < 0 else np.nan

    alpha = beta = info_ratio = treynor = np.nan
    if benchmark is not None:
        b = benchmark.reindex(close.index).pct_change(fill_method=None).fillna(0.0)
        cov = np.cov(port_ret, b)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
        alpha = (port_ret.mean() - beta * b.mean()) * ann if not np.isnan(beta) else np.nan
        active = port_ret - b
        te = active.std() * np.sqrt(ann)
        info_ratio = active.mean() * ann / te if te > 0 else np.nan
        treynor = (port_ret.mean() * ann - config.RISK_FREE_RATE) / beta if beta and abs(beta) > 1e-6 else np.nan

    res = {
        "total_return": total_return, "cagr": cagr, "volatility": vol,
        "sharpe": sharpe, "sortino": sortino, "calmar": calmar,
        "max_dd": mdd, "dd_duration": dd_dur, "ulcer": ulcer,
        "recovery_factor": recovery, "exposure": exposure,
        "alpha": alpha, "beta": beta, "info_ratio": info_ratio, "treynor": treynor,
        "n_bars": n,
    }

    if trade_log:
        trades = extract_trades(held, open_, close)
    else:
        trades = extract_trades(held, open_, close)
    if len(trades):
        wins = trades[trades.ret > 0]
        losses = trades[trades.ret <= 0]
        res.update({
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades),
            "avg_win": wins.ret.mean() if len(wins) else 0.0,
            "avg_loss": losses.ret.mean() if len(losses) else 0.0,
            "profit_factor": (wins.ret.sum() / abs(losses.ret.sum())) if len(losses) and losses.ret.sum() != 0 else np.inf,
            "expectancy": trades.ret.mean(),
            "avg_holding_days": trades.days.mean(),
        })
    else:
        res.update({"total_trades": 0, "win_rate": np.nan, "avg_win": np.nan,
                    "avg_loss": np.nan, "profit_factor": np.nan, "expectancy": np.nan,
                    "avg_holding_days": np.nan})
    res["_equity"] = equity
    res["_returns"] = port_ret
    res["_trades"] = trades
    return res
