"""Performance metrics shared by all strategy backtests."""
import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return drawdown.min()


def summarize(equity: pd.Series, trades: pd.DataFrame, periods_per_year: int = 252) -> dict:
    """equity: daily equity curve (indexed by date). trades: one row per closed trade with a 'pnl_pct' column."""
    daily_ret = equity.pct_change().dropna()
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    vol_ann = daily_ret.std() * np.sqrt(periods_per_year)
    sharpe = (daily_ret.mean() * periods_per_year) / vol_ann if vol_ann > 0 else np.nan
    mdd = max_drawdown(equity)

    if len(trades) > 0:
        wins = trades.loc[trades["pnl_pct"] > 0, "pnl_pct"]
        losses = trades.loc[trades["pnl_pct"] <= 0, "pnl_pct"]
        profit_factor = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.nan
        win_rate = len(wins) / len(trades)
    else:
        profit_factor = np.nan
        win_rate = np.nan

    return {
        "CAGR": cagr,
        "Vol_annualisee": vol_ann,
        "Sharpe": sharpe,
        "Max_Drawdown": mdd,
        "Profit_Factor": profit_factor,
        "Win_Rate": win_rate,
        "Nb_Trades": len(trades),
        "Rendement_total": equity.iloc[-1] / equity.iloc[0] - 1,
    }
