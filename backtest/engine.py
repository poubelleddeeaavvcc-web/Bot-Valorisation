"""Small shared helper for turning a list of discrete trades into a daily equity curve.

Convention used across every strategy in this study: signals are computed using data
available *through* a bar's close, and the resulting order is assumed filled at that same
close (a "trade at close" / market-on-close convention, matching how the reference
strategies this study reproduces -- TASC, Connors, classic Donchian/Turtle writeups -- are
themselves usually backtested). This keeps the comparison consistent across strategies
without pretending to a level of execution precision the daily-bar data doesn't support.
"""
import pandas as pd


def trades_to_daily_equity(index: pd.DatetimeIndex, close: pd.Series, trades: list, cost_per_side: float):
    """trades: list of dicts with entry_date, entry_price, exit_date, exit_price, direction (+1/-1),
    sorted chronologically and non-overlapping (one position at a time).
    Returns (daily equity curve starting at 1.0, trades dataframe with pnl_pct per trade)."""
    equity = pd.Series(index=index, dtype=float)
    level = 1.0
    cursor = 0  # position in index up to which `equity` has been filled
    trade_rows = []

    for tr in trades:
        entry_date, exit_date = tr["entry_date"], tr["exit_date"]
        entry_price, exit_price, direction = tr["entry_price"], tr["exit_price"], tr["direction"]

        entry_pos = index.get_loc(entry_date)
        exit_pos = index.get_loc(exit_date)

        # flat period before this trade holds the previous level
        if entry_pos > cursor:
            equity.iloc[cursor:entry_pos] = level

        equity_after_entry_cost = level * (1 - cost_per_side)
        span = index[entry_pos:exit_pos + 1]
        for d in span:
            price = exit_price if d == exit_date else close.loc[d]
            raw_ret = direction * (price / entry_price - 1)
            equity.loc[d] = equity_after_entry_cost * (1 + raw_ret)

        level = equity.loc[exit_date] * (1 - cost_per_side)
        equity.loc[exit_date] = level
        cursor = exit_pos + 1

        gross_pnl_pct = direction * (exit_price / entry_price - 1)
        net_pnl_pct = (1 + gross_pnl_pct) * (1 - cost_per_side) ** 2 - 1
        trade_rows.append({
            "entry_date": entry_date, "exit_date": exit_date, "direction": direction,
            "entry_price": entry_price, "exit_price": exit_price, "pnl_pct": net_pnl_pct,
        })

    if cursor < len(index):
        equity.iloc[cursor:] = level

    trades_df = pd.DataFrame(trade_rows)
    return equity, trades_df
