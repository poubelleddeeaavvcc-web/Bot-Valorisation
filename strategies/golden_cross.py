"""Golden/Death Cross trend-following baseline: long when fast SMA > slow SMA, short otherwise."""
import pandas as pd

from backtest.engine import trades_to_daily_equity

NAME = "Golden Cross (SMA50/SMA200)"


def backtest(df: pd.DataFrame, fast: int = 50, slow: int = 200, cost_per_side: float = 0.0005):
    close = df["Close"]
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()

    signal = pd.Series(0, index=df.index)
    signal[sma_fast > sma_slow] = 1
    signal[sma_fast < sma_slow] = -1
    signal[sma_slow.isna()] = 0

    trades = []
    pos = 0
    entry_date = entry_price = None
    for d in df.index:
        s = int(signal.loc[d])
        if s != pos:
            if pos != 0:
                trades.append({"entry_date": entry_date, "entry_price": entry_price,
                                "exit_date": d, "exit_price": close.loc[d], "direction": pos})
            if s != 0:
                entry_date, entry_price = d, close.loc[d]
            pos = s
    if pos != 0:
        last = df.index[-1]
        trades.append({"entry_date": entry_date, "entry_price": entry_price,
                        "exit_date": last, "exit_price": close.loc[last], "direction": pos})

    return trades_to_daily_equity(df.index, close, trades, cost_per_side)
