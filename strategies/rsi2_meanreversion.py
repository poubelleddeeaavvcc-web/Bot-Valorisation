"""Connors RSI(2) mean reversion: in an established uptrend (close > SMA200), buy sharp
short-term pullbacks (RSI(2) < entry_th) and sell the snap-back (RSI(2) > exit_th).
A wide ATR stop and a max holding period are added as a risk-management backstop --
not part of Connors' original rules, but standard practice when trading this live."""
import pandas as pd

from backtest.engine import trades_to_daily_equity
from strategies.indicators import atr, wilder_rsi

NAME = "RSI(2) Mean Reversion (Connors)"


def backtest(df: pd.DataFrame, sma_len: int = 200, rsi_len: int = 2,
             entry_th: float = 10, exit_th: float = 70,
             atr_len: int = 20, atr_mult: float = 3.0, max_hold: int = 10,
             cost_per_side: float = 0.0005):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    sma = c.rolling(sma_len).mean()
    rsi2 = wilder_rsi(c, rsi_len)
    atr_series = atr(h, l, c, atr_len)

    warmup = max(sma_len, rsi_len, atr_len) + 1
    dates = df.index

    trades = []
    pos = 0
    entry_date = entry_price = stop_price = None
    hold_days = 0

    for i in range(warmup, len(dates)):
        d = dates[i]
        if pos == 0:
            if c.iloc[i] > sma.iloc[i] and rsi2.iloc[i] < entry_th:
                pos, entry_date, entry_price = 1, d, c.iloc[i]
                stop_price = entry_price - atr_mult * atr_series.iloc[i]
                hold_days = 0
        else:
            hold_days += 1
            if l.iloc[i] <= stop_price:
                fill = stop_price if o.iloc[i] >= stop_price else o.iloc[i]
                trades.append({"entry_date": entry_date, "entry_price": entry_price,
                                "exit_date": d, "exit_price": fill, "direction": 1})
                pos = 0
            elif rsi2.iloc[i] > exit_th or hold_days >= max_hold:
                trades.append({"entry_date": entry_date, "entry_price": entry_price,
                                "exit_date": d, "exit_price": c.iloc[i], "direction": 1})
                pos = 0

    if pos != 0:
        last = dates[-1]
        trades.append({"entry_date": entry_date, "entry_price": entry_price,
                        "exit_date": last, "exit_price": c.loc[last], "direction": 1})

    return trades_to_daily_equity(df.index, c, trades, cost_per_side)
