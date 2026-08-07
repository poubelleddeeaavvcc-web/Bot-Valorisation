"""Donchian Channel Breakout ("Turtle"-style): enter on an N-day extreme breakout,
exit on the opposite shorter-channel extreme or an ATR-based stop, whichever hits first.
Classic parameters (entry=20, exit=10, ATR stop=2x) per the original Turtle system."""
import pandas as pd

from backtest.engine import trades_to_daily_equity
from strategies.indicators import atr

NAME = "Donchian Breakout (Turtle 20/10, stop 2xATR)"


def backtest(df: pd.DataFrame, entry_n: int = 20, exit_n: int = 10,
             atr_len: int = 20, atr_mult: float = 2.0, cost_per_side: float = 0.0005,
             allow_short: bool = True):
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]

    entry_high = h.rolling(entry_n).max().shift(1)
    entry_low = l.rolling(entry_n).min().shift(1)
    exit_high = h.rolling(exit_n).max().shift(1)
    exit_low = l.rolling(exit_n).min().shift(1)
    atr_prev = atr(h, l, c, atr_len).shift(1)

    warmup = max(entry_n, exit_n, atr_len) + 1
    dates = df.index

    trades = []
    pos = 0  # 0 flat, 1 long, -1 short
    entry_date = entry_price = stop_price = None

    for i in range(warmup, len(dates)):
        d = dates[i]
        if pos == 0:
            if h.iloc[i] > entry_high.iloc[i] and pd.notna(entry_high.iloc[i]):
                fill = entry_high.iloc[i] if o.iloc[i] <= entry_high.iloc[i] else o.iloc[i]
                pos, entry_date, entry_price = 1, d, fill
                stop_price = entry_price - atr_mult * atr_prev.iloc[i]
            elif allow_short and l.iloc[i] < entry_low.iloc[i] and pd.notna(entry_low.iloc[i]):
                fill = entry_low.iloc[i] if o.iloc[i] >= entry_low.iloc[i] else o.iloc[i]
                pos, entry_date, entry_price = -1, d, fill
                stop_price = entry_price + atr_mult * atr_prev.iloc[i]
        elif pos == 1:
            if l.iloc[i] <= stop_price:
                fill = stop_price if o.iloc[i] >= stop_price else o.iloc[i]
                trades.append({"entry_date": entry_date, "entry_price": entry_price,
                                "exit_date": d, "exit_price": fill, "direction": 1})
                pos = 0
            elif pd.notna(exit_low.iloc[i]) and l.iloc[i] <= exit_low.iloc[i]:
                fill = exit_low.iloc[i] if o.iloc[i] >= exit_low.iloc[i] else o.iloc[i]
                trades.append({"entry_date": entry_date, "entry_price": entry_price,
                                "exit_date": d, "exit_price": fill, "direction": 1})
                pos = 0
        elif pos == -1:
            if h.iloc[i] >= stop_price:
                fill = stop_price if o.iloc[i] <= stop_price else o.iloc[i]
                trades.append({"entry_date": entry_date, "entry_price": entry_price,
                                "exit_date": d, "exit_price": fill, "direction": -1})
                pos = 0
            elif pd.notna(exit_high.iloc[i]) and h.iloc[i] >= exit_high.iloc[i]:
                fill = exit_high.iloc[i] if o.iloc[i] <= exit_high.iloc[i] else o.iloc[i]
                trades.append({"entry_date": entry_date, "entry_price": entry_price,
                                "exit_date": d, "exit_price": fill, "direction": -1})
                pos = 0

    if pos != 0:
        last = dates[-1]
        trades.append({"entry_date": entry_date, "entry_price": entry_price,
                        "exit_date": last, "exit_price": c.loc[last], "direction": pos})

    return trades_to_daily_equity(df.index, c, trades, cost_per_side)
