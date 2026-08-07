"""Split-period robustness check: does the Donchian edge hold up in both halves of the
window, or is it concentrated in one bull run? Quick sanity check, not a full walk-forward."""
import pandas as pd

from backtest.metrics import summarize
from run_backtests import COST_PER_SIDE, DATA_DIR, buy_and_hold
from strategies import donchian_turtle, rsi2_meanreversion

PERIODS = {
    "2017-08 -> 2022-02 (1re moitie)": ("2017-08-01", "2022-02-01"),
    "2022-02 -> 2026-08 (2e moitie)": ("2022-02-01", "2026-08-06"),
}

TARGETS = [
    ("crypto", "BTC-USD"),
    ("crypto", "ETH-USD"),
    ("equity", "SPY"),
    ("equity", "AAPL"),
]


def main():
    rows = []
    for asset_class, ticker in TARGETS:
        f = DATA_DIR / asset_class / f"{ticker}.csv"
        full = pd.read_csv(f, index_col="Date", parse_dates=True)
        cost = COST_PER_SIDE[asset_class]
        for label, (start, end) in PERIODS.items():
            df = full.loc[start:end].dropna()
            if len(df) < 300:
                continue
            bh = summarize(buy_and_hold(df), pd.DataFrame(columns=["pnl_pct"]))
            rows.append({"ticker": ticker, "periode": label, "strategie": "Buy & Hold", **bh})
            for name, fn in [(donchian_turtle.NAME, donchian_turtle.backtest), (rsi2_meanreversion.NAME, rsi2_meanreversion.backtest)]:
                eq, tr = fn(df, cost_per_side=cost)
                eq = eq.dropna()
                if len(eq) < 2:
                    continue
                m = summarize(eq, tr)
                rows.append({"ticker": ticker, "periode": label, "strategie": name, **m})

    out = pd.DataFrame(rows)
    print(out[["ticker", "periode", "strategie", "CAGR", "Sharpe", "Max_Drawdown", "Nb_Trades"]]
          .to_string(index=False, formatters={"CAGR": "{:+.1%}".format, "Sharpe": "{:.2f}".format,
                                               "Max_Drawdown": "{:.1%}".format}))
    out.to_csv("results/robustness_split_period.csv", index=False)


if __name__ == "__main__":
    main()
