"""Run every candidate strategy against every asset in the study universe and produce
a comparison table (results/comparison.csv) plus a per-asset detail (results/detail.csv)."""
import pathlib

import pandas as pd

from backtest.metrics import summarize
from strategies import donchian_turtle, golden_cross, rsi2_meanreversion

HERE = pathlib.Path(__file__).parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

COST_PER_SIDE = {"crypto": 0.0010, "equity": 0.0005, "forex": 0.0002}

STRATEGIES = {
    golden_cross.NAME: golden_cross.backtest,
    donchian_turtle.NAME: donchian_turtle.backtest,
    rsi2_meanreversion.NAME: rsi2_meanreversion.backtest,
}

# common window so every asset is compared over the same period
COMMON_START = "2017-08-01"
COMMON_END = "2026-08-06"


def load(asset_class: str, ticker_file: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(ticker_file, index_col="Date", parse_dates=True)
    df = df.loc[COMMON_START:COMMON_END]
    return df.dropna()


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    close = df["Close"]
    return close / close.iloc[0]


def main():
    rows = []
    for asset_class_dir in sorted(DATA_DIR.iterdir()):
        if not asset_class_dir.is_dir():
            continue
        asset_class = asset_class_dir.name
        cost = COST_PER_SIDE[asset_class]
        for f in sorted(asset_class_dir.glob("*.csv")):
            ticker = f.stem
            df = load(asset_class, f)
            if len(df) < 300:
                print(f"skip {ticker}: not enough data in common window")
                continue

            bh_equity = buy_and_hold(df)
            bh_metrics = summarize(bh_equity, pd.DataFrame(columns=["pnl_pct"]))
            rows.append({"asset_class": asset_class, "ticker": ticker, "strategy": "Buy & Hold", **bh_metrics})

            for strat_name, strat_fn in STRATEGIES.items():
                equity, trades = strat_fn(df, cost_per_side=cost)
                equity = equity.dropna()
                if len(equity) < 2:
                    continue
                m = summarize(equity, trades)
                rows.append({"asset_class": asset_class, "ticker": ticker, "strategy": strat_name, **m})
                print(f"{asset_class:8s} {ticker:12s} {strat_name:40s} "
                      f"CAGR={m['CAGR']:+.1%} Sharpe={m['Sharpe']:.2f} MaxDD={m['Max_Drawdown']:.1%} "
                      f"trades={m['Nb_Trades']}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_DIR / "comparison.csv", index=False)

    summary = (results.groupby(["strategy", "asset_class"])
               [["CAGR", "Sharpe", "Max_Drawdown", "Profit_Factor", "Win_Rate", "Nb_Trades"]]
               .mean(numeric_only=True))
    summary.to_csv(RESULTS_DIR / "summary_by_strategy_and_class.csv")
    print("\n=== Moyenne par stratégie / classe d'actif ===")
    print(summary.round(3))


if __name__ == "__main__":
    main()
