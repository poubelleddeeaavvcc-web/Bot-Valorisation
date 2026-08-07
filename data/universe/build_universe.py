"""Build the stock universe (S&P 500 + Euronext 100) from cached Wikipedia pages,
normalizing tickers to the Yahoo Finance convention."""
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).parent


def sp500():
    tables = pd.read_html(HERE / "sp500_wiki.html")
    df = tables[0][["Symbol", "Security", "GICS Sector"]].copy()
    df.columns = ["ticker", "name", "sector"]
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["market"] = "S&P 500"
    return df


def euronext100():
    tables = pd.read_html(HERE / "euronext100_wiki.html")
    df = tables[3][["Ticker", "Name", "Corporate form"]].copy()
    df.columns = ["ticker", "name", "sector"]
    df["sector"] = "n/a (" + df["sector"].astype(str) + ")"
    df["market"] = "Euronext 100"
    return df


def main():
    universe = pd.concat([sp500(), euronext100()], ignore_index=True)
    universe = universe.drop_duplicates(subset="ticker")
    universe.to_csv(HERE / "universe.csv", index=False)
    print(f"Univers construit : {len(universe)} tickers ({(universe['market'] == 'S&P 500').sum()} S&P 500 + "
          f"{(universe['market'] == 'Euronext 100').sum()} Euronext 100)")


if __name__ == "__main__":
    main()
