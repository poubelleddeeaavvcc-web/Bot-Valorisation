"""Broaden the universe beyond S&P500+Euronext100: full NYSE/NASDAQ common-stock
listing (via the public Nasdaq Trader symbol directories) plus the existing Europe list.
Excludes ETFs, SPAC-style shells, warrants/units/rights/preferred/notes -- these have no
meaningful P/E, ROE, or sector-momentum signal and would just be noise for a value+
momentum+quality screen. A market-cap floor is applied later, during data fetch, for
liquidity (an automated bot has no business trading illiquid micro-caps anyway)."""
import pathlib
import re

import pandas as pd

HERE = pathlib.Path(__file__).parent

JUNK_NAME_PATTERN = re.compile(
    r"warrant|\bunit(s)?\b|\bright(s)?\b|preferred|depositary|\bnotes?\b|"
    r"acquisition corp|acquisition company|\bspac\b|trust preferred",
    re.IGNORECASE,
)


def load_nasdaq_listed():
    df = pd.read_csv(HERE / "nasdaqlisted.txt", sep="|")
    df = df[df["Test Issue"] == "N"]
    df = df[df["ETF"] == "N"]
    df = df[~df["Security Name"].str.contains(JUNK_NAME_PATTERN, na=False)]
    out = df[["Symbol", "Security Name"]].rename(columns={"Symbol": "ticker", "Security Name": "name"})
    out["market"] = "US (NASDAQ)"
    return out


def load_other_listed():
    df = pd.read_csv(HERE / "otherlisted.txt", sep="|")
    df = df[df["Test Issue"] == "N"]
    df = df[df["ETF"] == "N"]
    df = df[~df["Security Name"].str.contains(JUNK_NAME_PATTERN, na=False)]
    out = df[["ACT Symbol", "Security Name"]].rename(columns={"ACT Symbol": "ticker", "Security Name": "name"})
    out["market"] = "US (NYSE/other)"
    return out


def load_europe():
    tables = pd.read_html(HERE / "euronext100_wiki.html")
    df = tables[3][["Ticker", "Name"]].copy()
    df.columns = ["ticker", "name"]
    df["market"] = "Euronext 100"
    return df


def main():
    us = pd.concat([load_nasdaq_listed(), load_other_listed()], ignore_index=True)
    us["ticker"] = us["ticker"].str.replace(".", "-", regex=False).str.strip()
    us = us[us["ticker"].str.match(r"^[A-Z-]+$", na=False)]  # drop tickers with stray symbols (data glitches)
    europe = load_europe()

    universe = pd.concat([us, europe], ignore_index=True).drop_duplicates(subset="ticker")
    universe.to_csv(HERE / "universe_v2.csv", index=False)
    print(f"Univers elargi : {len(universe)} tickers "
          f"({(universe['market'] == 'US (NASDAQ)').sum()} NASDAQ + "
          f"{(universe['market'] == 'US (NYSE/other)').sum()} NYSE/autres US + "
          f"{(universe['market'] == 'Euronext 100').sum()} Euronext 100)")


if __name__ == "__main__":
    main()
