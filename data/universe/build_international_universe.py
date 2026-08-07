"""Broaden the raw universe beyond US + Euronext 100: UK (FTSE 100/250), Germany
(DAX/MDAX), Switzerland (SMI), Hong Kong (Hang Seng) -- same method as Euronext 100
(scrape the Wikipedia constituent table), normalized to Yahoo Finance ticker suffixes.

Dropped from this pass: SDAX (Wikipedia's table has no ticker/symbol column at all) and
Nikkei 225 (no reliable constituent table with tickers found on Wikipedia). Both would
need a different source; not worth blocking the rest of the expansion on them.

These tickers carry no sector/market-cap pre-filter (unlike the US NASDAQ-bulk path) --
there's no equivalent free bulk API for these exchanges. That's fine: index membership
itself is already a de facto large/mid-cap filter, and fetch_cache.py's own market-cap
check + analyze_cache.py's sector-momentum filter still apply once fetched. We simply
don't get to skip the fetch budget for off-trend sectors the way we do for the full US
market -- acceptable given the throughput headroom (423/hour) relative to these being a
few hundred extra tickers.
"""
import pathlib
import re

import pandas as pd

HERE = pathlib.Path(__file__).parent


def ftse(file, table_idx, market):
    df = pd.read_html(HERE / file)[table_idx]
    df = df.rename(columns={df.columns[0]: "name", df.columns[1]: "ticker"})[["ticker", "name"]]
    df["ticker"] = df["ticker"].astype(str).str.strip() + ".L"
    df["market"] = market
    return df


def dax():
    df = pd.read_html(HERE / "dax_wiki.html")[4][["Ticker", "Company"]]
    df = df.rename(columns={"Ticker": "ticker", "Company": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip()  # already Yahoo-ready (mixed .DE/.PA/etc)
    df["market"] = "DAX"
    return df


def mdax():
    df = pd.read_html(HERE / "mdax_wiki.html")[2][["Symbol", "Name"]]
    df = df.rename(columns={"Symbol": "ticker", "Name": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip() + ".DE"
    df["market"] = "MDAX"
    return df


def smi():
    df = pd.read_html(HERE / "smi_wiki.html")[2][["Ticker", "Name"]]
    df = df.rename(columns={"Ticker": "ticker", "Name": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip() + ".SW"
    df["market"] = "SMI"
    return df


def hangseng():
    df = pd.read_html(HERE / "hangseng_wiki.html")[6][["Ticker", "Name"]]
    df = df.rename(columns={"Ticker": "ticker", "Name": "name"})
    # raw values look like "SEHK:​5" -- pull out the numeric code, zero-pad to 4 digits
    codes = df["ticker"].astype(str).str.extract(r"(\d+)")[0]
    df = df[codes.notna()].copy()
    df["ticker"] = codes.dropna().str.zfill(4) + ".HK"
    df["market"] = "Hang Seng"
    return df


def main():
    parts = [
        ftse("ftse100_wiki.html", 6, "FTSE 100"),
        ftse("ftse250_wiki.html", 3, "FTSE 250"),
        dax(),
        mdax(),
        smi(),
        hangseng(),
    ]
    combined = pd.concat(parts, ignore_index=True).drop_duplicates(subset="ticker")
    combined.to_csv(HERE / "international_universe.csv", index=False)
    print(f"Univers international : {len(combined)} tickers")
    print(combined["market"].value_counts().to_string())


if __name__ == "__main__":
    main()
