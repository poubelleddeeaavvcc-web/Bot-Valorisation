"""Broaden the raw universe beyond US + Euronext 100: UK (FTSE 100/250), Germany
(DAX/MDAX), Switzerland (SMI), Hong Kong (Hang Seng), Canada (S&P/TSX Composite), Spain
(IBEX 35), Sweden (OMX Stockholm 30), South Korea (KOSPI 200), Australia (S&P/ASX 200),
India (NIFTY 50 + BSE SENSEX) -- scrape the Wikipedia constituent table, normalized to
Yahoo Finance ticker suffixes.

Japan and Taiwan use a different source: neither Nikkei 225/TOPIX nor the Taiwan 50 Index
has a Wikipedia table with tickers (checked directly -- Taiwan 50's page has no <table> at
all; Nikkei 225/TOPIX only have historical index-level data, no per-company list). Both
exchanges publish their own official bulk listed-issues files instead, same idea as the
NASDAQ bulk screener build_trending_universe.py already uses for the US:
  - Japan (JPX): data_j.xls, filtered to Prime-market, TOPIX Core30/Large70/Mid400 (large
    + mid cap) -- unfiltered it's ~4400 rows including ETFs/ETNs/PRO Market, far more than
    a large/mid-cap value+momentum screen has any business fetching one-by-one.
  - Taiwan (TWSE): t187ap03_L open-data endpoint, main-board listed companies -- TWSE main-
    board listing standards already function as a de facto size/quality gate, same
    reasoning as "index membership" for the Wikipedia-sourced markets above, so no extra
    size filter applied.

Dropped from this pass: SDAX (Wikipedia's table has no ticker/symbol column at all).

These tickers carry no sector/market-cap pre-filter (unlike the US NASDAQ-bulk path) --
there's no equivalent free bulk API for these exchanges. That's fine: index membership
itself is already a de facto large/mid-cap filter, and fetch_cache.py's own market-cap
check + analyze_cache.py's sector-momentum filter still apply once fetched. We simply
don't get to skip the fetch budget for off-trend sectors the way we do for the full US
market -- acceptable given the throughput headroom (423/hour) relative to these being a
few hundred extra tickers.
"""
import json
import pathlib
import re

import pandas as pd

HERE = pathlib.Path(__file__).parent

JPX_SIZE_TIERS = {"TOPIX Core30", "TOPIX Large70", "TOPIX Mid400"}  # large + mid cap only
JPX_PRIME = "プライム（内国株式）"  # Prime market, domestic stock


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


def cac40():
    df = pd.read_html(HERE / "cac40_wiki.html")[4][["Ticker", "Company"]]
    df = df.rename(columns={"Ticker": "ticker", "Company": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip()  # already Yahoo-ready (.PA)
    df["market"] = "CAC 40"
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


def tsx():
    df = pd.read_html(HERE / "tsx_wiki.html")[3]
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Ticker", "Company"]].rename(columns={"Ticker": "ticker", "Company": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip() + ".TO"  # bare TSX symbols on Wikipedia
    df["market"] = "S&P/TSX Composite"
    return df


def ibex35():
    df = pd.read_html(HERE / "ibex35_wiki.html")[2]
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Ticker", "Company"]].rename(columns={"Ticker": "ticker", "Company": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip()  # already Yahoo-ready (.MC)
    df["market"] = "IBEX 35"
    return df


def omxs30():
    df = pd.read_html(HERE / "omxs30_wiki.html")[1]
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Ticker", "Company"]].rename(columns={"Ticker": "ticker", "Company": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip()  # already Yahoo-ready (.ST, incl. -A/-B classes)
    df["market"] = "OMX Stockholm 30"
    return df


def kospi200():
    df = pd.read_html(HERE / "kospi200_wiki.html")[2]
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Symbol", "Company"]].rename(columns={"Symbol": "ticker", "Company": "name"})
    # 6-digit KRX codes -- zero-pad (pandas/Wikipedia can drop a leading zero) then suffix
    df["ticker"] = df["ticker"].astype(str).str.strip().str.zfill(6) + ".KS"
    df["market"] = "KOSPI 200"
    return df


def asx200():
    df = pd.read_html(HERE / "asx200_wiki.html")[2]
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Code", "Company"]].rename(columns={"Code": "ticker", "Company": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip() + ".AX"
    df["market"] = "S&P/ASX 200"
    return df


def nifty50():
    df = pd.read_html(HERE / "nifty50_wiki.html")[1]
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df[["Symbol", "Company name"]].rename(columns={"Symbol": "ticker", "Company name": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip() + ".NS"  # NSE
    df["market"] = "NIFTY 50"
    return df


def sensex():
    df = pd.read_html(HERE / "sensex_wiki.html")[2]
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    # careful: this table's "Ticker" column is actually the raw BSE numeric scrip code;
    # "Symbol" is the one already suffixed ".BO" and Yahoo-ready.
    df = df[["Symbol", "Company"]].rename(columns={"Symbol": "ticker", "Company": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip()
    df["market"] = "BSE SENSEX"
    return df


def jpx():
    """Japan: no usable Wikipedia table exists for Nikkei 225 or TOPIX, so this uses JPX's
    own official listed-issues file instead (data_j.xls, refreshed by re-running the
    download by hand -- see module docstring). Restricted to Prime market + TOPIX
    Core30/Large70/Mid400 (large+mid cap): unfiltered the file is ~4400 rows including
    ETFs/ETNs/PRO Market, far more than a large/mid-cap screen should fetch one-by-one.
    """
    df = pd.read_excel(HERE / "jpx_listed_issues.xls")
    df = df[(df["市場・商品区分"] == JPX_PRIME) & (df["規模区分"].isin(JPX_SIZE_TIERS))]
    df = df[["コード", "銘柄名"]].rename(columns={"コード": "ticker", "銘柄名": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip() + ".T"
    df["market"] = "JPX Prime (Core30/Large70/Mid400)"
    return df


def twse():
    """Taiwan: same situation as Japan -- neither the Taiwan 50 Index nor TAIEX Wikipedia
    pages have a constituent table with tickers. TWSE's own open-data endpoint (basic info
    for all main-board listed companies) fills the gap; main-board listing standards are
    already a de facto size/quality gate, so no extra size filter is applied here."""
    data = json.loads((HERE / "twse_listed_issues.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(data)[["公司代號", "英文簡稱"]].rename(columns={"公司代號": "ticker", "英文簡稱": "name"})
    df["ticker"] = df["ticker"].astype(str).str.strip() + ".TW"
    df["market"] = "TWSE (main board)"
    return df


def main():
    parts = [
        ftse("ftse100_wiki.html", 6, "FTSE 100"),
        ftse("ftse250_wiki.html", 3, "FTSE 250"),
        dax(),
        mdax(),
        smi(),
        cac40(),
        hangseng(),
        tsx(),
        ibex35(),
        omxs30(),
        kospi200(),
        asx200(),
        nifty50(),
        sensex(),
        jpx(),
        twse(),
    ]
    combined = pd.concat(parts, ignore_index=True).drop_duplicates(subset="ticker")
    combined.to_csv(HERE / "international_universe.csv", index=False)
    print(f"Univers international : {len(combined)} tickers")
    print(combined["market"].value_counts().to_string())


if __name__ == "__main__":
    main()
