"""Step 1 of the progressive-cache pipeline: shrink the ~7200-ticker US market down to
only stocks in currently-trending sectors, using data sources that don't share Yahoo
Finance's per-ticker rate limit:
  - Sector + market cap for the whole US market: NASDAQ's public screener API, ONE bulk
    call, no per-ticker cost.
  - Sector momentum: 11 SPDR sector ETFs via yfinance -- only 11 requests total, cheap
    even under a tight rate limit.
This is what makes "no universe limit" practical: we only spend the expensive per-ticker
yfinance budget (fetch_cache.py) on names that already cleared the sector-momentum filter.
"""
import json
import pathlib
import re

import pandas as pd
import requests
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
UNIVERSE_DIR = HERE / "data/universe"
MARKET_CAP_FLOOR = 1_000_000_000

SECTOR_ETF = {
    "Technology": "XLK", "Health Care": "XLV", "Finance": "XLF",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Industrials": "XLI",
    "Basic Materials": "XLB", "Energy": "XLE", "Utilities": "XLU",
    "Real Estate": "XLRE", "Telecommunications": "XLC",
}
JUNK_INDUSTRIES = {"Blank Checks"}
# Depositary shares, preferred stock, and mortgage-REIT-style vehicles carry capital
# structures (or aren't common equity at all) that break the simple P/E-based valuation
# model -- excluded here rather than left to quietly produce nonsense valuation_gap outliers.
JUNK_NAME_PATTERN = re.compile(
    r"depositary|\bdepo\b|\bpreferred\b|\bpfd\b|\bnotes?\b|warrant|\bunit(?:s)?\b|\bright(?:s)?\b|"
    r"acquisition corp|acquisition company|\bspac\b",
    re.IGNORECASE,
)


def fetch_nasdaq_bulk(force=False):
    path = UNIVERSE_DIR / "nasdaq_screener_full.json"
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))
    headers = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    r = requests.get("https://api.nasdaq.com/api/screener/stocks",
                      params={"tableonly": "true", "limit": 10000, "offset": 0, "download": "true"},
                      headers=headers, timeout=30)
    r.raise_for_status()
    path.write_text(r.text, encoding="utf-8")
    return r.json()


def sector_momentum():
    rows = []
    for sector, etf in SECTOR_ETF.items():
        hist = yf.Ticker(etf).history(period="14mo", interval="1mo", auto_adjust=True)["Close"].dropna()
        mom = hist.iloc[-2] / hist.iloc[-13] - 1
        rows.append({"sector": sector, "etf": etf, "sector_mom_12_2": mom})
    df = pd.DataFrame(rows).sort_values("sector_mom_12_2", ascending=False)
    df.to_csv(UNIVERSE_DIR / "sector_momentum.csv", index=False)
    return df


def load_non_us_tickers() -> pd.DataFrame:
    """Euronext 100 + UK/Germany/Switzerland/Hong Kong (built by
    build_international_universe.py). No cheap bulk sector+cap source exists for these
    exchanges, so -- unlike the US NASDAQ-bulk path -- they carry no sector pre-filter;
    fetch_cache.py's own market-cap check and analyze_cache.py's sector-momentum filter
    apply once fetched instead. Was silently missing entirely before this fix -- the
    pipeline had been US-only despite everything saying "US + Euronext 100"."""
    frames = []
    euronext_path = UNIVERSE_DIR / "euronext100_wiki.html"
    if euronext_path.exists():
        tables = pd.read_html(euronext_path)
        eu = tables[3][["Ticker", "Name"]].rename(columns={"Ticker": "ticker", "Name": "name"})
        eu["market"] = "Euronext 100"
        frames.append(eu)
    intl_path = UNIVERSE_DIR / "international_universe.csv"
    if intl_path.exists():
        frames.append(pd.read_csv(intl_path))
    if not frames:
        return pd.DataFrame(columns=["ticker", "name", "sector", "industry", "marketCap"])
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(subset="ticker")
    for col in ["sector", "industry", "marketCap"]:
        combined[col] = None
    return combined[["ticker", "name", "sector", "industry", "marketCap"]]


def main():
    bulk = fetch_nasdaq_bulk()
    df = pd.DataFrame(bulk["data"]["rows"])
    df["marketCap"] = pd.to_numeric(df["marketCap"], errors="coerce")
    df = df[(df["marketCap"] >= MARKET_CAP_FLOOR) & (~df["industry"].isin(JUNK_INDUSTRIES))]
    df = df[~df["name"].str.contains(JUNK_NAME_PATTERN, na=False)]

    mom = sector_momentum()
    trending_sectors = mom[mom["sector_mom_12_2"] > 0]["sector"].tolist()
    print("Momentum sectoriel (12-2 mois) :")
    print(mom.to_string(index=False, formatters={"sector_mom_12_2": "{:+.1%}".format}))
    print(f"\nSecteurs retenus (momentum > 0) : {trending_sectors}")

    trending = df[df["sector"].isin(trending_sectors)][["symbol", "name", "sector", "industry", "marketCap"]]
    trending = trending.rename(columns={"symbol": "ticker"})
    trending["ticker"] = trending["ticker"].str.replace(".", "-", regex=False).str.strip()

    non_us = load_non_us_tickers()
    combined = pd.concat([trending, non_us], ignore_index=True).drop_duplicates(subset="ticker")
    combined.to_csv(UNIVERSE_DIR / "trending_universe.csv", index=False)

    print(f"\nUnivers US avant filtre secteur : {len(df)} tickers (cap >= {MARKET_CAP_FLOOR/1e9:.0f}Md$)")
    print(f"Univers US apres filtre secteur en hausse : {len(trending)} tickers")
    print(f"+ {len(non_us)} tickers non-US (Euronext 100, UK, Allemagne, Suisse, Hong Kong) sans pre-filtre secteur")
    print(f"Total : {len(combined)} tickers -> c'est ce sous-ensemble qui ira dans le fetch detaille (fetch_cache.py)")


if __name__ == "__main__":
    main()
