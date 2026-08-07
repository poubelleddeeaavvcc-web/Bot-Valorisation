"""Download daily OHLCV history for the study's universe via yfinance and cache to CSV."""
import pathlib
import sys

import yfinance as yf

HERE = pathlib.Path(__file__).parent

UNIVERSE = {
    "crypto": ["BTC-USD", "ETH-USD"],
    "equity": ["SPY", "QQQ", "AAPL", "MSFT"],
    "forex": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
}

START = "2012-01-01"
END = "2026-08-07"


def download_all():
    for asset_class, tickers in UNIVERSE.items():
        out_dir = HERE / asset_class
        out_dir.mkdir(exist_ok=True)
        for ticker in tickers:
            print(f"Downloading {ticker} ({asset_class})...")
            df = yf.download(ticker, start=START, end=END, interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                print(f"  WARNING: no data for {ticker}", file=sys.stderr)
                continue
            if isinstance(df.columns, __import__("pandas").MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index.name = "Date"
            path = out_dir / f"{ticker.replace('=', '_')}.csv"
            df.to_csv(path)
            print(f"  saved {len(df)} rows -> {path}")


if __name__ == "__main__":
    download_all()
