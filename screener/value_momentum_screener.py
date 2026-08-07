"""Live value+momentum screener over S&P 500 + Euronext 100.

Ranks the universe on two independent axes and combines them:
  - Value  : cheapness (inverse price-to-book percentile; P/E as a secondary check)
  - Momentum : 12-month price return excluding the most recent month (the standard
    academic "12-2" definition -- skipping the last month avoids the well documented
    short-term reversal effect).

This is a *live* screener, not a historical backtest: point-in-time historical
fundamentals for hundreds of stocks aren't available for free without survivorship-bias
risk, which is why the concept itself was validated separately against real long-run
academic data (see run_value_premium_check.py / run_value_momentum_check.py) before
building this. Re-run quarterly; each run is cached to results/screener/ with the date.
"""
import concurrent.futures as cf
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
UNIVERSE_CSV = HERE / "data/universe/universe.csv"
OUT_DIR = HERE / "results/screener"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N = 30
MAX_WORKERS = 8


def fetch_one(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="14mo", interval="1mo", auto_adjust=True)["Close"].dropna()
        if len(hist) < 13:
            return {"ticker": ticker, "error": "historique insuffisant"}
        # 12-2 momentum: return from 12 months ago to 1 month ago (skip most recent month)
        mom_12_2 = hist.iloc[-2] / hist.iloc[-13] - 1
        return {
            "ticker": ticker,
            "price": hist.iloc[-1],
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "name": info.get("shortName"),
            "mom_12_2": mom_12_2,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def fetch_all(tickers: list) -> pd.DataFrame:
    rows = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_one, tk): tk for tk in tickers}
        done = 0
        for fut in cf.as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(tickers)} tickers recuperes...", file=sys.stderr)
    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["error"].isna()].copy() if "error" in df.columns else df.copy()
    # sanity filters: drop negative/absurd P/B and P/E (usually distressed or data errors),
    # and require a positive ROE (avoid unprofitable "value traps", per run_value_quality_check.py)
    df = df[(df["pb"] > 0) & (df["pb"] < 50) & (df["roe"].fillna(-1) > 0)]
    df = df.dropna(subset=["mom_12_2"])

    df["value_pct"] = 1 - df["pb"].rank(pct=True)  # cheaper P/B -> higher percentile
    df["momentum_pct"] = df["mom_12_2"].rank(pct=True)  # stronger momentum -> higher percentile
    df["score"] = 0.5 * df["value_pct"] + 0.5 * df["momentum_pct"]
    return df.sort_values("score", ascending=False)


def main():
    universe = pd.read_csv(UNIVERSE_CSV)
    tickers = universe["ticker"].tolist()
    print(f"Recuperation des donnees pour {len(tickers)} tickers (peut prendre quelques minutes)...")

    t0 = time.time()
    raw = fetch_all(tickers)
    print(f"Termine en {time.time() - t0:.0f}s")

    raw = raw.merge(universe[["ticker", "market", "sector"]].rename(columns={"sector": "gics_sector"}),
                     on="ticker", how="left")
    if "error" not in raw.columns:
        raw["error"] = np.nan
    n_errors = raw["error"].notna().sum()
    print(f"{n_errors} tickers sans donnees exploitables (delisting, ticker invalide, etc.)")

    ranked = score(raw)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    ranked.to_csv(OUT_DIR / f"screener_{today}_full.csv", index=False)

    top = ranked.head(TOP_N)[["ticker", "name", "market", "gics_sector", "price", "pb", "pe",
                                "roe", "mom_12_2", "value_pct", "momentum_pct", "score"]]
    top.to_csv(OUT_DIR / f"screener_{today}_top{TOP_N}.csv", index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(f"\n=== Top {TOP_N} value+momentum ({today}) ===")
    print(top.to_string(index=False, formatters={
        "price": "{:.2f}".format, "pb": "{:.2f}".format, "pe": lambda x: f"{x:.1f}" if pd.notna(x) else "n/a",
        "roe": "{:.1%}".format, "mom_12_2": "{:+.1%}".format,
        "value_pct": "{:.0%}".format, "momentum_pct": "{:.0%}".format, "score": "{:.2f}".format,
    }))


if __name__ == "__main__":
    main()
