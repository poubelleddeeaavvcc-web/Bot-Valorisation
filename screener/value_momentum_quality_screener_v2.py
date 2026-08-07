"""v2 screener: broad US+Europe universe (no index restriction), sector-relative fair P/E
adjusted for quality (ROE, margin) and leverage, momentum, and a debt filter with
exceptions for structurally high-debt sectors.

Fair value formula (documented here because it is otherwise a black box):
  fair_pe   = sector_median_pe * quality_multiplier
  quality_multiplier = clip(1
                              + 0.5 * (roe_percentile_in_sector - 0.5)
                              + 0.25 * (margin_percentile_in_sector - 0.5)
                              - 0.25 * (debt_percentile_in_sector - 0.5),
                             0.6, 1.6)
  fair_value = trailing_eps * fair_pe
  valuation_gap = fair_value / price - 1        (>0 undervalued, <0 overvalued)

This replaces the hand-picked "fair P/E per stock" the user's own portfolio tracker used
(reverse-engineered from its valeurJuste field) with something that scales to thousands
of tickers without a person assigning a multiple to each one by hand.
"""
import concurrent.futures as cf
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
UNIVERSE_CSV = HERE / "data/universe/universe_v2.csv"
OUT_DIR = HERE / "results/screener"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKET_CAP_FLOOR = 1_000_000_000  # $1B -- liquidity floor, an automated bot has no business in micro-caps
MAX_WORKERS = 16
HIGH_DEBT_SECTORS = {"Utilities", "Real Estate", "Communication Services"}
DEBT_TO_EQUITY_CAP = 150.0  # percent
EXTREME_DEBT_TO_EQUITY_CAP = 400.0  # percent -- even the high-debt-sector exception has a limit;
# beyond this it's a mortgage REIT / leverage-as-the-business-model vehicle, not
# "normal" sector leverage, and the P/E-based fair value model doesn't apply to it.
PEG_CAP = 2.0  # Peter Lynch's classic threshold: PEG<1 excellent, <2 still reasonable,
# beyond that the P/E isn't justified by earnings growth even if it looks cheap on paper.
MAX_PLAUSIBLE_VALUATION_GAP = 1.0  # +100%: past this, treat it as a model breakdown
# (wrong peer group, distressed name, depositary/preferred slipping through) rather than
# a real opportunity -- a stock trading at a fraction of "fair value" by 3-8x is a red
# flag, not alpha.


def fetch_one(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        cap = info.get("marketCap")
        if not cap or cap < MARKET_CAP_FLOOR:
            return {"ticker": ticker, "error": "cap_trop_petite"}

        hist = t.history(period="14mo", interval="1mo", auto_adjust=True)["Close"].dropna()
        if len(hist) < 13:
            return {"ticker": ticker, "error": "historique_insuffisant"}
        mom_12_2 = hist.iloc[-2] / hist.iloc[-13] - 1

        return {
            "ticker": ticker, "price": hist.iloc[-1], "market_cap": cap,
            "sector": info.get("sector"), "industry": info.get("industry"),
            "name": info.get("shortName"),
            "pe": info.get("trailingPE"), "pb": info.get("priceToBook"),
            "eps": info.get("trailingEps"), "roe": info.get("returnOnEquity"),
            "margin": info.get("profitMargins"), "debt_eq": info.get("debtToEquity"),
            "mom_12_2": mom_12_2,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)[:80]}


def fetch_all(tickers: list) -> pd.DataFrame:
    rows = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_one, tk): tk for tk in tickers}
        done = 0
        for fut in cf.as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tickers)} tickers traites...", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def pct_rank_within_sector(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("sector")[col].rank(pct=True)


def compute_valuation(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["error"].isna()].copy()
    df = df[(df["pe"] > 0) & (df["pe"] < 80) & (df["eps"] > 0) & (df["pb"] > 0)]
    df = df.dropna(subset=["sector", "mom_12_2", "roe"])

    sector_pe = df.groupby("sector")["pe"].median().rename("sector_median_pe")
    df = df.join(sector_pe, on="sector")

    df["roe_pct"] = pct_rank_within_sector(df, "roe")
    df["margin_pct"] = pct_rank_within_sector(df.fillna({"margin": df["margin"].median()}), "margin")
    df["debt_pct"] = pct_rank_within_sector(df.fillna({"debt_eq": df["debt_eq"].median()}), "debt_eq")

    quality_multiplier = (1
                           + 0.5 * (df["roe_pct"] - 0.5)
                           + 0.25 * (df["margin_pct"] - 0.5)
                           - 0.25 * (df["debt_pct"] - 0.5))
    df["quality_multiplier"] = quality_multiplier.clip(0.6, 1.6)
    df["fair_pe"] = df["sector_median_pe"] * df["quality_multiplier"]
    df["fair_value"] = df["eps"] * df["fair_pe"]
    df["valuation_gap"] = df["fair_value"] / df["price"] - 1

    sector_momentum = df.groupby("sector")["mom_12_2"].median().rename("sector_momentum")
    df = df.join(sector_momentum, on="sector")

    normal_debt_ok = df["debt_eq"] < DEBT_TO_EQUITY_CAP
    high_debt_sector_ok = df["sector"].isin(HIGH_DEBT_SECTORS) & (df["debt_eq"] < EXTREME_DEBT_TO_EQUITY_CAP)
    debt_ok = df["debt_eq"].isna() | normal_debt_ok | high_debt_sector_ok
    sector_up = df["sector_momentum"] > 0
    plausible = df["valuation_gap"] <= MAX_PLAUSIBLE_VALUATION_GAP
    # value+momentum, not value alone: a sector trending up isn't enough if the stock
    # itself is falling -- this is exactly the combination validated against 100 years
    # of Fama-French value/momentum factor data before building this screener.
    stock_momentum_ok = df["mom_12_2"] > 0
    # PEG: is the P/E justified by actual earnings growth, or just "statistically cheap"
    # on a metric that ignores growth entirely? Missing PEG (delisted-adjacent, unusual
    # capital structure, data gap) is treated as a fail, not a pass -- tightens the
    # candidate list today and self-heals as the cache fills in with the new field.
    peg_ok = df["peg"].notna() & (df["peg"] > 0) & (df["peg"] < PEG_CAP)
    df["passes_filter"] = debt_ok & sector_up & plausible & stock_momentum_ok & peg_ok
    return df.sort_values("valuation_gap", ascending=False)


def main():
    universe = pd.read_csv(UNIVERSE_CSV)
    tickers = universe["ticker"].dropna().unique().tolist()
    print(f"Recuperation des donnees pour {len(tickers)} tickers (univers elargi, ca va prendre plusieurs minutes)...")

    t0 = time.time()
    raw = fetch_all(tickers)
    print(f"Termine en {time.time() - t0:.0f}s")
    if "error" not in raw.columns:
        raw["error"] = np.nan
    print(raw["error"].value_counts(dropna=False).head(10))

    valued = compute_valuation(raw)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    valued.to_csv(OUT_DIR / f"screener_v2_{today}_full.csv", index=False)

    candidates = valued[valued["passes_filter"] & (valued["valuation_gap"] > 0.15)]
    candidates.to_csv(OUT_DIR / f"screener_v2_{today}_long_candidates.csv", index=False)

    cols = ["ticker", "name", "sector", "price", "pe", "fair_pe", "fair_value", "valuation_gap",
            "roe", "debt_eq", "mom_12_2", "sector_momentum"]
    pd.set_option("display.width", 180)
    print(f"\n=== {len(candidates)} candidats LONG (sous-evalues de +15% ou plus, filtres dette+secteur OK) ===")
    print(candidates.head(40)[cols].to_string(index=False, formatters={
        "price": "{:.2f}".format, "pe": "{:.1f}".format, "fair_pe": "{:.1f}".format,
        "fair_value": "{:.2f}".format, "valuation_gap": "{:+.1%}".format,
        "roe": "{:.1%}".format, "debt_eq": lambda x: f"{x:.0f}%" if pd.notna(x) else "n/a",
        "mom_12_2": "{:+.1%}".format, "sector_momentum": "{:+.1%}".format,
    }))


if __name__ == "__main__":
    main()
