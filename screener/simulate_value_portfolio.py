"""Bot #42 Valeur (2026-10-12): owns the 15 cheapest companies of their own sector, and nothing else
-- no quality floor, no momentum, no stop-loss. Same mechanics as Bot #41 Qualite (quarterly
rebalance, top-KEEP_RANK buffer, sector cap, fills/accounting of sleeve_book.py), imported from
simulate_quality_portfolio.py; only the ranking differs.

Why not the screener's valuation_gap: it is eps * peer P/E * quality_multiplier / price - 1, i.e.
already a value x quality blend (rank correlation 0.41 with note_qualite_20 on 2026-10-12), and the
point of this bot is a cheapness signal independent of #41. VALUE_SCORE is the plain mean of three
within-sector percentiles (higher = cheaper), the classic academic value measures:
  - earnings yield 1/PE, book yield 1/PB (PE and PB > 0 are guaranteed by the screener's universe);
  - free-cash-flow yield free_cashflow / market_cap, only when the filer reports in its listing
    currency (TSM-style mismatches give nonsense yields, see quality_perspective_notes.py) and never
    for Financial Services (a bank's cash flow statement isn't a free cash flow).
At least 2 of the 3 must exist. Expect this list to look unappealing (falling, low-quality names):
that is what a value strategy holds, and the 20-year ETF check of 2026-10-12 showed US value
lagging growth by ~4.7 pts/year from 2007 to 2026 -- a long test, not a quick one.
"""
# Alias lisibilite (mapping perso) : V1 -- strategie Valeur seule
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.simulate_quality_portfolio import UNIVERSE_COLUMNS, run_factor_bot  # noqa: E402

FCF_EXCLUDED_SECTORS = {"Financial Services"}
MIN_METRICS = 2


def value_ranking(universe: pd.DataFrame) -> pd.DataFrame:
    df = universe.copy()
    df["earnings_yield"] = 1 / df["pe"]
    df["book_yield"] = 1 / df["pb"]
    fcf_ok = (df["financial_currency"] == df["currency"]) & ~df["sector"].isin(FCF_EXCLUDED_SECTORS)
    df["fcf_yield"] = (df["free_cashflow"] / df["market_cap"]).where(fcf_ok)
    metrics = ["earnings_yield", "book_yield", "fcf_yield"]
    pct = df.groupby("sector")[metrics].rank(pct=True)
    df["score"] = pct.mean(axis=1).where(pct.notna().sum(axis=1) >= MIN_METRICS)
    df = df.dropna(subset=["score"])[UNIVERSE_COLUMNS + ["score"]]
    return df.sort_values(["score", "ticker"], ascending=[False, True]).reset_index(drop=True)


def main():
    run_factor_bot("value", "Bot #42 Valeur", value_ranking)


if __name__ == "__main__":
    main()
