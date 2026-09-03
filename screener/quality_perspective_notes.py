"""Two-note quality/perspective read on any cached ticker, for humans, not the bot: reduces
a stock to a NOTE_QUALITE (/20) and a NOTE_PERSPECTIVE (/20) so the user can understand a
pick at a glance -- inspired by an Instagram account's stock-analysis format the user showed
(2026-09-03): a Bilan/Rendement du capital/Cash/Marges/Discipline quality score plus a
Valorisation/Momentum perspective score.

DELIBERATELY NOT a filter: nothing here feeds into value_momentum_quality_screener_v2.py's
passes_filter/compute_valuation, or select_top_picks.py's composite_score/select_diversified.
Those are already calibrated and under observation with too few closed trades (see the
~30-trade checkpoint) to safely fold in new dimensions -- this module is read-only annotation
on top of whatever the existing pipeline already decided to buy.

Design principles, established over several rounds of testing against fundamentals_cache.csv
(2026-09-03):

1. A note must reflect reality, not exist for its own sake. A pillar that isn't
   structurally measurable for a whole sector/company -- bank leverage & free cash flow
   (yfinance simply doesn't populate debtToEquity/freeCashflow/quickRatio for financial-
   services filers) or a foreign filer whose cashflow-statement currency differs from its
   listing currency (e.g. TSM reports in TWD, trades and is capped in USD) -- is EXCLUDED
   (None/N/A) from that ticker's average. Neither a bonus nor a penalty: comparing a bank's
   leverage to a tech company's on the same yardstick wouldn't mean anything.

2. A real, comparable absence must show up as a genuinely low score, not be silently
   dropped -- UNLESS the absence itself can't be interpreted. Concretely: a company that has
   NEVER done a single buyback or paid a single dividend (confirmed by testing on
   TSLA/RIVN: the "Repurchase Of Capital Stock" row doesn't exist in their cashflow
   statement at all, not just this year) is excluded from Discipline rather than scored,
   because a true zero on both could mean "100% reinvested in growth" (arguably good) just
   as much as "returns nothing to shareholders" (arguably bad) -- this specific pillar has
   no way to tell those two apart. Contrast with a company doing SOME buybacks/dividends but
   less than its peers: that DOES get a low (not excluded) percentile score, because the
   comparison is meaningful there.

3. Percentiles are computed within-sector across the full fundamentals_cache.csv universe
   (thousands of tickers per sector once backfilled), not some fixed absolute threshold --
   otherwise a structurally capital-intensive sector (utilities, REITs: routinely negative
   discretionary free cash flow because of continuous capex) reads as "bad Cash" against a
   tech-company yardstick when it's actually normal for that sector. Comparing utilities to
   other utilities fixes this without special-casing the sector.

4. Two years of iterating past the naive fields already caught a currency bug (freeCashflow
   reported in the filer's home currency divided by a market_cap reported in USD -- a
   nonsense 34% "yield" for TSM before this was guarded), a stock-split false positive
   (dilution-by-share-count read NVDA's June-2024 10:1 split as a 10x share dump -- avoided
   entirely here by measuring Discipline in $ returned, never in share count, so no split
   adjustment is needed at all), and a small-sample percentile clustering artifact (with only
   1-2 tickers per sector group, rank(pct=True) can only output a couple of discrete values
   -- confirmed to smooth out once the group has dozens+ members, see the 25-ticker Technology
   test in the design conversation). None of those are re-derivable from the code alone, so
   they're recorded here.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

CACHE_PATH = HERE / "results/screener/fundamentals_cache.csv"
OUT_PATH = HERE / "results/screener/quality_perspective_notes.csv"

# Cash stays excluded for Financial Services even though total_cash/total_debt AREN'T NaN
# for banks (unlike debt_eq/quick_ratio, which yfinance leaves genuinely empty and therefore
# self-excludes via NaN without needing this list) -- a bank's "cash" and "debt" are mostly
# customer deposits, not leverage in the normal sense, so the ratio computes a real number
# that means something different from every other sector's. Bilan needs no entry here: both
# its sub-signals (debt_eq, quick_ratio) are natively absent for banks, so it falls out to
# N/A on its own.
SECTOR_NA = {"Cash": {"Financial Services"}}


def _score_from_pct(pct: pd.Series, invert: bool = False) -> pd.Series:
    p = 1 - pct if invert else pct
    return (p * 20).round(1)


def _score_from_ratio(value: pd.Series, ceiling: float, floor: float = 0.0) -> pd.Series:
    p = ((value - floor) / (ceiling - floor)).clip(0.0, 1.0)
    return (p * 20).round(1)


def _blend(*scores: pd.Series) -> pd.Series:
    """Row-wise mean across sub-score columns, ignoring whichever are NaN for that row --
    NaN only if ALL of them are (pandas .mean(skipna=True) default)."""
    return pd.concat(scores, axis=1).mean(axis=1).round(1)


# Fields added to fetch_cache.py's schema on 2026-09-03 -- absent as COLUMNS (not just NaN
# values) from any fundamentals_cache.csv written before that migration's first run. Guarded
# here so this module works immediately (everything reads as N/A) rather than crashing until
# the weekly rotation has refetched at least one ticker under the new schema.
NEW_SCHEMA_COLUMNS = ["free_cashflow", "total_cash", "total_debt", "quick_ratio",
                       "operating_margin", "financial_currency", "buyback_avg_3y", "div_avg_3y"]


def compute_notes(cache: pd.DataFrame) -> pd.DataFrame:
    df = cache[cache["error"].isna()].copy()
    for col in NEW_SCHEMA_COLUMNS:
        if col not in df.columns:
            # float NaN, not None/object -- object-dtype columns break arithmetic (division
            # against an all-None column raises ZeroDivisionError instead of yielding NaN)
            df[col] = np.nan if col != "financial_currency" else pd.Series(pd.NA, index=df.index, dtype="object")

    # --- percentiles on fields already used by the real screener (full-universe sample) ---
    df["roe_pct"] = df.groupby("sector")["roe"].rank(pct=True)
    df["margin_pct"] = df.groupby("sector")["margin"].rank(pct=True)
    df["debt_eq_pct"] = df.groupby("sector")["debt_eq"].rank(pct=True)
    df["mom_pct"] = df.groupby("sector")["mom_12_2"].rank(pct=True)
    peer_median_pe = df.groupby("industry")["pe"].transform("median")
    df["pe_vs_peer"] = df["pe"] / peer_median_pe

    # --- Cash/Bilan/Marges/Discipline raw ratios (fields added to fetch_cache.py 2026-09-03,
    # will read as NaN for any row not yet refetched since then -- self-heals as the weekly
    # rotation backfills the cache, same mechanism as every prior schema migration) ---
    df["fcf_yield"] = df["free_cashflow"] / df["market_cap"]
    df["net_cash_ratio"] = (df["total_cash"] - df["total_debt"]) / df["market_cap"]
    df["capital_return_yield"] = (df["buyback_avg_3y"].abs().fillna(0)
                                   + df["div_avg_3y"].abs().fillna(0)) / df["market_cap"]
    both_missing = df["buyback_avg_3y"].isna() & df["div_avg_3y"].isna()
    df.loc[both_missing, "capital_return_yield"] = None
    df.loc[df["capital_return_yield"] == 0, "capital_return_yield"] = None  # principle #2:
    # a genuine zero on both sides is excluded, not scored -- see module docstring

    currency_mismatch = (df["financial_currency"].notna() & df["currency"].notna()
                          & (df["financial_currency"] != df["currency"]))
    df.loc[currency_mismatch, ["fcf_yield", "net_cash_ratio", "capital_return_yield"]] = None

    df["fcf_yield_pct"] = df.groupby("sector")["fcf_yield"].rank(pct=True)
    df["net_cash_pct"] = df.groupby("sector")["net_cash_ratio"].rank(pct=True)
    df["quick_ratio_pct"] = df.groupby("sector")["quick_ratio"].rank(pct=True)
    df["operating_margin_pct"] = df.groupby("sector")["operating_margin"].rank(pct=True)
    df["capital_return_pct"] = df.groupby("sector")["capital_return_yield"].rank(pct=True)

    # --- 5 piliers Qualite ---
    df["pilier_bilan"] = _blend(_score_from_pct(df["debt_eq_pct"], invert=True),
                                 _score_from_pct(df["quick_ratio_pct"]))
    df["pilier_rendement_capital"] = _score_from_pct(df["roe_pct"])
    df["pilier_marges"] = _blend(_score_from_pct(df["operating_margin_pct"]),
                                  _score_from_pct(df["margin_pct"]))
    cash_score = _blend(_score_from_pct(df["fcf_yield_pct"]), _score_from_pct(df["net_cash_pct"]))
    cash_na = df["sector"].isin(SECTOR_NA["Cash"])
    df["pilier_cash"] = cash_score.where(~cash_na, None)
    df["pilier_discipline"] = _score_from_pct(df["capital_return_pct"])

    q_cols = ["pilier_bilan", "pilier_rendement_capital", "pilier_cash", "pilier_marges", "pilier_discipline"]
    df["note_qualite_n_piliers"] = df[q_cols].notna().sum(axis=1)
    df["note_qualite_20"] = df[q_cols].mean(axis=1).round(1)
    df["note_qualite_low_confidence"] = df["note_qualite_n_piliers"] <= 2  # more than half of 5 missing

    # --- 2 piliers Perspective ---
    # pe_vs_peer=0.5 (moitie du P/E median de l'industrie) -> 20, pe_vs_peer=1.5 (50% plus
    # cher) -> 0. P/E negatif ou absent -> N/A (pas de resultat net positif = pas de P/E qui
    # veut dire quelque chose, pas "valorisation nulle")
    valo_score = _score_from_ratio(-df["pe_vs_peer"], ceiling=-0.5, floor=-1.5)
    pe_invalid = df["pe_vs_peer"].isna() | df["pe"].isna() | (df["pe"] <= 0)
    df["pilier_valorisation"] = valo_score.where(~pe_invalid, None)
    df["pilier_momentum"] = _score_from_pct(df["mom_pct"])

    p_cols = ["pilier_valorisation", "pilier_momentum"]
    df["note_perspective_n_piliers"] = df[p_cols].notna().sum(axis=1)
    df["note_perspective_20"] = df[p_cols].mean(axis=1).round(1)
    df["note_perspective_low_confidence"] = df["note_perspective_n_piliers"] <= 1  # 1 of 2 missing
    # means the note reflects only a single angle, not a real blend

    out_cols = ["ticker", "name", "sector", "industry"] + q_cols + [
        "note_qualite_20", "note_qualite_n_piliers", "note_qualite_low_confidence"] + p_cols + [
        "note_perspective_20", "note_perspective_n_piliers", "note_perspective_low_confidence"]
    return df[out_cols]


def main():
    cache = pd.read_csv(CACHE_PATH)
    notes = compute_notes(cache)
    notes.to_csv(OUT_PATH, index=False)

    coverage = notes["note_qualite_20"].notna().mean()
    print(f"{len(notes)} tickers notes | note qualite calculable pour {coverage:.0%} "
          f"(le reste attend son backfill de champs via la rotation hebdo de fetch_cache.py)")
    print(f"Ecrit dans {OUT_PATH}")

    sample = notes.dropna(subset=["note_qualite_20"]).sort_values("note_qualite_20", ascending=False).head(10)
    print("\nTop 10 par note qualite (parmi les tickers deja backfilles) :")
    print(sample[["ticker", "sector", "note_qualite_20", "note_qualite_n_piliers",
                   "note_perspective_20", "note_perspective_n_piliers"]].to_string(index=False))


if __name__ == "__main__":
    main()
