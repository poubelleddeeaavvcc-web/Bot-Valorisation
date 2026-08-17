"""Proof of concept: compare the user's actual current stock holdings against the
value+momentum+quality screener's live ranking, to see which positions already look
well-placed and which look like rotation candidates. Crypto and real-estate crowdfunding
positions are intentionally excluded (out of scope per the user's own direction).

Reads results/screener/full_valuation_latest.csv -- the same output the live hourly
pipeline (fetch_cache.py -> analyze_cache.py) produces -- rather than a hand-run,
never-refreshed snapshot from the older, simpler value_momentum_screener.py. That older
path had gone stale (last regenerated 2026-08-07) and used a different, cruder model
(plain value+momentum percentile, no PEG/quality/sector-relative fair value, no
cross-listing dedup), so this comparison had drifted out of sync with what the bot
actually does. full_valuation_latest.csv is already deduped (see CROSS_LISTING_GROUPS in
value_momentum_quality_screener_v2.py) so no extra dedup work is needed here.
"""
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from screener.value_momentum_quality_screener_v2 import fetch_one  # noqa: E402

HERE = pathlib.Path(__file__).parent.parent
PORTFOLIO_JSON = HERE / "data/portfolio_backup_2026-08-07.json"
VALUATION_CSV = HERE / "results/screener/full_valuation_latest.csv"

# the portfolio's internal "tk" codes aren't all Yahoo Finance tickers (European
# small/mid caps in particular need an exchange suffix) -- mapped by hand from the name.
TICKER_FIXES = {
    "TFI": "TFI.PA",      # TF1
    "DG": "DG.PA",         # Vinci
    "AM": "AM.PA",         # Dassault Aviation
    "ML": "ML.PA",         # Michelin
    "THEP": "THEP.PA",    # Thermador
    "HAG": "HAG.DE",       # Hensoldt (Frankfurt, not Euronext)
    "BNP": "BNP.PA",
    "TTE": "TTE.PA",
    "ASML": "ASML.AS",
}
EXCLUDE_TYPES = {"crypto", "autre"}
EXCLUDE_TICKERS = {"ARTINT"}  # ETF, not a single-company value/momentum candidate


def load_current_holdings():
    data = json.loads(PORTFOLIO_JSON.read_text(encoding="utf-8"))
    holdings = {}
    for a in data["actions"]:
        if a["t"] in EXCLUDE_TYPES or a["tk"] in EXCLUDE_TICKERS:
            continue
        ticker = TICKER_FIXES.get(a["tk"], a["tk"])
        holdings[ticker] = holdings.get(ticker, 0) + a.get("val", 0)  # merge duplicates (e.g. TTE on 2 brokers)
    return holdings


def main():
    holdings = load_current_holdings()
    universe = pd.read_csv(VALUATION_CSV).sort_values("valuation_gap", ascending=False).reset_index(drop=True)
    assert not universe["ticker"].duplicated().any(), "doublon inattendu dans full_valuation_latest.csv"

    print(f"Positions actions/ETF actuelles (hors crypto/immo) : {len(holdings)}")
    rows = []
    for ticker, value_eur in holdings.items():
        match = universe[universe["ticker"] == ticker]
        if len(match):
            row = match.iloc[0].to_dict()
            row["in_universe"] = True
            rank = universe.index.get_loc(match.index[0]) + 1
            row["rank"] = f"{rank}/{len(universe)}"
        else:
            # not in the live-tracked universe (or filtered out earlier, e.g. cap trop
            # petite) -- fetch live. No sector context for a lone ticker, so this can't
            # compute a valuation_gap/passes_filter the way the batch pipeline does.
            fetched = fetch_one(ticker)
            fetched["in_universe"] = False
            fetched["rank"] = "hors univers / filtre"
            row = fetched
        row["valeur_eur"] = value_eur
        rows.append(row)

    report = pd.DataFrame(rows)
    assert not report["ticker"].duplicated().any(), "doublon inattendu dans le rapport (holdings dedup a echoue ?)"
    report.to_csv(HERE / "results/portfolio_vs_screener.csv", index=False)

    cols = ["ticker", "valeur_eur", "pe", "peg", "roe", "mom_12_2", "valuation_gap", "passes_filter", "rank"]
    display_cols = [c for c in cols if c in report.columns]
    pd.set_option("display.width", 160)
    print("\n=== Tes positions actuelles vs le classement value+momentum+qualite ===")
    print(report[display_cols].sort_values("valuation_gap", ascending=False, na_position="last").to_string(
        index=False,
        formatters={
            "valeur_eur": "{:.2f}".format,
            "pe": lambda x: f"{x:.1f}" if pd.notna(x) else "n/a",
            "peg": lambda x: f"{x:.2f}" if pd.notna(x) else "n/a",
            "roe": lambda x: f"{x:.1%}" if pd.notna(x) else "n/a",
            "mom_12_2": lambda x: f"{x:+.1%}" if pd.notna(x) else "n/a",
            "valuation_gap": lambda x: f"{x:+.1%}" if pd.notna(x) else "n/a",
        }))

    candidates = universe[universe["passes_filter"]]
    not_held = candidates[~candidates["ticker"].isin(holdings.keys())]
    print(f"\n=== Candidats LONG (tous filtres OK) absents de ton portefeuille ({len(not_held)}) ===")
    print(not_held[["ticker", "name", "pe", "peg", "mom_12_2", "valuation_gap"]].to_string(
        index=False, formatters={
            "pe": "{:.1f}".format, "peg": "{:.2f}".format,
            "mom_12_2": "{:+.1%}".format, "valuation_gap": "{:+.1%}".format,
        }))


if __name__ == "__main__":
    main()
