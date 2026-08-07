"""Proof of concept: compare the user's actual current stock holdings against the
value+momentum screener ranking, to see which positions already look well-placed and
which look like rotation candidates. Crypto and real-estate crowdfunding positions are
intentionally excluded (out of scope per the user's own direction)."""
import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
from screener.value_momentum_screener import fetch_one, score  # noqa: E402

HERE = pathlib.Path(__file__).parent.parent
PORTFOLIO_JSON = HERE / "data/portfolio_backup_2026-08-07.json"
SCREENER_CSV = sorted((HERE / "results/screener").glob("screener_*_full.csv"))[-1]

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
    universe = pd.read_csv(SCREENER_CSV)
    universe_scored = score(universe.assign(error=None))

    print(f"Positions actions/ETF actuelles (hors crypto/immo) : {len(holdings)}")
    rows = []
    for ticker, value_eur in holdings.items():
        match = universe_scored[universe_scored["ticker"] == ticker]
        if len(match):
            row = match.iloc[0].to_dict()
            row["in_universe"] = True
            rank = universe_scored.index.get_loc(match.index[0]) + 1
            row["rank"] = f"{rank}/{len(universe_scored)}"
        else:
            # not in the S&P500+Euronext100 universe (or filtered out, e.g. negative ROE) -- fetch live
            fetched = fetch_one(ticker)
            fetched["in_universe"] = False
            fetched["rank"] = "hors univers / filtre"
            row = fetched
        row["valeur_eur"] = value_eur
        rows.append(row)

    report = pd.DataFrame(rows)
    report.to_csv(HERE / "results/portfolio_vs_screener.csv", index=False)

    cols = ["ticker", "valeur_eur", "pb", "roe", "mom_12_2", "score", "rank"]
    display_cols = [c for c in cols if c in report.columns]
    pd.set_option("display.width", 160)
    print("\n=== Tes positions actuelles vs le classement value+momentum ===")
    print(report[display_cols].sort_values("score", ascending=False, na_position="last").to_string(
        index=False,
        formatters={
            "valeur_eur": "{:.2f}".format,
            "pb": lambda x: f"{x:.2f}" if pd.notna(x) else "n/a",
            "roe": lambda x: f"{x:.1%}" if pd.notna(x) else "n/a",
            "mom_12_2": lambda x: f"{x:+.1%}" if pd.notna(x) else "n/a",
            "score": lambda x: f"{x:.2f}" if pd.notna(x) else "n/a",
        }))

    top30 = universe_scored.head(30)
    not_held = top30[~top30["ticker"].isin(holdings.keys())]
    print(f"\n=== Dans le top 30 mais absent de ton portefeuille ({len(not_held)}) ===")
    print(not_held[["ticker", "name", "pb", "mom_12_2", "score"]].to_string(
        index=False, formatters={"pb": "{:.2f}".format, "mom_12_2": "{:+.1%}".format, "score": "{:.2f}".format}))


if __name__ == "__main__":
    main()
