"""Bot#34 (blind + momentum entry margin): same "buy every LONG candidate, no ranking, no
capital constraint" mechanics as simulate_portfolio.py (Bot#1), except a candidate is only
bought if its mom_12_2 beats its sector's momentum by at least MIN_ENTRY_MOM_MARGIN at entry
(see that constant in simulate_portfolio.py for the 2026-10-03 analysis behind it). Exit rules
are untouched: a position still exits as soon as it falls back to its sector's momentum.

Everything else (fresh-check gate, exit rules, benchmarks) is Bot#1's, reused by import --
the margin is the only variable isolated against Bot#1.
"""
# Alias lisibilite (mapping perso) : A9 -- famille Alpha (aveugle), variante + marge momentum a l'entree
import pathlib
import sys

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.simulate_portfolio import (  # noqa: E402
    LEDGER_COLUMNS, BENCHMARKS, MIN_ENTRY_MOM_MARGIN, open_new_positions, recheck_open_positions,
)

LEDGER_PATH = HERE / "results/simulation/portfolio_ledger_marge.csv"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
SUMMARY_PATH = HERE / "results/simulation/summary_marge.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/equity_curve_marge.csv"


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH)
        for c in LEDGER_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[LEDGER_COLUMNS]
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def write_summary(ledger: pd.DataFrame):
    closed = ledger[ledger["status"] == "closed"]
    open_pos = ledger[ledger["status"] == "open"]
    summary = {
        "nb_open": len(open_pos),
        "nb_closed": len(closed),
        "win_rate_closed": float((closed["return_pct"] > 0).mean()) if len(closed) else None,
        "avg_return_closed": float(closed["return_pct"].mean()) if len(closed) else None,
        "avg_unrealized_open": float(open_pos["unrealized_return_pct"].mean()) if len(open_pos) else None,
    }
    SUMMARY_PATH.write_text(pd.Series(summary).to_json(), encoding="utf-8")
    print(f"\n=== Resume simulation (Bot#34, marge momentum) : {summary['nb_open']} ouvertes, "
          f"{summary['nb_closed']} cloturees ===")
    if summary["win_rate_closed"] is not None:
        print(f"Taux de reussite (cloturees) : {summary['win_rate_closed']:.0%} | "
              f"Retour moyen (cloturees) : {summary['avg_return_closed']:+.1%}")


def append_equity_curve_point(ledger: pd.DataFrame):
    closed = ledger[ledger["status"] == "closed"]["return_pct"]
    open_ = ledger[ledger["status"] == "open"]["unrealized_return_pct"]
    all_returns = pd.concat([closed, open_]).dropna()
    if len(all_returns) == 0:
        return
    row = {"timestamp": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
           "strategy_avg_return": all_returns.mean(),
           "n_open": int((ledger["status"] == "open").sum()),
           "n_closed": int((ledger["status"] == "closed").sum())}
    for col, bench_ticker in BENCHMARKS.items():
        try:
            price = yf.Ticker(bench_ticker).history(period="5d")["Close"].dropna().iloc[-1]
        except Exception as e:
            print(f"  echec fetch benchmark {bench_ticker}: {e}", file=sys.stderr)
            price = None
        row[col] = price

    header = not EQUITY_CURVE_PATH.exists()
    pd.DataFrame([row]).to_csv(EQUITY_CURVE_PATH, mode="a", header=header, index=False)


def main():
    if not CANDIDATES_PATH.exists() or not VALUATION_PATH.exists():
        print("Pas encore de resultats de screener -- rien a simuler.")
        return
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    candidates = pd.read_csv(CANDIDATES_PATH)
    valuation = pd.read_csv(VALUATION_PATH)

    ledger = load_ledger()
    ledger, newly_opened = open_new_positions(ledger, candidates, valuation, today,
                                              min_mom_margin=MIN_ENTRY_MOM_MARGIN)
    ledger = recheck_open_positions(ledger, valuation, today, skip_tickers=newly_opened)

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    write_summary(ledger)
    append_equity_curve_point(ledger)


if __name__ == "__main__":
    main()
