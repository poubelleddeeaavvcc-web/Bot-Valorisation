"""Bot#7 (blind + notes gate): same "buy every LONG candidate, no ranking, no capital
constraint" mechanics as simulate_portfolio.py (Bot#1), except a candidate whose 2-note
quality/perspective read (screener/quality_perspective_notes.py) doesn't clear a minimum bar is
skipped entirely -- everything else (fresh-check gate, exit rules, benchmarks) is identical to
Bot#1 and reused directly by import, per this repo's "duplicate the one thing that changes,
import the rest" convention (see simulate_constrained_portfolio_notes.py's docstring for Bot#8,
the same idea applied to a different mechanic).

Why a threshold here and not a ranking (unlike Bot#8/#9, which rank by notes_score): Bot#1's
mechanic has no capital scarcity forcing a choice between candidates -- it buys everything that
clears its gate, so applying the notes improvement "blindly" means a pass/fail bar, not an order.
Per the user's direction (2026-09-04): buy only if note_qualite_20 > QUALITE_THRESHOLD AND
note_perspective_20 > PERSPECTIVE_THRESHOLD. Thresholds (12, 15) were calibrated against the
actual distribution in quality_perspective_notes.csv joined to long_candidates_latest.csv that
day (mean qualite ~13.9/std ~2.2, mean perspective ~14.5/std ~1.4): this combination passed 9/49
candidates (~18%) -- selective without being so tight the bot rarely buys anything. Fixed values,
not a percentile/relative cut, consistent with how the rest of the repo avoids implicitly-moving
thresholds. A candidate with no computable note (NaN on either pillar) is excluded, same rule as
Bot#8/#9.
"""
import pathlib
import sys

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.simulate_portfolio import (  # noqa: E402
    LEDGER_COLUMNS, BENCHMARKS, fetch_fresh_single, fails_fresh_check, recheck_open_positions,
)
from screener.simulate_constrained_portfolio_notes import _load_notes_score  # noqa: E402

LEDGER_PATH = HERE / "results/simulation/portfolio_ledger_notes.csv"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
NOTES_PATH = HERE / "results/screener/quality_perspective_notes.csv"
SUMMARY_PATH = HERE / "results/simulation/summary_notes.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/equity_curve_notes.csv"

QUALITE_THRESHOLD = 12
PERSPECTIVE_THRESHOLD = 15


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH)
        for c in LEDGER_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[LEDGER_COLUMNS]
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def open_new_positions(ledger: pd.DataFrame, candidates: pd.DataFrame, valuation: pd.DataFrame,
                        today: str) -> tuple:
    open_tickers = set(ledger.loc[ledger["status"] == "open", "ticker"])
    sector_pe = valuation.groupby("sector")["sector_median_pe"].first()
    sector_mom = valuation.groupby("sector")["sector_momentum"].first()
    industry_pe = valuation.groupby("industry")["industry_median_pe"].first()
    industry_count = valuation.groupby("industry")["industry_count"].first()

    scored = _load_notes_score(candidates)

    new_rows = []
    for _, c in scored.iterrows():
        if c["ticker"] in open_tickers:
            continue
        if pd.isna(c["note_qualite_20"]) or pd.isna(c["note_perspective_20"]):
            continue  # unscored candidates test nothing about the notes' predictive value --
            # dropped entirely rather than treated as a pass or a fail, same rule as Bot#8/#9
        if c["note_qualite_20"] <= QUALITE_THRESHOLD or c["note_perspective_20"] <= PERSPECTIVE_THRESHOLD:
            continue
        fresh = fetch_fresh_single(c["ticker"])
        if fresh is None or fresh["price"] is None or fresh["eps"] is None:
            print(f"  achat ignore {c['ticker']} : echec verification fraiche", file=sys.stderr)
            continue
        fails, state = fails_fresh_check(fresh, c["quality_multiplier"], sector_pe, sector_mom,
                                          industry_pe, industry_count,
                                          fallback_valuation_gap=c["valuation_gap"])
        if fails:
            print(f"  achat ecarte {c['ticker']} : ne passe plus le filtre momentum/valorisation "
                  f"en verification fraiche")
            continue
        new_rows.append({
            "ticker": c["ticker"], "name": c["name"], "sector": state["sector"] or c["sector"],
            "status": "open",
            "entry_date": today, "entry_price": fresh["price"], "entry_valuation_gap": state["valuation_gap"],
            "entry_quality_multiplier": c["quality_multiplier"], "entry_mom_12_2": fresh["mom_12_2"],
            "entry_sector_momentum": state["sector_momentum"],
            "last_check_date": today, "last_price": fresh["price"], "last_valuation_gap": state["valuation_gap"],
            "last_mom_12_2": fresh["mom_12_2"], "last_sector_momentum": state["sector_momentum"],
            "unrealized_return_pct": 0.0, "peak_unrealized_return_pct": 0.0, "peak_date": today,
            "exit_date": None, "exit_price": None, "exit_reason": None,
            "return_pct": None, "holding_days": None,
        })
    new_tickers = {r["ticker"] for r in new_rows}
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"{len(new_rows)} nouvelle(s) position(s) ouverte(s) (notes) : {sorted(new_tickers)}")
    return ledger, new_tickers


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
    print(f"\n=== Resume simulation (Bot#7, seuil notes) : {summary['nb_open']} ouvertes, "
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
    strategy_avg_return = all_returns.mean()

    row = {"timestamp": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
           "strategy_avg_return": strategy_avg_return,
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
    if not NOTES_PATH.exists():
        print("quality_perspective_notes.csv n'existe pas encore -- lancer "
              "screener/quality_perspective_notes.py d'abord. Rien a simuler ce run.")
        return
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    candidates = pd.read_csv(CANDIDATES_PATH)
    valuation = pd.read_csv(VALUATION_PATH)

    ledger = load_ledger()
    ledger, newly_opened = open_new_positions(ledger, candidates, valuation, today)
    ledger = recheck_open_positions(ledger, valuation, today, skip_tickers=newly_opened)

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    write_summary(ledger)
    append_equity_curve_point(ledger)


if __name__ == "__main__":
    main()
