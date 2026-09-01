"""Bot#4: same blind, no-hindsight paper-trading mechanics as simulate_portfolio.py (Bot#1,
kept deliberately unfiltered as the honest baseline -- see its module docstring), except
every candidate is passed through news_filter.news_verdict() before being bought. A
candidate flagged by the local Ollama model as having an obvious news red flag (fraud
investigation, profit warning, lawsuit, bankruptcy, delisting, ...) is skipped rather than
bought; everything else about entry and exit is identical to Bot#1, so any difference in
the two ledgers' performance isolates the effect of the news gate itself.

Exit rules, fresh single-ticker fetch, and equity-curve/benchmark logic are reused directly
from simulate_portfolio.py (recheck_open_positions, fetch_fresh_single, resolve_peer_pe,
BENCHMARKS, STOP_LOSS_PCT) -- none of those touch the ledger/candidate file paths, so
they're safe to import as-is. Only open_new_positions() (which decides what to buy) and the
path-coupled load/save/summary functions are duplicated here with this bot's own files, the
same pattern simulate_large_portfolio.py already uses to reuse simulate_constrained_portfolio.py.
"""
import pathlib
import sys

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.simulate_portfolio import (  # noqa: E402
    BENCHMARKS, STOP_LOSS_PCT, fetch_fresh_single, recheck_open_positions, resolve_peer_pe,
)
from screener import news_filter  # noqa: E402

LEDGER_PATH = HERE / "results/simulation/portfolio_ledger_newsgated.csv"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
SUMMARY_PATH = HERE / "results/simulation/summary_newsgated.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/equity_curve_newsgated.csv"

LEDGER_COLUMNS = [
    "ticker", "name", "sector", "status",
    "entry_date", "entry_price", "entry_valuation_gap", "entry_quality_multiplier",
    "entry_mom_12_2", "entry_sector_momentum",
    "last_check_date", "last_price", "last_valuation_gap", "last_mom_12_2", "last_sector_momentum",
    "unrealized_return_pct", "peak_unrealized_return_pct", "peak_date",
    "exit_date", "exit_price", "exit_reason", "return_pct", "holding_days",
    "news_source", "news_reason",
]


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH)
        for c in LEDGER_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[LEDGER_COLUMNS]
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def open_new_positions(ledger: pd.DataFrame, candidates: pd.DataFrame, today: str) -> pd.DataFrame:
    open_tickers = set(ledger.loc[ledger["status"] == "open", "ticker"])
    new_rows = []
    skipped = []
    for _, c in candidates.iterrows():
        if c["ticker"] in open_tickers:
            continue
        verdict = news_filter.news_verdict(c["ticker"], c["name"], c["sector"], today)
        if not verdict["relevant"]:
            skipped.append((c["ticker"], verdict["reason"]))
            continue
        new_rows.append({
            "ticker": c["ticker"], "name": c["name"], "sector": c["sector"], "status": "open",
            "entry_date": today, "entry_price": c["price"], "entry_valuation_gap": c["valuation_gap"],
            "entry_quality_multiplier": c["quality_multiplier"], "entry_mom_12_2": c["mom_12_2"],
            "entry_sector_momentum": c["sector_momentum"],
            "last_check_date": today, "last_price": c["price"], "last_valuation_gap": c["valuation_gap"],
            "last_mom_12_2": c["mom_12_2"], "last_sector_momentum": c["sector_momentum"],
            "unrealized_return_pct": 0.0, "peak_unrealized_return_pct": 0.0, "peak_date": today,
            "exit_date": None, "exit_price": None, "exit_reason": None,
            "return_pct": None, "holding_days": None,
            "news_source": verdict["source"], "news_reason": verdict["reason"],
        })
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"{len(new_rows)} nouvelle(s) position(s) ouverte(s) : {[r['ticker'] for r in new_rows]}")
    for ticker, reason in skipped:
        print(f"  SKIP {ticker} (actu Ollama) : {reason}")
    return ledger


def write_summary(ledger: pd.DataFrame):
    closed = ledger[ledger["status"] == "closed"]
    open_pos = ledger[ledger["status"] == "open"]
    summary = {
        "nb_open": len(open_pos),
        "nb_closed": len(closed),
        "win_rate_closed": float((closed["return_pct"] > 0).mean()) if len(closed) else None,
        "avg_return_closed": float(closed["return_pct"].mean()) if len(closed) else None,
        "avg_unrealized_open": float(open_pos["unrealized_return_pct"].mean()) if len(open_pos) else None,
        "best_closed": closed.loc[closed["return_pct"].idxmax()][["ticker", "return_pct"]].to_dict() if len(closed) else None,
        "worst_closed": closed.loc[closed["return_pct"].idxmin()][["ticker", "return_pct"]].to_dict() if len(closed) else None,
    }
    SUMMARY_PATH.write_text(pd.Series(summary).to_json(), encoding="utf-8")
    print(f"\n=== Resume simulation (news-gated) : {summary['nb_open']} ouvertes, {summary['nb_closed']} cloturees ===")
    if summary["win_rate_closed"] is not None:
        print(f"Taux de reussite (cloturees) : {summary['win_rate_closed']:.0%} | "
              f"Retour moyen (cloturees) : {summary['avg_return_closed']:+.1%}")
    if summary["avg_unrealized_open"] is not None:
        print(f"Retour latent moyen (ouvertes) : {summary['avg_unrealized_open']:+.1%}")


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
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    candidates = pd.read_csv(CANDIDATES_PATH)
    valuation = pd.read_csv(VALUATION_PATH)

    ledger = load_ledger()
    ledger = open_new_positions(ledger, candidates, today)
    ledger = recheck_open_positions(ledger, valuation, today)

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    write_summary(ledger)
    append_equity_curve_point(ledger)


if __name__ == "__main__":
    main()
