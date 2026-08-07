"""Step 5 of the pipeline: a blind, no-hindsight forward paper-trading simulation.

The idea: instead of trusting a backtest (which can always be curve-fit after the
fact), simulate buying EVERY LONG candidate the screener ever produces, with no human
stock-picking involved, and track what actually happens going forward in real time.
This is the honest test of whether the screener works -- accumulated evidence, not a
historical fit.

Mechanics:
  - Every run, any current LONG candidate not already an open position gets "bought"
    at today's price.
  - Every run, every OPEN position gets a FRESH single-ticker price/fundamentals check
    (independent of fetch_cache.py's 90-day staleness -- that cadence is fine for
    discovering new candidates but far too slow for tracking positions you're
    supposedly holding). This is cheap: a handful of tickers, not the whole universe.
  - Exit ("sell") when EITHER:
      - valuation_gap has closed to <=0 (the thesis played out: no longer undervalued)
      - momentum is lost: mom_12_2 <= 0, or mom_12_2 <= its sector's momentum (the
        same relative-strength bar required to enter in the first place)
  - No short-side simulation yet (long-only exit to cash) -- deliberately kept simple
    until the long side has a track record worth trusting.
"""
import pathlib
import sys

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

LEDGER_PATH = HERE / "results/simulation/portfolio_ledger.csv"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
SUMMARY_PATH = HERE / "results/simulation/summary.json"

LEDGER_COLUMNS = [
    "ticker", "name", "sector", "status",
    "entry_date", "entry_price", "entry_valuation_gap", "entry_quality_multiplier",
    "entry_mom_12_2", "entry_sector_momentum",
    "last_check_date", "last_price", "last_valuation_gap", "last_mom_12_2", "unrealized_return_pct",
    "exit_date", "exit_price", "exit_reason", "return_pct", "holding_days",
]


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH)
        for c in LEDGER_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[LEDGER_COLUMNS]
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def fetch_fresh_single(ticker: str) -> dict | None:
    """Independent of the big cache -- always current, cheap because it's one ticker."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        hist = t.history(period="14mo", interval="1mo", auto_adjust=True)["Close"].dropna()
        if len(hist) < 13:
            return None
        mom_12_2 = hist.iloc[-2] / hist.iloc[-13] - 1
        return {
            "price": hist.iloc[-1], "eps": info.get("trailingEps"),
            "sector": info.get("sector"), "mom_12_2": mom_12_2,
        }
    except Exception as e:
        print(f"  echec fetch frais pour {ticker}: {e}", file=sys.stderr)
        return None


def open_new_positions(ledger: pd.DataFrame, candidates: pd.DataFrame, today: str) -> pd.DataFrame:
    open_tickers = set(ledger.loc[ledger["status"] == "open", "ticker"])
    new_rows = []
    for _, c in candidates.iterrows():
        if c["ticker"] in open_tickers:
            continue
        new_rows.append({
            "ticker": c["ticker"], "name": c["name"], "sector": c["sector"], "status": "open",
            "entry_date": today, "entry_price": c["price"], "entry_valuation_gap": c["valuation_gap"],
            "entry_quality_multiplier": c["quality_multiplier"], "entry_mom_12_2": c["mom_12_2"],
            "entry_sector_momentum": c["sector_momentum"],
            "last_check_date": today, "last_price": c["price"], "last_valuation_gap": c["valuation_gap"],
            "last_mom_12_2": c["mom_12_2"], "unrealized_return_pct": 0.0,
            "exit_date": None, "exit_price": None, "exit_reason": None,
            "return_pct": None, "holding_days": None,
        })
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
        print(f"{len(new_rows)} nouvelle(s) position(s) ouverte(s) : {[r['ticker'] for r in new_rows]}")
    return ledger


def recheck_open_positions(ledger: pd.DataFrame, valuation: pd.DataFrame, today: str) -> pd.DataFrame:
    sector_pe = valuation.groupby("sector")["sector_median_pe"].first()
    sector_mom = valuation.groupby("sector")["sector_momentum"].first()

    for idx in ledger.index[ledger["status"] == "open"]:
        ticker = ledger.at[idx, "ticker"]
        fresh = fetch_fresh_single(ticker)
        if fresh is None or fresh["price"] is None or fresh["eps"] is None:
            continue

        sector = fresh["sector"] or ledger.at[idx, "sector"]
        today_sector_pe = sector_pe.get(sector)
        today_sector_mom = sector_mom.get(sector, 0.0)
        qmult = ledger.at[idx, "entry_quality_multiplier"]

        if today_sector_pe is not None and pd.notna(qmult):
            fair_value_now = fresh["eps"] * today_sector_pe * qmult
            valuation_gap_now = fair_value_now / fresh["price"] - 1
        else:
            valuation_gap_now = ledger.at[idx, "last_valuation_gap"]

        entry_price = ledger.at[idx, "entry_price"]
        unrealized = fresh["price"] / entry_price - 1

        ledger.at[idx, "last_check_date"] = today
        ledger.at[idx, "last_price"] = fresh["price"]
        ledger.at[idx, "last_valuation_gap"] = valuation_gap_now
        ledger.at[idx, "last_mom_12_2"] = fresh["mom_12_2"]
        ledger.at[idx, "unrealized_return_pct"] = unrealized

        momentum_lost = fresh["mom_12_2"] <= 0 or fresh["mom_12_2"] <= today_sector_mom
        valuation_reached = pd.notna(valuation_gap_now) and valuation_gap_now <= 0

        if momentum_lost or valuation_reached:
            reason = "valorisation_atteinte" if valuation_reached else "momentum_perdu"
            entry_date = pd.Timestamp(ledger.at[idx, "entry_date"])
            ledger.at[idx, "status"] = "closed"
            ledger.at[idx, "exit_date"] = today
            ledger.at[idx, "exit_price"] = fresh["price"]
            ledger.at[idx, "exit_reason"] = reason
            ledger.at[idx, "return_pct"] = unrealized
            ledger.at[idx, "holding_days"] = (pd.Timestamp(today) - entry_date).days
            print(f"  VENTE {ticker} : {reason}, retour {unrealized:+.1%}")

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
    print(f"\n=== Résumé simulation : {summary['nb_open']} ouvertes, {summary['nb_closed']} clôturées ===")
    if summary["win_rate_closed"] is not None:
        print(f"Taux de réussite (clôturées) : {summary['win_rate_closed']:.0%} | "
              f"Retour moyen (clôturées) : {summary['avg_return_closed']:+.1%}")
    if summary["avg_unrealized_open"] is not None:
        print(f"Retour latent moyen (ouvertes) : {summary['avg_unrealized_open']:+.1%}")


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


if __name__ == "__main__":
    main()
