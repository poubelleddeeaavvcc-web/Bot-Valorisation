"""A second, parallel paper-trading simulation: instead of blindly buying every LONG
candidate (simulate_portfolio.py, kept deliberately unfiltered as the honest baseline
track record), this one starts from a fixed amount of capital and behaves like an
investor who can only afford a handful of positions at once.

Mechanics:
  - Starts at STARTING_CAPITAL, all in cash.
  - Every run, existing open positions get a fresh price/fundamentals check and exit on
    the exact same rules as the blind simulation (momentum lost / valuation reached /
    stop loss -- see simulate_portfolio.py for the reasoning behind each; duplicated here
    rather than imported because this ledger tracks euro amounts, not just percentages).
  - Every sale's proceeds return to cash and get reinvested: whenever cash >=
    TARGET_POSITION_SIZE, the next slot is filled with the best available LONG candidate
    NOT already held, ranked by select_top_picks.composite_score and capped at
    MAX_PER_SECTOR per sector (relaxed one step at a time only if too few sectors are
    available to fill every slot otherwise) -- same diversification logic used for the
    "top picks today" report.
  - No hard position-count ceiling: as gains compound, cash naturally clears the
    TARGET_POSITION_SIZE bar more than once per run, so a big win can open 2+ new slots
    in the same cycle. Growth scales the *number* of positions, not just their size --
    per the user's instruction, a windfall should diversify into a 5th/6th/7th name
    rather than doubling down on fewer, bigger bets.
"""
import json
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.select_top_picks import composite_score  # noqa: E402
from screener.simulate_portfolio import fetch_fresh_single, STOP_LOSS_PCT  # noqa: E402

LEDGER_PATH = HERE / "results/simulation/constrained_portfolio_ledger.csv"
STATE_PATH = HERE / "results/simulation/constrained_state.json"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
SUMMARY_PATH = HERE / "results/simulation/constrained_summary.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/constrained_equity_curve.csv"

STARTING_CAPITAL = 500.0      # illustrative -- the user's own example; change freely
STARTING_SLOTS = 4            # -> TARGET_POSITION_SIZE = 500/4 = 125 EUR per slot
TARGET_POSITION_SIZE = STARTING_CAPITAL / STARTING_SLOTS
MAX_PER_SECTOR = 1

LEDGER_COLUMNS = [
    "ticker", "name", "sector", "status",
    "entry_date", "entry_price", "shares", "entry_value_eur",
    "entry_valuation_gap", "entry_quality_multiplier", "entry_mom_12_2", "entry_sector_momentum",
    "last_check_date", "last_price", "last_valuation_gap", "last_mom_12_2",
    "current_value_eur", "unrealized_return_pct",
    "exit_date", "exit_price", "exit_reason", "exit_value_eur", "return_pct", "holding_days",
]


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH)
        for c in LEDGER_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[LEDGER_COLUMNS]
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def load_cash() -> float:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))["cash_eur"]
    return STARTING_CAPITAL


def save_cash(cash: float):
    STATE_PATH.write_text(json.dumps({"cash_eur": cash}), encoding="utf-8")


def recheck_and_exit(ledger: pd.DataFrame, valuation: pd.DataFrame, today: str, cash: float) -> tuple:
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
        shares = ledger.at[idx, "shares"]
        unrealized = fresh["price"] / entry_price - 1
        current_value = shares * fresh["price"]

        ledger.at[idx, "last_check_date"] = today
        ledger.at[idx, "last_price"] = fresh["price"]
        ledger.at[idx, "last_valuation_gap"] = valuation_gap_now
        ledger.at[idx, "last_mom_12_2"] = fresh["mom_12_2"]
        ledger.at[idx, "current_value_eur"] = current_value
        ledger.at[idx, "unrealized_return_pct"] = unrealized

        momentum_lost = fresh["mom_12_2"] <= 0 or fresh["mom_12_2"] <= today_sector_mom
        valuation_reached = pd.notna(valuation_gap_now) and valuation_gap_now <= 0
        stop_loss_hit = unrealized <= STOP_LOSS_PCT  # see simulate_portfolio.py

        if momentum_lost or valuation_reached or stop_loss_hit:
            reason = ("stop_loss" if stop_loss_hit else
                      "valorisation_atteinte" if valuation_reached else "momentum_perdu")
            entry_date = pd.Timestamp(ledger.at[idx, "entry_date"])
            ledger.at[idx, "status"] = "closed"
            ledger.at[idx, "exit_date"] = today
            ledger.at[idx, "exit_price"] = fresh["price"]
            ledger.at[idx, "exit_reason"] = reason
            ledger.at[idx, "exit_value_eur"] = current_value
            ledger.at[idx, "return_pct"] = unrealized
            ledger.at[idx, "holding_days"] = (pd.Timestamp(today) - entry_date).days
            cash += current_value
            print(f"  VENTE {ticker} : {reason}, retour {unrealized:+.1%}, "
                  f"{current_value:.2f} EUR reinjectes en cash")

    return ledger, cash


def fill_slots(ledger: pd.DataFrame, candidates: pd.DataFrame, cash: float, today: str) -> tuple:
    held_tickers = set(ledger.loc[ledger["status"] == "open", "ticker"])
    sector_counts = ledger.loc[ledger["status"] == "open", "sector"].value_counts().to_dict()

    pool = candidates[~candidates["ticker"].isin(held_tickers)].copy()
    if not len(pool):
        return ledger, cash
    pool["score"] = composite_score(pool)
    pool = pool.sort_values("score", ascending=False)

    new_rows = []
    while cash >= TARGET_POSITION_SIZE:
        pick = None
        for cap in range(MAX_PER_SECTOR, 10):  # relax the sector cap only if truly starved of options
            eligible = pool[~pool["ticker"].isin(held_tickers)]
            eligible = eligible[eligible["sector"].map(lambda s: sector_counts.get(s, 0)) < cap]
            if len(eligible):
                pick = eligible.iloc[0]
                break
        if pick is None:
            break  # no eligible candidate at all, however relaxed -- stop, keep the cash

        shares = TARGET_POSITION_SIZE / pick["price"]
        new_rows.append({
            "ticker": pick["ticker"], "name": pick["name"], "sector": pick["sector"], "status": "open",
            "entry_date": today, "entry_price": pick["price"], "shares": shares,
            "entry_value_eur": TARGET_POSITION_SIZE,
            "entry_valuation_gap": pick["valuation_gap"], "entry_quality_multiplier": pick["quality_multiplier"],
            "entry_mom_12_2": pick["mom_12_2"], "entry_sector_momentum": pick["sector_momentum"],
            "last_check_date": today, "last_price": pick["price"], "last_valuation_gap": pick["valuation_gap"],
            "last_mom_12_2": pick["mom_12_2"], "current_value_eur": TARGET_POSITION_SIZE,
            "unrealized_return_pct": 0.0,
            "exit_date": None, "exit_price": None, "exit_reason": None,
            "exit_value_eur": None, "return_pct": None, "holding_days": None,
        })
        cash -= TARGET_POSITION_SIZE
        held_tickers.add(pick["ticker"])
        sector_counts[pick["sector"]] = sector_counts.get(pick["sector"], 0) + 1
        print(f"  ACHAT {pick['ticker']} ({pick['sector']}) : {TARGET_POSITION_SIZE:.2f} EUR "
              f"@ {pick['price']:.2f}, score {pick['score']:.2f}")

    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
    return ledger, cash


def write_summary(ledger: pd.DataFrame, cash: float):
    closed = ledger[ledger["status"] == "closed"]
    open_pos = ledger[ledger["status"] == "open"]
    total_equity = cash + open_pos["current_value_eur"].sum()
    summary = {
        "cash_eur": cash,
        "total_equity_eur": total_equity,
        "total_return_pct": total_equity / STARTING_CAPITAL - 1,
        "nb_open": len(open_pos),
        "nb_closed": len(closed),
        "win_rate_closed": float((closed["return_pct"] > 0).mean()) if len(closed) else None,
        "avg_return_closed": float(closed["return_pct"].mean()) if len(closed) else None,
    }
    SUMMARY_PATH.write_text(pd.Series(summary).to_json(), encoding="utf-8")
    print(f"\n=== Portefeuille contraint : {summary['nb_open']} positions, "
          f"{cash:.2f} EUR cash, valeur totale {total_equity:.2f} EUR "
          f"({summary['total_return_pct']:+.1%} depuis le depart) ===")


def append_equity_curve_point(cash: float, total_equity: float, nb_open: int, nb_closed: int):
    row = {"timestamp": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
           "cash_eur": cash, "total_equity_eur": total_equity,
           "n_open": nb_open, "n_closed": nb_closed}
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
    cash = load_cash()

    ledger, cash = recheck_and_exit(ledger, valuation, today, cash)
    ledger, cash = fill_slots(ledger, candidates, cash, today)

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    save_cash(cash)
    write_summary(ledger, cash)

    open_pos = ledger[ledger["status"] == "open"]
    total_equity = cash + open_pos["current_value_eur"].sum()
    append_equity_curve_point(cash, total_equity, len(open_pos), len(ledger[ledger["status"] == "closed"]))


if __name__ == "__main__":
    main()
