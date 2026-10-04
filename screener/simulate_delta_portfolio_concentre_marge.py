"""Bot#40 (Delta, 6 positions + momentum margin): the momentum-margin bot of its
family (see the original docstring below) with fewer, larger positions -- STARTING_CAPITAL / 6
per slot instead of Bot#25's default -- so the flat 1 EUR sale fee weighs less per trade. Picked
from the 2026-10-04 point-in-time backtest (screener/backtest_pit_variants.json, fewer positions +
margin led in all 3 capital families -- on 1-5 closed trades each, an indication, not proof).

Bot#37 (Delta + momentum entry margin): same conviction-averaging mechanics as
simulate_delta_portfolio.py (Bot#25 -- 300 EUR, 9 slots, reinforce_convictions gate, exit rules),
except a brand-new candidate is only bought if its mom_12_2 beats its sector's momentum by at
least MIN_ENTRY_MOM_MARGIN (see simulate_portfolio.py), and a conviction reinforcement needs that
same margin too (instead of Delta's 0/+5 points) -- both are money going into a name, and a thin
margin is what preceded the momentum_perdu exits (2026-10-03, user's choice: "margin on every
buy"). Exit rules are untouched.

reinforce_convictions/fill_slots/recheck_and_exit are Bot#25's own, reused by import with the
margin passed in -- the margin on every buy is the only variable isolated against Bot#25.
"""
# Alias lisibilite (mapping perso) : D10 -- Delta (conviction), 6 positions + marge momentum ; source : D9 -- famille Delta (conviction), variante + marge momentum a l'entree et au renfort
import json
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.simulate_delta_portfolio import (  # noqa: E402
    LEDGER_COLUMNS, STARTING_CAPITAL, FX_PAIR, fetch_fx_rates, fill_slots, recheck_and_exit,
    reinforce_convictions,
)
from screener.simulate_portfolio import MIN_ENTRY_MOM_MARGIN  # noqa: E402

SLOTS = 6
TARGET_POSITION_SIZE = STARTING_CAPITAL / SLOTS
# pas de nouvelle position sous la moitie de la taille cible : le backtest qui a retenu ce bot n'en
# ouvre jamais sous la taille cible, et une petite ligne paie le meme 1 EUR de frais a la vente.
MIN_POSITION_EUR = TARGET_POSITION_SIZE / 2

LEDGER_PATH = HERE / "results/simulation/delta_portfolio_ledger_concentre_marge.csv"
STATE_PATH = HERE / "results/simulation/delta_state_concentre_marge.json"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
SUMMARY_PATH = HERE / "results/simulation/delta_summary_concentre_marge.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/delta_equity_curve_concentre_marge.csv"


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
        "nb_reinforced": int((open_pos["reinforcement_count"].fillna(0) > 0).sum()) if len(open_pos) else 0,
        "win_rate_closed": float((closed["return_pct"] > 0).mean()) if len(closed) else None,
        "avg_return_closed": float(closed["return_pct"].mean()) if len(closed) else None,
    }
    SUMMARY_PATH.write_text(pd.Series(summary).to_json(), encoding="utf-8")
    print(f"\n=== Portefeuille Delta (6 positions + marge momentum) : {summary['nb_open']} positions "
          f"({summary['nb_reinforced']} renforcees), {cash:.2f} EUR cash, valeur totale {total_equity:.2f} EUR "
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

    fx_rates = fetch_fx_rates(set(FX_PAIR.keys()))

    ledger, cash = recheck_and_exit(ledger, valuation, today, cash, fx_rates)
    ledger, cash = reinforce_convictions(ledger, valuation, today, cash, fx_rates,
                                         min_mom_margin=MIN_ENTRY_MOM_MARGIN, target_size=TARGET_POSITION_SIZE)
    ledger, cash = fill_slots(ledger, candidates, valuation, cash, today, fx_rates,
                              min_mom_margin=MIN_ENTRY_MOM_MARGIN, target_size=TARGET_POSITION_SIZE,
                              min_position_eur=MIN_POSITION_EUR)

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    save_cash(cash)
    write_summary(ledger, cash)

    open_pos = ledger[ledger["status"] == "open"]
    total_equity = cash + open_pos["current_value_eur"].sum()
    append_equity_curve_point(cash, total_equity, len(open_pos), len(ledger[ledger["status"] == "closed"]))


if __name__ == "__main__":
    main()
