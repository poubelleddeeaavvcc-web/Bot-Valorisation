"""Mise a l'echelle unique de l'historique des 30 bots a capital (2026-10-19).

Le capital simule passe de 500 a 2000 EUR (famille contrainte), de 500 a 4000 EUR (large) et de
300 a 1200 EUR (delta), pour que chaque ligne fasse ~133 EUR au lieu de ~17-33 EUR (plancher
MIN_POSITION_EUR = 100 EUR, frais de 1 EUR a l'achat ET a la vente -- voir
simulate_constrained_portfolio.py). Plutot que de repartir de zero, l'historique est multiplie
par le meme facteur : nombre d'actions, montants EUR du journal, cash, courbe de valeur et
resume. Les rendements en % (lignes et portefeuille) sont inchanges ; les frais passes restent
ceux payes sur les petites lignes (1 EUR a la vente seulement), ce qui est l'historique reel.

Idempotent : le premier point de la courbe de valeur EST le capital de depart. S'il vaut deja
le nouveau capital, le bot est saute ; s'il vaut l'ancien, il est mis a l'echelle ; sinon le
script s'arrete sans rien ecrire pour ce bot (etat inattendu, a regarder a la main).

    python screener/rescale_capital_bots.py            # applique
    python screener/rescale_capital_bots.py --dry-run  # affiche seulement
"""
import argparse
import json
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
SIM = HERE / "results/simulation"

FAMILIES = {"constrained": (500.0, 2000.0), "large": (500.0, 4000.0), "delta": (300.0, 1200.0)}
SUFFIXES = ["", "_notes", "_sector_outlook", "_notes_sector_outlook", "_newsgated", "_newsgated_notes",
            "_newsgated_sector_outlook", "_newsgated_notes_sector_outlook", "_marge", "_concentre_marge"]
LEDGER_EUR_COLUMNS = ["shares", "entry_value_eur", "current_value_eur", "exit_value_eur"]
CURVE_EUR_COLUMNS = ["cash_eur", "total_equity_eur"]
SUMMARY_EUR_KEYS = ["cash_eur", "total_equity_eur"]


def paths(family: str, suffix: str) -> dict:
    return {"ledger": SIM / f"{family}_portfolio_ledger{suffix}.csv",
            "state": SIM / f"{family}_state{suffix}.json",
            "curve": SIM / f"{family}_equity_curve{suffix}.csv",
            "summary": SIM / f"{family}_summary{suffix}.json"}


def rescale_bot(family: str, suffix: str, old: float, new: float, dry_run: bool) -> str:
    p = paths(family, suffix)
    if not p["curve"].exists():
        return "pas encore lance"
    curve = pd.read_csv(p["curve"])
    start = float(curve["total_equity_eur"].iloc[0])
    if abs(start - new) < 0.01 * new:
        return f"deja a {new:.0f} EUR"
    if abs(start - old) > 0.01 * old:
        raise SystemExit(f"{family}{suffix} : premier point {start:.2f} EUR, ni {old:.0f} ni {new:.0f} -- rien ecrit")
    k = new / old
    ledger = pd.read_csv(p["ledger"])
    for c in LEDGER_EUR_COLUMNS:
        ledger[c] = pd.to_numeric(ledger[c], errors="coerce") * k
    for c in CURVE_EUR_COLUMNS:
        curve[c] = curve[c] * k
    state = json.loads(p["state"].read_text(encoding="utf-8"))
    state["cash_eur"] = state["cash_eur"] * k
    summary = json.loads(p["summary"].read_text(encoding="utf-8")) if p["summary"].exists() else None
    if summary is not None:
        for key in SUMMARY_EUR_KEYS:
            if summary.get(key) is not None:
                summary[key] = summary[key] * k
    if not dry_run:
        ledger.to_csv(p["ledger"], index=False)
        curve.to_csv(p["curve"], index=False)
        p["state"].write_text(json.dumps(state), encoding="utf-8")
        if summary is not None:
            p["summary"].write_text(json.dumps(summary), encoding="utf-8")
    return f"x{k:g} : {start:.0f} -> {new:.0f} EUR, cash {state['cash_eur']:.2f} EUR, valeur {curve['total_equity_eur'].iloc[-1]:.2f} EUR"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    bots = [(f, s, old, new) for f, (old, new) in FAMILIES.items() for s in SUFFIXES]
    # verification complete avant toute ecriture : un etat inattendu sur un bot ne doit pas
    # laisser les autres a moitie convertis
    for family, suffix, old, new in bots:
        rescale_bot(family, suffix, old, new, dry_run=True)
    for family, suffix, old, new in bots:
        print(f"{family}{suffix or '_base'} : {rescale_bot(family, suffix, old, new, args.dry_run)}")


if __name__ == "__main__":
    main()
