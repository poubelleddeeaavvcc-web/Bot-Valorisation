"""Bot#7: same capital-constrained paper-trading mechanics as
simulate_constrained_portfolio.py (Bot#2 -- capital, slots, sector cap, NA cap, FX and
fractional-share handling all identical), except candidates are ranked by the 2-note
quality/perspective system (screener/quality_perspective_notes.py) instead of
select_top_picks.composite_score.

Why this bot exists (2026-09-03, per the user): the notes were originally built as a
read-only annotation, deliberately NOT feeding into any bot's selection (see
quality_perspective_notes.py's module docstring). The user then asked for a way to
actually compare the two ranking methods head-to-head on performance and risk -- which
needs its own live paper-trading track record, not a retroactive backtest (Cash/Discipline
history didn't exist before 2026-09-03, so there's no historical note data to replay).

Deliberately draws from the SAME candidate pool as every other bot
(long_candidates_latest.csv, i.e. already passed value_momentum_quality_screener_v2.py's
valuation_gap/debt/momentum filter) rather than defining its own universe from the notes
alone -- an independent universe would confound "does note-based ranking pick better
winners" with "does note-based filtering select a different, unrelated set of candidates
to begin with", making a performance gap uninterpretable. Everything except fill_slots()
(which decides what to buy, and here also what "best" means) is reused directly from
simulate_constrained_portfolio.py, so any difference between Bot#2's and Bot#7's ledgers
isolates the effect of ranking-by-notes vs ranking-by-composite_score on an otherwise
identical entry/exit/diversification mechanism -- exactly the controlled comparison asked
for. Feeds into the same portfolio/bot_checkin.py report (win rate, avg return/win/loss,
30-clean-trade threshold) as every other bot once added to its BOTS list.

Ranking: notes_score = mean(note_qualite_20, note_perspective_20) -- both already
percentile-based /20 scores (see quality_perspective_notes.py), so a plain average treats
"how good is this business" and "is this a good moment to buy it" as equally important,
mirroring composite_score's equal-weighting philosophy without re-deriving it. Candidates
with no computable note yet (still NaN pre-backfill, or permanently N/A for structural
reasons -- see SECTOR_NA in quality_perspective_notes.py) are dropped from the pool
entirely rather than ranked last: buying a candidate this bot couldn't actually score would
not test the notes' predictive value, it would just be noise in the comparison.
"""
import json
import math
import pathlib
import sys
import time

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.select_top_picks import ticker_region, is_state_linked, NORTH_AMERICA_MAX_SHARE, STATE_LINKED_MAX_SHARE  # noqa: E402
from screener.simulate_constrained_portfolio import (  # noqa: E402
    LEDGER_COLUMNS, MAX_PER_SECTOR, MAX_WHOLE_SHARE_OVERSHOOT, STARTING_CAPITAL, STARTING_SLOTS,
    TARGET_POSITION_SIZE, FX_PAIR, fetch_fx_rates, fractional_eligible, recheck_and_exit, to_eur,
)
from screener.simulate_portfolio import fails_fresh_check  # noqa: E402
from screener.fetch_cache import fetch_one as fetch_cache_one  # noqa: E402

LEDGER_PATH = HERE / "results/simulation/constrained_portfolio_ledger_notes.csv"
STATE_PATH = HERE / "results/simulation/constrained_state_notes.json"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
NOTES_PATH = HERE / "results/screener/quality_perspective_notes.csv"
SUMMARY_PATH = HERE / "results/simulation/constrained_summary_notes.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/constrained_equity_curve_notes.csv"

NOTES_LEDGER_COLUMNS = LEDGER_COLUMNS + [
    "entry_note_qualite", "entry_note_qualite_low_confidence",
    "entry_note_perspective", "entry_note_perspective_low_confidence",
]


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH)
        for c in NOTES_LEDGER_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[NOTES_LEDGER_COLUMNS]
    return pd.DataFrame(columns=NOTES_LEDGER_COLUMNS)


def load_cash() -> float:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))["cash_eur"]
    return STARTING_CAPITAL


def save_cash(cash: float):
    STATE_PATH.write_text(json.dumps({"cash_eur": cash}), encoding="utf-8")


def _load_notes_score(candidates: pd.DataFrame) -> pd.DataFrame:
    """Left-joins quality_perspective_notes.csv onto candidates and computes notes_score.
    Missing file (module never run yet) or missing ticker both degrade to NaN score, same
    end result (dropped from the pool below) -- never a hard error, this bot just can't
    trade anything until notes exist, same posture as every other best-effort join in this
    codebase."""
    candidates = candidates.copy()
    if not NOTES_PATH.exists():
        candidates["notes_score"] = float("nan")
        candidates["note_qualite_20"] = float("nan")
        candidates["note_qualite_low_confidence"] = None
        candidates["note_perspective_20"] = float("nan")
        candidates["note_perspective_low_confidence"] = None
        return candidates
    notes = pd.read_csv(NOTES_PATH)[["ticker", "note_qualite_20", "note_qualite_low_confidence",
                                      "note_perspective_20", "note_perspective_low_confidence"]]
    candidates = candidates.merge(notes, on="ticker", how="left")
    candidates["notes_score"] = candidates[["note_qualite_20", "note_perspective_20"]].mean(axis=1)
    return candidates


def fill_slots(ledger: pd.DataFrame, candidates: pd.DataFrame, valuation: pd.DataFrame, cash: float, today: str,
               fx_rates: dict) -> tuple:
    held_tickers = set(ledger.loc[ledger["status"] == "open", "ticker"])
    sector_counts = ledger.loc[ledger["status"] == "open", "sector"].value_counts().to_dict()
    total_held = len(held_tickers)
    sector_pe = valuation.groupby("sector")["sector_median_pe"].first()
    sector_mom = valuation.groupby("sector")["sector_momentum"].first()
    industry_pe = valuation.groupby("industry")["industry_median_pe"].first()
    industry_count = valuation.groupby("industry")["industry_count"].first()
    na_count = sum(1 for t in held_tickers if ticker_region(t) == "North America")
    state_count = int(ledger.loc[ledger["status"] == "open", "country"].map(is_state_linked).sum())

    pool = candidates[~candidates["ticker"].isin(held_tickers)].copy()
    pool = _load_notes_score(pool)
    pool = pool.dropna(subset=["notes_score"])  # see module docstring: unscored candidates
    # are excluded from this bot's universe entirely, not ranked last
    if not len(pool):
        return ledger, cash
    pool["score"] = pool["notes_score"]
    pool = pool.sort_values("score", ascending=False)
    rejected = set()

    new_rows = []
    while True:
        pick_row = None
        for enforce_geo_caps in (True, False):
            for cap in range(MAX_PER_SECTOR, 10):
                eligible = pool[(~pool["ticker"].isin(held_tickers)) & (~pool["ticker"].isin(rejected))]
                eligible = eligible[eligible["sector"].map(lambda s: sector_counts.get(s, 0)) < cap]
                if enforce_geo_caps:
                    max_na = math.floor((total_held + 1) * NORTH_AMERICA_MAX_SHARE)
                    eligible = eligible[eligible["ticker"].map(
                        lambda t: ticker_region(t) != "North America" or na_count < max_na)]
                    max_state = math.floor((total_held + 1) * STATE_LINKED_MAX_SHARE)
                    eligible = eligible[eligible.get("country", pd.Series(index=eligible.index, dtype=object)).map(
                        lambda c: not is_state_linked(c) or state_count < max_state)]
                if len(eligible):
                    pick_row = eligible.iloc[0]
                    break
            if pick_row is not None:
                break
        if pick_row is None:
            break

        ticker = pick_row["ticker"]
        fresh = fetch_cache_one(ticker)
        time.sleep(0.4)
        if fresh.get("price") is None or fresh.get("error"):
            rejected.add(ticker)
            continue

        price_eur = to_eur(fresh["price"], fresh.get("currency"), fx_rates)
        if price_eur is None or price_eur <= 0 or price_eur > cash:
            rejected.add(ticker)
            continue

        avg_vol = fresh.get("avg_volume")
        market_cap_eur = to_eur(fresh.get("market_cap"), fresh.get("currency"), fx_rates)
        adv_eur = (to_eur(avg_vol * fresh["price"], fresh.get("currency"), fx_rates)
                   if avg_vol is not None and pd.notna(avg_vol) else None)
        fractional = fractional_eligible(ticker, market_cap_eur, adv_eur)

        if not fractional and price_eur > MAX_WHOLE_SHARE_OVERSHOOT * TARGET_POSITION_SIZE:
            rejected.add(ticker)
            continue

        # fresh momentum/valuation gate, same entry bar as every other bot: this ranks by
        # notes, but the candidate still has to clear the base screener's own filter on
        # today's data before being bought, not just at whatever moment it entered
        # long_candidates_latest.csv (2026-09-02 same-day round-trip fix, see
        # simulate_portfolio.py's module docstring).
        fails, state = fails_fresh_check(fresh, pick_row["quality_multiplier"], sector_pe, sector_mom,
                                          industry_pe, industry_count, fallback_valuation_gap=pick_row["valuation_gap"])
        if fails:
            rejected.add(ticker)
            continue

        if fractional:
            cost = min(TARGET_POSITION_SIZE, cash)
            shares = cost / price_eur
        else:
            target_shares = max(1, int(TARGET_POSITION_SIZE // price_eur))
            max_affordable = int(cash // price_eur)
            shares = min(target_shares, max_affordable)
            cost = shares * price_eur

        new_rows.append({
            "ticker": ticker, "name": pick_row["name"], "sector": pick_row["sector"],
            "country": pick_row.get("country"), "status": "open",
            "currency": fresh.get("currency"), "fractional": bool(fractional),
            "entry_date": today, "entry_price": fresh["price"], "shares": shares,
            "entry_value_eur": cost,
            "entry_valuation_gap": state["valuation_gap"], "entry_quality_multiplier": pick_row["quality_multiplier"],
            "entry_mom_12_2": fresh["mom_12_2"], "entry_sector_momentum": state["sector_momentum"],
            "last_check_date": today, "last_price": fresh["price"], "last_valuation_gap": state["valuation_gap"],
            "last_mom_12_2": fresh["mom_12_2"], "current_value_eur": cost,
            "unrealized_return_pct": 0.0,
            "exit_date": None, "exit_price": None, "exit_reason": None,
            "exit_value_eur": None, "return_pct": None, "holding_days": None,
            "entry_note_qualite": pick_row.get("note_qualite_20"),
            "entry_note_qualite_low_confidence": pick_row.get("note_qualite_low_confidence"),
            "entry_note_perspective": pick_row.get("note_perspective_20"),
            "entry_note_perspective_low_confidence": pick_row.get("note_perspective_low_confidence"),
        })
        cash -= cost
        held_tickers.add(ticker)
        sector_counts[pick_row["sector"]] = sector_counts.get(pick_row["sector"], 0) + 1
        total_held += 1
        if ticker_region(ticker) == "North America":
            na_count += 1
        if is_state_linked(pick_row.get("country")):
            state_count += 1
        kind = "fractionne" if fractional else "entier"
        print(f"  ACHAT {ticker} ({pick_row['sector']}) : {cost:.2f} EUR ({shares:.4f} actions, {kind}) "
              f"@ {fresh['price']:.2f} {fresh.get('currency') or '?'}, notes_score {pick_row['score']:.1f}")

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
    print(f"\n=== Portefeuille contraint (selection par notes) : {summary['nb_open']} positions, "
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
    if not NOTES_PATH.exists():
        print("quality_perspective_notes.csv n'existe pas encore -- lancer "
              "screener/quality_perspective_notes.py d'abord. Rien a simuler ce run.")
        return
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    candidates = pd.read_csv(CANDIDATES_PATH)
    valuation = pd.read_csv(VALUATION_PATH)

    ledger = load_ledger()
    cash = load_cash()

    fx_rates = fetch_fx_rates(set(FX_PAIR.keys()))

    ledger, cash = recheck_and_exit(ledger, valuation, today, cash, fx_rates)
    ledger, cash = fill_slots(ledger, candidates, valuation, cash, today, fx_rates)

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    save_cash(cash)
    write_summary(ledger, cash)

    open_pos = ledger[ledger["status"] == "open"]
    total_equity = cash + open_pos["current_value_eur"].sum()
    append_equity_curve_point(cash, total_equity, len(open_pos), len(ledger[ledger["status"] == "closed"]))


if __name__ == "__main__":
    main()
