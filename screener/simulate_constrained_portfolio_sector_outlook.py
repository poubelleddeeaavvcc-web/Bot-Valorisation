"""Bot#11: same capital-constrained paper-trading mechanics as simulate_constrained_portfolio.py
(Bot#2 -- capital, slots, sector cap, NA cap, FX and fractional-share handling all identical),
same ranking (select_top_picks.composite_score, unchanged), except a candidate whose sector is
currently rated "sous_pression" in data/universe/sector_outlook.csv -- Ollama's daily synthesis
of the user's own Gmail newsletters, see screener/newsletter_digest.py -- is excluded from the
pool entirely. "neutre"/"florissant"/no signal at all (missing file) don't exclude anything, same
fail-open posture as every other best-effort join in this repo.

Isolates a single variable against Bot#2, same controlled-comparison principle as Bot#8 (notes
ranking) vs Bot#2: everything except the extra sector exclusion in fill_slots() is reused
directly from simulate_constrained_portfolio.py.
"""
import json
import math
import pathlib
import sys
import time

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.select_top_picks import (  # noqa: E402
    composite_score, ticker_region, is_state_linked, NORTH_AMERICA_MAX_SHARE, STATE_LINKED_MAX_SHARE,
)
from screener.simulate_constrained_portfolio import (  # noqa: E402
    LEDGER_COLUMNS, MAX_PER_SECTOR, MAX_WHOLE_SHARE_OVERSHOOT, STARTING_CAPITAL, STARTING_SLOTS,
    TARGET_POSITION_SIZE, FX_PAIR, fetch_fx_rates, fractional_eligible, recheck_and_exit, to_eur,
)
from screener.simulate_portfolio import fails_fresh_check  # noqa: E402
from screener.fetch_cache import fetch_one as fetch_cache_one  # noqa: E402
from screener.newsletter_digest import load_pressured_sectors  # noqa: E402

LEDGER_PATH = HERE / "results/simulation/constrained_portfolio_ledger_sector_outlook.csv"
STATE_PATH = HERE / "results/simulation/constrained_state_sector_outlook.json"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
SUMMARY_PATH = HERE / "results/simulation/constrained_summary_sector_outlook.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/constrained_equity_curve_sector_outlook.csv"


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
    pressured = load_pressured_sectors()

    pool = candidates[~candidates["ticker"].isin(held_tickers)].copy()
    pool = pool[~pool["sector"].isin(pressured)]
    if not len(pool):
        return ledger, cash
    pool["score"] = composite_score(pool)
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
              f"@ {fresh['price']:.2f} {fresh.get('currency') or '?'}, score {pick_row['score']:.2f}")

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
    print(f"\n=== Portefeuille contraint (veto sectoriel) : {summary['nb_open']} positions, "
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
