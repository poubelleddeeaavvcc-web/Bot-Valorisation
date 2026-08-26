"""Bot#3 ("large"): same capital-constrained paper-trading mechanics as
simulate_constrained_portfolio.py ("restreint", Bot#2) -- same entry/exit rules, same FX and
fractional-share handling, same "no hard position-count ceiling, growth opens new slots"
compounding -- but a different capital-allocation philosophy, per the user's instruction
(2026-08-26): no max-per-sector cap. Instead, every new buy goes to whichever sector
currently holds the FEWEST open positions (ties broken by best available score), so the
sector mix stays as even as possible over time instead of being capped at a fixed count.

When the least-held sector has no fresh candidate available (exhausted for now):
  - if there's already an open position in that sector, reinforce it (buy more of the same
    ticker, averaging the entry price) instead of leaving that slot's cash idle -- "sauf s'il
    n'y a plus rien, alors il faut renforcer les positions actuelles."
  - if there's no open position in that sector either (never got a foothold), that sector is
    marked exhausted for this run and skipped in favor of the next least-held one.
Reinforcement can make one legacy position disproportionately large if its sector stays
starved of fresh candidates run after run -- an accepted consequence of "don't leave cash
idle," not a bug; keep an eye on entry_value_eur outliers in the ledger if that happens.

round_counts (separate from the real sector_counts used to seed the next run) is bumped on
every buy AND every reinforcement so that, within a single run, one chronically-starved
sector can't monopolize all of that run's cash at the expense of sectors that still have
fresh candidates waiting their turn.

Starts at STARTING_SLOTS=30 (~17 EUR/slot on the illustrative 500 EUR capital) vs Bot#2's 15
-- "large" as in more, smaller positions.

See simulate_constrained_portfolio.py for the FX conversion, fractional-share eligibility,
and exit-rule reasoning -- reused directly here (recheck_and_exit, to_eur, fetch_fx_rates,
fractional_eligible all operate on the same ledger schema and aren't bot#3-specific, so
they're imported rather than re-duplicated a third time).
"""
import json
import math
import pathlib
import sys
import time

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.select_top_picks import composite_score, ticker_region, NORTH_AMERICA_MAX_SHARE  # noqa: E402
from screener.fetch_cache import fetch_one as fetch_cache_one  # noqa: E402
from screener.simulate_constrained_portfolio import (  # noqa: E402
    LEDGER_COLUMNS, recheck_and_exit, to_eur, fetch_fx_rates, fractional_eligible,
)

LEDGER_PATH = HERE / "results/simulation/large_portfolio_ledger.csv"
STATE_PATH = HERE / "results/simulation/large_state.json"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
SUMMARY_PATH = HERE / "results/simulation/large_summary.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/large_equity_curve.csv"

STARTING_CAPITAL = 500.0       # illustrative, same as Bot#2 -- change freely
STARTING_SLOTS = 30            # "large" vs Bot#2's 15 -- more, smaller positions
TARGET_POSITION_SIZE = STARTING_CAPITAL / STARTING_SLOTS
# a whole-share buy (see fractional_eligible() in simulate_constrained_portfolio.py) costing
# more than this multiple of the slot target is rejected rather than forced -- at a ~17 EUR
# slot even more than Bot#2's, a single non-fractional name could otherwise eat several
# slots' worth of budget in one line.
MAX_WHOLE_SHARE_OVERSHOOT = 2.5


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


def fill_slots(ledger: pd.DataFrame, candidates: pd.DataFrame, cash: float, today: str,
               fx_rates: dict) -> tuple:
    held_tickers = set(ledger.loc[ledger["status"] == "open", "ticker"])
    sector_counts = ledger.loc[ledger["status"] == "open", "sector"].value_counts().to_dict()
    total_held = len(held_tickers)
    na_count = sum(1 for t in held_tickers if ticker_region(t) == "North America")

    pool = candidates[~candidates["ticker"].isin(held_tickers)].copy()
    pool["score"] = composite_score(pool)

    all_sectors = sorted(set(candidates["sector"].dropna().unique()) | set(sector_counts.keys()))
    round_counts = dict(sector_counts)
    rejected = set()
    exhausted = set()

    while len(exhausted) < len(all_sectors):
        remaining = [s for s in all_sectors if s not in exhausted]
        target_sector = min(remaining, key=lambda s: round_counts.get(s, 0))

        sector_pool = pool[(pool["sector"] == target_sector) &
                            (~pool["ticker"].isin(held_tickers)) &
                            (~pool["ticker"].isin(rejected))]
        # NB: sector_pool/eligible are only ever narrowed via boolean masks on a non-empty
        # frame below -- filtering an already-0-row frame with a map()-derived mask drops
        # its columns entirely in pandas (empty object-dtype mask on an empty frame), which
        # then blows up the sort_values("score") call. Guard on len(sector_pool) instead.
        if len(sector_pool):
            max_na = math.floor((total_held + 1) * NORTH_AMERICA_MAX_SHARE)
            na_ok = sector_pool["ticker"].map(
                lambda t: ticker_region(t) != "North America" or na_count < max_na)
            eligible = sector_pool[na_ok]
            if not len(eligible):
                eligible = sector_pool  # NA cap starving this sector -- relax it as a last resort
            eligible = eligible.sort_values("score", ascending=False)
        else:
            eligible = sector_pool  # already empty -- no fresh candidate left in this sector

        is_new = len(eligible) > 0
        if is_new:
            pick_row = eligible.iloc[0]
            ticker = pick_row["ticker"]
        else:
            # nothing fresh left for this sector -- reinforce the most-attractive open
            # position there instead of leaving this slot's cash idle
            open_in_sector = ledger[(ledger["status"] == "open") &
                                     (ledger["sector"] == target_sector) &
                                     (~ledger["ticker"].isin(rejected))]
            if not len(open_in_sector):
                exhausted.add(target_sector)
                continue
            ticker = open_in_sector.sort_values("last_valuation_gap", ascending=False)["ticker"].iloc[0]

        fresh = fetch_cache_one(ticker)
        time.sleep(0.4)  # same pacing as fetch_cache.py -- this can run right after it in CI
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

        if fractional:
            cost = min(TARGET_POSITION_SIZE, cash)
            add_shares = cost / price_eur
        else:
            target_shares = max(1, int(TARGET_POSITION_SIZE // price_eur))
            max_affordable = int(cash // price_eur)
            add_shares = min(target_shares, max_affordable)
            cost = add_shares * price_eur

        if is_new:
            new_row = {
                "ticker": ticker, "name": pick_row["name"], "sector": pick_row["sector"], "status": "open",
                "currency": fresh.get("currency"), "fractional": bool(fractional),
                "entry_date": today, "entry_price": fresh["price"], "shares": add_shares,
                "entry_value_eur": cost,
                "entry_valuation_gap": pick_row["valuation_gap"], "entry_quality_multiplier": pick_row["quality_multiplier"],
                "entry_mom_12_2": pick_row["mom_12_2"], "entry_sector_momentum": pick_row["sector_momentum"],
                "last_check_date": today, "last_price": fresh["price"], "last_valuation_gap": pick_row["valuation_gap"],
                "last_mom_12_2": pick_row["mom_12_2"], "current_value_eur": cost,
                "unrealized_return_pct": 0.0,
                "exit_date": None, "exit_price": None, "exit_reason": None,
                "exit_value_eur": None, "return_pct": None, "holding_days": None,
            }
            # appended straight into ledger (not batched) so a position bought earlier in
            # this same run is immediately visible to the reinforcement fallback below, if
            # its sector runs dry of fresh candidates later in this same run
            ledger = pd.concat([ledger, pd.DataFrame([new_row])], ignore_index=True)
            held_tickers.add(ticker)
            total_held += 1
            if ticker_region(ticker) == "North America":
                na_count += 1
            sector_counts[target_sector] = sector_counts.get(target_sector, 0) + 1
            kind = "fractionne" if fractional else "entier"
            print(f"  ACHAT {ticker} ({target_sector}) : {cost:.2f} EUR ({add_shares:.4f} actions, {kind}) "
                  f"@ {fresh['price']:.2f} {fresh.get('currency') or '?'}, score {pick_row['score']:.2f}")
        else:
            idx = ledger.index[(ledger["status"] == "open") & (ledger["ticker"] == ticker)][0]
            old_shares = ledger.at[idx, "shares"]
            new_shares = old_shares + add_shares
            new_entry_price = (ledger.at[idx, "entry_price"] * old_shares +
                                fresh["price"] * add_shares) / new_shares
            ledger.at[idx, "entry_price"] = new_entry_price
            ledger.at[idx, "shares"] = new_shares
            ledger.at[idx, "entry_value_eur"] = ledger.at[idx, "entry_value_eur"] + cost
            # current_value_eur/unrealized_return_pct must move with the added shares too --
            # otherwise a reinforced position understates the portfolio's true equity until
            # the *next* run's recheck_and_exit happens to refresh it (caught 2026-08-26: a
            # bot that had just spent its cash showed a bogus -27% "return" on its very first
            # run because these two were left stale after RENFORCE).
            ledger.at[idx, "last_check_date"] = today
            ledger.at[idx, "last_price"] = fresh["price"]
            ledger.at[idx, "current_value_eur"] = ledger.at[idx, "current_value_eur"] + cost
            ledger.at[idx, "unrealized_return_pct"] = fresh["price"] / new_entry_price - 1
            print(f"  RENFORCE {ticker} ({target_sector}) : +{cost:.2f} EUR ({add_shares:.4f} actions) "
                  f"@ {fresh['price']:.2f} {fresh.get('currency') or '?'} -- plus de candidat frais dans ce secteur")

        cash -= cost
        round_counts[target_sector] = round_counts.get(target_sector, 0) + 1

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
    print(f"\n=== Portefeuille large : {summary['nb_open']} positions, "
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

    fx_rates = fetch_fx_rates({
        "USD", "GBP", "GBp", "JPY", "CHF", "SEK", "CAD", "HKD", "AUD", "KRW", "TWD", "INR",
        "BRL", "MXN",
    })

    ledger, cash = recheck_and_exit(ledger, valuation, today, cash, fx_rates)
    ledger, cash = fill_slots(ledger, candidates, cash, today, fx_rates)

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    save_cash(cash)
    write_summary(ledger, cash)

    open_pos = ledger[ledger["status"] == "open"]
    total_equity = cash + open_pos["current_value_eur"].sum()
    append_equity_curve_point(cash, total_equity, len(open_pos), len(ledger[ledger["status"] == "closed"]))


if __name__ == "__main__":
    main()
