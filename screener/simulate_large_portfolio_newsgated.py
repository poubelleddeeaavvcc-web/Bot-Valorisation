"""Bot#6: same capital-constrained mechanics as simulate_large_portfolio.py (Bot#3 --
no max-per-sector cap, buys go to whichever sector currently holds the fewest open
positions, reinforces an existing position when its sector has no fresh candidate left),
except each candidate is passed through news_filter.news_verdict() right before it would
actually be bought or reinforced. A candidate flagged by the local Ollama model as having
an obvious news red flag is treated like any other rejection (unaffordable, fetch failed,
...): skipped this run, free to come up again next run. Everything else -- exits, FX,
fractional-share eligibility, sector-balancing -- is reused directly from
simulate_large_portfolio.py / simulate_constrained_portfolio.py, so any difference between
the two bots' ledgers isolates the effect of the news gate.

Only fill_slots() (which decides what to buy/reinforce) and the path-coupled
load/save/summary functions are duplicated here with this bot's own files.
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
from screener.fetch_cache import fetch_one as fetch_cache_one  # noqa: E402
from screener.simulate_constrained_portfolio import (  # noqa: E402
    LEDGER_COLUMNS, FX_PAIR, recheck_and_exit, to_eur, fetch_fx_rates, fractional_eligible,
)
from screener.simulate_large_portfolio import MAX_WHOLE_SHARE_OVERSHOOT  # noqa: E402
from screener import news_filter  # noqa: E402

LEDGER_PATH = HERE / "results/simulation/large_portfolio_ledger_newsgated.csv"
STATE_PATH = HERE / "results/simulation/large_state_newsgated.json"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
SUMMARY_PATH = HERE / "results/simulation/large_summary_newsgated.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/large_equity_curve_newsgated.csv"

STARTING_CAPITAL = 500.0       # illustrative, same as Bot#3 -- change freely
STARTING_SLOTS = 30            # same as Bot#3 -- "large" vs Bot#2's 15
TARGET_POSITION_SIZE = STARTING_CAPITAL / STARTING_SLOTS

NEWSGATED_LEDGER_COLUMNS = LEDGER_COLUMNS + [
    "news_source", "news_sentiment", "news_reason",
    "customer_concentration", "customer_concentration_reason",
]


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH)
        for c in NEWSGATED_LEDGER_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[NEWSGATED_LEDGER_COLUMNS]
    return pd.DataFrame(columns=NEWSGATED_LEDGER_COLUMNS)


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
    state_count = int(ledger.loc[ledger["status"] == "open", "country"].map(is_state_linked).sum())

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
        if len(sector_pool):
            max_na = math.floor((total_held + 1) * NORTH_AMERICA_MAX_SHARE)
            max_state = math.floor((total_held + 1) * STATE_LINKED_MAX_SHARE)
            geo_ok = sector_pool["ticker"].map(
                lambda t: ticker_region(t) != "North America" or na_count < max_na)
            geo_ok &= sector_pool.get("country", pd.Series(index=sector_pool.index, dtype=object)).map(
                lambda c: not is_state_linked(c) or state_count < max_state)
            eligible = sector_pool[geo_ok]
            if not len(eligible):
                eligible = sector_pool
            eligible = eligible.sort_values("score", ascending=False)
        else:
            eligible = sector_pool

        is_new = len(eligible) > 0
        if is_new:
            pick_row = eligible.iloc[0]
            ticker = pick_row["ticker"]
        else:
            open_in_sector = ledger[(ledger["status"] == "open") &
                                     (ledger["sector"] == target_sector) &
                                     (~ledger["ticker"].isin(rejected))]
            if not len(open_in_sector):
                exhausted.add(target_sector)
                continue
            ticker = open_in_sector.sort_values("last_valuation_gap", ascending=False)["ticker"].iloc[0]

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

        # news gate: last check, whether this is a brand-new buy or a reinforcement of an
        # existing position -- only reached once every other filter has already passed.
        if is_new:
            news_name, news_country = pick_row["name"], pick_row.get("country")
        else:
            open_row = ledger.loc[(ledger["status"] == "open") & (ledger["ticker"] == ticker)].iloc[0]
            news_name, news_country = open_row["name"], open_row.get("country")
        verdict = news_filter.news_verdict(ticker, news_name, target_sector, today)
        if not verdict["relevant"]:
            rejected.add(ticker)
            print(f"  SKIP {ticker} (actu Ollama) : {verdict['reason']}")
            continue

        # customer-concentration: measured, not a veto -- see the same check in
        # simulate_constrained_portfolio_newsgated.fill_slots for the reasoning.
        concentration = news_filter.customer_concentration_verdict(ticker, news_name, news_country)
        size_factor = news_filter.CONCENTRATION_SIZE_FACTOR.get(concentration["concentration"], 1.0)
        target_size = TARGET_POSITION_SIZE * size_factor

        if fractional:
            cost = min(target_size, cash)
            add_shares = cost / price_eur
        else:
            target_shares = max(1, int(target_size // price_eur))
            max_affordable = int(cash // price_eur)
            add_shares = min(target_shares, max_affordable)
            cost = add_shares * price_eur

        if is_new:
            new_row = {
                "ticker": ticker, "name": pick_row["name"], "sector": pick_row["sector"],
                "country": pick_row.get("country"), "status": "open",
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
                "news_source": verdict["source"], "news_sentiment": verdict.get("sentiment"),
                "news_reason": verdict["reason"],
                "customer_concentration": concentration["concentration"],
                "customer_concentration_reason": concentration["reason"],
            }
            ledger = pd.concat([ledger, pd.DataFrame([new_row])], ignore_index=True)
            held_tickers.add(ticker)
            total_held += 1
            if ticker_region(ticker) == "North America":
                na_count += 1
            if is_state_linked(pick_row.get("country")):
                state_count += 1
            sector_counts[target_sector] = sector_counts.get(target_sector, 0) + 1
            kind = "fractionne" if fractional else "entier"
            size_note = ", position reduite (clients concentres)" if size_factor < 1.0 else ""
            print(f"  ACHAT {ticker} ({target_sector}) : {cost:.2f} EUR ({add_shares:.4f} actions, {kind}) "
                  f"@ {fresh['price']:.2f} {fresh.get('currency') or '?'}, score {pick_row['score']:.2f}{size_note}")
        else:
            idx = ledger.index[(ledger["status"] == "open") & (ledger["ticker"] == ticker)][0]
            old_shares = ledger.at[idx, "shares"]
            new_shares = old_shares + add_shares
            new_entry_price = (ledger.at[idx, "entry_price"] * old_shares +
                                fresh["price"] * add_shares) / new_shares
            ledger.at[idx, "entry_price"] = new_entry_price
            ledger.at[idx, "shares"] = new_shares
            ledger.at[idx, "entry_value_eur"] = ledger.at[idx, "entry_value_eur"] + cost
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
    print(f"\n=== Portefeuille large (news-gated) : {summary['nb_open']} positions, "
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
