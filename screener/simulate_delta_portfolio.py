"""Bot #25 ("Delta"): same capital-constrained paper-trading mechanics as
simulate_constrained_portfolio.py (Bot#2, "Beta") -- same entry/exit rules, same FX and
fractional-share handling, same diversified reinvestment into new slots -- but with one added
mechanic: conviction-based averaging down.

The idea, from the user's own framing (2026-09-10): a position that has pulled back from its
entry cost isn't automatically abandoned. If the same momentum/valuation signal that gates
every entry and exit still says the thesis is intact (the position hasn't hit any of
recheck_and_exit's exit triggers), the bot can reinject capital into it -- lowering the average
cost basis and increasing the eventual gain if/when the thesis plays out, instead of just
waiting it out at the original size. This isolates the effect of conviction-averaging against
Bot#2's baseline: same candidate selection, same starting capital shape, same exit rules, the
only difference is what happens to an existing position while it's underwater.

Mechanics:
  - Starts at STARTING_CAPITAL (300 EUR), all in cash -- deliberately smaller and separate
    from Bot#2/3's illustrative 500 EUR, its own pool, its own ledger.
  - Every run, in order: recheck_and_exit (identical to Bot#2/3, including the ratcheting
    stop -- see simulate_portfolio.py) closes anything that broke its thesis or hit a
    stop; reinforce_convictions (see below) gets first claim on the freed-up + existing cash
    for any open position that still qualifies; fill_slots then spends whatever cash is left
    on brand-new diversified candidates, exactly like Bot#2.
  - See reinforce_convictions() for the conviction-averaging gate and its reasoning.

Currency/fractional-share handling, exit rules (including the ratcheting stop): identical to
Bot#2/3, imported directly from simulate_constrained_portfolio.py rather than duplicated -- same
"single authoritative implementation, every capital-tracking bot imports it" pattern Bot#3
already uses for these, since none of it depends on how a position was opened.
"""
# Alias lisibilite (mapping perso) : D1 -- famille Delta (conviction), variante base
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
from screener.simulate_portfolio import fails_fresh_check, fetch_fresh_single, STOP_LOSS_PCT  # noqa: E402
from screener.simulate_constrained_portfolio import (  # noqa: E402
    LEDGER_COLUMNS as BASE_LEDGER_COLUMNS, MAX_PER_SECTOR, MAX_WHOLE_SHARE_OVERSHOOT, FX_PAIR,
    fetch_fx_rates, fractional_eligible, recheck_and_exit, to_eur,
)
from screener.fetch_cache import fetch_one as fetch_cache_one  # noqa: E402

LEDGER_PATH = HERE / "results/simulation/delta_portfolio_ledger.csv"
STATE_PATH = HERE / "results/simulation/delta_state.json"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
SUMMARY_PATH = HERE / "results/simulation/delta_summary.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/delta_equity_curve.csv"

STARTING_CAPITAL = 300.0      # separate, smaller pool than Bot#2/3's 500 EUR -- the user's own example
STARTING_SLOTS = 9            # -> TARGET_POSITION_SIZE ~= 33 EUR/slot, same granularity as Bot#2/3
TARGET_POSITION_SIZE = STARTING_CAPITAL / STARTING_SLOTS
# MAX_PER_SECTOR/MAX_WHOLE_SHARE_OVERSHOOT reused as-is from Bot#2/3 (imported above) -- no
# reason for Delta's diversification/whole-share caps to differ from Bot#2's.

# --- conviction-averaging gate, see reinforce_convictions() ------------------------------
# Below this drop, a pullback is just noise -- not worth reinforcing. Kept comfortably short of
# STOP_LOSS_PCT (-15%, see simulate_portfolio.py) so a reinforcement never lands right at the
# edge of the hard stop; in practice averaging down also *helps* here, since a lower entry price
# pulls unrealized_return_pct back toward 0 and buys room before the floor (2026-09-10, user
# observation).
DELTA_MIN_DROP_PCT = -0.07
# Past this drop (and still short of STOP_LOSS_PCT), the pullback counts as "deep" -- the
# stricter conviction bar below applies instead of the base one.
DELTA_DEEP_DROP_PCT = -0.10
# Required valuation_gap_now (still-undervalued margin, not just >0) to reinforce a "moderate"
# pullback (DELTA_MIN_DROP_PCT..DELTA_DEEP_DROP_PCT).
DELTA_CONVICTION_MARGIN_LOW = 0.10
# Required valuation_gap_now for a "deep" pullback (below DELTA_DEEP_DROP_PCT) -- meaningfully
# higher bar, per the user's "plus la baisse est forte, plus la conviction doit etre forte".
DELTA_CONVICTION_MARGIN_HIGH = 0.20
# Required momentum edge over the sector (mom_12_2 - sector_momentum) for a "deep" pullback --
# not required at all for a "moderate" one (already-open positions have cleared momentum_lost's
# bar, which only requires the edge to be positive, not any particular size).
DELTA_MOM_SPREAD_MIN = 0.05
# Cap on total capital (entry_value_eur, every buy + reinforcement combined) a single position
# can absorb, as a share of STARTING_CAPITAL -- however strong the conviction looks, one falling
# name is never allowed to swallow more than this fraction of the 300 EUR pool.
DELTA_MAX_POSITION_SHARE = 0.25

LEDGER_COLUMNS = BASE_LEDGER_COLUMNS + ["reinforcement_count"]


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


def reinforce_convictions(ledger: pd.DataFrame, valuation: pd.DataFrame, today: str, cash: float,
                           fx_rates: dict) -> tuple:
    """The bot's namesake mechanic: an open position that has pulled back from its cost basis
    but whose momentum/valuation signal still looks bullish (it hasn't hit any of
    recheck_and_exit's exit triggers, called right before this) gets MORE capital rather than
    being left alone -- lowers the average cost basis and increases the eventual gain if the
    thesis plays out. Runs before fill_slots so an existing conviction gets first claim on cash
    over a brand-new candidate.

    Gate, from the user's own framing (2026-09-10) -- 'plus la baisse est forte, plus la
    conviction doit etre forte a court terme': the required margin scales with how deep the
    pullback already is, and a short-term signal (mom_1m, the single most recent month's
    return -- the one thing mom_12_2 deliberately excludes) has to be positive either way, since
    averaging into a name that's still actively falling right now defeats the purpose:
      - DELTA_MIN_DROP_PCT <= unrealized < DELTA_DEEP_DROP_PCT ("moderate" pullback): needs
        valuation_gap_now >= DELTA_CONVICTION_MARGIN_LOW and mom_1m > 0.
      - unrealized < DELTA_DEEP_DROP_PCT ("deep" pullback, closer to STOP_LOSS_PCT): needs the
        stricter valuation_gap_now >= DELTA_CONVICTION_MARGIN_HIGH, a real momentum edge over
        the sector (mom_12_2 - sector_momentum >= DELTA_MOM_SPREAD_MIN), and mom_1m > 0.
    A position never absorbs more than DELTA_MAX_POSITION_SHARE of STARTING_CAPITAL in total
    (entry_value_eur across every buy and reinforcement combined).

    mom_1m needs its own fresh fetch (recheck_and_exit's fetch_fresh_single call already
    happened this run but didn't need it) -- kept cheap by only fetching for positions that
    already cleared the valuation/momentum gate above, not every open position.
    """
    sector_mom = valuation.groupby("sector")["sector_momentum"].first()
    cap_eur = DELTA_MAX_POSITION_SHARE * STARTING_CAPITAL

    candidates_idx = ledger.index[
        (ledger["status"] == "open") &
        (ledger["unrealized_return_pct"] <= DELTA_MIN_DROP_PCT) &
        (ledger["unrealized_return_pct"] > STOP_LOSS_PCT)
    ]
    for idx in candidates_idx:
        if ledger.at[idx, "entry_value_eur"] >= cap_eur:
            continue  # already at the position cap -- no room left regardless of conviction

        unrealized = ledger.at[idx, "unrealized_return_pct"]
        valuation_gap_now = ledger.at[idx, "last_valuation_gap"]
        mom_12_2 = ledger.at[idx, "last_mom_12_2"]
        if pd.isna(valuation_gap_now) or pd.isna(mom_12_2):
            continue

        deep = unrealized <= DELTA_DEEP_DROP_PCT
        margin_needed = DELTA_CONVICTION_MARGIN_HIGH if deep else DELTA_CONVICTION_MARGIN_LOW
        spread_needed = DELTA_MOM_SPREAD_MIN if deep else 0.0
        today_sector_mom = sector_mom.get(ledger.at[idx, "sector"], 0.0)
        if not (valuation_gap_now >= margin_needed and mom_12_2 - today_sector_mom >= spread_needed):
            continue

        ticker = ledger.at[idx, "ticker"]
        fresh = fetch_fresh_single(ticker)
        if fresh is None or fresh.get("price") is None or pd.isna(fresh.get("mom_1m")) or fresh["mom_1m"] <= 0:
            continue

        price_eur = to_eur(fresh["price"], ledger.at[idx, "currency"], fx_rates)
        if price_eur is None or price_eur <= 0:
            continue

        headroom = cap_eur - ledger.at[idx, "entry_value_eur"]
        budget = min(TARGET_POSITION_SIZE, headroom, cash)
        if budget <= 0:
            continue

        if bool(ledger.at[idx, "fractional"]):
            add_shares = budget / price_eur
            cost = budget
        else:
            add_shares = int(budget // price_eur)
            if add_shares < 1:
                continue
            cost = add_shares * price_eur

        old_shares = ledger.at[idx, "shares"]
        new_shares = old_shares + add_shares
        new_entry_price = (ledger.at[idx, "entry_price"] * old_shares + fresh["price"] * add_shares) / new_shares
        ledger.at[idx, "entry_price"] = new_entry_price
        ledger.at[idx, "shares"] = new_shares
        ledger.at[idx, "entry_value_eur"] = ledger.at[idx, "entry_value_eur"] + cost
        # move with the added shares immediately, same fix as Bot#3's RENFORCE -- see
        # simulate_large_portfolio.py's reinforcement branch for the bug this avoids.
        ledger.at[idx, "current_value_eur"] = ledger.at[idx, "current_value_eur"] + cost
        ledger.at[idx, "unrealized_return_pct"] = fresh["price"] / new_entry_price - 1
        prior_count = ledger.at[idx, "reinforcement_count"]
        ledger.at[idx, "reinforcement_count"] = (0 if pd.isna(prior_count) else prior_count) + 1
        cash -= cost
        band = "profonde" if deep else "moderee"
        print(f"  RENFORT CONVICTION {ticker} (baisse {band}, {unrealized:+.1%} avant renfort) : "
              f"+{cost:.2f} EUR ({add_shares:.4f} actions) @ {fresh['price']:.2f} "
              f"{ledger.at[idx, 'currency'] or '?'}, nouveau prix de revient {new_entry_price:.4f}")

    return ledger, cash


def fill_slots(ledger: pd.DataFrame, candidates: pd.DataFrame, valuation: pd.DataFrame, cash: float, today: str,
               fx_rates: dict) -> tuple:
    """Identical to Bot#2/3's diversified-buy logic -- see simulate_constrained_portfolio.py."""
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
            "unrealized_return_pct": 0.0, "peak_unrealized_return_pct": 0.0, "peak_date": today,
            "reinforcement_count": 0,
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
        "nb_reinforced": int((open_pos["reinforcement_count"].fillna(0) > 0).sum()) if len(open_pos) else 0,
        "win_rate_closed": float((closed["return_pct"] > 0).mean()) if len(closed) else None,
        "avg_return_closed": float(closed["return_pct"].mean()) if len(closed) else None,
    }
    SUMMARY_PATH.write_text(pd.Series(summary).to_json(), encoding="utf-8")
    print(f"\n=== Portefeuille Delta : {summary['nb_open']} positions ({summary['nb_reinforced']} renforcees), "
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
    ledger, cash = reinforce_convictions(ledger, valuation, today, cash, fx_rates)
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
