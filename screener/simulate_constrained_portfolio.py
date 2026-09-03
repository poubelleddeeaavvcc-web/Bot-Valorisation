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
    MAX_PER_SECTOR per sector, and at select_top_picks.NORTH_AMERICA_MAX_SHARE (75%) of
    open positions for US+Canada combined -- both relaxed one step at a time (sector cap
    first) only if too few candidates are available to fill every slot otherwise -- same
    diversification logic used for the "top picks today" report.
  - No hard position-count ceiling: as gains compound, cash naturally clears the
    TARGET_POSITION_SIZE bar more than once per run, so a big win can open 2+ new slots
    in the same cycle. Growth scales the *number* of positions, not just their size --
    per the user's instruction, a windfall should diversify into a 5th/6th/7th name
    rather than doubling down on fewer, bigger bets. Starts at STARTING_SLOTS=15 (~33 EUR
    each) rather than a handful of strong convictions -- the user wants many small,
    incrementally-reinvested gains over a few large bets; existing open positions keep
    their original entry size when this target changes, only new buys use the new one.

Currency: candidate prices come straight from yfinance in the listing's native currency
(USD, JPY, GBp/pence, ...), not EUR -- naive division against a EUR cash balance was
silently wrong before this was caught (2026-08-17). All cash/cost arithmetic below goes
through to_eur() using live FX rates fetched once per run.

Fractional shares: IBKR only supports fractional trading on a specific allowlist -- US
and Canadian stocks broadly, European stocks only above a size+liquidity bar (~$5B market
cap, ~$5M average daily $ volume; see https://www.interactivebrokers.com/en/trading/
fractional-trading.php, checked 2026-08-17), and NOT AT ALL on Japan/Taiwan/South
Korea/Hong Kong/India/Australia listings, which is a meaningful chunk of this bot's
universe. fractional_eligible() approximates that allowlist; ineligible or liquidity-
unknown candidates buy whole shares instead (at least 1, if affordable) rather than being
skipped -- per the user's direction, don't leave cash idle by avoiding a good candidate
just because it isn't fractional-eligible.
"""
import json
import math
import pathlib
import sys
import time

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.select_top_picks import (  # noqa: E402
    composite_score, ticker_region, is_state_linked, NORTH_AMERICA_MAX_SHARE, STATE_LINKED_MAX_SHARE,
)
from screener.simulate_portfolio import fails_fresh_check, fetch_fresh_single, resolve_peer_pe, STOP_LOSS_PCT  # noqa: E402
from screener.fetch_cache import fetch_one as fetch_cache_one  # noqa: E402

LEDGER_PATH = HERE / "results/simulation/constrained_portfolio_ledger.csv"
STATE_PATH = HERE / "results/simulation/constrained_state.json"
CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
SUMMARY_PATH = HERE / "results/simulation/constrained_summary.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/constrained_equity_curve.csv"

STARTING_CAPITAL = 500.0      # illustrative -- the user's own example; change freely
STARTING_SLOTS = 15           # -> TARGET_POSITION_SIZE = 500/15 ~= 33 EUR per slot
TARGET_POSITION_SIZE = STARTING_CAPITAL / STARTING_SLOTS
# raised from 1 alongside STARTING_SLOTS 4->15 (2026-08-26, per the user's direction to grow
# toward many small positions instead of a handful of strong convictions): with only ~8
# sectors typically represented in long_candidates_latest.csv, a strict 1/sector cap would
# make 15 concurrent slots unreachable even before the NA-share and relaxation logic kicks in.
MAX_PER_SECTOR = 3
# a whole-share buy (see fractional_eligible()) costing more than this multiple of the slot
# target is rejected rather than forced -- at a 33 EUR slot, a single ~70+ EUR non-fractional
# name (e.g. TSMC) would otherwise eat 2-3 slots' worth of budget in one line, defeating the
# "many small positions" goal. Fractional-eligible names are unaffected: they always buy
# exactly TARGET_POSITION_SIZE regardless of share price.
MAX_WHOLE_SHARE_OVERSHOOT = 2.5

LEDGER_COLUMNS = [
    "ticker", "name", "sector", "country", "status", "currency", "fractional",
    "entry_date", "entry_price", "shares", "entry_value_eur",
    "entry_valuation_gap", "entry_quality_multiplier", "entry_mom_12_2", "entry_sector_momentum",
    "last_check_date", "last_price", "last_valuation_gap", "last_mom_12_2",
    "current_value_eur", "unrealized_return_pct",
    "exit_date", "exit_price", "exit_reason", "exit_value_eur", "return_pct", "holding_days",
]

# Yahoo FX ticker for EUR/<currency> -- rate = units of <currency> per 1 EUR, so
# price_in_that_currency / rate = price_in_EUR. GBp (pence) shares the GBP rate but needs
# an extra /100 -- handled in to_eur(), not here.
FX_PAIR = {
    "USD": "EURUSD=X", "GBP": "EURGBP=X", "GBp": "EURGBP=X", "JPY": "EURJPY=X",
    "CHF": "EURCHF=X", "SEK": "EURSEK=X", "CAD": "EURCAD=X", "HKD": "EURHKD=X",
    "AUD": "EURAUD=X", "KRW": "EURKRW=X", "TWD": "EURTWD=X", "INR": "EURINR=X",
    "BRL": "EURBRL=X", "MXN": "EURMXN=X",
}

# Suffix -> IBKR fractional-trading region bucket. US (no suffix) and Canada (.TO) are
# broadly eligible; European exchanges are gated by size+liquidity (checked per-candidate
# in fractional_eligible()); everything else on this list is never fractional-eligible.
CANADA_SUFFIX = ".TO"
EUROPE_SUFFIXES = (".PA", ".DE", ".L", ".SW", ".MI", ".AS", ".ST", ".IR", ".LS", ".BR", ".OL")
# Brazil (.SA) and Mexico (.MX) added 2026-08-20 alongside the new B3/IPC universe sources
# -- not on IBKR's fractional-trading allowlist, whole shares only.
NEVER_FRACTIONAL_SUFFIXES = (".T", ".TW", ".KS", ".HK", ".NS", ".BO", ".AX", ".SA", ".MX")
EUROPE_MARKET_CAP_FLOOR_EUR = 4_600_000_000   # ~$5B at a rough EURUSD~1.08
EUROPE_ADV_FLOOR_EUR = 4_600_000              # ~$5M


def fetch_fx_rates(currencies: set) -> dict:
    rates = {"EUR": 1.0}
    for ccy in currencies:
        key = "GBP" if ccy == "GBp" else ccy
        if key in rates or key not in FX_PAIR:
            continue
        try:
            hist = yf.Ticker(FX_PAIR[key]).history(period="5d")["Close"].dropna()
            rates[key] = float(hist.iloc[-1]) if len(hist) else None
        except Exception as e:
            print(f"  echec fetch FX pour {key}: {e}", file=sys.stderr)
            rates[key] = None
    return rates


def to_eur(amount, currency, fx_rates):
    if amount is None or pd.isna(amount):
        return None
    if currency is None or pd.isna(currency) or currency == "EUR":
        return amount  # unknown currency assumed EUR/USD-ish -- see module docstring on self-healing
    rate = fx_rates.get("GBP" if currency == "GBp" else currency)
    if rate is None or rate == 0:
        return None
    converted = amount / rate
    return converted / 100 if currency == "GBp" else converted


def fractional_eligible(ticker: str, market_cap_eur, avg_dollar_vol_eur) -> bool:
    if "." not in ticker or ticker.endswith(CANADA_SUFFIX):
        return True  # US (no suffix) or Canada
    if ticker.endswith(EUROPE_SUFFIXES):
        return (pd.notna(market_cap_eur) and market_cap_eur >= EUROPE_MARKET_CAP_FLOOR_EUR and
                pd.notna(avg_dollar_vol_eur) and avg_dollar_vol_eur >= EUROPE_ADV_FLOOR_EUR)
    if ticker.endswith(NEVER_FRACTIONAL_SUFFIXES):
        return False
    return False  # unrecognized suffix -- conservative default, whole shares only


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


def recheck_and_exit(ledger: pd.DataFrame, valuation: pd.DataFrame, today: str, cash: float,
                      fx_rates: dict) -> tuple:
    sector_pe = valuation.groupby("sector")["sector_median_pe"].first()
    sector_mom = valuation.groupby("sector")["sector_momentum"].first()
    industry_pe = valuation.groupby("industry")["industry_median_pe"].first()
    industry_count = valuation.groupby("industry")["industry_count"].first()

    for idx in ledger.index[ledger["status"] == "open"]:
        ticker = ledger.at[idx, "ticker"]
        fresh = fetch_fresh_single(ticker)
        if fresh is None or fresh["price"] is None or fresh["eps"] is None:
            continue

        sector = fresh["sector"] or ledger.at[idx, "sector"]
        today_peer_pe = resolve_peer_pe(sector, fresh.get("industry"), sector_pe, industry_pe, industry_count)
        today_sector_mom = sector_mom.get(sector, 0.0)
        qmult = ledger.at[idx, "entry_quality_multiplier"]

        if today_peer_pe is not None and pd.notna(qmult):
            fair_value_now = fresh["eps"] * today_peer_pe * qmult
            valuation_gap_now = fair_value_now / fresh["price"] - 1
        else:
            valuation_gap_now = ledger.at[idx, "last_valuation_gap"]

        entry_price = ledger.at[idx, "entry_price"]
        shares = ledger.at[idx, "shares"]
        unrealized = fresh["price"] / entry_price - 1
        # currency is fixed at entry (a listing doesn't change currency), so reuse it rather
        # than needing fetch_fresh_single to report it -- only the FX *rate* needs refreshing.
        current_value = to_eur(shares * fresh["price"], ledger.at[idx, "currency"], fx_rates)
        if current_value is None:
            continue  # FX rate unavailable this run -- skip valuing/exiting, retry next run

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
    # unlike NA (derivable from the ticker string alone via ticker_region), state-linkage
    # depends on Yahoo's HQ-country data, so it has to be persisted on the ledger row at
    # entry time (see "country" in LEDGER_COLUMNS) rather than recomputed from the ticker.
    state_count = int(ledger.loc[ledger["status"] == "open", "country"].map(is_state_linked).sum())

    pool = candidates[~candidates["ticker"].isin(held_tickers)].copy()
    if not len(pool):
        return ledger, cash
    pool["score"] = composite_score(pool)
    pool = pool.sort_values("score", ascending=False)
    # tickers tried and rejected this run (fetch failed, unaffordable, ...) -- don't retry
    # them within the same run, but they're free to come up again next run.
    rejected = set()

    new_rows = []
    while True:
        pick_row = None
        # try honoring the North America and state-linked caps first (see
        # select_top_picks.NORTH_AMERICA_MAX_SHARE / STATE_LINKED_MAX_SHARE), across every
        # sector-cap relaxation level; only if that combination is truly starved of options
        # does the second pass drop both caps together -- same "don't leave cash idle over a
        # diversification target" logic as the sector cap relaxation below.
        for enforce_geo_caps in (True, False):
            for cap in range(MAX_PER_SECTOR, 10):  # relax the sector cap only if truly starved of options
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
            break  # nothing left to try, however relaxed -- stop, keep the cash

        ticker = pick_row["ticker"]
        # fresh single-ticker fetch at the moment of purchase: currency/avg_volume are new
        # fields (added 2026-08-17) that the 7-day progressive cache hasn't necessarily
        # backfilled for this ticker yet, and affordability/fractional-eligibility depend on
        # both -- guessing wrong here means real money math, not just a stale display.
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

        # fresh momentum/valuation gate: fetch_cache_one's data can be hours to a week stale
        # (the same rotation cadence as the screener's candidate list), so re-verify the
        # entry bar with the fetch we just did before actually spending the cash -- a
        # candidate that's already failing it shouldn't be bought at all (2026-09-02, see
        # simulate_portfolio.py's module docstring for the same-day round-trip bug this fixes).
        fails, state = fails_fresh_check(fresh, pick_row["quality_multiplier"], sector_pe, sector_mom,
                                          industry_pe, industry_count, fallback_valuation_gap=pick_row["valuation_gap"])
        if fails:
            rejected.add(ticker)
            continue

        if fractional:
            cost = min(TARGET_POSITION_SIZE, cash)
            shares = cost / price_eur
        else:
            # whole shares only (not IBKR fractional-eligible): buy as many as get closest to
            # the target size without exceeding available cash -- at least 1, guaranteed
            # affordable by the price_eur<=cash check above.
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

    # fetched unconditionally (not just currencies already in play) -- fill_slots discovers
    # a candidate's currency one at a time via a fresh per-ticker fetch, so which currencies
    # will be needed isn't known upfront; there are only 11 pairs total, cheap either way.
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
