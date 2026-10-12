"""Bot #41 Qualite (2026-10-12): owns the 15 companies with the best NOTE_QUALITE (/20, see
quality_perspective_notes.py: Bilan, Rendement du capital, Cash, Marges, Discipline -- each a
within-sector percentile), and nothing else -- no valuation, no momentum, no stop-loss. One of the
three single-idea bots asked for by the user (with #42 Valeur and #43 Tendance), meant to be
combined later; see sleeve_book.py for fills (close of the next session) and total-return
accounting, which all three share.

Rules (fixed before any result, per the multiple-testing point discussed 2026-10-09):
  - universe: full_valuation_latest.csv (profitable, deduplicated share classes) with a quality
    note computed on >= MIN_PILLARS pillars and not flagged low-confidence; closed-end funds and
    royalty trusts excluded (EXCLUDED_INDUSTRIES / EXCLUDED_NAME_RE): on 2026-10-12 Gabelli Equity
    Trust and Sabine Royalty Trust ranked in the top 15 although a fund's "margins" or "buybacks"
    say nothing about a business; listings trading under MIN_ADV_EUR a day excluded too (mostly
    Taiwan depositary receipts and secondary listings, e.g. 911608.TW at ~2,000 EUR/day: their
    last price can be days old, the stale-fill problem again);
  - SLOTS lines of equity/SLOTS each, at most MAX_PER_SECTOR per sector;
  - rebalanced once per calendar quarter (first run of the quarter): a held line is sold only if it
    falls out of the top KEEP_RANK, so ordinary rank noise doesn't cost a 1 EUR fee on a 33 EUR line;
  - between rebalances an empty slot (unaffordable whole share, failed fill...) is refilled from the
    ranking saved at the last rebalance, never from a fresher one.

Run order in CI: after the `setup` job (notes + valuation refreshed), see update-screener.yml.
"""
# Alias lisibilite (mapping perso) : Q1 -- strategie Qualite seule
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.simulate_constrained_portfolio import (  # noqa: E402
    MAX_WHOLE_SHARE_OVERSHOOT, fractional_eligible, to_eur,
)
from screener.sleeve_book import Book, now_utc, quarter_key  # noqa: E402

VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
NOTES_PATH = HERE / "results/screener/quality_perspective_notes.csv"

STARTING_CAPITAL = 500.0
SLOTS = 15
KEEP_RANK = 45
MAX_PER_SECTOR = 3
MIN_POSITION_EUR = STARTING_CAPITAL / SLOTS / 2  # same floor as bots #38-40
RANKING_DEPTH = 120  # names saved at each rebalance for in-quarter refills
MIN_PILLARS = 4
EXCLUDED_INDUSTRIES = {"Asset Management", "Shell Companies"}
EXCLUDED_NAME_RE = r"Royalty Trust"
MIN_ADV_EUR = 1_000_000  # 3,217 of 3,653 listings pass on 2026-10-12
UNIVERSE_COLUMNS = ["ticker", "name", "sector", "industry", "country", "currency", "price", "market_cap", "avg_volume"]


def load_universe(fx_rates: dict) -> pd.DataFrame:
    df = pd.read_csv(VALUATION_PATH)
    df = df[~df["industry"].isin(EXCLUDED_INDUSTRIES)]
    df = df[~df["name"].fillna("").str.contains(EXCLUDED_NAME_RE, case=False, regex=True)]
    df = df.dropna(subset=["sector", "price", "avg_volume"])
    adv = [to_eur(v * p, c, fx_rates) for v, p, c in zip(df["avg_volume"], df["price"], df["currency"])]
    return df[pd.Series(adv, index=df.index, dtype=float) >= MIN_ADV_EUR]


def quality_ranking(universe: pd.DataFrame) -> pd.DataFrame:
    notes = pd.read_csv(NOTES_PATH)
    ok = (notes["note_qualite_n_piliers"] >= MIN_PILLARS) & ~notes["note_qualite_low_confidence"].fillna(True).astype(bool)
    notes = notes.loc[ok & notes["note_qualite_20"].notna(), ["ticker", "note_qualite_20"]]
    df = universe[UNIVERSE_COLUMNS].merge(notes, on="ticker").rename(columns={"note_qualite_20": "score"})
    return df.sort_values(["score", "ticker"], ascending=[False, True]).reset_index(drop=True)


def _ranking_records(book: Book, ranked: pd.DataFrame) -> list:
    records = []
    for row in ranked.head(RANKING_DEPTH).itertuples(index=False):
        price_eur = to_eur(row.price, row.currency, book.fx)
        mcap_eur = to_eur(row.market_cap, row.currency, book.fx)
        adv_eur = to_eur(row.avg_volume * row.price, row.currency, book.fx) if pd.notna(row.avg_volume) else None
        records.append({"ticker": row.ticker, "name": row.name, "sector": row.sector, "country": row.country,
                        "currency": row.currency, "score": round(float(row.score), 4),
                        "price_eur": price_eur, "fractional": bool(fractional_eligible(row.ticker, mcap_eur, adv_eur))})
    return records


def fill_slots(book: Book, now: pd.Timestamp, reason: str):
    pending = book.state["pending"]
    selling = {o["ticker"] for o in pending if o["side"] == "sell"}
    buying = [o for o in pending if o["side"] == "buy"]
    lots = book.open_lots()
    kept = lots[~lots["ticker"].isin(selling)]
    taken = set(kept["ticker"]) | {o["ticker"] for o in buying}
    sectors = kept.drop_duplicates("ticker")["sector"].value_counts().to_dict()
    for o in buying:
        sectors[o["sector"]] = sectors.get(o["sector"], 0) + 1
    free_cash = (book.state["cash"]["pool"] - sum(o["amount_eur"] for o in buying)
                 + sum(book.position_value(ticker=t) for t in selling))
    target = book.equity() / SLOTS
    dropped = set(book.state.get("dropped", []))
    for cand in book.state.get("ranking", []):
        if len(taken) >= SLOTS or free_cash < MIN_POSITION_EUR:
            break
        if (cand["ticker"] in taken or cand["ticker"] in selling or cand["ticker"] in dropped
                or sectors.get(cand["sector"], 0) >= MAX_PER_SECTOR):
            continue
        if not cand["fractional"] and (cand["price_eur"] is None or cand["price_eur"] > MAX_WHOLE_SHARE_OVERSHOOT * target):
            continue
        amount = min(target, free_cash)
        book.add_order("buy", cand["ticker"], "pool", f"{reason}, score {cand['score']:g}", now, amount_eur=amount,
                       name=cand["name"], sector=cand["sector"], country=cand["country"], currency=cand["currency"],
                       fractional=cand["fractional"], score=cand["score"])
        taken.add(cand["ticker"])
        sectors[cand["sector"]] = sectors.get(cand["sector"], 0) + 1
        free_cash -= amount


def rebalance(book: Book, ranked: pd.DataFrame, now: pd.Timestamp):
    rank = {t: i + 1 for i, t in enumerate(ranked["ticker"])}
    book.state["pending"] = []  # unfilled orders of the previous quarter are superseded
    book.state["dropped"] = []
    for ticker in sorted(set(book.open_lots()["ticker"])):
        r = rank.get(ticker)
        if r is None or r > KEEP_RANK:
            book.add_order("sell", ticker, "pool", f"hors du top {KEEP_RANK} (rang {r or 'hors univers'})", now, fraction=1.0)
    book.state["ranking"] = _ranking_records(book, ranked)
    book.state["rebalanced_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    fill_slots(book, now, "revision trimestrielle")


def run_factor_bot(stem: str, label: str, ranking_fn):
    book = Book(stem, label, STARTING_CAPITAL, {"pool": STARTING_CAPITAL})
    if not VALUATION_PATH.exists():
        print("Pas encore de resultats de screener -- rien a simuler.")
        return
    book.load_prices()
    book.mark_to_market()
    book.execute_pending(MIN_POSITION_EUR, MAX_WHOLE_SHARE_OVERSHOOT * STARTING_CAPITAL / SLOTS)
    book.mark_to_market()  # lots just filled: from their entry close to the latest bar

    now = now_utc()
    key = quarter_key(now)
    if book.state.get("rebalance_key") != key:
        ranked = ranking_fn(load_universe(book.fx))
        print(f"Revision {key} : {len(ranked)} titres classes, top 5 : "
              + ", ".join(f"{t} ({s:g})" for t, s in zip(ranked["ticker"].head(5), ranked["score"].head(5))))
        rebalance(book, ranked, now)
        book.state["rebalance_key"] = key
    else:
        fill_slots(book, now, "remplacement")
    book.save({"rebalance_key": key, "slots": SLOTS})


def main():
    run_factor_bot("quality", "Bot #41 Qualite", quality_ranking)


if __name__ == "__main__":
    main()
