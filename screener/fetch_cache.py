"""Step 2 of the progressive-cache pipeline: fetch fundamentals+momentum for the
trending-sector universe (from build_trending_universe.py), respecting Yahoo Finance's
undocumented rate limit -- small batch, low concurrency, a delay between requests, and
exponential backoff that ABORTS the run early on repeated rate-limit errors rather than
digging the hole deeper. Results accumulate in a persistent CSV cache across runs.

REFRESH SCHEDULE (rewritten 2026-09-02, then again same day per the user's follow-up): each
hourly run refreshes a slice of the universe determined by market_hours.py -- a ticker is only
ever eligible during its OWN exchange's trading hours (see market_hours.MARKET_HOURS), and
within that window it's deterministically assigned to one of that market's slots (a stable
hash of the ticker, not its position in the list, so slot assignment doesn't shift just
because build_trending_universe.py's output reorders or resizes day to day). Over one BUSINESS
week (5 trading days) every ticker gets refreshed exactly once, spread evenly across whichever
of its market's hourly session-hours the cron happens to land on -- never while that market is
closed, since re-fetching a closed market just re-reads the same last close.

This is the second iteration of the schedule: the first (same day, commit 068d52d) replaced a
pure staleness check (refresh only if last fetched >7 days ago -- completes in ~13h then goes
fully IDLE for ~6.5 days, so the whole universe's momentum updated in one clustered weekly
burst, flipping several stocks' momentum status at once) with a flat 168-slot calendar-week
rotation, evenly spread but blind to whether any given market was actually open. This version
additionally ties each ticker's slot to its own market's real trading calendar, per the user's
direction that calendar-week/anytime-of-day refreshing isn't "clean" -- fetching AAPL at 3am
New York time doesn't produce fresher data, it just spends rate-limit budget for nothing.

New tickers (never in the cache at all -- just entered the trending universe) are fetched
immediately rather than waiting for their slot, since they might not get one for days.
"""
import hashlib
import pathlib
import sys
import time

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))  # this script is invoked directly (python screener/fetch_cache.py,
# see update-screener.yml), so "screener" isn't importable as a package without this -- same
# pattern used by every other intra-package import in this project (e.g.
# simulate_constrained_portfolio.py)

from screener import market_hours  # noqa: E402

TRENDING_UNIVERSE = HERE / "data/universe/trending_universe.csv"
CACHE_PATH = HERE / "results/screener/fundamentals_cache.csv"

BATCH_SIZE = 423          # hard ceiling on requests/run regardless of schedule -- 70% of the
# 604 that worked in the original one-shot test, still a guess (Yahoo publishes no real
# limit). With market-hours rotation this is rarely reached (each run's regular slice is a
# small fraction of whichever markets happen to be open) except while the
# LAST_SCHEMA_MIGRATION backlog below is draining.
DELAY_BETWEEN_CALLS = 0.4  # seconds, single-threaded on purpose
COOLDOWN_EVERY = 100      # community-reported pattern (yfinance GH discussion #2431):
COOLDOWN_SECONDS = 20     # ~100 requests before Yahoo wants a breather -- so take one voluntarily
MAX_RETRIES = 2

LAST_SCHEMA_MIGRATION = "2026-09-03"  # date fetch_one() last gained new fields (cash/quality
# fields for the 2-note display system -- see quality_perspective_notes.py -- plus the prior
# country/analyst-consensus migration from 2026-09-01/02, unified under this one cutoff). A
# row fetched before this date predates those columns entirely and would otherwise wait for
# its normal weekly slot to backfill them -- rows below this cutoff are pulled into every run
# (on top of that run's regular slice) until they've all been refetched once, then this has
# no further effect (a refetched row's date moves past the cutoff, so it can never re-trigger).


def _avg_nonnull(cf: pd.DataFrame, row_names: list, lookback: int = 3) -> float | None:
    """3-year rolling average of a cashflow-statement line, for quality_perspective_notes.py's
    Discipline pillar (buybacks/dividends) -- smooths a single lumpy fiscal year rather than
    reading only the latest one. None only if the cashflow statement itself failed to fetch
    (cf.empty); 0.0 if the line genuinely doesn't exist for this filer or its recent values
    are all NaN -- a real zero (never buys back / never pays a dividend), not missing data.
    Confirmed in testing (2026-09-03, TSLA/RIVN): treating an absent row as None instead of 0
    let a company that returns nothing to shareholders rank as "best in sector" once its two
    zero-return peers got excluded from the percentile -- the opposite of reality."""
    if cf.empty:
        return None
    for row_name in row_names:
        if row_name in cf.index:
            series = cf.loc[row_name].iloc[:lookback].dropna()
            return float(series.mean()) if len(series) else 0.0
    return 0.0


def fetch_one(ticker: str) -> dict:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    for attempt in range(MAX_RETRIES + 1):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            hist = t.history(period="14mo", interval="1mo", auto_adjust=True)["Close"].dropna()
            mom_12_2 = hist.iloc[-2] / hist.iloc[-13] - 1 if len(hist) >= 13 else None
            # cashflow statement -- one extra request, needed only for the Discipline pillar
            # (buybacks/dividends aren't in `info`). Any failure here falls through to the
            # same except/retry path as info/history above, same as every other field.
            cf = t.cashflow
            buyback_avg_3y = _avg_nonnull(cf, ["Repurchase Of Capital Stock"])
            div_avg_3y = _avg_nonnull(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"])
            return {
                "ticker": ticker, "fetched_at": today,
                "price": hist.iloc[-1] if len(hist) else None,
                "sector": info.get("sector"), "industry": info.get("industry"),
                # HQ/incorporation country -- used by select_top_picks.is_state_linked() for the
                # geopolitical concentration cap (2026-09-02): a purely quantitative ratio can't
                # tell genuine value from a market position propped up by state subsidy/dumping
                # (the BYD case that prompted this), so exposure to specific countries is capped
                # at the portfolio level instead of trying to score it per-stock.
                "country": info.get("country"),
                # longName over shortName: shortName gets hard-truncated by Yahoo, which can
                # cut off the very word (Depositary, Preferred...) JUNK_NAME_PATTERN needs.
                "name": info.get("longName") or info.get("shortName"),
                # needed by simulate_constrained_portfolio.py: "price" above is in the
                # listing's native currency (JPY, GBp/pence, ...), not EUR or USD, and
                # fractional-share eligibility (IBKR) depends on both liquidity and listing
                # currency/market.
                "currency": info.get("currency"), "avg_volume": info.get("averageVolume"),
                "pe": info.get("trailingPE"), "pb": info.get("priceToBook"),
                "eps": info.get("trailingEps"), "roe": info.get("returnOnEquity"),
                "margin": info.get("profitMargins"), "debt_eq": info.get("debtToEquity"),
                "market_cap": info.get("marketCap"), "peg": info.get("pegRatio"),
                # Analyst consensus (2026-09-01) -- third-party data Yahoo aggregates, not its
                # own number: an independent corroboration signal alongside our own valuation_gap.
                # See compute_valuation() in value_momentum_quality_screener_v2.py for how it's used.
                "target_low_price": info.get("targetLowPrice"), "target_mean_price": info.get("targetMeanPrice"),
                "target_high_price": info.get("targetHighPrice"), "recommendation_key": info.get("recommendationKey"),
                "num_analyst_opinions": info.get("numberOfAnalystOpinions"),
                # Cash/Bilan/Marges/Discipline pillars for quality_perspective_notes.py
                # (2026-09-03) -- free_cashflow/total_cash/total_debt/quick_ratio/
                # operating_margin all come from `info`, no extra request beyond the one
                # already made above. financial_currency flags foreign filers (e.g. TSM
                # reports in TWD while its ADR trades and is capped in USD) whose cashflow
                # figures aren't directly comparable to market_cap -- quality_perspective_notes
                # must treat these as N/A rather than computing a ratio across currencies.
                "free_cashflow": info.get("freeCashflow"), "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"), "quick_ratio": info.get("quickRatio"),
                "operating_margin": info.get("operatingMargins"),
                "financial_currency": info.get("financialCurrency"),
                "buyback_avg_3y": buyback_avg_3y, "div_avg_3y": div_avg_3y,
                "mom_12_2": mom_12_2, "error": None,
            }
        except Exception as e:
            msg = str(e)
            if "Rate limited" in msg or "Too Many Requests" in msg:
                if attempt < MAX_RETRIES:
                    backoff = 5 * (attempt + 1)
                    print(f"  rate-limit sur {ticker}, pause {backoff}s...", file=sys.stderr)
                    time.sleep(backoff)
                    continue
                # NOT marked fresh today on purpose: a persistent rate limit isn't the
                # ticker's fault, so it should be retried on the very next run instead
                # of waiting out the 90-day staleness window.
                return {"ticker": ticker, "fetched_at": None, "error": "RATE_LIMITED"}
            # a real per-ticker failure (e.g. delisted, bad symbol) IS marked fresh today,
            # so a permanently-broken ticker doesn't get retried every single run.
            return {"ticker": ticker, "fetched_at": today, "error": msg[:100]}


def load_cache() -> pd.DataFrame:
    if not CACHE_PATH.exists():
        return pd.DataFrame(columns=["ticker", "fetched_at"])
    df = pd.read_csv(CACHE_PATH)
    # format="mixed": pandas otherwise infers a single format from the first value and
    # silently NaTs every row that doesn't match it -- real bug hit in practice here
    # when the column had a mix of "YYYY-MM-DD" and "YYYY-MM-DD HH:MM:SS" strings.
    df["fetched_at"] = pd.to_datetime(df["fetched_at"], format="mixed", errors="coerce")
    return df


def _slot_of(ticker: str, total_slots: int) -> int:
    """Deterministic, stable across processes/runs (unlike Python's built-in hash(), which is
    randomized per-process by default) -- a ticker's slot within its market's weekly rotation
    never moves just because build_trending_universe.py reordered or resized its output."""
    return int(hashlib.md5(ticker.encode("utf-8")).hexdigest(), 16) % total_slots


class _MarketSlots:
    """Memoized (tz_name, open_hour, close_hour) -> current (slot, total_slots) | None --
    computed once per distinct market (there are ~25 in this project's universe) rather than
    once per ticker (thousands), since each lookup does a zoneinfo-aware "what time is it
    there right now" conversion."""

    def __init__(self):
        self._cache = {}

    def get(self, ticker: str):
        market = market_hours.market_of(ticker)[:3]  # (tz_name, open_hour, close_hour)
        if market not in self._cache:
            self._cache[market] = market_hours.current_slot_for_market(*market)
        return self._cache[market]

    def is_open(self, ticker: str) -> bool:
        return self.get(ticker) is not None

    def n_open_markets(self) -> int:
        return sum(1 for v in self._cache.values() if v is not None)


def main():
    universe = pd.read_csv(TRENDING_UNIVERSE)
    universe = universe[universe["ticker"].notna()]  # build_trending_universe.py has been
    # seen to emit one stray all-blank row -- drop it here rather than let it flow into
    # fetch_one(nan) or a hash that can't .encode() a float.
    cache = load_cache()
    cache = cache[cache["ticker"].notna()]  # same defensive drop, cache side (a NaN ticker
    # ends up here too once fetch_one(nan) is called for one from the universe)

    cached_tickers = set(cache["ticker"]) if len(cache) else set()
    never_fetched_all = [t for t in universe["ticker"] if t not in cached_tickers]

    migration_cutoff = pd.Timestamp(LAST_SCHEMA_MIGRATION)
    premigration_all = (sorted(set(cache.loc[cache["fetched_at"] < migration_cutoff, "ticker"]))
                         if len(cache) else [])

    # market-hours gate (see market_hours.py): never fetch a ticker while its own exchange is
    # closed -- re-reading a closed market just returns the same last close, not fresher data.
    # Applies to new tickers and the schema catch-up backlog too (any time their market is
    # open, not slot-restricted -- both want to drain as fast as possible, not spread over
    # another full week), as well as the regular rotation slice below (market open AND this
    # hour is specifically this ticker's slot).
    slots = _MarketSlots()
    never_fetched = [t for t in never_fetched_all if slots.is_open(t)]
    premigration = [t for t in premigration_all if slots.is_open(t)]
    scheduled = []
    for t in universe["ticker"]:
        slot_info = slots.get(t)
        if slot_info is not None and _slot_of(t, slot_info[1]) == slot_info[0]:
            scheduled.append(t)

    # priority order when BATCH_SIZE caps the total: brand-new tickers first (they might not
    # get a slot for days otherwise), then the one-time schema catch-up backlog, then this
    # run's regular weekly-rotation slice. dict.fromkeys dedupes while keeping first-seen order
    # (a ticker can legitimately appear in more than one of the three lists).
    todo = list(dict.fromkeys(never_fetched + premigration + scheduled))[:BATCH_SIZE]

    print(f"Univers cible : {len(universe)} tickers | marches actuellement ouverts : "
          f"{slots.n_open_markets()} | tranche de rotation (marche ouvert) : {len(scheduled)} | "
          f"jamais fetches, marche ouvert : {len(never_fetched)}/{len(never_fetched_all)} | "
          f"rattrapage schema, marche ouvert : {len(premigration)}/{len(premigration_all)} | "
          f"a traiter ce run : {len(todo)}")

    if not todo:
        print("Rien a faire -- cache deja a jour pour tout l'univers cible.")
        return

    new_rows = []
    for i, ticker in enumerate(todo):
        row = fetch_one(ticker)
        new_rows.append(row)
        if row.get("error") == "RATE_LIMITED":
            print(f"Rate-limit persistant apres {i} tickers -- arret propre du run, "
                  f"sauvegarde du progres, reessayer plus tard.", file=sys.stderr)
            break
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(todo)} traites...", file=sys.stderr, flush=True)
        if (i + 1) % COOLDOWN_EVERY == 0:
            print(f"  pause preventive de {COOLDOWN_SECONDS}s apres {i + 1} tickers...", file=sys.stderr)
            time.sleep(COOLDOWN_SECONDS)
        else:
            time.sleep(DELAY_BETWEEN_CALLS)

    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([cache[~cache["ticker"].isin(new_df["ticker"])], new_df], ignore_index=True)
    # normalize to one consistent string format on write, so old (datetime64) and new
    # (plain string) values can never again mix into the format-inference bug above.
    combined["fetched_at"] = pd.to_datetime(combined["fetched_at"], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    combined.to_csv(CACHE_PATH, index=False)

    ok = new_df["error"].isna().sum() if "error" in new_df.columns else len(new_df)
    coverage = len(cached_tickers | set(new_df["ticker"]))
    remaining_premigration = max(0, len(premigration_all) - len(new_df))  # total backlog, not
    # just the market-open subset attempted this run -- the closed-market rest still counts
    print(f"\n{ok}/{len(new_df)} succes ce run. Couverture cache : {coverage}/{len(universe)} "
          f"({coverage / len(universe):.0%})."
          + (f" Rattrapage schema restant : ~{remaining_premigration}." if remaining_premigration else ""))


if __name__ == "__main__":
    main()
