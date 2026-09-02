"""Step 2 of the progressive-cache pipeline: fetch fundamentals+momentum for the
trending-sector universe (from build_trending_universe.py), respecting Yahoo Finance's
undocumented rate limit -- small batch, low concurrency, a delay between requests, and
exponential backoff that ABORTS the run early on repeated rate-limit errors rather than
digging the hole deeper. Results accumulate in a persistent CSV cache across runs.

REFRESH SCHEDULE (rewritten 2026-09-02): each hourly run refreshes a fixed ROTATING SLICE of
the universe -- every ticker is deterministically assigned to one of RUNS_PER_CYCLE slices
(a stable hash of the ticker, not its position in the list, so slice membership doesn't shift
just because build_trending_universe.py's output reorders or resizes day to day), and the
active slice each run is whatever wall-clock hour it is, mod RUNS_PER_CYCLE. Over one week
every ticker gets refreshed exactly once, evenly spread across all 168 hourly runs.

This replaced a pure staleness check (refresh only if last fetched >STALENESS_DAYS ago): that
scheme completes a full pass in ~13h (5571 tickers / 423 per run) and then goes completely
IDLE for the remaining ~6.5 days once nothing is stale -- so the whole universe's momentum
data updates in one clustered burst once a week rather than drifting continuously, which is
exactly what caused several stocks to flip momentum status simultaneously right after a
burst (flagged by the user 2026-09-02) instead of one at a time as their own data aged. Pure
rotation fixes that directly: ~33 tickers/run, every run, no idle stretch.

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
TRENDING_UNIVERSE = HERE / "data/universe/trending_universe.csv"
CACHE_PATH = HERE / "results/screener/fundamentals_cache.csv"

BATCH_SIZE = 423          # hard ceiling on requests/run regardless of schedule -- 70% of the
# 604 that worked in the original one-shot test, still a guess (Yahoo publishes no real
# limit). With rotation this is rarely reached (a slice is ~33 tickers) except while the
# LAST_SCHEMA_MIGRATION backlog below is draining.
DELAY_BETWEEN_CALLS = 0.4  # seconds, single-threaded on purpose
COOLDOWN_EVERY = 100      # community-reported pattern (yfinance GH discussion #2431):
COOLDOWN_SECONDS = 20     # ~100 requests before Yahoo wants a breather -- so take one voluntarily
MAX_RETRIES = 2

RUNS_PER_CYCLE = 24 * 7  # one slice per hourly cron tick ("35 * * * *" in
# update-screener.yml) -- 168 slices/week is what makes "every ticker refreshed once a week,
# spread evenly" concrete: universe_size / 168 tickers per run (~33 today).

LAST_SCHEMA_MIGRATION = "2026-09-02"  # date fetch_one() last gained new fields (country;
# the analyst-consensus fields landed one day earlier, 2026-09-01 -- this single cutoff
# covers both). A row fetched before this date predates those columns entirely and would
# otherwise wait for its normal weekly slot to backfill them -- rows below this cutoff are
# pulled into every run (on top of that run's regular slice) until they've all been
# refetched once, then this has no further effect (a refetched row's date moves past the
# cutoff, so it can never re-trigger).


def fetch_one(ticker: str) -> dict:
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    for attempt in range(MAX_RETRIES + 1):
        try:
            t = yf.Ticker(ticker)
            info = t.info
            hist = t.history(period="14mo", interval="1mo", auto_adjust=True)["Close"].dropna()
            mom_12_2 = hist.iloc[-2] / hist.iloc[-13] - 1 if len(hist) >= 13 else None
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


def _slice_of(ticker: str) -> int:
    """Deterministic, stable across processes/runs (unlike Python's built-in hash(), which is
    randomized per-process by default) -- a ticker's slice never moves just because
    build_trending_universe.py reordered or resized its output."""
    return int(hashlib.md5(ticker.encode("utf-8")).hexdigest(), 16) % RUNS_PER_CYCLE


def _current_slice() -> int:
    hours_since_epoch = int(pd.Timestamp.now(tz="UTC").timestamp() // 3600)
    return hours_since_epoch % RUNS_PER_CYCLE


def main():
    universe = pd.read_csv(TRENDING_UNIVERSE)
    universe = universe[universe["ticker"].notna()]  # build_trending_universe.py has been
    # seen to emit one stray all-blank row -- drop it here rather than let it flow into
    # fetch_one(nan) or a hash that can't .encode() a float.
    cache = load_cache()
    cache = cache[cache["ticker"].notna()]  # same defensive drop, cache side (a NaN ticker
    # ends up here too once fetch_one(nan) is called for one from the universe)

    cached_tickers = set(cache["ticker"]) if len(cache) else set()
    never_fetched = [t for t in universe["ticker"] if t not in cached_tickers]

    migration_cutoff = pd.Timestamp(LAST_SCHEMA_MIGRATION)
    premigration = (sorted(set(cache.loc[cache["fetched_at"] < migration_cutoff, "ticker"]))
                     if len(cache) else [])

    current_slice = _current_slice()
    scheduled = [t for t in universe["ticker"] if _slice_of(t) == current_slice]

    # priority order when BATCH_SIZE caps the total: brand-new tickers first (they might not
    # get a slot for days otherwise), then the one-time schema catch-up backlog, then this
    # run's regular weekly-rotation slice. dict.fromkeys dedupes while keeping first-seen order
    # (a ticker can legitimately appear in more than one of the three lists).
    todo = list(dict.fromkeys(never_fetched + premigration + scheduled))[:BATCH_SIZE]

    print(f"Univers cible : {len(universe)} tickers | tranche horaire #{current_slice}/{RUNS_PER_CYCLE} "
          f"({len(scheduled)} tickers) | jamais fetches : {len(never_fetched)} | "
          f"rattrapage schema restant : {len(premigration)} | a traiter ce run : {len(todo)}")

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
    remaining_premigration = max(0, len(premigration) - len(new_df))
    print(f"\n{ok}/{len(new_df)} succes ce run. Couverture cache : {coverage}/{len(universe)} "
          f"({coverage / len(universe):.0%})."
          + (f" Rattrapage schema restant : ~{remaining_premigration}." if remaining_premigration else ""))


if __name__ == "__main__":
    main()
