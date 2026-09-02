"""Step 2 of the progressive-cache pipeline: fetch fundamentals+momentum for the
trending-sector universe (from build_trending_universe.py), respecting Yahoo Finance's
undocumented rate limit -- small batch, low concurrency, a delay between requests, and
exponential backoff that ABORTS the run early on repeated rate-limit errors rather than
digging the hole deeper. Results accumulate in a persistent CSV cache across runs, so
re-running this daily (e.g. via a scheduled task) gradually completes the full universe
without ever tripping the block that happened when we tried to do it all in one go.
"""
import pathlib
import sys
import time

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
TRENDING_UNIVERSE = HERE / "data/universe/trending_universe.csv"
CACHE_PATH = HERE / "results/screener/fundamentals_cache.csv"

BATCH_SIZE = 423          # 70% of the 604 that worked in the original one-shot test -- still a
# guess (Yahoo publishes no real limit), just a less conservative one than the initial 25%
DELAY_BETWEEN_CALLS = 0.4  # seconds, single-threaded on purpose
COOLDOWN_EVERY = 100      # community-reported pattern (yfinance GH discussion #2431):
COOLDOWN_SECONDS = 20     # ~100 requests before Yahoo wants a breather -- so take one voluntarily
MAX_RETRIES = 2
STALENESS_DAYS = 7        # short enough to keep momentum/price (the whole point of the screener)
# from going stale for months once the cache reaches full coverage of the target universe --
# slower-moving fundamentals (P/E, ROE, debt) get refreshed more often than strictly needed as a
# side effect, but that's a cheap trade given the batch (423/run, hourly) clears the full ~3000-
# ticker universe in well under a day once it goes stale. Was 90 days; that let the cache freeze
# solid the moment it reached 100% coverage (see incident 2026-08-07 -> 2026-08-14, zero refresh).


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


def main():
    universe = pd.read_csv(TRENDING_UNIVERSE)
    cache = load_cache()

    fresh_cutoff = pd.Timestamp.today() - pd.Timedelta(days=STALENESS_DAYS)
    if len(cache):
        already_fresh = set(cache.loc[cache["fetched_at"] >= fresh_cutoff, "ticker"])
    else:
        already_fresh = set()

    todo = [t for t in universe["ticker"] if t not in already_fresh][:BATCH_SIZE]
    print(f"Univers cible : {len(universe)} tickers | deja en cache (frais) : {len(already_fresh)} | "
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
    total_fresh = len(already_fresh) + ok
    print(f"\n{ok}/{len(new_df)} succes ce run. Cache total frais : {total_fresh}/{len(universe)} "
          f"({total_fresh / len(universe):.0%}). Relancer ce script pour continuer.")


if __name__ == "__main__":
    main()
