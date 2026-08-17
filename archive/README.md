# Archive

Code superseded by the live pipeline, kept for context rather than deleted outright --
these document earlier iterations of the screener and why they were replaced.

## screener/value_momentum_screener.py

The original screener: plain value (P/B percentile) + momentum (12-2) percentile,
averaged into a single score. No PEG, no quality multiplier, no sector-relative fair
value, no cross-listing dedup. Universe: S&P 500 + Euronext 100 only
(`data/universe/universe.csv`, built by `build_universe.py`).

Superseded by `screener/value_momentum_quality_screener_v2.py` (`compute_valuation`),
which the live hourly pipeline (`fetch_cache.py` -> `analyze_cache.py`) runs today --
sector-relative fair P/E adjusted for quality (ROE/margin/debt), a PEG filter, an
explicit debt/leverage gate, and (as of 2026-08-14) cross-listing dedup and a much
broader universe (US trending sectors + 15 international markets).

`portfolio/compare_portfolio.py` used to read this script's stale, hand-run output
(`screener_2026-08-07_full.csv`, never regenerated after that date) instead of the live
pipeline's `full_valuation_latest.csv` -- fixed 2026-08-15. If you're looking at this
file wondering why a comparison looks stale or uses a different model than expected,
that's the mistake this archive note is here to prevent repeating.

## screener/value_momentum_quality_screener_v2.py's own standalone run path

`compute_valuation()` and `fetch_one()` from this file are still actively imported by
`analyze_cache.py` and `portfolio/compare_portfolio.py` -- the file itself is NOT
archived. Only its own `main()` / one-shot full-universe fetch path is dead: it used
`data/universe/universe_v2.csv` (built by `build_universe_v2.py`, US NASDAQ+NYSE bulk +
Euronext 100, no progressive caching), which did the whole fetch in one run rather than
respecting Yahoo's rate limit across many small runs. Superseded by the
`build_trending_universe.py` + `fetch_cache.py` progressive-cache approach the live
pipeline uses instead.

## Stale dated snapshots (deleted, not archived)

`results/screener/screener_2026-08-07_full.csv`, `screener_2026-08-07_top30.csv`,
`screener_v2_2026-08-07_full.csv`, `screener_v2_2026-08-07_long_candidates.csv` -- one-off
outputs of the two paths above, dated 2026-08-07 and never regenerated. Deleted rather
than archived: they were data snapshots with no code/methodology value of their own, and
their presence risked being mistaken for current data (nearly happened once already,
see `portfolio/compare_portfolio.py`'s git history). Recoverable from git history if
ever needed.
