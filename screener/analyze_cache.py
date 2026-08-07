"""Step 3 of the pipeline: run the value+momentum+quality analysis against whatever is
currently in the fundamentals cache (built by fetch_cache.py). Unlike the cache fetch,
this does no network calls at all -- it's just pandas over a local CSV -- so it's safe to
run after every single cache update, even while the cache is still partially filled.
Results only get more complete (more tickers, better sector medians) as the cache grows.

Deliberately does NOT know about the user's actual portfolio: that data lives only in a
gitignored local file and never reaches this repo, so the SHORT-on-reversal logic (needs
current holdings) stays a separate, local-only script. This produces the universe-wide
LONG candidate list only.
"""
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.value_momentum_quality_screener_v2 import compute_valuation  # noqa: E402
from screener.build_trending_universe import JUNK_NAME_PATTERN  # noqa: E402

CACHE_PATH = HERE / "results/screener/fundamentals_cache.csv"
OUT_DIR = HERE / "results/screener"


def main():
    if not CACHE_PATH.exists():
        print("Pas encore de cache (fetch_cache.py n'a jamais tourne) -- rien a analyser.")
        return

    cache = pd.read_csv(CACHE_PATH)
    # defense in depth: some entries may predate the junk-name filter in
    # build_trending_universe.py (e.g. depositary/preferred shares fetched before that
    # filter existed) -- catch them here too regardless of how they got into the cache.
    cache = cache[~cache["name"].fillna("").str.contains(JUNK_NAME_PATTERN)]
    n_usable = cache["error"].isna().sum()
    if n_usable < 20:
        print(f"Cache encore trop petit ({n_usable} tickers exploitables) pour une analyse "
              f"fiable -- medianes sectorielles pas assez robustes. On attend que le cache grossisse.")
        return

    valued = compute_valuation(cache)
    valued.to_csv(OUT_DIR / "full_valuation_latest.csv", index=False)

    candidates = valued[valued["passes_filter"]]
    candidates.to_csv(OUT_DIR / "long_candidates_latest.csv", index=False)

    cols = ["ticker", "name", "sector", "price", "pe", "fair_pe", "fair_value", "valuation_gap",
            "roe", "debt_eq", "mom_12_2", "sector_momentum", "explication"]
    pd.set_option("display.width", 220)
    print(f"Analyse sur {n_usable} tickers en cache (sur un univers cible de tendance connu).")
    print(f"\n=== {len(candidates)} candidats LONG (tous les filtres OK : valeur, qualite, momentum relatif, PEG) ===")
    print(candidates.head(40)[cols].to_string(index=False, formatters={
        "price": "{:.2f}".format, "pe": "{:.1f}".format, "fair_pe": "{:.1f}".format,
        "fair_value": "{:.2f}".format, "valuation_gap": "{:+.1%}".format,
        "roe": "{:.1%}".format, "debt_eq": lambda x: f"{x:.0f}%" if pd.notna(x) else "n/a",
        "mom_12_2": "{:+.1%}".format, "sector_momentum": "{:+.1%}".format,
    }))


if __name__ == "__main__":
    main()
