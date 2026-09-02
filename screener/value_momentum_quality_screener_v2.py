"""v2 screener: broad US+Europe universe (no index restriction), sector-relative fair P/E
adjusted for quality (ROE, margin) and leverage, momentum, and a debt filter with
exceptions for structurally high-debt sectors.

Fair value formula (documented here because it is otherwise a black box):
  fair_pe   = sector_median_pe * quality_multiplier
  quality_multiplier = clip(1
                              + 0.5 * (roe_percentile_in_sector - 0.5)
                              + 0.25 * (margin_percentile_in_sector - 0.5)
                              - 0.25 * (debt_percentile_in_sector - 0.5),
                             0.6, 1.6)
  fair_value = trailing_eps * fair_pe
  valuation_gap = fair_value / price - 1        (>0 undervalued, <0 overvalued)

This replaces the hand-picked "fair P/E per stock" the user's own portfolio tracker used
(reverse-engineered from its valeurJuste field) with something that scales to thousands
of tickers without a person assigning a multiple to each one by hand.
"""
import concurrent.futures as cf
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
UNIVERSE_CSV = HERE / "data/universe/universe_v2.csv"
OUT_DIR = HERE / "results/screener"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MARKET_CAP_FLOOR = 1_000_000_000  # $1B -- liquidity floor, an automated bot has no business in micro-caps
MAX_WORKERS = 16
HIGH_DEBT_SECTORS = {"Utilities", "Real Estate", "Communication Services"}
DEBT_TO_EQUITY_CAP = 150.0  # percent
EXTREME_DEBT_TO_EQUITY_CAP = 400.0  # percent -- even the high-debt-sector exception has a limit;
# beyond this it's a mortgage REIT / leverage-as-the-business-model vehicle, not
# "normal" sector leverage, and the P/E-based fair value model doesn't apply to it.
PEG_CAP = 2.0  # Peter Lynch's classic threshold: PEG<1 excellent, <2 still reasonable,
# beyond that the P/E isn't justified by earnings growth even if it looks cheap on paper.
MAX_PLAUSIBLE_VALUATION_GAP = 0.75  # was 1.0 (+100%), lowered 2026-08-18: even after the
# industry-peer-group and cyclical P/B fixes above, 23% of candidates still showed a gap
# > 75% (13% > 90%) on real cache data -- not just a mining-specific artifact, so tightened
# here too rather than trusting the model past this point (wrong peer group, distressed
# name, depositary/preferred slipping through, stale/bad EPS). A stock trading at a
# fraction of "fair value" by 4x+ is a red flag, not alpha.
MIN_VALUATION_GAP = 0.25  # margin of safety, raised from 0.15 -- fewer, more convicted picks
MIN_ROE = 0.15  # absolute quality floor regardless of sector (Buffett-style baseline),
# on top of the existing relative-to-sector quality_multiplier

MIN_INDUSTRY_PEERS = 5  # below this, an industry (e.g. "Gold") median P/E is too noisy to
# trust over the broader sector ("Basic Materials") median -- falls back to sector in that case

MIN_ANALYST_COVERAGE = 3  # below this, "recommendationKey" is one or two opinions dressed up
# as a consensus -- too noisy to act on, so the analyst-recommendation check below is skipped
# entirely rather than excluding a candidate on thin data
BEARISH_RECOMMENDATIONS = {"sell", "strong_sell", "underperform"}  # third-party analyst
# consensus explicitly bearish despite our own model calling it undervalued -- treated as a
# corroboration failure (see compute_valuation): analysts may be pricing in something (pending
# litigation, accounting concerns, sector headwind) our purely quantitative screen can't see.
# Missing/thin coverage is NOT penalized (fail open, same posture as the rest of this module).
CYCLICAL_SECTORS = {"Basic Materials", "Energy"}  # trailing EPS is unreliable here: earnings
# swing hard with the commodity cycle, so trough-of-cycle EPS can make the P/E look
# artificially cheap and inflate valuation_gap without the company actually being
# undervalued (flagged 2026-08-18 on gold miners AU/HMY/DPM.TO, all ~97-98% "undervalued").
# P/B doesn't swing with the earnings cycle, so it's used as a corroborating check here.

# Same company, cross-listed on two exchanges under two different tickers (and, critically,
# two different name strings -- "Equinor ASA" vs "EQUINOR" -- so the name-based dedup below
# doesn't catch them). Found by hand after the ledger opened both EQNR and EQNR.OL, and both
# LOGI and LOGN.SW, as if they were independent bets on two unrelated companies. Deliberately
# a manual, curated list rather than fuzzy name-matching -- this universe has plenty of
# distinct companies that share a name prefix (three separate "Grupo Aeroportuario..." airport
# operators, several unrelated "First Merchants"/"First Bancorp"-style banks), where an
# automated match would silently merge companies that have nothing to do with each other.
CROSS_LISTING_GROUPS = [
    {"ALC", "ALC.SW"},        # Alcon
    {"AMRZ", "AMRZ.SW"},      # Amrize
    {"BTI", "BATS.L"},        # British American Tobacco
    {"CCEP", "CCEP.L"},       # Coca-Cola Europacific Partners
    {"DB", "DBK.DE"},         # Deutsche Bank
    {"EQNR", "EQNR.OL"},      # Equinor
    {"HSBC", "0005.HK"},      # HSBC
    {"LOGI", "LOGN.SW"},      # Logitech
    {"NVS", "NOVN.SW"},       # Novartis
    {"SNY", "SAN.PA"},        # Sanofi
    {"SNN", "SN.L"},          # Smith & Nephew
    {"TTE", "TTE.PA"},        # TotalEnergies
    {"TSM", "2330.TW"},       # Taiwan Semiconductor Manufacturing -- found 2026-08-18 when
    # both showed up as separate LONG candidates
]
CROSS_LISTING_KEY = {tk: f"xlisting:{i}" for i, group in enumerate(CROSS_LISTING_GROUPS) for tk in group}


def fetch_one(ticker: str) -> dict:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        cap = info.get("marketCap")
        if not cap or cap < MARKET_CAP_FLOOR:
            return {"ticker": ticker, "error": "cap_trop_petite"}

        hist = t.history(period="14mo", interval="1mo", auto_adjust=True)["Close"].dropna()
        if len(hist) < 13:
            return {"ticker": ticker, "error": "historique_insuffisant"}
        mom_12_2 = hist.iloc[-2] / hist.iloc[-13] - 1

        return {
            "ticker": ticker, "price": hist.iloc[-1], "market_cap": cap,
            "sector": info.get("sector"), "industry": info.get("industry"),
            "country": info.get("country"),  # see fetch_cache.fetch_one for why
            # longName over shortName: shortName gets hard-truncated by Yahoo (e.g. "First
            # Merchants Corporation - D...") which can cut off the very word (Depositary,
            # Preferred...) that JUNK_NAME_PATTERN needs to see to filter the security out.
            "name": info.get("longName") or info.get("shortName"),
            "pe": info.get("trailingPE"), "pb": info.get("priceToBook"),
            "eps": info.get("trailingEps"), "roe": info.get("returnOnEquity"),
            "margin": info.get("profitMargins"), "debt_eq": info.get("debtToEquity"),
            "target_low_price": info.get("targetLowPrice"), "target_mean_price": info.get("targetMeanPrice"),
            "target_high_price": info.get("targetHighPrice"), "recommendation_key": info.get("recommendationKey"),
            "num_analyst_opinions": info.get("numberOfAnalystOpinions"),
            "mom_12_2": mom_12_2,
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)[:80]}


def fetch_all(tickers: list) -> pd.DataFrame:
    rows = []
    with cf.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_one, tk): tk for tk in tickers}
        done = 0
        for fut in cf.as_completed(futures):
            rows.append(fut.result())
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tickers)} tickers traites...", file=sys.stderr, flush=True)
    return pd.DataFrame(rows)


def pct_rank_within_sector(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("sector")[col].rank(pct=True)


def _explain_row(r) -> str:
    """Plain-language reasons the model flags this valuation_gap -- mechanical (derived
    from the inputs already computed), not a claim about market psychology or news we
    don't have data for."""
    parts = []
    if pd.notna(r["pe"]) and pd.notna(r["peer_median_pe"]) and r["peer_median_pe"] > 0 and r["pe"] / r["peer_median_pe"] < 0.85:
        peer_label = "l'industrie" if r.get("industry_count", 0) >= MIN_INDUSTRY_PEERS else "du secteur"
        parts.append(f"P/E de {r['pe']:.1f} nettement sous la mediane de {peer_label} ({r['peer_median_pe']:.1f})")
    if r["roe_pct"] >= 0.75:
        parts.append(f"ROE ({r['roe']*100:.0f}%) dans le haut du secteur")
    if r["margin_pct"] >= 0.75:
        parts.append("marge superieure a la mediane du secteur")
    if r["debt_pct"] <= 0.35:
        parts.append("dette plus faible que la mediane du secteur")
    if pd.notna(r["mom_12_2"]) and pd.notna(r["sector_momentum"]) and r["mom_12_2"] > r["sector_momentum"]:
        parts.append(f"momentum ({r['mom_12_2']*100:+.0f}%) superieur au secteur ({r['sector_momentum']*100:+.0f}%)")
    if pd.notna(r.get("peg")) and r["peg"] < 1:
        parts.append(f"PEG bas ({r['peg']:.2f}) : croissance qui justifie plus que le P/E actuel")
    if (pd.notna(r.get("num_analyst_opinions")) and r["num_analyst_opinions"] >= MIN_ANALYST_COVERAGE
            and pd.notna(r.get("analyst_gap")) and r["analyst_gap"] > 0):
        parts.append(f"cible analystes ({r['num_analyst_opinions']:.0f} analystes, "
                      f"moyenne {r['target_mean_price']:.2f}) {r['analyst_gap']*100:+.0f}% au-dessus du prix actuel")
    if not parts:
        parts.append("ecart porte principalement par le multiple median du secteur")
    return " ; ".join(parts[:3])


def compute_valuation(df: pd.DataFrame) -> pd.DataFrame:
    # defensive: analyst fields are only populated by fetch_cache.py / this module's own
    # fetch_one (added 2026-09-01) -- older cached rows or an unrelated caller's raw df may
    # not have them yet, so backfill as missing rather than KeyError downstream.
    for col in ("target_low_price", "target_mean_price", "target_high_price",
                "recommendation_key", "num_analyst_opinions"):
        if col not in df.columns:
            df[col] = np.nan
    if "country" not in df.columns:  # added 2026-09-02, same backfill reasoning
        df["country"] = np.nan

    df = df[df["error"].isna()].copy()
    df = df[(df["pe"] > 0) & (df["pe"] < 80) & (df["eps"] > 0) & (df["pb"] > 0)]
    df = df.dropna(subset=["sector", "mom_12_2", "roe"])

    # dual/multi share classes (GOOG/GOOGL, BRK.A/BRK.B, FOXA/FOX...) are the same
    # company, not two independent opportunities -- keep only the larger-cap class per
    # name so they don't inflate the candidate count or double up in the simulation.
    # Cross-listed tickers (same company, different exchange AND different name string,
    # e.g. Equinor/EQNR.OL) dedupe on CROSS_LISTING_KEY instead since "name" alone misses them.
    df = df.sort_values("market_cap", ascending=False)
    dedupe_key = df["ticker"].map(CROSS_LISTING_KEY).fillna(df["name"])
    df = df[~dedupe_key.duplicated(keep="first")]

    sector_pe = df.groupby("sector")["pe"].median().rename("sector_median_pe")
    df = df.join(sector_pe, on="sector")

    # tighter peer group when there's enough of it (e.g. "Gold" miners specifically, not all
    # of "Basic Materials" lumped in with chemical makers) -- falls back to the coarser
    # sector median when the industry bucket is too thin to trust (MIN_INDUSTRY_PEERS)
    industry_pe = df.groupby("industry")["pe"].median().rename("industry_median_pe")
    df = df.join(industry_pe, on="industry")
    df["industry_count"] = df.groupby("industry")["pe"].transform("size")
    industry_reliable = (df["industry"].notna() & (df["industry_count"] >= MIN_INDUSTRY_PEERS)
                          & df["industry_median_pe"].notna())
    df["peer_median_pe"] = np.where(industry_reliable, df["industry_median_pe"], df["sector_median_pe"])

    df["roe_pct"] = pct_rank_within_sector(df, "roe")
    df["margin_pct"] = pct_rank_within_sector(df.fillna({"margin": df["margin"].median()}), "margin")
    df["debt_pct"] = pct_rank_within_sector(df.fillna({"debt_eq": df["debt_eq"].median()}), "debt_eq")

    quality_multiplier = (1
                           + 0.5 * (df["roe_pct"] - 0.5)
                           + 0.25 * (df["margin_pct"] - 0.5)
                           - 0.25 * (df["debt_pct"] - 0.5))
    df["quality_multiplier"] = quality_multiplier.clip(0.6, 1.6)
    df["fair_pe"] = df["peer_median_pe"] * df["quality_multiplier"]
    df["fair_value"] = df["eps"] * df["fair_pe"]
    df["valuation_gap"] = df["fair_value"] / df["price"] - 1

    sector_momentum = df.groupby("sector")["mom_12_2"].median().rename("sector_momentum")
    df = df.join(sector_momentum, on="sector")
    sector_pb = df.groupby("sector")["pb"].median().rename("sector_median_pb")
    df = df.join(sector_pb, on="sector")
    # Analyst-implied upside, same shape as our own valuation_gap -- a second, independent
    # (third-party) opinion on the same question, not a replacement for it.
    df["analyst_gap"] = df["target_mean_price"] / df["price"] - 1
    df["explication"] = df.apply(_explain_row, axis=1)

    normal_debt_ok = df["debt_eq"] < DEBT_TO_EQUITY_CAP
    high_debt_sector_ok = df["sector"].isin(HIGH_DEBT_SECTORS) & (df["debt_eq"] < EXTREME_DEBT_TO_EQUITY_CAP)
    debt_ok = df["debt_eq"].isna() | normal_debt_ok | high_debt_sector_ok
    sector_up = df["sector_momentum"] > 0
    plausible = df["valuation_gap"] <= MAX_PLAUSIBLE_VALUATION_GAP
    # Relative strength: the stock must be beating its own sector, i.e. actually a leader
    # rather than passively drifting up with the tide. No separate "mom_12_2 > 0" filter is
    # needed alongside this: combined with sector_up (sector_momentum > 0) just above, this
    # already guarantees mom_12_2 > sector_momentum > 0 -- verified 2026-08-18 that a
    # standalone stock_momentum_ok filter excluded zero rows beyond what these two already cut.
    relative_momentum_ok = df["mom_12_2"] > df["sector_momentum"]
    # PEG: is the P/E justified by actual earnings growth, or just "statistically cheap"
    # on a metric that ignores growth entirely? Missing PEG (delisted-adjacent, unusual
    # capital structure, data gap) is treated as a fail, not a pass -- tightens the
    # candidate list today and self-heals as the cache fills in with the new field.
    peg_ok = df["peg"].notna() & (df["peg"] > 0) & (df["peg"] < PEG_CAP)
    margin_of_safety_ok = df["valuation_gap"] >= MIN_VALUATION_GAP
    quality_floor_ok = df["roe"] >= MIN_ROE
    # cyclical earnings can make trailing P/E look artificially cheap at the bottom of the
    # cycle (see CYCLICAL_SECTORS above) -- P/B must also confirm cheapness-vs-sector before
    # trusting a big valuation_gap coming from one of these sectors
    cyclical_pb_confirms = ~df["sector"].isin(CYCLICAL_SECTORS) | (df["pb"] <= df["sector_median_pb"])
    # quality_multiplier should amplify a real discount, not manufacture one: without this,
    # a stock already at/above its peer group's P/E can still pass on quality_multiplier alone
    # (found 2026-08-18 on NVDA: P/E 34.4 vs Semiconductors peer median 31.6, still a 32%
    # "valuation_gap" from quality_multiplier=1.44 alone). Checked before adding: only 4/47
    # current candidates trade above their peer median P/E, so this isn't overly restrictive.
    pe_below_peer_ok = df["pe"] <= df["peer_median_pe"]
    # Corroboration, not a second independent hurdle: only excludes when analysts explicitly
    # disagree (bearish consensus) with decent coverage to trust it -- see BEARISH_RECOMMENDATIONS.
    # Missing/thin coverage passes (fail open), same posture as peg_ok's missing-PEG handling
    # is the opposite (fail closed) -- deliberate: PEG is OUR data pipeline, always computable
    # from fields we already require; a missing analyst consensus just means thin coverage,
    # which says nothing about the stock either way.
    analyst_recommendation_ok = (df["recommendation_key"].isna()
                                  | (df["num_analyst_opinions"].fillna(0) < MIN_ANALYST_COVERAGE)
                                  | ~df["recommendation_key"].isin(BEARISH_RECOMMENDATIONS))
    df["passes_filter"] = (debt_ok & sector_up & plausible
                            & relative_momentum_ok & peg_ok & margin_of_safety_ok & quality_floor_ok
                            & cyclical_pb_confirms & pe_below_peer_ok & analyst_recommendation_ok)
    return df.sort_values("valuation_gap", ascending=False)


def main():
    universe = pd.read_csv(UNIVERSE_CSV)
    tickers = universe["ticker"].dropna().unique().tolist()
    print(f"Recuperation des donnees pour {len(tickers)} tickers (univers elargi, ca va prendre plusieurs minutes)...")

    t0 = time.time()
    raw = fetch_all(tickers)
    print(f"Termine en {time.time() - t0:.0f}s")
    if "error" not in raw.columns:
        raw["error"] = np.nan
    print(raw["error"].value_counts(dropna=False).head(10))

    valued = compute_valuation(raw)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    valued.to_csv(OUT_DIR / f"screener_v2_{today}_full.csv", index=False)

    candidates = valued[valued["passes_filter"] & (valued["valuation_gap"] > 0.15)]
    candidates.to_csv(OUT_DIR / f"screener_v2_{today}_long_candidates.csv", index=False)

    cols = ["ticker", "name", "sector", "price", "pe", "fair_pe", "fair_value", "valuation_gap",
            "roe", "debt_eq", "mom_12_2", "sector_momentum"]
    pd.set_option("display.width", 180)
    print(f"\n=== {len(candidates)} candidats LONG (sous-evalues de +15% ou plus, filtres dette+secteur OK) ===")
    print(candidates.head(40)[cols].to_string(index=False, formatters={
        "price": "{:.2f}".format, "pe": "{:.1f}".format, "fair_pe": "{:.1f}".format,
        "fair_value": "{:.2f}".format, "valuation_gap": "{:+.1%}".format,
        "roe": "{:.1%}".format, "debt_eq": lambda x: f"{x:.0f}%" if pd.notna(x) else "n/a",
        "mom_12_2": "{:+.1%}".format, "sector_momentum": "{:+.1%}".format,
    }))


if __name__ == "__main__":
    main()
