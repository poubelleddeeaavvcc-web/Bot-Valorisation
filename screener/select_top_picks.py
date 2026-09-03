"""Position selection for capital-constrained live trading: when only enough capital for
3-4 concurrent positions exists (vs. the paper simulation's blind "buy every candidate"
approach, deliberately kept simple to produce an honest, unbiased track record), rank
LONG candidates by a composite score and enforce sector diversification, rather than
buying everything or picking the single most extreme metric.

Composite score: equal-weighted average of percentile ranks across value (valuation_gap,
winsorized), momentum margin over sector, and quality -- combining independent ranks
rather than optimizing one axis, per Asness/Moskowitz/Pedersen "Value and Momentum
Everywhere" (Journal of Finance, 2013): value and momentum are negatively correlated, so
averaging ranks captures most of the diversification benefit without overfitting a
bespoke weighting to a handful of trades. Quality (Novy-Marx 2013; AQR "Quality Minus
Junk") is added as a third, distinct premium that guards against value traps.

Also includes a retroactive backtest: replay the blind bot's full trade history and
simulate what a capital-constrained (n positions, max_per_sector cap) version would have
picked and held, reusing each position's actual historical exit outcome (return_pct,
exit_date, exit_reason) since exit rules evaluate each position independently of
portfolio composition -- an exact replay, not an approximation, of "what if we'd only
been able to afford a handful of these."

IMPORTANT CAVEAT: as of 2026-08-17 the blind ledger has only 8 closed trades total; the
constrained subset will have even fewer. Any comparison below is a mechanism check (does
the selection/diversification logic behave sensibly), not a performance conclusion --
see the ~30-closed-trade checkpoint already agreed on before treating this as evidence.

Also caps North America (US + Canada) at NORTH_AMERICA_MAX_SHARE of the n slots: the raw
screener universe skews heavily US even after the international sourcing pass (see
build_trending_universe.py / build_international_universe.py), so an unconstrained
top-n-by-score pick tends to come back all-US -- 3 US + 1 CAN was the actual result that
prompted this (2026-08-20), and CAN is close enough to the US market (NAFTA trade ties,
BoC tracks the Fed) that it's not meaningfully different exposure. Relaxed the same way
the sector cap is: only if too few non-NA candidates clear the screener's filters to fill
the remaining slots otherwise -- per the user's standing direction, don't leave cash idle
or force a materially worse pick just to hit a diversification target.

Same treatment for STATE_LINKED_COUNTRIES (2026-09-02) -- a name whose HQ country is a
state known for large-scale industrial subsidy can pass every quantitative filter while
carrying a policy risk none of those ratios would catch; see STATE_LINKED_MAX_SHARE for
the reasoning.
"""
import math
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

CANDIDATES_PATH = HERE / "results/screener/long_candidates_latest.csv"
LEDGER_PATH = HERE / "results/simulation/portfolio_ledger.csv"

# cap valuation_gap before ranking: MAX_PLAUSIBLE_VALUATION_GAP=1.0 (100%) in
# value_momentum_quality_screener_v2.py already treats gaps beyond that as likely model
# breakdown rather than real opportunity; capping here at 75% keeps the ranking from
# being dominated by the most extreme (and least trustworthy) end of that range.
VALUATION_GAP_WINSOR_CAP = 0.75

# Upper bound on the North America (US + Canada) share of a selected portfolio -- see
# module docstring. Applied to whatever n is passed to select_diversified(), so it scales
# automatically as the constrained bot's slot count grows (e.g. n=4 -> max 3 NA, n=8 ->
# max 6 NA).
NORTH_AMERICA_MAX_SHARE = 0.75
CANADA_SUFFIX = ".TO"  # kept in sync with screener.simulate_constrained_portfolio.CANADA_SUFFIX

# Geopolitical/state-subsidy concentration cap (2026-09-02, per the user + a friend's
# feedback: BYD can look undervalued on ordinary ratios while that "value" is really the
# Chinese state flooding the market to win a monopoly -- if that policy reverses, the ratios
# never warned you). Unlike NORTH_AMERICA_MAX_SHARE (home-market bias, not a red flag), this
# isn't trying to score any single stock's subsidy exposure -- that's judged qualitatively
# per-candidate instead, by the Ollama news gate's state-dependency check in news_filter.py
# (Bot#4/5/6 only). This cap is the portfolio-level backstop: even a name that clears that
# check individually shouldn't let the WHOLE portfolio lean on one government's industrial
# policy. Applies across all bots that call select_diversified()/fill_slots() with sector/NA
# caps -- i.e. not Bot#1 (kept deliberately blind, see simulate_portfolio.py).
STATE_LINKED_COUNTRIES = {"China"}  # start narrow (the actual case raised); add countries
# here deliberately if another one becomes a live concern, not preemptively.
STATE_LINKED_MAX_SHARE = 0.30  # tighter than the 75% NA cap -- this is a single-policy-actor
# risk, not ordinary home-market concentration, so it should bind well before NA's does.


def ticker_region(ticker: str) -> str:
    """North America (no exchange suffix = US, or .TO = Canada) vs everything else --
    same suffix convention used for IBKR fractional-eligibility in
    simulate_constrained_portfolio.fractional_eligible()."""
    if "." not in ticker or ticker.endswith(CANADA_SUFFIX):
        return "North America"
    return "International"


def is_state_linked(country) -> bool:
    """See STATE_LINKED_COUNTRIES. Missing/unknown country fails open (not flagged) --
    same posture as the rest of this project's diversification/quality checks: absence of
    data isn't evidence of the risk."""
    return isinstance(country, str) and country in STATE_LINKED_COUNTRIES


def composite_score(df: pd.DataFrame) -> pd.Series:
    value_pct = df["valuation_gap"].clip(upper=VALUATION_GAP_WINSOR_CAP).rank(pct=True)
    # margin over sector, not just "positive" -- a thin margin is exactly what flipped
    # ALV/HMY to an exit within hours of entry (see 2026-08-15 finding); a wider margin
    # should be more resilient to that kind of noise.
    momentum_margin = df["mom_12_2"] - df["sector_momentum"]
    momentum_pct = momentum_margin.rank(pct=True)
    quality_pct = df["quality_multiplier"].rank(pct=True)
    return (value_pct + momentum_pct + quality_pct) / 3


def select_diversified(df: pd.DataFrame, n: int, max_per_sector: int) -> pd.DataFrame:
    """Greedy pick by score, capping how many can come from one sector -- relaxed one
    step at a time only if too few sectors are represented to fill n slots otherwise.
    Also caps North America at NORTH_AMERICA_MAX_SHARE and state-linked countries (see
    STATE_LINKED_COUNTRIES) at STATE_LINKED_MAX_SHARE of n, both tried first with the
    sector cap fully respected-and-relaxed; only if that combination still can't fill n
    slots are the NA/state-linked caps themselves dropped together, on the same "don't
    leave a slot empty over a diversification target" logic as the sector cap relaxation."""
    ranked = df.sort_values("score", ascending=False)
    max_na = math.floor(n * NORTH_AMERICA_MAX_SHARE)
    max_state = math.floor(n * STATE_LINKED_MAX_SHARE)
    id_col = "row_id" if "row_id" in df.columns else "ticker"
    has_country = "country" in df.columns

    picked_ids = []
    for enforce_geo_caps in (True, False):
        for cap in range(max_per_sector, n + 1):
            picked_ids, sector_counts, na_count, state_count = [], {}, 0, 0
            for _, row in ranked.iterrows():
                if len(picked_ids) >= n:
                    break
                rid = row.get("row_id", row["ticker"])
                if rid in picked_ids:
                    continue
                if sector_counts.get(row["sector"], 0) >= cap:
                    continue
                is_na = ticker_region(row["ticker"]) == "North America"
                is_state = has_country and is_state_linked(row.get("country"))
                if enforce_geo_caps and is_na and na_count >= max_na:
                    continue
                if enforce_geo_caps and is_state and state_count >= max_state:
                    continue
                picked_ids.append(rid)
                sector_counts[row["sector"]] = sector_counts.get(row["sector"], 0) + 1
                if is_na:
                    na_count += 1
                if is_state:
                    state_count += 1
            if len(picked_ids) >= n:
                break
        if len(picked_ids) >= n:
            break

    return df[df[id_col].isin(picked_ids)].sort_values("score", ascending=False)


def today_top_picks(n=4, max_per_sector=1) -> pd.DataFrame:
    candidates = pd.read_csv(CANDIDATES_PATH)
    candidates["score"] = composite_score(candidates)
    return select_diversified(candidates, n, max_per_sector)


def _rename_entry_cols(ledger: pd.DataFrame) -> pd.DataFrame:
    return ledger.rename(columns={
        "entry_valuation_gap": "valuation_gap", "entry_mom_12_2": "mom_12_2",
        "entry_sector_momentum": "sector_momentum", "entry_quality_multiplier": "quality_multiplier",
    })


def backtest_capital_constrained(n=4, max_per_sector=1):
    ledger = pd.read_csv(LEDGER_PATH).reset_index().rename(columns={"index": "row_id"})
    ledger["score"] = composite_score(_rename_entry_cols(ledger))
    ledger["entry_date"] = pd.to_datetime(ledger["entry_date"])
    ledger["exit_date"] = pd.to_datetime(ledger["exit_date"])

    all_dates = sorted(set(ledger["entry_date"]) | set(ledger["exit_date"].dropna()))

    pool, held, sector_counts = {}, {}, {}
    constrained_closed = []

    for date in all_dates:
        for _, row in ledger[ledger["exit_date"] == date].iterrows():
            rid = row["row_id"]
            pool.pop(rid, None)
            if rid in held:
                sector_counts[row["sector"]] -= 1
                constrained_closed.append(row)
                del held[rid]

        for _, row in ledger[ledger["entry_date"] == date].iterrows():
            pool[row["row_id"]] = row

        free_slots = n - len(held)
        if free_slots > 0 and pool:
            ranked = pd.DataFrame(pool.values()).sort_values("score", ascending=False)
            for cap in range(max_per_sector, n + 1):
                if free_slots <= 0:
                    break
                for _, row in ranked.iterrows():
                    if free_slots <= 0:
                        break
                    rid = row["row_id"]
                    if rid not in pool:
                        continue
                    if sector_counts.get(row["sector"], 0) >= cap:
                        continue
                    held[rid] = row
                    sector_counts[row["sector"]] = sector_counts.get(row["sector"], 0) + 1
                    del pool[rid]
                    free_slots -= 1

    return pd.DataFrame(constrained_closed), pd.DataFrame(held.values()), ledger


def _with_quality_perspective_notes(picks: pd.DataFrame) -> pd.DataFrame:
    """Left-joins the 2 read-only notes (see quality_perspective_notes.py) onto the picks
    table purely for display -- doesn't touch score/ranking/selection above. Missing file or
    ticker just means no notes columns get added, never an error: this module must never be
    a hard dependency for the actual buy pipeline."""
    notes_path = HERE / "results/screener/quality_perspective_notes.csv"
    if not notes_path.exists():
        return picks
    notes = pd.read_csv(notes_path)[["ticker", "note_qualite_20", "note_qualite_low_confidence",
                                      "note_perspective_20", "note_perspective_low_confidence"]]
    return picks.merge(notes, on="ticker", how="left")


def main():
    print(f"=== Top picks aujourd'hui (n=4, max 1/secteur) ===")
    picks = today_top_picks(n=4, max_per_sector=1)
    picks = _with_quality_perspective_notes(picks)
    cols = ["ticker", "name", "sector", "pe", "peg", "mom_12_2", "valuation_gap", "quality_multiplier", "score"]
    formatters = {
        "pe": "{:.1f}".format, "peg": lambda x: f"{x:.2f}" if pd.notna(x) else "n/a",
        "mom_12_2": "{:+.1%}".format, "valuation_gap": "{:+.1%}".format,
        "quality_multiplier": "{:.2f}".format, "score": "{:.2f}".format,
    }
    if "note_qualite_20" in picks.columns:
        cols += ["note_qualite_20", "note_perspective_20"]
        # "*" marks a note built on too few pillars to trust at face value (see
        # note_qualite_low_confidence/note_perspective_low_confidence in quality_perspective_notes.py)
        picks["note_qualite_20"] = picks.apply(
            lambda r: (f"{r['note_qualite_20']:.1f}*" if r.get("note_qualite_low_confidence")
                       else f"{r['note_qualite_20']:.1f}") if pd.notna(r["note_qualite_20"]) else "n/a", axis=1)
        picks["note_perspective_20"] = picks.apply(
            lambda r: (f"{r['note_perspective_20']:.1f}*" if r.get("note_perspective_low_confidence")
                       else f"{r['note_perspective_20']:.1f}") if pd.notna(r["note_perspective_20"]) else "n/a", axis=1)
    print(picks[cols].to_string(index=False, formatters=formatters))

    print(f"\n=== Backtest retroactif : portefeuille a 4 positions max vs achat en aveugle ===")
    print("ATTENTION : echantillon minuscule (8 clotures dans le ledger complet a ce jour) -- "
          "verification du mecanisme, pas une conclusion de performance.\n")
    constrained_closed, constrained_open, full_ledger = backtest_capital_constrained(n=4, max_per_sector=1)

    blind_closed = full_ledger[full_ledger["status"] == "closed"]
    print(f"Achat en aveugle (tout) : {len(blind_closed)} clotures | "
          f"win rate {  (blind_closed['return_pct']>0).mean():.0%} | "
          f"retour moyen {blind_closed['return_pct'].mean():+.2%}")

    if len(constrained_closed):
        print(f"Portefeuille contraint (n=4) : {len(constrained_closed)} clotures | "
              f"win rate {(constrained_closed['return_pct']>0).mean():.0%} | "
              f"retour moyen {constrained_closed['return_pct'].mean():+.2%}")
        print(constrained_closed[["ticker", "sector", "entry_date", "exit_date", "exit_reason", "return_pct"]]
              .to_string(index=False, formatters={"return_pct": "{:+.2%}".format}))
    else:
        print("Portefeuille contraint (n=4) : aucune cloture pour l'instant.")

    if len(constrained_open):
        print(f"\nPositions actuellement tenues par le portefeuille contraint : {len(constrained_open)}")
        print(constrained_open[["ticker", "sector", "entry_date", "unrealized_return_pct"]]
              .to_string(index=False, formatters={"unrealized_return_pct": "{:+.2%}".format}))


if __name__ == "__main__":
    main()
