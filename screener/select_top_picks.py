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


def ticker_region(ticker: str) -> str:
    """North America (no exchange suffix = US, or .TO = Canada) vs everything else --
    same suffix convention used for IBKR fractional-eligibility in
    simulate_constrained_portfolio.fractional_eligible()."""
    if "." not in ticker or ticker.endswith(CANADA_SUFFIX):
        return "North America"
    return "International"


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
    Also caps North America at NORTH_AMERICA_MAX_SHARE of n (see module docstring),
    tried first with the sector cap fully respected-and-relaxed; only if that combination
    still can't fill n slots is the NA cap itself dropped, on the same "don't leave a
    slot empty over a diversification target" logic as the sector cap relaxation."""
    ranked = df.sort_values("score", ascending=False)
    max_na = math.floor(n * NORTH_AMERICA_MAX_SHARE)
    id_col = "row_id" if "row_id" in df.columns else "ticker"

    picked_ids = []
    for enforce_na_cap in (True, False):
        for cap in range(max_per_sector, n + 1):
            picked_ids, sector_counts, na_count = [], {}, 0
            for _, row in ranked.iterrows():
                if len(picked_ids) >= n:
                    break
                rid = row.get("row_id", row["ticker"])
                if rid in picked_ids:
                    continue
                if sector_counts.get(row["sector"], 0) >= cap:
                    continue
                is_na = ticker_region(row["ticker"]) == "North America"
                if enforce_na_cap and is_na and na_count >= max_na:
                    continue
                picked_ids.append(rid)
                sector_counts[row["sector"]] = sector_counts.get(row["sector"], 0) + 1
                if is_na:
                    na_count += 1
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


def main():
    print(f"=== Top picks aujourd'hui (n=4, max 1/secteur) ===")
    picks = today_top_picks(n=4, max_per_sector=1)
    cols = ["ticker", "name", "sector", "pe", "peg", "mom_12_2", "valuation_gap", "quality_multiplier", "score"]
    print(picks[cols].to_string(index=False, formatters={
        "pe": "{:.1f}".format, "peg": lambda x: f"{x:.2f}" if pd.notna(x) else "n/a",
        "mom_12_2": "{:+.1%}".format, "valuation_gap": "{:+.1%}".format,
        "quality_multiplier": "{:.2f}".format, "score": "{:.2f}".format,
    }))

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
