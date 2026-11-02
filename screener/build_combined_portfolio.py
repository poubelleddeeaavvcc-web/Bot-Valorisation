"""Portefeuille combine (2026-10-12): the user's "LE modele" -- a virtual mix of independent strategies
weighted by RISK, tracked live from the bots' own equity curves (no orders of its own, no fees beyond the
bots'). Goal stated by the user: raise profit/risk, lower risk is welcome.

Components (each normalised to 1 at the mix's start or last rebalance):
  - Sous-evaluation: equal mix of the three base capital bots of the screener strategy, B1 #2
    (constrained_*), C1 #3 (large_*) and D1 #25 (delta_*) -- the family, not a hand-picked variant;
  - R1 #44 retour a la moyenne (rsi2_*);
  - T1 #43 tendance (trend_*);
  - I1 #45 inities petites capitalisations (insider_*), added 2026-10-13 -- in the 2008-2026 backtest it
    lifted the mix's Sharpe from 0.66 to 0.79 (its excess return runs -0.31 against the value proxy).
  - G1 #46 commandes publiques (govcontracts_*) and W1 #47 attention Wikipedia (wikiattention_*), added
    2026-10-14 at the user's request. Backtest 2015-10 -> 2026-09 (the window where W1's page views exist,
    scratchpad combo_m1_g1w1.py): mix Sharpe 0.86 -> 0.93 (2015-20 0.66 -> 0.80, 2021-26 1.10 -> 1.09), worst
    drawdown -17.9% -> -21.0%. Caveats: both are mostly equity exposure (monthly correlation with the value
    proxy 0.76 for G1, 0.59 for W1), and G1's backtest only holds contractors still listed today.
Weights: risk parity, inverse of each component's long-run volatility, FIXED from the 2007-2026 backtest
of 2026-10-12 (proxies: 50/50 IWD+EFV for the screener, SPY RSI2, 8-ETF trend, small-cap insider book;
median trailing-252-day vols 15.2% / 5.8% / 6.7% / 16.2%; G1 20.9% and W1 10.7% over 2015-2026, the window of their
data). Weights changed on 2026-10-13 when I1 joined and on 2026-10-14 when G1 and W1 joined:
the curve is rebuilt from the launch each run, so its first day now reflects them too. Re-estimate them from the live curves once they hold a year of history --
not before (two months of data would just chase noise). Back to target weights on the first run of each
month. Backtest 2007-2026: Sharpe 0.61 vs 0.33 for the undervaluation proxy alone, worst drawdown -18% vs
-62%, worst year -6% vs -39% -- but not above R1 alone (0.66): the gain is robustness, not a new edge.

Variants (2026-11-02, user's request: other compositions and weights), tracked only from their launch so that
nothing in their rules or curve was chosen with hindsight; compositions decided from the earlier edge tests, not from
the three weeks of live results. M1 stays as it is.
  - M2 tous les bots: all 12 active strategy components at equal weight (M1's six + Q1 V1 H1 E1 L1 F1).
  - M3 sans cash: M1's components and weights, but only the invested part of each counts, so idle cash (about 39% of
    M1 at launch) is spread over the invested sleeves at each step. Invested shares are read at the previous point (no
    look-ahead), scale capped at x2, 0.1% fee on the virtual turnover.
  - M4 preuves seules: risk parity of the bots whose edge survived a test (R1 T1 I1 G1 W1), without the screener,
    which showed no measurable edge.
  - M5 facteurs: risk parity of Q1 V1 T1 R1, the original "independent strategies" plan. Q1/V1 vols of 15% are an
    ASSUMPTION (no history to measure them).
Left out everywhere: #52 Courrier copy-all (negative edge), Courrier real (no position), frozen screener variants;
PEA-PME and Pari du jour have no curve in this repo.
Proxy backtest 2007-03 -> 2026-10 (rough: I1 half IWC, G1 ITA, W1 half SPY): see the commit message. M2 and M5 cannot
be backtested (Q1 V1 H1 E1 L1 F1 have no usable history). Compare on live curves after about a year.

Writes results/simulation/combined_equity_curve.csv (same columns as the capital bots, 1000 EUR base) and
combined_summary.json, recomputed from scratch each run (deterministic, idempotent). Clock = the trend
bot's curve (one point per CI run); each component is read as of that time.
"""
import json
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
SIM = HERE / "results/simulation"
START = pd.Timestamp("2026-10-12T07:00:00Z")      # launch of the trend bot, first component curve
BASE_EUR = 1000.0
UNDERVALUATION = ["constrained_equity_curve.csv", "large_equity_curve.csv", "delta_equity_curve.csv"]
COMPONENTS = {"Sous-evaluation": UNDERVALUATION, "R1": ["rsi2_equity_curve.csv"], "T1": ["trend_equity_curve.csv"],
              "I1": ["insider_equity_curve.csv"], "G1": ["govcontracts_equity_curve.csv"],
              "W1": ["wikiattention_equity_curve.csv"], "Q1": ["quality_equity_curve.csv"],
              "V1": ["value_equity_curve.csv"], "H1": ["congress_equity_curve.csv"], "E1": ["tone_equity_curve.csv"],
              "L1": ["lazy_equity_curve.csv"], "F1": ["bestideas_equity_curve.csv"]}
M1_COMPONENTS = ["Sous-evaluation", "R1", "T1", "I1", "G1", "W1"]
LONG_RUN_VOL = {"Sous-evaluation": 0.152, "R1": 0.058, "T1": 0.067, "I1": 0.162, "G1": 0.209, "W1": 0.107,
                "Q1": 0.15, "V1": 0.15}


def risk_parity(names: list) -> dict:
    inv = {k: 1 / LONG_RUN_VOL[k] for k in names}
    return {k: v / sum(inv.values()) for k, v in inv.items()}


WEIGHTS = risk_parity(M1_COMPONENTS)   # ~0.11 / 0.29 / 0.25 / 0.10 / 0.08 / 0.16
VARIANTS_START = pd.Timestamp("2026-11-02T12:00:00Z")
M3_MAX_SCALE, M3_FEE = 2.0, 0.001
VARIANTS = {  # stem: (title, weights, invested only)
    "combined_m2": ("Portefeuille combine M2 (tous les bots)", {k: 1 / len(COMPONENTS) for k in COMPONENTS}, False),
    "combined_m3": ("Portefeuille combine M3 (sans cash)", WEIGHTS, True),
    "combined_m4": ("Portefeuille combine M4 (preuves seules)", risk_parity(["R1", "T1", "I1", "G1", "W1"]), False),
    "combined_m5": ("Portefeuille combine M5 (facteurs)", risk_parity(["Q1", "V1", "T1", "R1"]), False),
}


def curve(name: str, col: str = "total_equity_eur") -> pd.Series:
    df = pd.read_csv(SIM / name, usecols=["timestamp", col])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.dropna().sort_values("timestamp").set_index("timestamp")[col]


def asof(s: pd.Series, clock: pd.DatetimeIndex) -> pd.Series:
    # value as of each clock time; before a bot's first point it is still its starting cash
    return s.reindex(s.index.union(clock)).ffill().bfill().reindex(clock)


def components(clock: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each component's curve normalised to 1, and its invested share of equity (0 to 1)."""
    comp, inv = pd.DataFrame(index=clock), pd.DataFrame(index=clock)
    for k, files in COMPONENTS.items():
        parts, shares = [], []
        for f in files:
            if not (SIM / f).exists():         # bot launched but no run yet: still all cash
                parts.append(pd.Series(1.0, index=clock))
                shares.append(pd.Series(0.0, index=clock))
                continue
            eq = asof(curve(f), clock)
            parts.append(eq)
            shares.append((1 - asof(curve(f, "cash_eur"), clock) / eq).clip(0, 1))
        comp[k] = pd.concat(parts, axis=1).div(pd.concat(parts, axis=1).iloc[0]).mean(axis=1)  # equal mix, normalised
        inv[k] = pd.concat(shares, axis=1).mean(axis=1)
    return comp.dropna(), inv


def monthly_nav(comp: pd.DataFrame, weights: dict) -> pd.Series:
    nav, out, ref_nav, ref = 1.0, [], 1.0, comp.iloc[0]
    month = comp.index[0].strftime("%Y-%m")
    for t, row in comp.iterrows():
        if t.strftime("%Y-%m") != month:           # monthly reset to the target weights
            ref_nav, ref, month = nav, row, t.strftime("%Y-%m")
        nav = ref_nav * sum(weights[k] * row[k] / ref[k] for k in weights)
        out.append(nav)
    return pd.Series(out, index=comp.index)


def no_cash_nav(comp: pd.DataFrame, inv: pd.DataFrame, weights: dict) -> pd.Series:
    w = pd.Series(weights)
    nav, out, prev_alloc = 1.0, [1.0], None
    for i in range(1, len(comp)):
        step = comp.iloc[i][w.index] / comp.iloc[i - 1][w.index] - 1
        held = w * inv.iloc[i - 1][w.index]
        if held.sum() > 0:
            alloc = held / held.sum()
            fee = 0.0 if prev_alloc is None else M3_FEE * (alloc - prev_alloc).abs().sum() / 2
            nav *= 1 + min(1 / held.sum(), M3_MAX_SCALE) * (w * step).sum() - fee
            prev_alloc = alloc
        out.append(nav)
    return pd.Series(out, index=comp.index)


def write(stem: str, title: str, nav_s: pd.Series, comp: pd.DataFrame, weights: dict, start: pd.Timestamp):
    out = [{"timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "cash_eur": 0.0, "total_equity_eur": round(BASE_EUR * nav, 4),
            "n_open": len(weights), "n_closed": 0} for t, nav in nav_s.items()]
    pd.DataFrame(out).to_csv(SIM / f"{stem}_equity_curve.csv", index=False)
    nav, last = nav_s.iloc[-1], comp.iloc[-1]
    summary = {
        "strategy": title,
        "total_equity_eur": BASE_EUR * nav, "total_return_pct": nav - 1, "cash_eur": 0.0,
        "nb_open": len(weights), "nb_closed": 0, "win_rate_closed": None, "avg_return_closed": None,
        "launched_at": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "component_return_pct": {k: float(last[k] - 1) for k in weights},
        "as_of": comp.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (SIM / f"{stem}_summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    print(f"{title} : {BASE_EUR * nav:.2f} EUR ({nav - 1:+.2%}) au {summary['as_of']}, poids "
          + ", ".join(f"{k} {v:.0%}" for k, v in weights.items()))


def main():
    clock_src = curve("trend_equity_curve.csv")
    clock = clock_src[clock_src.index >= START].index
    if not len(clock):
        print("Pas encore de releves depuis le lancement -- rien a combiner.")
        return
    comp, _ = components(clock)
    write("combined", "Portefeuille combine (risque egal)", monthly_nav(comp, WEIGHTS), comp, WEIGHTS, START)

    clock = clock_src[clock_src.index >= VARIANTS_START].index
    if not len(clock):
        print("Variantes M2-M5 : pas encore de releves depuis leur lancement.")
        return
    comp, inv = components(clock)
    for stem, (title, weights, invested_only) in VARIANTS.items():
        nav = no_cash_nav(comp, inv, weights) if invested_only else monthly_nav(comp, weights)
        write(stem, title, nav, comp, weights, VARIANTS_START)


if __name__ == "__main__":
    main()
