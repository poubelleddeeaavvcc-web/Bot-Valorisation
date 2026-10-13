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
Weights: risk parity, inverse of each component's long-run volatility, FIXED from the 2007-2026 backtest
of 2026-10-12 (proxies: 50/50 IWD+EFV for the screener, SPY RSI2, 8-ETF trend, small-cap insider book;
median trailing-252-day vols 15.2% / 5.8% / 6.7% / 16.2%). Weights changed on 2026-10-13 when I1 joined:
the curve is rebuilt from the launch each run, so its first day now reflects them too. Re-estimate them from the live curves once they hold a year of history --
not before (two months of data would just chase noise). Back to target weights on the first run of each
month. Backtest 2007-2026: Sharpe 0.61 vs 0.33 for the undervaluation proxy alone, worst drawdown -18% vs
-62%, worst year -6% vs -39% -- but not above R1 alone (0.66): the gain is robustness, not a new edge.

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
              "I1": ["insider_equity_curve.csv"]}
LONG_RUN_VOL = {"Sous-evaluation": 0.152, "R1": 0.058, "T1": 0.067, "I1": 0.162}
_inv = {k: 1 / v for k, v in LONG_RUN_VOL.items()}
WEIGHTS = {k: v / sum(_inv.values()) for k, v in _inv.items()}   # ~0.15 / 0.38 / 0.33 / 0.14


def curve(name: str) -> pd.Series:
    df = pd.read_csv(SIM / name, usecols=["timestamp", "total_equity_eur"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.dropna().sort_values("timestamp").set_index("timestamp")["total_equity_eur"]


def main():
    clock_src = curve("trend_equity_curve.csv")
    clock = clock_src[clock_src.index >= START].index
    if not len(clock):
        print("Pas encore de releves depuis le lancement -- rien a combiner.")
        return
    comp = pd.DataFrame(index=clock)
    for k, files in COMPONENTS.items():
        parts = []
        for f in files:
            if not (SIM / f).exists():         # bot launched but no run yet: still all cash
                parts.append(pd.Series(1.0, index=clock))
                continue
            s = curve(f)
            # value as of each clock time; before a bot's first point it is still its starting cash
            parts.append(s.reindex(s.index.union(clock)).ffill().bfill().reindex(clock))
        comp[k] = pd.concat(parts, axis=1).div(pd.concat(parts, axis=1).iloc[0]).mean(axis=1)  # equal mix, normalised
    comp = comp.dropna()

    nav, out, ref_nav, ref = 1.0, [], 1.0, comp.iloc[0]
    month = comp.index[0].strftime("%Y-%m")
    for t, row in comp.iterrows():
        if t.strftime("%Y-%m") != month:           # monthly reset to the target weights
            ref_nav, ref, month = nav, row, t.strftime("%Y-%m")
        nav = ref_nav * sum(WEIGHTS[k] * row[k] / ref[k] for k in WEIGHTS)
        out.append({"timestamp": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "cash_eur": 0.0,
                    "total_equity_eur": round(BASE_EUR * nav, 4), "n_open": len(WEIGHTS), "n_closed": 0})
    pd.DataFrame(out).to_csv(SIM / "combined_equity_curve.csv", index=False)
    last = comp.iloc[-1]
    summary = {
        "strategy": "Portefeuille combine (risque egal)",
        "total_equity_eur": BASE_EUR * nav, "total_return_pct": nav - 1, "cash_eur": 0.0,
        "nb_open": len(WEIGHTS), "nb_closed": 0, "win_rate_closed": None, "avg_return_closed": None,
        "launched_at": START.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "weights": {k: round(v, 4) for k, v in WEIGHTS.items()},
        "component_return_pct": {k: float(last[k] - 1) for k in WEIGHTS},
        "as_of": comp.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (SIM / "combined_summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    print(f"Portefeuille combine : {BASE_EUR * nav:.2f} EUR ({nav - 1:+.2%}) au {summary['as_of']}, poids "
          + ", ".join(f"{k} {v:.0%}" for k, v in WEIGHTS.items()))


if __name__ == "__main__":
    main()
