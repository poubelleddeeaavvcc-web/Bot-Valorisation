"""Bilan annexe des portefeuilles combines (2026-11-02, user's request): M1-M5 side by side, written to
results/simulation/combined_bilan.md at each CI run from the curves build_combined_portfolio.py just wrote.

Two tables on purpose: the official curves (M2-M5 only count from their launch) and a reconstruction from the day all
of M1's components existed, which gives a longer comparison but was known when the variants were designed, so it
proves nothing. S&P 500 = index price from equity_curve.csv, without dividends. Does not rank or pick a winner before
MIN_DAYS of official history (noise, see the module docstring of build_combined_portfolio.py).
"""
import sys

import pandas as pd

sys.path.insert(0, __file__.rsplit("screener", 1)[0] + "screener")
import build_combined_portfolio as b  # noqa: E402

RECON_START = pd.Timestamp("2026-10-14T12:00:00Z")
MIN_DAYS = 365
NAMES = {"combined": "M1 Risque égal", "combined_m2": "M2 Tous les bots", "combined_m3": "M3 Sans cash",
         "combined_m4": "M4 Preuves seules", "combined_m5": "M5 Facteurs"}


def pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x * 100:+.2f} %".replace(".", ",")


def perf_since(s: pd.Series, days: int):
    t = s.index[-1] - pd.Timedelta(days=days)
    return s.iloc[-1] / s[s.index <= t].iloc[-1] - 1 if s.index[0] <= t else None


def benchmark() -> pd.Series:
    df = pd.read_csv(b.SIM / "equity_curve.csv", usecols=["timestamp", "spy_price"]).dropna()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").set_index("timestamp")["spy_price"]


def main():
    spy, rows, days = benchmark(), [], 0
    for stem, name in NAMES.items():
        f = b.SIM / f"{stem}_equity_curve.csv"
        if not f.exists():
            rows.append(f"| {name} | pas encore de relevé | | | | | |")
            continue
        s = b.curve(f.name)
        ref = b.asof(spy, s.index)
        if stem != "combined":
            days = (s.index[-1] - s.index[0]).days
        rows.append(f"| {name} | {s.index[0]:%d/%m/%Y} | {f"{s.iloc[-1]:.2f}".replace(".", ",")} € | {pct(s.iloc[-1] / s.iloc[0] - 1)} "
                    f"| {pct(perf_since(s, 7))} | {pct(ref.iloc[-1] / ref.iloc[0] - 1)} | {pct((s / s.cummax() - 1).min())} |")

    clock_src = b.curve("trend_equity_curve.csv")
    clock = clock_src[clock_src.index >= RECON_START].index
    comp, inv = b.components(clock)
    ref = b.asof(spy, comp.index)
    specs = {"combined": (b.WEIGHTS, False), **{k: (w, o) for k, (_, w, o) in b.VARIANTS.items()}}
    recon = []
    for stem, (w, invested_only) in specs.items():
        nav = b.no_cash_nav(comp, inv, w) if invested_only else b.monthly_nav(comp, w)
        cash = 1 - sum(w[k] * inv[k].iloc[-1] for k in w)
        recon.append(f"| {NAMES[stem]} | {pct(nav.iloc[-1] - 1)} | {pct((nav / nav.cummax() - 1).min())} "
                     f"| {'0 % (redistribué)' if invested_only else f'{cash * 100:.0f} %'} |")
    comps = []
    for k in comp.columns:
        cells = " | ".join(f"{w[k] * 100:.0f} %" if k in w else "–" for w, _ in specs.values())
        comps.append(f"| {k} | {pct(comp[k].iloc[-1] - 1)} | {inv[k].iloc[-1] * 100:.0f} % | {cells} |")

    if days < MIN_DAYS:
        verdict = (f"**Pas de classement pour l'instant :** {days} jour(s) de suivi officiel sur les {MIN_DAYS} prévus. "
                   "À si court terme, les écarts relèvent surtout du hasard.")
    else:
        verdict = "**Un an de suivi atteint :** les portefeuilles peuvent être comparés et leurs poids réestimés."
    text = "\n".join([
        f"# Bilan annexe — portefeuilles combinés ({comp.index[-1]:%d/%m/%Y %H:%M} UTC)", "", verdict, "",
        "Portefeuilles virtuels, base 1 000 €. Le S&P 500 est hors dividendes, les bots dividendes compris.", "",
        "## Suivi officiel", "",
        "| Portefeuille | Départ | Valeur | Depuis le départ | 7 jours | S&P 500 même période | Pire baisse |",
        "|---|---|---:|---:|---:|---:|---:|", *rows, "",
        f"## Reconstitution depuis le {RECON_START:%d/%m/%Y}", "",
        "Calculée après coup avec les poids d'aujourd'hui : indicative, elle ne prouve rien. "
        f"S&P 500 sur la période : {pct(ref.iloc[-1] / ref.iloc[0] - 1)}.", "",
        "| Portefeuille | Performance | Pire baisse | Cash actuel |", "|---|---:|---:|---:|", *recon, "",
        "## Composantes sur la même période", "",
        "| Composante | Performance | Part investie | " + " | ".join(n.split()[0] for n in NAMES.values()) + " |",
        "|---|---:|---:|" + "---:|" * len(NAMES), *comps, "",
    ])
    (b.SIM / "combined_bilan.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
