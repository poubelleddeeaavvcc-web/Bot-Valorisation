"""Bi-weekly check-in report comparing the 3 paper-trading bots (results/simulation/),
so that a "which bot do I actually base my real strategy on" decision can be made once
each bot has enough clean closed trades to be judged -- the decision must wait for
CLEAN_TRADE_THRESHOLD closed trades per bot, and must exclude closes that are artifacts
of a code/strategy change rather than the bot's normal signal.

Deliberately NOT auto-detected by a general heuristic (same-day batch, holding_days==0,
etc.) -- that flagged trades never confirmed as artifacts (e.g. the 08-17 momentum_perdu
batch). Only KNOWN_ARTIFACT_TRADES, individually identified and confirmed, are excluded.
When a new batch of closes looks suspicious in a future check-in, investigate it (same
way the 08-07 and 08-18 ones were: check `git log` for a same-day commit touching the
valuation/momentum model) and add confirmed entries to the list below -- don't infer.

Run every ~2 weeks: `python -m portfolio.bot_checkin`.
"""
import pathlib
import sys
from datetime import date

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
SIM_DIR = HERE / "results/simulation"

CLEAN_TRADE_THRESHOLD = 30  # don't judge a bot's win rate/avg return before this many clean closes

# Individually confirmed artifacts -- a closed trade caused by a same-day code/strategy
# change rather than the bot's own signal, so it shouldn't count toward performance.
# (bot_key, ticker, exit_date) -> reason. Add new rows only after confirming via git log,
# never by pattern-matching.
KNOWN_ARTIFACT_TRADES = {
    ("bot1_blind", "ALV", "2026-08-07"): "jour-0 : recalibrage du signal au lancement du bot (pas un vrai trade)",
    ("bot1_blind", "HMY", "2026-08-07"): "jour-0 : recalibrage du signal au lancement du bot (pas un vrai trade)",
    ("bot1_blind", "DAL", "2026-08-18"): "commit 47c9e91 (durcissement du modele de valorisation) a declenche 5 sorties le meme jour",
    ("bot1_blind", "EME", "2026-08-18"): "commit 47c9e91 (durcissement du modele de valorisation) a declenche 5 sorties le meme jour",
    ("bot1_blind", "KALU", "2026-08-18"): "commit 47c9e91 (durcissement du modele de valorisation) a declenche 5 sorties le meme jour",
    ("bot1_blind", "NBIX", "2026-08-18"): "commit 47c9e91 (durcissement du modele de valorisation) a declenche 5 sorties le meme jour",
    ("bot1_blind", "VMI", "2026-08-18"): "commit 47c9e91 (durcissement du modele de valorisation) a declenche 5 sorties le meme jour",
}

BOTS = [
    {
        "key": "bot1_blind",
        "label": "Bot #1 (blind, sans contrainte de capital)",
        "ledger": SIM_DIR / "portfolio_ledger.csv",
        "summary": SIM_DIR / "summary.json",
        "has_eur_equity": False,
    },
    {
        "key": "bot2_constrained",
        "label": "Bot #2 (constrained, NA<=75%, max 3/secteur)",
        "ledger": SIM_DIR / "constrained_portfolio_ledger.csv",
        "summary": SIM_DIR / "constrained_summary.json",
        "has_eur_equity": True,
    },
    {
        "key": "bot3_large",
        "label": "Bot #3 (large, ~30 lignes, equilibrage secteur sans plafond)",
        "ledger": SIM_DIR / "large_portfolio_ledger.csv",
        "summary": SIM_DIR / "large_summary.json",
        "has_eur_equity": True,
    },
]


def split_clean_vs_artifacts(bot_key: str, closed: pd.DataFrame):
    """Return (clean, excluded) closed-trade rows, matched against KNOWN_ARTIFACT_TRADES."""
    closed = closed.copy()

    def lookup_reason(row):
        return KNOWN_ARTIFACT_TRADES.get((bot_key, row["ticker"], str(row["exit_date"])))

    closed["_exclude_reason"] = closed.apply(lookup_reason, axis=1)
    excluded = closed[closed["_exclude_reason"].notna()]
    clean = closed[closed["_exclude_reason"].isna()]
    return clean, excluded


def trade_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "win_rate": None, "avg_return": None, "avg_win": None, "avg_loss": None}
    wins = df[df["return_pct"] > 0]["return_pct"]
    losses = df[df["return_pct"] <= 0]["return_pct"]
    return {
        "n": len(df),
        "win_rate": len(wins) / len(df),
        "avg_return": df["return_pct"].mean(),
        "avg_win": wins.mean() if len(wins) else None,
        "avg_loss": losses.mean() if len(losses) else None,
    }


def check_bot(bot: dict) -> dict:
    ledger = pd.read_csv(bot["ledger"])
    launch_date = pd.to_datetime(ledger["entry_date"]).min().date()
    days_running = (date.today() - launch_date).days

    closed = ledger[ledger["exit_date"].notna()].copy()
    closed["holding_days"] = closed["holding_days"].astype(float)
    raw_stats = trade_stats(closed)

    clean, excluded = split_clean_vs_artifacts(bot["key"], closed) if len(closed) else (closed, closed)
    clean_stats = trade_stats(clean)

    result = {
        "bot": bot["label"],
        "launch_date": launch_date,
        "days_running": days_running,
        "nb_open": int((ledger["status"] == "open").sum()) if "status" in ledger else None,
        "raw": raw_stats,
        "clean": clean_stats,
        "excluded_rows": excluded,
        "clean_pace_per_day": clean_stats["n"] / days_running if days_running else 0,
    }

    if result["clean_pace_per_day"] > 0:
        remaining = CLEAN_TRADE_THRESHOLD - clean_stats["n"]
        result["eta_days_to_threshold"] = max(0, remaining) / result["clean_pace_per_day"]
    else:
        result["eta_days_to_threshold"] = None

    if bot["has_eur_equity"] and bot["summary"].exists():
        import json
        summ = json.loads(bot["summary"].read_text(encoding="utf-8"))
        result["total_equity_eur"] = summ.get("total_equity_eur")
        result["total_return_pct"] = summ.get("total_return_pct")
    else:
        import json
        summ = json.loads(bot["summary"].read_text(encoding="utf-8"))
        result["avg_unrealized_open"] = summ.get("avg_unrealized_open")

    return result


def fmt_pct(x):
    return "n/a" if x is None else f"{x:+.1%}"


def print_report(results: list[dict]):
    print(f"=== Check-in bots -- {date.today().isoformat()} ===")
    print(f"(seuil de decision : {CLEAN_TRADE_THRESHOLD} clotures propres par bot)\n")

    for r in results:
        print(f"--- {r['bot']} ---")
        print(f"  Lance le {r['launch_date']} ({r['days_running']} jours de fonctionnement)")
        if r.get("nb_open") is not None:
            print(f"  Positions ouvertes : {r['nb_open']}")
        if "total_equity_eur" in r:
            print(f"  Equity totale : {r['total_equity_eur']:.2f} EUR ({fmt_pct(r['total_return_pct'])})")
        if "avg_unrealized_open" in r:
            print(f"  Retour moyen non-realise (positions ouvertes) : {fmt_pct(r['avg_unrealized_open'])}")

        raw, clean = r["raw"], r["clean"]
        print(f"  Clotures brutes   : n={raw['n']:>3}  win_rate={fmt_pct(raw['win_rate'])}  "
              f"retour_moyen={fmt_pct(raw['avg_return'])}")
        print(f"  Clotures propres  : n={clean['n']:>3}  win_rate={fmt_pct(clean['win_rate'])}  "
              f"retour_moyen={fmt_pct(clean['avg_return'])}  "
              f"gain_moyen={fmt_pct(clean['avg_win'])}  perte_moyenne={fmt_pct(clean['avg_loss'])}")

        excl = r["excluded_rows"]
        if len(excl):
            print(f"  -> {len(excl)} cloture(s) exclue(s) comme artefact :")
            for _, row in excl.iterrows():
                print(f"       {row['ticker']:<10} {row['exit_date']}  {row['exit_reason']:<20} "
                      f"({row['_exclude_reason']})")

        if clean["n"] >= CLEAN_TRADE_THRESHOLD:
            print(f"  >> SEUIL ATTEINT ({clean['n']}/{CLEAN_TRADE_THRESHOLD}) -- ce bot est jugeable.")
        elif r["eta_days_to_threshold"] is not None:
            eta = pd.Timestamp.today() + pd.Timedelta(days=r["eta_days_to_threshold"])
            print(f"  Progression : {clean['n']}/{CLEAN_TRADE_THRESHOLD} clotures propres -- "
                  f"estimation seuil atteint vers {eta.date().isoformat()} au rythme actuel.")
        else:
            print(f"  Progression : {clean['n']}/{CLEAN_TRADE_THRESHOLD} clotures propres -- "
                  f"pas encore de clotures, rythme non estimable.")
        print()

    ready = [r for r in results if r["clean"]["n"] >= CLEAN_TRADE_THRESHOLD]
    if ready:
        print(f"Bots jugeables des maintenant : {', '.join(r['bot'] for r in ready)}")
    else:
        soonest = min((r for r in results if r["eta_days_to_threshold"] is not None),
                      key=lambda r: r["eta_days_to_threshold"], default=None)
        if soonest:
            eta = pd.Timestamp.today() + pd.Timedelta(days=soonest["eta_days_to_threshold"])
            print(f"Aucun bot n'a encore atteint le seuil. Le plus proche ({soonest['bot']}) "
                  f"l'atteindrait vers {eta.date().isoformat()} au rythme actuel.")
        else:
            print("Aucun bot n'a encore de clotures propres -- rien n'est estimable pour l'instant.")


def main():
    results = [check_bot(bot) for bot in BOTS]
    print_report(results)

    rows = []
    for bot, r in zip(BOTS, results):
        rows.append({
            "bot": bot["key"],
            "checkin_date": date.today().isoformat(),
            "days_running": r["days_running"],
            "n_closed_raw": r["raw"]["n"],
            "n_closed_clean": r["clean"]["n"],
            "win_rate_clean": r["clean"]["win_rate"],
            "avg_return_clean": r["clean"]["avg_return"],
            "avg_win_clean": r["clean"]["avg_win"],
            "avg_loss_clean": r["clean"]["avg_loss"],
            "total_equity_eur": r.get("total_equity_eur"),
            "total_return_pct": r.get("total_return_pct"),
            "n_artifacts_excluded": len(r["excluded_rows"]),
            "threshold_reached": r["clean"]["n"] >= CLEAN_TRADE_THRESHOLD,
        })
    out = pd.DataFrame(rows)
    out_path = HERE / "results/bot_checkin_history.csv"
    header = not out_path.exists()
    out.to_csv(out_path, mode="a", index=False, header=header)
    print(f"\nRapport ajoute a {out_path.relative_to(HERE)}")


if __name__ == "__main__":
    sys.exit(main())
