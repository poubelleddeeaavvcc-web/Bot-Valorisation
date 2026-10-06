"""Resume compact des courbes d'equity de tous les bots, pour le dashboard.

Chaque bot ajoute une ligne a sa courbe a chaque cycle (~29 par jour) : les ~40 fichiers
results/simulation/*equity_curve*.csv pesaient 1,9 Mo le 2026-10-06 et grossissent sans fin
(~15 Mo projetes a un an), retelecharges en entier par le dashboard a chaque nouveau cycle. Ce
script publie results/simulation/equity_curves_summary.json, un seul fichier qui contient :
  - les 31 derniers jours de chaque courbe en entier (periodes 7/14/30 j du dashboard) ;
  - au-dela, 1 releve par jour, pris au MEME cycle pour tous les bots (dernier releve du jour de
    la courbe de reference du Bot #1, puis pour chaque courbe le releve le plus proche a 20 min
    pres -- meme tolerance d'appariement que le dashboard), pour que les courbes restent
    alignees entre elles ;
  - tous les releves autour du lancement de chaque bot et de l'introduction des frais : le
    dashboard y rebase ses courbes ("Depart commun", "Sans frais"), ils doivent rester exacts ;
  - le premier et le dernier releve de chaque courbe.
Les lignes gardees sont recopiees telles quelles (aucun nombre reformate) : le dashboard lit le
meme CSV, simplement avec moins de lignes anciennes.

Usage : python screener/build_equity_summary.py (dernier job de update-screener.yml, apres
tous les jobs de bots).
"""
import json
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parent.parent / "results" / "simulation"
OUT_PATH = SIM_DIR / "equity_curves_summary.json"
FULL_RES_DAYS = 31  # > 30 j, la plus longue periode du dashboard, avec une marge d'un jour
JOIN_TOLERANCE = timedelta(minutes=20)  # = SIM_JOIN_TOLERANCE_MS du dashboard
LAUNCH_WINDOW = (timedelta(hours=1), timedelta(hours=3))  # avant / apres chaque lancement
# Introduction des frais de 1 EUR par vente (commit d2a9b88a0) : le dashboard lit le dernier
# releve avant cette date pour sa vue "Sans frais" (SIM_FEE_START_MS).
FEE_START = datetime(2026, 9, 4, 10, 2, 3, tzinfo=timezone.utc)
FEE_WINDOW = (timedelta(hours=2), timedelta(hours=1))
REFERENCE = "equity_curve.csv"  # Bot #1, la courbe la plus ancienne


def parse_ts(line):
    try:
        return datetime.fromisoformat(line.split(",", 1)[0].replace("Z", "+00:00"))
    except ValueError:
        return None


def read_curve(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return "", []
    rows = [line for line in lines[1:] if line.strip()]
    return lines[0], [(line, parse_ts(line)) for line in rows]


def daily_anchors(rows, cutoff):
    """Dernier releve de chaque jour UTC, avant la coupure."""
    last_of_day = {}
    for _, ts in rows:
        if ts is not None and ts < cutoff:
            day = ts.date()
            if day not in last_of_day or ts > last_of_day[day]:
                last_of_day[day] = ts
    return sorted(last_of_day.values())


def kept_indices(rows, cutoff, windows, anchors):
    keep = {0, len(rows) - 1}
    old = []  # (timestamp, index) des releves anciens, pour les ancres journalieres
    for i, (_, ts) in enumerate(rows):
        if ts is None or ts >= cutoff or any(a <= ts <= b for a, b in windows):
            keep.add(i)
        if ts is not None and ts < cutoff:
            old.append((ts, i))
    old.sort()
    old_ts = [ts for ts, _ in old]
    for anchor in anchors:
        j = bisect_left(old_ts, anchor)
        best = min((k for k in (j - 1, j) if 0 <= k < len(old)),
                   key=lambda k: abs(old_ts[k] - anchor), default=None)
        if best is not None and abs(old_ts[best] - anchor) <= JOIN_TOLERANCE:
            keep.add(old[best][1])
    return sorted(keep)


def main():
    paths = sorted(p for p in SIM_DIR.glob("*equity_curve*.csv") if not p.name.startswith("mail_signal"))
    curves = {p.name: read_curve(p) for p in paths}
    curves = {name: c for name, c in curves.items() if c[1]}
    if not curves:
        print("Aucune courbe a resumer.")
        return

    latest = max(ts for _, rows in curves.values() for _, ts in rows if ts is not None)
    cutoff = latest - timedelta(days=FULL_RES_DAYS)
    windows = [(FEE_START - FEE_WINDOW[0], FEE_START + FEE_WINDOW[1])]
    for _, rows in curves.values():
        first = next((ts for _, ts in rows if ts is not None), None)
        if first is not None:
            windows.append((first - LAUNCH_WINDOW[0], first + LAUNCH_WINDOW[1]))
    anchors = daily_anchors(curves[REFERENCE][1], cutoff) if REFERENCE in curves else []

    summary = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "full_resolution_days": FULL_RES_DAYS,
        "cutoff": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {},
        "source_rows": {},
    }
    kept_total = source_total = 0
    for name, (header, rows) in curves.items():
        idx = kept_indices(rows, cutoff, windows, anchors)
        summary["files"][name] = "\n".join([header] + [rows[i][0] for i in idx]) + "\n"
        summary["source_rows"][name] = len(rows)
        kept_total += len(idx)
        source_total += len(rows)

    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"{len(curves)} courbes, {kept_total}/{source_total} releves gardes, "
          f"{OUT_PATH.stat().st_size // 1024} Ko (coupure {summary['cutoff']})")


if __name__ == "__main__":
    main()
