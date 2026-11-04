"""La Lettre du Courrier (2026-11-04, demande de l'utilisateur) : une newsletter hebdomadaire
rédigée par le bot lui-même après digestion de toutes les newsletters de la semaine.

Matière (rien d'autre) :
  - results/screener/mail_signal_journal.csv : avis vérifiés de la semaine (citation mot pour mot)
  - results/screener/newsletter_database.csv : objets des newsletters financières reçues
  - results/screener/mail_signal_source_scorecard.csv : track record de chaque source
  - results/simulation/mail_signal_real_ledger.csv : ce que la stratégie réelle a fait

Partage du travail :
  - Python calcule tout ce qui est factuel (tickers les plus cités, consensus, désaccords, track
    record des sources, actions du bot). Aucun chiffre ne vient d'Ollama.
  - Ollama (même modèle que mail_signal_bot.py) ne rédige que les "thèmes de la semaine" à partir
    d'une liste numérotée de citations et d'objets. Chaque thème doit citer ses numéros ; un thème
    qui ne s'appuie pas sur au moins 2 éléments valides de 2 publications différentes est écarté
    (règle d'ancrage du projet : Ollama ne devine jamais un fait).

Semaine = samedi -> vendredi (UTC, date du mail). La lettre est écrite par son propre workflow
(.github/workflows/courrier-lettre.yml, le samedi), séparé du bot pour ne pas allonger son job.
Archive : results/screener/mail_signal_lettres.json (une entrée par semaine, la plus récente d'abord).

Usage : python screener/mail_signal_lettre.py [--semaine YYYY-MM-DD (vendredi)] [--force]
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import sys

import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
JOURNAL = ROOT / "results/screener/mail_signal_journal.csv"
SUBJECTS = ROOT / "results/screener/newsletter_database.csv"
SCORECARD = ROOT / "results/screener/mail_signal_source_scorecard.csv"
REAL_LEDGER = ROOT / "results/simulation/mail_signal_real_ledger.csv"
OUT = ROOT / "results/screener/mail_signal_lettres.json"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.environ.get("LETTRE_OLLAMA_MODEL", "mistral:7b")  # meme modele que mail_signal_bot.py
OLLAMA_TIMEOUT = int(os.environ.get("LETTRE_OLLAMA_TIMEOUT", "1500"))
# contexte plus large que le bot (2048) : ici un seul appel lit toute la semaine
OLLAMA_OPTIONS = {"temperature": 0.2, "num_ctx": 6144, "num_predict": 1200}
MAX_ITEMS = 45        # elements numerotes envoyes a Ollama (citations d'abord, puis objets)
CITATION_CHARS = 160
TOP_TICKERS = 8
MAX_THEMES = 4


def _read_csv(path: pathlib.Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def last_friday(today: dt.date) -> dt.date:
    """Dernier vendredi entierement ecoule (strictement avant aujourd'hui)."""
    days = (today.weekday() - 4) % 7 or 7
    return today - dt.timedelta(days=days)


def _source_record(scorecard: pd.DataFrame) -> dict:
    rec = {}
    for _, r in scorecard.iterrows():
        n20 = r.get("n_evalues_20j")
        ex20 = r.get("excess_moy_20j")
        rec[str(r["source"])] = {
            "statut": r.get("statut", ""),
            "n_evalues_20j": int(n20) if pd.notna(n20) else 0,
            "excess_moy_20j": float(ex20) if pd.notna(ex20) else None,
        }
    return rec


def build_facts(start: dt.date, end: dt.date) -> dict:
    j = _read_csv(JOURNAL)
    if not j.empty:
        j["d"] = j["mail_date_utc"].astype(str).str[:10]
        j = j[(j["d"] >= start.isoformat()) & (j["d"] <= end.isoformat())]
        if "backfill" in j:
            j = j[j["backfill"].astype(str) != "True"]
    subj = _read_csv(SUBJECTS)
    if not subj.empty:
        subj = subj[(subj["date"].astype(str) >= start.isoformat()) & (subj["date"].astype(str) <= end.isoformat())]
    record = _source_record(_read_csv(SCORECARD))

    tickers = []
    desaccords = []
    if not j.empty:
        for tk, g in j.groupby("ticker"):
            n_long = int((g["side"] == "long").sum())
            n_short = int((g["side"] == "short").sum())
            pubs = sorted(set(g["publication"].astype(str)))
            row = {
                "ticker": tk, "nom": str(g["name"].iloc[0]), "n_avis": len(g),
                "n_achat": n_long, "n_vente": n_short, "publications": pubs,
                "sources": [{"publication": p, **record.get(p, {})} for p in pubs],
            }
            tickers.append(row)
            if n_long and n_short:
                desaccords.append({
                    **row,
                    "citations": [{"publication": r.publication, "sens": r.side,
                                   "citation": str(r.citation)[:CITATION_CHARS]} for r in g.itertuples()],
                })
        tickers.sort(key=lambda r: (-len(r["publications"]), -r["n_avis"], r["ticker"]))

    pubs_count = j["publication"].value_counts().to_dict() if not j.empty else {}

    actions = []
    led = _read_csv(REAL_LEDGER)
    for _, r in led.iterrows():
        for kind, col in (("achat", "entry_date"), ("vente", "exit_date")):
            d = str(r.get(col) or "")[:10]
            if d and start.isoformat() <= d <= end.isoformat():
                actions.append({"date": d, "type": kind, "ticker": r["ticker"], "nom": r.get("name", ""),
                                "newsletter": r.get("newsletter_principale", ""),
                                "rendement": float(r["return_pct"]) if kind == "vente" and pd.notna(r.get("return_pct")) else None,
                                "motif": r.get("exit_reason", "") if kind == "vente" else ""})
    actions.sort(key=lambda a: a["date"])

    return {
        "journal": j, "subjects": subj,
        "stats": {"n_avis": int(len(j)), "n_achat": int((j["side"] == "long").sum()) if not j.empty else 0,
                  "n_vente": int((j["side"] == "short").sum()) if not j.empty else 0,
                  "n_newsletters": int(len(subj)), "n_publications": len(pubs_count)},
        "publications": [{"publication": p, "n_avis": int(n), **record.get(p, {})} for p, n in pubs_count.items()],
        "tickers": tickers[:TOP_TICKERS],
        "desaccords": desaccords,
        "actions_bot": actions,
    }


def build_items(journal: pd.DataFrame, subjects: pd.DataFrame) -> list[dict]:
    items = []
    for r in journal.itertuples():
        sens = "achat" if r.side == "long" else "vente"
        items.append({"publication": str(r.publication), "texte": f"[avis {sens} {r.ticker}] {str(r.citation)[:CITATION_CHARS]}"})
    for r in subjects.itertuples():
        items.append({"publication": "objet de newsletter", "texte": f"[objet] {r.subject}"})
    items = items[:MAX_ITEMS]
    for i, it in enumerate(items, 1):
        it["id"] = i
    return items


PROMPT = """Tu es le redacteur d'une lettre financiere hebdomadaire en francais. Voici la matiere de la
semaine : des extraits numerotes de newsletters reellement recues (avis d'achat/vente avec citation
exacte, et objets de newsletters). N'utilise QUE ces extraits. N'invente aucun chiffre, aucun fait,
aucune societe absente des extraits. Ne donne aucun conseil d'investissement personnel.

Identifie de 2 a 4 THEMES de la semaine : des sujets sur lesquels PLUSIEURS extraits convergent
(un secteur, une tendance, une inquietude, un type d'entreprise...). Pour chaque theme : un titre
court, un paragraphe de 2 a 4 phrases qui resume ce que disent les extraits, et la liste des numeros
d'extraits utilises (au moins 2). Ne recopie pas les extraits, donne seulement leurs numeros.

Reponds uniquement en JSON :
{{"themes": [{{"titre": "...", "texte": "...", "refs": [1, 4, 7]}}]}}

Extraits :
{items}
"""


def _call_ollama_json(prompt: str) -> dict:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json", "keep_alive": "20m",
               "options": OLLAMA_OPTIONS}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    return json.loads(json.loads(resp.content)["response"])


def write_themes(items: list[dict]) -> tuple[list[dict], str]:
    if len(items) < 2:
        return [], "pas assez de matiere cette semaine"
    lines = "\n".join(f"{it['id']}. ({it['publication']}) {it['texte']}" for it in items)
    try:
        raw = _call_ollama_json(PROMPT.format(items=lines))
    except Exception as e:  # Ollama absent / reponse invalide : la lettre sort quand meme, sans themes
        print(f"  echec Ollama : {e}", file=sys.stderr)
        return [], "redaction Ollama indisponible"
    by_id = {it["id"]: it for it in items}
    themes = []
    for t in raw.get("themes", []) if isinstance(raw, dict) else []:
        if not isinstance(t, dict):
            continue
        refs = []
        for x in t.get("refs", []) or []:
            try:
                x = int(x)
            except (TypeError, ValueError):
                continue
            if x in by_id and x not in refs:
                refs.append(x)
        pubs = {by_id[x]["publication"] for x in refs}
        titre, texte = str(t.get("titre", "")).strip(), str(t.get("texte", "")).strip()
        # ancrage : au moins 2 extraits valides, de 2 publications differentes
        # pas deux fois le meme titre (le 8b a tendance a decliner un theme en serie), 4 themes max
        if titre.lower() in {t["titre"].lower() for t in themes} or len(themes) >= MAX_THEMES:
            continue
        if titre and texte and len(refs) >= 2 and len(pubs) >= 2:
            themes.append({"titre": titre, "texte": texte,
                           "sources": [{"id": x, "publication": by_id[x]["publication"],
                                        "texte": by_id[x]["texte"]} for x in refs]})
    return themes, "" if themes else "aucun theme assez etaye"


def make_letter(friday: dt.date) -> dict:
    start = friday - dt.timedelta(days=6)
    facts = build_facts(start, friday)
    items = build_items(facts["journal"], facts["subjects"])
    themes, note = write_themes(items)
    print(f"  semaine {start} -> {friday} : {facts['stats']['n_avis']} avis, "
          f"{facts['stats']['n_newsletters']} newsletters, {len(themes)} theme(s)")
    return {
        "semaine_fin": friday.isoformat(), "semaine_debut": start.isoformat(),
        "genere_le": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "modele": OLLAMA_MODEL, "stats": facts["stats"], "themes": themes, "note_themes": note,
        "tickers": facts["tickers"], "desaccords": facts["desaccords"],
        "publications": facts["publications"], "actions_bot": facts["actions_bot"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semaine", help="vendredi de fin de semaine (YYYY-MM-DD)")
    ap.add_argument("--force", action="store_true", help="reecrit la lettre si elle existe deja")
    a = ap.parse_args()
    friday = dt.date.fromisoformat(a.semaine) if a.semaine else last_friday(dt.datetime.now(dt.timezone.utc).date())
    if friday.weekday() != 4:
        sys.exit(f"{friday} n'est pas un vendredi")
    letters = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    if any(l["semaine_fin"] == friday.isoformat() for l in letters) and not a.force:
        print(f"Lettre du {friday} deja ecrite.")
        return
    letter = make_letter(friday)
    letters = [l for l in letters if l["semaine_fin"] != friday.isoformat()] + [letter]
    letters.sort(key=lambda l: l["semaine_fin"], reverse=True)
    OUT.write_text(json.dumps(letters, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Lettre ecrite -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
