"""Test inverse pre-enregistre sur deux lettres Seeking Alpha (2026-11-13, demande de l'utilisateur).

POURQUOI
--------
Apres la separation des lettres Seeking Alpha par nom (ca3990bc1), deux lettres ressortaient nettement
negatives sur leurs avis de septembre-octobre : Tech Daily (-10,3 % vs marche a 20 seances, t -3,3) et
Stock Ideas (-6,4 %, t -2,3). Idee : faire l'INVERSE de leurs avis (acheter quand elles disent de vendre,
vendre a decouvert quand elles disent d'acheter, couvert par l'ETF du marche).

Ces deux lettres ont ete CHOISIES sur ces memes donnees : les rejouer ne prouverait rien. Le test ne
compte donc que leurs avis envoyes APRES START, et toutes les regles ci-dessous sont fixees avant le
premier resultat. Ne pas les modifier apres coup (sinon le test ne vaut plus rien) -- un changement
serait un nouveau test, avec un nouveau START.

REGLES (fixees le 2026-11-13, avant tout avis du test)
------------------------------------------------------
- Echantillon : avis du journal (mail_signal_journal.csv) de LETTRES, envoyes a partir de START,
  achats et ventes, hors repetitions (meme lettre, meme titre, meme sens sous 7 jours).
- Mesure : rendement de la position inverse a 20 seances, moins le marche = -ex_20 (ex_20 est deja
  l'ecart au marche dans le sens de l'avis, voir mail_signal_scoring.py). A 5 seances : pour info.
- Verdict, sur les deux lettres ENSEMBLE, une seule fois, quand N_VERDICT avis sont mesures a 20
  seances (les N_VERDICT premiers par date d'envoi, pour ne pas choisir le moment) :
    * "confirme"  : moyenne > COUT_ESTIME et t >= T_VERDICT (ecart-type de l'echantillon) ;
    * "infirme"   : sinon.
  Avant : "en cours". Apres le verdict, les avis suivants restent affiches pour info.
- COUT_ESTIME = 0,5 % par avis : frais d'aller-retour + emprunt du titre vendu a decouvert sur un mois,
  ordre de grandeur pour un particulier. Le resultat brut est aussi affiche.
- Aucune position reelle ni capital : c'est une mesure, comme le journal.

Sortie : results/screener/mail_signal_inverse_test.json (lu par le dashboard Portfolio, onglet Courrier).
"""
import json
import math
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
JOURNAL_PATH = HERE / "results/screener/mail_signal_journal.csv"
OUT_PATH = HERE / "results/screener/mail_signal_inverse_test.json"

START = pd.Timestamp("2026-11-13T11:00:00Z")
LETTRES = ["Tech Daily (Seeking Alpha)", "Stock Ideas (Seeking Alpha)"]
N_VERDICT = 30
T_VERDICT = 1.5
COUT_ESTIME = 0.005


def _flag(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def _stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    n = len(x)
    if not n:
        return {"n": 0, "moyenne": None, "t": None, "taux_reussite": None}
    m = float(x.mean())
    sd = float(x.std()) if n > 1 else float("nan")
    t = m * math.sqrt(n) / sd if n > 1 and sd > 0 else None
    return {"n": n, "moyenne": m, "t": t, "taux_reussite": float((x > 0).mean())}


def build() -> dict:
    out = {
        "regles": {"start": START.isoformat(), "lettres": LETTRES, "n_verdict": N_VERDICT,
                   "t_verdict": T_VERDICT, "cout_estime": COUT_ESTIME, "horizon_seances": 20},
        "statut": "en cours", "verdict": None,
    }
    if not JOURNAL_PATH.exists():
        out.update({"global": _stats(pd.Series(dtype=float)), "par_lettre": {}, "avis": []})
        return out
    j = pd.read_csv(JOURNAL_PATH)
    j["ts"] = pd.to_datetime(j["mail_date_utc"], utc=True, errors="coerce", format="mixed")
    j = j[j["publication"].isin(LETTRES) & (j["ts"] >= START) & ~_flag(j["repeat"])].sort_values("ts")
    j["inv_20"] = -pd.to_numeric(j["ex_20"], errors="coerce")
    j["inv_5"] = -pd.to_numeric(j["ex_5"], errors="coerce")

    out["global"] = _stats(j["inv_20"])
    out["global_5j"] = _stats(j["inv_5"])
    out["par_lettre"] = {p: {"20j": _stats(g["inv_20"]), "5j": _stats(g["inv_5"]), "avis_recus": int(len(g))}
                         for p, g in j.groupby("publication")}
    out["avis_recus"] = int(len(j))
    measured = j[j["inv_20"].notna()]
    if len(measured) >= N_VERDICT:
        s = _stats(measured["inv_20"].iloc[:N_VERDICT])
        ok = s["moyenne"] > COUT_ESTIME and s["t"] is not None and s["t"] >= T_VERDICT
        out["statut"] = "termine"
        out["verdict"] = {"resultat": "confirme" if ok else "infirme", **s,
                          "moyenne_nette": s["moyenne"] - COUT_ESTIME,
                          "dernier_avis_compte": measured["ts"].iloc[N_VERDICT - 1].isoformat()}
    out["avis"] = [{"date": r.ts.strftime("%Y-%m-%d"), "lettre": r.publication.replace(" (Seeking Alpha)", ""),
                    "ticker": r.ticker, "sens_avis": "achat" if r.side == "long" else "vente",
                    "inverse_5j": None if pd.isna(r.inv_5) else float(r.inv_5),
                    "inverse_20j": None if pd.isna(r.inv_20) else float(r.inv_20)}
                   for r in j.itertuples()]
    return out


def main():
    out = build()
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    g = out["global"]
    print(f"Test inverse Tech Daily + Stock Ideas : {out.get('avis_recus', 0)} avis depuis {START.date()}, "
          f"{g['n']}/{N_VERDICT} mesures a 20 seances, statut {out['statut']}"
          + (f", verdict {out['verdict']['resultat']}" if out["verdict"] else ""))


if __name__ == "__main__":
    main()
