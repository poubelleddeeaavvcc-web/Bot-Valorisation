"""Bot #46 Commandes publiques (2026-10-14): buys the US federal contractors whose government order book grows more
than expected ("unexpected government receivables", TenderAlpha/FactSet 2013-2020: Sharpe 0.77-1.27; government
sales dependence predicts returns, Parajuli & Pontiff, Review of Asset Pricing Studies 2023). Sixth single-idea
bot (with #41-45), fills/accounting by sleeve_book.py (close of the first session after the decision, total return).

Backtest (scratchpad study usa2_*, 2026-10-14), on a hand-checked list of ~70 listed contractors, subsidiaries
included through keyword search of the recipient name (Electric Boat -> General Dynamics, CACI NSS -> CACI...):
monthly contract obligations from USAspending (new awards, modifications AND de-obligations), used only 3 months
later (the Defense Department publishes with a 90-day delay); G = obligations of the last 12 known months; SGR =
G / market cap; UGR = (SGR - mean of its 12 previous monthly values) / their std. Monthly terciles, 2013-2026:
top - bottom +5.7 %/yr (t=1.5), positive in 2013-19 and 2020-26; bottom - all contractors -3.2 %/yr -> rising
receivables are good news, falling ones bad. Net of the 1 EUR sale fee with this bot's rules: +2.3 %/yr over the
equal-weight contractor list (t=1.2). The edge is plausible but NOT proven: this bot is its out-of-sample test.

Rules, fixed:
  - once a month, refresh every keyword's monthly obligations (48 months, USAspending spending_over_time, at most
    MAX_CALLS_PER_RUN calls per run -- the refresh spreads over a few runs, ~110 calls in all);
  - rank the eligible contractors (>= 24 months with obligations, positive G, price and market cap known) by UGR;
  - target = names in the top third, plus names already held that are still above the median (buffer: halves
    the fees, 39 sales a year instead of 56 in the backtest); equal weight for new buys, held lines not resized;
  - sell the rest at the next close. 5,000 EUR so that ~23 lines of ~200 EUR keep the 1 EUR fee near 0.8 %/yr.
The bottom third is tracked as a shadow series (no money) to watch the direction of the effect live.
"""
# Alias lisibilite (mapping perso) : G1 -- strategie Commandes publiques
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.sleeve_book import Book, now_utc  # noqa: E402

STARTING_CAPITAL = 5000.0
ENTER_Q, EXIT_Q = 2 / 3, 0.5
LAG_MONTHS = 3
HISTORY_MONTHS = 48
MIN_HISTORY = 24
MAX_CALLS_PER_RUN = 40
API = "https://api.usaspending.gov/api/v2/search/spending_over_time/"
SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json", "User-Agent": "TV-strategies-personal-project"})

# ticker: [(recipient keyword, first month 'YYYY-MM' or None, last month or None)] -- date limits when a
# subsidiary changed hands (old SAIC -> Leidos, Sikorsky -> Lockheed 11/2015, Jacobs gov services -> Amentum)
CONTRACTORS = {
    "LMT": [("LOCKHEED MARTIN", None, None), ("SIKORSKY", "2015-11", None)],
    "NOC": [("NORTHROP GRUMMAN", None, None)],
    "GD": [("GENERAL DYNAMICS", None, None), ("ELECTRIC BOAT", None, None), ("BATH IRON WORKS", None, None),
           ("NATIONAL STEEL AND SHIPBUILDING", None, None), ("GULFSTREAM AEROSPACE", None, None)],
    "HII": [("HUNTINGTON INGALLS", None, None), ("NEWPORT NEWS SHIPBUILDING", None, None)],
    "BA": [("BOEING", None, None)],
    "LHX": [("L3HARRIS", None, None), ("HARRIS CORPORATION", None, None), ("EXELIS", "2015-06", None),
            ("L-3 COMMUNICATIONS", "2019-07", None), ("L3 TECHNOLOGIES", "2019-07", None)],
    "TXT": [("TEXTRON", None, None), ("BELL TEXTRON", None, None), ("AAI CORPORATION", None, None)],
    "BAH": [("BOOZ ALLEN", None, None)],
    "CACI": [("CACI", None, None)],
    "SAIC": [("SCIENCE APPLICATIONS INTERNATIONAL", "2013-10", None), ("ENGILITY", "2019-01", None)],
    "LDOS": [("LEIDOS", None, None), ("SCIENCE APPLICATIONS INTERNATIONAL", None, "2013-09")],
    "PSN": [("PARSONS GOVERNMENT SERVICES", None, None), ("PARSONS CORPORATION", None, None)],
    "KBR": [("KBR", None, None), ("WYLE", "2016-07", None)],
    "VVX": [("VECTRUS", None, None), ("V2X", None, None), ("VERTEX AEROSPACE", "2022-07", None)],
    "MMS": [("MAXIMUS", None, None)],
    "ICFI": [("ICF INCORPORATED", None, None), ("ICF MACRO", None, None), ("ICF INTERNATIONAL", None, None)],
    "TTEK": [("TETRA TECH", None, None)],
    "J": [("JACOBS ENGINEERING", None, "2024-08"), ("JACOBS TECHNOLOGY", None, "2024-08"),
          ("JACOBS GOVERNMENT", None, "2024-08"), ("CH2M", "2017-12", "2024-08")],
    "ACM": [("AECOM", None, None)],
    "FLR": [("FLUOR", None, None)],
    "DLHC": [("DLH", None, None)],
    "TLS": [("TELOS CORPORATION", None, None)],
    "WYY": [("WIDEPOINT", None, None)],
    "UIS": [("UNISYS", None, None)],
    "PLTR": [("PALANTIR", None, None)],
    "VSEC": [("VSE CORPORATION", None, None)],
    "KTOS": [("KRATOS", None, None)],
    "AVAV": [("AEROVIRONMENT", None, None)],
    "MRCY": [("MERCURY SYSTEMS", None, None)],
    "CW": [("CURTISS-WRIGHT", None, None)],
    "TDY": [("TELEDYNE", None, None)],
    "OSK": [("OSHKOSH", None, None)],
    "AXON": [("AXON ENTERPRISE", None, None)],
    "GEO": [("GEO GROUP", None, None)],
    "CXW": [("CORECIVIC", None, None)],
    "SIGA": [("SIGA TECHNOLOGIES", None, None)],
    "EBS": [("EMERGENT BIO", None, None)],
    "NVAX": [("NOVAVAX", None, None)],
    "BCRX": [("BIOCRYST", None, None)],
    "PESI": [("PERMA-FIX", None, None)],
    "AMRC": [("AMERESCO", None, None)],
    "IRDM": [("IRIDIUM", None, None)],
    "VSAT": [("VIASAT", None, None)],
    "RKLB": [("ROCKET LAB", None, None)],
    "LUNR": [("INTUITIVE MACHINES", None, None)],
    "RDW": [("REDWIRE", None, None)],
    "PL": [("PLANET LABS", None, None)],
    "BWXT": [("BWX", None, None)],
    "DCO": [("DUCOMMUN", None, None)],
    "AIR": [("AAR AIRLIFT", None, None), ("AAR CORP", None, None), ("AAR GOVERNMENT SERVICES", None, None)],
    "NPK": [("NATIONAL PRESTO", None, None), ("AMTEC CORPORATION", None, None)],
    "OSIS": [("OSI SYSTEMS", None, None), ("RAPISCAN", None, None)],
    "GVA": [("GRANITE CONSTRUCTION", None, None)],
    "TPC": [("TUTOR PERINI", None, None), ("PERINI MANAGEMENT", None, None)],
    "ORN": [("ORION MARINE", None, None)],
    "HUM": [("HUMANA GOVERNMENT", None, None)],
    "CNC": [("HEALTH NET FEDERAL", "2016-03", None)],
    "NNI": [("NELNET", None, None)],
    "NAVI": [("NAVIENT", None, None)],
    "CNDT": [("CONDUENT", None, None)],
    "ATRO": [("ASTRONICS", None, None)],
    "ESP": [("ESPEY", None, None)],
    "BKTI": [("BK TECHNOLOGIES", None, None)],
    "FEIM": [("FREQUENCY ELECTRONICS", None, None)],
    "KVHI": [("KVH INDUSTRIES", None, None)],
    "ISSC": [("INNOVATIVE SOLUTIONS & SUPPORT", None, None)],
}


def fetch_keyword(keyword: str, start: str, end: str):
    """Monthly federal contract obligations (award types A-D) of recipients matching `keyword`, as
    {'YYYY-MM': amount} in calendar months, or None if the API keeps failing."""
    body = {"group": "month", "filters": {"recipient_search_text": [keyword], "award_type_codes": ["A", "B", "C", "D"],
                                          "time_period": [{"start_date": start, "end_date": end}]}}
    for k in range(4):
        try:
            r = SESSION.post(API, data=json.dumps(body), timeout=120)
            if r.status_code == 200:
                out = {}
                for x in r.json()["results"]:
                    fy, m = int(x["time_period"]["fiscal_year"]), int(x["time_period"]["month"])  # fiscal month 1 = October
                    y, cm = (fy - 1, m + 9) if m <= 3 else (fy, m - 3)
                    out[f"{y}-{cm:02d}"] = float(x["aggregated_amount"] or 0.0)
                return out
        except requests.RequestException:
            pass
        time.sleep(10 * (k + 1))
    return None


def monthly_obligations(store: dict, months: pd.PeriodIndex) -> pd.DataFrame:
    O = pd.DataFrame(0.0, index=months, columns=list(CONTRACTORS))
    for tic, kws in CONTRACTORS.items():
        for kw, a, b in kws:
            ser = pd.Series(store.get(f"{tic}|{kw}", {}), dtype=float)
            if ser.empty:
                continue
            ser.index = pd.PeriodIndex(ser.index, freq="M")
            ser = ser.reindex(months).fillna(0.0)
            if a:
                ser[ser.index < pd.Period(a, "M")] = 0.0
            if b:
                ser[ser.index > pd.Period(b, "M")] = 0.0
            O[tic] += ser
    return O


def market_caps(tickers) -> dict:
    caps = {}
    for t in tickers:
        try:
            fi = yf.Ticker(t).fast_info
            cap = fi.get("marketCap") if hasattr(fi, "get") else getattr(fi, "market_cap", None)
            if cap:
                caps[t] = float(cap)
        except Exception:
            continue
    return caps


def signal(store: dict, now: pd.Timestamp):
    """UGR of each eligible contractor, as of the last complete month, plus month-end closes (for the shadow)."""
    last = pd.Period(now.strftime("%Y-%m"), "M") - 1
    months = pd.period_range(last - HISTORY_MONTHS + 1, last, freq="M")
    O = monthly_obligations(store, months)
    px = yf.download(list(CONTRACTORS), period="3y", interval="1d", auto_adjust=False, progress=False, threads=True)
    close = px["Close"]; adj = px["Adj Close"]
    cm = close.resample("ME").last(); cm.index = cm.index.to_period("M")
    am = adj.resample("ME").last(); am.index = am.index.to_period("M")
    caps = market_caps([t for t in CONTRACTORS if t in cm.columns and cm[t].notna().any()])
    G = O.rolling(12).sum().shift(LAG_MONTHS)
    known = O.ne(0).cumsum().shift(LAG_MONTHS)
    rows = {}
    for t in O.columns:
        if t not in caps or t not in cm.columns:
            continue
        c = cm[t].dropna()
        if len(c) < 14 or last not in c.index:
            continue
        mcap = caps[t] * c / c.iloc[-1]
        sgr = (G[t] / mcap.reindex(G.index)).loc[last - 12:last]
        if sgr.isna().any() or known.at[last, t] < MIN_HISTORY or G.at[last, t] <= 0:
            continue
        hist = sgr.iloc[:-1]
        if hist.std() <= 0:
            continue
        rows[t] = {"ugr": float((sgr.iloc[-1] - hist.mean()) / hist.std()), "sgr": float(sgr.iloc[-1]),
                   "g_usd": float(G.at[last, t]), "mcap_usd": caps[t]}
    df = pd.DataFrame(rows).T
    if len(df):
        df["q"] = df["ugr"].rank(pct=True)
    return df.sort_values("ugr", ascending=False) if len(df) else df, am


def update_shadow(st: dict, am: pd.DataFrame, now: pd.Timestamp):
    """Equal-weight return, over the last complete month, of the previous ranking's thirds (no money)."""
    prev = st.get("last_ranking")
    month = pd.Period(now.strftime("%Y-%m"), "M") - 1
    if not prev or prev.get("for_month") != str(month) or month not in am.index or month - 1 not in am.index:
        return
    r = am.loc[month] / am.loc[month - 1] - 1
    out = {"month": str(month)}
    for k in ("haut", "bas", "univers"):
        x = r.reindex(prev[k]).dropna()
        out[k] = float(x.mean()) if len(x) else None
    st["shadow"] = [s for s in st.get("shadow", []) if s["month"] != out["month"]] + [out]


def rebalance(book: Book, ranked: pd.DataFrame, now: pd.Timestamp):
    st = book.state
    st["pending"] = []
    held = set(book.open_lots()["ticker"])
    q = ranked["q"].to_dict()
    target = {t for t in ranked.index if q[t] > ENTER_Q} | {t for t in held if q.get(t, 0) > EXIT_Q}
    for t in sorted(held - target):
        why = f"surprise du carnet de commandes publiques retombee (rang {q[t]:.0%})" if t in q else "plus classe ce mois-ci"
        book.add_order("sell", t, "pool", why, now, fraction=1.0)
    new = [t for t in ranked.index if t in target and t not in held]
    free = st["cash"]["pool"] + sum(book.position_value(ticker=t) for t in held - target)
    amount = book.equity() / max(len(target), 1)
    for t in new:
        if free < amount * 0.5:
            break
        a = min(amount, free)
        r = ranked.loc[t]
        book.add_order("buy", t, "pool", f"carnet public en hausse inattendue (UGR {r['ugr']:+.2f}, rang {r['q']:.0%})", now,
                       amount_eur=a, name=t, sector="Sous-traitants de l'Etat federal", country="United States",
                       currency="USD", fractional=True, score=round(float(r["ugr"]), 3))
        free -= a
    n = len(ranked)
    st["last_ranking"] = {"for_month": now.strftime("%Y-%m"),
                          "haut": [t for t in ranked.index if q[t] > 2 / 3], "bas": [t for t in ranked.index if q[t] <= 1 / 3],
                          "univers": list(ranked.index)}
    st["ranking"] = [{"ticker": t, "ugr": round(float(r.ugr), 3), "rang": round(float(r.q), 3),
                      "contrats_12m_musd": round(float(r.g_usd) / 1e6, 1), "dependance_pct": round(float(r.sgr) * 100, 2)}
                     for t, r in ranked.iterrows()]
    print(f"Revision {now.strftime('%Y-%m')} : {n} sous-traitants classes, cible {len(target)} lignes, "
          f"{len(held - target)} vente(s), {len(new)} achat(s) envisage(s)")


def main():
    book = Book("govcontracts", "Bot #46 Commandes publiques", STARTING_CAPITAL, {"pool": STARTING_CAPITAL})
    st = book.state
    st.setdefault("obligations", {})
    book.load_prices()
    book.mark_to_market()
    book.execute_pending()
    book.mark_to_market()

    now = now_utc()
    key = now.strftime("%Y-%m")
    if st.get("rebalance_key") != key:
        if st.get("fetch_key") != key:
            st["fetch_key"], st["fetched"] = key, []
        last = pd.Period(key, "M") - 1
        start = (last - HISTORY_MONTHS + 1).start_time.strftime("%Y-%m-%d")
        end = last.end_time.strftime("%Y-%m-%d")
        todo = [f"{t}|{kw}" for t, kws in CONTRACTORS.items() for kw, _, _ in kws if f"{t}|{kw}" not in st["fetched"]]
        for k in todo[:MAX_CALLS_PER_RUN]:
            ser = fetch_keyword(k.split("|", 1)[1], start, end)
            if ser is None:
                print(f"  USAspending indisponible pour {k} -- nouvel essai au prochain passage")
                continue
            st["obligations"][k] = ser
            st["fetched"].append(k)
            time.sleep(0.5)
        left = len(todo) - min(len(todo), MAX_CALLS_PER_RUN)
        print(f"Commandes publiques {key} : {len(st['fetched'])} mots-cles a jour, {left} restant(s)")
        if all(f"{t}|{kw}" in st["fetched"] for t, kws in CONTRACTORS.items() for kw, _, _ in kws):
            ranked, am = signal(st["obligations"], now)
            update_shadow(st, am, now)
            if len(ranked) >= 15:
                rebalance(book, ranked, now)
                st["rebalance_key"] = key
            else:
                print(f"  seulement {len(ranked)} sous-traitants classables -- revision reportee")
    shadow = st.get("shadow", [])
    book.save({"rebalance_key": st.get("rebalance_key"), "n_ranked": len(st.get("ranking", [])),
               "shadow_months": len(shadow), "shadow_last": shadow[-1] if shadow else None})


if __name__ == "__main__":
    main()
