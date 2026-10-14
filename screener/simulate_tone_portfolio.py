"""Bot #49 Ton des communiques (2026-10-14): reads the earnings press release (8-K item 2.02, exhibit 99.1) of every
S&P 500 company the day it is filed and measures its ABNORMAL tone -- the optimism that the numbers do not explain
(Huang, Teoh & Zhang, The Accounting Review 2014, 'Tone Management'). Ninth single-idea bot (with #41-48),
fills/accounting by sleeve_book.py (close of the first session after the decision, total return).

The study (annual releases 1997-2007): abnormally optimistic releases are followed by NEGATIVE returns, the market
being first misled then correcting: ranked abnormal tone, top - bottom = -2.44 % over CR[+2,+61], -4.70 % over
CR[+2,+121]. Our replication on S&P 500 releases 2014-2026 (scratchpad audit_tone.py, 2026-10-14): same sign but
~1/5 of the size and not significant (-0.42 % over 61 sessions, t=-0.9; -1.0 % in 2020-26). Long only, the bot
holds the LEAST abnormally optimistic fifth (the study's long leg) and tracks the most optimistic fifth as a
shadow list (no money) to watch the direction live.

Rules: TONE = (positive - negative words, a positive word right after no/not/none/neither/never/nobody counts as
negative) / words, financial word lists of the backtest; ABTONE = residual of an OLS of TONE on the prior 12-month
return, log market cap, monthly return volatility, the 2-session announcement reaction vs SPY (the earnings-news
proxy; no Compustat here) and log(1 + years of price history), fitted on the last 12 months of releases (seeded
with the backtest's, data/strategies/tone_seed.json); the release's rank among those residuals decides: bottom
20 % -> buy POSITION_EUR at the close after the reaction session (CR[+2, ...] of the study), hold HOLD_BDAYS
sessions; top 20 % -> shadow list. 5,000 EUR in 100 EUR lines, at most MAX_POSITIONS.
"""
# Alias lisibilite (mapping perso) : E1 -- strategie Ton des communiques
import io
import json
import pathlib
import re
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener import sec_daily  # noqa: E402
from screener.sleeve_book import Book, now_utc  # noqa: E402

STARTING_CAPITAL = 5000.0
POSITION_EUR = 100.0
MAX_POSITIONS = 50
HOLD_BDAYS = 60
LOW_Q, HIGH_Q = 0.2, 0.8
POOL_DAYS = 365
FIRST_FILING_DAY = pd.Timestamp("2026-10-14")
MAX_DAYS_PER_RUN = 2
NY = ZoneInfo("America/New_York")
SEED = HERE / "data/strategies/tone_seed.json"
FEATURES = ["ret12", "size", "stdret", "react", "age"]
WORD = re.compile(r"[a-z]+")
NEGATORS = {"no", "not", "none", "neither", "never", "nobody"}
POS = set("""achieve achieved achievement attractive beneficial benefit benefited better boost boosted breakthrough
enhance enhanced exceed exceeded exceeding exceeds excellent exceptional expand expanded expanding expansion favorable
gain gains grew growing growth highest improve improved improvement improvements improving increase increased increases
increasing momentum outperform outperformed outperforming positive profitable profitability progress raise raised
raising record records robust solid strength strengthen strengthened strong stronger strongest success successful
surpassed upside accelerate accelerated accelerating opportunity opportunities resilient resilience""".split())
NEG = set("""adverse adversely challenge challenges challenging decline declined declines declining decrease decreased
decreases decreasing deficit delay delayed delays deteriorate deteriorated deterioration difficult disappointing downturn
drop dropped falls fell headwind headwinds impairment impairments loss losses negative negatively pressure pressures
reduce reduced reduction reductions restructuring shortfall slow slowdown slowed slower soft softer softness uncertain
uncertainty uncertainties unfavorable weak weaker weakness worse writedown layoffs litigation""".split())


def tone(text: str):
    w = WORD.findall(text[:60_000])               # the narrative is at the top; tables follow
    if len(w) < 200:
        return None
    pos = neg = 0
    for k, x in enumerate(w):
        if x in POS:
            if k and w[k - 1] in NEGATORS:
                neg += 1
            else:
                pos += 1
        elif x in NEG:
            neg += 1
    return (pos - neg) / len(w)


def sp500_by_cik() -> dict:
    sp = pd.read_html(io.StringIO((HERE / "data/universe/sp500_wiki.html").read_text(encoding="utf-8")))[0]
    cikmap = json.loads((HERE / "data/universe/sec_ticker_cik.json").read_text(encoding="utf-8"))
    out = {}
    for t in sp["Symbol"].astype(str):
        cik = cikmap.get(t) or cikmap.get(t.replace(".", "-")) or cikmap.get(t.replace(".", ""))
        if cik:
            out[int(cik)] = t.replace(".", "-")
    return out


def features(ticker: str, accepted: pd.Timestamp):
    """Reaction session and the regressors, once that session has closed; None if not yet / no data."""
    px = yf.download([ticker, "SPY"], start=(accepted - pd.Timedelta(days=400)).strftime("%Y-%m-%d"),
                     auto_adjust=False, progress=False, threads=False)
    if not len(px):
        return None
    a = px["Adj Close"].dropna()
    if ticker not in a.columns or len(a) < 260:
        return None
    day = pd.Timestamp(accepted.tz_convert(NY).date())
    sessions = a.index[a.index >= day]
    if accepted.tz_convert(NY).hour >= 16:
        sessions = sessions[sessions > day]
    if not len(sessions):
        return None
    d0 = sessions[0]
    if pd.Timestamp.now(tz=NY) < pd.Timestamp(d0.date()).tz_localize(NY) + pd.Timedelta(hours=17):
        return None                                # reaction session not closed yet
    i = a.index.get_loc(d0)
    s, p = a["SPY"].values, a[ticker].values
    mret = pd.Series(p[i - 253:i - 1]).iloc[::21].pct_change().dropna()
    cap = yf.Ticker(ticker).fast_info
    mcap = cap.get("marketCap") if hasattr(cap, "get") else getattr(cap, "market_cap", None)
    first = yf.download(ticker, start="2005-12-01", auto_adjust=False, progress=False, threads=False)
    years = (d0 - first.index[0]).days / 365.25 if len(first) else 10.0
    if not mcap:
        return None
    return {"day0": d0.strftime("%Y-%m-%d"), "ret12": p[i - 2] / p[i - 253] - 1, "size": float(np.log(mcap)),
            "stdret": float(mret.std()), "react": (p[i] / p[i - 2] - 1) - (s[i] / s[i - 2] - 1), "age": float(np.log1p(years))}


def abnormal(pool: list, x: dict):
    df = pd.DataFrame(pool).dropna(subset=FEATURES + ["tone"])
    X = np.column_stack([np.ones(len(df)), df[FEATURES].values])
    beta, *_ = np.linalg.lstsq(X, df["tone"].values, rcond=None)
    res = df["tone"].values - X @ beta
    ab = x["tone"] - float(np.array([1.0] + [x[f] for f in FEATURES]) @ beta)
    return ab, float((res <= ab).mean())


def main():
    book = Book("tone", "Bot #49 Ton des communiques", STARTING_CAPITAL, {"pool": STARTING_CAPITAL})
    st = book.state
    if "pool" not in st:
        st["pool"] = json.loads(SEED.read_text(encoding="utf-8"))
    for k in ("edgar_days_done", "awaiting", "shadow", "events_log"):
        st.setdefault(k, [])
    st.setdefault("hold_until", {})
    st.setdefault("counters", {"jours": 0, "communiques": 0, "evalues": 0, "achetes": 0, "ombre": 0, "complet": 0})
    book.load_prices()
    book.mark_to_market()
    book.execute_pending()
    book.mark_to_market()
    now = now_utc()
    today = pd.Timestamp(now.tz_convert(NY).date())

    held = set(book.open_lots()["ticker"])
    pend_s = {o["ticker"] for o in st["pending"] if o["side"] == "sell"}
    pend_b = {o["ticker"] for o in st["pending"] if o["side"] == "buy"}
    st["hold_until"] = {t: d for t, d in st["hold_until"].items() if t in held or t in pend_b}
    for t in sorted(held - pend_s):
        u = st["hold_until"].get(t)
        if u and pd.Timestamp(u) <= today:
            book.add_order("sell", t, "pool", f"detention de {HOLD_BDAYS} seances terminee", now, fraction=1.0)

    # 1. new earnings releases of the S&P 500 members (daily index of day D, complete the next morning)
    members = sp500_by_cik()
    todo = [d for d in pd.bdate_range(FIRST_FILING_DAY, today - pd.Timedelta(days=1)) if d.strftime("%Y-%m-%d") not in st["edgar_days_done"]]
    for day in todo[:MAX_DAYS_PER_RUN]:
        rows = sec_daily.daily_filings(day, ("8-K",))
        if rows is None:
            if (today - day).days > 4:
                st["edgar_days_done"].append(day.strftime("%Y-%m-%d"))
            continue
        for r in rows:
            if r["cik"] not in members:
                continue
            ix = sec_daily.filing_index(r["cik"], r["acc"])
            if "2.02" not in sec_daily.items_of(ix):
                continue
            url = sec_daily.document_link(ix, r["cik"], r["acc"], r"EX-99(?:\.1|\.01)?")
            txt = sec_daily.document_text(url) if url else None
            tn = tone(txt) if txt else None
            acc_at = sec_daily.accepted_at(ix)
            if tn is None or acc_at is None:
                continue
            st["awaiting"].append({"ticker": members[r["cik"]], "acc": r["acc"], "accepted": acc_at.isoformat(), "tone": tn,
                                   "company": r["company"]})
            st["counters"]["communiques"] += 1
        st["counters"]["jours"] += 1
        st["edgar_days_done"].append(day.strftime("%Y-%m-%d"))

    # 2. releases whose reaction session has closed: abnormal tone, rank, decision
    left = []
    for ev in st["awaiting"]:
        x = features(ev["ticker"], pd.Timestamp(ev["accepted"]))
        if x is None:
            if now - pd.Timestamp(ev["accepted"]) < pd.Timedelta(days=10):
                left.append(ev)
            continue
        x["tone"] = ev["tone"]
        ab, q = abnormal(st["pool"], x)
        st["pool"].append({"ticker": ev["ticker"], "date": x["day0"], **{k: round(float(x[k]), 6) for k in FEATURES + ["tone"]}})
        st["counters"]["evalues"] += 1
        decision = "neutre"
        if q <= LOW_Q:
            lots, buys = book.open_lots(), [o for o in st["pending"] if o["side"] == "buy"]
            until = (pd.Timestamp(x["day0"]) + pd.offsets.BDay(HOLD_BDAYS + 1)).strftime("%Y-%m-%d")
            if ev["ticker"] in set(lots["ticker"]) | {o["ticker"] for o in buys}:
                st["hold_until"][ev["ticker"]] = max(st["hold_until"].get(ev["ticker"], until), until)
                decision = "deja detenu : detention prolongee"
            elif len(set(lots["ticker"])) + len(buys) >= MAX_POSITIONS or st["cash"]["pool"] - sum(o["amount_eur"] for o in buys) < POSITION_EUR:
                st["counters"]["complet"] += 1
                decision = "portefeuille complet"
            else:
                book.add_order("buy", ev["ticker"], "pool", f"ton peu optimiste au regard des chiffres (rang {q:.0%})", now,
                               amount_eur=POSITION_EUR, name=ev["company"], sector="Communiques de resultats",
                               country="United States", currency="USD", fractional=True, score=round(ab, 6))
                st["hold_until"][ev["ticker"]] = until
                st["counters"]["achetes"] += 1
                decision = "achat"
        elif q > HIGH_Q:
            st["shadow"].append({"ticker": ev["ticker"], "day0": x["day0"], "rang": round(q, 3)})
            st["counters"]["ombre"] += 1
            decision = "ombre (ton trop optimiste)"
        st["events_log"].append({"ticker": ev["ticker"], "company": ev["company"], "day0": x["day0"], "tone": round(ev["tone"], 5),
                                 "abtone": round(ab, 5), "rang": round(q, 3), "decision": decision})
        print(f"    {ev['ticker']:6} {ev['company'][:28]:28} ton {ev['tone']:+.4f} anormal {ab:+.4f} rang {q:.0%} -> {decision}")
    st["awaiting"] = left
    cut = (today - pd.Timedelta(days=POOL_DAYS)).strftime("%Y-%m-%d")
    st["pool"] = [p for p in st["pool"] if p["date"] >= cut]
    st["events_log"] = st["events_log"][-400:]
    st["shadow"] = st["shadow"][-600:]
    st["edgar_days_done"] = sorted(set(st["edgar_days_done"]))
    book.save({"counters": st["counters"], "pool_size": len(st["pool"]), "awaiting": len(st["awaiting"]),
               "max_positions": MAX_POSITIONS, "position_eur": POSITION_EUR})


if __name__ == "__main__":
    main()
