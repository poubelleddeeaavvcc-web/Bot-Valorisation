"""Bot #50 Rapports inchanges (2026-10-14): 'Lazy Prices' (Cohen, Malloy & Nguyen, Journal of Finance 2020) -- firms
whose annual / quarterly report barely changes from the same report a year earlier do better than firms that
rewrite it (changes hide bad news that investors, who don't read the 200 pages, discover later). Tenth single-idea
bot (with #41-49), fills/accounting by sleeve_book.py (close of the first session after the decision, total return).

The study (all US firms 1995-2014, 10-K and 10-Q): quintiles of year-on-year similarity on the prior year's
distribution, a stock enters the month after its filing and is held 3 months; non-changers minus changers
18-45 bp/month equal-weighted, up to 58 bp value-weighted, 188 bp on the Risk Factors section; most of the spread
comes from the changers' later fall. Our replication on S&P 500 10-Ks 2006-2026 (scratchpad audit_lazy.py,
2026-10-14), with the study's portfolio rules: nothing (whole document -20 to -48 bp/month, Risk Factors +14 bp,
t=0.4) -- the same as an independent public replication on the S&P 500 (Risk Factors +15 bp, p=0.08). The study's
universe is all firms and three quarters of its reports are 10-Qs: this bot follows the study (10-K AND 10-Q, US
firms worth >= MIN_CAP_USD), so it tests the idea where our backtest could not.

Rules: every 10-K / 10-Q of the universe (EDGAR daily index), compared with the same form whose period ended
~1 year earlier (word counts, 3+ letters: cosine similarity); quintile breakpoints from the last 12 months of
similarities of that form (10-K seeded with the backtest's, data/strategies/lazy_seed.json; 10-Q after a warm-up of
MIN_HISTORY filings). On the first run of each month: buy the N_NEW most similar 'non-changers' (top fifth) filed
the previous month, hold 3 months; the N_NEW biggest 'changers' (bottom fifth) go to a shadow list (no money).
5,000 EUR, ~30 lines of ~160 EUR.
"""
# Alias lisibilite (mapping perso) : L1 -- strategie Rapports inchanges (Lazy Prices)
import json
import pathlib
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener import sec_daily  # noqa: E402
from screener.sleeve_book import Book, now_utc  # noqa: E402

STARTING_CAPITAL = 5000.0
N_NEW = 10
HOLD_MONTHS = 3
MIN_CAP_USD = 1e9
MIN_HISTORY = 200
FIRST_FILING_DAY = pd.Timestamp("2026-10-14")
MAX_DAYS_PER_RUN = 2
MAX_FILINGS_PER_RUN = 30
SEED = HERE / "data/strategies/lazy_seed.json"
WORD = re.compile(r"[a-z]{3,}")


def universe() -> dict:
    """CIK -> ticker of US-listed companies worth >= MIN_CAP_USD (Nasdaq screener snapshot of the repo)."""
    scr = json.loads((HERE / "data/universe/nasdaq_screener_full.json").read_text(encoding="utf-8"))["data"]["rows"]
    caps = {r["symbol"].replace("/", "-"): float(r["marketCap"] or 0) for r in scr}
    cikmap = json.loads((HERE / "data/universe/sec_ticker_cik.json").read_text(encoding="utf-8"))
    out = {}
    for t, cik in cikmap.items():
        if caps.get(t, 0) >= MIN_CAP_USD and (int(cik) not in out or len(t) < len(out[int(cik)])):
            out[int(cik)] = t
    return out


def similarity(cik: int, acc: str, form: str):
    """Cosine and Jaccard similarity of a filing's primary document with the same form ~1 year earlier."""
    raw = sec_daily.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    if raw is None:
        return None
    rec = json.loads(raw)["filings"]["recent"]
    df = pd.DataFrame({k: rec[k] for k in ("accessionNumber", "form", "reportDate", "primaryDocument")})
    df = df[df["form"] == form]
    me = df[df["accessionNumber"] == acc]
    if not len(me) or not me["reportDate"].iloc[0]:
        return None
    period = pd.Timestamp(me["reportDate"].iloc[0])
    df = df[df["reportDate"] != ""].assign(rd=lambda d: pd.to_datetime(d["reportDate"], errors="coerce"))
    prev = df[(df["rd"] - (period - pd.DateOffset(years=1))).abs() <= pd.Timedelta(days=25)]
    if not len(prev):
        return None
    docs = []
    for a, d in [(acc, me["primaryDocument"].iloc[0]), (prev["accessionNumber"].iloc[0], prev["primaryDocument"].iloc[0])]:
        txt = sec_daily.document_text(f"https://www.sec.gov/Archives/edgar/data/{cik}/{a.replace('-', '')}/{d}")
        if not txt:
            return None
        docs.append(Counter(WORD.findall(txt)))
    a, b = docs
    if len(a) < 500 or len(b) < 500:
        return None
    common = set(a) & set(b)
    cos = sum(a[w] * b[w] for w in common) / (np.sqrt(sum(v * v for v in a.values())) * np.sqrt(sum(v * v for v in b.values())))
    return float(cos), len(common) / len(set(a) | set(b)), period.strftime("%Y-%m-%d")


def breakpoints(st: dict, form: str):
    vals = [s["cos"] for s in st["scores"] if s["form"] == form]
    if form == "10-K":
        vals += st["seed_10k"]
    if len(vals) < MIN_HISTORY:
        return None
    return float(np.quantile(vals, 0.2)), float(np.quantile(vals, 0.8))


def main():
    book = Book("lazy", "Bot #50 Rapports inchanges", STARTING_CAPITAL, {"pool": STARTING_CAPITAL})
    st = book.state
    if "seed_10k" not in st:
        st["seed_10k"] = json.loads(SEED.read_text(encoding="utf-8"))["10-K"]
    for k in ("edgar_days_done", "queue", "scores", "shadow"):
        st.setdefault(k, [])
    st.setdefault("hold_until", {})
    st.setdefault("counters", {"rapports": 0, "compares": 0, "sans_comparable": 0, "achetes": 0})
    book.load_prices()
    book.mark_to_market()
    book.execute_pending()
    book.mark_to_market()
    now = now_utc()
    today = pd.Timestamp(now.date())

    held = set(book.open_lots()["ticker"])
    pend_s = {o["ticker"] for o in st["pending"] if o["side"] == "sell"}
    for t in sorted(held - pend_s):
        u = st["hold_until"].get(t)
        if u and pd.Timestamp(u) <= today:
            book.add_order("sell", t, "pool", f"detention de {HOLD_MONTHS} mois terminee", now, fraction=1.0)

    # 1. queue the new 10-K / 10-Q filings of the universe, then score a batch
    uni = universe()
    todo = [d for d in pd.bdate_range(FIRST_FILING_DAY, today - pd.Timedelta(days=1)) if d.strftime("%Y-%m-%d") not in st["edgar_days_done"]]
    for day in todo[:MAX_DAYS_PER_RUN]:
        rows = sec_daily.daily_filings(day, ("10-K", "10-Q"))
        if rows is None:
            if (today - day).days > 4:
                st["edgar_days_done"].append(day.strftime("%Y-%m-%d"))
            continue
        st["queue"] += [{"cik": r["cik"], "acc": r["acc"], "form": r["form"], "ticker": uni[r["cik"]], "date": day.strftime("%Y-%m-%d")}
                        for r in rows if r["cik"] in uni]
        st["edgar_days_done"].append(day.strftime("%Y-%m-%d"))
    batch, st["queue"] = st["queue"][:MAX_FILINGS_PER_RUN], st["queue"][MAX_FILINGS_PER_RUN:]
    for f in batch:
        st["counters"]["rapports"] += 1
        res = similarity(f["cik"], f["acc"], f["form"])
        if res is None:
            st["counters"]["sans_comparable"] += 1
            continue
        st["counters"]["compares"] += 1
        st["scores"].append({"ticker": f["ticker"], "form": f["form"], "date": f["date"], "cos": round(res[0], 6),
                             "jac": round(res[1], 5), "period": res[2]})
        print(f"    {f['ticker']:6} {f['form']:5} {f['date']} similarite {res[0]:.5f} (Jaccard {res[1]:.3f})")

    # 2. monthly entries: filings of the previous month, ranked on their form's prior-12-month breakpoints
    key = now.strftime("%Y-%m")
    if st.get("entry_key") != key and today.day >= 2:
        prev_m = (pd.Period(key, "M") - 1)
        month = [s for s in st["scores"] if pd.Period(s["date"][:7], "M") == prev_m]
        hi, lo = [], []
        for s in month:
            bp = breakpoints({**st, "scores": [x for x in st["scores"] if x["date"] < s["date"]]}, s["form"])
            if bp is None:
                continue
            (hi if s["cos"] >= bp[1] else lo if s["cos"] <= bp[0] else []).append(s)
        hi = sorted({s["ticker"]: s for s in sorted(hi, key=lambda s: s["cos"])}.values(), key=lambda s: -s["cos"])[:N_NEW]
        lo = sorted({s["ticker"]: s for s in sorted(lo, key=lambda s: -s["cos"])}.values(), key=lambda s: s["cos"])[:N_NEW]
        until = (pd.Period(key, "M") + HOLD_MONTHS).start_time.strftime("%Y-%m-%d")
        lots = set(book.open_lots()["ticker"])
        amount = STARTING_CAPITAL / (N_NEW * HOLD_MONTHS)
        for s in hi:
            if s["ticker"] in lots:
                st["hold_until"][s["ticker"]] = max(st["hold_until"].get(s["ticker"], until), until)
                continue
            free = st["cash"]["pool"] - sum(o["amount_eur"] for o in st["pending"] if o["side"] == "buy")
            if free < amount * 0.5:
                break
            book.add_order("buy", s["ticker"], "pool", f"{s['form']} quasi inchange (similarite {s['cos']:.5f})", now,
                           amount_eur=min(amount, free), name=s["ticker"], sector="Rapports inchanges",
                           country="United States", currency="USD", fractional=True, score=s["cos"])
            st["hold_until"][s["ticker"]] = until
            st["counters"]["achetes"] += 1
        st["shadow"] += [{"ticker": s["ticker"], "form": s["form"], "month": key, "cos": s["cos"]} for s in lo]
        st["entry_key"] = key
        print(f"Entrees {key} : {len(month)} rapports compares en {prev_m}, {len(hi)} inchanges achetes, {len(lo)} changeants en ombre")
    cut = (today - pd.Timedelta(days=400)).strftime("%Y-%m-%d")
    st["scores"] = [s for s in st["scores"] if s["date"] >= cut]
    st["shadow"] = st["shadow"][-300:]
    st["edgar_days_done"] = sorted(set(st["edgar_days_done"]))
    book.save({"counters": st["counters"], "queue": len(st["queue"]), "scores": len(st["scores"]),
               "entry_key": st.get("entry_key")})


if __name__ == "__main__":
    main()
