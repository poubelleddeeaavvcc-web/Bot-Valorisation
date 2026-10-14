"""Bot #48 Copie des elus (2026-10-14): copies the stock PURCHASES that members of the US House of Representatives
disclose in their periodic transaction reports (STOCK Act, filed up to 45 days after the trade). Eighth single-idea
bot (with #41-47), fills/accounting by sleeve_book.py (close of the first session after the decision, total return).

This is a TRACKER, not a bet: the backtest (scratchpad house_*, 2026-10-14) found no edge. 18,408 purchases
2014-2026 (Clerk PDFs, median filing delay 28 days), entry after the filing date, vs other stocks on the same date
with a similar prior move: -0.72 % at 120 sessions (p=0.007; 2014-19 -0.15 % n.s., 2020-26 -1.06 %), in line with
Belmont, Sacerdote, Sehgal & Van Hoek (J. Public Economics 2022: -26 bp at 6 months, 2012-2020). The large gains
of congressional LEADERS (Wei & Zhou, NBER 2025: +47 points a year once in leadership, 1995-2021) could not be
checked -- 66 leader purchases of listed stocks in our data. The user asked for one bot per idea (2026-10-14):
this one shows live whether copying Congress helps or hurts, and logs every disclosed purchase.

Rules = the backtest's: every new periodic transaction report of the current year's index (Clerk zip, refreshed
daily), purchases ('P') of stocks ([ST] or untagged lines with a ticker); one event per member x ticker x report;
buy POSITION_EUR at the next close, hold HOLD_BDAYS sessions; a new purchase of a held stock extends the holding;
at most MAX_POSITIONS lines (first come, first served). 5,000 EUR in 100 EUR lines, like #45.
"""
# Alias lisibilite (mapping perso) : H1 -- strategie Copie des elus
import io
import pathlib
import re
import sys
import time
import zipfile

import pandas as pd
import requests
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.sec_edgar import SEC_HEADERS  # noqa: E402  (identified User-Agent)
from screener.sleeve_book import Book, now_utc  # noqa: E402

STARTING_CAPITAL = 5000.0
POSITION_EUR = 100.0
MAX_POSITIONS = 50
HOLD_BDAYS = 120
MAX_DOCS_PER_RUN = 40
FIRST_FILING_DAY = pd.Timestamp("2026-10-13")
BASE = "https://disclosures-clerk.house.gov/public_disc"
# (TICKER) [ST] P|S|S (partial)|E  trade-date  notification-date  $amount-low  -- same pattern as the backtest
RX = re.compile(r"\(([A-Za-z][A-Za-z.\-]{0,6})\)\s*(?:\[ST\]\s*)?(P|S \(partial\)|S|E)\s+(\d{1,2}/\d{1,2}/\d{4})\s+"
                r"(\d{1,2}/\d{1,2}/\d{4})\s+\$([\d,]+)", re.I)
SESSION = requests.Session()
SESSION.headers.update(SEC_HEADERS)


def get(url):
    for k in range(3):
        try:
            r = SESSION.get(url, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(5 * (k + 1))
    return None


def new_reports(year: int, done: set) -> list:
    r = get(f"{BASE}/financial-pdfs/{year}FD.zip")
    if r is None:
        return []
    x = zipfile.ZipFile(io.BytesIO(r.content)).read(f"{year}FD.xml").decode("utf-8-sig", "ignore")
    out = []
    for m in re.findall(r"<Member>(.*?)</Member>", x, re.S):
        g = lambda tag: (re.search(f"<{tag}>(.*?)</{tag}>", m, re.S) or [None, ""])[1].strip()  # noqa: E731
        if g("FilingType") != "P" or g("DocID") in done:
            continue
        fd = pd.to_datetime(g("FilingDate"), format="%m/%d/%Y", errors="coerce")
        if pd.isna(fd) or fd < FIRST_FILING_DAY:
            continue
        out.append({"doc": g("DocID"), "member": f"{g('First')} {g('Last')}".strip(), "district": g("StateDst"),
                    "filing_date": fd, "year": year})
    return sorted(out, key=lambda d: d["filing_date"])


def purchases(rep: dict):
    """Stock purchases parsed from the report PDF; None if the PDF can't be read (scanned paper filing)."""
    import fitz  # PyMuPDF, only needed here
    r = get(f"{BASE}/ptr-pdfs/{rep['year']}/{rep['doc']}.pdf")
    if r is None or not r.content.startswith(b"%PDF"):
        return None
    try:
        text = re.sub(r"\s+", " ", "".join(p.get_text() for p in fitz.open(stream=r.content, filetype="pdf")))
    except Exception:
        return None
    rows = []
    for tk, typ, td, nd, amt in RX.findall(text):
        if typ.upper() == "P":
            rows.append({"ticker": tk.upper().replace(".", "-"), "trade_date": td, "amount_low": int(amt.replace(",", ""))})
    return rows


def main():
    book = Book("congress", "Bot #48 Copie des elus", STARTING_CAPITAL, {"pool": STARTING_CAPITAL})
    st = book.state
    st.setdefault("docs_done", [])
    st.setdefault("hold_until", {})
    st.setdefault("events_log", [])
    st.setdefault("counters", {"rapports": 0, "illisibles": 0, "achats_declares": 0, "achetes": 0, "complet": 0,
                               "prolonges": 0, "sans_cours": 0})
    book.load_prices()
    book.mark_to_market()
    book.execute_pending()
    book.mark_to_market()
    now = now_utc()
    today = pd.Timestamp(now.date())

    held = set(book.open_lots()["ticker"])
    pending_sells = {o["ticker"] for o in st["pending"] if o["side"] == "sell"}
    pending_buys = {o["ticker"] for o in st["pending"] if o["side"] == "buy"}
    st["hold_until"] = {t: d for t, d in st["hold_until"].items() if t in held or t in pending_buys}
    for t in sorted(held - pending_sells):
        until = st["hold_until"].get(t)
        if until and pd.Timestamp(until) <= today:
            book.add_order("sell", t, "pool", f"detention de {HOLD_BDAYS} seances terminee", now, fraction=1.0)

    done = set(st["docs_done"])
    reps = []
    for y in sorted({today.year, (today - pd.Timedelta(days=45)).year}):
        reps += new_reports(y, done)
    for rep in reps[:MAX_DOCS_PER_RUN]:
        rows = purchases(rep)
        st["docs_done"].append(rep["doc"])
        st["counters"]["rapports"] += 1
        if rows is None:
            st["counters"]["illisibles"] += 1
            continue
        events = {}
        for x in rows:
            events.setdefault(x["ticker"], x)            # one event per member x ticker x report
        if events:
            px = yf.download(sorted(events), period="1mo", auto_adjust=False, progress=False, threads=True)
        for tk, x in events.items():
            st["counters"]["achats_declares"] += 1
            try:
                c = px["Close"][tk].dropna() if isinstance(px["Close"], pd.DataFrame) else px["Close"].dropna()
            except Exception:
                c = pd.Series(dtype=float)
            if not len(c):
                decision = "pas de cours Yahoo"
                st["counters"]["sans_cours"] += 1
            else:
                until = (rep["filing_date"] + pd.offsets.BDay(HOLD_BDAYS + 1)).strftime("%Y-%m-%d")
                open_t = set(book.open_lots()["ticker"])
                buys = [o for o in st["pending"] if o["side"] == "buy"]
                if tk in open_t or tk in {o["ticker"] for o in buys}:
                    st["hold_until"][tk] = max(st["hold_until"].get(tk, until), until)
                    st["counters"]["prolonges"] += 1
                    decision = "deja detenu : detention prolongee"
                elif len(open_t) + len(buys) >= MAX_POSITIONS or st["cash"]["pool"] - sum(o["amount_eur"] for o in buys) < POSITION_EUR:
                    st["counters"]["complet"] += 1
                    decision = "portefeuille complet : ignore"
                else:
                    book.add_order("buy", tk, "pool", f"achat declare par {rep['member']} ({rep['district']}), "
                                   f">= {x['amount_low']:,} $ le {x['trade_date']}", now, amount_eur=POSITION_EUR, name=tk,
                                   sector="Achats d'elus", country="United States", currency="USD", fractional=True,
                                   score=float(x["amount_low"]))
                    st["hold_until"][tk] = until
                    st["counters"]["achetes"] += 1
                    decision = "achat"
            st["events_log"].append({"filing_date": rep["filing_date"].strftime("%Y-%m-%d"), "member": rep["member"],
                                     "district": rep["district"], "ticker": tk, "trade_date": x["trade_date"],
                                     "amount_low": x["amount_low"], "decision": decision})
            print(f"    {rep['member'][:24]:24} {tk:6} >= {x['amount_low']:>9,} $ -> {decision}")
        time.sleep(0.5)
    st["events_log"] = st["events_log"][-400:]
    st["docs_done"] = sorted(set(st["docs_done"]))[-5000:]
    book.save({"counters": st["counters"], "max_positions": MAX_POSITIONS, "position_eur": POSITION_EUR,
               "reports_waiting": max(0, len(reps) - MAX_DOCS_PER_RUN)})


if __name__ == "__main__":
    main()
