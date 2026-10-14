"""Bot #51 Meilleures idees des gerants (2026-10-14): each quarter, from the SEC's Form 13F data sets, finds each
active manager's 'best idea' -- the holding they overweight the most against the other managers' combined book --
and buys the best ideas shared by the most managers (Cohen, Polk & Silli 2010 / Anton, Cohen & Polk, 'Best
Ideas'). Eleventh single-idea bot (with #41-50), fills/accounting by sleeve_book.py.

The study (active US equity mutual funds 1983-2018): best ideas beat the market by ~37 bp/month (six-factor alpha,
t=3.3), the rest of the managers' books by about as much. Our test (scratchpad f13_*, 2013-2026, 209,000 manager x
quarter best ideas of all 13F filers with 10-300 stock positions): -0.37 % per quarter vs other stocks on the same
date with a similar prior 6-month move (p=0.17) -- no edge either way once the method bias is corrected; best-idea
portfolio alpha vs SPY -1.5 %/yr (t=-0.7). Closer to the study (f13_audit.py: best idea = tilt x idiosyncratic
volatility, 25 % most active managers, formed at the quarter end like the study or at the filing date): alpha
-3 to +4 bp/month (study 37); only 'best minus the manager's other top-5 tilts' stays positive (+0.4-0.5 %/quarter,
p=0.06-0.2), a long-short effect. Kept as a live test because the user asked for one bot per idea; the simpler
tilt definition is used here (the information-ratio one needs a year of prices for ~25,000 candidate holdings).

Rules: when the SEC publishes a new quarterly data set (zip, ~70 MB), keep the on-time 13F-HR holdings reports
(originals filed <= 50 days after the quarter end), managers with 10-300 stock positions (options excluded);
tilt = weight in the manager's book - the stock's weight in all those managers' combined book; best idea = the
largest tilt; ACTIVE managers = the 25 % with the largest best-idea tilt (the study keeps its 25 % most active
funds). CUSIP -> ticker through OpenFIGI (US common stocks / ADRs only). Hold the N_HOLD best ideas named by the
most active managers (ties: larger mean tilt), equal weight, price >= $5; a held name stays while it ranks within
KEEP_RANK. The data set comes out ~1 month after the filing deadline, so entries are later than the backtest's
filing-date entries. 2,000 EUR, 25 lines.
"""
# Alias lisibilite (mapping perso) : F1 -- strategie Meilleures idees des gerants (13F)
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

from screener.sec_edgar import SEC_HEADERS  # noqa: E402
from screener.sleeve_book import Book, now_utc  # noqa: E402

STARTING_CAPITAL = 2000.0
N_HOLD, KEEP_RANK = 25, 50
ACTIVE_Q = 0.75
MIN_PRICE_USD = 5.0
PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
FIGI = "https://api.openfigi.com/v3/mapping"
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
SESSION = requests.Session()
SESSION.headers.update(SEC_HEADERS)


def latest_dataset():
    r = SESSION.get(PAGE, timeout=60)
    # the newest files sit under another directory (/files/datastandardsinnovation/... since mid-2026)
    links = sorted(set(re.findall(r'/files/[a-z]+/data/form-13f-data-sets/[^"]+_form13f\.zip', r.text)))

    def end_of(link):                         # "01jun2026-31aug2026_form13f.zip" -> 2026-08-31 ; "2023q4_..." -> 2023-12-31
        m = re.search(r"-(\d{2})([a-z]{3})(\d{4})_form13f", link)
        if m:
            return pd.Timestamp(int(m.group(3)), MONTHS[m.group(2)], int(m.group(1)))
        m = re.search(r"(\d{4})q(\d)_form13f", link)
        return pd.Timestamp(int(m.group(1)), 3 * int(m.group(2)), 1) + pd.offsets.MonthEnd(0) if m else pd.Timestamp(0)
    return max(links, key=end_of) if links else None


def download(url: str, attempts: int = 8):
    """~100 MB zip; the SEC often cuts long transfers, so a broken one RESUMES where it stopped (HTTP Range)."""
    buf = bytearray()
    total = None
    for k in range(attempts):
        headers = {"Range": f"bytes={len(buf)}-"} if buf else {}
        try:
            with SESSION.get(url, headers=headers, timeout=120, stream=True) as r:
                if r.status_code not in (200, 206):
                    time.sleep(10); continue
                if r.status_code == 200:
                    buf = bytearray()                       # server ignored the range: start over
                    total = int(r.headers.get("Content-Length", 0)) or None
                elif total is None:
                    m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
                    total = int(m.group(1)) if m else None
                for chunk in r.iter_content(1 << 20):
                    buf.extend(chunk)
            if total is None or len(buf) >= total:
                return bytes(buf)
        except requests.RequestException as e:
            print(f"  telechargement du jeu 13F coupe a {len(buf) / 1e6:.0f} Mo ({type(e).__name__}), reprise {k + 1}/{attempts}")
        time.sleep(5)
    return None


def best_ideas(content: bytes) -> tuple:
    z = zipfile.ZipFile(io.BytesIO(content))
    member = {i.filename.split("/")[-1]: i.filename for i in z.infolist()}
    sub = pd.read_csv(z.open(member["SUBMISSION.tsv"]), sep="\t", dtype=str)
    cov = pd.read_csv(z.open(member["COVERPAGE.tsv"]), sep="\t", dtype=str, usecols=["ACCESSION_NUMBER", "ISAMENDMENT", "REPORTTYPE"])
    sub = sub.merge(cov, on="ACCESSION_NUMBER")
    sub = sub[(sub["SUBMISSIONTYPE"] == "13F-HR") & (sub["ISAMENDMENT"].fillna("N") != "Y")
              & sub["REPORTTYPE"].fillna("").str.contains("HOLDINGS|COMBINATION", case=False)].copy()
    sub["FILING_DATE"] = pd.to_datetime(sub["FILING_DATE"], format="%d-%b-%Y", errors="coerce")
    sub["PERIOD"] = pd.to_datetime(sub["PERIODOFREPORT"], format="%d-%b-%Y", errors="coerce")
    sub = sub[(sub["FILING_DATE"] - sub["PERIOD"]).dt.days.between(0, 50)]
    period = sub["PERIOD"].mode().iloc[0]
    sub = sub[sub["PERIOD"] == period].drop_duplicates("CIK", keep="last")
    keep, parts = set(sub["ACCESSION_NUMBER"]), []
    for ch in pd.read_csv(z.open(member["INFOTABLE.tsv"]), sep="\t", dtype=str, chunksize=500_000,
                          usecols=["ACCESSION_NUMBER", "CUSIP", "VALUE", "PUTCALL", "NAMEOFISSUER", "SSHPRNAMTTYPE"]):
        ch = ch[ch["PUTCALL"].isna() & ch["ACCESSION_NUMBER"].isin(keep) & (ch["SSHPRNAMTTYPE"].str.upper() == "SH")]
        ch = ch.assign(VALUE=pd.to_numeric(ch["VALUE"], errors="coerce"), CUSIP=ch["CUSIP"].str.upper().str[:9])
        parts.append(ch[ch["VALUE"] > 0][["ACCESSION_NUMBER", "CUSIP", "VALUE", "NAMEOFISSUER"]])
    info = pd.concat(parts)
    names = info.drop_duplicates("CUSIP").set_index("CUSIP")["NAMEOFISSUER"]
    h = info.groupby(["ACCESSION_NUMBER", "CUSIP"], as_index=False)["VALUE"].sum()
    npos = h.groupby("ACCESSION_NUMBER")["CUSIP"].transform("size")
    h = h[(npos >= 10) & (npos <= 300)].copy()
    h["w"] = h["VALUE"] / h.groupby("ACCESSION_NUMBER")["VALUE"].transform("sum")
    W = h.groupby("CUSIP")["VALUE"].sum() / h["VALUE"].sum()
    h["excess"] = h["w"] - h["CUSIP"].map(W)
    b = h.loc[h.groupby("ACCESSION_NUMBER")["excess"].idxmax()]
    b = b[b["excess"] >= b["excess"].quantile(ACTIVE_Q)]
    # one line per ISSUER (first 6 CUSIP characters): BRK-A / BRK-B or GOOG / GOOGL are one idea, carried by the
    # share class most managers hold
    b = b.assign(ISSUER=b["CUSIP"].str[:6])
    main_cls = b.groupby(["ISSUER", "CUSIP"]).size().reset_index(name="k").sort_values("k").drop_duplicates("ISSUER", keep="last")
    ideas = b.groupby("ISSUER").agg(n=("ACCESSION_NUMBER", "size"), excess=("excess", "mean")).reset_index()
    ideas = ideas.merge(main_cls[["ISSUER", "CUSIP"]], on="ISSUER")
    ideas["issuer"] = ideas["CUSIP"].map(names)
    return ideas.sort_values(["n", "excess"], ascending=False), period, int(len(b))


def map_cusips(cusips, cache: dict):
    todo = [c for c in cusips if c not in cache]
    s = requests.Session(); s.headers.update({"Content-Type": "application/json"})
    for k in range(0, len(todo), 10):
        batch = todo[k:k + 10]
        for t in range(5):
            try:
                r = s.post(FIGI, json=[{"idType": "ID_CUSIP", "idValue": c, "exchCode": "US"} for c in batch], timeout=30)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(30)
        else:
            continue
        for c, res in zip(batch, r.json()):
            d = (res.get("data") or [None])[0]
            ok = d is not None and d.get("securityType2") in ("Common Stock", "Depositary Receipt") and d.get("ticker")
            cache[c] = d["ticker"].replace("/", "-").replace(".", "-") if ok else None
        time.sleep(2.6)                          # OpenFIGI without a key: 25 requests a minute


def main():
    book = Book("bestideas", "Bot #51 Meilleures idees des gerants", STARTING_CAPITAL, {"pool": STARTING_CAPITAL})
    st = book.state
    st.setdefault("figi", {})
    book.load_prices()
    book.mark_to_market()
    book.execute_pending()
    book.mark_to_market()
    now = now_utc()

    link = latest_dataset()
    if link and st.get("dataset") != link.rsplit("/", 1)[1]:
        content = download("https://www.sec.gov" + link)
        if content is not None:
            ideas, period, n_mgr = best_ideas(content)
            top = ideas.head(KEEP_RANK * 4)
            map_cusips(list(top["CUSIP"]), st["figi"])
            top = top.assign(ticker=top["CUSIP"].map(st["figi"])).dropna(subset=["ticker"]).drop_duplicates("ticker")
            px = yf.download(list(top["ticker"]), period="5d", auto_adjust=False, progress=False, threads=True)["Close"]
            last = px.ffill().iloc[-1] if len(px) else pd.Series(dtype=float)
            top = top[top["ticker"].map(lambda t: last.get(t, 0) >= MIN_PRICE_USD)].head(KEEP_RANK).reset_index(drop=True)
            rank = {t: i + 1 for i, t in enumerate(top["ticker"])}
            st["pending"] = []
            held = set(book.open_lots()["ticker"])
            for t in sorted(held):
                if rank.get(t, 999) > KEEP_RANK:
                    book.add_order("sell", t, "pool", "n'est plus parmi les meilleures idees partagees", now, fraction=1.0)
            target = [t for t in top["ticker"].head(N_HOLD)]
            keep = [t for t in held if rank.get(t, 999) <= KEEP_RANK]
            new = [t for t in target if t not in held][:max(0, N_HOLD - len(keep))]
            amount = book.equity() / N_HOLD
            free = st["cash"]["pool"] + sum(book.position_value(ticker=t) for t in held if t not in keep)
            for t in new:
                if free < amount * 0.5:
                    break
                row = top[top["ticker"] == t].iloc[0]
                book.add_order("buy", t, "pool", f"meilleure idee de {int(row['n'])} gerant(s) actif(s), surpoids moyen {row['excess']:.1%}",
                               now, amount_eur=min(amount, free), name=str(row["issuer"]).title(), sector="Idees des gerants (13F)",
                               country="United States", currency="USD", fractional=True, score=int(row["n"]))
                free -= min(amount, free)
            st["dataset"] = link.rsplit("/", 1)[1]
            st["period"] = period.strftime("%Y-%m-%d")
            st["n_managers"] = n_mgr
            st["rebalance_key"] = now.strftime("%Y-%m-%d")
            st["ideas"] = [{"ticker": x.ticker, "issuer": str(x.issuer).title(), "n": int(x.n), "excess": round(float(x.excess), 4)}
                           for x in top.itertuples()]
            print(f"Jeu 13F {st['dataset']} (trimestre clos le {st['period']}) : {n_mgr} gerants actifs, "
                  f"{len(top)} idees retenues, {len(new)} achat(s)")
    book.save({"dataset": st.get("dataset"), "period": st.get("period"), "n_managers": st.get("n_managers"),
               "rebalance_key": st.get("rebalance_key")})


if __name__ == "__main__":
    main()
