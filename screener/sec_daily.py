"""EDGAR daily-index helpers shared by the filing-text bots (2026-10-14): #49 Ton des communiques (8-K earnings
releases, simulate_tone_portfolio.py) and #50 Rapports inchanges (10-K / 10-Q year-on-year text, simulate_lazy_
portfolio.py). Same fair-access discipline as sec_form4.py (repo SEC User-Agent, ~5 requests/second at most).

daily_filings(day, forms) -> rows of that day's form index (form, company, cik, accession, archive path), or None
when the index does not exist (not published yet, or EDGAR closed: a missing index answers 403);
filing_index(cik, acc) -> the filing's index page (items of an 8-K, acceptance time, document table);
document_text(url) -> lower-cased plain text of an HTML/text document."""
import html as htmlmod
import pathlib
import re
import sys
import time

import pandas as pd
import requests

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.sec_edgar import SEC_HEADERS  # noqa: E402

DAILY_INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/form.{ymd}.idx"
PAUSE = 0.2
_session = requests.Session()
_session.headers.update(SEC_HEADERS)
_TAG, _WS = re.compile(r"<[^>]+>"), re.compile(r"\s+")


def get(url: str, retries: int = 4, missing_codes=(404,)):
    for k in range(retries):
        try:
            r = _session.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(2 * (k + 1))
            continue
        time.sleep(PAUSE)
        if r.status_code == 200:
            return r.text
        if r.status_code in missing_codes:
            return None
        time.sleep(5 * (k + 1))
    return None


def daily_filings(day: pd.Timestamp, forms=("8-K",)):
    q = (day.month - 1) // 3 + 1
    text = get(DAILY_INDEX_URL.format(y=day.year, q=q, ymd=day.strftime("%Y%m%d")), missing_codes=(403, 404))
    if text is None:
        return None
    rows = []
    for line in text.splitlines():
        form = line[:17].strip()
        if form not in forms:
            continue
        m = re.search(r"\s(\d{1,10})\s+(\d{8})\s+(edgar/data/\S+\.txt)\s*$", line)
        if m:
            path = m.group(3)
            rows.append({"form": form, "company": line[17:79].strip(), "cik": int(m.group(1)), "path": path,
                         "acc": path.rsplit("/", 1)[1][:-4]})
    return rows


def filing_index(cik: int, acc: str):
    return get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{acc}-index.htm")


def items_of(index_html: str) -> list:
    return re.findall(r"Item\s+(\d+\.\d+)", index_html or "")


def accepted_at(index_html: str):
    """Acceptance time (New York) of the filing, from its index page."""
    m = re.search(r'<div class="info">(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})</div>', index_html or "")
    return pd.Timestamp(m.group(1)).tz_localize("America/New_York") if m else None


def document_link(index_html: str, cik: int, acc: str, doc_type_re: str):
    """URL of the first document whose 'Type' column matches doc_type_re (e.g. r'EX-99(\\.1|\\.01)?' or r'10-K')."""
    # iXBRL documents carry an extra '<span>iXBRL</span>' after the link and an '/ix?doc=' viewer prefix
    m = re.search(r'<a href="([^"]+)">[^<]*</a>[^<]*(?:<span[^>]*>[^<]*</span>)?\s*</td>\s*<td scope="row">\s*('
                  + doc_type_re + r')\s*</td>', index_html or "", re.I)
    if not m:
        return None
    href = m.group(1)
    if href.startswith("/ix?doc="):
        href = href[len("/ix?doc="):]
    return "https://www.sec.gov" + href if href.startswith("/") else \
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{href}"


def document_text(url: str, max_chars: int = 3_000_000):
    raw = get(url)
    if raw is None:
        return None
    return _WS.sub(" ", htmlmod.unescape(_TAG.sub(" ", raw[:max_chars]))).strip().lower()
