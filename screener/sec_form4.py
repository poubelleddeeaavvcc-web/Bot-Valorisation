"""Live feed of insider open-market purchases from SEC EDGAR (Form 4), for Bot #45 (2026-10-13).

The SEC's quarterly "Insider Transactions Data Sets" (used for the 2006-2026 backtest) are published after
each quarter -- too late to trade on. EDGAR's daily form index (daily-index/YYYY/QTRn/form.YYYYMMDD.idx) lists
every Form 4 filed that day (~400-800); each one is fetched and its XML ownershipDocument parsed. Same
selection as the backtest's insider_parse step: non-derivative, transaction code P, acquired (A), a common
or ordinary share title, filed by a director or an officer (a pure 10% owner is a fund buying for its own
reasons). Amendments (4/A) are left out, like in the backtest. Requests carry the repo's SEC User-Agent and
stay under the SEC's fair-access limit (~5 requests/second here, 10 allowed).
"""
import pathlib
import re
import sys
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.sec_edgar import SEC_HEADERS  # noqa: E402

DAILY_INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/form.{ymd}.idx"
ARCHIVE_URL = "https://www.sec.gov/Archives/{path}"
REQUEST_PAUSE = 0.2
_session = requests.Session()
_session.headers.update(SEC_HEADERS)


def _get(url: str, retries: int = 4):
    for k in range(retries):
        try:
            r = _session.get(url, timeout=30)
        except requests.RequestException:
            time.sleep(2 * (k + 1))
            continue
        if r.status_code == 200:
            return r.text
        if r.status_code == 404:
            return None
        time.sleep(5 * (k + 1))  # 429 / 503: back off
    return None


def form4_paths(day: pd.Timestamp):
    """Archive paths of the Form 4 filings of `day`, or None if that day's index doesn't exist (yet)."""
    q = (day.month - 1) // 3 + 1
    text = _get(DAILY_INDEX_URL.format(y=day.year, q=q, ymd=day.strftime("%Y%m%d")))
    if text is None:
        return None
    paths = set()
    for line in text.splitlines():
        if line.startswith("4 "):                       # "4/A" (amendments) deliberately excluded
            m = re.search(r"(edgar/data/\S+\.txt)\s*$", line)
            if m:
                paths.add(m.group(1))
    return sorted(paths)


def _val(node, path):
    x = node.find(path)
    if x is None:
        return None
    v = x.find("value")
    txt = (v.text if v is not None else x.text) or ""
    return txt.strip() or None


def _flag(node, tag):
    return (_val(node, tag) or "").lower() in ("1", "true")


def parse_form4(text: str) -> list:
    """Open-market purchases by directors/officers in one Form 4 submission (.txt with an embedded XML)."""
    m = re.search(r"<ownershipDocument>.*?</ownershipDocument>", text, re.S)
    if not m:
        return []
    try:
        doc = ET.fromstring(m.group(0))
    except ET.ParseError:
        return []
    ticker = (_val(doc, "issuer/issuerTradingSymbol") or "").upper()
    issuer = _val(doc, "issuer/issuerName")
    owners = []
    for ro in doc.findall("reportingOwner"):
        rel = ro.find("reportingOwnerRelationship")
        if rel is None:
            continue
        if _flag(rel, "isDirector") or _flag(rel, "isOfficer"):
            owners.append((_val(ro, "reportingOwnerId/rptOwnerCik"), _val(rel, "officerTitle") or
                           ("Director" if _flag(rel, "isDirector") else "Officer")))
    if not owners or not ticker:
        return []
    out = []
    for tr in doc.findall("nonDerivativeTable/nonDerivativeTransaction"):
        if (_val(tr, "transactionCoding/transactionCode") or "") != "P":
            continue
        if (_val(tr, "transactionAmounts/transactionAcquiredDisposedCode") or "") != "A":
            continue
        title = _val(tr, "securityTitle") or ""
        if not re.search(r"common|ordinary", title, re.I):
            continue
        try:
            shares = float(_val(tr, "transactionAmounts/transactionShares") or "nan")
            price = float(_val(tr, "transactionAmounts/transactionPricePerShare") or "nan")
        except ValueError:
            continue
        out.append({"ticker": ticker, "issuer": issuer, "trans_date": _val(tr, "transactionDate"),
                    "shares": shares, "price": price, "value": shares * price,
                    "owner_cik": owners[0][0], "owner_title": owners[0][1]})
    return out


def purchases_for_day(day: pd.Timestamp, log=print):
    """All qualifying purchase rows filed on `day` (filing_date added), or None if the index isn't there."""
    paths = form4_paths(day)
    time.sleep(REQUEST_PAUSE)
    if paths is None:
        return None
    rows = []
    for p in paths:
        text = _get(ARCHIVE_URL.format(path=p))
        time.sleep(REQUEST_PAUSE)
        if text:
            for r in parse_form4(text):
                r["filing_date"] = day.strftime("%Y-%m-%d")
                rows.append(r)
    log(f"  EDGAR {day.date()} : {len(paths)} formulaires 4, {len(rows)} achats en Bourse de dirigeants")
    return rows
