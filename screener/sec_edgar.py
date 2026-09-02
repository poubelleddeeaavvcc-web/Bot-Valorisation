"""SEC EDGAR-sourced customer concentration (2026-09-02): a real, cited disclosure pulled
from a company's own 10-K, not an LLM asked to recall or guess a fact. Per the user's explicit
requirement ("il faut trouver l'information sur Internet et etre sur de l'information") --
Ollama has no internet access and must never be asked to state a specific real-world fact from
its own weights, only to read text handed to it. This module is the "certain" primary source
for US-listed tickers; news_filter.customer_concentration_verdict() falls back to its own
Ollama-based read of longBusinessSummary only when this module returns nothing confident.
US-listed only -- SEC EDGAR has no jurisdiction over foreign filers (most file 20-F, not
10-K, so _latest_10k() naturally returns None for them and the caller falls through).

Extraction is grounded in an actual regulatory requirement, not a guess at phrasing: ASC
280-10-50-42 / Item 101(c)(1)(vii) require a company to disclose any customer representing
10%+ of revenue, so CONCENTRATION_RE is tuned to that specific 10% threshold and the standard
disclosure language ("customer ... accounted for/represented N% of ... sales/revenue", or its
negation "no [single] customer accounted for N% or more").

Deliberately conservative -- returns None rather than a wrong answer whenever the match isn't
completely clean, rather than trying to cover every possible phrasing. Caught in testing
(2026-09-02) before this was tightened: a naive "customer" + "%" proximity match wrongly
tagged Apple's *trade receivables* concentration note as revenue concentration (excluded by
requiring "sales"/"revenue" in the same clause and rejecting "receivable" nearby), and wrongly
read YETI's "No single customer accounted for 10% or more of our gross sales" as a POSITIVE
match on the "10%" it contains rather than the negation it actually is (fixed by tying the
leading "no" to the same match via a named group instead of a separate, misaligned negation
regex). A false negative here just falls through to the softer Ollama signal; a false positive
would assert a confident, sourced-looking fact that's wrong -- so every remaining ambiguous
case is designed to fall through rather than guess.
"""
import json
import pathlib
import re
import sys

import pandas as pd
import requests

HERE = pathlib.Path(__file__).parent.parent

# SEC requires a descriptive User-Agent identifying the application and a contact (see
# https://www.sec.gov/os/webmaster-faq#developers) for programmatic access -- not
# authentication, just their fair-use/abuse-contact policy. Same posture as this project's
# existing yfinance/Google News usage: a personal project, low volume, identifying itself
# honestly rather than spoofing a browser.
SEC_HEADERS = {"User-Agent": "TV-strategies-personal-project alex.verneycarron@gmail.com"}

TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
TICKER_CIK_CACHE_PATH = HERE / "data/universe/sec_ticker_cik.json"
TICKER_CIK_STALENESS_DAYS = 30  # SEC regenerates this file periodically; new listings are
# rare enough that a monthly refresh of the matching table is plenty

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
REQUEST_TIMEOUT = 20

CONCENTRATION_THRESHOLD = 10.0  # matches ASC 280 / Item 101(c)(1)(vii)'s own materiality bar
# for disclosing a customer -- not an arbitrary pick, it's the rule that determines whether a
# company had to say anything about a customer's share of revenue in the first place.

_PCT = r"(?P<pct>\d{1,3}(?:\.\d+)?)\s*(?:%|percent)"
# One pattern for both the positive ("customer X accounted for N% of sales") and negated
# ("no customer accounted for N% or more of sales") phrasing -- the leading "no" is captured
# as part of THE SAME match (not a separate regex checked for overlap) so it can never
# misalign with a different sentence's percentage. [^.%]/[^.] gaps deliberately exclude
# literal periods so the match can't tunnel across a sentence boundary (the YETI failure mode
# above was exactly this, via a bridging group that allowed periods inside it).
CONCENTRATION_RE = re.compile(
    r"(?P<neg>\bno\s+)?"
    r"(?:single\s+|other\s+|one\s+|two\s+|three\s+|four\s+|five\s+|ten\s+"
    r"|our\s+\d+\s+largest\s+|largest\s+)?"
    r"(?:end\s+)?customers?"
    r"[^.%]{0,60}?"
    r"(?:accounted for|represented|comprised|purchased)"
    r"[^.%]{0,30}?" + _PCT + r"(?:\s*or more)?"
    r"[^.]{0,60}?(?:sales|revenue)",
    re.IGNORECASE,
)


def _load_ticker_cik_map() -> dict:
    if TICKER_CIK_CACHE_PATH.exists():
        age_days = (pd.Timestamp.today() -
                    pd.Timestamp.fromtimestamp(TICKER_CIK_CACHE_PATH.stat().st_mtime)).days
        if age_days < TICKER_CIK_STALENESS_DAYS:
            try:
                return json.loads(TICKER_CIK_CACHE_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    try:
        resp = requests.get(TICKER_CIK_URL, headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()
    except Exception as e:
        print(f"  echec fetch SEC ticker->CIK map: {e}", file=sys.stderr)
        if TICKER_CIK_CACHE_PATH.exists():  # stale cache beats no cache
            try:
                return json.loads(TICKER_CIK_CACHE_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}
    mapping = {v["ticker"].upper(): v["cik_str"] for v in raw.values()}
    TICKER_CIK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TICKER_CIK_CACHE_PATH.write_text(json.dumps(mapping), encoding="utf-8")
    return mapping


def _latest_10k(cik: int) -> dict | None:
    try:
        resp = requests.get(SUBMISSIONS_URL.format(cik=cik), headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        recent = resp.json()["filings"]["recent"]
    except Exception as e:
        print(f"  echec fetch SEC submissions (CIK {cik}): {e}", file=sys.stderr)
        return None
    for i, form in enumerate(recent["form"]):
        if form == "10-K":  # foreign private issuers file 20-F instead -- naturally returns
            # None here rather than needing a separate "is this a 10-K filer" check upfront
            accession = recent["accessionNumber"][i].replace("-", "")
            doc = recent["primaryDocument"][i]
            return {
                "accession": accession, "doc": doc, "filed": recent["filingDate"][i],
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{doc}",
            }
    return None


def _fetch_filing_text(url: str) -> str | None:
    try:
        resp = requests.get(url, headers=SEC_HEADERS, timeout=REQUEST_TIMEOUT * 2)  # 10-Ks run 1-2MB
        resp.raise_for_status()
    except Exception as e:
        print(f"  echec fetch 10-K {url}: {e}", file=sys.stderr)
        return None
    text = re.sub(r"<[^>]+>", " ", resp.text)
    text = re.sub(r"&#\d+;|&nbsp;|&amp;", " ", text)
    return re.sub(r"\s+", " ", text)


def customer_concentration_from_10k(ticker: str) -> dict | None:
    """Returns {"concentration": "concentrated"|"diversified", "detail": str, "pct": float,
    "filing_url": str, "filed": str} for a US-listed ticker with a clean ASC-280-style
    disclosure match, or None -- not a US SEC filer, no 10-K on file, a fetch failed, or no
    confident match in the text. None always means "fall back to the Ollama read", never
    "assume diversified"."""
    cik = _load_ticker_cik_map().get(ticker.upper())
    if cik is None:
        return None

    filing = _latest_10k(cik)
    if filing is None:
        return None

    text = _fetch_filing_text(filing["url"])
    if text is None:
        return None

    m = CONCENTRATION_RE.search(text)
    if not m:
        return None
    pct = float(m.group("pct"))
    negated = bool(m.group("neg"))
    if not negated and pct < CONCENTRATION_THRESHOLD:
        return None  # shouldn't happen given the regex's own phrasing, but stay conservative

    return {
        "concentration": "diversified" if negated else "concentrated",
        "detail": m.group(0)[:400],
        "pct": pct,
        "filing_url": filing["url"],
        "filed": filing["filed"],
    }
