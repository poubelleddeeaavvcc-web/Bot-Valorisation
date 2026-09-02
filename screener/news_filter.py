"""Ollama-backed news relevance gate for the "newsgated" bot variants (Bot#4/5/6): once a
candidate has already cleared the valuation/momentum screen those bots inherit unchanged
from Bot#1/2/3, ask a local LLM whether recent news is flagging an obvious reason to avoid
buying it right now, before the trade is actually placed.

Deliberately NOT used by the original Bot#1/2/3 -- those stay blind on purpose (see
simulate_portfolio.py's module docstring) as the honest, unbiased baseline track record.
This is a parallel experiment testing whether a news filter improves on that baseline, not
a replacement for it -- see portfolio/bot_checkin.py for the side-by-side comparison.

Fails OPEN (treats the candidate as relevant, i.e. buyable) on any error: no news found,
Ollama unreachable, malformed response. The point is a soft veto on names with an obvious
red flag, not a hard dependency that silently starves every bot of every buy whenever a
network hiccup or a CI cold-start makes the news fetch or the model call fail. The
verdict's "source" field records whether a real LLM verdict was reached ("ollama") or a
fallback kicked in ("no_news"/"ollama_error"), so a string of fallbacks is visible in the
ledger/logs rather than looking identical to a genuine veto.

Headlines are pulled from two independent sources -- yfinance's built-in .news (Yahoo-only)
and Google News' public RSS search feed (aggregates many outlets: press releases, wire
services, sector press) -- merged and deduplicated by title, so the model sees a broader
slice of coverage than Yahoo alone provides. The RSS feed is public and documented for
"personal ... use in a feed reader" (its own copyright header); this is a personal project
consuming it the same way, at a low per-ticker-per-day rate (see caching below), same
posture as this project's existing yfinance usage.

Cached on disk per (ticker, date) and shared across all three newsgated bots -- they draw
candidates from the same long_candidates_latest.csv, so without this a ticker considered
by all three bots in the same hourly run would trigger 3 identical Ollama calls. Separately,
every headline set + verdict actually computed (i.e. not a cache hit) is appended to
NEWS_DB_PATH -- an ever-growing, never-pruned archive (unlike the cache) meant as the user's
own historical dataset for later analysis, distinct from the cache's job of avoiding
duplicate Ollama calls within a single day.

KEYWORD SAFETY NET, two-stage (added 2026-09-01, revised same day): the first production run
showed llama3.2:3b is NOT reliable at turning its own reading into the "relevant" boolean --
caught live on AMZN, where the model's "reason" text correctly summarized "AMZN Stock
Declines After FTC Accuses Amazon of Secretly Inflating Ad Prices by $20 Billion" (a major
lawsuit, explicitly one of the prompt's own red-flag categories) but still returned
relevant=true. A first fix (auto-veto on any RED_FLAG_KEYWORDS match) over-corrected: a
keyword alone can't tell "lawsuit newly filed, risk ongoing" from "lawsuit WON, charges
DROPPED, case DISMISSED" -- exactly the kind of favorable resolution that can leave a stock
undervalued and worth buying. Fix: a keyword match is now a TRIGGER, not a verdict. It forces
a second, narrow, single-purpose Ollama call (_call_ollama_followup) asking only "is this one
headline an ACTIVE unresolved risk, or an already-favorable resolution?" -- a small model is
far more reliable on one constrained yes/no question than on the original broad multi-part
judgment. Only an active-risk answer vetoes; a resolved/favorable answer keeps the original
verdict. The follow-up call failing (network/Ollama error) defaults to veto (fail CLOSED),
unlike the rest of this module's fail-open default -- once a keyword has already surfaced a
plausible risk, "can't confirm it's resolved" should block, not wave through unexamined.

The base Ollama call's own "relevant" boolean is IGNORED for the buy/skip decision (same
production run, revised again same day): on 9942.TW, a ticker with zero red-flag-relevant
headlines (generic unrelated market reports), the base call still returned relevant=false
with a reason that itself said "no clearly disruptive news" -- the boolean contradicted its
own reasoning text, in the opposite direction from the AMZN miss (false negative there, false
positive here). A 3B local model's structured "relevant" field is noise uncorrelated with its
own prose in BOTH directions, so it no longer decides anything: only the deterministic
keyword trigger + narrow follow-up (above) can veto. The base call's "sentiment"/"reason" are
kept purely as informational context (shown in the dashboard), never as the gate itself.
"""
import json
import pathlib
import sys
import xml.etree.ElementTree as ET
from urllib.parse import quote

import pandas as pd
import requests
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
CACHE_PATH = HERE / "results/screener/news_verdict_cache.json"
CACHE_RETENTION_DAYS = 14  # verdicts are only ever looked up under today's date -- older
# entries are dead weight, pruned on every save rather than left to grow unbounded

NEWS_DB_PATH = HERE / "results/screener/news_database.csv"
NEWS_DB_COLUMNS = [
    "ticker", "date", "name", "sector", "n_headlines", "sources",
    "sentiment", "relevant", "reason", "headlines",
]

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 90  # generous: covers a cold model load (first call after `ollama serve`
# starts, e.g. in CI) as well as a warm local server

MAX_HEADLINES_PER_SOURCE = 5

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
GOOGLE_NEWS_TIMEOUT = 15

# Deterministic backstop -- see module docstring (KEYWORD SAFETY NET). Matched
# case-insensitively as a substring against each headline title.
RED_FLAG_KEYWORDS = [
    "fraud", "fraudulent", "sec investigation", "sec charges", "sec probe",
    "ftc lawsuit", "ftc sues", "ftc accuses", "class action", "class-action",
    "lawsuit", "sues", "sued", "indictment", "subpoena", "investigation",
    "bankruptcy", "chapter 11", "insolvency", "going concern",
    "delisting", "delisted", "profit warning", "guidance cut", "cuts guidance",
    "restated earnings", "restatement", "accounting scandal", "accounting fraud",
    "short seller", "short-seller", "recall", "resigns amid", "ousted",
]

PROMPT_TEMPLATE = """Tu es un analyste actions. Voici l'actualite recente pour {name} ({ticker}, secteur {sector}) :

{headlines}

D'apres CES SEULES actualites, le ton general est-il plutot positif, neutre ou negatif pour l'action ? Resume aussi en une phrase ce qui ressort de cette actualite.

Reponds UNIQUEMENT en JSON : {{"sentiment": "positive"|"neutral"|"negative", "reason": "<une phrase courte>"}}
"""

# Narrow follow-up, triggered only when a RED_FLAG_KEYWORDS match forces a closer look at ONE
# headline -- see module docstring (KEYWORD SAFETY NET). Deliberately a single yes/no question
# about a single headline, not the broad multi-criteria judgment above.
FOLLOWUP_PROMPT = """Voici un titre d'actualite pour {name} ({ticker}) qui contient un terme d'alerte potentiel ('{keyword}') :

"{title}"

Ce titre decrit-il un risque ACTIF et NON RESOLU (proces ou enquete EN COURS, accusation recente, sanction en attente, faillite en cours) qui justifie d'eviter un achat maintenant ? Ou decrit-il au contraire une resolution FAVORABLE pour l'entreprise (proces GAGNE, charges ABANDONNEES, enquete CLASSEE SANS SUITE, affaire REJETEE, recours REUSSI) -- auquel cas ce n'est PAS une raison d'eviter, l'incertitude est levee en faveur de l'entreprise ?

Reponds UNIQUEMENT en JSON : {{"active_risk": true|false, "reason": "<une phrase courte>"}}
"""


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict):
    cutoff = (pd.Timestamp.today() - pd.Timedelta(days=CACHE_RETENTION_DAYS)).strftime("%Y-%m-%d")
    pruned = {k: v for k, v in cache.items() if k.rsplit("_", 1)[-1] >= cutoff}
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(pruned, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_headlines_yfinance(ticker: str, max_items: int = MAX_HEADLINES_PER_SOURCE) -> list[dict]:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"  echec fetch yfinance news pour {ticker}: {e}", file=sys.stderr)
        return []
    out = []
    for item in items[:max_items]:
        content = item.get("content", item)  # yfinance>=1.4 nests fields under "content"
        title = content.get("title")
        if title:
            out.append({"source": "yfinance", "title": title,
                        "summary": content.get("summary") or "", "pub_date": content.get("pubDate") or ""})
    return out


def fetch_headlines_google_news(ticker: str, name: str, max_items: int = MAX_HEADLINES_PER_SOURCE) -> list[dict]:
    """Google News' public RSS search feed -- aggregates many outlets (wire services, press
    releases, sector press) beyond Yahoo alone. Queried by company name (more precise than a
    bare ticker symbol, which often collides with unrelated words/other tickers)."""
    query = f'"{name}" stock' if name else f"{ticker} stock"
    try:
        resp = requests.get(GOOGLE_NEWS_RSS.format(query=quote(query)), timeout=GOOGLE_NEWS_TIMEOUT,
                             headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception as e:
        print(f"  echec fetch Google News pour {ticker}: {e}", file=sys.stderr)
        return []
    out = []
    for item in root.findall(".//item")[:max_items]:
        title = item.findtext("title")
        if not title:
            continue
        source_el = item.find("source")
        source_name = (source_el.text or "").strip() if source_el is not None else "?"
        out.append({"source": f"google_news:{source_name}", "title": title,
                    "summary": "", "pub_date": item.findtext("pubDate") or ""})
    return out


def fetch_all_headlines(ticker: str, name: str) -> list[dict]:
    """Merges both sources, deduplicated by normalized title -- the same story often gets
    picked up by both yfinance and a Google News-indexed outlet."""
    combined = fetch_headlines_yfinance(ticker) + fetch_headlines_google_news(ticker, name)
    seen, deduped = set(), []
    for h in combined:
        key = h["title"].strip().lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(h)
    return deduped


def _keyword_red_flag(headlines: list[dict]) -> dict | None:
    """First (keyword, headline) match, or None. See RED_FLAG_KEYWORDS / module docstring."""
    for h in headlines:
        title_lower = h["title"].lower()
        for kw in RED_FLAG_KEYWORDS:
            if kw in title_lower:
                return {"keyword": kw, "title": h["title"]}
    return None


def _headlines_to_text(headlines: list[dict]) -> str:
    lines = []
    for h in headlines:
        line = f"- [{h['pub_date']}] {h['title']}"
        if h["summary"]:
            line += f" -- {h['summary'][:200]}"
        lines.append(line)
    return "\n".join(lines)


def _append_to_database(ticker: str, today: str, name: str, sector: str, headlines: list[dict], verdict: dict):
    row = {
        "ticker": ticker, "date": today, "name": name, "sector": sector,
        "n_headlines": len(headlines),
        "sources": "|".join(sorted({h["source"].split(":")[0] for h in headlines})),
        "sentiment": verdict.get("sentiment"), "relevant": verdict.get("relevant"),
        "reason": verdict.get("reason"),
        "headlines": " || ".join(h["title"] for h in headlines),
    }
    NEWS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = not NEWS_DB_PATH.exists()
    pd.DataFrame([row], columns=NEWS_DB_COLUMNS).to_csv(NEWS_DB_PATH, mode="a", header=header, index=False)


def _call_ollama(ticker: str, name: str, sector: str, headlines: list[dict]) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        name=name, ticker=ticker, sector=sector or "inconnu",
        headlines=_headlines_to_text(headlines),
    )
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json",
        "options": {"temperature": 0},  # deterministic-as-possible -- see FOLLOWUP_VOTES below
    }, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    # decode from raw bytes (json.loads assumes UTF-8 per RFC 8259) rather than resp.json(),
    # which goes through resp.text's charset guess -- that guess falls back to ISO-8859-1 for
    # some responses and mangles accented French reasons into replacement characters.
    outer = json.loads(resp.content)
    raw = json.loads(outer["response"])
    sentiment = raw.get("sentiment")
    # "relevant" is deliberately NOT read from raw here -- see module docstring (the 9942.TW
    # false positive): this call's boolean is unreliable noise, kept out of the gate entirely.
    # Always true at this stage; only the keyword trigger + follow-up (in news_verdict) can flip it.
    return {
        "relevant": True,
        "sentiment": sentiment if sentiment in ("positive", "neutral", "negative") else None,
        "reason": str(raw.get("reason", ""))[:300],
        "source": "ollama",
    }


def _call_ollama_followup(ticker: str, name: str, keyword: str, title: str) -> dict:
    prompt = FOLLOWUP_PROMPT.format(name=name, ticker=ticker, keyword=keyword, title=title)
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json",
    }, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    outer = json.loads(resp.content)
    raw = json.loads(outer["response"])
    return {"active_risk": bool(raw.get("active_risk", True)), "reason": str(raw.get("reason", ""))[:300]}


FOLLOWUP_VOTES = 3  # sampled multiple times rather than trusted on a single call: the same
# AMZN headline got a different active_risk answer across two otherwise-identical runs of this
# very question (temperature isn't pinned here on purpose -- see below) -- a small model's
# single judgment on this is noisy even when the question is narrow. An ABSOLUTE MAJORITY of
# "resolved" votes is required to lift the veto; a tie, a majority still seeing active risk, or
# a vote erroring all keep it -- same fail-closed posture as before, just harder to talk down.


def _keyword_followup_votes(ticker: str, name: str, keyword: str, title: str, n: int = FOLLOWUP_VOTES) -> list[dict]:
    votes = []
    for _ in range(n):
        try:
            votes.append(_call_ollama_followup(ticker, name, keyword, title))
        except Exception as e:
            print(f"  echec verification ciblee pour {ticker}: {e}", file=sys.stderr)
            votes.append({"active_risk": True, "reason": f"verification indisponible ({e})"})
    return votes


def news_verdict(ticker: str, name: str, sector: str, today: str) -> dict:
    """Returns {"relevant": bool, "sentiment": str|None, "reason": str, "source": str},
    cached per (ticker, today). Also appends the raw headlines + verdict to NEWS_DB_PATH on
    every real fetch (cache miss) -- see module docstring."""
    cache = _load_cache()
    key = f"{ticker}_{today}"
    if key in cache:
        return cache[key]

    headlines = fetch_all_headlines(ticker, name)
    if not headlines:
        verdict = {"relevant": True, "sentiment": None, "reason": "aucune actualite recente trouvee", "source": "no_news"}
    else:
        try:
            verdict = _call_ollama(ticker, name, sector, headlines)
        except Exception as e:
            print(f"  echec appel Ollama pour {ticker}: {e}", file=sys.stderr)
            verdict = {"relevant": True, "sentiment": None, "reason": f"ollama indisponible ({e})", "source": "ollama_error"}

        flag = _keyword_red_flag(headlines)
        if flag:
            votes = _keyword_followup_votes(ticker, name, flag["keyword"], flag["title"])
            resolved_votes = [v for v in votes if not v["active_risk"]]
            # strict majority (not just "any resolved vote") -- see FOLLOWUP_VOTES
            is_resolved = len(resolved_votes) > len(votes) / 2
            if not is_resolved:
                active_reasons = "; ".join(v["reason"] for v in votes if v["active_risk"] and v["reason"])[:250]
                verdict = {
                    "relevant": False, "sentiment": verdict.get("sentiment"),
                    "reason": f"'{flag['keyword']}' actif ({len(votes) - len(resolved_votes)}/{len(votes)} votes) : "
                              f"{active_reasons} (titre: {flag['title'][:150]})",
                    "source": "keyword_veto",
                }
            else:
                verdict["reason"] = (verdict.get("reason") or "") + \
                    f" [verification '{flag['keyword']}' : resolu favorablement ({len(resolved_votes)}/{len(votes)} votes) " \
                    f"-- {resolved_votes[0]['reason']}]"
        _append_to_database(ticker, today, name, sector, headlines, verdict)

    cache[key] = verdict
    _save_cache(cache)
    return verdict
