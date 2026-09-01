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

Cached on disk per (ticker, date) and shared across all three newsgated bots -- they draw
candidates from the same long_candidates_latest.csv, so without this a ticker considered
by all three bots in the same hourly run would trigger 3 identical Ollama calls.
"""
import json
import pathlib
import sys

import pandas as pd
import requests
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
CACHE_PATH = HERE / "results/screener/news_verdict_cache.json"
CACHE_RETENTION_DAYS = 14  # verdicts are only ever looked up under today's date -- older
# entries are dead weight, pruned on every save rather than left to grow unbounded

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 90  # generous: covers a cold model load (first call after `ollama serve`
# starts, e.g. in CI) as well as a warm local server

MAX_HEADLINES = 5

PROMPT_TEMPLATE = """Tu es un analyste actions prudent. Voici l'actualite recente pour {name} ({ticker}, secteur {sector}) :

{headlines}

Le screener quantitatif a deja juge cette action sous-evaluee avec un momentum positif. Ta seule tache : d'apres CES SEULES actualites, y a-t-il une raison de red flag evidente d'EVITER cet achat maintenant (fraude, enquete, profit warning, proces majeur, faillite, scandale, delisting) ? Une actualite neutre ou positive, ou l'absence d'actualite notable, n'est PAS une raison d'eviter.

Reponds UNIQUEMENT en JSON : {{"relevant": true|false, "reason": "<une phrase courte>"}}
relevant=false seulement si tu vois un red flag clair et recent. Par defaut relevant=true.
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


def fetch_headlines(ticker: str, max_items: int = MAX_HEADLINES) -> list[str]:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception as e:
        print(f"  echec fetch news pour {ticker}: {e}", file=sys.stderr)
        return []
    headlines = []
    for item in items[:max_items]:
        content = item.get("content", item)  # yfinance>=1.4 nests fields under "content"
        title = content.get("title")
        summary = content.get("summary") or ""
        pub_date = content.get("pubDate") or ""
        if title:
            line = f"- [{pub_date}] {title}"
            if summary:
                line += f" -- {summary[:200]}"
            headlines.append(line)
    return headlines


def _call_ollama(ticker: str, name: str, sector: str, headlines: list[str]) -> dict:
    prompt = PROMPT_TEMPLATE.format(
        name=name, ticker=ticker, sector=sector or "inconnu",
        headlines="\n".join(headlines),
    )
    resp = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json",
    }, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    # decode from raw bytes (json.loads assumes UTF-8 per RFC 8259) rather than resp.json(),
    # which goes through resp.text's charset guess -- that guess falls back to ISO-8859-1 for
    # some responses and mangles accented French reasons into replacement characters.
    outer = json.loads(resp.content)
    raw = json.loads(outer["response"])
    return {
        "relevant": bool(raw.get("relevant", True)),
        "reason": str(raw.get("reason", ""))[:300],
        "source": "ollama",
    }


def news_verdict(ticker: str, name: str, sector: str, today: str) -> dict:
    """Returns {"relevant": bool, "reason": str, "source": str}, cached per (ticker, today)."""
    cache = _load_cache()
    key = f"{ticker}_{today}"
    if key in cache:
        return cache[key]

    headlines = fetch_headlines(ticker)
    if not headlines:
        verdict = {"relevant": True, "reason": "aucune actualite recente trouvee", "source": "no_news"}
    else:
        try:
            verdict = _call_ollama(ticker, name, sector, headlines)
        except Exception as e:
            print(f"  echec appel Ollama pour {ticker}: {e}", file=sys.stderr)
            verdict = {"relevant": True, "reason": f"ollama indisponible ({e})", "source": "ollama_error"}

    cache[key] = verdict
    _save_cache(cache)
    return verdict
