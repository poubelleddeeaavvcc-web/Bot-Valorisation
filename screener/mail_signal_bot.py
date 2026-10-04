"""Bot #33 ("Courrier"): paper-trading LONG and SHORT positions on individual stock tips found
directly in the user's own Gmail newsletters -- the per-ticker counterpart to
screener/newsletter_digest.py's per-sector qualitative signal (added 2026-09-11, per the
user's explicit request: "des mails il y a souvent des suggestions d'actions a fort potentiel
ou au contraire des chutes -- je veux des long ou des shorts sur ces actions, et savoir de
quelle newsletter ca vient pour identifier les bons/mauvais investisseurs").

THREE LAYERS (since 2026-10-03)
-------------------------------
  1. JOURNAL (mail_signal_scoring.py): every validated tip from every newsletter, evaluated at
     J+5/J+20/J+60 against its market's benchmark. Decides which newsletters are "fiable",
     "bruit" or "observation", and measures the crowd's consensus. This is the scorecard.
  2. LAB (this module's ledger, mail_signal_ledger.csv): the original uncapped long/short paper
     book -- every live tip opens a position. Kept for continuity; no longer what the scorecard is
     built from.
  3. STRATEGIE REELLE (mail_signal_real.py): capital-limited, long-only, fee-aware book that only
     follows reliable newsletters, with the crowd's consensus adjusting conviction.

2026-10-03 OVERHAUL (the user's go-live review: "fais toutes ces corrections")
-----------------------------------------------------------------------------
A hand review of the first 168 lab positions found ~43% of tips wrong: ticker mapping errors
("Chevron" -> CHEV = Charging Robotics, the lab's only big winner; "BCE" = the European Central
Bank -> Bell Canada; "The Dollar Went Up" -> USD, a 2x semiconductor ETF), inverted direction
("Paychex Plunges, Providing the Entry Investors Have Been Waiting For" -> short), plain news or
page boilerplate taken as tips, and 22 ETFs/funds. Root cause: the prompt asked the model for its
"best guess" ticker (a guessed fact -- against this repo's grounding rule) and _resolve_ticker()
only checked that the symbol had a price, not that it was the right company. Now every tip must
pass, in order (see _validate_tip / _verify_tip / _resolve_instrument):
  - deterministic text checks: a verbatim citation that really is in the email, the company
    actually named in it, not a macro subject (central bank, currency, index...), no "pas d'avis"
    style self-negation, no ad/boilerplate text, no explicit upgrade/downgrade wording
    contradicting the claimed direction;
  - a second, independent Ollama call that only sees the citation and must confirm an explicit
    investment opinion on that company, in the same direction (it is not told which direction
    was claimed);
  - a Yahoo lookup: the ticker is only taken from the email when it is literally written there,
    otherwise looked up by company name; the listing must be an EQUITY on a primary exchange
    (no ETF, fund, OTC) whose Yahoo name matches the company named in the email.
Rejected tips are logged with their reason (mail_signal_rejects.csv) so the filter itself can be
audited.

The lab's existing wrong positions were removed once (cash refunded at entry value, same as the
2026-09-16 FDX/AF.PA cleanup) and archived with their reason in mail_signal_annulled.csv -- see
cleanup_legacy_rows().

Also fixed: fees are charged on BOTH orders (was exit only); position values are in EUR with the
current FX rate (was the entry rate forever) -- stops still trigger on the local price, like a
broker stop; one CI job only (the newsletter-digest-bot repo used to run this same script on the
same ledger in parallel, losing whichever push came second); the summary now reports P&L and
average return per position, because "total_return_pct" against the nominal 300 EUR was
misleading once ~5,000 EUR of notional was deployed.

Backfill: the first runs after this overhaul also walk back over the last BACKFILL_DAYS of mail
from senders already known as newsletters, BACKFILL_MAX_PER_RUN emails per run, feeding the
journal only (no lab trade on old tips) -- so newsletters get a J+20 track record in weeks instead
of months.

Deliberately a SEPARATE script/workflow job from newsletter_digest.py, not folded into it: this
module makes its own full pass of Ollama calls on top of whatever else already runs in the same
CI window. Duplicates small helpers rather than sharing them, per this repo's standing style.

ARTICLE FETCH (see _fetch_article_extract(), added 2026-09-16): most newsletters only excerpt a
couple of sentences before a "read more" link to the sender's own site. Each newsletter's own links
are tried and, for the domains hand-confirmed fetchable with a plain HTTP GET (zonebourse.com,
tradingsat.com -- see FETCHABLE_DOMAINS), the full article text replaces the teaser. Seeking Alpha
is deliberately NOT in that list: it answers a plain GET with a PerimeterX CAPTCHA wall -- that's
bot-detection, not something this bot tries to bypass.

ATTRIBUTION (see _publication()): PRIVACY / REPO-PUBLIC CONSTRAINT -- this repo pushes to a public
GitHub remote. The user's explicit choice (2026-09-11): never persist the sender's email address.
The sending domain is kept (e.g. "seekingalpha.com"). Since 2026-10-03, for newsletter PLATFORMS
(beehiiv, substack, sailthru...) where the domain is shared by dozens of unrelated newsletters, the
sender's display name is used instead (e.g. "Some Newsletter (beehiiv.com)") -- otherwise "which
newsletter calls it right" is unanswerable there. Still never the address itself. An author name
is kept only when the email literally contains it (journal column, informational).

LAB TRADING: own ledger, own cash file -- long or short depending on the extracted sentiment,
sized at TARGET_POSITION_SIZE per tip. NO CAP on concurrent open positions (2026-09-15, user's
request: "l'idee de ce bot c'est de savoir quelles analystes sont bons, je ne veux pas de
plafond"); cash_eur can go negative and is just a running counter.
- A tip for a ticker not currently held opens a position.
- A tip that CONTRADICTS an open position closes it ("signal_inverse") only once the price has
  already moved at least MIN_REVERSAL_CONFIRM_PCT against the held side (2026-09-14 NVDA whipsaw);
  otherwise it is dropped. Note this makes signal_inverse a loss-taking exit by construction.
- A tip for a ticker already held on the same side is a no-op for the lab (it still counts in the
  journal and the consensus).
Exits: STOP_LOSS_PCT / the ratcheting stop / TAKE_PROFIT_PCT; no max-holding force-close in the lab
(removed 2026-09-14 at the user's request). The real strategy does have one -- see
mail_signal_real.py.

SIMPLIFICATION (documented, not hidden): a short's cash accounting mirrors a long's rather than
modeling real short-sale mechanics (borrow fees, margin calls) -- one more reason shorts stay in
the lab and the real strategy is long-only.
"""
import base64
import difflib
import html
import json
import os
import pathlib
import re
import sys
import time
import unicodedata
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.simulate_portfolio import (  # noqa: E402
    STOP_LOSS_PCT, RATCHET_STEP_PCT, RATCHET_GIVEBACK_PCT, reconcile_fresh_price,
)
from screener.simulate_constrained_portfolio import fetch_fx_rates, to_eur, fractional_eligible  # noqa: E402
from screener import mail_signal_scoring as scoring  # noqa: E402
from screener.mail_signal_real import run_real_layer  # noqa: E402

STATE_PATH = HERE / "results/screener/mail_signal_state.json"
CASH_PATH = HERE / "results/simulation/mail_signal_state.json"
LEDGER_PATH = HERE / "results/simulation/mail_signal_ledger.csv"
SUMMARY_PATH = HERE / "results/simulation/mail_signal_summary.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/mail_signal_equity_curve.csv"
SCORECARD_PATH = HERE / "results/screener/mail_signal_source_scorecard.csv"
REJECTS_PATH = HERE / "results/screener/mail_signal_rejects.csv"
ANNULLED_PATH = HERE / "results/simulation/mail_signal_annulled.csv"

TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
GMAIL_QUERY = "newer_than:2d"  # same buffer/reasoning as newsletter_digest.py
MAX_MESSAGES = 300
BODY_TRUNCATE = 900       # classification excerpt -- a yes/no call doesn't need more
TEXT_TRUNCATE = 6000      # full email text kept in memory for extraction + citation checks
EXTRACT_TRUNCATE = 1800   # what the extraction call actually sees (email text or fetched article) --
# was 3000 for one day (2026-10-03): on the CPU-only CI runner each extraction then exceeded the
# 240 s timeout and 3 runs in a row hit the 60-min job limit without saving anything.
MAX_REJECTS_KEPT = 1000
MAX_ATTEMPTS_PER_MAIL = 3  # an email whose Ollama calls keep failing/timing out is given up after this

BACKFILL_DAYS = 30
BACKFILL_MAX_PER_RUN = 15  # and only with whatever time is left after the live emails (see RUN_BUDGET)

# Time budget per run (CI job limit is 60 min, ~2 min of setup before this script starts and a few
# minutes of Yahoo work after the mail pass). New Ollama work is not STARTED past the deadline; an
# email not fully processed by then is simply left for the next run (it is not marked processed).
RUN_BUDGET_MIN = float(os.environ.get("MAIL_BOT_BUDGET_MIN", "40"))

# ARTICLE FETCH -- see module docstring. A domain not listed here is never fetched.
FETCHABLE_DOMAINS = ("zonebourse.com", "tradingsat.com")
ARTICLE_FETCH_TIMEOUT = 15
ARTICLE_FETCH_TRUNCATE = EXTRACT_TRUNCATE
ARTICLE_LINK_CANDIDATES = 5
ARTICLE_FETCH_MAX_WORKERS = 4
# A full "Chrome 120" UA without the matching Sec-* headers tripped zonebourse.com's bot-detection
# (403, tested 2026-09-16); this generic one passes on both fetchable domains.
_FETCH_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
_EXCLUDE_LINK_SUBSTR = ("mailto:", "unsubscribe", "preferences", "facebook.com", "twitter.com",
                         "x.com", "linkedin.com", "instagram.com", "youtube.com", "privacy")

STARTING_CAPITAL = 300.0        # lab pool -- see module docstring
STARTING_SLOTS = 9
TARGET_POSITION_SIZE = STARTING_CAPITAL / STARTING_SLOTS
MAX_WHOLE_SHARE_OVERSHOOT = 2.5
TRADE_FEE_EUR = 1.0             # per ORDER -- charged at entry and at exit since 2026-10-03

TAKE_PROFIT_PCT = 0.30
MIN_REVERSAL_CONFIRM_PCT = 0.02
MAX_TICKERS_PER_EMAIL = 3

_PLACEHOLDER_TICKERS = {"N/A", "NA", "NONE", "AUCUN", "AUCUNE", "INCONNU", "UNKNOWN", "TBD", "-", "?"}

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"  # same as newsletter_digest.py
OLLAMA_TIMEOUT = 300
# SEQUENTIAL calls (was 2 workers): on the CPU-only CI runner two concurrent requests just queue
# behind each other inside Ollama, so each one's wall time doubled and hit the timeout. Sequential
# calls of the SAME prompt type in a row also let Ollama reuse the cached prefix of the (long, fixed)
# instructions, which is why every prompt below puts the instructions FIRST and the email LAST --
# measured locally 2026-10-03: prompt evaluation ~60% faster from the second call on.
# temperature 0: extraction/verification are lookups, not creative writing, and the same email must
# give the same answer on a re-run. num_ctx 2048 fits instructions + an EXTRACT_TRUNCATE extract +
# the answer; num_predict bounds generation time (the slowest part on CPU).
OLLAMA_OPTIONS = {"temperature": 0, "num_ctx": 2048, "num_predict": 400}

# Newsletter platforms whose sending domain is shared by many unrelated newsletters -- see
# ATTRIBUTION in the module docstring.
PLATFORM_DOMAINS = ("beehiiv.com", "substack.com", "sailthru.com", "mailchimpapp.com", "mcsv.net",
                    "convertkit.com", "kit.com", "ghost.io", "mailerlite.com", "sendgrid.net",
                    "klaviyomail.com", "hubspotemail.net", "createsend.com", "list-manage.com")

# Primary exchanges (Yahoo codes -- the same exchange can appear under two codes depending on the
# endpoint, e.g. NYQ/NYSE). Excludes OTC (PNK, OTCM...) and Brazilian/other depositary listings.
MAJOR_EXCHANGES = {
    "NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "NYSE", "NASDAQ", "AMEX",           # US
    "PAR", "GER", "FRA", "LSE", "AMS", "MIL", "MCE", "VIE", "SWX", "EBS", "BRU", "LIS",  # Europe
    "ISE", "STO", "CPH", "HEL", "OSL",
    "TOR", "TSE", "TAI", "TWO", "HKG", "JPX", "KSC", "KOE",                              # other
}

# Subjects a model turns into a "company" although they are not one (2026-09-16 "Fed" -> FedEx,
# 2026-09-13 "BCE" = Banque centrale europeenne -> Bell Canada). Exact match on the normalized
# company name, plus the substrings below.
_MACRO_TERMS = {
    "bce", "fed", "la fed", "federal reserve", "reserve federale", "banque centrale europeenne", "ecb",
    "boe", "boj", "bank of japan", "bank of england", "pboc", "snb", "fmi", "imf", "opep", "opec", "ocde",
    "oecd", "usd", "eur", "dollar", "euro", "yen", "yuan", "bitcoin", "btc", "ethereum", "or", "gold",
    "petrole", "oil", "brent", "wti", "sp 500", "s p 500", "nasdaq", "nasdaq 100", "dow jones", "cac 40",
    "dax", "stoxx 600", "euro stoxx 50", "tresor", "treasury", "us treasury", "wall street", "france",
    "etats unis", "chine", "china", "japon", "allemagne", "europe", "usa",
}
_MACRO_SUBSTRINGS = ("banque centrale", "central bank", "reserve federale", "federal reserve")
# The model's own reason/citation saying there is no opinion (seen verbatim in 23 lab rows).
_NEGATION_PATTERNS = ("pas d'avis", "aucun avis", "sans avis", "pas mentionné", "non mentionné",
                      "pas d'information", "aucune information", "non renseigné", "pas de recommandation",
                      "aucune recommandation", "no explicit", "not mentioned", "no opinion")
# Page/ad boilerplate that was taken as tips (TipRanks footer -> TRKR/TSLA, BFM menu -> Rexel).
_BOILERPLATE_PATTERNS = ("with tipranks you can", "follow the expert of your choice", "devenez membre",
                         "se connecter", "rester connecté", "unsubscribe", "se désabonner", "privacy policy",
                         "terms of service", "paid advertisement", "this is a paid", "sponsored content",
                         "contenu sponsorisé", "publicité")
_BULLISH_WORDS = ("rating upgrade", "upgraded", "upgrades", "upgrade to buy", "strong buy", "buy rating",
                  "outperform", "overweight", "relevée à l'achat", "recommandation à l'achat", "conseil achat")
_BEARISH_WORDS = ("rating downgrade", "downgraded", "downgrades", "downgrade to sell", "strong sell",
                  "sell rating", "underperform", "underweight", "recommandation à la vente", "conseil vente")
# Legal-form and filler words ignored when comparing company names.
_NAME_STOP_TOKENS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "companies", "ltd", "limited", "plc", "sa",
    "se", "nv", "ag", "spa", "ab", "asa", "oyj", "lp", "llc", "group", "groupe", "holding", "holdings", "the",
    "de", "du", "la", "le", "les", "et", "and", "of", "class", "cl", "ord", "adr", "ads", "sponsored", "new",
    "com", "reit",
}
# Brand name -> legal-name token, when they share no word at all (kept tiny on purpose).
_NAME_ALIASES = {"google": "alphabet", "facebook": "meta", "instagram": "meta", "whatsapp": "meta"}

CLASSIFY_PROMPT = """L'email ci-dessous est-il une newsletter financiere/economique (actualite des marches, d'un secteur, ou macroeconomique) -- par opposition a un email personnel, professionnel, transactionnel, ou publicitaire non lie a la finance ?

Reponds UNIQUEMENT en JSON : {{"is_finance_newsletter": true|false, "reason": "<une phrase courte>"}}

Expediteur : {sender}
Sujet : {subject}
Extrait : {body}
"""

# ANALYSE vs ACTUALITE (2026-09-15) and CITATION / NO-GUESS TICKER (2026-10-03) -- see module
# docstring. The model no longer maps a company to a ticker itself: it only copies a ticker that is
# literally written in the extract; otherwise the code looks it up by name.
EXTRACT_TICKER_PROMPT = """Dans l'extrait de newsletter financiere donne a la fin, identifie chaque action d'une SOCIETE COTEE PRECISE ET NOMMEE qui fait l'objet d'un AVIS D'INVESTISSEMENT EXPLICITE : recommandation d'achat/vente, notation relevee/abaissee, objectif de cours, ou these d'investissement argumentee concluant a un fort potentiel de hausse ou a un risque de forte baisse.

N'INCLUS PAS :
(1) une action seulement mentionnee en passant ;
(2) une simple actualite (resultats, annonce, partenariat, mouvement de cours, proces...) sans avis d'investissement explicite -- meme pour une grande entreprise connue ;
(3) un sujet qui n'est pas une societe cotee precise : banque centrale (Fed, BCE), pays, devise, indice, matiere premiere, crypto, fonds/ETF, secteur en general ;
(4) la banque ou le courtier qui EMET l'avis (dans "Bank of America releve Nvidia a l'achat", la societe analysee est Nvidia, pas Bank of America) ;
(5) les publicites, menus, mentions legales, pieds de page et textes d'abonnement.

SENS DE L'AVIS : deduis-le de l'opinion exprimee, jamais du mouvement passe du cours. Une action qui a chute et qui est presentee comme une opportunite d'achat est "haussier". Une action qui a monte et qui est jugee trop chere est "baissier". Si l'extrait compare deux titres, seul celui qui est explicitement recommande ou deconseille compte -- l'autre n'est pas un avis.

Pour chaque action retenue (maximum {max_tickers}), donne :
- "company" : le nom de la societe EXACTEMENT tel qu'ecrit dans l'extrait
- "ticker" : le symbole boursier SEULEMENT s'il est ecrit tel quel dans l'extrait (ex: "(NVDA)"), sinon "" -- ne devine jamais un symbole
- "sentiment" : "haussier" ou "baissier"
- "citation" : la phrase de l'extrait qui exprime l'avis, COPIEE MOT POUR MOT (ne la reformule pas, ne la traduis pas)
- "author" : le nom de l'auteur de l'analyse SEULEMENT s'il est ecrit dans l'extrait, sinon ""

Si aucune action ne remplit ces conditions, reponds avec une liste vide.

Reponds UNIQUEMENT en JSON : {{"tips": [{{"company": "<NOM>", "ticker": "<SYMBOLE ou vide>", "sentiment": "haussier|baissier", "citation": "<phrase copiee>", "author": "<auteur ou vide>"}}, ...]}}

Sujet : {subject}
Extrait : {body}
"""

# Independent second opinion on ONE tip (2026-10-03): sees only the subject and the citation, and
# is not told which direction the extraction claimed -- so it can't just agree.
VERIFY_PROMPT = """Question : la citation de newsletter financiere donnee a la fin contient-elle un AVIS D'INVESTISSEMENT EXPLICITE sur l'action de la societe nommee a la fin, elle-meme ? Un avis d'investissement = recommandation d'achat ou de vente, notation relevee ou abaissee, objectif de cours, ou conclusion argumentee sur le potentiel de hausse ou le risque de baisse de l'action.
Ce n'est PAS un avis : une simple actualite (resultats, partenariat, contrat, proces, nomination), un mouvement de cours passe sans opinion, un avis sur une autre societe, une publicite.

Si c'est un avis, quel est son sens ? "haussier" (acheter, potentiel de hausse -- y compris une baisse passee presentee comme une opportunite d'achat) ou "baissier" (vendre, risque de baisse).

Reponds UNIQUEMENT en JSON : {{"avis_explicite": true|false, "sens": "haussier|baissier|aucun", "raison": "<une phrase courte>"}}

Societe : {company}
Sujet de la newsletter : {subject}
Citation : "{citation}"
"""


def _call_ollama_json(prompt: str) -> dict:
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json", "keep_alive": "20m",
               "options": OLLAMA_OPTIONS}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    outer = json.loads(resp.content)
    return json.loads(outer["response"])


def _load_json(path: pathlib.Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return default
    return default


def _save_json(path: pathlib.Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# Gmail
# ---------------------------------------------------------------------------------------------

def get_access_token() -> str:
    resp = requests.post(TOKEN_URL, data={
        "refresh_token": os.environ["GMAIL_REFRESH_TOKEN"],
        "client_id": os.environ["GMAIL_CLIENT_ID"],
        "client_secret": os.environ["GMAIL_CLIENT_SECRET"],
        "grant_type": "refresh_token",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def list_message_ids(token: str, query: str, max_messages: int = MAX_MESSAGES) -> list[str]:
    ids = []
    params = {"q": query, "maxResults": min(max_messages, 500)}
    headers = {"Authorization": f"Bearer {token}"}
    while True:
        resp = requests.get(f"{GMAIL_API}/messages", params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        ids.extend(m["id"] for m in data.get("messages", []))
        if len(ids) >= max_messages or "nextPageToken" not in data:
            break
        params["pageToken"] = data["nextPageToken"]
    return ids[:max_messages]


def _part_charset(part: dict) -> str:
    for h in part.get("headers", []) or []:
        if h.get("name", "").lower() == "content-type":
            m = re.search(r'charset="?([\w-]+)"?', h.get("value", ""), re.IGNORECASE)
            if m:
                return m.group(1)
    return "utf-8"


def _decode_part(data_b64url: str, charset: str = "utf-8") -> str:
    raw = base64.urlsafe_b64decode(data_b64url + "=" * (-len(data_b64url) % 4))
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _html_to_text(page_html: str) -> str:
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", cleaned)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _extract_text(payload: dict) -> str:
    """text/plain part if there is one, else the HTML part converted to text (script/style blocks
    removed and entities unescaped since 2026-10-03 -- an HTML-only newsletter used to hand the
    model its CSS as the first 900 characters)."""
    stack = [payload]
    html_fallback = None
    while stack:
        part = stack.pop()
        mime = part.get("mimeType", "")
        body_data = part.get("body", {}).get("data")
        if mime == "text/plain" and body_data:
            return re.sub(r"[ \t]+", " ", _decode_part(body_data, _part_charset(part))).strip()
        if mime == "text/html" and body_data and html_fallback is None:
            html_fallback = _decode_part(body_data, _part_charset(part))
        stack.extend(part.get("parts", []) or [])
    return _html_to_text(html_fallback) if html_fallback else ""


def _extract_source(sender: str) -> str:
    """Sending domain only -- never the local part / full address (see ATTRIBUTION)."""
    m = re.search(r'@([\w.-]+\.[A-Za-z]{2,})', sender or "")
    return m.group(1).lower() if m else "inconnu"


def _publication(sender: str, domain: str) -> str:
    """Scoring key of a newsletter -- see ATTRIBUTION in the module docstring."""
    platform = next((p for p in PLATFORM_DOMAINS if domain == p or domain.endswith("." + p)), None)
    if not platform:
        return domain
    name = re.sub(r"<[^>]*>", "", sender or "")
    name = re.sub(r"\S*@\S*", "", name).strip().strip('"\'').strip()
    name = re.sub(r"\s+", " ", name)[:60]
    return f"{name} ({platform})" if name else domain


def _extract_html_part(payload: dict) -> str:
    stack = [payload]
    while stack:
        part = stack.pop()
        if part.get("mimeType") == "text/html":
            body_data = part.get("body", {}).get("data")
            if body_data:
                return _decode_part(body_data, _part_charset(part))
        stack.extend(part.get("parts", []) or [])
    return ""


def _extract_article_links(html_body: str) -> list[str]:
    """Plausible "read more" links copied verbatim from the email, in order."""
    if not html_body:
        return []
    out, seen = [], set()
    for link in re.findall(r'href=["\']((?:https?:)?//[^"\']+)', html_body, re.IGNORECASE):
        if link.startswith("//"):
            link = "https:" + link
        low = link.lower()
        if any(x in low for x in _EXCLUDE_LINK_SUBSTR) or link in seen:
            continue
        seen.add(link)
        out.append(link)
    return out


def _extract_article_text_from_html(page_html: str) -> str:
    """<article> region if any (whole page otherwise), script/style stripped first --
    zonebourse.com embeds a large JS blob at the top of its <article>. Validated by hand against
    real zonebourse.com/tradingsat.com pages, 2026-09-16."""
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", page_html, flags=re.DOTALL | re.IGNORECASE)
    start = cleaned.find("<article")
    if start == -1:
        chunk = cleaned
    else:
        end = cleaned.find("</article>", start)
        chunk = cleaned[start:end] if end != -1 else cleaned[start:]
    text = re.sub(r"<[^>]+>", " ", chunk)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_article_extract(links: list[str]) -> str | None:
    """Full article text from the first link that resolves to a FETCHABLE_DOMAINS page, or None."""
    for link in links[:ARTICLE_LINK_CANDIDATES]:
        try:
            resp = requests.get(link, headers=_FETCH_HEADERS, timeout=ARTICLE_FETCH_TIMEOUT, allow_redirects=True)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        host = urllib.parse.urlsplit(resp.url).netloc.lower()
        if not any(host == d or host.endswith("." + d) for d in FETCHABLE_DOMAINS):
            continue
        text = _extract_article_text_from_html(resp.content.decode(resp.encoding or "utf-8", errors="replace"))
        if len(text) > 200:
            return text[:ARTICLE_FETCH_TRUNCATE]
    return None


def fetch_message(token: str, message_id: str) -> dict | None:
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(f"{GMAIL_API}/messages/{message_id}", params={"format": "full"},
                         headers=headers, timeout=30)
    if resp.status_code != 200:
        return None
    data = resp.json()
    hdrs = data.get("payload", {}).get("headers", [])

    def get_header(name):
        return next((h["value"] for h in hdrs if h["name"].lower() == name.lower()), "")

    payload = data.get("payload", {})
    text = _extract_text(payload)[:TEXT_TRUNCATE]
    sender = get_header("From")
    domain = _extract_source(sender)
    try:
        mail_date = datetime.fromtimestamp(int(data["internalDate"]) / 1000, tz=timezone.utc)
    except (KeyError, ValueError, TypeError):
        mail_date = datetime.now(timezone.utc)
    return {"id": message_id, "sender": sender, "subject": get_header("Subject"), "source": domain,
            "publication": _publication(sender, domain), "date_utc": mail_date.isoformat(),
            "body": text[:BODY_TRUNCATE], "text": text, "html": _extract_html_part(payload)}


def classify_newsletter(msg: dict) -> bool | None:
    """True/False, or None if Ollama itself failed (see process_messages: an all-None batch means
    Ollama is down, and those emails must NOT be marked processed)."""
    prompt = CLASSIFY_PROMPT.format(sender=msg["sender"], subject=msg["subject"], body=msg["body"])
    try:
        raw = _call_ollama_json(prompt)
        return bool(raw.get("is_finance_newsletter", False))
    except Exception as e:
        print(f"  echec classification Ollama pour un mail: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------------------------
# Tip extraction + validation (2026-10-03 -- see module docstring)
# ---------------------------------------------------------------------------------------------

def _norm_text(s: str) -> str:
    """ASCII-folded, lowercase, punctuation -> space, whitespace collapsed. Apostrophes are
    dropped (not spaced) so "Dick's"/"L'Oreal" compare the same on both sides."""
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"['’`]s", "", s)  # possessive: "AMD's" -> "amd"
    s = re.sub(r"['’`]", "", s)
    s = s.replace("&", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# Patterns above are written naturally; matched in the same normalized form as the text.
_NEGATION_PATTERNS = tuple(_norm_text(p) for p in _NEGATION_PATTERNS)
_BOILERPLATE_PATTERNS = tuple(_norm_text(p) for p in _BOILERPLATE_PATTERNS)
_BULLISH_WORDS = tuple(_norm_text(p) for p in _BULLISH_WORDS)
_BEARISH_WORDS = tuple(_norm_text(p) for p in _BEARISH_WORDS)
_MACRO_TERMS = {_norm_text(p) for p in _MACRO_TERMS}
_MACRO_SUBSTRINGS = tuple(_norm_text(p) for p in _MACRO_SUBSTRINGS)


def _name_tokens(name: str, drop_stops: bool = True) -> list[str]:
    toks = [t for t in _norm_text(name).split() if len(t) >= 2]
    return [t for t in toks if t not in _NAME_STOP_TOKENS] if drop_stops else toks


def _names_match(company: str, listed_name: str) -> bool:
    """Is `listed_name` (Yahoo) the company the email names? Every significant word of the
    company must be in the listed name, and cover at least half of it (so "Mistral" no longer
    matches "Mistral Iberia Real Estate Socimi" -- 2026-09-16 lab row); or the company is an
    acronym of the listed name ("TSMC", "AMD"). Callers try both Yahoo's long and short name
    ("LVMH" is the short name of a 7-word long name)."""
    ct = {_NAME_ALIASES.get(t, t) for t in _name_tokens(company)}
    rt = set(_name_tokens(listed_name))
    if not ct or not rt:
        return False
    if ct <= rt and (len(rt) <= 2 or len(ct) / len(rt) >= 0.5):
        return True
    if rt <= ct:
        return True
    if len(ct) == 1:
        acro = next(iter(ct))
        initials = "".join(t[0] for t in _name_tokens(listed_name, drop_stops=False))
        if 2 <= len(acro) <= 6 and initials.startswith(acro):
            return True
    return False


def _citation_span(citation: str, text: str):
    """(start, end) of the citation inside _norm_text(text), or None if it isn't really there
    (exact, or a single contiguous block covering 80%+ of it -- tolerates a dropped quote mark,
    not a paraphrase)."""
    c, t = _norm_text(citation), _norm_text(text)
    if len(c) < 15 or not t:
        return None
    pos = t.find(c)
    if pos != -1:
        return pos, pos + len(c)
    m = difflib.SequenceMatcher(None, c, t, autojunk=False).find_longest_match(0, len(c), 0, len(t))
    return (m.b - m.a, m.b - m.a + len(c)) if m.size >= 0.8 * len(c) else None


# How far around the citation the company must be named. A citation that only says "we rate the
# stock a Buy" belongs to whichever company is named right before/after it -- tested 2026-10-03:
# llama3.1:8b attached a Chevron sentence to Nvidia, named 300+ characters further down the digest.
CITATION_CONTEXT_BEFORE = 250
CITATION_CONTEXT_AFTER = 150


def _ticker_in_text(ticker: str, text: str) -> bool:
    return bool(ticker) and re.search(rf"(?<![A-Za-z0-9.]){re.escape(ticker)}(?![A-Za-z0-9])", text) is not None


def _validate_tip(tip: dict, text: str) -> tuple:
    """Deterministic checks on one raw tip -- returns (cleaned tip, None) or (None, reason)."""
    company = str(tip.get("company") or "").strip()
    sentiment = tip.get("sentiment")
    citation = str(tip.get("citation") or "").strip()
    ticker = str(tip.get("ticker") or "").strip().upper()
    if not company or sentiment not in ("haussier", "baissier"):
        return None, "societe ou sens manquant"
    nc = _norm_text(company)
    if nc in _MACRO_TERMS or any(s in nc for s in _MACRO_SUBSTRINGS) or company.upper() in _PLACEHOLDER_TICKERS:
        return None, f"'{company}' n'est pas une societe cotee"
    ncit = _norm_text(citation)
    if any(p in ncit for p in _NEGATION_PATTERNS):
        return None, "la citation dit elle-meme qu'il n'y a pas d'avis"
    if any(p in ncit for p in _BOILERPLATE_PATTERNS):
        return None, "texte publicitaire ou de page, pas un avis"
    span = _citation_span(citation, text)
    if span is None:
        return None, "citation introuvable mot pour mot dans le mail"
    if ticker in _PLACEHOLDER_TICKERS or " " in ticker:
        ticker = ""
    if ticker and not _ticker_in_text(ticker, text):
        ticker = ""  # not literally in the email -> never trusted, looked up by name instead
    norm = _norm_text(text)
    context = set(norm[max(0, span[0] - CITATION_CONTEXT_BEFORE):span[1] + CITATION_CONTEXT_AFTER].split())
    company_tokens = [t for t in _name_tokens(company) if len(t) >= 3] or _name_tokens(company)
    if not any(t in context for t in company_tokens) and not (ticker and ticker.lower() in context):
        return None, f"'{company}' n'est pas nomme a cote de la citation (avis sur une autre societe ?)"
    bull = any(w in ncit for w in _BULLISH_WORDS)
    bear = any(w in ncit for w in _BEARISH_WORDS)
    if sentiment == "haussier" and bear and not bull:
        return None, "citation explicitement baissiere (downgrade...) pour un tip haussier"
    if sentiment == "baissier" and bull and not bear:
        return None, "citation explicitement haussiere (upgrade...) pour un tip baissier"
    author = str(tip.get("author") or "").strip()
    if author and _norm_text(author) not in _norm_text(text):
        author = ""
    return {"company": company, "ticker_in_text": ticker, "sentiment": sentiment, "citation": citation,
            "author": author[:80]}, None


def _extract_raw_tips(msg: dict) -> list[dict] | None:
    prompt = EXTRACT_TICKER_PROMPT.format(subject=msg["subject"], body=msg["extract_text"],
                                           max_tickers=MAX_TICKERS_PER_EMAIL)
    try:
        raw = _call_ollama_json(prompt)
    except Exception as e:
        print(f"  echec extraction tickers pour \"{msg['subject']}\": {e}", file=sys.stderr)
        return None  # retried next run, unlike a genuine empty answer
    tips = raw.get("tips") or []
    return [t for t in tips[:MAX_TICKERS_PER_EMAIL] if isinstance(t, dict)]


def _verify_tip(subject: str, tip: dict) -> tuple:
    """Second, independent Ollama opinion -- see VERIFY_PROMPT. (True|False, reason), or (None, ...)
    if Ollama itself failed -- the email is then retried next run rather than its tip dropped."""
    prompt = VERIFY_PROMPT.format(subject=subject, citation=tip["citation"][:600], company=tip["company"])
    try:
        raw = _call_ollama_json(prompt)
    except Exception as e:
        print(f"  echec verification pour \"{subject}\": {e}", file=sys.stderr)
        return None, f"verification impossible ({e})"
    if not raw.get("avis_explicite"):
        return False, f"verification : pas d'avis explicite ({str(raw.get('raison', ''))[:120]})"
    if raw.get("sens") != tip["sentiment"]:
        return False, f"verification : sens contradictoire ({raw.get('sens')} vs {tip['sentiment']})"
    return True, None


def _fetch_price(symbol: str) -> dict | None:
    """Last price + trading currency, no validation -- for symbols already validated (rechecks)."""
    try:
        tk = yf.Ticker(symbol)
        hist = tk.history(period="5d")["Close"].dropna()
        if hist.empty:
            return None
        currency = None
        try:
            currency = tk.fast_info.get("currency")
        except Exception:
            pass
        if not currency:
            try:
                currency = (tk.history_metadata or {}).get("currency")
            except Exception:
                pass
        return {"price": float(hist.iloc[-1]), "currency": currency}
    except Exception as e:
        print(f"  echec prix {symbol}: {e}", file=sys.stderr)
        return None


def _search_quotes(query: str) -> list[dict]:
    try:
        return yf.Search(query, max_results=10).quotes or []
    except Exception as e:
        print(f"  echec recherche Yahoo \"{query}\": {e}", file=sys.stderr)
        return []


def _resolve_instrument(company: str, ticker_in_text: str) -> tuple:
    """GROUNDING BACKSTOP -- returns ({ticker, price, currency, name, exchange}, None) or
    (None, reason). Candidates: the ticker only if literally written in the email (exact symbol
    match on Yahoo), then Yahoo's own search by the company name copied from the email. Each must
    be an EQUITY on a primary exchange whose Yahoo name matches the company. 2026-09-15 note still
    holds: never search by a guessed ticker ("TSMC" search ranks an unrelated Italian OTC stock
    above Taiwan Semiconductor)."""
    candidates = []
    if ticker_in_text:
        candidates += [q for q in _search_quotes(ticker_in_text) if str(q.get("symbol", "")).upper() == ticker_in_text][:1]
    candidates += _search_quotes(company)
    first_reason, tried, hopped = None, set(), False
    while candidates:
        q = candidates.pop(0)
        sym = q.get("symbol")
        if not sym or sym in tried:
            continue
        tried.add(sym)
        names = [n for n in (q.get("longname"), q.get("shortname")) if n]
        if q.get("quoteType") != "EQUITY":
            reason = f"{sym} n'est pas une action ({q.get('quoteType')})"
        elif str(q.get("exchange", "")).upper() not in MAJOR_EXCHANGES:
            reason = f"{sym} cote hors place principale ({q.get('exchange')})"
            # Right company, secondary listing only (e.g. "TSMC" -> TSMC34.SA, a Sao Paulo
            # receipt): search once more by Yahoo's own long name of that listing -- a name Yahoo
            # gave us, not one we guessed -- to reach the primary listing (TSM).
            if not hopped and q.get("longname") and any(_names_match(company, n) for n in names):
                hopped = True
                candidates += _search_quotes(q["longname"])
        elif not any(_names_match(company, n) for n in names):
            reason = f"{sym} = '{names[0] if names else '?'}' ne correspond pas a '{company}'"
        else:
            px = _fetch_price(sym)
            if px is None:
                reason = f"{sym} sans prix"
            else:
                if first_reason:
                    print(f"  ticker corrige pour '{company}' : {first_reason} -> {sym}")
                return {"ticker": sym, "price": px["price"], "currency": px["currency"],
                        "name": names[0] if names else sym, "exchange": str(q.get("exchange", "")).upper()}, None
        first_reason = first_reason or reason
    return None, first_reason or f"aucune cotation trouvee pour '{company}'"


def process_messages(token: str, ids: list[str], deadline: float, backfill: bool = False,
                     attempts: dict | None = None) -> tuple:
    """Full pipeline for a batch of Gmail ids -> (validated signals, rejected tips, ids DONE).
    Phase by phase (classify all, then extract all, then verify all -- same prompt type in a row so
    Ollama reuses its cached instructions), one Ollama call at a time, and no call STARTED after
    `deadline` (time.monotonic()). An id is DONE only once it is fully handled (not a newsletter, or
    every tip of it extracted, verified and resolved); the rest is retried next run, up to
    MAX_ATTEMPTS_PER_MAIL times (`attempts`, persisted by the caller)."""
    attempts = attempts if attempts is not None else {}
    msgs = [m for m in (fetch_message(token, mid) for mid in ids) if m is not None]
    msgs.sort(key=lambda m: m["date_utc"])  # oldest first: they are closest to leaving the listing window
    done, failed = set(), set()
    label = "rattrapage" if backfill else "nouveau(x)"

    def out_of_time() -> bool:
        return time.monotonic() >= deadline

    classified = []
    n_errors = 0
    for m in msgs:
        if out_of_time():
            break
        verdict = classify_newsletter(m)
        if verdict is None:
            n_errors += 1
            failed.add(m["id"])
        elif verdict:
            classified.append(m)
        else:
            done.add(m["id"])
    if msgs and n_errors and not done and not classified:
        raise RuntimeError("Ollama n'a repondu a aucune classification -- mails laisses non traites")
    print(f"{len(msgs)} mail(s) {label} a examiner, {len(done) + len(classified)} classe(s) ce run, "
          f"{len(classified)} newsletter(s) financiere(s).")

    with ThreadPoolExecutor(max_workers=ARTICLE_FETCH_MAX_WORKERS) as ex:  # plain HTTP, no Ollama
        fetched = dict(zip((m["id"] for m in classified),
                            ex.map(lambda m: _fetch_article_extract(_extract_article_links(m.get("html", ""))),
                                   classified)))
    for m in classified:
        m["extract_text"] = fetched.get(m["id"]) or m["text"][:EXTRACT_TRUNCATE]

    extracted = []
    for m in classified:
        if out_of_time():
            break
        tips = _extract_raw_tips(m)
        if tips is None:
            failed.add(m["id"])
        else:
            extracted.append((m, tips))

    rejects, signals, seen = [], [], set()
    for m, tips in extracted:
        complete = True
        verified = []
        for tip in tips:
            # citation checked against what the model saw AND the email itself (an article tip may
            # quote the email's teaser)
            clean, reason = _validate_tip(tip, m["extract_text"] + "\n" + m["text"])
            if clean is None:
                rejects.append(_reject_row(m, tip, reason))
                continue
            if out_of_time():
                complete = False
                break
            ok, reason = _verify_tip(m["subject"], clean)
            if ok is None:
                complete = False
                failed.add(m["id"])
                break
            if not ok:
                rejects.append(_reject_row(m, clean, reason))
            else:
                verified.append(clean)
        if not complete:
            continue  # retried next run -- its rejects so far are dropped too, to avoid duplicates
        for tip in verified:
            resolved, reason = _resolve_instrument(tip["company"], tip["ticker_in_text"])
            if resolved is None:
                rejects.append(_reject_row(m, tip, reason))
                continue
            if (m["id"], resolved["ticker"]) in seen:
                continue
            seen.add((m["id"], resolved["ticker"]))
            signals.append({
                "message_id": m["id"], "mail_date_utc": m["date_utc"], "publication": m["publication"],
                "domain": m["source"], "author": tip["author"], "company": tip["company"],
                "ticker": resolved["ticker"], "name": resolved["name"], "exchange": resolved["exchange"],
                "currency": resolved["currency"], "price": resolved["price"],
                "side": "long" if tip["sentiment"] == "haussier" else "short",
                "citation": tip["citation"], "backfill": backfill,
            })
        done.add(m["id"])
    rejects = [r for r in rejects if r["_id"] in done]

    for mid in failed - done:
        attempts[mid] = attempts.get(mid, 0) + 1
        if attempts[mid] >= MAX_ATTEMPTS_PER_MAIL:
            print(f"  mail abandonne apres {attempts[mid]} echecs Ollama", file=sys.stderr)
            done.add(mid)
    pending = len(msgs) - len(done)
    if pending:
        print(f"  {pending} mail(s) {label} reporte(s) au prochain run (budget de temps ou echec Ollama).")
    for r in rejects:
        print(f"  rejete [{r['publication']}] {r['company']} ({r['sentiment']}) : {r['motif']}")
    return signals, rejects, done


def _reject_row(m: dict, tip: dict, reason: str) -> dict:
    return {"_id": m["id"], "date_utc": m["date_utc"], "publication": m["publication"], "company": str(tip.get("company") or "")[:80],
            "ticker": str(tip.get("ticker") or tip.get("ticker_in_text") or "")[:12],
            "sentiment": tip.get("sentiment"), "citation": str(tip.get("citation") or "")[:300], "motif": reason}


def append_rejects(rejects: list[dict]):
    if not rejects:
        return
    cols = ["date_utc", "publication", "company", "ticker", "sentiment", "citation", "motif"]
    rejects = [{k: r[k] for k in cols} for r in rejects]
    old = pd.read_csv(REJECTS_PATH) if REJECTS_PATH.exists() else pd.DataFrame(columns=cols)
    out = pd.concat([old, pd.DataFrame(rejects, columns=cols)], ignore_index=True).tail(MAX_REJECTS_KEPT)
    REJECTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(REJECTS_PATH, index=False)


# ---------------------------------------------------------------------------------------------
# Lab ledger
# ---------------------------------------------------------------------------------------------

LEDGER_COLUMNS = [
    "ticker", "name", "side", "source", "status", "currency", "fractional",
    "entry_date", "entry_price", "shares", "entry_value_eur", "entry_fee_eur",
    "last_check_date", "last_price", "current_value_eur", "unrealized_return_pct",
    "peak_unrealized_return_pct", "peak_date",
    "exit_date", "exit_price", "exit_reason", "exit_value_eur", "return_pct", "holding_days",
    "signal_reason",
]


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH)
        for c in LEDGER_COLUMNS:
            if c not in df.columns:
                df[c] = None
        df = df[LEDGER_COLUMNS]
    else:
        df = pd.DataFrame(columns=LEDGER_COLUMNS)
    for c in ("ticker", "name", "side", "source", "status", "currency", "entry_date", "last_check_date",
              "peak_date", "exit_date", "exit_reason", "signal_reason"):
        df[c] = df[c].astype(object)
    return df


def load_cash() -> float:
    return _load_json(CASH_PATH, {"cash_eur": STARTING_CAPITAL})["cash_eur"]


def save_cash(cash: float):
    _save_json(CASH_PATH, {"cash_eur": cash})


def _unrealized_return(side: str, entry_price: float, last_price: float) -> float:
    raw = last_price / entry_price - 1
    return raw if side == "long" else -raw


def _ensure_fx(currency, fx_rates: dict) -> dict:
    """Extends fx_rates in place with whichever currency a position needs (tickers come from
    arbitrary mail tips, not a pre-scoped universe)."""
    key = "GBP" if currency == "GBp" else (currency if isinstance(currency, str) and currency else "EUR")
    if key not in fx_rates:
        fx_rates.update(fetch_fx_rates({key}))
    return fx_rates


def _value_eur(ledger: pd.DataFrame, idx, side: str, price: float, fx_rates: dict):
    """Current EUR value at today's FX rate (None if no rate). Entry price in EUR is recovered as
    entry_value_eur / shares (exact for both the fractional and whole-share sizing paths)."""
    currency = ledger.at[idx, "currency"]
    _ensure_fx(currency, fx_rates)
    price_eur = to_eur(price, currency if isinstance(currency, str) else None, fx_rates)
    shares, entry_value = float(ledger.at[idx, "shares"]), float(ledger.at[idx, "entry_value_eur"])
    if price_eur is None or shares <= 0:
        return None
    if side == "long":
        return shares * price_eur
    return entry_value * (2 - price_eur / (entry_value / shares))  # short: mirror of a long


def _close_lab(ledger, idx, today, price, value_eur, reason) -> float:
    entry_value = float(ledger.at[idx, "entry_value_eur"])
    entry_fee = float(ledger.at[idx, "entry_fee_eur"]) if pd.notna(ledger.at[idx, "entry_fee_eur"]) else 0.0
    net_exit_value = value_eur - TRADE_FEE_EUR
    net_return = (net_exit_value - entry_value - entry_fee) / entry_value
    ledger.at[idx, "status"] = "closed"
    ledger.at[idx, "exit_date"] = today
    ledger.at[idx, "exit_price"] = price
    ledger.at[idx, "exit_reason"] = reason
    ledger.at[idx, "exit_value_eur"] = net_exit_value
    ledger.at[idx, "return_pct"] = net_return
    ledger.at[idx, "holding_days"] = (pd.Timestamp(today) - pd.Timestamp(ledger.at[idx, "entry_date"])).days
    print(f"  CLOTURE {str(ledger.at[idx, 'side']).upper()} {ledger.at[idx, 'ticker']} : {reason}, "
          f"retour net {net_return:+.1%} (frais d'entree et de sortie deduits)")
    return net_exit_value


def recheck_and_exit(ledger: pd.DataFrame, today: str, cash: float, fx_rates: dict) -> tuple:
    for idx in ledger.index[ledger["status"] == "open"]:
        ticker = ledger.at[idx, "ticker"]
        side = ledger.at[idx, "side"]
        px = _fetch_price(ticker)
        if px is None:
            continue  # transient fetch failure -- retry next run, don't force an exit on it
        price_check, split_factor = reconcile_fresh_price(ticker, px["price"], ledger.at[idx, "last_price"],
                                                          ledger.at[idx, "last_check_date"])
        if price_check == "suspect":
            continue
        if price_check == "split":
            ledger.at[idx, "entry_price"] = ledger.at[idx, "entry_price"] / split_factor
            ledger.at[idx, "shares"] = ledger.at[idx, "shares"] * split_factor

        # stops on the LOCAL price move, like a broker stop; value in EUR at today's FX rate
        unrealized = _unrealized_return(side, ledger.at[idx, "entry_price"], px["price"])
        current_value = _value_eur(ledger, idx, side, px["price"], fx_rates)
        if current_value is None:
            current_value = float(ledger.at[idx, "entry_value_eur"]) * (1 + unrealized)

        ledger.at[idx, "last_check_date"] = today
        ledger.at[idx, "last_price"] = px["price"]
        ledger.at[idx, "current_value_eur"] = current_value
        ledger.at[idx, "unrealized_return_pct"] = unrealized

        peak = ledger.at[idx, "peak_unrealized_return_pct"]
        if pd.isna(peak) or unrealized > peak:
            ledger.at[idx, "peak_unrealized_return_pct"] = unrealized
            ledger.at[idx, "peak_date"] = today
        peak = ledger.at[idx, "peak_unrealized_return_pct"]

        stop_loss_hit = unrealized <= STOP_LOSS_PCT
        take_profit_hit = unrealized >= TAKE_PROFIT_PCT
        milestone = int(peak // RATCHET_STEP_PCT) if pd.notna(peak) else 0
        trailing_stop_hit = milestone >= 1 and unrealized <= milestone * RATCHET_STEP_PCT - RATCHET_GIVEBACK_PCT
        if stop_loss_hit or take_profit_hit or trailing_stop_hit:
            reason = ("trailing_stop" if trailing_stop_hit else
                      "stop_loss" if stop_loss_hit else "take_profit")
            cash += _close_lab(ledger, idx, today, px["price"], current_value, reason)
    return ledger, cash


def _open_position(ledger: pd.DataFrame, sig: dict, cash: float, today: str, fx_rates: dict) -> tuple:
    """No cash/affordability gate on purpose -- see LAB TRADING in the module docstring."""
    _ensure_fx(sig.get("currency"), fx_rates)
    price_eur = to_eur(sig["price"], sig.get("currency"), fx_rates)
    if price_eur is None or price_eur <= 0:
        return ledger, cash, False
    ticker = sig["ticker"]
    fractional = fractional_eligible(ticker, None, None)
    if fractional:
        cost = TARGET_POSITION_SIZE
        shares = cost / price_eur
    else:
        if price_eur > MAX_WHOLE_SHARE_OVERSHOOT * TARGET_POSITION_SIZE:
            return ledger, cash, False
        shares = max(1, int(TARGET_POSITION_SIZE // price_eur))
        cost = shares * price_eur

    new_row = {c: None for c in LEDGER_COLUMNS}
    new_row.update({
        "ticker": ticker, "name": sig.get("name") or ticker, "side": sig["side"], "source": sig["publication"],
        "status": "open", "currency": sig.get("currency"), "fractional": bool(fractional),
        "entry_date": today, "entry_price": sig["price"], "shares": shares, "entry_value_eur": cost,
        "entry_fee_eur": TRADE_FEE_EUR, "last_check_date": today, "last_price": sig["price"],
        "current_value_eur": cost, "unrealized_return_pct": 0.0, "peak_unrealized_return_pct": 0.0,
        "peak_date": today, "signal_reason": sig["citation"][:300],
    })
    ledger = pd.concat([ledger, pd.DataFrame([new_row], columns=LEDGER_COLUMNS)], ignore_index=True)
    cash -= cost + TRADE_FEE_EUR
    kind = "fractionne" if fractional else "entier"
    print(f"  OUVERTURE {sig['side'].upper()} {ticker} ({sig['publication']}) : {cost:.2f} EUR "
          f"({shares:.4f} actions, {kind}) @ {sig['price']:.2f} {sig.get('currency') or '?'}")
    return ledger, cash, True


def apply_signals(ledger: pd.DataFrame, signals: list[dict], cash: float, today: str, fx_rates: dict) -> tuple:
    """Lab decision tree (not held / same side / opposite side) for tips that were JUST added to
    the journal -- already validated and resolved to their canonical symbol."""
    for sig in signals:
        ticker, side = sig["ticker"], sig["side"]
        open_row = ledger[(ledger["ticker"] == ticker) & (ledger["status"] == "open")]
        if len(open_row):
            existing_side = open_row.iloc[0]["side"]
            if existing_side == side:
                continue
            idx = open_row.index[0]
            unrealized = _unrealized_return(existing_side, ledger.at[idx, "entry_price"], sig["price"])
            if unrealized > -MIN_REVERSAL_CONFIRM_PCT:
                continue  # see 2026-09-14 NVDA whipsaw note in the module docstring
            value = _value_eur(ledger, idx, existing_side, sig["price"], fx_rates)
            if value is None:
                value = float(ledger.at[idx, "entry_value_eur"]) * (1 + unrealized)
            cash += _close_lab(ledger, idx, today, sig["price"], value, "signal_inverse")
        ledger, cash, _ = _open_position(ledger, sig, cash, today, fx_rates)
    return ledger, cash


# Lab rows found wrong in the 2026-10-03 hand review that no automatic rule below would catch
# (wrong company behind a real equity ticker). Keyed by (ticker, entry_date, source).
MANUAL_ANNULMENTS = {
    ("CHEV", "2026-09-13", "seekingalpha.com"): "l'article parlait de Chevron (CVX), pas de Charging Robotics (CHEV)",
    ("BCE", "2026-09-13", "news.meilleurtaux.com"): "BCE = Banque centrale europeenne, pas Bell Canada",
    ("YMIB.MC", "2026-09-16", "aktionnaire.com"): "Mistral (IA, non cotee) confondu avec Mistral Iberia Real Estate",
    ("CS.PA", "2026-09-23", "aktionnaire.com"): "aucun avis sur AXA (article sur un prix litteraire)",
    ("B", "2026-09-27", "seekingalpha.com"): "Barrick sans rapport avec l'article (puces Nvidia en Chine)",
    ("ABBV", "2026-09-18", "analystratings.net"): "publicite sur l'IA/robotique, AbbVie sans rapport",
    ("BLK", "2026-09-17", "analystratings.net"): "publicite ; BlackRock y est un actionnaire cite, pas le titre recommande",
    ("BAC", "2026-09-15", "seekingalpha.com"): "Bank of America est l'analyste (semi-conducteurs), pas le titre analyse",
    ("BAC", "2026-09-24", "substack.com"): "estimation de BofA sur les bons du Tresor, aucun avis sur l'action",
    ("NDX", "2026-09-17", "tipranks.com"): "indice Nasdaq 100, pas une action",
}


def _quote_type(symbol: str) -> str | None:
    for q in _search_quotes(symbol):
        if str(q.get("symbol", "")).upper() == symbol.upper():
            return q.get("quoteType")
    try:
        return yf.Ticker(symbol).info.get("quoteType")
    except Exception:
        return None


def cleanup_legacy_rows(ledger: pd.DataFrame, cash: float, today: str) -> tuple:
    """One-off (state flag) application of the 2026-10-03 rules to lab rows opened before them:
    non-equity instruments, reasons that deny being an opinion / page boilerplate / empty, and
    MANUAL_ANNULMENTS. Same treatment as the 2026-09-16 FDX/AF.PA cleanup -- the trade never
    really existed, so cash is restored as if it had never been opened (open row: + entry value;
    closed row: + entry value - exit value, i.e. its realized P&L is reversed). Archived with the
    reason in mail_signal_annulled.csv rather than silently deleted."""
    reasons = {}
    for idx, r in ledger.iterrows():
        key = (r["ticker"], str(r["entry_date"]), r["source"])
        if key in MANUAL_ANNULMENTS:
            reasons[idx] = "revue manuelle 2026-10-03 : " + MANUAL_ANNULMENTS[key]
            continue
        reason_txt = _norm_text(r["signal_reason"]) if isinstance(r["signal_reason"], str) else ""
        if not reason_txt:
            reasons[idx] = "aucune justification enregistree"
        elif any(p in reason_txt for p in _NEGATION_PATTERNS):
            reasons[idx] = "la justification dit elle-meme qu'il n'y a pas d'avis"
        elif any(p in reason_txt for p in _BOILERPLATE_PATTERNS):
            reasons[idx] = "justification = texte de page ou publicite"
    types = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for sym, qt in zip(ledger["ticker"].unique(), ex.map(_quote_type, ledger["ticker"].unique())):
            types[sym] = qt
    for idx, r in ledger.iterrows():
        qt = types.get(r["ticker"])
        if idx not in reasons and qt is not None and qt != "EQUITY":
            reasons[idx] = f"instrument {qt}, pas une action"
    if not reasons:
        return ledger, cash
    annulled = ledger.loc[list(reasons)].copy()
    annulled["motif_annulation"] = [reasons[i] for i in annulled.index]
    annulled["date_annulation"] = today
    for idx in annulled.index:
        entry_value = float(ledger.at[idx, "entry_value_eur"])
        if ledger.at[idx, "status"] == "closed":
            cash += entry_value - float(ledger.at[idx, "exit_value_eur"])
        else:
            cash += entry_value
    old = pd.read_csv(ANNULLED_PATH) if ANNULLED_PATH.exists() else None
    annulled = pd.concat([old, annulled], ignore_index=True) if old is not None else annulled
    ANNULLED_PATH.parent.mkdir(parents=True, exist_ok=True)
    annulled.to_csv(ANNULLED_PATH, index=False)
    print(f"  nettoyage : {len(reasons)} position(s) du labo annulee(s) (detail dans {ANNULLED_PATH.name})")
    return ledger.drop(index=list(reasons)).reset_index(drop=True), cash


def write_summary(ledger: pd.DataFrame, cash: float) -> dict:
    closed = ledger[ledger["status"] == "closed"]
    open_pos = ledger[ledger["status"] == "open"]
    total_equity = cash + open_pos["current_value_eur"].sum()
    per_position = pd.concat([closed["return_pct"], open_pos["unrealized_return_pct"]]).dropna()
    summary = {
        "cash_eur": cash, "total_equity_eur": total_equity,
        # kept for the dashboard, but misleading on its own: the lab is uncapped, so its notional
        # can be many times the 300 EUR baseline -- read pnl_eur / rendement_moyen_par_position
        "total_return_pct": total_equity / STARTING_CAPITAL - 1,
        "pnl_eur": total_equity - STARTING_CAPITAL,
        "capital_engage_eur": float(open_pos["entry_value_eur"].sum()),
        "rendement_moyen_par_position": float(per_position.mean()) if len(per_position) else None,
        "nb_open": len(open_pos), "nb_closed": len(closed),
        "nb_long_open": int((open_pos["side"] == "long").sum()),
        "nb_short_open": int((open_pos["side"] == "short").sum()),
        "win_rate_closed": float((closed["return_pct"] > 0).mean()) if len(closed) else None,
        "avg_return_closed": float(closed["return_pct"].mean()) if len(closed) else None,
    }
    SUMMARY_PATH.write_text(pd.Series(summary).to_json(), encoding="utf-8")
    print(f"\n=== Bot #33 Courrier (labo) : {summary['nb_open']} positions ouvertes "
          f"({summary['nb_long_open']} long / {summary['nb_short_open']} short), "
          f"P&L {summary['pnl_eur']:+.2f} EUR pour {summary['capital_engage_eur']:.0f} EUR engages ===")
    return summary


def append_equity_curve_point(cash: float, total_equity: float, nb_open: int, nb_closed: int):
    row = {"timestamp": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
           "cash_eur": cash, "total_equity_eur": total_equity, "n_open": nb_open, "n_closed": nb_closed}
    header = not EQUITY_CURVE_PATH.exists()
    EQUITY_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(EQUITY_CURVE_PATH, mode="a", header=header, index=False)


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------

def _backfill_ids(token: str, state: dict, ledger: pd.DataFrame, journal: pd.DataFrame, live_ids: set) -> list[str]:
    """Next batch of older emails to backfill -- fixed window ending the day the backfill started
    (the live 2-day listing covers everything after), only from senders already seen as
    newsletters. Marks the backfill done once the window is exhausted."""
    bf = state.setdefault("backfill", {})
    if bf.get("done"):
        return []
    if "end_date" not in bf:
        bf["end_date"] = datetime.now(timezone.utc).date().isoformat()
        bf["processed_ids"] = []
    domains = set(journal["domain"].dropna()) | {s for s in ledger["source"].dropna() if " " not in s and "." in s}
    if not domains:
        bf["done"] = True
        return []
    end = datetime.fromisoformat(bf["end_date"]).date()
    start = end - timedelta(days=BACKFILL_DAYS)
    query = (f"after:{start:%Y/%m/%d} before:{end:%Y/%m/%d} from:(" +
             " OR ".join(sorted(domains)) + ")")
    done_ids = set(bf["processed_ids"]) | live_ids
    ids = [i for i in list_message_ids(token, query, max_messages=2000) if i not in done_ids]
    if not ids:
        bf["done"] = True
        print("Rattrapage termine.")
        return []
    batch = ids[-BACKFILL_MAX_PER_RUN:]  # Gmail lists newest first -> walk from the oldest
    print(f"Rattrapage : {len(ids)} mail(s) restant(s) sur {BACKFILL_DAYS} jours, {len(batch)} traite(s) ce run.")
    return batch


def main():
    deadline = time.monotonic() + RUN_BUDGET_MIN * 60
    state = _load_json(STATE_PATH, {})
    now = pd.Timestamp.now(tz="UTC")
    today = now.date().isoformat()

    ledger = load_ledger()
    cash = load_cash()
    journal = scoring.load_journal()
    if not state.get("legacy_cleanup_2026_10_03"):
        ledger, cash = cleanup_legacy_rows(ledger, cash, today)
        state["legacy_cleanup_2026_10_03"] = today

    signals, rejects = [], []
    message_ids = None  # set only once the live batch was fully processed
    missing = [v for v in ("GMAIL_REFRESH_TOKEN", "GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET") if not os.environ.get(v)]
    if missing:
        print(f"Variables manquantes ({', '.join(missing)}) -- lecture des mails ignoree.", file=sys.stderr)
    else:
        try:
            token = get_access_token()
            listed = list_message_ids(token, GMAIL_QUERY)
            previously = set(state.get("processed_message_ids", [])) | set(state.get("backfill", {}).get("processed_ids", []))
            new_ids = [m for m in listed if m not in previously]
            attempts = state.setdefault("ollama_attempts", {})
            s, r, done = process_messages(token, new_ids, deadline, attempts=attempts)
            signals += s
            rejects += r
            # only ids actually handled -- the others stay "new" and are retried next run
            message_ids = [m for m in listed if m in previously or m in done]
            if time.monotonic() < deadline:
                batch = _backfill_ids(token, state, ledger, journal, set(listed))
                if batch:
                    s, r, done_bf = process_messages(token, batch, deadline, backfill=True, attempts=attempts)
                    signals += s
                    rejects += r
                    state["backfill"]["processed_ids"] = state["backfill"]["processed_ids"] + sorted(done_bf)
            # ids given up on are now in a processed list -- only still-pending counters are worth keeping
            state["ollama_attempts"] = {k: v for k, v in attempts.items() if v < MAX_ATTEMPTS_PER_MAIL}
        except Exception as e:
            print(f"echec acces Gmail: {e} -- lecture des mails ignoree ce run.", file=sys.stderr)

    journal, added = scoring.add_signals(journal, signals, now.isoformat())
    if added:
        print(f"{len(added)} avis ajoute(s) au journal : "
              + "; ".join(f"{s['ticker']}({s['side']},{s['publication']}{',rattrapage' if s['backfill'] else ''})"
                          for s in added))
    append_rejects(rejects)
    journal = scoring.evaluate_journal(journal, today)
    scores, crowd = scoring.score_sources(journal, state.get("source_status", {}))
    state["source_status"] = scoring.save_status(scores, crowd)
    fiables = [p for p, st in state["source_status"].items() if st == "fiable"]
    print(f"Newsletters fiables : {', '.join(fiables) if fiables else 'aucune pour l instant'} ; foule : "
          f"edge {crowd['edge']:+.2%} sur {crowd['n']} evenement(s) ({crowd['status']}).")

    currencies = {c for c in ledger["currency"].dropna() if isinstance(c, str)} | {"USD"}
    fx_rates = fetch_fx_rates(currencies)
    ledger, cash = recheck_and_exit(ledger, today, cash, fx_rates)
    live = [s for s in added if not s["backfill"]]
    if live:
        ledger, cash = apply_signals(ledger, live, cash, today, fx_rates)

    write_summary(ledger, cash)
    scoring.build_scorecard(scores, crowd, ledger).to_csv(SCORECARD_PATH, index=False)
    open_pos = ledger[ledger["status"] == "open"]
    append_equity_curve_point(cash, cash + open_pos["current_value_eur"].sum(), len(open_pos),
                               int((ledger["status"] == "closed").sum()))
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    save_cash(cash)
    scoring.save_journal(journal)

    run_real_layer(journal, scores, crowd, now, fx_rates, _fetch_price, _ensure_fx)

    if message_ids is not None:
        state["processed_message_ids"] = message_ids
    state["last_run_date"] = today
    _save_json(STATE_PATH, state)


if __name__ == "__main__":
    main()
