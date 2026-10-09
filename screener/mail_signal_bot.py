"""Bot #33 ("Courrier"): individual stock tips found directly in the user's own Gmail
newsletters, scored per newsletter, and a capital-limited strategy that only follows the
newsletters that have proven themselves -- the per-ticker counterpart to
screener/newsletter_digest.py's per-sector qualitative signal (added 2026-09-11, per the user's
explicit request: "des mails il y a souvent des suggestions d'actions a fort potentiel ou au
contraire des chutes [...] et savoir de quelle newsletter ca vient pour identifier les
bons/mauvais investisseurs").

TWO LAYERS (since 2026-10-04)
-----------------------------
  1. JOURNAL (mail_signal_scoring.py): every validated tip from every newsletter, evaluated at
     J+5/J+20/J+60 against its market's benchmark -> the per-site performance the dashboard shows,
     each newsletter's status ("fiable" / "bruit" / "observation") and the crowd's consensus.
  2. STRATEGIE REELLE (mail_signal_real.py): capital-limited, long-only, fee-aware book that only
     follows reliable newsletters, with the crowd's consensus adjusting conviction.
The original uncapped long/short paper book ("labo", mail_signal_ledger.csv + its 300 EUR
notional pool) was retired on 2026-10-04 at the user's request ("supprime le labo, je n'ai jamais
compris ce que c'etait"): the journal measures every tip without needing a position per tip, and
the real strategy is the only book with an actual capital. retire_lab() deletes its files once
(they stay in git history) after saving the newsletter domains it knew, which the backfill needs.

2026-10-03 OVERHAUL (the user's go-live review: "fais toutes ces corrections")
-----------------------------------------------------------------------------
A hand review of the first 168 tips found ~43% of them wrong: ticker mapping errors ("Chevron" ->
CHEV = Charging Robotics; "BCE" = the European Central Bank -> Bell Canada; "The Dollar Went Up" ->
USD, a 2x semiconductor ETF), inverted direction ("Paychex Plunges, Providing the Entry Investors
Have Been Waiting For" -> short), plain news or page boilerplate taken as tips, and 22 ETFs/funds.
Root cause: the prompt asked the model for its "best guess" ticker (a guessed fact -- against this
repo's grounding rule) and the resolver only checked that the symbol had a price, not that it was
the right company. Now every tip must pass, in order (see _validate_tip / _verify_tip /
_resolve_instrument):
  - deterministic text checks: a verbatim citation that really is in the email, the company named
    right next to it, not a macro subject (central bank, currency, index...), no "pas d'avis"
    style self-negation, no ad/boilerplate text, no explicit upgrade/downgrade wording
    contradicting the claimed direction;
  - a second, independent Ollama call that only sees the citation and must confirm an explicit
    investment opinion on that company, in the same direction (it is not told which direction
    was claimed);
  - a Yahoo lookup: the ticker is only taken from the email when it is literally written there,
    otherwise looked up by company name; the listing must be an EQUITY on a primary exchange
    (no ETF, fund, OTC) whose Yahoo name matches the company named in the email.
Rejected tips are logged with their reason (mail_signal_rejects.csv) so the filter itself can be
audited. One CI job only (the newsletter-digest-bot repo used to run this same script on the same
files in parallel, losing whichever push came second).

CI TIME BUDGET (2026-10-04): the CPU-only runner needs ~1.7 min per email (classify + extract +
verify). Ollama calls are sequential, phase by phase, with the fixed instructions first in every
prompt (Ollama reuses the cached prefix), and no call is started after RUN_BUDGET_MIN -- an email not
fully handled by then is retried next run instead of being marked processed.

Backfill: the runs after the overhaul also walk back over the last BACKFILL_DAYS of mail from
senders known as newsletters, feeding the journal only -- so newsletters get a J+20 track record in
weeks instead of months.

ARTICLE FETCH (see _fetch_article_extract(), added 2026-09-16): most newsletters only excerpt a
couple of sentences before a "read more" link to the sender's own site. For the domains
hand-confirmed fetchable with a plain HTTP GET (zonebourse.com, tradingsat.com -- see
FETCHABLE_DOMAINS), the full article text replaces the teaser. Seeking Alpha is deliberately NOT in
that list: it answers a plain GET with a PerimeterX CAPTCHA wall -- bot-detection, not something
this bot tries to bypass.

ATTRIBUTION (see _publication()): PRIVACY / REPO-PUBLIC CONSTRAINT -- this repo pushes to a public
GitHub remote. The user's explicit choice (2026-09-11): never persist the sender's email address.
The sending domain is kept (e.g. "seekingalpha.com"). For newsletter PLATFORMS (beehiiv, substack,
sailthru...) where the domain is shared by dozens of unrelated newsletters, the sender's display
name is used instead (e.g. "Some Newsletter (beehiiv.com)"). Still never the address itself. An
author name is kept only when the email literally contains it (journal column, informational).
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

from screener.simulate_constrained_portfolio import fetch_fx_rates  # noqa: E402
from screener import mail_signal_scoring as scoring  # noqa: E402
from screener.mail_signal_real import run_real_layer  # noqa: E402

STATE_PATH = HERE / "results/screener/mail_signal_state.json"
SCORECARD_PATH = HERE / "results/screener/mail_signal_source_scorecard.csv"
REJECTS_PATH = HERE / "results/screener/mail_signal_rejects.csv"
# Files of the retired "labo" -- deleted once by retire_lab() (see module docstring)
LEGACY_LAB_LEDGER = HERE / "results/simulation/mail_signal_ledger.csv"
LEGACY_LAB_FILES = (
    LEGACY_LAB_LEDGER,
    HERE / "results/simulation/mail_signal_state.json",
    HERE / "results/simulation/mail_signal_summary.json",
    HERE / "results/simulation/mail_signal_equity_curve.csv",
    HERE / "results/simulation/mail_signal_annulled.csv",
)

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
CHUNK_SIZE = 4  # emails taken through every phase together -- see process_messages()

BACKFILL_DAYS = 30
BACKFILL_MAX_PER_RUN = 25  # upper bound only -- RUN_BUDGET_MIN is what actually stops a run (first CI
# run with the budget, 2026-10-04: 18 emails in 30 min; the 30-day window held 717 newsletter emails)

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
# URL path words of pages that are never the teased article (2026-10-08): tradingsat.com's "valeur du
# jour" email links its "abonnement Prestige" page FIRST, so for weeks the bot read that subscription
# page -- a site menu, no tip -- instead of the stock analysis three links further down, and
# llama3.1:8b then made up "citations" shaped like the prompt's own examples (all caught by the
# verbatim check, but the real tip was lost). The homepage and images are skipped too.
_NON_ARTICLE_PATH_WORDS = ("abonnement", "abonnes", "membres", "newsletter", "inscription", "desinscription",
                           "services", "contact", "outils", "emailing", "connexion", "compte", "login",
                           "images", "img", "img2")
_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp")

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
OLLAMA_OPTIONS = {"temperature": 0, "num_ctx": 2048, "num_predict": 700}  # 400 cut a 3-tip answer mid-JSON (2026-10-04)

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
# A rating action written out in the citation itself -- the tip is then explicit by construction,
# and the Ollama second opinion is skipped (see _explicit_rating). 2026-10-08: llama3.1:8b
# rejected "Market Pricing Closed Russian Flows (Downgrade To Hold)", "Bloom Energy Just Won
# Another Catalyst (Rating Upgrade)" or "Palantir: Rating Downgrade" as "pas d'avis explicite" --
# 386 of 534 rejects were that verdict. Whole-word matches only ("underperformed" is past
# performance, not a rating). A downgrade to Hold counts as bearish: it is a negative revision.
_EXPLICIT_BULLISH = ("rating upgrade", "upgrade to buy", "upgrade to strong buy", "upgrade to outperform",
                     "upgrade to overweight", "upgraded to buy", "upgraded to outperform", "upgraded to overweight",
                     "upgrade to hold", "upgraded to hold", "raised to buy", "initiated at buy", "initiate at buy",
                     "strong buy", "buy rating", "rated buy", "rated a buy", "is a buy", "outperform rating",
                     "overweight rating", "relevée à l'achat", "relevé à l'achat", "relève à l'achat",
                     "passe à l'achat", "se positionner à l'achat", "recommandation à l'achat", "conseil achat",
                     "conseil à l'achat", "recommande l'achat", "objectif de cours relevé", "relève son objectif")
_EXPLICIT_BEARISH = ("rating downgrade", "downgrade to hold", "downgrade to sell", "downgrade to underperform",
                     "downgrade to underweight", "downgraded to hold", "downgraded to sell",
                     "downgraded to underperform", "downgraded to underweight", "cut to sell", "cut to hold",
                     "strong sell", "sell rating", "rated sell", "rated a sell", "is a sell",
                     "underperform rating", "underweight rating", "abaissée à la vente", "abaissé à la vente",
                     "passe à la vente", "se positionner à la vente", "recommandation à la vente", "conseil vente",
                     "conseil à la vente", "recommande la vente", "objectif de cours abaissé", "abaisse son objectif")
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

SENS DE L'AVIS : deduis-le de l'opinion exprimee (acheter, conserver, vendre, sous-evaluee, trop chere), jamais du mouvement passe du cours : une baisse passee peut accompagner un avis haussier, une hausse passee un avis baissier. Si l'extrait compare deux titres, seul celui qui est explicitement recommande ou deconseille compte -- l'autre n'est pas un avis.

Pour chaque action retenue (maximum {max_tickers}), donne :
- "company" : le nom de la societe EXACTEMENT tel qu'ecrit dans l'extrait
- "ticker" : le symbole boursier SEULEMENT s'il est ecrit tel quel dans l'extrait (ex: "(NVDA)"), sinon "" -- ne devine jamais un symbole
- "sentiment" : "haussier" ou "baissier"
- "citation" : la phrase de l'EXTRAIT qui exprime l'avis, COPIEE MOT POUR MOT (ne la reformule pas, ne la traduis pas, ne l'invente pas, ne la prends jamais dans ces instructions)
- "author" : le nom de l'auteur de l'analyse SEULEMENT s'il est ecrit dans l'extrait, sinon ""

Si aucune action ne remplit ces conditions -- ou si l'extrait n'est qu'un menu de site, une page d'abonnement ou un commentaire de marche sans avis sur une action precise --, reponds avec une liste vide.

Reponds UNIQUEMENT en JSON : {{"tips": [{{"company": "<NOM>", "ticker": "<SYMBOLE ou vide>", "sentiment": "haussier|baissier", "citation": "<phrase copiee>", "author": "<auteur ou vide>"}}, ...]}}

Sujet : {subject}
Extrait : {body}
"""

# Independent second opinion on ONE tip (2026-10-03): sees only the subject and the citation, and
# is not told which direction the extraction claimed -- so it can't just agree.
VERIFY_PROMPT = """Question : la citation de newsletter financiere donnee a la fin contient-elle un AVIS D'INVESTISSEMENT EXPLICITE sur l'action de la societe nommee a la fin, elle-meme ? Un avis d'investissement = recommandation d'achat ou de vente, notation relevee ou abaissee, objectif de cours, ou conclusion argumentee sur le potentiel de hausse ou le risque de baisse de l'action.
Une citation courte peut etre un avis : un titre d'article qui annonce la notation de l'auteur ou un changement de notation est un avis explicite, meme entre parentheses -- "(Rating Upgrade)", "Downgrade To Hold", "Still A Sell", "relevee a l'achat", "se positionner a l'achat". Une simple notation "Hold"/"conserver" sans changement n'est ni haussiere ni baissiere : ce n'est pas un avis.
Ce n'est PAS un avis : une simple actualite (resultats, partenariat, contrat, proces, nomination), un mouvement de cours passe sans opinion, un avis sur une autre societe, une publicite.

Si c'est un avis, quel est son sens ? "haussier" (acheter, notation relevee, potentiel de hausse -- y compris une baisse passee presentee comme une opportunite d'achat) ou "baissier" (vendre, notation abaissee -- y compris vers "conserver" --, risque de baisse).

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


_HTML_TAG_RE = re.compile(r"</?(?:div|table|tbody|thead|tr|td|th|p|span|br|a|img|b|i|u|strong|em|font|center"
                          r"|ul|ol|li|h[1-6])\b[^>]*>", re.IGNORECASE)


def _clean_plain_text(plain: str) -> str:
    """text/plain parts are not always plain (2026-10-08): zonebourse.com's carries raw HTML markup
    (sponsor <div>/<table> blocks filled most of the 1800-character extract), news.meilleurtaux.com's
    HTML entities ("l&rsquo;IA" -- a citation the model wrote with a real apostrophe then failed the
    verbatim check). Only real tag names are stripped, so "<<Lire la suite>>" survives."""
    plain = html.unescape(_HTML_TAG_RE.sub(" ", plain))
    plain = re.sub(r"[ \t]+", " ", plain)
    return re.sub(r"\n\s*\n+", "\n", plain).strip()


def _extract_text(payload: dict) -> str:
    """text/plain part if there is one (see _clean_plain_text), else the HTML part converted to
    text (script/style blocks removed and entities unescaped since 2026-10-03 -- an HTML-only
    newsletter used to hand the model its CSS as the first 900 characters)."""
    stack = [payload]
    html_fallback = None
    while stack:
        part = stack.pop()
        mime = part.get("mimeType", "")
        body_data = part.get("body", {}).get("data")
        if mime == "text/plain" and body_data:
            return _clean_plain_text(_decode_part(body_data, _part_charset(part)))
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


def _rank_article_links(links: list[str], subject: str) -> list[str]:
    """Links that can be an article (see _NON_ARTICLE_PATH_WORDS), those whose URL path shares the
    most words with the email subject first ("La valeur du jour : MERSEN" -> /mersen-.../conseils/...),
    email order otherwise."""
    subject_words = {w for w in _norm_text(subject).split() if len(w) >= 4}
    ranked = []
    for i, link in enumerate(links):
        path = urllib.parse.urlsplit(link).path.lower()
        segments = [s for s in path.split("/") if s]
        if not segments or path.endswith(_IMAGE_EXTENSIONS):
            continue
        # first word of a segment only: "/outils-de-trading/" is a tools page, "/aws-services-..." an article
        if any((_norm_text(s).split() or [""])[0] in _NON_ARTICLE_PATH_WORDS for s in segments):
            continue
        ranked.append((-len(subject_words & set(_norm_text(path).split())), i, link))
    return [link for _, _, link in sorted(ranked)]


def _fetch_article_extract(links: list[str], subject: str = "") -> str | None:
    """Full article text from the best-ranked link (see _rank_article_links) that resolves to a
    FETCHABLE_DOMAINS HTML page, or None."""
    for link in _rank_article_links(links, subject)[:ARTICLE_LINK_CANDIDATES]:
        try:
            resp = requests.get(link, headers=_FETCH_HEADERS, timeout=ARTICLE_FETCH_TIMEOUT, allow_redirects=True)
        except Exception:
            continue
        if resp.status_code != 200 or "html" not in resp.headers.get("content-type", "").lower():
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
_EXPLICIT_BULLISH = tuple(_norm_text(p) for p in _EXPLICIT_BULLISH)
_EXPLICIT_BEARISH = tuple(_norm_text(p) for p in _EXPLICIT_BEARISH)
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


def _explicit_rating(citation: str) -> str | None:
    """"haussier"/"baissier" when the citation spells out a rating action in one direction only
    (see _EXPLICIT_BULLISH), else None."""
    padded = f" {_norm_text(citation)} "
    bull = any(f" {w} " in padded for w in _EXPLICIT_BULLISH)
    bear = any(f" {w} " in padded for w in _EXPLICIT_BEARISH)
    if bull != bear:
        return "haussier" if bull else "baissier"
    return None


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
                     attempts: dict | None = None, known_domains: set | None = None) -> tuple:
    """Full pipeline for a batch of Gmail ids -> (validated signals, rejected tips, ids DONE,
    domains of the emails classified as financial newsletters).

    CHUNKS of CHUNK_SIZE emails, each taken through every phase (classify, extract, verify,
    resolve) before the next chunk starts -- 2026-10-04: classifying a whole 25-email backfill batch
    first used up 17 of the 40 budget minutes, then extraction ran out of time and NOTHING of that
    batch was saved. Inside a chunk, calls of the same type still run back to back (Ollama reuses the
    cached instructions). One Ollama call at a time, none STARTED after `deadline`
    (time.monotonic()). An id is DONE only once fully handled; the rest is retried next run, up to
    MAX_ATTEMPTS_PER_MAIL times when Ollama itself fails (`attempts`, persisted by the caller).

    Emails from `known_domains` (senders already seen as financial newsletters) skip the
    classification call: it costs ~40 s per email on the CI runner and was "yes" for 24 of 25
    such emails -- an occasional non-newsletter from those senders just yields no tip."""
    attempts = attempts if attempts is not None else {}
    known_domains = known_domains or set()
    msgs = [m for m in (fetch_message(token, mid) for mid in ids) if m is not None]
    msgs.sort(key=lambda m: m["date_utc"])  # oldest first: they are closest to leaving the listing window
    label = "rattrapage" if backfill else "nouveau(x)"
    done, failed, newsletter_domains = set(), set(), set()
    signals, rejects, seen = [], [], set()
    n_classify_ok = n_classify_err = 0

    def out_of_time() -> bool:
        return time.monotonic() >= deadline

    for start in range(0, len(msgs), CHUNK_SIZE):
        if out_of_time():
            break
        chunk = msgs[start:start + CHUNK_SIZE]

        classified = []
        for m in chunk:
            if m["source"] in known_domains:
                classified.append(m)
                continue
            if out_of_time():
                break
            verdict = classify_newsletter(m)
            if verdict is None:
                n_classify_err += 1
                failed.add(m["id"])
            else:
                n_classify_ok += 1
                if verdict:
                    classified.append(m)
                else:
                    done.add(m["id"])
        if n_classify_err and not n_classify_ok and not classified:
            raise RuntimeError("Ollama n'a repondu a aucune classification -- mails laisses non traites")
        newsletter_domains |= {m["source"] for m in classified}

        with ThreadPoolExecutor(max_workers=ARTICLE_FETCH_MAX_WORKERS) as ex:  # plain HTTP, no Ollama
            fetched = dict(zip((m["id"] for m in classified),
                                ex.map(lambda m: _fetch_article_extract(_extract_article_links(m.get("html", "")),
                                                                        m["subject"]),
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

        for m, tips in extracted:
            complete, verified, mail_rejects = True, [], []
            for tip in tips:
                # citation checked against what the model saw AND the email itself (an article tip
                # may quote the email's teaser)
                clean, reason = _validate_tip(tip, m["extract_text"] + "\n" + m["text"])
                if clean is None:
                    mail_rejects.append(_reject_row(m, tip, reason))
                    continue
                if _explicit_rating(clean["citation"]) == clean["sentiment"]:
                    verified.append(clean)  # rating action written out -- no second opinion needed
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
                    mail_rejects.append(_reject_row(m, clean, reason))
                else:
                    verified.append(clean)
            if not complete:
                continue  # retried next run -- its rejects so far are dropped too, to avoid duplicates
            for tip in verified:
                resolved, reason = _resolve_instrument(tip["company"], tip["ticker_in_text"])
                if resolved is None:
                    mail_rejects.append(_reject_row(m, tip, reason))
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
            rejects += mail_rejects
            done.add(m["id"])

    for mid in failed - done:
        attempts[mid] = attempts.get(mid, 0) + 1
        if attempts[mid] >= MAX_ATTEMPTS_PER_MAIL:
            print(f"  mail abandonne apres {attempts[mid]} echecs Ollama", file=sys.stderr)
            done.add(mid)
    print(f"{len(msgs)} mail(s) {label} a examiner, {len(done)} traite(s) ce run.")
    pending = len(msgs) - len(done)
    if pending:
        print(f"  {pending} mail(s) {label} reporte(s) au prochain run (budget de temps ou echec Ollama).")
    for r in rejects:
        print(f"  rejete [{r['publication']}] {r['company']} ({r['sentiment']}) : {r['motif']}")
    return signals, rejects, done, newsletter_domains


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
# Helpers for the real strategy + retirement of the old lab
# ---------------------------------------------------------------------------------------------

def _ensure_fx(currency, fx_rates: dict) -> dict:
    """Extends fx_rates in place with whichever currency a position needs (tickers come from
    arbitrary mail tips, not a pre-scoped universe)."""
    key = "GBP" if currency == "GBp" else (currency if isinstance(currency, str) and currency else "EUR")
    if key not in fx_rates:
        fx_rates.update(fetch_fx_rates({key}))
    return fx_rates


def retire_lab(state: dict, journal: pd.DataFrame):
    """One-off (2026-10-04, see module docstring): keep the newsletter domains the old lab ledger
    knew -- the backfill searches Gmail by sender domain and the journal alone doesn't know them
    all yet -- then delete the lab's files (they remain in git history)."""
    domains = set(state.get("known_domains", [])) | set(journal["domain"].dropna())
    if LEGACY_LAB_LEDGER.exists():
        try:
            lab = pd.read_csv(LEGACY_LAB_LEDGER)
            domains |= {s for s in lab["source"].dropna() if " " not in s and "." in s}
        except Exception as e:
            print(f"  lecture de l'ancien labo impossible: {e}", file=sys.stderr)
    state["known_domains"] = sorted(domains)
    removed = [f.name for f in LEGACY_LAB_FILES if f.exists()]
    for f in LEGACY_LAB_FILES:
        f.unlink(missing_ok=True)
    if removed:
        print(f"  labo retire : {', '.join(removed)} supprime(s) (historique conserve dans git).")


# ---------------------------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------------------------

def _backfill_ids(token: str, state: dict, journal: pd.DataFrame, live_ids: set) -> list[str]:
    """Next batch of older emails to backfill -- fixed window ending the day the backfill started
    (the live 2-day listing covers everything after), only from senders already seen as
    newsletters. Marks the backfill done once the window is exhausted."""
    bf = state.setdefault("backfill", {})
    if bf.get("done"):
        return []
    if "end_date" not in bf:
        bf["end_date"] = datetime.now(timezone.utc).date().isoformat()
        bf["processed_ids"] = []
    domains = set(journal["domain"].dropna()) | set(state.get("known_domains", []))
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

    journal = scoring.load_journal()
    if not state.get("lab_retired"):
        retire_lab(state, journal)
        state["lab_retired"] = today
    state.pop("legacy_cleanup_2026_10_03", None)  # flag of the lab's one-off cleanup, now moot

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
            known = set(state.get("known_domains", [])) | set(journal["domain"].dropna())
            s, r, done, domains = process_messages(token, new_ids, deadline, attempts=attempts, known_domains=known)
            signals += s
            rejects += r
            state["known_domains"] = sorted(set(state.get("known_domains", [])) | domains)
            # only ids actually handled -- the others stay "new" and are retried next run
            message_ids = [m for m in listed if m in previously or m in done]
            if time.monotonic() < deadline:
                batch = _backfill_ids(token, state, journal, set(listed))
                if batch:
                    s, r, done_bf, _ = process_messages(token, batch, deadline, backfill=True, attempts=attempts,
                                                        known_domains=known | domains)
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
    scoring.build_scorecard(scores, crowd).to_csv(SCORECARD_PATH, index=False)
    scoring.save_journal(journal)

    fx_rates = fetch_fx_rates({"USD"})
    run_real_layer(journal, scores, crowd, now, fx_rates, _fetch_price, _ensure_fx)

    if message_ids is not None:
        state["processed_message_ids"] = message_ids
    state["last_run_date"] = today
    _save_json(STATE_PATH, state)


if __name__ == "__main__":
    main()
