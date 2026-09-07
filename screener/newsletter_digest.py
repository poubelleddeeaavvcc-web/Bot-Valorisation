"""Daily sector-outlook signal derived from the user's own Gmail newsletters, feeding the
"veto sectoriel" bots (#10/#11/#12): a qualitative, forward-looking read ("this sector is under
pressure / thriving") to sit alongside sector_momentum.csv's purely backward-looking 12-2 month
ETF price momentum.

Gmail access reuses the read-only OAuth grant already set up in the (unrelated) FreelanceCopilot
project -- same Google account, same `gmail.readonly` scope, refreshed here non-interactively via
plain HTTP (no google-api-python-client: this repo's dependency style stays on `requests`
throughout, and pulling in the full SDK for two REST calls isn't worth it). Locally, put
GMAIL_REFRESH_TOKEN/GMAIL_CLIENT_ID/GMAIL_CLIENT_SECRET in a gitignored `.env` (see FreelanceCopilot
for the values, or extract them from its token.json/credentials.json); in CI they come from repo
secrets. Missing/invalid credentials fail this run closed (skip the digest entirely, logged, no
crash) -- same posture as a missing dependency elsewhere in the pipeline, not a hard requirement
for the rest of the screener to run.

No sender/label pre-filter, per the user's explicit direction (2026-09-04): every message from the
last 2 days is read and Ollama itself judges whether it's a financial/economic newsletter or
something else (personal, professional, transactional, unrelated advertising) -- see
classify_newsletter(). Only messages classified as newsletters ever get their content folded into
the sector-classification prompt (see _classify_extract_sector()).

GROUNDING RULE (same as screener/news_filter.py -- see that module's docstring and the project's
own standing rule): Ollama must never invent a sector's outlook. Each extract is classified against
only the text actually fetched this run (see _classify_extract_sector()). A single extract never
drives a sector's outlook on its own either: readings feed a sliding per-sector window (see
WINDOW_SIZE / _update_window()) and a sector only gets a directional outlook once WINDOW_SIZE
independent readings have accumulated for it -- short of that (or if nobody's newsletter mentioned
it at all) it stays "no_data". The window has no calendar expiry: a sector that stops appearing in
newsletters simply keeps whatever readings it last had, however old, until fresh ones eventually
earn their way back in (2026-09-07, per the user's explicit direction -- replaces an earlier
calendar-based staleness check that erased a reading after N silent days regardless of whether
anything had actually contradicted it).

PRIVACY / REPO-PUBLIC CONSTRAINT: this repo pushes to a public GitHub remote with a Pages-served
dashboard. Raw email BODY TEXT and sender addresses are therefore NEVER written to disk beyond
the run's own memory and the minimal state file (a timestamp/id list, no content) -- only Ollama's
own synthesized outlook + 0-10 score + one-line reason per sector reaches data/universe/sector_outlook.csv.
The one exception (added 2026-09-04, per the user's request for traceability): the SUBJECT line of
each email actually classified as a financial newsletter is appended to
results/screener/newsletter_database.csv -- a subject line is already headline-length public-ish
text, the same sensitivity level as the headline titles news_filter.py already persists to
news_database.csv, so it's an acceptable exception to the no-content rule; the body and sender
stay excluded.

Auto-gated to once per UTC day (same shape as portfolio/bot_checkin.due_for_checkin): the workflow
that calls this runs hourly, but reclassifying the same ~2-day mail window every hour would burn
Ollama calls for a signal that only meaningfully changes once a day.
"""
import base64
import json
import os
import pathlib
import re
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.build_trending_universe import SECTOR_ETF  # noqa: E402

STATE_PATH = HERE / "results/screener/newsletter_digest_state.json"
WINDOW_PATH = HERE / "results/screener/sector_outlook_window.json"
OUT_PATH = HERE / "data/universe/sector_outlook.csv"
NEWSLETTER_DB_PATH = HERE / "results/screener/newsletter_database.csv"
NEWSLETTER_DB_COLUMNS = ["date", "subject"]

TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
GMAIL_QUERY = "newer_than:2d"  # 2 days, not 1: buffer against a run being skipped/late without
# losing a day's newsletters entirely
MAX_MESSAGES = 300
BODY_TRUNCATE = 1500  # per-message character cap fed to the classifier prompt
SYNTHESIS_EXTRACT_TRUNCATE = 500  # per-message cap when building the sector-synthesis prompt

WINDOW_SIZE = 3  # readings needed in a sector's sliding window before it gets a directional
# outlook (see GROUNDING RULE above) -- below this, a single email (or even two) could still
# swing an entire sector's veto signal on its own say-so (observed 2026-09-06: Industrials'
# "florissant" traced back to one extract about CRH, a single construction-materials company).
# The oldest reading is evicted only once a WINDOW_SIZE+1'th one arrives for that sector (see
# _update_window()) -- there is deliberately no separate calendar-based expiry.

OLLAMA_URL = "http://localhost:11434/api/generate"
# 8b, not the 3b shared with news_filter.py's bots #4/5/6: this module asks the model to judge
# which single sector (out of 11 candidates) an extract is actually about -- a subtler relevance
# call than news_filter.py's per-ticker buy/veto question, and 3b was demonstrably getting it
# wrong (2026-09-04: confidently linked an unrelated Nvidia/pharma extract to Finance, Industrials,
# Telecommunications...). Slower per call on the CPU-only Actions runner, but this step already
# waits on Ollama serially either way.
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_TIMEOUT = 180  # 8b cold-loads and infers slower than the 3b other modules use

# outlook is derived from SCORE, never asked for directly -- asking a 3B model for a category
# ("sous_pression"/"neutre"/"florissant") AND a free-text reason in the same call let the two
# drift apart (observed 2026-09-04: reason argued "sous pression" while outlook said "florissant").
# A single 0-10 axis removes that failure mode by construction: there's only one number to get
# consistent with itself, and outlook becomes pure arithmetic on it (see _score_to_outlook).
SCORE_SOUS_PRESSION_MAX = 3  # score <= this -> sous_pression
SCORE_FLORISSANT_MIN = 7     # score >= this -> florissant, else neutre

CLASSIFY_PROMPT = """Voici un email recu aujourd'hui :

Expediteur : {sender}
Sujet : {subject}
Extrait : {body}

Ceci est-il une newsletter financiere/economique (actualite des marches, d'un secteur, ou macroeconomique) -- par opposition a un email personnel, professionnel, transactionnel, ou publicitaire non lie a la finance ?

Reponds UNIQUEMENT en JSON : {{"is_finance_newsletter": true|false, "reason": "<une phrase courte>"}}
"""

# One call per EXTRACT (asking it to pick its one main sector out of 11), not one call per
# SECTOR (asking of every sector in turn "does this extract concern you?") -- the per-sector
# version, even on the 8b model, kept forcing a link between an extract and every sector loosely
# adjacent to its topic (observed 2026-09-05: an AI-chips earnings extract got cited as the source
# for Finance, Industrials, Basic Materials, Energy AND Telecommunications, because each of those
# 11 questions individually gave the model room to rationalize a connection). Asking once per
# extract for its SINGLE main sector removes that structural incentive: there's no separate
# question to talk itself into answering yes to for each unrelated sector.
SECTOR_LIST = "\n".join(f"- {s}" for s in SECTOR_ETF)

EXTRACT_SECTOR_PROMPT = """Voici un extrait de newsletter financiere recue aujourd'hui :

Sujet : {subject}
Extrait : {body}

Parmi les secteurs suivants, lequel est le sujet PRINCIPAL et EXPLICITE de cet extrait -- pas un secteur seulement mentionne en passant ou relie de facon indirecte ou supposee (ex: un extrait sur les semi-conducteurs IA concerne "Technology", pas "Finance", "Industrials" ou "Telecommunications" juste parce que ces secteurs achetent, vendent ou utilisent aussi de la technologie) :

{sector_list}

Il est normal et attendu qu'un extrait ne corresponde a AUCUN de ces secteurs -- dans le doute, reponds "sector": null plutot que de forcer un rapprochement.

Si un secteur est identifie, note de 0 a 10 la tendance qu'exprime cet extrait pour ce secteur : 0 = clairement sous pression, 5 = neutre ou avis partages, 10 = clairement florissant.

Reponds UNIQUEMENT en JSON : {{"sector": "<un secteur EXACT de la liste ci-dessus, ou null>", "score": <entier 0-10, ou null si sector est null>, "reason": "<une phrase courte citant ce que dit cet extrait>"}}
"""


def _call_ollama_json(prompt: str) -> dict:
    """Same request/parse shape as screener/news_filter.py's _call_ollama_json -- duplicated
    rather than imported (that module is a private, per-bot helper, not a shared library; see
    simulate_constrained_portfolio.py's docstring for why this repo duplicates small helpers
    instead of introducing cross-module coupling for a few lines)."""
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=OLLAMA_TIMEOUT)
    resp.raise_for_status()
    outer = json.loads(resp.content)
    return json.loads(outer["response"])


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_access_token() -> str:
    resp = requests.post(TOKEN_URL, data={
        "refresh_token": os.environ["GMAIL_REFRESH_TOKEN"],
        "client_id": os.environ["GMAIL_CLIENT_ID"],
        "client_secret": os.environ["GMAIL_CLIENT_SECRET"],
        "grant_type": "refresh_token",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["access_token"]


def list_recent_message_ids(token: str) -> list[str]:
    ids = []
    params = {"q": GMAIL_QUERY, "maxResults": min(MAX_MESSAGES, 500)}
    headers = {"Authorization": f"Bearer {token}"}
    while True:
        resp = requests.get(f"{GMAIL_API}/messages", params=params, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        ids.extend(m["id"] for m in data.get("messages", []))
        if len(ids) >= MAX_MESSAGES or "nextPageToken" not in data:
            break
        params["pageToken"] = data["nextPageToken"]
    return ids[:MAX_MESSAGES]


def _part_charset(part: dict) -> str:
    """Reads the charset declared in this MIME part's own Content-Type header (Gmail API exposes
    each part's raw headers under "headers") -- falls back to utf-8 when absent/unrecognized
    rather than assuming every email is UTF-8. Non-UTF-8 newsletters (ISO-8859-1/Windows-1252 is
    common for French senders) were otherwise coming out with accented characters replaced by
    literal U+FFFD, corrupting text that later reaches the public sector_outlook.csv (observed
    2026-09-06/07, e.g. "r�f�rence" instead of "référence")."""
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
    except LookupError:  # charset name Python's codecs module doesn't recognize
        return raw.decode("utf-8", errors="replace")


def _extract_text(payload: dict) -> str:
    """Walk MIME parts, preferring text/plain, falling back to a tag-stripped text/html --
    same goal as FreelanceCopilot's gmail_client.get_message_raw_html, reimplemented here in
    pure stdlib/requests so the two projects share only the Gmail account, not any code."""
    stack = [payload]
    html_fallback = None
    while stack:
        part = stack.pop()
        mime = part.get("mimeType", "")
        body_data = part.get("body", {}).get("data")
        if mime == "text/plain" and body_data:
            return _decode_part(body_data, _part_charset(part))
        if mime == "text/html" and body_data and html_fallback is None:
            html_fallback = _decode_part(body_data, _part_charset(part))
        stack.extend(part.get("parts", []) or [])
    if html_fallback:
        return re.sub(r"<[^>]+>", " ", html_fallback)
    return ""


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

    body = _extract_text(data.get("payload", {}))
    return {"id": message_id, "sender": get_header("From"), "subject": get_header("Subject"),
            "body": body[:BODY_TRUNCATE]}


def classify_newsletter(msg: dict) -> bool:
    prompt = CLASSIFY_PROMPT.format(sender=msg["sender"], subject=msg["subject"], body=msg["body"])
    try:
        raw = _call_ollama_json(prompt)
        return bool(raw.get("is_finance_newsletter", False))
    except Exception as e:
        # fail CLOSED here (unlike news_filter.py's fail-open buy gates): an email that can't
        # be classified is simply never used as a source -- there's no equivalent of "missing
        # a good trade" risk on the other side, so there's no reason to guess yes.
        print(f"  echec classification Ollama pour un mail: {e}", file=sys.stderr)
        return False


def _score_to_outlook(score: float | None) -> str:
    if score is None:
        return "no_data"
    if score <= SCORE_SOUS_PRESSION_MAX:
        return "sous_pression"
    if score >= SCORE_FLORISSANT_MIN:
        return "florissant"
    return "neutre"


def _parse_score(raw_score) -> float | None:
    if raw_score is None:
        return None
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(10.0, score))


def _classify_extract_sector(newsletter: dict) -> dict | None:
    """One Ollama call for this single extract: which sector (if any) is it actually about,
    and what score does it express for that sector. Returns None if the model failed, named no
    sector, or named something outside SECTOR_ETF (never trusted as a free-text match -- the
    prompt requires an exact name from the list, so anything else is treated as a non-answer)."""
    prompt = EXTRACT_SECTOR_PROMPT.format(subject=newsletter["subject"],
                                           body=newsletter["body"][:SYNTHESIS_EXTRACT_TRUNCATE],
                                           sector_list=SECTOR_LIST)
    try:
        raw = _call_ollama_json(prompt)
    except Exception as e:
        print(f"  echec classification sectorielle pour \"{newsletter['subject']}\": {e}",
              file=sys.stderr)
        return None
    sector = raw.get("sector")
    if sector not in SECTOR_ETF:
        return None
    score = _parse_score(raw.get("score"))
    if score is None:
        return None
    return {"sector": sector, "score": score, "reason": str(raw.get("reason", ""))[:300],
            "subject": newsletter["subject"]}


def classify_newsletters_by_sector(newsletters: list[dict], today: str) -> dict[str, list[dict]]:
    """Classifies each extract independently (see _classify_extract_sector) and groups today's
    readings by sector. Callers feed this straight into the sliding window (_update_window()) --
    it does NOT decide an outlook itself, since a single day's (or even a single extract's)
    reading is never enough on its own (see WINDOW_SIZE)."""
    by_sector: dict[str, list[dict]] = {}
    for n in newsletters:
        classified = _classify_extract_sector(n)
        if classified is not None:
            by_sector.setdefault(classified["sector"], []).append({
                "date": today, "score": classified["score"], "reason": classified["reason"],
                "subject": classified["subject"],
            })
    return by_sector


def _load_window() -> dict:
    if WINDOW_PATH.exists():
        try:
            return json.loads(WINDOW_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_window(window: dict):
    WINDOW_PATH.parent.mkdir(parents=True, exist_ok=True)
    WINDOW_PATH.write_text(json.dumps(window, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_window(window: dict, by_sector_today: dict[str, list[dict]]) -> dict:
    """Pushes today's per-extract readings onto each sector's sliding window. Only the
    WINDOW_SIZE most recent readings are kept per sector -- the oldest is evicted exactly when a
    new one for that sector pushes the window past WINDOW_SIZE, never by a calendar timer (see
    GROUNDING RULE in the module docstring)."""
    for sector, entries in by_sector_today.items():
        bucket = window.setdefault(sector, [])
        bucket.extend(entries)
        del bucket[:-WINDOW_SIZE]  # no-op while len(bucket) <= WINDOW_SIZE
    return window


def _compute_output_rows(window: dict) -> dict:
    """Turns the sliding window into the public sector_outlook.csv rows. A sector needs
    WINDOW_SIZE independent readings in its window before it gets a directional outlook; short of
    that it's "no_data" -- there is no separate staleness check, a sector's window (and hence its
    outlook) simply doesn't change until a fresh reading actually arrives for it."""
    results = {}
    for sector in SECTOR_ETF:
        entries = window.get(sector, [])
        if len(entries) < WINDOW_SIZE:
            reason = (f"en attente de davantage de lectures ({len(entries)}/{WINDOW_SIZE})" if entries
                      else "aucune newsletter n'a mentionne ce secteur")
            results[sector] = {"outlook": "no_data", "score": None, "reason": reason,
                                "last_updated": entries[-1]["date"] if entries else None}
            continue
        avg_score = sum(e["score"] for e in entries) / len(entries)
        reason = "; ".join(f"[{e['subject']}] {e['reason']}" for e in entries)[:300]
        results[sector] = {"outlook": _score_to_outlook(avg_score), "score": avg_score,
                            "reason": reason, "last_updated": entries[-1]["date"]}
    return results


def _append_to_newsletter_database(today: str, subject: str):
    """One row per email actually classified as a financial newsletter -- an ever-growing,
    never-pruned audit trail of what fed each day's synthesis (see module docstring), same
    shape as news_filter.py's NEWS_DB_PATH. Subject only: never the body or sender."""
    row = {"date": today, "subject": subject}
    NEWSLETTER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = not NEWSLETTER_DB_PATH.exists()
    pd.DataFrame([row], columns=NEWSLETTER_DB_COLUMNS).to_csv(
        NEWSLETTER_DB_PATH, mode="a", header=header, index=False)


def load_pressured_sectors() -> set:
    """Sectors currently rated "sous_pression" in sector_outlook.csv -- used by the veto bots
    (#10/#11/#12) to skip a candidate. Missing file (digest never run yet) or the column simply
    not having any "sous_pression" row both fail open to an empty set: no veto, rather than
    blocking every buy because the signal hasn't arrived yet, same posture as every other
    best-effort join in this repo."""
    if not OUT_PATH.exists():
        return set()
    df = pd.read_csv(OUT_PATH)
    return set(df.loc[df["outlook"] == "sous_pression", "sector"])


def main():
    state = _load_state()
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("last_run_date") == today:
        print(f"newsletter_digest deja execute aujourd'hui ({today}) -- rien a faire.")
        return

    missing = [v for v in ("GMAIL_REFRESH_TOKEN", "GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET")
               if not os.environ.get(v)]
    if missing:
        print(f"Variables manquantes ({', '.join(missing)}) -- digest Gmail ignore ce run.",
              file=sys.stderr)
        return

    try:
        token = get_access_token()
        message_ids = list_recent_message_ids(token)
    except Exception as e:
        print(f"echec acces Gmail: {e} -- digest ignore ce run.", file=sys.stderr)
        return

    # processed_message_ids holds exactly LAST run's fetched window (not an ever-growing
    # union): since GMAIL_QUERY always looks back only 2 days, anything older simply stops
    # being returned by Gmail on its own, so there's nothing to manually prune here.
    previously_processed = set(state.get("processed_message_ids", []))
    new_ids = [m for m in message_ids if m not in previously_processed]

    newsletters = []
    for mid in new_ids:
        msg = fetch_message(token, mid)
        if msg is not None and classify_newsletter(msg):
            newsletters.append(msg)
            _append_to_newsletter_database(today, msg["subject"])

    print(f"{len(new_ids)} nouveau(x) mail(s) examine(s), {len(newsletters)} newsletter(s) "
          f"financiere(s) retenue(s).")

    if newsletters:
        by_sector_today = classify_newsletters_by_sector(newsletters, today)
        window = _update_window(_load_window(), by_sector_today)
        _save_window(window)
        rows = [{"sector": s, **v} for s, v in _compute_output_rows(window).items()]
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False, encoding="utf-8")
    else:
        print("Aucune newsletter financiere aujourd'hui -- sector_outlook.csv inchange.")

    state["processed_message_ids"] = message_ids
    state["last_run_date"] = today
    _save_state(state)


if __name__ == "__main__":
    main()
