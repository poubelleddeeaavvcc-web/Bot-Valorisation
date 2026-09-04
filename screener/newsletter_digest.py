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
the per-sector synthesis prompt.

GROUNDING RULE (same as screener/news_filter.py -- see that module's docstring and the project's
own standing rule): Ollama must never invent a sector's outlook. synthesize_sector_outlook() only
ever asks about text actually fetched this run; a sector nobody's newsletter mentioned today comes
back "no_data" and keeps its last known value in sector_outlook.csv rather than being overwritten
by a guess.

PRIVACY / REPO-PUBLIC CONSTRAINT: this repo pushes to a public GitHub remote with a Pages-served
dashboard. Raw email text, subjects, and sender addresses are therefore NEVER written to disk
beyond the run's own memory and the minimal state file (a timestamp/id list, no content) -- only
Ollama's own short synthesized outlook + one-line reason per sector ever reaches
data/universe/sector_outlook.csv.

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
from datetime import datetime, timezone

import pandas as pd
import requests

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.build_trending_universe import SECTOR_ETF  # noqa: E402

STATE_PATH = HERE / "results/screener/newsletter_digest_state.json"
OUT_PATH = HERE / "data/universe/sector_outlook.csv"

TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
GMAIL_QUERY = "newer_than:2d"  # 2 days, not 1: buffer against a run being skipped/late without
# losing a day's newsletters entirely
MAX_MESSAGES = 300
BODY_TRUNCATE = 1500  # per-message character cap fed to the classifier prompt
SYNTHESIS_EXTRACT_TRUNCATE = 500  # per-message cap when building the sector-synthesis prompt

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_TIMEOUT = 90

VALID_OUTLOOKS = {"sous_pression", "neutre", "florissant", "no_data"}

CLASSIFY_PROMPT = """Voici un email recu aujourd'hui :

Expediteur : {sender}
Sujet : {subject}
Extrait : {body}

Ceci est-il une newsletter financiere/economique (actualite des marches, d'un secteur, ou macroeconomique) -- par opposition a un email personnel, professionnel, transactionnel, ou publicitaire non lie a la finance ?

Reponds UNIQUEMENT en JSON : {{"is_finance_newsletter": true|false, "reason": "<une phrase courte>"}}
"""

SECTOR_PROMPT = """Voici des extraits de newsletters financieres recues aujourd'hui :

{extracts}

D'apres CES SEULS extraits, le secteur "{sector}" est-il actuellement plutot sous pression, neutre, ou florissant ? Si aucun extrait ne parle de ce secteur, reponds "no_data" -- ne devine jamais a partir d'autre chose que ce texte.

Reponds UNIQUEMENT en JSON : {{"outlook": "sous_pression"|"neutre"|"florissant"|"no_data", "reason": "<une phrase courte>"}}
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


def _decode_part(data_b64url: str) -> str:
    padded = data_b64url + "=" * (-len(data_b64url) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace")


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
            return _decode_part(body_data)
        if mime == "text/html" and body_data and html_fallback is None:
            html_fallback = _decode_part(body_data)
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


def synthesize_sector_outlook(newsletters: list[dict]) -> dict:
    extracts = "\n\n".join(f"[{n['subject']}] {n['body'][:SYNTHESIS_EXTRACT_TRUNCATE]}"
                            for n in newsletters)
    results = {}
    for sector in SECTOR_ETF:
        prompt = SECTOR_PROMPT.format(extracts=extracts, sector=sector)
        try:
            raw = _call_ollama_json(prompt)
            outlook = raw.get("outlook")
            if outlook not in VALID_OUTLOOKS:
                outlook = "no_data"
            results[sector] = {"outlook": outlook, "reason": str(raw.get("reason", ""))[:300]}
        except Exception as e:
            print(f"  echec synthese Ollama pour {sector}: {e}", file=sys.stderr)
            results[sector] = {"outlook": "no_data", "reason": f"ollama indisponible ({e})"}
    return results


def merge_into_output(new_results: dict, today: str):
    """A sector with no fresh signal today (no_data) keeps whatever it last had -- overwriting
    a real prior read with an absence-of-mail would throw away information the bots downstream
    still rely on."""
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH).set_index("sector").to_dict("index")
    else:
        existing = {}
    for sector in SECTOR_ETF:
        res = new_results.get(sector)
        if res and res["outlook"] != "no_data":
            existing[sector] = {"outlook": res["outlook"], "reason": res["reason"], "last_updated": today}
        elif sector not in existing:
            existing[sector] = {"outlook": "no_data",
                                 "reason": "aucune newsletter n'a mentionne ce secteur",
                                 "last_updated": today}
    rows = [{"sector": s, **v} for s, v in existing.items()]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)


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

    print(f"{len(new_ids)} nouveau(x) mail(s) examine(s), {len(newsletters)} newsletter(s) "
          f"financiere(s) retenue(s).")

    if newsletters:
        results = synthesize_sector_outlook(newsletters)
        merge_into_output(results, today)
    else:
        print("Aucune newsletter financiere aujourd'hui -- sector_outlook.csv inchange.")

    state["processed_message_ids"] = message_ids
    state["last_run_date"] = today
    _save_state(state)


if __name__ == "__main__":
    main()
