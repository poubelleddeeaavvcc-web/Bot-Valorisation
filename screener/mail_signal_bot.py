"""Bot #33 ("Echo"): paper-trading LONG and SHORT positions on individual stock tips found
directly in the user's own Gmail newsletters -- the per-ticker counterpart to
screener/newsletter_digest.py's per-sector qualitative signal (added 2026-09-11, per the
user's explicit request: "des mails il y a souvent des suggestions d'actions a fort potentiel
ou au contraire des chutes -- je veux des long ou des shorts sur ces actions, et savoir de
quelle newsletter ca vient pour identifier les bons/mauvais investisseurs").

Deliberately a SEPARATE script/workflow job from newsletter_digest.py, not folded into it,
for the same reason bots-delta got its own job in update-screener.yml and the digest itself
got split into its own repo (newsletter-digest-bot) on 2026-09-08: this module makes its own
full pass of Ollama calls (is-this-a-newsletter + per-extract ticker/sentiment extraction) on
top of whatever else already runs in the same CI window, and a shared 60-min job budget with
those doesn't leave enough margin -- see newsletter-digest-bot's own history (run cancelled
2026-09-10 at 56min after a state reset forced a full backlog reclassification). Independent
Gmail fetch + independent daily gate (own STATE_PATH), duplicated rather than shared, per this
repo's standing "duplicate small helpers instead of introducing cross-module coupling" style
(see simulate_constrained_portfolio.py's docstring).

MECHANICS
---------
Extraction (see _extract_ticker_signals()): each email already classified as a financial
newsletter (same classify_newsletter() gate as newsletter_digest.py) is passed to Ollama once,
asked to list every EXPLICIT stock tip it contains -- a ticker/company clearly framed as having
strong upside potential ("haussier") or headed for a drop ("baissier"). GROUNDING RULE (same
standing rule as news_filter.py/newsletter_digest.py): sentiment must be grounded in what the
extract actually says, never invented. The ticker SYMBOL itself is allowed to be the model's own
mapping from a clearly-named company (that's a lookup, not a fact the model could hallucinate
about the market) -- but it is NEVER trusted blindly: _resolve_ticker() fetches real price
history for it before any trade happens, and a symbol that doesn't resolve to real market data is
simply dropped, extract by extract.

Attribution (see _extract_source()): PRIVACY / REPO-PUBLIC CONSTRAINT (same standing rule as
newsletter_digest.py -- this repo pushes to a public GitHub remote). The user's own explicit
choice (2026-09-11): never persist the sender's full email address (that would publish which
newsletters they're personally subscribed to); only the sending domain (e.g. "fool.com") is
public-safe enough to keep, and it's exactly enough to build the "which sources call it right"
scorecard this bot exists for (see mail_signal_source_scorecard.csv).

Trading: one dedicated 300 EUR pool (own ledger, own cash file -- same "propre pool, propre
ledger" convention as Bot#25 "Delta"), long or short depending on the extracted sentiment.
- A signal for a ticker not currently held opens a position sized at TARGET_POSITION_SIZE (or
  whatever whole-share amount gets closest to it, same fractional_eligible() split as every
  other capital-tracking bot -- imported, not reimplemented).
- A signal that CONTRADICTS an existing open position (bullish while short, bearish while long)
  closes it immediately ("signal_inverse") -- a fresh, explicit tip against the current thesis
  is stronger grounds to bail than waiting for the price to prove it, and the freed cash is
  eligible to reopen the position on the new side the same run.
- A signal for a ticker already held on the SAME side is a no-op (already positioned).

Exit logic on every run, independent of new signals -- the user's own explicit direction
(2026-09-11: "il faut qu'on instaure une logique de quand vendre"), since this is a fresh
mechanic with no fundamentals model of its own to fall back on for continued-thesis checks:
  - STOP_LOSS_PCT / the ratcheting stop (RATCHET_STEP_PCT/RATCHET_GIVEBACK_PCT), same constants
    imported from simulate_portfolio.py, applied to whichever side's own "unrealized" (a short's
    gain is the mirror of a long's -- see _unrealized_return()) -- symmetric risk discipline
    rather than leaving a short's theoretically-unbounded downside unmanaged.
  - TAKE_PROFIT_PCT: a single newsletter tip, unlike the valuation model the other bots use,
    carries no ongoing thesis to re-check for "still has room to run" -- so a big favorable move
    locks in profit rather than riding indefinitely.
  - MAX_HOLDING_DAYS: a tip's informational edge decays; unlike a valuation gap (which can stay
    open for months until the price catches up), nothing here re-confirms the thesis is still
    live, so a stale position is force-closed regardless of P&L.

SIMPLIFICATION (documented, not hidden): a short's cash accounting mirrors a long's --
entry_value_eur leaves the cash pool at open and current_value_eur = entry_value_eur * (1 +
unrealized) returns at close, rather than modeling real short-sale mechanics (borrow fees,
margin calls, proceeds-from-sale-generates-cash-immediately). That keeps the ledger's
open/close cash bookkeeping identical for both sides; it does NOT reproduce a real broker's
short economics, only this bot's own directional bet's P&L.
"""
import base64
import json
import os
import pathlib
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.simulate_portfolio import STOP_LOSS_PCT, RATCHET_STEP_PCT, RATCHET_GIVEBACK_PCT  # noqa: E402
from screener.simulate_constrained_portfolio import fetch_fx_rates, to_eur, fractional_eligible  # noqa: E402

STATE_PATH = HERE / "results/screener/mail_signal_state.json"
CASH_PATH = HERE / "results/simulation/mail_signal_state.json"
LEDGER_PATH = HERE / "results/simulation/mail_signal_ledger.csv"
SUMMARY_PATH = HERE / "results/simulation/mail_signal_summary.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/mail_signal_equity_curve.csv"
SCORECARD_PATH = HERE / "results/screener/mail_signal_source_scorecard.csv"

TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_API = "https://gmail.googleapis.com/gmail/v1/users/me"
GMAIL_QUERY = "newer_than:2d"  # same buffer/reasoning as newsletter_digest.py
MAX_MESSAGES = 300
BODY_TRUNCATE = 900
EXTRACT_TRUNCATE = 900  # a per-ticker tip can be buried deeper in the email than the
# sector-level gist newsletter_digest.py extracts, so this module truncates less aggressively

STARTING_CAPITAL = 300.0        # own pool, separate from every other bot -- see module docstring
STARTING_SLOTS = 9              # -> TARGET_POSITION_SIZE ~= 33 EUR/slot, same granularity as Bot#2/3/25
TARGET_POSITION_SIZE = STARTING_CAPITAL / STARTING_SLOTS
MAX_WHOLE_SHARE_OVERSHOOT = 2.5  # same convention as simulate_constrained_portfolio.py
TRADE_FEE_EUR = 1.0

TAKE_PROFIT_PCT = 0.30    # see module docstring's EXIT LOGIC section
MAX_HOLDING_DAYS = 20     # ~1 trading month -- a tip's edge decays, unlike a valuation gap
MAX_TICKERS_PER_EMAIL = 3  # bounds noise/cost: a newsletter that name-drops a dozen tickers in
# passing is diluting its own conviction, not producing a dozen real tips

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"  # same as newsletter_digest.py -- see that module's docstring for
# why the 3b model shared with news_filter.py's bots isn't reliable enough for this kind of call
OLLAMA_TIMEOUT = 180
OLLAMA_MAX_WORKERS = 2  # see newsletter_digest.py's own constant for the memory-ceiling reasoning

CLASSIFY_PROMPT = """Voici un email recu aujourd'hui :

Expediteur : {sender}
Sujet : {subject}
Extrait : {body}

Ceci est-il une newsletter financiere/economique (actualite des marches, d'un secteur, ou macroeconomique) -- par opposition a un email personnel, professionnel, transactionnel, ou publicitaire non lie a la finance ?

Reponds UNIQUEMENT en JSON : {{"is_finance_newsletter": true|false, "reason": "<une phrase courte>"}}
"""

# One call per EXTRACT returning a LIST (not one ticker per call) -- same "ask once, let the
# model enumerate" shape as newsletter_digest.py's per-sector call, but this module wants every
# explicit tip in the email, not just its single main topic.
EXTRACT_TICKER_PROMPT = """Voici un extrait de newsletter financiere recue aujourd'hui :

Sujet : {subject}
Extrait : {body}

Identifie chaque action individuelle presentee dans cet extrait comme ayant soit un FORT POTENTIEL DE HAUSSE (recommandation d'achat, catalyseur positif, objectif de cours releve...), soit un RISQUE DE FORTE BAISSE (avertissement, degradation, catalyseur negatif...). Ignore les actions seulement mentionnees en passant sans avis directionnel clair.

Pour chaque action identifiee (maximum {max_tickers}), donne son ticker boursier exact (le symbole utilise sur les marches, ex: AAPL, MC.PA -- si seul le nom de l'entreprise est donne, indique le ticker que tu connais pour cette entreprise) et le sens ("haussier" ou "baissier").

Si aucune action n'a d'avis directionnel clair et explicite, reponds avec une liste vide.

Reponds UNIQUEMENT en JSON : {{"tips": [{{"ticker": "<SYMBOLE>", "sentiment": "haussier|baissier", "reason": "<une phrase courte citant ce que dit l'extrait>"}}, ...]}}
"""


def _call_ollama_json(prompt: str) -> dict:
    """Same shape as newsletter_digest.py's _call_ollama_json -- duplicated per this repo's
    small-helper convention (see module docstring)."""
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json", "keep_alive": "20m"}
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


def _extract_text(payload: dict) -> str:
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


def _extract_source(sender: str) -> str:
    """Sending domain only -- e.g. "Morning Brew <crew@morningbrew.com>" -> "morningbrew.com".
    Never the local part / full address (see PRIVACY note in module docstring)."""
    m = re.search(r'@([\w.-]+\.[A-Za-z]{2,})', sender or "")
    return m.group(1).lower() if m else "inconnu"


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
            "source": _extract_source(get_header("From")), "body": body[:BODY_TRUNCATE]}


def classify_newsletter(msg: dict) -> bool:
    prompt = CLASSIFY_PROMPT.format(sender=msg["sender"], subject=msg["subject"], body=msg["body"])
    try:
        raw = _call_ollama_json(prompt)
        return bool(raw.get("is_finance_newsletter", False))
    except Exception as e:
        print(f"  echec classification Ollama pour un mail: {e}", file=sys.stderr)
        return False


def _extract_ticker_signals(msg: dict) -> list[dict]:
    """One Ollama call per newsletter, returns every grounded (ticker, sentiment) tip it
    contains -- NOT yet validated against real market data, see _resolve_ticker()."""
    prompt = EXTRACT_TICKER_PROMPT.format(subject=msg["subject"], body=msg["body"][:EXTRACT_TRUNCATE],
                                           max_tickers=MAX_TICKERS_PER_EMAIL)
    try:
        raw = _call_ollama_json(prompt)
    except Exception as e:
        print(f"  echec extraction tickers pour \"{msg['subject']}\": {e}", file=sys.stderr)
        return []
    tips = raw.get("tips") or []
    out = []
    for tip in tips[:MAX_TICKERS_PER_EMAIL]:
        ticker = str(tip.get("ticker") or "").strip().upper()
        sentiment = tip.get("sentiment")
        if not ticker or sentiment not in ("haussier", "baissier"):
            continue
        out.append({"ticker": ticker, "side": "long" if sentiment == "haussier" else "short",
                     "reason": str(tip.get("reason", ""))[:300], "source": msg["source"],
                     "subject": msg["subject"]})
    return out


def _resolve_ticker(ticker: str) -> dict | None:
    """GROUNDING BACKSTOP: never trusts Ollama's ticker mapping blindly -- fetches real price
    history before this ticker is ever allowed to drive a trade. Returns None (dropped, not
    guessed at) if it doesn't resolve to real, current market data."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")["Close"].dropna()
        if hist.empty:
            return None
        price = float(hist.iloc[-1])
        fast_info = tk.fast_info
        currency = fast_info.get("currency") if fast_info else None
        name = None
        try:
            name = tk.info.get("shortName")
        except Exception:
            pass
        return {"price": price, "currency": currency, "name": name or ticker}
    except Exception as e:
        print(f"  echec resolution ticker {ticker}: {e}", file=sys.stderr)
        return None


LEDGER_COLUMNS = [
    "ticker", "name", "side", "source", "status", "currency", "fractional",
    "entry_date", "entry_price", "shares", "entry_value_eur",
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
        return df[LEDGER_COLUMNS]
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def load_cash() -> float:
    return _load_json(CASH_PATH, {"cash_eur": STARTING_CAPITAL})["cash_eur"]


def save_cash(cash: float):
    _save_json(CASH_PATH, {"cash_eur": cash})


def _unrealized_return(side: str, entry_price: float, last_price: float) -> float:
    """A short's gain is the mirror of a long's -- see module docstring's SIMPLIFICATION note."""
    raw = last_price / entry_price - 1
    return raw if side == "long" else -raw


def recheck_and_exit(ledger: pd.DataFrame, today: str, cash: float) -> tuple:
    for idx in ledger.index[ledger["status"] == "open"]:
        ticker = ledger.at[idx, "ticker"]
        side = ledger.at[idx, "side"]
        resolved = _resolve_ticker(ticker)
        if resolved is None:
            continue  # transient fetch failure -- retry next run, don't force an exit on it

        entry_price = ledger.at[idx, "entry_price"]
        unrealized = _unrealized_return(side, entry_price, resolved["price"])
        entry_value_eur = ledger.at[idx, "entry_value_eur"]
        current_value = entry_value_eur * (1 + unrealized)  # see SIMPLIFICATION in module docstring

        ledger.at[idx, "last_check_date"] = today
        ledger.at[idx, "last_price"] = resolved["price"]
        ledger.at[idx, "current_value_eur"] = current_value
        ledger.at[idx, "unrealized_return_pct"] = unrealized

        peak = ledger.at[idx, "peak_unrealized_return_pct"]
        if pd.isna(peak) or unrealized > peak:
            ledger.at[idx, "peak_unrealized_return_pct"] = unrealized
            ledger.at[idx, "peak_date"] = today
        peak = ledger.at[idx, "peak_unrealized_return_pct"]

        entry_date = pd.Timestamp(ledger.at[idx, "entry_date"])
        holding_days_elapsed = (pd.Timestamp(today) - entry_date).days

        stop_loss_hit = unrealized <= STOP_LOSS_PCT
        take_profit_hit = unrealized >= TAKE_PROFIT_PCT
        milestone = int(peak // RATCHET_STEP_PCT) if pd.notna(peak) else 0
        trailing_stop_hit = milestone >= 1 and unrealized <= milestone * RATCHET_STEP_PCT - RATCHET_GIVEBACK_PCT
        max_holding_hit = holding_days_elapsed >= MAX_HOLDING_DAYS

        if stop_loss_hit or take_profit_hit or trailing_stop_hit or max_holding_hit:
            reason = ("trailing_stop" if trailing_stop_hit else
                      "stop_loss" if stop_loss_hit else
                      "take_profit" if take_profit_hit else "duree_max_atteinte")
            net_exit_value = current_value - TRADE_FEE_EUR
            net_return = unrealized - TRADE_FEE_EUR / entry_value_eur
            ledger.at[idx, "status"] = "closed"
            ledger.at[idx, "exit_date"] = today
            ledger.at[idx, "exit_price"] = resolved["price"]
            ledger.at[idx, "exit_reason"] = reason
            ledger.at[idx, "exit_value_eur"] = net_exit_value
            ledger.at[idx, "return_pct"] = net_return
            ledger.at[idx, "holding_days"] = holding_days_elapsed
            cash += net_exit_value
            print(f"  CLOTURE {side.upper()} {ticker} : {reason}, retour net {net_return:+.1%} "
                  f"(frais {TRADE_FEE_EUR:.2f} EUR deduits), {net_exit_value:.2f} EUR reinjectes en cash")
    return ledger, cash


def _fx_rate_for(currency: str | None, fx_rates: dict) -> dict:
    """Extends fx_rates in place with whichever currency this position needs -- tickers here
    come from arbitrary mail tips, not a pre-scoped universe, so the set of currencies needed
    can't be known ahead of a single fetch_fx_rates() call the way the other bots do it."""
    key = "GBP" if currency == "GBp" else (currency or "EUR")
    if key not in fx_rates:
        fx_rates.update(fetch_fx_rates({key}))
    return fx_rates


def _open_position(ledger: pd.DataFrame, ticker: str, side: str, source: str, reason: str,
                    resolved: dict, cash: float, today: str, fx_rates: dict) -> tuple:
    _fx_rate_for(resolved.get("currency"), fx_rates)
    price_eur = to_eur(resolved["price"], resolved.get("currency"), fx_rates)
    if price_eur is None or price_eur <= 0 or price_eur > cash:
        return ledger, cash, False

    fractional = fractional_eligible(ticker, None, None)
    if fractional:
        cost = min(TARGET_POSITION_SIZE, cash)
        shares = cost / price_eur
    else:
        if price_eur > MAX_WHOLE_SHARE_OVERSHOOT * TARGET_POSITION_SIZE:
            return ledger, cash, False
        target_shares = max(1, int(TARGET_POSITION_SIZE // price_eur))
        max_affordable = int(cash // price_eur)
        shares = min(target_shares, max_affordable)
        if shares < 1:
            return ledger, cash, False
        cost = shares * price_eur

    new_row = {
        "ticker": ticker, "name": resolved.get("name") or ticker, "side": side, "source": source,
        "status": "open", "currency": resolved.get("currency"), "fractional": bool(fractional),
        "entry_date": today, "entry_price": resolved["price"], "shares": shares,
        "entry_value_eur": cost, "last_check_date": today, "last_price": resolved["price"],
        "current_value_eur": cost, "unrealized_return_pct": 0.0,
        "peak_unrealized_return_pct": 0.0, "peak_date": today,
        "exit_date": None, "exit_price": None, "exit_reason": None,
        "exit_value_eur": None, "return_pct": None, "holding_days": None,
        "signal_reason": reason,
    }
    ledger = pd.concat([ledger, pd.DataFrame([new_row])], ignore_index=True)
    cash -= cost
    kind = "fractionne" if fractional else "entier"
    print(f"  OUVERTURE {side.upper()} {ticker} ({source}) : {cost:.2f} EUR ({shares:.4f} actions, {kind}) "
          f"@ {resolved['price']:.2f} {resolved.get('currency') or '?'}")
    return ledger, cash, True


def apply_signals(ledger: pd.DataFrame, signals: list[dict], cash: float, today: str, fx_rates: dict) -> tuple:
    """Opens/reverses positions from today's freshly-extracted signals -- see module docstring's
    Trading section for the same-side/opposite-side/not-held decision tree."""
    for sig in signals:
        ticker, side, source, reason = sig["ticker"], sig["side"], sig["source"], sig["reason"]
        open_row = ledger[(ledger["ticker"] == ticker) & (ledger["status"] == "open")]

        if len(open_row):
            existing_side = open_row.iloc[0]["side"]
            if existing_side == side:
                continue  # already positioned this direction -- no-op
            # opposite signal: close the existing position now, regardless of its current P&L
            idx = open_row.index[0]
            resolved = _resolve_ticker(ticker)
            if resolved is None:
                continue
            entry_price = ledger.at[idx, "entry_price"]
            unrealized = _unrealized_return(existing_side, entry_price, resolved["price"])
            entry_value_eur = ledger.at[idx, "entry_value_eur"]
            net_exit_value = entry_value_eur * (1 + unrealized) - TRADE_FEE_EUR
            net_return = unrealized - TRADE_FEE_EUR / entry_value_eur
            ledger.at[idx, "status"] = "closed"
            ledger.at[idx, "exit_date"] = today
            ledger.at[idx, "exit_price"] = resolved["price"]
            ledger.at[idx, "exit_reason"] = "signal_inverse"
            ledger.at[idx, "exit_value_eur"] = net_exit_value
            ledger.at[idx, "return_pct"] = net_return
            ledger.at[idx, "holding_days"] = (pd.Timestamp(today) - pd.Timestamp(ledger.at[idx, "entry_date"])).days
            cash += net_exit_value
            print(f"  CLOTURE {existing_side.upper()} {ticker} : signal_inverse, retour net {net_return:+.1%}")

        if cash < 1.0:
            continue
        resolved = _resolve_ticker(ticker)
        if resolved is None:
            continue
        ledger, cash, _ = _open_position(ledger, ticker, side, source, reason, resolved, cash, today, fx_rates)
    return ledger, cash


def write_scorecard(ledger: pd.DataFrame):
    """The "quels sont les bons/mauvais investisseurs" leaderboard this bot exists for --
    recomputed fresh from the ledger every run, grouped by sending domain (see _extract_source
    / PRIVACY note in module docstring). Rows with too few closed signals to mean anything are
    still listed (n_closed=0 is informative on its own -- a source with many open signals and
    no verdict yet), just not rankable by win_rate."""
    rows = []
    for source, grp in ledger.groupby("source"):
        closed_grp = grp[grp["status"] == "closed"]
        rows.append({
            "source": source,
            "n_signals_total": len(grp),
            "n_closed": len(closed_grp),
            "n_open": int((grp["status"] == "open").sum()),
            "win_rate_closed": float((closed_grp["return_pct"] > 0).mean()) if len(closed_grp) else None,
            "avg_return_closed": float(closed_grp["return_pct"].mean()) if len(closed_grp) else None,
        })
    columns = ["source", "n_signals_total", "n_closed", "n_open", "win_rate_closed", "avg_return_closed"]
    out = pd.DataFrame(rows, columns=columns)
    if len(out):
        out = out.sort_values(by=["win_rate_closed", "n_closed"], ascending=[False, False], na_position="last")
    SCORECARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SCORECARD_PATH, index=False)


def write_summary(ledger: pd.DataFrame, cash: float):
    closed = ledger[ledger["status"] == "closed"]
    open_pos = ledger[ledger["status"] == "open"]
    total_equity = cash + open_pos["current_value_eur"].sum()
    summary = {
        "cash_eur": cash, "total_equity_eur": total_equity,
        "total_return_pct": total_equity / STARTING_CAPITAL - 1,
        "nb_open": len(open_pos), "nb_closed": len(closed),
        "nb_long_open": int((open_pos["side"] == "long").sum()),
        "nb_short_open": int((open_pos["side"] == "short").sum()),
        "win_rate_closed": float((closed["return_pct"] > 0).mean()) if len(closed) else None,
        "avg_return_closed": float(closed["return_pct"].mean()) if len(closed) else None,
    }
    SUMMARY_PATH.write_text(pd.Series(summary).to_json(), encoding="utf-8")
    print(f"\n=== Bot #33 Echo : {summary['nb_open']} positions ouvertes "
          f"({summary['nb_long_open']} long / {summary['nb_short_open']} short), "
          f"{cash:.2f} EUR cash, valeur totale {total_equity:.2f} EUR "
          f"({summary['total_return_pct']:+.1%} depuis le depart) ===")


def append_equity_curve_point(cash: float, total_equity: float, nb_open: int, nb_closed: int):
    row = {"timestamp": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
           "cash_eur": cash, "total_equity_eur": total_equity, "n_open": nb_open, "n_closed": nb_closed}
    header = not EQUITY_CURVE_PATH.exists()
    EQUITY_CURVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(EQUITY_CURVE_PATH, mode="a", header=header, index=False)


def main():
    state = _load_json(STATE_PATH, {})
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("last_run_date") == today:
        print(f"mail_signal_bot deja execute aujourd'hui ({today}) -- rien a faire.")
        return

    ledger = load_ledger()
    cash = load_cash()

    signals: list[dict] = []
    message_ids = None
    missing = [v for v in ("GMAIL_REFRESH_TOKEN", "GMAIL_CLIENT_ID", "GMAIL_CLIENT_SECRET")
               if not os.environ.get(v)]
    if missing:
        print(f"Variables manquantes ({', '.join(missing)}) -- run ignore.", file=sys.stderr)
    else:
        try:
            token = get_access_token()
            message_ids = list_recent_message_ids(token)
            previously_processed = set(state.get("processed_message_ids", []))
            new_ids = [m for m in message_ids if m not in previously_processed]

            msgs = [m for m in (fetch_message(token, mid) for mid in new_ids) if m is not None]
            with ThreadPoolExecutor(max_workers=OLLAMA_MAX_WORKERS) as ex:
                is_newsletter = dict(zip((m["id"] for m in msgs), ex.map(classify_newsletter, msgs)))
            newsletters = [m for m in msgs if is_newsletter.get(m["id"])]

            print(f"{len(new_ids)} nouveau(x) mail(s) examine(s), {len(newsletters)} newsletter(s) "
                  f"financiere(s) retenue(s).")

            with ThreadPoolExecutor(max_workers=OLLAMA_MAX_WORKERS) as ex:
                futures = [ex.submit(_extract_ticker_signals, m) for m in newsletters]
                for fut in as_completed(futures):
                    try:
                        signals.extend(fut.result())
                    except Exception as e:
                        print(f"  echec extraction (parallele, inattendu): {e}", file=sys.stderr)
        except Exception as e:
            print(f"echec acces Gmail: {e} -- run ignore.", file=sys.stderr)

    if signals:
        print(f"{len(signals)} tip(s) extrait(s) : "
              + "; ".join(f"{s['ticker']}({s['side']},{s['source']})" for s in signals))

    fx_rates = fetch_fx_rates({"USD", "EUR"})  # warm the common-case cache; per-trade calls
    # above fetch the rest on demand since tickers are unpredictable ahead of time here
    ledger, cash = recheck_and_exit(ledger, today, cash)
    if signals:
        ledger, cash = apply_signals(ledger, signals, cash, today, fx_rates)

    write_summary(ledger, cash)
    write_scorecard(ledger)
    open_pos = ledger[ledger["status"] == "open"]
    append_equity_curve_point(cash, cash + open_pos["current_value_eur"].sum(), len(open_pos),
                               int((ledger["status"] == "closed").sum()))

    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    save_cash(cash)

    if message_ids is not None:
        state["processed_message_ids"] = message_ids
    state["last_run_date"] = today
    _save_json(STATE_PATH, state)


if __name__ == "__main__":
    main()
