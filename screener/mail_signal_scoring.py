"""Bot #33 Courrier -- journal of every newsletter tip, per-newsletter scoring, and consensus
(added 2026-10-03, at the user's explicit request after the go-live analysis of the bot).

WHY A JOURNAL (and why the old lab ledger was retired on 2026-10-04)
--------------------------------------------------------------------
The old lab ledger (mail_signal_ledger.csv) only recorded the tips that opened a position: a second
newsletter recommending a ticker already held on the same side was a silent no-op, so its opinion
was lost. And the old scorecard could only rank a newsletter once its positions had CLOSED --
after 3 weeks, 9 of the 13 sources had zero closed positions, i.e. no verdict at all.

The journal records EVERY validated tip from EVERY newsletter (good or bad), one row per
(email, ticker), and evaluates each one mechanically at fixed horizons -- J+5, J+20, J+60 trading
days -- as an excess return against a benchmark of the same market (see BENCHMARKS). That is what
the per-newsletter score is built from, independent of any exit rule, so the scorecard fills in on
its own as days pass.

NEWSLETTER STATUS ("fiable" / "bruit" / "observation")
------------------------------------------------------
The user's own direction (2026-10-03): bad newsletters must not be engaged in the real strategy,
but their noise must still be heard -- "si tout le monde dit d'acheter une action en particulier
leur avis est positif et peut etre utile, idem si tout le monde rejette une action sauf 1
newsletter". So every newsletter gets a status from its J+20 track record:
  - "observation": fewer than N_MIN evaluated tips -- not enough data to judge either way.
  - "fiable": N_MIN+ evaluated tips and a t-stat of its mean excess return >= T_PROMOTE. Only
    these can trigger a position in the real strategy (see mail_signal_real.py).
  - "bruit": N_MIN+ evaluated tips, not reliable. Never triggers anything on its own.
Hysteresis: a "fiable" newsletter only gets demoted once its t-stat falls below T_DEMOTE, so a
single bad week doesn't flip it back and forth (each flip would mean real trading fees).

The t-stat uses a POOLED volatility (std of every evaluated tip, all newsletters together) rather
than each newsletter's own: with 15 tips, a newsletter's own std is too noisy to divide by -- one
lucky newsletter with 15 tips that happen to be close together would look spectacular.

EDGE ("rendement espere") AND THE CROWD
---------------------------------------
Each newsletter's edge is its mean J+20 excess return SHRUNK toward 0 with the weight of
K_SHRINK virtual zero-return tips (edge = sum / (n + K_SHRINK)): a newsletter with 3 lucky tips
gets a small edge, one with 60 consistent tips keeps almost all of its mean.

"La foule" = every newsletter that is NOT fiable, taken together. Its consensus on a ticker is
measured like a newsletter of its own: an "event" is a tip that agrees with at least one OTHER
non-reliable newsletter on the same ticker/side within CONSENSUS_WINDOW_DAYS, and the crowd's
edge is the shrunk mean J+20 excess return of those events. Its prior is CROWD_PRIOR_EDGE (> 0,
the user's own hypothesis that agreement is informative) so the consensus has a modest effect
from day one; the measured data then overrides the prior -- including flipping its sign if the
crowd turns out to be a contrarian indicator (when everyone agrees, the stock underperforms).

consensus_for() combines both for one ticker (see its docstring): reliable votes give the base
expected excess return, the crowd adds or subtracts its own measured edge times its vote balance.
If the crowd itself ever earns "fiable" status on its events, a strong crowd consensus can trigger
a position on its own (the bad newsletters individually still never can).
"""
import json
import math
import pathlib
import sys

import numpy as np
import warnings

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
# appending a row whose not-yet-filled columns are all None -- intended, and harmless under pandas<3
warnings.filterwarnings("ignore", category=FutureWarning, message=".*concatenation with empty or all-NA.*")
JOURNAL_PATH = HERE / "results/screener/mail_signal_journal.csv"

HORIZONS = (5, 20, 60)        # trading days
PRIMARY_HORIZON = 20          # the horizon newsletters are scored and ranked on
N_MIN = 15                    # evaluated tips before a newsletter can leave "observation"
T_PROMOTE = 1.5               # t-stat to become "fiable" (~7% one-sided false-positive per source)
T_DEMOTE = 1.0                # a "fiable" newsletter keeps its status down to this t-stat
K_SHRINK = 10                 # virtual zero-excess tips added to each newsletter's mean
DEFAULT_SIGMA = 0.065         # J+20 excess-return dispersion measured on the first 166 lab
# positions (2026-10-03 analysis) -- used until the journal has 30+ evaluations of its own
MIN_POOLED_SIGMA = 0.02       # floor, so an unusually calm sample can't inflate every t-stat
CONSENSUS_WINDOW_DAYS = 14    # a vote counts for this long, with linear decay (see consensus_for)
CROWD_MIN_VOTERS = 2          # distinct non-reliable newsletters agreeing = a crowd "event"
CROWD_PRIOR_EDGE = 0.005      # +0.5% at J+20: the user's hypothesis that agreement is informative
CROWD_PUBLICATION = "(consensus des newsletters non fiables)"
REPEAT_WINDOW_DAYS = 7        # same newsletter, same ticker, same side within this = a repeat
GIVE_UP_EVAL_DAYS = 150       # past this, a still-unevaluated tip (delisted...) is closed out

# Benchmark per trading currency -- ETFs (not price indices) so adjusted closes include
# dividends, same as the adjusted closes used for the stock itself.
BENCHMARKS = {
    "USD": "SPY", "EUR": "EXSA.DE", "CHF": "EXSA.DE", "SEK": "EXSA.DE", "DKK": "EXSA.DE",
    "NOK": "EXSA.DE", "GBp": "ISF.L", "GBP": "ISF.L", "CAD": "XIU.TO", "JPY": "1306.T",
    "HKD": "2800.HK", "TWD": "0050.TW",
}
DEFAULT_BENCHMARK = "SPY"
# Earliest plausible UTC close hour per market (summer time, i.e. the earliest of the year). An
# email received at/after it is anchored on the NEXT session's close -- conservative on purpose:
# when in doubt the tip is evaluated from a later price, never from one that predates it.
CLOSE_CUTOFF_UTC = {"USD": 20, "CAD": 20, "EUR": 14, "CHF": 14, "SEK": 14, "DKK": 14, "NOK": 14,
                    "GBp": 15, "GBP": 15, "JPY": 6, "HKD": 8, "TWD": 5}

JOURNAL_COLUMNS = [
    "signal_id", "message_id", "mail_date_utc", "recorded_at_utc", "publication", "domain", "author",
    "company", "ticker", "name", "exchange", "currency", "side", "citation", "backfill", "repeat",
    "anchor_date", "anchor_price", "benchmark", "bench_anchor",
    "ret_5", "ex_5", "ret_20", "ex_20", "ret_60", "ex_60", "eval_done",
]


def _flag(series: pd.Series) -> pd.Series:
    """CSV round-trips booleans as bool, "True"/"False" strings or NaN depending on the column's
    history -- normalize to a real bool Series."""
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def load_journal() -> pd.DataFrame:
    if JOURNAL_PATH.exists():
        df = pd.read_csv(JOURNAL_PATH)
        for c in JOURNAL_COLUMNS:
            if c not in df.columns:
                df[c] = None
        df = df[JOURNAL_COLUMNS]
        for c in ("signal_id", "message_id", "mail_date_utc", "recorded_at_utc", "publication", "domain",
                  "author", "company", "ticker", "name", "exchange", "currency", "side", "citation",
                  "anchor_date", "benchmark"):
            df[c] = df[c].astype(object)
        return df
    return pd.DataFrame(columns=JOURNAL_COLUMNS)


def save_journal(journal: pd.DataFrame):
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    journal.to_csv(JOURNAL_PATH, index=False)


def add_signals(journal: pd.DataFrame, signals: list[dict], now_iso: str) -> tuple:
    """Appends validated tips. IDEMPOTENT: a (message_id, ticker) already in the journal is
    skipped -- this is the key that keeps a re-run, a retried push or an overlapping backfill from
    ever counting (or trading) the same tip twice. Returns (journal, list of newly added signal
    dicts) -- only the newly added ones may drive the lab ledger's trades."""
    known = set(journal["signal_id"].astype(str))
    added_rows, added = [], []
    existing = journal[["publication", "ticker", "side", "mail_date_utc"]].copy()
    existing["mail_date_utc"] = pd.to_datetime(existing["mail_date_utc"], utc=True, errors="coerce")
    for sig in signals:
        sid = f"{sig['message_id']}:{sig['ticker']}"
        if sid in known:
            continue
        known.add(sid)
        mail_ts = pd.Timestamp(sig["mail_date_utc"])
        prior = existing[(existing["publication"] == sig["publication"]) & (existing["ticker"] == sig["ticker"]) &
                         (existing["side"] == sig["side"]) &
                         (existing["mail_date_utc"] < mail_ts) &
                         (existing["mail_date_utc"] >= mail_ts - pd.Timedelta(days=REPEAT_WINDOW_DAYS))]
        row = {c: None for c in JOURNAL_COLUMNS}
        row.update({
            "signal_id": sid, "message_id": sig["message_id"], "mail_date_utc": sig["mail_date_utc"],
            "recorded_at_utc": now_iso, "publication": sig["publication"], "domain": sig["domain"],
            "author": sig.get("author") or None, "company": sig["company"], "ticker": sig["ticker"],
            "name": sig["name"], "exchange": sig.get("exchange"), "currency": sig.get("currency"),
            "side": sig["side"], "citation": sig["citation"][:300], "backfill": bool(sig.get("backfill")),
            "repeat": len(prior) > 0, "benchmark": BENCHMARKS.get(sig.get("currency"), DEFAULT_BENCHMARK),
            "eval_done": False,
        })
        added_rows.append(row)
        added.append(sig)
        existing = pd.concat([existing, pd.DataFrame([{"publication": sig["publication"], "ticker": sig["ticker"],
                                                        "side": sig["side"], "mail_date_utc": mail_ts}])],
                             ignore_index=True)
    if added_rows:
        new = pd.DataFrame(added_rows, columns=JOURNAL_COLUMNS)
        journal = pd.concat([journal, new], ignore_index=True) if len(journal) else new
    return journal, added


def _download_closes(tickers, start: str) -> dict:
    """Adjusted daily closes per ticker (dict ticker -> Series indexed by tz-naive date). Batched;
    a ticker that fails to download is simply absent (evaluated on a later run)."""
    out = {}
    tickers = sorted({t for t in tickers if isinstance(t, str) and t})
    for i in range(0, len(tickers), 80):
        chunk = tickers[i:i + 80]
        try:
            data = yf.download(chunk, start=start, auto_adjust=True, progress=False, threads=True,
                               group_by="ticker")
        except Exception as e:
            print(f"  echec telechargement historique ({len(chunk)} tickers): {e}", file=sys.stderr)
            continue
        if data is None or data.empty:
            continue
        for t in chunk:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if t not in data.columns.get_level_values(0):
                        continue
                    s = data[t]["Close"]
                else:
                    s = data["Close"]
                s = s.dropna()
                if len(s):
                    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
                    out[t] = s
            except Exception:
                continue
    return out


def _anchor_position(closes: pd.Series, mail_ts: pd.Timestamp, currency, today: pd.Timestamp):
    """Index (in `closes`) of the first session close that is unambiguously AFTER the email
    arrived -- see CLOSE_CUTOFF_UTC. Only completed sessions (strictly before today, UTC) are
    used, so a run during market hours never anchors on a still-moving intraday bar."""
    mail_day = mail_ts.tz_convert("UTC").tz_localize(None).normalize()
    strict = mail_ts.tz_convert("UTC").hour >= CLOSE_CUTOFF_UTC.get(currency, 20)
    eligible = closes.index > mail_day if strict else closes.index >= mail_day
    eligible &= closes.index < today
    positions = np.flatnonzero(eligible)
    return int(positions[0]) if len(positions) else None


def evaluate_journal(journal: pd.DataFrame, today_iso: str) -> pd.DataFrame:
    """Fills ret_h / ex_h for every horizon that has become measurable since the last run.
    ex_h is SIGNED by the tip's side: positive = the newsletter was right (a "baissier" tip whose
    stock underperforms its benchmark scores positive)."""
    if journal.empty:
        return journal
    today = pd.Timestamp(today_iso).normalize()
    mail_ts = pd.to_datetime(journal["mail_date_utc"], utc=True, errors="coerce")
    done = _flag(journal["eval_done"])
    pending_idx = []
    for idx in journal.index[~done]:
        ts = mail_ts.loc[idx]
        if pd.isna(ts):
            journal.at[idx, "eval_done"] = True
            continue
        age_days = (today - ts.tz_localize(None).normalize()).days
        if age_days > GIVE_UP_EVAL_DAYS:
            journal.at[idx, "eval_done"] = True
            continue
        missing = [h for h in HORIZONS if pd.isna(journal.at[idx, f"ex_{h}"])]
        if not missing:
            journal.at[idx, "eval_done"] = True
            continue
        if age_days >= math.ceil(min(missing) * 7 / 5) + 1:  # earliest pending horizon plausibly due
            pending_idx.append(idx)
    if not pending_idx:
        return journal

    start = (mail_ts.loc[pending_idx].min().tz_localize(None).normalize() - pd.Timedelta(days=7)).date().isoformat()
    tickers = set(journal.loc[pending_idx, "ticker"]) | set(journal.loc[pending_idx, "benchmark"].dropna())
    closes = _download_closes(tickers, start)
    n_new = 0
    for idx in pending_idx:
        row = journal.loc[idx]
        series, bench = closes.get(row["ticker"]), closes.get(row["benchmark"])
        if series is None or bench is None:
            continue
        pos0 = _anchor_position(series, mail_ts.loc[idx], row["currency"], today)
        if pos0 is None:
            continue
        p0, d0 = float(series.iloc[pos0]), series.index[pos0]
        b0 = bench.asof(d0)
        if pd.isna(b0) or p0 <= 0:
            continue
        journal.at[idx, "anchor_date"] = d0.date().isoformat()
        journal.at[idx, "anchor_price"] = p0
        journal.at[idx, "bench_anchor"] = float(b0)
        sign = 1.0 if row["side"] == "long" else -1.0
        for h in HORIZONS:
            if pd.notna(journal.at[idx, f"ex_{h}"]) or pos0 + h >= len(series):
                continue
            dh = series.index[pos0 + h]
            if dh >= today:
                continue
            bh = bench.asof(dh)
            if pd.isna(bh):
                continue
            ret = float(series.iloc[pos0 + h]) / p0 - 1
            bret = float(bh) / float(b0) - 1
            journal.at[idx, f"ret_{h}"] = ret
            journal.at[idx, f"ex_{h}"] = sign * (ret - bret)
            n_new += 1
        if all(pd.notna(journal.at[idx, f"ex_{h}"]) for h in HORIZONS):
            journal.at[idx, "eval_done"] = True
    if n_new:
        print(f"  journal : {n_new} evaluation(s) a horizon fixe ajoutee(s).")
    return journal


def _pooled_sigma(scored: pd.DataFrame) -> float:
    x = scored[f"ex_{PRIMARY_HORIZON}"].dropna()
    sigma = float(x.std()) if len(x) >= 30 else DEFAULT_SIGMA
    return max(sigma, MIN_POOLED_SIGMA)


def _status(n: int, t: float, previous: str | None) -> str:
    if n < N_MIN or pd.isna(t):
        return "observation"
    if t >= T_PROMOTE or (previous == "fiable" and t >= T_DEMOTE):
        return "fiable"
    return "bruit"


# Per-site performance columns, one block per horizon (2026-10-04, user's request: "affiche-moi la
# performance de chaque site"). For each horizon h (trading days after the email):
#   n_evalues_{h}j      tips of that site already measurable at h
#   excess_moy_{h}j     mean return in the advised direction MINUS the benchmark's (a "baissier"
#                       tip whose stock falls more than the market scores positive)
#   rendement_brut_{h}j mean return in the advised direction, market not subtracted
#   taux_reussite_{h}j  share of tips that beat the market in the advised direction
SCORE_COLUMNS = ["statut", "n_signaux", "n_achat", "n_vente"] + [
    f"{c}_{h}j" for h in HORIZONS for c in ("n_evalues", "excess_moy", "rendement_brut", "taux_reussite")
] + ["edge_estime", "t_stat"]


def score_sources(journal: pd.DataFrame, previous_status: dict) -> tuple:
    """Returns (scores DataFrame indexed by publication, crowd dict). Status / edge / t-stat use
    PRIMARY_HORIZON only -- see module docstring; the other horizons are informational."""
    if journal.empty:
        crowd = {"edge": CROWD_PRIOR_EDGE, "n": 0, "t": float("nan"), "status": "observation", "mean": float("nan")}
        return pd.DataFrame(columns=SCORE_COLUMNS), crowd
    scored = journal[~_flag(journal["repeat"])]
    sigma = _pooled_sigma(scored)
    rows = {}
    for pub, g in scored.groupby("publication"):
        sign = g["side"].map({"long": 1.0, "short": -1.0})
        row = {}
        for h in HORIZONS:
            ex = pd.to_numeric(g[f"ex_{h}"], errors="coerce")
            raw = pd.to_numeric(g[f"ret_{h}"], errors="coerce") * sign
            ok = ex.notna()
            row[f"n_evalues_{h}j"] = int(ok.sum())
            row[f"excess_moy_{h}j"] = float(ex[ok].mean()) if ok.any() else float("nan")
            row[f"rendement_brut_{h}j"] = float(raw[ok].mean()) if ok.any() else float("nan")
            row[f"taux_reussite_{h}j"] = float((ex[ok] > 0).mean()) if ok.any() else float("nan")
        x = pd.to_numeric(g[f"ex_{PRIMARY_HORIZON}"], errors="coerce").dropna()
        n = len(x)
        mean = float(x.mean()) if n else float("nan")
        t = mean * math.sqrt(n) / sigma if n else float("nan")
        all_rows = journal[journal["publication"] == pub]
        row.update({
            "statut": _status(n, t, previous_status.get(pub)),
            "n_signaux": int(len(all_rows)),
            "n_achat": int((all_rows["side"] == "long").sum()),
            "n_vente": int((all_rows["side"] == "short").sum()),
            "edge_estime": float(x.sum()) / (n + K_SHRINK) if n else 0.0,
            "t_stat": t,
        })
        rows[pub] = row
    scores = pd.DataFrame.from_dict(rows, orient="index", columns=SCORE_COLUMNS)
    status_map = scores["statut"].to_dict()
    crowd = _crowd_stats(scored, status_map, sigma, previous_status.get(CROWD_PUBLICATION))
    return scores, crowd


def _crowd_stats(scored: pd.DataFrame, status_map: dict, sigma: float, previous: str | None) -> dict:
    """The crowd's own track record -- see module docstring. One event per (ticker, side) per
    window: the tip that brings a ticker to CROWD_MIN_VOTERS distinct agreeing non-reliable
    newsletters, evaluated from its own anchor (= the moment the consensus formed, i.e. when the
    real strategy would have acted on it)."""
    nonrel = scored[scored["publication"].map(lambda p: status_map.get(p) != "fiable")].copy()
    nonrel["ts"] = pd.to_datetime(nonrel["mail_date_utc"], utc=True, errors="coerce")
    nonrel = nonrel.dropna(subset=["ts"]).sort_values("ts")
    events = []
    window = pd.Timedelta(days=CONSENSUS_WINDOW_DAYS)
    for (ticker, side), g in nonrel.groupby(["ticker", "side"]):
        last_event_ts = None
        for i, (idx, r) in enumerate(g.iterrows()):
            if last_event_ts is not None and r["ts"] - last_event_ts < window:
                continue
            before = g.iloc[:i]
            voters = set(before.loc[before["ts"] >= r["ts"] - window, "publication"]) - {r["publication"]}
            if len(voters) + 1 >= CROWD_MIN_VOTERS:
                last_event_ts = r["ts"]
                if pd.notna(r[f"ex_{PRIMARY_HORIZON}"]):
                    events.append(float(r[f"ex_{PRIMARY_HORIZON}"]))
    n = len(events)
    mean = float(np.mean(events)) if n else float("nan")
    t = mean * math.sqrt(n) / sigma if n else float("nan")
    edge = (sum(events) + K_SHRINK * CROWD_PRIOR_EDGE) / (n + K_SHRINK)
    return {"edge": edge, "n": n, "t": t, "status": _status(n, t, previous), "mean": mean}


def recent_votes(journal: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    """Every journal row still inside the consensus window, with its age -- repeats included
    (a newsletter repeating its call keeps its vote fresh, it just doesn't count twice in its own
    score)."""
    if journal.empty:
        return journal.assign(ts=pd.Series(dtype="datetime64[ns, UTC]"), age_days=pd.Series(dtype=float))
    j = journal.copy()
    j["ts"] = pd.to_datetime(j["mail_date_utc"], utc=True, errors="coerce")
    j = j[(j["ts"] >= now - pd.Timedelta(days=CONSENSUS_WINDOW_DAYS)) & (j["ts"] <= now)]
    j["age_days"] = (now - j["ts"]).dt.total_seconds() / 86400
    return j


def _diminishing_sum(values: list[float]) -> float:
    """Second reliable newsletter agreeing adds half its edge, third a quarter... -- newsletters
    often relay the same news/analyst note, so N agreeing votes are not N independent bets."""
    return sum(v * 0.5 ** i for i, v in enumerate(sorted(values, reverse=True)))


def consensus_for(votes: pd.DataFrame, ticker: str, scores: pd.DataFrame, crowd: dict) -> dict:
    """Expected J+20 excess return ("alpha") of being LONG `ticker` right now, from the votes in
    the consensus window (latest vote per newsletter, linearly decayed to 0 over the window):
      alpha_fiable = diminishing sum of reliable long votes' edges - same for reliable short votes
      foule        = (n_long - n_short) / (n_long + n_short + 1) over non-reliable newsletters
      alpha        = alpha_fiable + crowd_edge * foule        if any reliable newsletter voted
                   = crowd_edge * foule                       if none did, the crowd is itself
                                                              "fiable" and >= CROWD_MIN_VOTERS agree
                   = 0                                        otherwise
    A crowd that disagrees with a reliable newsletter lowers its alpha (or raises it, if the
    crowd's measured edge is negative -- a contrarian crowd). Bad newsletters never create an
    alpha on their own unless their CONSENSUS has itself proven reliable."""
    v = votes[votes["ticker"] == ticker].sort_values("ts")
    status = scores["statut"].to_dict() if len(scores) else {}
    edges = scores["edge_estime"].to_dict() if len(scores) else {}
    rel_long, rel_short, sources_long = [], [], []
    crowd_long = crowd_short = 0.0
    n_crowd_long = n_crowd_short = 0
    last_rel_long_ts = None
    for pub, g in v.groupby("publication"):
        last = g.iloc[-1]
        decay = max(0.0, 1 - float(last["age_days"]) / CONSENSUS_WINDOW_DAYS)
        if status.get(pub) == "fiable":
            e = max(float(edges.get(pub, 0.0)), 0.0) * decay
            if last["side"] == "long":
                rel_long.append(e)
                sources_long.append((e, pub))
                last_rel_long_ts = max(last_rel_long_ts, last["ts"]) if last_rel_long_ts is not None else last["ts"]
            else:
                rel_short.append(e)
        elif last["side"] == "long":
            crowd_long += decay
            n_crowd_long += 1
        else:
            crowd_short += decay
            n_crowd_short += 1
    alpha_fiable = _diminishing_sum(rel_long) - _diminishing_sum(rel_short)
    balance = (crowd_long - crowd_short) / (crowd_long + crowd_short + 1)
    crowd_term = crowd["edge"] * balance
    if rel_long or rel_short:
        alpha = alpha_fiable + crowd_term
    elif crowd["status"] == "fiable" and max(n_crowd_long, n_crowd_short) >= CROWD_MIN_VOTERS:
        alpha = crowd_term
    else:
        alpha = 0.0
    lead = max(sources_long)[1] if sources_long else (CROWD_PUBLICATION if alpha > 0 else None)
    return {
        "ticker": ticker, "alpha": alpha, "alpha_fiable": alpha_fiable, "foule_terme": crowd_term,
        "foule_equilibre": balance, "n_fiables_long": len(rel_long), "n_fiables_short": len(rel_short),
        "n_foule_long": n_crowd_long, "n_foule_short": n_crowd_short, "newsletter_principale": lead,
        "newsletters_fiables_long": ";".join(p for _, p in sorted(sources_long, reverse=True)),
        "dernier_vote_fiable_long": last_rel_long_ts.isoformat() if last_rel_long_ts is not None else None,
    }


def build_scorecard(scores: pd.DataFrame, crowd: dict) -> pd.DataFrame:
    """Per-site performance + status, one row per newsletter, plus one row for the crowd's
    consensus (its own J+20 track record -- see module docstring). Reliable sites first, then by
    t-stat."""
    out = scores.copy()
    pk = f"{PRIMARY_HORIZON}j"
    out.loc[CROWD_PUBLICATION, ["statut", f"n_evalues_{pk}", f"excess_moy_{pk}", "edge_estime", "t_stat"]] = [
        crowd["status"], crowd["n"], crowd["mean"], crowd["edge"], crowd["t"]]
    out.index.name = "source"
    out = out.reset_index()
    out["statut"] = out["statut"].fillna("observation")
    for c in ["n_signaux", "n_achat", "n_vente"] + [f"n_evalues_{h}j" for h in HORIZONS]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    order = {"fiable": 0, "observation": 1, "bruit": 2}
    out["_o"] = out["statut"].map(order).fillna(3)
    out = out.sort_values(["_o", "t_stat", "n_signaux"], ascending=[True, False, False],
                          na_position="last").drop(columns="_o")
    return out[["source"] + SCORE_COLUMNS]


def save_status(scores: pd.DataFrame, crowd: dict) -> dict:
    status = scores["statut"].to_dict() if len(scores) else {}
    status[CROWD_PUBLICATION] = crowd["status"]
    return status


if __name__ == "__main__":
    j = load_journal()
    s, c = score_sources(j, {})
    print(s.to_string())
    print(json.dumps(c, default=str))
