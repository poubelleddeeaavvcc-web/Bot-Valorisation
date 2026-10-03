"""Bot #33 Courrier -- STRATEGIE REELLE (capital limite), added 2026-10-03 at the user's explicit
request: "les newsletters mauvaises, que tu entendes leur bruit mais dans la strategie reelle elles
ne soient pas engagees [...] j'aurais un capital limite, je ne peux pas acheter toutes les bonnes
newsletters, je te laisse voir pour faire ca de maniere intelligente".

Still a SIMULATION (no broker is connected yet), but one built to the constraints of real money,
unlike the uncapped lab ledger (mail_signal_bot.py), which keeps testing every tip for the
scorecard:
  - Own capital (CONFIG "capital_eur"), own ledger, own cash. Long only: shorting needs a margin
    account, borrow fees and is often unavailable on small caps -- shorts stay in the lab.
  - Only listings a French retail account can actually trade (REAL_EXCHANGES: US + European
    primary exchanges; the journal already excludes ETFs, funds and OTC listings).
  - A real fee on BOTH orders (CONFIG "frais_par_ordre_eur"), so a position is never smaller than
    "taille_min_position_eur" (at 1 EUR/order, 150 EUR = 1.3% round trip).

WHO CAN TRIGGER A POSITION
--------------------------
Only a newsletter with "fiable" status (see mail_signal_scoring.py) -- or the crowd's consensus,
if that consensus has itself earned "fiable" status. "Bruit"/"observation" newsletters are heard,
never followed: their votes move a candidate's expected return up or down through the crowd term
of consensus_for(), nothing more. As long as no newsletter has proven itself, this strategy simply
stays in cash -- by design.

ALLOCATION ("de maniere intelligente")
--------------------------------------
Every ticker with votes in the consensus window gets an expected J+20 excess return ("alpha")
from consensus_for(): the measured edge of the reliable newsletters recommending it (decayed with
the age of their call), plus or minus the crowd's measured edge times its vote balance. Then:
  1. Candidates are ranked by alpha, best first -- capital goes to the best-supported ideas, not
     to whichever newsletter happened to write first.
  2. Size grows with alpha: taille_max_position_pct of equity at alpha >= alpha_plein, linearly
     less below, never under taille_min_position_eur.
  3. An idea must pay for its own fees: alpha >= 2 * fee / size + marge_alpha.
  4. No newsletter may carry more than max_par_newsletter_pct of equity (positions are attributed
     to the reliable newsletter with the highest edge behind them) -- one newsletter's bad streak
     can't sink the whole account.
  5. Capital full: a new idea only replaces the weakest held position (lowest current alpha, held
     >= detention_min_jours) if it beats it by marge_rotation plus the 2 extra fees of the swap --
     otherwise fees would eat the switch.
Exits: same stop-loss / ratchet / take-profit constants as the lab (on the local price, like a
broker stop), "these_inversee" when reliable newsletters now net against the position (alpha < 0),
"horizon_atteint" detention_max_jours after the LAST reliable call on the ticker (a newsletter
repeating its call extends it), "rotation" when sold for a better idea.

Execution prices are the price at run time (same as the lab) -- the 2026-10-03 analysis measured
that this differs from the next session's open by -0.07% on average, i.e. no material bias.
"""
import json
import math
import pathlib
import sys

import warnings

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
# appending a row whose not-yet-filled columns are all None -- intended, and harmless under pandas<3
warnings.filterwarnings("ignore", category=FutureWarning, message=".*concatenation with empty or all-NA.*")
sys.path.insert(0, str(HERE))

from screener.simulate_portfolio import (  # noqa: E402
    STOP_LOSS_PCT, RATCHET_STEP_PCT, RATCHET_GIVEBACK_PCT, reconcile_fresh_price,
)
from screener.simulate_constrained_portfolio import to_eur, fractional_eligible  # noqa: E402
from screener.mail_signal_scoring import consensus_for, recent_votes, CROWD_PUBLICATION  # noqa: E402

CONFIG_PATH = HERE / "screener/mail_signal_real_config.json"
LEDGER_PATH = HERE / "results/simulation/mail_signal_real_ledger.csv"
STATE_PATH = HERE / "results/simulation/mail_signal_real_state.json"
SUMMARY_PATH = HERE / "results/simulation/mail_signal_real_summary.json"
EQUITY_CURVE_PATH = HERE / "results/simulation/mail_signal_real_equity_curve.csv"
CANDIDATES_PATH = HERE / "results/simulation/mail_signal_real_candidates.csv"

TAKE_PROFIT_PCT = 0.30  # same as the lab -- see mail_signal_bot.py

DEFAULT_CONFIG = {
    "capital_eur": 1000.0,
    "frais_par_ordre_eur": 1.0,
    "taille_min_position_eur": 150.0,
    "taille_max_position_pct": 0.30,
    "max_par_newsletter_pct": 0.50,
    "alpha_plein": 0.04,
    "marge_alpha": 0.005,
    "marge_rotation": 0.01,
    "detention_min_jours": 3,
    "detention_max_jours": 30,
}

REAL_EXCHANGES = {
    "NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "NYSE", "NASDAQ", "AMEX",           # US
    "PAR", "GER", "FRA", "AMS", "MIL", "MCE", "BRU", "LIS", "VIE", "SWX", "EBS",         # Europe
    "STO", "CPH", "HEL", "OSL", "ISE", "LSE",
}

COLUMNS = [
    "ticker", "name", "newsletter_principale", "newsletters_fiables", "status", "currency", "exchange",
    "fractional", "entry_date", "entry_price", "shares", "entry_value_eur", "entry_alpha",
    "last_signal_date", "last_check_date", "last_price", "current_value_eur", "unrealized_return_pct",
    "current_alpha", "peak_unrealized_return_pct", "peak_date",
    "exit_date", "exit_price", "exit_reason", "exit_value_eur", "fees_eur", "return_pct", "holding_days",
]


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            user = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in user.items() if k in DEFAULT_CONFIG})
        except (json.JSONDecodeError, OSError) as e:
            print(f"  config strategie reelle illisible ({e}) -- valeurs par defaut", file=sys.stderr)
    return cfg


def load_ledger() -> pd.DataFrame:
    if LEDGER_PATH.exists():
        df = pd.read_csv(LEDGER_PATH)
        for c in COLUMNS:
            if c not in df.columns:
                df[c] = None
        df = df[COLUMNS]
    else:
        df = pd.DataFrame(columns=COLUMNS)
    for c in ("ticker", "name", "newsletter_principale", "newsletters_fiables", "status", "currency", "exchange",
              "entry_date", "last_signal_date", "last_check_date", "peak_date", "exit_date", "exit_reason"):
        df[c] = df[c].astype(object)
    return df


def _load_state(cfg: dict) -> dict:
    state = {"cash_eur": cfg["capital_eur"], "capital_initial_eur": cfg["capital_eur"], "apports_eur": 0.0}
    if STATE_PATH.exists():
        try:
            state.update(json.loads(STATE_PATH.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    # capital_eur changed in the config since last run = a deposit (or withdrawal) of the
    # difference, not a reset -- positions already open keep their history
    delta = float(cfg["capital_eur"]) - float(state["capital_initial_eur"]) - float(state.get("apports_eur", 0.0))
    if abs(delta) > 1e-9:
        state["cash_eur"] = float(state["cash_eur"]) + delta
        state["apports_eur"] = float(state.get("apports_eur", 0.0)) + delta
        print(f"  strategie reelle : capital configure modifie, {delta:+.2f} EUR ajoutes au cash")
    return state


def _json_default(o):
    """numpy scalars (pandas sums/means) -> plain Python numbers; NaN -> null."""
    try:
        v = float(o)
    except (TypeError, ValueError):
        return str(o)
    return None if math.isnan(v) else v


def _close(ledger, idx, today, price, value_eur, fee, reason):
    entry_value = float(ledger.at[idx, "entry_value_eur"])
    fees = float(ledger.at[idx, "fees_eur"] or 0) + fee
    ledger.at[idx, "status"] = "closed"
    ledger.at[idx, "exit_date"] = today
    ledger.at[idx, "exit_price"] = price
    ledger.at[idx, "exit_reason"] = reason
    ledger.at[idx, "exit_value_eur"] = value_eur - fee
    ledger.at[idx, "fees_eur"] = fees
    ledger.at[idx, "return_pct"] = (value_eur - entry_value - fees) / entry_value
    ledger.at[idx, "holding_days"] = (pd.Timestamp(today) - pd.Timestamp(ledger.at[idx, "entry_date"])).days
    print(f"  REEL CLOTURE {ledger.at[idx, 'ticker']} : {reason}, retour net {ledger.at[idx, 'return_pct']:+.1%}")
    return value_eur - fee


def run_real_layer(journal, scores, crowd, now: pd.Timestamp, fx_rates: dict, fetch_price, ensure_fx) -> dict:
    """See module docstring. `fetch_price(ticker) -> {"price", "currency"} | None` and
    `ensure_fx(currency, fx_rates)` are passed in by mail_signal_bot.py (single owner of the
    Yahoo plumbing)."""
    cfg = load_config()
    fee = float(cfg["frais_par_ordre_eur"])
    min_pos = float(cfg["taille_min_position_eur"])
    today = now.date().isoformat()
    ledger = load_ledger()
    state = _load_state(cfg)
    cash = float(state["cash_eur"])
    votes = recent_votes(journal, now)
    n_fiables = int((scores["statut"] == "fiable").sum()) if len(scores) else 0

    # 1. mark to market + exits
    for idx in ledger.index[ledger["status"] == "open"]:
        ticker = ledger.at[idx, "ticker"]
        px = fetch_price(ticker)
        if px is None:
            continue
        check, factor = reconcile_fresh_price(ticker, px["price"], ledger.at[idx, "last_price"],
                                              ledger.at[idx, "last_check_date"])
        if check == "suspect":
            continue
        if check == "split":
            ledger.at[idx, "entry_price"] = float(ledger.at[idx, "entry_price"]) / factor
            ledger.at[idx, "shares"] = float(ledger.at[idx, "shares"]) * factor
        currency = ledger.at[idx, "currency"]
        ensure_fx(currency, fx_rates)
        price_eur = to_eur(px["price"], currency, fx_rates)
        if price_eur is None:
            continue
        value = float(ledger.at[idx, "shares"]) * price_eur
        unrealized = px["price"] / float(ledger.at[idx, "entry_price"]) - 1
        cons = consensus_for(votes, ticker, scores, crowd)
        ledger.at[idx, "last_check_date"] = today
        ledger.at[idx, "last_price"] = px["price"]
        ledger.at[idx, "current_value_eur"] = value
        ledger.at[idx, "unrealized_return_pct"] = unrealized
        ledger.at[idx, "current_alpha"] = cons["alpha"]
        if cons["dernier_vote_fiable_long"]:
            new_sig = pd.Timestamp(cons["dernier_vote_fiable_long"]).date().isoformat()
            if pd.isna(ledger.at[idx, "last_signal_date"]) or new_sig > str(ledger.at[idx, "last_signal_date"]):
                ledger.at[idx, "last_signal_date"] = new_sig
        peak = ledger.at[idx, "peak_unrealized_return_pct"]
        if pd.isna(peak) or unrealized > float(peak):
            ledger.at[idx, "peak_unrealized_return_pct"] = unrealized
            ledger.at[idx, "peak_date"] = today
        peak = float(ledger.at[idx, "peak_unrealized_return_pct"])
        milestone = int(peak // RATCHET_STEP_PCT)
        age_signal = (pd.Timestamp(today) - pd.Timestamp(ledger.at[idx, "last_signal_date"])).days
        reason = None
        if milestone >= 1 and unrealized <= milestone * RATCHET_STEP_PCT - RATCHET_GIVEBACK_PCT:
            reason = "trailing_stop"
        elif unrealized <= STOP_LOSS_PCT:
            reason = "stop_loss"
        elif unrealized >= TAKE_PROFIT_PCT:
            reason = "take_profit"
        elif cons["alpha"] < 0:
            reason = "these_inversee"
        elif age_signal >= cfg["detention_max_jours"]:
            reason = "horizon_atteint"
        if reason:
            cash += _close(ledger, idx, today, px["price"], value, fee, reason)

    # 2. candidates, best expected excess return first
    open_mask = ledger["status"] == "open"
    held = set(ledger.loc[open_mask, "ticker"])
    latest = (journal.sort_values("mail_date_utc").groupby("ticker").last()
              if len(journal) else pd.DataFrame(columns=["name", "exchange", "currency"]))
    rows = []
    for ticker in sorted(set(votes["ticker"]) if len(votes) else set()):
        cons = consensus_for(votes, ticker, scores, crowd)
        info = latest.loc[ticker] if ticker in latest.index else {}
        cons.update({"name": info.get("name"), "exchange": info.get("exchange"), "currency": info.get("currency")})
        rows.append(cons)
    cands = pd.DataFrame(rows)
    decisions = {}
    if len(cands):
        cands = cands.sort_values("alpha", ascending=False).reset_index(drop=True)
        for i, c in cands.iterrows():
            ticker, alpha = c["ticker"], float(c["alpha"])
            if ticker in held:
                decisions[i] = ("detenu", None)
                continue
            if alpha <= 0:
                decisions[i] = ("pas d'avis fiable a l'achat", None)
                continue
            if c["exchange"] not in REAL_EXCHANGES:
                decisions[i] = (f"place non accessible ({c['exchange']})", None)
                continue
            open_mask = ledger["status"] == "open"
            equity = cash + pd.to_numeric(ledger.loc[open_mask, "current_value_eur"], errors="coerce").fillna(0).sum()
            target = max(equity * cfg["taille_max_position_pct"] * min(1.0, alpha / cfg["alpha_plein"]), min_pos)
            lead = c["newsletter_principale"]
            exposure = pd.to_numeric(ledger.loc[open_mask & (ledger["newsletter_principale"] == lead),
                                                "current_value_eur"], errors="coerce").fillna(0).sum()
            target = min(target, cfg["max_par_newsletter_pct"] * equity - exposure)
            if target < min_pos:
                decisions[i] = (f"plafond atteint pour {lead}", None)
                continue
            if alpha < 2 * fee / target + cfg["marge_alpha"]:
                decisions[i] = ("rendement espere trop faible face aux frais", target)
                continue
            if cash < target + fee:
                if cash >= min_pos + fee:
                    target = cash - fee
                else:
                    held_rows = ledger[open_mask].copy()
                    held_rows["age"] = (pd.Timestamp(today) - pd.to_datetime(held_rows["entry_date"])).dt.days
                    held_rows = held_rows[held_rows["age"] >= cfg["detention_min_jours"]]
                    if held_rows.empty:
                        decisions[i] = ("capital plein (positions trop recentes pour etre remplacees)", target)
                        continue
                    weakest = pd.to_numeric(held_rows["current_alpha"], errors="coerce").fillna(0).idxmin()
                    weak_alpha = float(pd.to_numeric(pd.Series([ledger.at[weakest, "current_alpha"]]),
                                                     errors="coerce").fillna(0).iloc[0])
                    if alpha - weak_alpha < cfg["marge_rotation"] + 2 * fee / target:
                        decisions[i] = (f"capital plein ({ledger.at[weakest, 'ticker']} reste meilleur apres frais)", target)
                        continue
                    weak_value = float(ledger.at[weakest, "current_value_eur"])
                    cash += _close(ledger, weakest, today, ledger.at[weakest, "last_price"], weak_value, fee, "rotation")
                    held.discard(ledger.at[weakest, "ticker"])
                    target = min(target, cash - fee)
                    if target < min_pos:
                        decisions[i] = ("capital insuffisant apres rotation", target)
                        continue
            px = fetch_price(ticker)
            if px is None:
                decisions[i] = ("prix indisponible", target)
                continue
            currency = px["currency"] or c["currency"]
            ensure_fx(currency, fx_rates)
            price_eur = to_eur(px["price"], currency, fx_rates)
            if price_eur is None or price_eur <= 0:
                decisions[i] = ("taux de change indisponible", target)
                continue
            fractional = fractional_eligible(ticker, None, None)
            if fractional:
                shares = target / price_eur
            else:
                shares = math.floor(target / price_eur)
                if shares == 0 and price_eur <= min(1.25 * target, cash - fee):
                    shares = 1
                if shares == 0:
                    decisions[i] = ("action trop chere pour la taille de position", target)
                    continue
            cost = shares * price_eur
            if cost < 0.8 * min_pos or cost + fee > cash:
                decisions[i] = ("taille finale hors limites", target)
                continue
            new_row = {col: None for col in COLUMNS}
            new_row.update({
                "ticker": ticker, "name": c["name"] or ticker, "newsletter_principale": lead,
                "newsletters_fiables": c["newsletters_fiables_long"], "status": "open", "currency": currency,
                "exchange": c["exchange"], "fractional": bool(fractional), "entry_date": today,
                "entry_price": px["price"], "shares": shares, "entry_value_eur": cost, "entry_alpha": alpha,
                "last_signal_date": (pd.Timestamp(c["dernier_vote_fiable_long"]).date().isoformat()
                                     if c["dernier_vote_fiable_long"] else today),
                "last_check_date": today, "last_price": px["price"], "current_value_eur": cost,
                "unrealized_return_pct": 0.0, "current_alpha": alpha, "peak_unrealized_return_pct": 0.0,
                "peak_date": today, "fees_eur": fee,
            })
            new = pd.DataFrame([new_row], columns=COLUMNS)
            ledger = pd.concat([ledger, new], ignore_index=True) if len(ledger) else new
            cash -= cost + fee
            held.add(ticker)
            decisions[i] = ("achete", cost)
            print(f"  REEL ACHAT {ticker} ({lead}) : {cost:.2f} EUR, rendement espere {alpha:+.1%} a 20j")
        cands["decision"] = [decisions.get(i, ("", None))[0] for i in cands.index]
        cands["taille_cible_eur"] = [decisions.get(i, ("", None))[1] for i in cands.index]

    # 3. outputs
    open_pos = ledger[ledger["status"] == "open"]
    closed = ledger[ledger["status"] == "closed"]
    invested = pd.to_numeric(open_pos["current_value_eur"], errors="coerce").fillna(0).sum()
    equity = cash + invested
    capital = float(state["capital_initial_eur"]) + float(state.get("apports_eur", 0.0))
    exposure = (open_pos.groupby("newsletter_principale")["current_value_eur"]
                .apply(lambda s: float(pd.to_numeric(s, errors="coerce").fillna(0).sum())).to_dict())
    if n_fiables == 0 and crowd["status"] != "fiable":
        statut = ("en attente : aucune newsletter n'est encore fiable (il faut au moins 15 avis evalues a "
                  "20 jours et un t-stat >= 1.5) -- le capital reste en cash")
    elif open_pos.empty:
        statut = "pret : newsletters fiables identifiees, en attente d'un avis assez fort pour acheter"
    else:
        statut = "investi"
    summary = {
        "capital_eur": capital, "cash_eur": cash, "investi_eur": invested, "total_equity_eur": equity,
        "total_return_pct": equity / capital - 1 if capital else None,
        "nb_open": int(len(open_pos)), "nb_closed": int(len(closed)),
        "frais_payes_eur": float(pd.to_numeric(ledger["fees_eur"], errors="coerce").fillna(0).sum()),
        "win_rate_closed": float((closed["return_pct"] > 0).mean()) if len(closed) else None,
        "avg_return_closed": float(closed["return_pct"].mean()) if len(closed) else None,
        "exposition_par_newsletter_eur": exposure, "nb_newsletters_fiables": n_fiables,
        "foule": {"statut": crowd["status"], "edge_estime": crowd["edge"], "n_evenements": crowd["n"],
                  "t_stat": None if pd.isna(crowd["t"]) else crowd["t"]},
        "statut": statut, "config": cfg, "derniere_maj_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    state["cash_eur"] = float(cash)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    cand_cols = ["ticker", "name", "alpha", "alpha_fiable", "foule_terme", "n_fiables_long", "n_fiables_short",
                 "n_foule_long", "n_foule_short", "newsletter_principale", "newsletters_fiables_long",
                 "decision", "taille_cible_eur"]
    (cands.reindex(columns=cand_cols) if len(cands) else pd.DataFrame(columns=cand_cols)).to_csv(CANDIDATES_PATH, index=False)
    point = {"timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "cash_eur": cash, "total_equity_eur": equity,
             "n_open": len(open_pos), "n_closed": len(closed)}
    pd.DataFrame([point]).to_csv(EQUITY_CURVE_PATH, mode="a", header=not EQUITY_CURVE_PATH.exists(), index=False)
    print(f"\n=== Strategie reelle Courrier : {len(open_pos)} position(s), {cash:.2f} EUR cash, "
          f"valeur {equity:.2f} EUR ({summary['total_return_pct']:+.1%}) -- {statut} ===")
    return summary


__all__ = ["run_real_layer", "load_config", "CROWD_PUBLICATION"]
