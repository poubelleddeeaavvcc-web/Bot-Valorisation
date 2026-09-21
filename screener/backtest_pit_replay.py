"""Point-in-time replay backtest (Bot #1/#2/#3/#25) using REAL historical valuation/momentum
snapshots mined from this repo's own git history, instead of backtest_momentum_proxy.py's
momentum-only approximation replayed against TODAY's frozen fundamentals.

WHY THIS EXISTS (2026-09-14, user: "est-ce que tu pourrais deja backtester tout sur cette
echantillon ?"): results/screener/full_valuation_latest.csv gets committed to git on every
screener run (roughly hourly, via GitHub Actions). Its git history is therefore an accidental
point-in-time fundamentals archive: 253 snapshots since 2026-08-07, each carrying the EXACT
price/valuation_gap/fair_value/mom_12_2/sector_momentum/quality_multiplier the live bots saw
at that exact moment. This closes the lookahead-bias gap backtest_momentum_proxy.py's
docstring flags repeatedly -- for THIS ~5.5-week window (not the 12-month window that other
script covers), valuation_reached exits and Bot #25's real reinforcement gate
(valuation_gap_now >= entry_valuation_gap, see simulate_delta_portfolio.reinforce_convictions)
can be replayed faithfully instead of approximated or skipped.

CAVEATS (documented, not hidden -- same spirit as backtest_momentum_proxy.py):
  - Window is short: ~5.5 weeks (2026-08-07 -> today), not 12 months -- this repo simply
    hasn't existed longer. Far fewer trades, far less statistical weight than the momentum
    proxy; a mechanism check ("does the logic behave sensibly"), not a performance verdict.
  - The universe grew hugely over the window (364 tickers on 2026-08-07 vs ~3636 today, per
    trending_universe.csv's own history) -- early snapshots see a much smaller candidate
    pool. NOT a bug to fix: it's the honest historical universe at each point, the same one
    the live bots actually saw and could actually have bought.
  - The valuation model itself evolved during the window (industry peer-group blending,
    cyclical P/B check and other refinements landed over these weeks -- see git log on
    select_top_picks.py/value_momentum_quality_screener_v2.py). valuation_gap at an early
    snapshot was computed with whatever model version was live then, not today's -- that is
    fidelity, not noise: it's what the live bot actually used to decide at that moment.
  - `currency` wasn't tracked before 2026-08-21. Backfilled per ticker from the earliest
    snapshot where it IS known (a ticker's listing currency essentially never changes,
    unlike anything price/valuation-related, which is why only this field is backfilled
    this way).
  - Snapshot cadence is ~hourly but not perfectly uniform (a CI job can be late or skipped).
    The exit scan below walks snapshot-to-snapshot in whatever order they actually happened,
    not a fixed calendar grid, so a gap just means a slightly coarser check through it, not
    a wrong one.
  - Whole-share vs. fractional-share rounding, and the exact NA/state-linked cap relaxation
    order: same simplifications as backtest_momentum_proxy.py, not modeled here either.

Reuses _pick_capped/_pick_even_sector from backtest_momentum_proxy.py (candidate picking
under sector/NA caps doesn't care where the score came from) and ticker_region/
NORTH_AMERICA_MAX_SHARE/composite_score from select_top_picks.py -- single authoritative
implementation, not a re-fork, same pattern already used throughout this repo.
"""
import json
import pathlib
import sys
import time

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.backtest_momentum_proxy import (  # noqa: E402
    _pick_capped, _pick_even_sector, _chunked_download, FX_PAIR,
)
from screener.select_top_picks import ticker_region, composite_score  # noqa: E402
from screener.simulate_portfolio import STOP_LOSS_PCT, RATCHET_STEP_PCT, RATCHET_GIVEBACK_PCT  # noqa: E402
from screener.simulate_delta_portfolio import (  # noqa: E402
    DELTA_MIN_DROP_PCT, DELTA_DEEP_DROP_PCT, DELTA_MOM_SPREAD_MIN, DELTA_MAX_POSITION_SHARE,
)

VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
OUT_DIR = HERE / "results/backtest_pit"
SNAP_DIR = OUT_DIR / "snapshots"
SNAP_DIR.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 500.0
DELTA_STARTING_CAPITAL = 300.0
# The first ~2 weeks (2026-08-07 -> ~08-21) are excluded -- found by running this script
# unfiltered first: 197 of 198 Bot#1 "valorisation_atteinte" exits in that first run had
# entry_date in the single week of 2026-08-07..09, median hold time 19 minutes, median
# return_pct exactly 0.0. That's not the model validating a thesis -- it's the valuation
# pipeline still being actively tuned right at project launch (universe 364 tickers vs
# ~3636 today, fewer columns, no industry-peer-group blending yet -- see git log on
# select_top_picks.py from that week), so fair_value swung between consecutive early runs
# for reasons that have nothing to do with price. Re-running from 2026-08-21 onward (when
# the `currency` column first appears, a convenient marker of a more settled schema)
# produced zero further valorisation_atteinte instant-flips -- confirms this was a
# bootstrap artifact, not an ongoing data-quality problem across the whole window.
WINDOW_START = pd.Timestamp("2026-08-21")
MAX_PER_SECTOR = 3
STARTING_SLOTS_B2 = 15
STARTING_SLOTS_B3 = 30
STARTING_SLOTS_DELTA = 9

# Parametres du moteur "slotted" (Bot #2/#3/#25 et toutes les variantes). Les valeurs par defaut
# reproduisent exactement le comportement des bots reels -- un run sans surcharge donne les memes
# resultats qu'avant l'ajout des variantes.
DEFAULT_PARAMS = {
    "stop_loss_pct": STOP_LOSS_PCT,
    "ratchet_step_pct": RATCHET_STEP_PCT,
    "ratchet_giveback_pct": RATCHET_GIVEBACK_PCT,
    "exit_on_momentum_lost": True,
    "exit_on_valuation_reached": True,
    "min_valuation_gap": float("-inf"),      # filtres d'entree supplementaires
    "min_mom_12_2": float("-inf"),
    "min_quality_multiplier": float("-inf"),
    "delta_min_drop_pct": DELTA_MIN_DROP_PCT,
    "delta_deep_drop_pct": DELTA_DEEP_DROP_PCT,
    "delta_mom_spread_min": DELTA_MOM_SPREAD_MIN,
    "delta_max_position_share": DELTA_MAX_POSITION_SHARE,
    # Apport mensuel (EUR) reserve au renfort de conviction : credite au 1er snapshot de chaque nouveau
    # mois calendaire (pas au tout premier : le capital de depart est deja la), jamais utilise pour de
    # nouvelles positions. Reparti a chaque snapshot entre les positions qui passent la porte de
    # conviction, dans la limite du plafond par position ; le reliquat attend la prochaine occasion.
    "monthly_contribution_eur": 0.0,
    "contribution_split": "valuation",       # "valuation" (poids = sous-evaluation actuelle) | "equal"
}

# Bots de reference sur lesquels une variante s'appuie (champ "base" du fichier de config).
BASES = {
    "bot2_constrained": {"slots": STARTING_SLOTS_B2, "max_per_sector": MAX_PER_SECTOR, "mode": "capped",
                          "capital": STARTING_CAPITAL, "reinforce": False},
    "bot3_large": {"slots": STARTING_SLOTS_B3, "max_per_sector": None, "mode": "even_sector",
                    "capital": STARTING_CAPITAL, "reinforce": False},
    "bot25_delta": {"slots": STARTING_SLOTS_DELTA, "max_per_sector": MAX_PER_SECTOR, "mode": "capped",
                     "capital": DELTA_STARTING_CAPITAL, "reinforce": True},
}
VARIANTS_PATH = HERE / "screener/backtest_pit_variants.json"

BOTS = {
    "bot1_blind": "Bot #1 (blind, PIT reel)",
    "bot2_constrained": "Bot #2 (constrained, PIT reel)",
    "bot3_large": "Bot #3 (large, PIT reel)",
    "bot25_delta": "Bot #25 Delta (constrained+conviction REELLE, PIT reel)",
}


# ============ 1. Extraction de l'historique Git (snapshots point-in-time) ============

def list_snapshot_commits() -> list:
    """(commit_hash, timestamp) for every commit that touched full_valuation_latest.csv,
    oldest first -- git log --follow survives the file having been renamed/moved."""
    rel = str(VALUATION_PATH.relative_to(HERE)).replace("\\", "/")
    import subprocess
    out = subprocess.run(
        ["git", "log", "--follow", "--format=%H|%ad", "--date=format:%Y-%m-%dT%H-%M-%S", "--", rel],
        cwd=HERE, capture_output=True, text=True, check=True,
    ).stdout
    rows = [line.split("|") for line in out.strip().splitlines() if line]
    rows.sort(key=lambda r: r[1])  # oldest first
    return rows


def extract_snapshots() -> list:
    """Writes each historical version of full_valuation_latest.csv to SNAP_DIR (cached --
    skipped if already on disk), returns the sorted list of (timestamp, path)."""
    import subprocess
    rel = str(VALUATION_PATH.relative_to(HERE)).replace("\\", "/")
    commits = list_snapshot_commits()
    print(f"Historique Git : {len(commits)} snapshots de {rel} depuis {commits[0][1]}", file=sys.stderr)
    out = []
    for commit_hash, ts in commits:
        dest = SNAP_DIR / f"{ts}.csv"
        if not dest.exists():
            content = subprocess.run(
                ["git", "show", f"{commit_hash}:{rel}"],
                cwd=HERE, capture_output=True, text=True, check=True,
            ).stdout
            dest.write_text(content, encoding="utf-8")
        out.append((ts, dest))
    return out


def load_panel(snapshots: list) -> pd.DataFrame:
    """One long DataFrame, every snapshot stacked, tagged with snapshot_time (parsed from
    the filename, not `fetched_at` -- fetched_at is per-ticker and can straddle the actual
    commit instant by a few minutes across a large universe; the commit time is the single
    instant every row in that snapshot was actually live together)."""
    frames = []
    for ts, path in snapshots:
        try:
            df = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        if not len(df):
            continue
        df["snapshot_time"] = pd.to_datetime(ts, format="%Y-%m-%dT%H-%M-%S")
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    for col in ("valuation_gap", "mom_12_2", "sector_momentum", "quality_multiplier",
                "fair_value", "price", "passes_filter"):
        if col not in panel.columns:
            panel[col] = pd.NA
    # passes_filter can land as real bool in some snapshots and as the string "True"/"False"
    # in others once concatenated across files with slightly different dtypes -- normalize
    # once here rather than risk a silent `== True` mismatch downstream.
    panel["passes_filter"] = panel["passes_filter"].map(
        {True: True, False: False, "True": True, "False": False}).fillna(False)
    return panel.sort_values("snapshot_time")


def last_known_prices(panel: pd.DataFrame, tickers) -> pd.Series:
    """Last non-null price for each of `tickers` anywhere in the panel, regardless of which
    exact snapshot it came from -- fixes a real bug found 2026-09-14: a position still
    genuinely held (never hit an exit condition) can be silently dropped from the final
    "open positions" report if its ticker happens to be missing from the one single LAST
    snapshot (a transient per-run fetch failure), even though it's present in every
    snapshot around it. Requiring exact presence in `last_snap` was wrong; the position was
    never actually closed, just under-reported."""
    sub = panel[panel["ticker"].isin(tickers) & panel["price"].notna()]
    sub = sub.sort_values("snapshot_time").drop_duplicates("ticker", keep="last")
    return sub.set_index("ticker")["price"]


def backfill_currency(panel: pd.DataFrame) -> dict:
    """currency wasn't tracked before 2026-08-21 -- map every ticker to the earliest known
    currency for it across the whole panel (a ticker's listing currency doesn't change)."""
    if "currency" not in panel.columns:
        return {}
    known = panel.dropna(subset=["currency"]).sort_values("snapshot_time")
    return known.drop_duplicates("ticker", keep="first").set_index("ticker")["currency"].to_dict()


# ============ 2. FX historique pour la fenetre (courte, telechargement rapide) ============

def download_fx_window(currencies: set, months: int) -> pd.DataFrame:
    pairs = {ccy: FX_PAIR[ccy] for ccy in currencies if ccy in FX_PAIR and ccy != "GBp"}
    if "GBp" in currencies:
        pairs["GBP"] = FX_PAIR["GBP"]
    closes = _chunked_download(list(set(pairs.values())), period=f"{months}mo", interval="1d")
    ticker_to_ccy = {v: k for k, v in pairs.items()}
    renamed = {ticker_to_ccy[tk]: s for tk, s in closes.items() if tk in ticker_to_ccy}
    fx = pd.DataFrame(renamed)
    fx.index = pd.to_datetime(fx.index).tz_localize(None)
    return fx.sort_index()


def fx_rate_asof(fx_daily: pd.DataFrame, ccy, when: pd.Timestamp):
    if ccy is None or pd.isna(ccy):
        return None
    if ccy == "EUR":
        return 1.0
    key = "GBP" if ccy == "GBp" else ccy
    if fx_daily is None or key not in fx_daily.columns:
        return None
    s = fx_daily[key].loc[:when.normalize() + pd.Timedelta(days=1)].dropna()
    if not len(s):
        return None
    rate = float(s.iloc[-1])
    return rate * 100 if ccy == "GBp" else rate


def to_eur_pit(price, ccy, fx_daily, when):
    rate = fx_rate_asof(fx_daily, ccy, when)
    if rate is None or rate <= 0 or price is None or pd.isna(price):
        return None
    return price / rate


# ============ 3. Sortie -- exacte replique de recheck_and_exit(), donnees reelles ============

def check_exit(unrealized: float, peak: float, stop_loss: float = STOP_LOSS_PCT,
               ratchet_step: float = RATCHET_STEP_PCT, ratchet_giveback: float = RATCHET_GIVEBACK_PCT):
    """Returns (exit_reason_or_None, new_peak) -- mirrors simulate_portfolio.recheck_and_exit's
    stop_loss/trailing_stop math exactly (2026-09-11 staircase ratchet). momentum_lost and
    valuation_reached are checked by the caller (need the live row, not just unrealized/peak)."""
    peak = unrealized if pd.isna(peak) or unrealized > peak else peak
    stop_loss_hit = unrealized <= stop_loss
    milestone = int(peak // ratchet_step) if pd.notna(peak) else 0
    trailing_stop_hit = milestone >= 1 and unrealized <= milestone * ratchet_step - ratchet_giveback
    if trailing_stop_hit:
        return "trailing_stop", peak
    if stop_loss_hit:
        return "stop_loss", peak
    return None, peak


# ============ 4. Bot #1 (aveugle, sans capital) ============

def run_bot1_pit(panel: pd.DataFrame) -> tuple:
    held = {}
    closed = []
    times = sorted(panel["snapshot_time"].unique())
    for t in times:
        snap = panel[panel["snapshot_time"] == t].set_index("ticker")
        for tk in list(held):
            info = held[tk]
            if tk not in snap.index:
                continue  # ticker temporarily missing this snapshot (fetch failure) -- skip, don't force an exit
            row = snap.loc[tk]
            price = row["price"]
            if pd.isna(price):
                continue
            unrealized = price / info["entry_price"] - 1
            reason, peak = check_exit(unrealized, info.get("peak"))
            info["peak"] = peak
            momentum_lost = pd.notna(row["mom_12_2"]) and (row["mom_12_2"] <= 0 or row["mom_12_2"] <= row["sector_momentum"])
            valuation_reached = pd.notna(row["valuation_gap"]) and row["valuation_gap"] <= 0
            if reason is None and valuation_reached:
                reason = "valorisation_atteinte"
            if reason is None and momentum_lost:
                reason = "momentum_perdu"
            if reason is not None:
                closed.append({**info, "ticker": tk, "exit_date": t, "exit_price": price,
                               "exit_reason": reason, "return_pct": unrealized})
                del held[tk]

        cands = snap[(snap["passes_filter"] == True) & (~snap.index.isin(held))]  # noqa: E712
        for tk, row in cands.iterrows():
            if pd.isna(row["price"]):
                continue
            held[tk] = {"sector": row["sector"], "entry_date": t, "entry_price": row["price"],
                        "entry_valuation_gap": row["valuation_gap"], "peak": 0.0}

    open_rows = []
    last_prices = last_known_prices(panel, held.keys())
    for tk, info in held.items():
        if tk not in last_prices.index:
            continue
        px = last_prices[tk]
        open_rows.append({**info, "ticker": tk, "unrealized_return_pct": px / info["entry_price"] - 1})
    return pd.DataFrame(closed), pd.DataFrame(open_rows)


# ============ 5. Bots #2/#3/#25 (capital contraint, EUR, diversifie) ============

def conviction_gate(info: dict, row, p: dict):
    """Porte de conviction de Bot #25 (reinforce_convictions) : renvoie valuation_gap_now si la
    position peut etre renforcee, sinon None. Extraite pour servir aussi a l'apport mensuel."""
    price, vg_now = row["price"], row["valuation_gap"]
    if pd.isna(price) or pd.isna(vg_now):
        return None
    unrealized = price / info["entry_price"] - 1
    if not (unrealized <= p["delta_min_drop_pct"] and unrealized > p["stop_loss_pct"]):
        return None
    if vg_now < info["entry_valuation_gap"]:
        return None  # coeur de la regle du 2026-09-14 : pas moins sous-evalue qu'a l'achat
    if unrealized <= p["delta_deep_drop_pct"]:
        mom_12_2, sm = row["mom_12_2"], row["sector_momentum"]
        if pd.isna(mom_12_2) or pd.isna(sm) or (mom_12_2 - sm) < p["delta_mom_spread_min"]:
            return None
    return vg_now


def apply_reinforcement(info: dict, price, price_eur, budget: float):
    old_v = info["entry_value_eur"]
    new_v = old_v + budget
    info["entry_price"] = (info["entry_price"] * old_v + price * budget) / new_v
    info["entry_price_eur"] = (info["entry_price_eur"] * old_v + price_eur * budget) / new_v
    info["entry_value_eur"] = new_v
    info["reinforcement_count"] = info.get("reinforcement_count", 0) + 1


def run_slotted_pit(panel: pd.DataFrame, currency_of: dict, fx_daily: pd.DataFrame,
                     starting_slots: int, max_per_sector, mode: str,
                     starting_capital: float, reinforce: bool = False, params: dict = None) -> tuple:
    p = {**DEFAULT_PARAMS, **(params or {})}
    target_size = starting_capital / starting_slots
    cash = starting_capital
    held = {}
    closed = []
    nav_curve = {}
    reserve = 0.0          # part de `cash` reservee au renfort (apport mensuel)
    contributed = 0.0
    last_month = None
    times = sorted(panel["snapshot_time"].unique())

    for t in times:
        snap = panel[panel["snapshot_time"] == t].set_index("ticker")

        month = (t.year, t.month)
        if p["monthly_contribution_eur"] and last_month is not None and month != last_month:
            cash += p["monthly_contribution_eur"]
            reserve += p["monthly_contribution_eur"]
            contributed += p["monthly_contribution_eur"]
            print(f"  APPORT {t.date()} : +{p['monthly_contribution_eur']:.2f} EUR (reserve renfort)", file=sys.stderr)
        last_month = month

        for tk in list(held):
            info = held[tk]
            if tk not in snap.index or pd.isna(snap.loc[tk, "price"]):
                continue
            row = snap.loc[tk]
            price = row["price"]
            unrealized = price / info["entry_price"] - 1
            reason, peak = check_exit(unrealized, info.get("peak"), p["stop_loss_pct"],
                                       p["ratchet_step_pct"], p["ratchet_giveback_pct"])
            info["peak"] = peak
            momentum_lost = pd.notna(row["mom_12_2"]) and (row["mom_12_2"] <= 0 or row["mom_12_2"] <= row["sector_momentum"])
            valuation_reached = pd.notna(row["valuation_gap"]) and row["valuation_gap"] <= 0
            if reason is None and valuation_reached and p["exit_on_valuation_reached"]:
                reason = "valorisation_atteinte"
            if reason is None and momentum_lost and p["exit_on_momentum_lost"]:
                reason = "momentum_perdu"
            if reason is None:
                continue

            price_eur = to_eur_pit(price, info["currency"], fx_daily, t)
            exit_value_eur = (info["entry_value_eur"] * price_eur / info["entry_price_eur"]
                               if price_eur and info.get("entry_price_eur") else info["entry_value_eur"])
            cash += exit_value_eur
            closed.append({**info, "ticker": tk, "exit_date": t, "exit_price": price,
                           "exit_reason": reason, "return_pct": unrealized, "exit_value_eur": exit_value_eur})
            del held[tk]

        # ---- renfort de conviction REEL (Bot #25 uniquement) ----
        if reinforce:
            cap_eur = p["delta_max_position_share"] * starting_capital
            for tk, info in held.items():
                free_cash = cash - reserve
                if info["entry_value_eur"] >= cap_eur or free_cash <= 0 or tk not in snap.index:
                    continue
                row = snap.loc[tk]
                vg_now = conviction_gate(info, row, p)
                if vg_now is None:
                    continue
                price = row["price"]
                price_eur = to_eur_pit(price, info["currency"], fx_daily, t)
                if not price_eur or price_eur <= 0:
                    continue
                budget = min(target_size, cap_eur - info["entry_value_eur"], free_cash)
                if budget <= 0:
                    continue
                apply_reinforcement(info, price, price_eur, budget)
                cash -= budget
                print(f"  RENFORT REEL {tk} @ {t.date()} (valorisation {vg_now:+.1%} vs "
                      f"{info['entry_valuation_gap']:+.1%} a l'achat) : +{budget:.2f} EUR", file=sys.stderr)

            # ---- reserve d'apports : reparti entre les positions qui passent la porte de conviction ----
            if reserve > 0.01:
                eligible = []
                for tk, info in held.items():
                    if tk not in snap.index or info["entry_value_eur"] >= cap_eur:
                        continue
                    row = snap.loc[tk]
                    vg_now = conviction_gate(info, row, p)
                    price_eur = to_eur_pit(row["price"], info["currency"], fx_daily, t)
                    if vg_now is None or not price_eur or price_eur <= 0:
                        continue
                    eligible.append((tk, info, row["price"], price_eur, vg_now))
                if eligible:
                    weights = [1.0 if p["contribution_split"] == "equal" else max(e[4], 1e-6) for e in eligible]
                    total_w = sum(weights)
                    pot = reserve
                    for (tk, info, price, price_eur, vg_now), w in zip(eligible, weights):
                        budget = min(pot * w / total_w, cap_eur - info["entry_value_eur"], reserve)
                        if budget <= 0.01:
                            continue
                        apply_reinforcement(info, price, price_eur, budget)
                        cash -= budget
                        reserve -= budget
                        print(f"  RENFORT APPORT {tk} @ {t.date()} (valorisation {vg_now:+.1%}) : +{budget:.2f} EUR",
                              file=sys.stderr)

        # ---- nouvelles positions ----
        cand = snap[(snap["passes_filter"] == True) & (~snap.index.isin(held))].copy()  # noqa: E712
        cand = cand.dropna(subset=["price", "valuation_gap", "mom_12_2", "sector_momentum", "quality_multiplier"])
        cand = cand[(cand["valuation_gap"] >= p["min_valuation_gap"]) & (cand["mom_12_2"] >= p["min_mom_12_2"])
                    & (cand["quality_multiplier"] >= p["min_quality_multiplier"])]
        if len(cand):
            cand = cand.reset_index()
            cand["score"] = composite_score(cand)
            rejected = set()
            exhausted = set()
            while cash - reserve >= target_size and len(cand):
                sector_counts = {}
                for info in held.values():
                    sector_counts[info["sector"]] = sector_counts.get(info["sector"], 0) + 1
                na_count = sum(1 for tk in held if ticker_region(tk) == "North America")
                total_held = len(held)
                ranked = cand.sort_values("score", ascending=False)
                if mode == "capped":
                    pick = _pick_capped(ranked, held, sector_counts, na_count, total_held, max_per_sector, rejected)
                else:
                    pick = _pick_even_sector(ranked, held, sector_counts, na_count, total_held, exhausted, rejected)
                if pick is None:
                    break
                tk = pick["ticker"]
                price = pick["price"]
                price_eur = to_eur_pit(price, currency_of.get(tk), fx_daily, t)
                if price_eur is None or price_eur <= 0:
                    rejected.add(tk)
                    cand = cand[cand["ticker"] != tk]
                    continue
                held[tk] = {"sector": pick["sector"], "entry_date": t, "entry_price": price,
                            "entry_price_eur": price_eur, "entry_value_eur": target_size,
                            "entry_valuation_gap": pick["valuation_gap"], "currency": currency_of.get(tk),
                            "peak": 0.0, "reinforcement_count": 0}
                cash -= target_size
                cand = cand[cand["ticker"] != tk]

        mtm = 0.0
        for tk, info in held.items():
            price_eur = to_eur_pit(snap.loc[tk, "price"], info["currency"], fx_daily, t) if tk in snap.index else None
            mtm += (info["entry_value_eur"] * price_eur / info["entry_price_eur"]
                    if price_eur and info.get("entry_price_eur") else info["entry_value_eur"])
        nav_curve[t] = cash + mtm

    open_rows = []
    last_t = times[-1]
    last_prices = last_known_prices(panel, held.keys())
    for tk, info in held.items():
        if tk not in last_prices.index:
            continue
        px = last_prices[tk]
        price_eur = to_eur_pit(px, info["currency"], fx_daily, last_t)
        cur_val = (info["entry_value_eur"] * price_eur / info["entry_price_eur"]
                   if price_eur and info.get("entry_price_eur") else info["entry_value_eur"])
        open_rows.append({**info, "ticker": tk, "unrealized_return_pct": px / info["entry_price"] - 1,
                           "current_value_eur": cur_val})
    return pd.DataFrame(closed), pd.DataFrame(open_rows), pd.Series(nav_curve), cash, contributed


# ============ 6. Variantes (fichier de config) ============

def load_variants() -> list:
    """Variantes definies dans screener/backtest_pit_variants.json. Chaque variante = un bot de
    reference (`base`) + surcharges. Cles reconnues : slots, max_per_sector, mode, capital, reinforce
    + toutes les cles de DEFAULT_PARAMS. Une cle inconnue leve une erreur (une faute de frappe ne doit
    pas passer pour une variante qui ne change rien)."""
    if not VARIANTS_PATH.exists():
        return []
    variants = json.loads(VARIANTS_PATH.read_text(encoding="utf-8")).get("variants", [])
    allowed = set(DEFAULT_PARAMS) | {"slots", "max_per_sector", "mode", "capital", "reinforce"}
    meta = {"name", "label", "base", "description"}
    seen = set()
    for v in variants:
        unknown = set(v) - allowed - meta
        if unknown:
            raise ValueError(f"variante {v.get('name')!r} : cles inconnues {sorted(unknown)}")
        if v.get("monthly_contribution_eur") and not (v.get("reinforce", BASES.get(v.get("base"), {}).get("reinforce"))):
            raise ValueError(f"variante {v.get('name')!r} : un apport mensuel exige reinforce=true")
        if v.get("contribution_split", "valuation") not in ("valuation", "equal"):
            raise ValueError(f"variante {v.get('name')!r} : contribution_split doit etre valuation|equal")
        if v.get("base") not in BASES:
            raise ValueError(f"variante {v.get('name')!r} : base doit etre parmi {sorted(BASES)}")
        if not v.get("name") or v["name"] in seen or v["name"] in BOTS:
            raise ValueError(f"variante : nom manquant, duplique ou reserve ({v.get('name')!r})")
        seen.add(v["name"])
    return variants


def run_variant(panel, currency_of, fx_daily, v: dict) -> dict:
    over = ("slots", "max_per_sector", "mode", "capital", "reinforce")
    cfg = {**BASES[v["base"]], **{k: v[k] for k in over if k in v}}
    params = {k: v[k] for k in DEFAULT_PARAMS if k in v}
    closed, open_df, nav, cash, contributed = run_slotted_pit(panel, currency_of, fx_daily, cfg["slots"], cfg["max_per_sector"],
                                                  cfg["mode"], cfg["capital"], cfg["reinforce"], params)
    label = v.get("label") or f"Variante {v['name']} (base {v['base']}, PIT reel)"
    result = summarize(label, closed, open_df, nav, cash, cfg["capital"], contributed)
    closed.to_csv(OUT_DIR / f"variant_{v['name']}_pit_trades.csv", index=False)
    open_df.to_csv(OUT_DIR / f"variant_{v['name']}_pit_open.csv", index=False)
    return result


# ============ 7. Sortie / orchestration ============

def summarize(label, closed, open_df, nav=None, final_cash=None, starting_capital=None, contributed=0.0):
    n_closed = len(closed)
    win_rate = float((closed["return_pct"] > 0).mean()) if n_closed else None
    avg_return = float(closed["return_pct"].mean()) if n_closed else None
    print(f"--- {label} ---")
    print(f"  Clotures : n={n_closed}  win_rate={'n/a' if win_rate is None else f'{win_rate:+.1%}'}  "
          f"retour_moyen={'n/a' if avg_return is None else f'{avg_return:+.1%}'}")
    if n_closed:
        print(f"  Motifs de sortie : {closed['exit_reason'].value_counts().to_dict()}")
    print(f"  Positions ouvertes en fin de periode : n={len(open_df)}")
    result = {"bot": label, "n_closed": n_closed, "win_rate": win_rate, "avg_return_closed": avg_return,
              "n_open": len(open_df)}
    if nav is not None and len(nav):
        cap = starting_capital
        total_return = nav.iloc[-1] / (cap + contributed) - 1  # sur tout l'argent verse, apports compris
        print(f"  Equity EUR (depart {cap:.0f}) : {nav.iloc[-1]:.2f} EUR ({total_return:+.1%})  "
              f"cash_final={final_cash:.2f} EUR")
        result["final_equity_eur"] = float(nav.iloc[-1])
        result["total_return_pct"] = float(total_return)
        result["contributions_eur"] = float(contributed)
    if "reinforcement_count" in closed.columns or "reinforcement_count" in open_df.columns:
        n_r = int((closed["reinforcement_count"] > 0).sum()) if "reinforcement_count" in closed.columns and len(closed) else 0
        n_r += int((open_df["reinforcement_count"] > 0).sum()) if "reinforcement_count" in open_df.columns and len(open_df) else 0
        print(f"  Renforts de conviction reels declenches : {n_r} position(s)")
    print()
    return result


def main():
    print("Extraction de l'historique Git de full_valuation_latest.csv...", file=sys.stderr)
    snapshots = extract_snapshots()
    print(f"{len(snapshots)} snapshots extraits/en cache.", file=sys.stderr)
    panel = load_panel(snapshots)
    n_before = panel["snapshot_time"].nunique()
    panel = panel[panel["snapshot_time"] >= WINDOW_START]
    print(f"Fenetre tronquee a partir de {WINDOW_START.date()} (voir WINDOW_START) : "
          f"{n_before} -> {panel['snapshot_time'].nunique()} points retenus.", file=sys.stderr)
    n_tickers_first = panel[panel["snapshot_time"] == panel["snapshot_time"].min()]["ticker"].nunique()
    n_tickers_last = panel[panel["snapshot_time"] == panel["snapshot_time"].max()]["ticker"].nunique()
    print(f"Fenetre : {panel['snapshot_time'].min()} -> {panel['snapshot_time'].max()} "
          f"({panel['snapshot_time'].nunique()} points, univers {n_tickers_first}->{n_tickers_last} tickers)\n")

    currency_of = backfill_currency(panel)
    currencies = set(currency_of.values())
    months = max(2, int((panel["snapshot_time"].max() - panel["snapshot_time"].min()).days / 30) + 1)
    print(f"Telechargement FX historique ({months} mois, {len(currencies)} devises)...", file=sys.stderr)
    fx_daily = download_fx_window(currencies, months)
    print(f"FX obtenu pour : {list(fx_daily.columns)}\n")

    results = []

    closed1, open1 = run_bot1_pit(panel)
    results.append(summarize(BOTS["bot1_blind"], closed1, open1))
    closed1.to_csv(OUT_DIR / "bot1_blind_pit_trades.csv", index=False)
    open1.to_csv(OUT_DIR / "bot1_blind_pit_open.csv", index=False)

    closed2, open2, nav2, cash2, _ = run_slotted_pit(panel, currency_of, fx_daily, STARTING_SLOTS_B2,
                                                    MAX_PER_SECTOR, "capped", STARTING_CAPITAL)
    results.append(summarize(BOTS["bot2_constrained"], closed2, open2, nav2, cash2, STARTING_CAPITAL))
    closed2.to_csv(OUT_DIR / "bot2_constrained_pit_trades.csv", index=False)
    open2.to_csv(OUT_DIR / "bot2_constrained_pit_open.csv", index=False)

    closed3, open3, nav3, cash3, _ = run_slotted_pit(panel, currency_of, fx_daily, STARTING_SLOTS_B3,
                                                    None, "even_sector", STARTING_CAPITAL)
    results.append(summarize(BOTS["bot3_large"], closed3, open3, nav3, cash3, STARTING_CAPITAL))
    closed3.to_csv(OUT_DIR / "bot3_large_pit_trades.csv", index=False)
    open3.to_csv(OUT_DIR / "bot3_large_pit_open.csv", index=False)

    closed25, open25, nav25, cash25, _ = run_slotted_pit(panel, currency_of, fx_daily, STARTING_SLOTS_DELTA,
                                                        MAX_PER_SECTOR, "capped", DELTA_STARTING_CAPITAL,
                                                        reinforce=True)
    results.append(summarize(BOTS["bot25_delta"], closed25, open25, nav25, cash25, DELTA_STARTING_CAPITAL))
    closed25.to_csv(OUT_DIR / "bot25_delta_pit_trades.csv", index=False)
    open25.to_csv(OUT_DIR / "bot25_delta_pit_open.csv", index=False)

    for v in load_variants():
        results.append(run_variant(panel, currency_of, fx_daily, v))

    pd.DataFrame(results).to_csv(OUT_DIR / "summary_pit.csv", index=False)
    print(f"Resultats ecrits dans {OUT_DIR.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
