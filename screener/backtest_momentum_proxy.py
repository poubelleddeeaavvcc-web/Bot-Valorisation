"""Proxy backtest (price/momentum only) for Bot #1/#2/#3 over the trailing ~12 months.

WHY A PROXY, NOT A REAL BACKTEST OF THE BOTS' ACTUAL STRATEGY: entries in the live bots
(simulate_portfolio.py / simulate_constrained_portfolio.py / simulate_large_portfolio.py)
are gated on valuation_gap -- a sector-relative fair P/E adjusted for quality (ROE, margin,
debt) -- computed from yfinance's CURRENT `.info` snapshot (trailing EPS, trailing P/E,
ROE...). There is no point-in-time fundamentals data in this repo (or readily available
for free) to recompute those as they stood at each past rebalance date. Applying TODAY's
fundamentals to PAST prices would be lookahead bias, not a backtest -- which is exactly
why the live bots were built as forward paper-trading in the first place (see
simulate_portfolio.py's docstring). That gap is NOT closed by anything below -- entries
here are still momentum-only.

What IS closed, in this version, vs. the first proxy attempt (per the user's "make it
match the bots more closely" ask):
  - Daily stop-loss detection: the live bots recheck every open position roughly hourly
    (update-screener.yml), so a hard -25% floor can trigger almost any day, not just at a
    month boundary. The first version only checked exits once a month, understating how
    fast the stop-loss backstop actually fires. This version scans DAILY closes between
    rebalance points for the stop-loss check; momentum_lost is still evaluated monthly,
    because mom_12_2 is itself derived from monthly closes and genuinely does not change
    within a month (recomputing it more often would just repeat the same number).
  - EUR-denominated capital for bot2/bot3: the real ledgers track euros, converted via
    live FX rates (simulate_constrained_portfolio.to_eur) -- a naive 1:1 read of foreign
    prices against a EUR cash balance was an actual bug caught in this project's history
    (2026-08-17). This version fetches historical FX series for every currency in the
    universe and converts EUR position sizing/cash the same way, while keeping the
    stop-loss/momentum-lost trigger itself on the NATIVE price ratio -- matching
    recheck_and_exit(), which compares fresh price to entry price before any EUR
    conversion.
  - Cash-gated reinvestment instead of a fixed slot count: the real bots have "no hard
    position-count ceiling" -- they buy the next candidate whenever cash clears
    TARGET_POSITION_SIZE, so a good run can open several new positions at once. The first
    version used a fixed slots=15/30 capacity instead; this version mirrors the real
    cash-gated loop (see fill_slots() in simulate_constrained_portfolio.py /
    simulate_large_portfolio.py).
  - Quality as a secondary ranking axis: composite_score in select_top_picks.py averages
    value + momentum + quality percentile ranks. Value is still impossible to replay
    (see above), but quality_multiplier (ROE/margin/debt-based) is read from TODAY's
    snapshot and frozen across the whole backtest window -- a real approximation (quality
    metrics come from quarterly financials and do move), but a much smaller lookahead risk
    than valuation_gap, which is dominated by price moves the model must not have known in
    advance. Selection ranking for bot2/bot3 is now 0.5*momentum_pct + 0.5*quality_pct
    instead of momentum margin alone.

Still NOT modeled (flagged, not silently ignored): whole-share vs. fractional-share
rounding, the North-America-share cap's exact relaxation order in edge cases, and bot3's
"reinforce an existing position when its sector has no fresh candidate" rule (approximated
here as: skip that sector for the rest of the round instead of averaging into the existing
position). None of these materially change the shape of the result; all are cheaper
simplifications than the valuation-data gap that remains the real limitation.
"""
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.select_top_picks import ticker_region, NORTH_AMERICA_MAX_SHARE  # noqa: E402

VALUATION_PATH = HERE / "results/screener/full_valuation_latest.csv"
FUNDAMENTALS_CACHE_PATH = HERE / "results/screener/fundamentals_cache.csv"
OUT_DIR = HERE / "results/backtest"
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOP_LOSS_PCT = -0.25          # same as simulate_portfolio.STOP_LOSS_PCT
LOOKBACK_MONTHS_TOTAL = 26     # 12 rebalance months + 13 months of momentum lookback each needs
REBALANCE_MONTHS = 12          # walk-forward window we report on
CHUNK_SIZE = 300               # yfinance batch download chunk (fetch_cache.py precedent)
CHUNK_DELAY = 2.0              # seconds between chunks

STARTING_CAPITAL = 500.0       # same illustrative amount as the live bots

# Same FX ticker map as simulate_constrained_portfolio.FX_PAIR -- duplicated because that
# module's fetch_fx_rates() only fetches a live snapshot, not a historical series.
FX_PAIR = {
    "USD": "EURUSD=X", "GBP": "EURGBP=X", "GBp": "EURGBP=X", "JPY": "EURJPY=X",
    "CHF": "EURCHF=X", "SEK": "EURSEK=X", "CAD": "EURCAD=X", "HKD": "EURHKD=X",
    "AUD": "EURAUD=X", "KRW": "EURKRW=X", "TWD": "EURTWD=X", "INR": "EURINR=X",
    "BRL": "EURBRL=X", "MXN": "EURMXN=X",
}

BOTS = {
    "bot1_blind": {"label": "Bot #1 (blind, proxy momentum)"},
    "bot2_constrained": {"label": "Bot #2 (constrained, proxy momentum+quality)",
                          "starting_slots": 15, "max_per_sector": 3, "mode": "capped"},
    "bot3_large": {"label": "Bot #3 (large, proxy momentum+quality)",
                   "starting_slots": 30, "max_per_sector": None, "mode": "even_sector"},
}

BENCHMARKS = {"sp500": "^GSPC", "euronext100": "^N100"}

UNIVERSE_G = None


def load_universe() -> pd.DataFrame:
    val = pd.read_csv(VALUATION_PATH)
    val = val.dropna(subset=["sector"])[["ticker", "sector", "industry", "quality_multiplier"]].drop_duplicates("ticker")
    fund = pd.read_csv(FUNDAMENTALS_CACHE_PATH)[["ticker", "currency"]].drop_duplicates("ticker")
    df = val.merge(fund, on="ticker", how="left")
    df["quality_pct"] = df["quality_multiplier"].rank(pct=True)
    return df.reset_index(drop=True)


def _chunked_download(tickers: list, period: str, interval: str) -> dict:
    closes = {}
    n_chunks = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        chunk_no = i // CHUNK_SIZE + 1
        print(f"  telechargement chunk {chunk_no}/{n_chunks} ({len(chunk)} tickers, {interval})...", file=sys.stderr)
        try:
            data = yf.download(chunk, period=period, interval=interval,
                                auto_adjust=True, group_by="ticker", threads=True, progress=False)
        except Exception as e:
            print(f"    echec chunk {chunk_no}: {e}", file=sys.stderr)
            continue
        for tk in chunk:
            try:
                s = data[tk]["Close"].dropna() if len(chunk) > 1 else data["Close"].dropna()
            except (KeyError, TypeError):
                continue
            if len(s):
                closes[tk] = s
        if chunk_no < n_chunks:
            time.sleep(CHUNK_DELAY)
    return closes


def download_daily_panel(tickers: list) -> pd.DataFrame:
    closes = _chunked_download(tickers, period=f"{LOOKBACK_MONTHS_TOTAL}mo", interval="1d")
    panel = pd.DataFrame(closes)
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    return panel.sort_index()


def download_fx_daily(currencies: set) -> pd.DataFrame:
    pairs = {ccy: FX_PAIR[ccy] for ccy in currencies if ccy in FX_PAIR and ccy != "GBp"}
    if "GBp" in currencies:
        pairs["GBP"] = FX_PAIR["GBP"]
    closes = _chunked_download(list(set(pairs.values())), period=f"{LOOKBACK_MONTHS_TOTAL}mo", interval="1d")
    ticker_to_ccy = {v: k for k, v in pairs.items()}
    renamed = {ticker_to_ccy[tk]: s for tk, s in closes.items() if tk in ticker_to_ccy}
    panel = pd.DataFrame(renamed)
    panel.index = pd.to_datetime(panel.index).tz_localize(None)
    return panel.sort_index()


def to_eur_panel(daily_panel: pd.DataFrame, currency_of: dict, fx_daily: pd.DataFrame) -> pd.DataFrame:
    """Converts a native-currency daily price panel to EUR, ticker by ticker, using the
    FX rate on/before each date (forward-filled across weekends/holidays -- FX and equity
    calendars don't line up exactly). Same rate convention as
    simulate_constrained_portfolio.to_eur(): rate = units of <currency> per 1 EUR, GBp
    (pence) shares GBP's rate but needs an extra /100."""
    fx_ff = fx_daily.reindex(daily_panel.index).ffill().bfill()
    out = {}
    for tk in daily_panel.columns:
        ccy = currency_of.get(tk)
        if ccy is None or pd.isna(ccy) or ccy == "EUR":
            out[tk] = daily_panel[tk]
            continue
        key = "GBP" if ccy == "GBp" else ccy
        if key not in fx_ff.columns:
            continue  # no FX series available -- ticker excluded from EUR-denominated bots, same as real to_eur() returning None
        rate = fx_ff[key]
        converted = daily_panel[tk] / rate
        out[tk] = converted / 100 if ccy == "GBp" else converted
    return pd.DataFrame(out)


def compute_momentum(monthly_panel: pd.DataFrame) -> pd.DataFrame:
    """mom_12_2 per ticker per month-index: close[i-1]/close[i-13] - 1."""
    return monthly_panel.shift(1) / monthly_panel.shift(13) - 1


def sector_momentum_series(mom: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    sector_of = universe.set_index("ticker")["sector"]
    common = [t for t in mom.columns if t in sector_of.index]
    mom = mom[common]
    sectors = sector_of.loc[common]
    sec_mom = mom.T.groupby(sectors).transform("median").T
    return sec_mom


def rank_candidates(date_i, mom, sec_mom, universe, held: set) -> pd.DataFrame:
    m = mom.loc[date_i].reindex(sec_mom.columns)
    sm = sec_mom.loc[date_i]
    ok = (sm > 0) & (m > sm) & m.notna() & sm.notna()
    cand_tickers = [t for t in m.index[ok] if t not in held]
    if not cand_tickers:
        return pd.DataFrame(columns=["ticker", "sector", "mom", "sector_mom", "margin", "quality_pct", "score"])
    uni = universe.set_index("ticker")
    momentum_pct = m[cand_tickers].rank(pct=True)
    rows = []
    for t in cand_tickers:
        rows.append({"ticker": t, "sector": uni.loc[t, "sector"] if t in uni.index else None,
                      "mom": m[t], "sector_mom": sm[t], "margin": m[t] - sm[t],
                      "momentum_pct": momentum_pct[t],
                      "quality_pct": uni.loc[t, "quality_pct"] if t in uni.index else 0.5})
    df = pd.DataFrame(rows)
    df["quality_pct"] = df["quality_pct"].fillna(0.5)
    df["score"] = 0.5 * df["momentum_pct"] + 0.5 * df["quality_pct"]
    return df.sort_values("score", ascending=False)


def asof_panel(daily_df: pd.DataFrame, dates) -> pd.DataFrame:
    """Forward-filled snapshot of daily_df at each of `dates` -- the last known price on or
    before that date, not an exact-match reindex. Needed because a monthly resample's
    month-end label (e.g. "2025-11-30", a Sunday) frequently isn't itself a trading day, so
    reindexing a daily series to that exact label alone (with no earlier context to fill
    from) silently returns NaN -- confirmed on this data: 5 of 13 rebalance dates in the
    trailing-12-month window aren't real trading days."""
    idx = pd.DatetimeIndex(dates)
    union_idx = daily_df.index.union(idx).sort_values()
    filled = daily_df.reindex(union_idx).ffill()
    return filled.loc[idx]


def scan_daily_stop_loss(daily_native: pd.DataFrame, tk: str, entry_price: float,
                          window_start, window_end):
    """First trading day (window_start, window_end] where native-price unrealized return
    breaches STOP_LOSS_PCT, or None if it never does in this window. Matches
    recheck_and_exit(): the stop-loss decision uses the native price ratio, not EUR."""
    if tk not in daily_native.columns or entry_price is None or pd.isna(entry_price) or entry_price == 0:
        return None
    s = daily_native[tk]
    mask = (s.index > window_start) & (s.index <= window_end)
    sub = s[mask].dropna()
    if sub.empty:
        return None
    unreal = sub / entry_price - 1
    hit = unreal[unreal <= STOP_LOSS_PCT]
    if hit.empty:
        return None
    return hit.index[0], float(sub.loc[hit.index[0]])


def run_bot1_blind(dates, mom, sec_mom, daily_native: pd.DataFrame, native_at_dates: pd.DataFrame) -> tuple:
    """Blind buy-every-candidate, no capital constraint -- tracks % return per bet only,
    same as the real bot1 (has_eur_equity=False in bot_checkin.py)."""
    held = {}
    closed = []
    prev_date = dates[0] - pd.DateOffset(months=1)
    for date_i in dates:
        price_row = native_at_dates.loc[date_i]
        for tk in list(held):
            info = held[tk]
            hit = scan_daily_stop_loss(daily_native, tk, info["entry_price"], prev_date, date_i)
            if hit is not None:
                hit_date, hit_price = hit
                closed.append({**info, "ticker": tk, "exit_date": hit_date, "exit_price": hit_price,
                               "exit_reason": "stop_loss", "return_pct": hit_price / info["entry_price"] - 1})
                del held[tk]
                continue
            if tk not in mom.columns:
                continue
            m, sm = mom.loc[date_i].get(tk), sec_mom.loc[date_i].get(tk)
            px = price_row.get(tk)
            if pd.isna(m) or pd.isna(sm) or px is None or pd.isna(px):
                continue
            unreal = px / info["entry_price"] - 1
            if m <= 0 or m <= sm:
                closed.append({**info, "ticker": tk, "exit_date": date_i, "exit_price": px,
                               "exit_reason": "momentum_perdu", "return_pct": unreal})
                del held[tk]

        cands = rank_candidates(date_i, mom, sec_mom, universe=UNIVERSE_G, held=set(held))
        for _, c in cands.iterrows():
            entry_px = price_row.get(c["ticker"])
            if entry_px is None or pd.isna(entry_px):
                continue
            held[c["ticker"]] = {"sector": c["sector"], "entry_date": date_i, "entry_price": entry_px}
        prev_date = date_i

    open_rows = []
    last_date = dates[-1]
    last_price_row = native_at_dates.loc[last_date]
    for tk, info in held.items():
        px = last_price_row.get(tk)
        if px is None or pd.isna(px):
            continue
        open_rows.append({**info, "ticker": tk, "unrealized_return_pct": px / info["entry_price"] - 1})
    return pd.DataFrame(closed), pd.DataFrame(open_rows)


def _pick_capped(cands: pd.DataFrame, held: dict, sector_counts: dict, na_count: int,
                  total_held: int, max_per_sector, rejected: set):
    for enforce_na_cap in (True, False):
        cap_range = range(max_per_sector, 10) if max_per_sector is not None else [10**9]
        for cap in cap_range:
            elig = cands[~cands["ticker"].isin(held) & ~cands["ticker"].isin(rejected)]
            if max_per_sector is not None:
                elig = elig[elig["sector"].map(lambda s: sector_counts.get(s, 0)) < cap]
            if enforce_na_cap:
                max_na = int((total_held + 1) * NORTH_AMERICA_MAX_SHARE)
                elig = elig[elig["ticker"].map(lambda t: ticker_region(t) != "North America" or na_count < max_na)]
            if len(elig):
                return elig.iloc[0]
    return None


def _pick_even_sector(cands: pd.DataFrame, held: dict, sector_counts: dict, na_count: int,
                       total_held: int, exhausted_sectors: set, rejected: set):
    max_na = int((total_held + 1) * NORTH_AMERICA_MAX_SHARE)
    sectors_by_count = sorted(set(sector_counts) | set(cands["sector"].dropna()), key=lambda s: sector_counts.get(s, 0))
    for sector in sectors_by_count:
        if sector in exhausted_sectors:
            continue
        elig = cands[(cands["sector"] == sector) & ~cands["ticker"].isin(held) & ~cands["ticker"].isin(rejected)]
        elig = elig[elig["ticker"].map(lambda t: ticker_region(t) != "North America" or na_count < max_na)]
        if len(elig):
            return elig.iloc[0]
        exhausted_sectors.add(sector)
    return None


def _price_asof(series: pd.Series, date):
    """Last known value on or before `date` -- used for stop-loss exit dates, which land on
    an arbitrary trading day inside a month, not on one of the 13 rebalance dates that
    asof_panel() precomputes."""
    s = series.loc[:date].dropna()
    return s.iloc[-1] if len(s) else None


def run_slotted_bot(dates, mom, sec_mom, daily_native, daily_eur, currency_of,
                     native_at_dates, eur_at_dates,
                     starting_slots, max_per_sector, mode) -> tuple:
    target_size = STARTING_CAPITAL / starting_slots
    cash = STARTING_CAPITAL
    held = {}  # ticker -> dict
    closed = []
    prev_date = dates[0] - pd.DateOffset(months=1)
    nav_curve = {}

    for date_i in dates:
        native_row = native_at_dates.loc[date_i]
        eur_row = eur_at_dates.loc[date_i]
        for tk in list(held):
            info = held[tk]
            hit = scan_daily_stop_loss(daily_native, tk, info["entry_price"], prev_date, date_i)
            if hit is not None:
                hit_date, hit_price = hit
            else:
                if tk not in mom.columns:
                    continue
                m, sm = mom.loc[date_i].get(tk), sec_mom.loc[date_i].get(tk)
                if pd.isna(m) or pd.isna(sm) or (m > 0 and m > sm):
                    continue  # still passes the momentum bar, keep holding
                hit_date, hit_price = date_i, native_row.get(tk)
                if hit_price is None or pd.isna(hit_price):
                    continue

            exit_value_eur = None
            if tk in daily_eur.columns:
                px_eur_now = eur_row.get(tk) if hit_date == date_i else _price_asof(daily_eur[tk], hit_date)
                if pd.notna(px_eur_now) and pd.notna(info.get("entry_price_eur")) and info["entry_price_eur"]:
                    exit_value_eur = info["entry_value_eur"] * (px_eur_now / info["entry_price_eur"])
            if exit_value_eur is None:
                exit_value_eur = info["entry_value_eur"]  # FX unavailable at exit -- fall back to entry value, same spirit as recheck_and_exit skipping on missing FX
            cash += exit_value_eur
            closed.append({**info, "ticker": tk, "exit_date": hit_date, "exit_price": hit_price,
                           "exit_reason": "stop_loss" if hit is not None else "momentum_perdu",
                           "return_pct": hit_price / info["entry_price"] - 1, "exit_value_eur": exit_value_eur})
            del held[tk]

        cands = rank_candidates(date_i, mom, sec_mom, universe=UNIVERSE_G, held=set(held))
        rejected = set()
        exhausted_sectors = set()
        while cash >= target_size and len(cands):
            sector_counts = {}
            for info in held.values():
                sector_counts[info["sector"]] = sector_counts.get(info["sector"], 0) + 1
            na_count = sum(1 for t in held if ticker_region(t) == "North America")
            total_held = len(held)

            if mode == "capped":
                pick = _pick_capped(cands, held, sector_counts, na_count, total_held, max_per_sector, rejected)
            else:
                pick = _pick_even_sector(cands, held, sector_counts, na_count, total_held, exhausted_sectors, rejected)
            if pick is None:
                break
            tk = pick["ticker"]
            entry_px_native = native_row.get(tk)
            entry_px_eur = eur_row.get(tk)
            if entry_px_native is None or pd.isna(entry_px_native) or entry_px_eur is None or pd.isna(entry_px_eur) or entry_px_eur <= 0:
                rejected.add(tk)
                continue
            held[tk] = {"sector": pick["sector"], "entry_date": date_i, "entry_price": entry_px_native,
                        "entry_price_eur": entry_px_eur, "entry_value_eur": target_size,
                        "currency": currency_of.get(tk)}
            cash -= target_size

        # month-end NAV: cash + mark-to-market EUR value of every open position
        mtm = 0.0
        for tk, info in held.items():
            px_eur_now = eur_row.get(tk)
            if pd.notna(px_eur_now) and info.get("entry_price_eur"):
                mtm += info["entry_value_eur"] * (px_eur_now / info["entry_price_eur"])
            else:
                mtm += info["entry_value_eur"]
        nav_curve[date_i] = cash + mtm
        prev_date = date_i

    open_rows = []
    last_date = dates[-1]
    last_native_row = native_at_dates.loc[last_date]
    last_eur_row = eur_at_dates.loc[last_date]
    for tk, info in held.items():
        px = last_native_row.get(tk)
        if px is None or pd.isna(px):
            continue
        px_eur_now = last_eur_row.get(tk)
        cur_val = (info["entry_value_eur"] * (px_eur_now / info["entry_price_eur"])
                   if pd.notna(px_eur_now) and info.get("entry_price_eur") else info["entry_value_eur"])
        open_rows.append({**info, "ticker": tk, "unrealized_return_pct": px / info["entry_price"] - 1,
                           "current_value_eur": cur_val})
    return pd.DataFrame(closed), pd.DataFrame(open_rows), pd.Series(nav_curve), cash


def build_equity_curve_bot1(dates, closed: pd.DataFrame, native_at_dates: pd.DataFrame, entry_info: dict) -> pd.Series:
    """Same 'average return of every bet placed so far' methodology as
    simulate_portfolio.append_equity_curve_point: not a compounding NAV, an honest running
    average of every position's return (locked-in for closed, mark-to-market for open)."""
    curve = {}
    for date_i in dates:
        price_row = native_at_dates.loc[date_i]
        rets = []
        if len(closed):
            c = closed[closed["exit_date"] <= date_i]
            rets.extend(c["return_pct"].tolist())
        for tk, info in entry_info.items():
            entry_date = info["entry_date"]
            if entry_date > date_i or tk not in price_row.index:
                continue
            still_open = not (len(closed) and ((closed["ticker"] == tk) & (closed["entry_date"] == entry_date) & (closed["exit_date"] <= date_i)).any())
            if still_open:
                px_now = price_row.get(tk)
                if pd.notna(px_now):
                    rets.append(px_now / info["entry_price"] - 1)
        curve[date_i] = np.mean(rets) if rets else np.nan
    return pd.Series(curve)


def summarize(label, closed: pd.DataFrame, open_df: pd.DataFrame, nav: pd.Series = None, final_cash: float = None) -> dict:
    n_closed = len(closed)
    win_rate = float((closed["return_pct"] > 0).mean()) if n_closed else None
    avg_return = float(closed["return_pct"].mean()) if n_closed else None
    avg_unrealized = float(open_df["unrealized_return_pct"].mean()) if len(open_df) else None
    med_unrealized = float(open_df["unrealized_return_pct"].median()) if len(open_df) else None
    print(f"--- {label} ---")
    print(f"  Clotures (proxy momentum+stop-loss quotidien) : n={n_closed}  "
          f"win_rate={'n/a' if win_rate is None else f'{win_rate:+.1%}'}  "
          f"retour_moyen={'n/a' if avg_return is None else f'{avg_return:+.1%}'}")
    print(f"  Positions ouvertes en fin de periode : n={len(open_df)}  "
          f"retour_latent_moyen={'n/a' if avg_unrealized is None else f'{avg_unrealized:+.1%}'}  "
          f"retour_latent_median={'n/a' if med_unrealized is None else f'{med_unrealized:+.1%}'}")
    result = {"bot": label, "n_closed": n_closed, "win_rate": win_rate, "avg_return_closed": avg_return,
              "n_open": len(open_df), "avg_unrealized_open": avg_unrealized, "median_unrealized_open": med_unrealized}
    if nav is not None and len(nav):
        total_return = nav.iloc[-1] / STARTING_CAPITAL - 1
        print(f"  Equity EUR (depart {STARTING_CAPITAL:.0f}) : {nav.iloc[-1]:.2f} EUR ({total_return:+.1%})  "
              f"cash_final={final_cash:.2f} EUR")
        result["final_equity_eur"] = float(nav.iloc[-1])
        result["total_return_pct"] = float(total_return)
    print()
    return result


def main():
    global UNIVERSE_G
    universe = load_universe()
    UNIVERSE_G = universe
    print(f"Univers : {len(universe)} tickers (secteur/industrie/quality figes a aujourd'hui, prix+FX historiques reels)")

    tickers = universe["ticker"].tolist() + list(BENCHMARKS.values())
    panel_cache = OUT_DIR / "price_panel_daily_cache.csv"
    if panel_cache.exists():
        print(f"Panel de prix quotidien trouve en cache ({panel_cache.name}), pas de re-telechargement.")
        daily_native = pd.read_csv(panel_cache, index_col=0, parse_dates=True)
    else:
        print(f"Telechargement de {LOOKBACK_MONTHS_TOTAL} mois de cloture QUOTIDIENNE pour {len(tickers)} tickers...")
        daily_native = download_daily_panel(tickers)
        daily_native.to_csv(panel_cache)
    print(f"Panel quotidien obtenu : {daily_native.shape[1]} tickers, {daily_native.shape[0]} jours de bourse")

    fx_cache = OUT_DIR / "fx_daily_cache.csv"
    currencies = set(universe["currency"].dropna().unique())
    if fx_cache.exists():
        print(f"Panel FX trouve en cache ({fx_cache.name}), pas de re-telechargement.")
        fx_daily = pd.read_csv(fx_cache, index_col=0, parse_dates=True)
    else:
        print(f"Telechargement des taux de change historiques pour {len(currencies)} devises...")
        fx_daily = download_fx_daily(currencies)
        fx_daily.to_csv(fx_cache)
    print(f"FX obtenu pour : {list(fx_daily.columns)}")

    currency_of = universe.set_index("ticker")["currency"].to_dict()
    daily_eur = to_eur_panel(daily_native, currency_of, fx_daily)

    monthly_native = daily_native.resample("ME").last()
    mom = compute_momentum(monthly_native)
    universe = universe[universe["ticker"].isin(daily_native.columns)].reset_index(drop=True)
    UNIVERSE_G = universe
    sec_mom = sector_momentum_series(mom, universe)

    valid_months = [d for d in monthly_native.index if mom.loc[d].notna().any()]
    dates = valid_months[-REBALANCE_MONTHS:]
    print(f"Fenetre de backtest : {dates[0].date()} -> {dates[-1].date()} ({len(dates)} points de rebalancement mensuels, "
          f"stop-loss verifie quotidiennement entre chaque point)\n")

    # as-of price snapshots at each of the 13 rebalance dates -- NOT an exact-match reindex:
    # several month-end labels (e.g. "2025-11-30", a Sunday) aren't real trading days, so a
    # plain reindex([date]) would silently return NaN for those months. See asof_panel().
    native_at_dates = asof_panel(daily_native, dates)
    eur_at_dates = asof_panel(daily_eur, dates)

    results = []

    closed1, open1 = run_bot1_blind(dates, mom, sec_mom, daily_native, native_at_dates)
    entry_info_1 = {}
    for _, r in closed1.iterrows():
        entry_info_1[r["ticker"]] = {"entry_date": r["entry_date"], "entry_price": r["entry_price"]}
    for _, r in open1.iterrows():
        entry_info_1[r["ticker"]] = {"entry_date": r["entry_date"], "entry_price": r["entry_price"]}
    eq1 = build_equity_curve_bot1(dates, closed1, native_at_dates, entry_info_1)
    results.append(summarize(BOTS["bot1_blind"]["label"], closed1, open1))
    closed1.to_csv(OUT_DIR / "bot1_blind_proxy_trades.csv", index=False)
    open1.to_csv(OUT_DIR / "bot1_blind_proxy_open.csv", index=False)

    b2 = BOTS["bot2_constrained"]
    closed2, open2, nav2, cash2 = run_slotted_bot(dates, mom, sec_mom, daily_native, daily_eur, currency_of,
                                                    native_at_dates, eur_at_dates,
                                                    b2["starting_slots"], b2["max_per_sector"], b2["mode"])
    results.append(summarize(b2["label"], closed2, open2, nav2, cash2))
    closed2.to_csv(OUT_DIR / "bot2_constrained_proxy_trades.csv", index=False)
    open2.to_csv(OUT_DIR / "bot2_constrained_proxy_open.csv", index=False)

    b3 = BOTS["bot3_large"]
    closed3, open3, nav3, cash3 = run_slotted_bot(dates, mom, sec_mom, daily_native, daily_eur, currency_of,
                                                    native_at_dates, eur_at_dates,
                                                    b3["starting_slots"], b3["max_per_sector"], b3["mode"])
    results.append(summarize(b3["label"], closed3, open3, nav3, cash3))
    closed3.to_csv(OUT_DIR / "bot3_large_proxy_trades.csv", index=False)
    open3.to_csv(OUT_DIR / "bot3_large_proxy_open.csv", index=False)

    bench_at_dates = asof_panel(daily_native[[tk for tk in BENCHMARKS.values() if tk in daily_native.columns]], dates)
    bench_curve = pd.DataFrame({name: bench_at_dates[tk] / bench_at_dates[tk].iloc[0] - 1
                                 for name, tk in BENCHMARKS.items() if tk in bench_at_dates.columns})
    curve_df = pd.DataFrame({"date": dates})
    curve_df["bot1_blind_avg_return"] = [eq1.get(d) for d in dates]
    curve_df["bot2_constrained_total_return"] = [(nav2.get(d) / STARTING_CAPITAL - 1) if d in nav2.index else None for d in dates]
    curve_df["bot3_large_total_return"] = [(nav3.get(d) / STARTING_CAPITAL - 1) if d in nav3.index else None for d in dates]
    for name in bench_curve.columns:
        curve_df[name] = bench_curve[name].values
    curve_df.to_csv(OUT_DIR / "equity_curve_proxy.csv", index=False)

    pd.DataFrame(results).to_csv(OUT_DIR / "summary_proxy.csv", index=False)
    print("Benchmarks sur la meme fenetre : " +
          ", ".join(f"{name}={bench_curve[name].iloc[-1]:+.1%}" for name in bench_curve.columns))
    print(f"\nResultats ecrits dans {OUT_DIR.relative_to(HERE)}/")


if __name__ == "__main__":
    main()
