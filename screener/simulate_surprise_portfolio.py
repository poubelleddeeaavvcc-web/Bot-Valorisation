"""Bot #54 Surprise de benefices petites capitalisations (2026-11-16): post-earnings-announcement drift (Bernard &
Thomas 1989; Foster, Olsen & Shevlin 1984) on US small caps -- stocks whose last quarterly EPS beat the same quarter a
year earlier by the most, relative to how much that change usually varies (SUE), keep outperforming for a few weeks.
Single-idea paper bot like #41-53, fills/accounting by sleeve_book.py (close of the first session after the decision,
total return).

Backtest (2026-11-16, Desktop/Trading/sec_insiders/wave8.py and wave8_lag.py, monthly 2010-2026, quarterly EPS from the
SEC XBRL frames API, universe = US stocks with $100k-1M/day traded and price >= $5): top SUE decile minus the universe
+10.7%/yr (t=2.3) in 2010-17 and +10.5%/yr (t=2.0) in 2018-26; with a 75-day availability lag +10.8/+11.1%/yr. The drift
fades within months: a 3-month hold gave only +3.7%/yr in 2010-17, so the book is rebuilt every month. ~17 names, 37%
monthly turnover: 1 EUR fees (~3%/yr on 5,000 EUR) and small-cap spreads (~4%/yr) leave roughly +3-4%/yr net. The same
signal on the S&P 500 failed (2026-10-12): the drift only lives where few analysts look. Correlation of its monthly
excess with #53 Rachats nets 0.36.
Verdict rule fixed at launch: after ~12 monthly rebalances, compare the bot with the equal-weighted universe recorded at
each rebalance (state["rebalances"]); keep it only if ahead after costs.

Rules (frozen 2026-11-16, same as the backtest's 'SMALL, SUE, 45-day lag' line):
- universe: as #53 (simulate_buyback_portfolio.py) -- Nasdaq-screener US stocks, last close >= $5, 60-session mean
  dollar volume $100k-1M, US-headquartered filers, one line per company, notes / preferreds dropped -- but financials
  KEPT (the backtest line that passed included them);
- SUE = (EPS_q - EPS_{q-4}) / standard deviation of the previous 8 such seasonal changes (at least 6), EPS = us-gaap
  EarningsPerShareBasic quarterly frames, quarter ends matched by date (+-40 days); a quarter counts 45 days after its
  end and only while its end is <= 135 days old;
- portfolio: the 10% of the universe with the highest SUE, equal weight, rebuilt on the first run of each month
  (lines still selected are kept, the others sold, new ones bought). 5,000 EUR, 1 EUR fee on buys and sales.
"""
# Alias lisibilite (mapping perso) : S1 -- strategie Surprise de benefices petites capitalisations
import json
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener import sec_daily  # noqa: E402
from screener.sec_edgar import _load_ticker_cik_map  # noqa: E402
from screener.simulate_buyback_portfolio import (  # noqa: E402
    DV_MAX, DV_MIN, MIN_PRICE, NOT_COMMON, PREFILTER_CAP, _num, dollar_volume, nasdaq_rows,
)
from screener.simulate_constrained_portfolio import TRADE_FEE_EUR  # noqa: E402
from screener.sleeve_book import Book, now_utc  # noqa: E402

STARTING_CAPITAL = 5000.0
DECILE = 0.10
LAG_DAYS = 45
MAX_AGE_DAYS = 135
MIN_UNIVERSE = 50
FRAME_URL = "https://data.sec.gov/api/xbrl/frames/us-gaap/EarningsPerShareBasic/USD-per-shares/CY{y}Q{q}.json"


def eps_frames(today: pd.Timestamp) -> pd.DataFrame:
    """cik, end, eps, loc for the last 14 calendar quarters (enough for 8 seasonal changes + the current one)."""
    rows = []
    q0 = pd.Period(today, "Q")
    for k in range(1, 15):
        p = q0 - k
        raw = sec_daily.get(FRAME_URL.format(y=p.year, q=p.quarter))
        if raw is None:
            continue
        for d in json.loads(raw)["data"]:
            rows.append((int(d["cik"]), pd.Timestamp(d["end"]), float(d["val"]), d.get("loc") or ""))
    return pd.DataFrame(rows, columns=["cik", "end", "eps", "loc"]).drop_duplicates(["cik", "end"], keep="last")


def sue_by_cik(eps: pd.DataFrame, today: pd.Timestamp) -> pd.Series:
    """Latest usable SUE per US-headquartered company (same computation as the backtest's wave8.surprise)."""
    us = set(eps.loc[eps["loc"].str.startswith("US-"), "cik"])
    out = {}
    for cik, g in eps[eps["cik"].isin(us)].sort_values("end").groupby("cik"):
        s = g.set_index("end")["eps"]
        ends = s.index
        diffs = {}
        for e in ends:
            prev = ends[(ends >= e - pd.Timedelta(days=405)) & (ends <= e - pd.Timedelta(days=325))]
            if len(prev):
                diffs[e] = s[e] - s[prev[-1]]
        d = pd.Series(diffs).sort_index()
        usable = [e for e in d.index if e + pd.Timedelta(days=LAG_DAYS) <= today and (today - e).days <= MAX_AGE_DAYS]
        if not usable:
            continue
        i = d.index.get_loc(usable[-1])
        hist = d.iloc[max(0, i - 8):i]
        if len(hist) < 6 or not hist.std() > 0:
            continue
        out[cik] = float(d.iloc[i] / hist.std())
    return pd.Series(out, dtype=float)


def select(today: pd.Timestamp):
    t2c = {t.upper().replace(".", "-"): int(c) for t, c in _load_ticker_cik_map().items()}
    sue = sue_by_cik(eps_frames(today), today)
    cand = {}
    for r in nasdaq_rows():
        t = str(r.get("symbol", "")).strip().upper().replace("/", "-").replace(".", "-")
        cap, px = _num(r.get("marketCap")), _num(r.get("lastsale"))
        if (not t or "^" in t or NOT_COMMON.search(str(r.get("name", "")))
                or not (PREFILTER_CAP[0] <= cap <= PREFILTER_CAP[1]) or not px >= MIN_PRICE):
            continue
        cik = t2c.get(t)
        if cik is not None and cik in sue.index:
            cand[t] = cik
    print(f"  {len(cand)} candidats (cotes US, BPA trimestriel recent et historique suffisant)")
    px = dollar_volume(cand)
    # one line per company: its most traded share class, chosen BEFORE the volume band (else a thin class B can
    # stand in for a class A too liquid for the band -- KELYB on 2026-11-16)
    px["cik"] = [cand[t] for t in px.index]
    px = px.sort_values("dv", ascending=False).drop_duplicates("cik")
    uni = px[(px["close"] >= MIN_PRICE) & px["dv"].between(DV_MIN, DV_MAX)].copy()
    uni["sue"] = [sue[c] for c in uni["cik"]]
    if len(uni) < MIN_UNIVERSE:
        return None, uni
    pick = uni[uni["sue"].rank(pct=True, method="first") > 1 - DECILE].sort_values("sue", ascending=False)
    return pick, uni


def main():
    book = Book("surprise", "Bot #54 Surprise de benefices petites capitalisations", STARTING_CAPITAL,
                {"pool": STARTING_CAPITAL})
    book.buy_fee_eur = TRADE_FEE_EUR
    st = book.state
    st.setdefault("rebalances", {})
    book.load_prices()
    book.mark_to_market()
    book.execute_pending()
    book.mark_to_market()
    now = now_utc()
    today = pd.Timestamp(now.date())
    key = today.strftime("%Y-%m")

    if st.get("rebalance_key") != key and not st["pending"]:
        pick, uni = select(today)
        if pick is None:
            print(f"Univers trop petit ({len(uni)} < {MIN_UNIVERSE}) : reessai au prochain passage")
        else:
            chosen = list(pick.index)
            held = set(book.open_lots()["ticker"])
            for t in sorted(held - set(chosen)):
                book.add_order("sell", t, "pool", "sorti du decile des plus fortes surprises", now, fraction=1.0)
            equity = book.equity() - TRADE_FEE_EUR * len(held - set(chosen))
            amount = equity / len(chosen)
            for t in chosen:
                if t in held:
                    continue
                book.add_order("buy", t, "pool", f"surprise de BPA {pick.at[t, 'sue']:+.1f} ecarts-types", now,
                               amount_eur=amount, name=t, sector="Surprise de benefices", country="United States",
                               currency="USD", fractional=True, score=float(pick.at[t, "sue"]))
            st["rebalances"][key] = {
                "date": today.strftime("%Y-%m-%d"),
                "selected": {t: round(float(pick.at[t, "sue"]), 3) for t in chosen},
                "universe": {t: round(float(uni.at[t, "close"]), 4) for t in uni.index},
            }
            st["rebalance_key"] = key
            print(f"Rebalancement {key} : univers {len(uni)} titres, {len(chosen)} retenus "
                  f"(SUE de {pick['sue'].min():+.1f} a {pick['sue'].max():+.1f})")
    book.save({"rebalance_key": st.get("rebalance_key"),
               "last_universe_size": len(st["rebalances"].get(st.get("rebalance_key"), {}).get("universe", {}))})


if __name__ == "__main__":
    main()
