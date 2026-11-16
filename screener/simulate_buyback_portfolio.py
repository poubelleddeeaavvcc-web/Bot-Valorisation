"""Bot #53 Rachats nets petites capitalisations (2026-11-16): US small caps whose share count shrank the most over 12
months (net buybacks) -- the net share issuance anomaly (Pontiff & Woodgate, Journal of Finance 2008; Daniel & Titman
2006). Single-idea bot like #41-51, fills/accounting by sleeve_book.py (close of the first session after the decision,
total return), paper only: an out-of-sample check of a backtest that survived but with caveats.

Backtest (2026-11-16, Desktop/Trading/sec_insiders/wave7.py and iss_robust.py, monthly 2010-2026, shares from the SEC
XBRL frames API point-in-time, universe = US stocks with $100k-1M/day traded and price >= $5): the most-shrinking
decile beat the universe by +7.6%/yr (t=2.2) in 2010-17 and +9.7%/yr (t=2.4) in 2018-26; ex-financials +9.0/+13.1%;
quarterly rebalance +9.5/+11.4%/yr (t 2.3/2.7). Caveats that make it a PAPER bot: within the same SIC2 sector 2010-17
it falls to +1.1%/yr, and among profitable firms only to +1.3/+6.6/+7.9% (2010-17/18-21/22-26, t<2) -- part of the
raw signal is a profitability and sector tilt; mid caps +3.4%/yr (t~1.5); buyback ETFs (PKW, SYLD) ~0 vs SPY.
Verdict rule fixed at launch: after ~4 quarters, compare the bot with the equal-weighted universe recorded at each
rebalance (state["rebalances"]); keep it only if ahead.

Rules (frozen 2026-11-16, same as the backtest's 'ex-financials, quarterly rebalance' line):
- universe: US-listed stocks (Nasdaq screener), last close >= $5, mean daily dollar volume over the last 60 sessions
  between $100k and $1M, SIC not 6xxx (financials; data/strategies/sic_map.json from the SEC financial statement data
  sets, unknown SIC kept);
- signal ISS = log(shares now / shares 12 months earlier), shares from the SEC frames (dei EntityCommonStockShares-
  Outstanding, else us-gaap CommonStockSharesOutstanding) usable 45 days after their date and at most 400 days old;
  ratios within 2% of a split factor or outside [0.5, 3] are dropped;
- portfolio: the 10% of the universe with the lowest ISS, equal weight, rebuilt on the first run of each calendar
  quarter (lines still selected are kept as they are, the others are sold, new ones bought).
Live deviations from the backtest universe (Form 4 issuers): US-headquartered filers only (frames 'loc' US-*), one
line per company, exchange-traded notes / preferreds dropped by name, empty SIC treated as financial. The live
universe is ~75 names (2026-11-16) vs 120-170 in the backtest, so the decile is ~7-8 lines.
5,000 EUR, 1 EUR fee on buys and sales.
"""
# Alias lisibilite (mapping perso) : N1 -- strategie Rachats nets petites capitalisations
import json
import pathlib
import re
import sys

import numpy as np
import pandas as pd
import requests
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener import sec_daily  # noqa: E402
from screener.sec_edgar import _load_ticker_cik_map  # noqa: E402
from screener.simulate_constrained_portfolio import TRADE_FEE_EUR  # noqa: E402
from screener.sleeve_book import Book, now_utc, quarter_key  # noqa: E402

STARTING_CAPITAL = 5000.0
DV_MIN, DV_MAX = 1e5, 1e6
MIN_PRICE = 5.0
DECILE = 0.10
SHARES_LAG_DAYS = 45
SHARES_MAX_AGE_DAYS = 400
MIN_UNIVERSE = 50           # live US-HQ ex-financials universe ~75 names (2026-11-16); fewer = data problem, retry
PREFILTER_CAP = (5e6, 5e9)  # Nasdaq screener market cap, only to keep the price download small
SPLIT_FACTORS = [2, 3, 4, 5, 10, 1.5, 4 / 3, 5 / 4, 20]
FRAME_URL = "https://data.sec.gov/api/xbrl/frames/{tax}/{tag}/shares/CY{y}Q{q}I.json"
SIC_MAP = HERE / "data/strategies/sic_map.json"
NOT_COMMON = re.compile(r"notes?|%|preferred|depositary|warrant|units?|rights?|debenture|due 20", re.I)


def nasdaq_rows() -> list:
    r = requests.get("https://api.nasdaq.com/api/screener/stocks",
                     params={"tableonly": "true", "limit": 10000, "offset": 0, "download": "true"},
                     headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=60)
    r.raise_for_status()
    return r.json()["data"]["rows"]


def _num(x):
    try:
        return float(str(x).replace("$", "").replace(",", ""))
    except ValueError:
        return np.nan


def share_frames(today: pd.Timestamp) -> pd.DataFrame:
    """cik, end, shares from the last ~9 calendar quarters of SEC frames (dei preferred over us-gaap)."""
    rows = []
    q0 = pd.Period(today, "Q")
    for k in range(1, 10):
        p = q0 - k
        for pri, (tax, tag) in enumerate([("dei", "EntityCommonStockSharesOutstanding"),
                                          ("us-gaap", "CommonStockSharesOutstanding")]):
            raw = sec_daily.get(FRAME_URL.format(tax=tax, tag=tag, y=p.year, q=p.quarter))
            if raw is None:
                continue
            for d in json.loads(raw)["data"]:
                rows.append((int(d["cik"]), pd.Timestamp(d["end"]), float(d["val"]), pri, d.get("loc") or ""))
    df = pd.DataFrame(rows, columns=["cik", "end", "sh", "pri", "loc"])
    return df.sort_values("pri").drop_duplicates(["cik", "end"]).drop(columns="pri")


def iss_by_cik(sh: pd.DataFrame, today: pd.Timestamp) -> pd.Series:
    """12-month log change in shares, US-headquartered filers only: the backtest universe came from Form 4 issuers,
    i.e. domestic filers (foreign private issuers file no Form 4, and their ADS / ordinary share counts get mixed)."""
    us = set(sh.loc[sh["loc"].str.startswith("US-"), "cik"])
    sh = sh[sh["cik"].isin(us)]
    def asof(m):
        x = sh[(sh["end"] + pd.Timedelta(days=SHARES_LAG_DAYS) <= m) & (sh["end"] >= m - pd.Timedelta(days=SHARES_MAX_AGE_DAYS))]
        return x.sort_values("end").groupby("cik")["sh"].last()
    now, prev = asof(today), asof(today - pd.DateOffset(months=12))
    r = (now / prev).dropna()
    r = r[(r > 0) & r.between(0.5, 3)]
    lr = np.log(r)
    split = pd.Series(False, index=lr.index)
    for k in SPLIT_FACTORS:
        split |= (np.abs(lr - np.log(k)) < 0.02) | (np.abs(lr + np.log(k)) < 0.02)
    return lr[~split]


def dollar_volume(tickers) -> pd.DataFrame:
    """last close and 60-session mean dollar volume, downloaded in chunks."""
    out = {}
    tickers = sorted(set(tickers))
    for i in range(0, len(tickers), 200):
        chunk = tickers[i:i + 200]
        try:
            d = yf.download(chunk, period="6mo", auto_adjust=False, progress=False, threads=True, group_by="ticker")
        except Exception as e:  # noqa: BLE001
            print(f"  echec telechargement {i}: {e}", file=sys.stderr)
            continue
        for t in chunk:
            try:
                x = d[t].dropna(subset=["Close"]) if len(chunk) > 1 else d.dropna(subset=["Close"])
            except KeyError:
                continue
            if len(x) < 40:
                continue
            dv = (x["Close"] * x["Volume"]).tail(60).mean()
            out[t] = (float(x["Close"].iloc[-1]), float(dv))
    return pd.DataFrame.from_dict(out, orient="index", columns=["close", "dv"])


def select(today: pd.Timestamp):
    rows = nasdaq_rows()
    t2c = {t.upper().replace(".", "-"): int(c) for t, c in _load_ticker_cik_map().items()}
    sic = json.loads(SIC_MAP.read_text(encoding="utf-8"))
    cand = {}
    for r in rows:
        t = str(r.get("symbol", "")).strip().upper().replace("/", "-").replace(".", "-")
        cap, px = _num(r.get("marketCap")), _num(r.get("lastsale"))
        if (not t or "^" in t or NOT_COMMON.search(str(r.get("name", "")))
                or not (PREFILTER_CAP[0] <= cap <= PREFILTER_CAP[1]) or not px >= MIN_PRICE):
            continue
        cik = t2c.get(t)
        if cik is None or str(sic.get(str(cik), "")).startswith("6"):
            continue
        cand[t] = cik
    iss = iss_by_cik(share_frames(today), today)
    cand = {t: c for t, c in cand.items() if c in iss.index}
    print(f"  {len(cand)} candidats (cotes US, hors financieres, nombre d'actions connu sur 12 mois)")
    px = dollar_volume(cand)
    # one line per company: its most traded share class, chosen BEFORE the volume band (else a thin class B can
    # stand in for a class A too liquid for the band -- KELYB on 2026-11-16)
    px["cik"] = [cand[t] for t in px.index]
    px = px.sort_values("dv", ascending=False).drop_duplicates("cik")
    uni = px[(px["close"] >= MIN_PRICE) & px["dv"].between(DV_MIN, DV_MAX)].copy()
    uni["iss"] = [iss[c] for c in uni["cik"]]
    # SIC missing from the map (BDCs, recent listings...): ask EDGAR, then drop financials
    for t in uni.index:
        c = int(uni.at[t, "cik"])
        if str(c) not in sic:
            raw = sec_daily.get(f"https://data.sec.gov/submissions/CIK{c:010d}.json")
            sic[str(c)] = str(json.loads(raw).get("sic") or "") if raw else ""
    # SIC still empty after EDGAR = business development companies / funds (WHF, BCIC, PFX on 2026-11-16): treated as
    # financials -- slightly stricter than the backtest, which kept unknown SIC
    uni = uni[~uni["cik"].map(lambda c: str(sic.get(str(int(c)), "")).startswith("6") or not sic.get(str(int(c))))]
    if len(uni) < MIN_UNIVERSE:
        return None, uni
    pick = uni[uni["iss"].rank(pct=True, method="first") <= DECILE].sort_values("iss")
    return pick, uni


def main():
    book = Book("buyback", "Bot #53 Rachats nets petites capitalisations", STARTING_CAPITAL, {"pool": STARTING_CAPITAL})
    book.buy_fee_eur = TRADE_FEE_EUR
    st = book.state
    st.setdefault("rebalances", {})
    book.load_prices()
    book.mark_to_market()
    book.execute_pending()
    book.mark_to_market()
    now = now_utc()
    today = pd.Timestamp(now.date())
    key = quarter_key(today)

    if st.get("rebalance_key") != key and not st["pending"]:
        pick, uni = select(today)
        if pick is None:
            print(f"Univers trop petit ({len(uni)} < {MIN_UNIVERSE}) : reessai au prochain passage")
        else:
            chosen = list(pick.index)
            held = set(book.open_lots()["ticker"])
            for t in sorted(held - set(chosen)):
                book.add_order("sell", t, "pool", "sorti du decile des plus forts rachats nets", now, fraction=1.0)
            equity = book.equity() - TRADE_FEE_EUR * len(held - set(chosen))
            amount = equity / len(chosen)
            for t in chosen:
                if t in held:
                    continue
                shrink = 1 - np.exp(pick.at[t, "iss"])
                book.add_order("buy", t, "pool", f"nombre d'actions -{shrink:.1%} sur 12 mois", now,
                               amount_eur=amount, name=t, sector="Rachats nets", country="United States",
                               currency="USD", fractional=True, score=float(pick.at[t, "iss"]))
            st["rebalances"][key] = {
                "date": today.strftime("%Y-%m-%d"),
                "selected": {t: round(float(pick.at[t, "iss"]), 5) for t in chosen},
                "universe": {t: round(float(uni.at[t, "close"]), 4) for t in uni.index},
            }
            st["rebalance_key"] = key
            print(f"Rebalancement {key} : univers {len(uni)} titres, {len(chosen)} retenus "
                  f"(baisse du nombre d'actions de {1 - np.exp(pick['iss'].max()):.1%} a {1 - np.exp(pick['iss'].min()):.1%})")
    book.save({"rebalance_key": st.get("rebalance_key"),
               "last_universe_size": len(st["rebalances"].get(st.get("rebalance_key"), {}).get("universe", {}))})


if __name__ == "__main__":
    main()
