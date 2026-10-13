"""Bot #45 Inities petites capitalisations (2026-10-13): buys US small caps after their directors/officers
bought shares on the open market, holds 6 months. Fifth single-idea bot (with #41-44), fills/accounting by
sleeve_book.py (close of the first session after the decision, total return).

Why small caps: the 2006-2026 test (SEC Form 3/4/5 data sets, scratchpad edge study of 2026-10-13) found
the classic insider-purchase effect gone on liquid names but alive where large funds can't trade -- names
trading $100k-$1M a day. Against the SAME stock at random dates with a matched prior-6-month return
(insiders buy after drops: -5.5% vs the S&P beforehand): +2.3% at 3 months, +5.0% at 6, +10.2% at 12, all
p<0.001, and still +3.8% (6m, p=0.02) / +9.7% (12m, p=0.001) over 2020-2026. Equal-weight book held 6
months: alpha vs that control +5.6%/yr (t=3.2); its excess return correlates -0.31 with the value proxy,
and adding it to the combined portfolio lifted the 2008-2026 Sharpe from 0.66 to 0.79. Caveats: firms that
later delisted are missing from Yahoo (2,768 events had no prices), and small caps have wider spreads.

Rules = the backtest's, fixed:
  - event = a (ticker, filing day) with open-market purchases (Form 4 code P) of common stock by a director
    or officer, >= MIN_VALUE_USD in total, price >= MIN_PRICE_USD, filed <= MAX_FILING_DELAY days after
    the trade (sec_form4.py, EDGAR daily index -- processed the morning after the filing day);
  - the stock trades MIN_ADV_USD to MAX_ADV_USD a day (20-session average up to the filing day), is quoted
    in USD, and Yahoo's close on the trade date is within x0.67-1.5 of the insider's price (ticker sanity);
  - buy POSITION_EUR at the close of the next session (= the first session after the filing day), hold
    HOLD_BDAYS business days (~6 months); a new event on a stock already held extends the holding instead
    of buying more (the backtest's calendar portfolio did the same);
  - at most MAX_POSITIONS lines: when full, new events are skipped (first come, first served -- the
    backtest held every event; this samples them at random in time, same expected edge, more noise).
Capital 5,000 EUR in 100 EUR lines: the effect is per event and small caps are noisy, so the book needs
breadth, and a 1 EUR sale fee costs 1% of a 100 EUR line (a 33 EUR line would pay 3% of a ~5% edge).
"""
# Alias lisibilite (mapping perso) : I1 -- strategie Inities petites capitalisations
import pathlib
import re
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.sec_form4 import purchases_for_day  # noqa: E402
from screener.sleeve_book import Book, now_utc  # noqa: E402

STARTING_CAPITAL = 5000.0
POSITION_EUR = 100.0
MAX_POSITIONS = 50
HOLD_BDAYS = 126
MIN_VALUE_USD = 25_000
MIN_PRICE_USD = 5.0
MAX_FILING_DELAY = 7
MIN_ADV_USD, MAX_ADV_USD = 100_000, 1_000_000
PRICE_SANITY = (0.67, 1.5)
FIRST_FILING_DAY = pd.Timestamp("2026-10-12")   # launch: filings from the last session before it
MAX_DAYS_PER_RUN = 3
NY = ZoneInfo("America/New_York")
TICKER_RE = re.compile(r"[A-Z]{1,5}([.\-][A-Z])?")


def events_from_rows(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # joint filers (a director and his holding company...) report the same trade in separate filings
    df = df.drop_duplicates(["ticker", "trans_date", "shares", "price"])
    df["trans_date"] = pd.to_datetime(df["trans_date"], errors="coerce")
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["delay"] = (df["filing_date"] - df["trans_date"]).dt.days
    df = df[(df["delay"] >= 0) & (df["delay"] <= MAX_FILING_DELAY) & (df["price"] >= MIN_PRICE_USD) & (df["value"] > 0)]
    df = df[df["ticker"].str.fullmatch(TICKER_RE)]
    if df.empty:
        return df
    ev = df.groupby(["ticker", "filing_date"]).agg(
        value=("value", "sum"), n_insiders=("owner_cik", "nunique"), trans_date=("trans_date", "min"),
        ins_price=("price", "median"), issuer=("issuer", "first"), title=("owner_title", "first")).reset_index()
    return ev[ev["value"] >= MIN_VALUE_USD]


def market_checks(ev: pd.DataFrame) -> pd.DataFrame:
    """Liquidity band, USD quote and price sanity, from Yahoo daily bars up to the filing day."""
    if ev.empty:
        return ev
    ytick = {t: t.replace(".", "-") for t in ev["ticker"]}
    raw = yf.download(sorted(set(ytick.values())), period="6mo", auto_adjust=False, progress=False, threads=True)
    out = []
    for r in ev.itertuples():
        yt = ytick[r.ticker]
        try:
            c = raw["Close"][yt].dropna() if isinstance(raw["Close"], pd.DataFrame) else raw["Close"].dropna()
            v = raw["Volume"][yt].dropna() if isinstance(raw["Volume"], pd.DataFrame) else raw["Volume"].dropna()
        except Exception:
            out.append((r.Index, "pas de cours Yahoo", None))
            continue
        c, v = c[c.index <= r.filing_date], v[v.index <= r.filing_date]
        if len(c) < 15:
            out.append((r.Index, "historique trop court", None))
            continue
        adv = float((c * v).tail(20).mean())
        at_trade = c[c.index <= r.trans_date]
        ratio = float(at_trade.iloc[-1] / r.ins_price) if len(at_trade) else np.nan
        if not (MIN_ADV_USD <= adv < MAX_ADV_USD):
            out.append((r.Index, f"liquidite hors bande ({adv / 1e6:.2f} M$/jour)", adv))
        elif not (PRICE_SANITY[0] <= ratio <= PRICE_SANITY[1]):
            out.append((r.Index, f"prix incoherent (x{ratio:.2f})", adv))
        else:
            out.append((r.Index, "retenu", adv))
    res = pd.DataFrame(out, columns=["i", "decision", "adv_usd"]).set_index("i")
    ev = ev.join(res)
    ev["yahoo"] = ev["ticker"].map(ytick)
    return ev


def main():
    book = Book("insider", "Bot #45 Inities petites capitalisations", STARTING_CAPITAL, {"pool": STARTING_CAPITAL})
    st = book.state
    st.setdefault("edgar_days_done", [])
    st.setdefault("hold_until", {})
    st.setdefault("events_log", [])
    st.setdefault("counters", {"jours": 0, "evenements": 0, "retenus": 0, "achetes": 0, "complet": 0, "prolonges": 0})
    book.load_prices()
    book.mark_to_market()
    book.execute_pending()
    book.mark_to_market()

    now = now_utc()
    ny_now = now.tz_convert(NY)
    today_ny = pd.Timestamp(ny_now.date())

    # 1. exits: holding period over (and forget the dates of buys that never filled)
    held = set(book.open_lots()["ticker"])
    pending_sells = {o["ticker"] for o in st["pending"] if o["side"] == "sell"}
    pending_buy_tickers = {o["ticker"] for o in st["pending"] if o["side"] == "buy"}
    st["hold_until"] = {t: d for t, d in st["hold_until"].items() if t in held or t in pending_buy_tickers}
    for t in sorted(held - pending_sells):
        until = st["hold_until"].get(t)
        if until and pd.Timestamp(until) <= today_ny:
            book.add_order("sell", t, "pool", f"detention de {HOLD_BDAYS} seances terminee", now, fraction=1.0)

    # 2. new filing days (the daily index of day D is complete the next morning, New York time)
    todo = [d for d in pd.bdate_range(FIRST_FILING_DAY, today_ny - pd.Timedelta(days=1))
            if d.strftime("%Y-%m-%d") not in st["edgar_days_done"]]
    if ny_now.hour < 6:
        todo = [d for d in todo if d < today_ny - pd.Timedelta(days=1)]
    for day in todo[:MAX_DAYS_PER_RUN]:
        rows = purchases_for_day(day)
        if rows is None:
            if (today_ny - day).days > 4:       # holiday: no index will ever come
                st["edgar_days_done"].append(day.strftime("%Y-%m-%d"))
            continue
        ev = market_checks(events_from_rows(rows))
        st["counters"]["jours"] += 1
        st["counters"]["evenements"] += len(ev)
        for r in ev.itertuples():
            decision = r.decision
            if decision == "retenu":
                st["counters"]["retenus"] += 1
                until = (r.filing_date + pd.offsets.BDay(HOLD_BDAYS + 1)).strftime("%Y-%m-%d")
                lots = book.open_lots()
                open_tickers = set(lots["ticker"])
                pending_buys = [o for o in st["pending"] if o["side"] == "buy"]
                if r.yahoo in open_tickers or r.yahoo in {o["ticker"] for o in pending_buys}:
                    st["hold_until"][r.yahoo] = max(st["hold_until"].get(r.yahoo, until), until)
                    st["counters"]["prolonges"] += 1
                    decision = "deja detenu : detention prolongee"
                else:
                    free = st["cash"]["pool"] - sum(o["amount_eur"] for o in pending_buys)
                    if len(open_tickers) + len(pending_buys) >= MAX_POSITIONS or free < POSITION_EUR:
                        st["counters"]["complet"] += 1
                        decision = "portefeuille complet : ignore"
                    else:
                        book.add_order("buy", r.yahoo, "pool",
                                       f"{r.n_insiders} initie(s), {r.value / 1e3:.0f} k$ le {r.trans_date.date()}",
                                       now, amount_eur=POSITION_EUR, name=r.issuer, sector="Inities (petites cap.)",
                                       country="United States", currency="USD", fractional=True,
                                       score=round(float(r.value), 0))
                        st["hold_until"][r.yahoo] = until
                        st["counters"]["achetes"] += 1
                        decision = "achat"
            st["events_log"].append({"filing_date": r.filing_date.strftime("%Y-%m-%d"), "ticker": r.ticker,
                                     "issuer": r.issuer, "value_usd": round(float(r.value)), "insiders": int(r.n_insiders),
                                     "title": r.title, "adv_usd": None if pd.isna(r.adv_usd) else round(float(r.adv_usd)),
                                     "decision": decision})
            print(f"    {r.ticker:6} {str(r.issuer)[:30]:30} {r.value / 1e3:7.0f} k$  -> {decision}")
        st["events_log"] = st["events_log"][-300:]
        st["edgar_days_done"].append(day.strftime("%Y-%m-%d"))
    st["edgar_days_done"] = sorted(set(st["edgar_days_done"]))   # ~250 a year, never trimmed: trimming
    # would make old days look unprocessed again (todo starts at FIRST_FILING_DAY)
    book.save({"counters": st["counters"], "last_edgar_day": (st["edgar_days_done"] or [None])[-1],
               "max_positions": MAX_POSITIONS, "position_eur": POSITION_EUR})


if __name__ == "__main__":
    main()
