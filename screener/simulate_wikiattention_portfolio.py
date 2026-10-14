"""Bot #47 Attention Wikipedia (2026-10-14): market timing on the views of the Wikipedia pages of the 30 Dow Jones
companies (Moat, Curme, ..., Preis, Scientific Reports 2013: page views rise before market falls). Seventh
single-idea bot (with #41-46), fills/accounting by sleeve_book.py (close of the first session after the decision).

The study (2007-2012): for each Dow page, n(t) = views of week t (weeks end on Sunday); if n(t) is above the mean
of the 3 previous weeks, be SHORT the Dow from the close of the first trading day of week t+1 to that of week t+2,
else LONG; mean result 0.50 standard deviations above random strategies (p=0.005).
Our re-test on 2015-10 -> 2026-09, i.e. AFTER publication (scratchpad audit_wiki.py, 2026-10-14): the 30 page
strategies are still +0.39 standard deviations above random ones (18/30 positive); without short selling, on the
summed views of the 30 pages: invested 55 % of the weeks, 12.6 %/yr, Sharpe 0.95, max drawdown -25 % vs SPY
15.0 %/yr, Sharpe 0.91, -33 % -- weeks in the market +41-49 bp vs +8-15 bp out, in 2015-20 and in 2021-26; ~10
sales a year (1 %/yr of fees on 1,000 EUR). Fragile point: the study's own aggregation (average of the 30 page
strategies) gives Sharpe 0.82 in long/cash form, below buy-and-hold -- this bot uses the summed views.

Rules, fixed: once a week (first run on the US first trading day of the week, before its close), sum the
user views of the 30 pages over the last full Monday-Sunday week (Wikimedia REST API) and compare with the mean of
the 3 previous weeks; views down or equal -> hold SPY with all the capital, views up -> cash. The order fills at
that day's close, as in the study. 1,000 EUR, like #43/#44.
"""
# Alias lisibilite (mapping perso) : W1 -- strategie Attention Wikipedia
import pathlib
import sys
import time
import urllib.parse
from zoneinfo import ZoneInfo

import pandas as pd
import requests

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.sec_edgar import SEC_HEADERS  # noqa: E402  (same identified User-Agent for Wikimedia)
from screener.sleeve_book import Book, now_utc  # noqa: E402

STARTING_CAPITAL = 1000.0
INSTRUMENT = "SPY"
LOOKBACK_WEEKS = 3
NY = ZoneInfo("America/New_York")
API = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{}/daily/{}/{}"
# the 30 Dow Jones companies (October 2026) and their English Wikipedia articles, as in the backtest
DOW_PAGES = {
    "AAPL": "Apple_Inc.", "AMGN": "Amgen", "AMZN": "Amazon_(company)", "AXP": "American_Express", "BA": "Boeing",
    "CAT": "Caterpillar_Inc.", "CRM": "Salesforce", "CSCO": "Cisco", "CVX": "Chevron_Corporation",
    "DIS": "The_Walt_Disney_Company", "GS": "Goldman_Sachs", "HD": "Home_Depot", "HON": "Honeywell", "IBM": "IBM",
    "JNJ": "Johnson_&_Johnson", "JPM": "JPMorgan_Chase", "KO": "The_Coca-Cola_Company", "MCD": "McDonald's", "MMM": "3M",
    "MRK": "Merck_&_Co.", "MSFT": "Microsoft", "NKE": "Nike,_Inc.", "NVDA": "Nvidia", "PG": "Procter_&_Gamble",
    "SHW": "Sherwin-Williams", "TRV": "The_Travelers_Companies", "UNH": "UnitedHealth_Group", "V": "Visa_Inc.",
    "VZ": "Verizon", "WMT": "Walmart",
}


def weekly_views(last_sunday: pd.Timestamp):
    """Summed daily user views of the 30 pages for the LOOKBACK_WEEKS + 1 weeks ending last_sunday, oldest first,
    or None when a page is missing (the API publishes a day's views within ~24 h)."""
    start = last_sunday - pd.Timedelta(days=7 * (LOOKBACK_WEEKS + 1) - 1)
    total = pd.Series(0.0, index=pd.date_range(start, last_sunday))
    s = requests.Session(); s.headers.update(SEC_HEADERS)
    for ticker, art in DOW_PAGES.items():
        url = API.format(urllib.parse.quote(art, safe=""), start.strftime("%Y%m%d"), last_sunday.strftime("%Y%m%d"))
        for k in range(3):
            try:
                r = s.get(url, timeout=30)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(5 * (k + 1))
        else:
            print(f"  vues indisponibles pour {art}")
            return None
        v = pd.Series({pd.Timestamp(x["timestamp"][:8]): x["views"] for x in r.json().get("items", [])}, dtype=float)
        if v.reindex(total.index).isna().any():
            print(f"  semaine incomplete pour {art} ({v.reindex(total.index).isna().sum()} jour(s) manquant(s))")
            return None
        total += v.reindex(total.index)
        time.sleep(0.2)
    return total.groupby((total.index - start).days // 7).sum().tolist()


def main():
    book = Book("wikiattention", "Bot #47 Attention Wikipedia", STARTING_CAPITAL, {"pool": STARTING_CAPITAL})
    st = book.state
    st.setdefault("signals", [])
    book.load_prices([INSTRUMENT])
    book.mark_to_market()
    book.execute_pending()
    book.mark_to_market()

    now = now_utc()
    ny = now.tz_convert(NY)
    week_key = f"{ny.isocalendar().year}-S{ny.isocalendar().week:02d}"
    # decide once a week, at the first run of a weekday before 16:00 New York (normally Monday, once Sunday's views
    # are published): the order fills at that day's close -- the close of the week's first trading day, as in the study
    if st.get("week_key") != week_key and ny.weekday() <= 4 and ny.hour < 16:
        last_sunday = pd.Timestamp((ny - pd.Timedelta(days=ny.weekday() + 1)).date())
        w = weekly_views(last_sunday)
        if w is not None:
            n_t, prev = w[-1], sum(w[:-1]) / LOOKBACK_WEEKS
            invest = n_t <= prev
            held = INSTRUMENT in set(book.open_lots()["ticker"])
            pending = {o["ticker"] for o in st["pending"]}
            change = n_t / prev - 1
            if invest and not held and INSTRUMENT not in pending:
                book.add_order("buy", INSTRUMENT, "pool", f"vues des pages du Dow {change:+.1%} vs 3 semaines : investi", now,
                               amount_eur=st["cash"]["pool"], name="SPDR S&P 500 ETF", sector="ETF actions US",
                               country="United States", currency="USD", fractional=True, score=round(change, 4))
            elif not invest and held:
                book.add_order("sell", INSTRUMENT, "pool", f"vues des pages du Dow {change:+.1%} vs 3 semaines : cash", now, fraction=1.0)
            st["signals"] = (st["signals"] + [{"week": week_key, "views": round(n_t), "mean_prev3": round(prev), "change": round(change, 4),
                                               "position": "investi" if invest else "cash"}])[-260:]
            st["week_key"] = week_key
            print(f"Semaine {week_key} : vues {n_t:,.0f} vs moyenne 3 semaines {prev:,.0f} ({change:+.1%}) -> "
                  f"{'investi' if invest else 'cash'}")
    last = st["signals"][-1] if st["signals"] else None
    book.save({"week_key": st.get("week_key"), "position": last["position"] if last else None,
               "last_change": last["change"] if last else None})


if __name__ == "__main__":
    main()
