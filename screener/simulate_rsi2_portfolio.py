"""Bot #44 Retour a la moyenne (2026-10-12): Connors' RSI(2) rule on the S&P 500, untouched -- buy when
RSI(2) < RSI_ENTRY and the index is above its SMA_TREND-day average, sell when it closes back above
its SMA_EXIT-day average. Holds a few days, ~8 round trips a year, invested ~11% of the time.

Edge test of 2026-10-12 (scratchpad, see memory "edge tests"): each real trade against random trades
of the same length in the same regime -- S&P 500 1994-2008 p<0.001, 2009-2026 p=0.006, last 10 years
p=0.015; significant on Nasdaq 100, Dow, Euro Stoxx 50, CAC 40, FTSE with the same rule; all 36
neighbouring parameter sets positive. But NO edge 1960-1993: a property of today's markets, which can
fade. Net of fees on 1,000 EUR: ~4.4%/yr (3.8% when filled a day late) vs 1.4% for cash.

Timing: the rule trades at the close. The decision is taken during the last hour of the US session
(DECISION_HOUR, New York time) with the live price standing in for the close, and sleeve_book fills it
at that session's close. A day whose last-hour run is missed (CI delay) is simply skipped -- the signal
usually lasts more than a day. SPY for the data and the fills (as in the backtest); a French investor
would use a UCITS S&P 500 ETF. All-in or all-out, one cash bucket.
"""
# Alias lisibilite (mapping perso) : R1 -- strategie Retour a la moyenne seule
import pathlib
import sys
from zoneinfo import ZoneInfo

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.sleeve_book import Book, fetch_daily, now_utc  # noqa: E402

TICKER = "SPY"
STARTING_CAPITAL = 1000.0
RSI_PERIOD, RSI_ENTRY = 2, 10
SMA_TREND, SMA_EXIT = 200, 5
DECISION_HOUR = 15  # 15:00-15:59 New York, the session's last hour
NY = ZoneInfo("America/New_York")


def rsi(closes: pd.Series, n: int = RSI_PERIOD) -> pd.Series:
    d = closes.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    return 100 - 100 / (1 + up.ewm(alpha=1 / n, adjust=False).mean() / dn.ewm(alpha=1 / n, adjust=False).mean())


def main():
    book = Book("rsi2", "Bot #44 Retour a la moyenne (RSI2)", STARTING_CAPITAL, {TICKER: STARTING_CAPITAL})
    book.load_prices()
    book.mark_to_market()
    book.execute_pending()
    book.mark_to_market()

    now = now_utc()
    ny = now.tz_convert(NY)
    today = ny.strftime("%Y-%m-%d")
    signal = book.state.get("signal")
    if ny.weekday() < 5 and ny.hour == DECISION_HOUR and book.state.get("decision_date") != today:
        closes, _ = fetch_daily([TICKER], now.tz_localize(None) - pd.Timedelta(days=420))
        s = closes[TICKER].dropna() if TICKER in closes.columns else pd.Series(dtype=float)
        # the decision needs TODAY's live bar (the price standing in for the close); without it
        # (holiday, data lag) the day is skipped
        if len(s) > SMA_TREND and s.index[-1].strftime("%Y-%m-%d") == today:
            r, sma_t, sma_x = float(rsi(s).iloc[-1]), float(s.tail(SMA_TREND).mean()), float(s.tail(SMA_EXIT).mean())
            last = float(s.iloc[-1])
            held = bool((book.open_lots()["ticker"] == TICKER).any())
            signal = {"date": today, "price": last, "rsi2": round(r, 2), "sma200": round(sma_t, 2),
                      "sma5": round(sma_x, 2), "held": held}
            print(f"Decision {today} : SPY {last:.2f}, RSI2 {r:.1f}, SMA200 {sma_t:.2f}, SMA5 {sma_x:.2f}, "
                  f"{'en position' if held else 'en cash'}")
            if not held and r < RSI_ENTRY and last > sma_t:
                amount = book.state["cash"].get(TICKER, 0.0)
                if amount > 1:
                    book.add_order("buy", TICKER, TICKER, f"RSI2 {r:.1f} < {RSI_ENTRY} et cours > moyenne {SMA_TREND} j",
                                   now, amount_eur=amount, name="SPDR S&P 500 ETF", sector="Actions US",
                                   country="United States", currency="USD", fractional=True, score=round(r, 2))
            elif held and last > sma_x:
                book.add_order("sell", TICKER, TICKER, f"cours {last:.2f} > moyenne {SMA_EXIT} j {sma_x:.2f}",
                               now, fraction=1.0)
            book.state["decision_date"] = today
            book.state["signal"] = signal
        else:
            print(f"Pas de cours du jour pour {TICKER} ({today}) -- decision sautee")
    book.save({"signal": signal, "invested": bool((book.open_lots()["ticker"] == TICKER).any())})


if __name__ == "__main__":
    main()
