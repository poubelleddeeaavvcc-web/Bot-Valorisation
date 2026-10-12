"""Bot #43 Tendance (2026-10-12): multi-asset trend following, the rule backtested over 2007-2026 on
2026-10-12 (CAGR ~5.9%, max drawdown ~-8%, vs S&P 500 11.0% / -51%; ~4.7%/yr after a 1 EUR fee per
order on 1,000 EUR): eight ETFs, each one held only while its month-end price is above its
SMA_MONTHS-month average, otherwise its sleeve sits in cash. It is Faber's published rule (2007),
deliberately untuned, so the live run is an out-of-sample test of the backtest rather than of a
parameter picked on the same data. Third single-idea bot with #41 Qualite / #42 Valeur, to be
combined later; fills/accounting in sleeve_book.py.

Mechanics, mirroring the backtest:
  - STARTING_CAPITAL split into one cash bucket ("sleeve") per ETF; a sleeve is either fully in its
    ETF or fully in cash, never lent to another sleeve;
  - decided at the first run of each month on the last COMPLETE month's adjusted close (the
    partial current month is dropped), filled at the next US close by sleeve_book;
  - every January, sleeves are brought back to equity/8: an invested sleeve more than
    REBALANCE_TOLERANCE away from it is trimmed or topped up, idle cash is redistributed.
US-listed ETFs because they carry the long history the signal and the backtest were built on; a
French investor would hold the UCITS equivalents (US-domiciled ETFs aren't sold to EU retail).
"""
# Alias lisibilite (mapping perso) : T1 -- strategie Tendance seule
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.sleeve_book import Book, fetch_daily, now_utc  # noqa: E402

ASSETS = {
    "SPY": ("SPDR S&P 500 ETF", "Actions US"),
    "EFA": ("iShares MSCI EAFE ETF", "Actions pays developpes hors US"),
    "EEM": ("iShares MSCI Emerging Markets ETF", "Actions emergentes"),
    "IEF": ("iShares 7-10 Year Treasury Bond ETF", "Obligations US 7-10 ans"),
    "TLT": ("iShares 20+ Year Treasury Bond ETF", "Obligations US 20 ans et plus"),
    "GLD": ("SPDR Gold Shares", "Or"),
    "DBC": ("Invesco DB Commodity Index Tracking Fund", "Matieres premieres"),
    "VNQ": ("Vanguard Real Estate ETF", "Immobilier US"),
}
STARTING_CAPITAL = 1000.0
SMA_MONTHS = 10
REBALANCE_MONTH = 1
REBALANCE_TOLERANCE = 0.10
MIN_ORDER_EUR = 2.0  # below this a sleeve's leftover cash isn't worth an order


def monthly_signals(now: pd.Timestamp) -> dict:
    _, adj = fetch_daily(list(ASSETS), now.tz_localize(None) - pd.Timedelta(days=430))
    month_start = pd.Timestamp(now.year, now.month, 1)
    monthly = adj[adj.index < month_start].resample("ME").last()
    out = {}
    for t in ASSETS:
        s = monthly[t].dropna() if t in monthly.columns else pd.Series(dtype=float)
        if len(s) < SMA_MONTHS:
            out[t] = None  # no signal -> leave the sleeve as it is this month
            continue
        last, sma = float(s.iloc[-1]), float(s.tail(SMA_MONTHS).mean())
        out[t] = {"month_end": s.index[-1].strftime("%Y-%m-%d"), "close": last, "sma": sma, "above": last > sma}
    return out


def decide(book: Book, now: pd.Timestamp, signals: dict):
    book.state["pending"] = []  # last month's unfilled orders are superseded
    cash = book.state["cash"]
    values = {t: book.position_value(ticker=t) for t in ASSETS}
    held = {t: bool((book.open_lots()["ticker"] == t).any()) for t in ASSETS}
    january = now.month == REBALANCE_MONTH
    target = book.equity() / len(ASSETS)
    orders, within = {}, set()
    for t in ASSETS:
        sig = signals.get(t)
        if sig is None:
            within.add(t)
            continue
        if sig["above"] and not held[t]:
            orders[t] = ("buy", None)  # amount = the sleeve's cash, known after reallocation
        elif not sig["above"] and held[t]:
            orders[t] = ("sell", 1.0)
        elif january and held[t]:
            dev = values[t] - target
            if dev > REBALANCE_TOLERANCE * target:
                orders[t] = ("sell", dev / values[t])
            elif dev < -REBALANCE_TOLERANCE * target:
                orders[t] = ("buy", -dev)
            else:
                within.add(t)
        else:
            within.add(t)

    if january:
        new_cash = {}
        for t in ASSETS:
            if t in within:
                new_cash[t] = cash.get(t, 0.0)
            elif held[t]:
                new_cash[t] = target - values[t]  # negative until a trim/sale's proceeds land
            else:
                new_cash[t] = target
        residual = sum(cash.values()) - sum(new_cash.values())
        flex = [t for t in ASSETS if not held[t]] or [t for t in ASSETS if t not in within] or list(ASSETS)
        for t in flex:
            new_cash[t] += residual / len(flex)
        book.state["cash"] = cash = new_cash
        print(f"  Reequilibrage annuel : cible {target:.2f} EUR par poche")

    for t, (side, x) in orders.items():
        name, label = ASSETS[t]
        sig = signals[t]
        why = f"cloture {sig['close']:.2f} {'>' if sig['above'] else '<='} moyenne {SMA_MONTHS} mois {sig['sma']:.2f}"
        if side == "sell":
            book.add_order("sell", t, t, ("reequilibrage annuel" if x < 1 else "sous sa moyenne") + f" ({why})",
                           now, fraction=x)
        else:
            amount = x if x is not None else cash.get(t, 0.0)
            if amount < MIN_ORDER_EUR:
                continue
            book.add_order("buy", t, t, ("reequilibrage annuel" if x is not None else "au-dessus de sa moyenne")
                           + f" ({why})", now, amount_eur=amount, name=name, sector=label,
                           country="United States", currency="USD", fractional=True, score=None)


def main():
    book = Book("trend", "Bot #43 Tendance", STARTING_CAPITAL,
                {t: STARTING_CAPITAL / len(ASSETS) for t in ASSETS})
    book.load_prices()
    book.mark_to_market()
    book.execute_pending()
    book.mark_to_market()

    now = now_utc()
    month = now.strftime("%Y-%m")
    if book.state.get("decision_month") != month:
        signals = monthly_signals(now)
        if all(s is None for s in signals.values()):
            print("Aucun signal calculable (telechargement en echec ?) -- decision reportee au prochain run.")
        else:
            print(f"Decision {month} : " + ", ".join(
                f"{t} {'IN' if s and s['above'] else 'OUT' if s else '?'}" for t, s in signals.items()))
            decide(book, now, signals)
            book.state["signals"] = signals
            book.state["decision_month"] = month
    book.save({"decision_month": book.state.get("decision_month"),
               "invested": [t for t in ASSETS if (book.open_lots()["ticker"] == t).any()]})


if __name__ == "__main__":
    main()
