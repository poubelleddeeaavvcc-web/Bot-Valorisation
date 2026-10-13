"""Shared order/accounting engine of the single-strategy bots (2026-10-12): #41 Qualite
(simulate_quality_portfolio.py), #42 Valeur (simulate_value_portfolio.py) and #43 Tendance
(simulate_trend_portfolio.py), then #44 Retour a la moyenne (simulate_rsi2_portfolio.py) and #45 Inities petites capitalisations
(simulate_insider_portfolio.py). Per the user (2026-10-12): the ~40 earlier bots are one strategy
(value + momentum screener) in many variants, so these three each test ONE independent idea, to be
combined later once each has its own track record.

Two things differ on purpose from the earlier bots, which fill at the last price Yahoo returns
whatever the hour (simulate_constrained_portfolio.fill_slots):

1. Fills at the CLOSE OF THE FIRST SESSION ENDING AFTER THE DECISION -- what a market-on-close
   order placed at decision time would get. Bot #40 (D10) bought B3SA3.SA on Sunday 2026-10-04 at
   Friday's close (19.00 BRL) and booked the +16% Monday gap (open 22.10) as its own gain, a fill
   no real order could have had. Here a decision only records a pending order; a later run books
   it once that session's daily bar is final (close + SETTLE_DELAY). Days without a bar (weekends,
   holidays) are skipped naturally.

2. Values positions in TOTAL RETURN: a lot is worth entry_cost_local * AdjClose(now) /
   AdjClose(entry day), both read from the SAME download each run (Yahoo rescales the whole
   adjusted series at every dividend/split, so a ratio taken within one fetch is the only stable
   quantity). Dividends are thus reinvested in the line, pre-tax, and splits need no special case.
   The earlier bots compare raw prices, so they leave dividends out -- a value strategy (high
   yields) would be penalised ~3-4 pts/year, and the trend bot's bond ETFs even more.

Cash lives in named buckets: "pool" for #41/#42, one bucket per ETF for #43 (its sleeves). Fees
follow the repo rule (simulate_constrained_portfolio.TRADE_FEE_EUR): 1 EUR per sale, nothing on
buys. No interest is credited on idle cash.
"""
import json
import pathlib
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from screener.market_hours import market_of  # noqa: E402
from screener.simulate_constrained_portfolio import (  # noqa: E402
    FX_PAIR, TRADE_FEE_EUR, fetch_fx_rates, to_eur,
)

SETTLE_DELAY = pd.Timedelta(hours=1)  # MARKET_HOURS uses whole hours (Paris "17" = 17:30 real close)
ORDER_EXPIRY = pd.Timedelta(days=10)  # a buy still unfilled after this (no data, delisted...) is dropped
SUSPECT_JUMP = 3.0                    # same bar as simulate_portfolio.MAX_PLAUSIBLE_PRICE_RATIO

LEDGER_COLUMNS = [
    "ticker", "name", "sector", "country", "status", "currency", "fractional", "bucket",
    "entry_decided_at", "entry_date", "entry_price", "shares", "entry_value_eur", "entry_cost_local",
    "entry_score", "last_check_date", "last_price", "current_value_eur", "unrealized_return_pct",
    "exit_decided_at", "exit_date", "exit_price", "exit_reason", "exit_value_eur", "return_pct",
    "holding_days",
]


def from_eur(amount_eur, currency, fx_rates):
    """Inverse of simulate_constrained_portfolio.to_eur."""
    if currency is None or pd.isna(currency) or currency == "EUR":
        return amount_eur
    rate = fx_rates.get("GBP" if currency == "GBp" else currency)
    if rate is None or rate == 0:
        return None
    local = amount_eur * rate
    return local * 100 if currency == "GBp" else local


def now_utc() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def session_close_utc(ticker: str, day) -> pd.Timestamp:
    tz_name, _, close_hour, _ = market_of(ticker)
    local = pd.Timestamp(pd.Timestamp(day).date()) + pd.Timedelta(hours=close_hour)
    return local.tz_localize(ZoneInfo(tz_name)).tz_convert("UTC")


def fetch_daily(tickers, start) -> tuple:
    """(close, adj_close) DataFrames, one column per ticker, index = session date."""
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    try:
        raw = yf.download(tickers, start=pd.Timestamp(start).strftime("%Y-%m-%d"), auto_adjust=False,
                          progress=False, threads=True)
    except Exception as e:
        print(f"  echec telechargement des cours : {e}", file=sys.stderr)
        return pd.DataFrame(), pd.DataFrame()
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    close, adj = raw["Close"], raw["Adj Close"]
    if isinstance(close, pd.Series):  # older yfinance, single ticker
        close, adj = close.to_frame(tickers[0]), adj.to_frame(tickers[0])
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    adj.index = close.index
    return close, adj


def execution_day(ticker: str, decided_at: pd.Timestamp, closes: pd.Series, now: pd.Timestamp):
    """Session date whose close fills an order decided at decided_at, or None if that session
    hasn't closed yet / has no data yet (the order stays pending)."""
    for day in closes.dropna().index:
        close_at = session_close_utc(ticker, day)
        if close_at > decided_at:
            return day if now >= close_at + SETTLE_DELAY else None
    return None


class Book:
    """Ledger + state + summary + equity curve of one bot, all under results/simulation/."""

    def __init__(self, stem: str, label: str, capital: float, buckets: dict):
        sim = HERE / "results/simulation"
        self.ledger_path = sim / f"{stem}_portfolio_ledger.csv"
        self.state_path = sim / f"{stem}_state.json"
        self.summary_path = sim / f"{stem}_summary.json"
        self.curve_path = sim / f"{stem}_equity_curve.csv"
        self.label, self.capital = label, capital
        if self.ledger_path.exists():
            df = pd.read_csv(self.ledger_path)
            for c in LEDGER_COLUMNS:
                if c not in df.columns:
                    df[c] = None
            self.ledger = df[LEDGER_COLUMNS].astype(object)
        else:
            self.ledger = pd.DataFrame(columns=LEDGER_COLUMNS, dtype=object)
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {"cash": dict(buckets), "pending": [], "launched_at": now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")}
        self.fx = fetch_fx_rates(set(FX_PAIR.keys()))
        self.close = self.adj = pd.DataFrame()

    # ---- data -------------------------------------------------------------------------------
    def open_lots(self) -> pd.DataFrame:
        return self.ledger[self.ledger["status"] == "open"]

    def load_prices(self, extra_tickers=()):
        lots = self.open_lots()
        tickers = set(lots["ticker"]) | {o["ticker"] for o in self.state["pending"]} | set(extra_tickers)
        starts = [pd.Timestamp(d) for d in lots["entry_date"].dropna()]
        start = min(starts + [now_utc().tz_localize(None).normalize() - pd.Timedelta(days=20)]) - pd.Timedelta(days=5)
        self.close, self.adj = fetch_daily(tickers, start)

    def _series(self, frame, ticker):
        return frame[ticker].dropna() if ticker in frame.columns else pd.Series(dtype=float)

    def lot_tr(self, idx, day=None):
        """Total-return factor of a lot from its entry session to `day` (default: last bar)."""
        ticker = self.ledger.at[idx, "ticker"]
        adj = self._series(self.adj, ticker)
        entry = pd.Timestamp(self.ledger.at[idx, "entry_date"])
        if entry not in adj.index or not len(adj):
            return None
        end = adj.loc[:day].iloc[-1] if day is not None else adj.iloc[-1]
        return float(end / adj.loc[entry])

    def bucket_of(self, ticker):
        lots = self.open_lots()
        rows = lots[lots["ticker"] == ticker]
        return rows["bucket"].iloc[0] if len(rows) else None

    # ---- marking ----------------------------------------------------------------------------
    def mark_to_market(self):
        for idx in self.open_lots().index:
            ticker = self.ledger.at[idx, "ticker"]
            tr = self.lot_tr(idx)
            if tr is None:
                continue
            prev = self.ledger.at[idx, "unrealized_return_pct"]
            prev_tr = 1 + float(prev) if prev is not None and pd.notna(prev) else 1.0
            if prev_tr > 0 and not (1 / SUSPECT_JUMP <= tr / prev_tr <= SUSPECT_JUMP):
                print(f"  COURS SUSPECT {ticker} : facteur {prev_tr:.3f} -> {tr:.3f}, ligne non reevaluee ce run",
                      file=sys.stderr)
                continue
            value = to_eur(float(self.ledger.at[idx, "entry_cost_local"]) * tr, self.ledger.at[idx, "currency"], self.fx)
            if value is None:
                continue
            closes = self._series(self.close, ticker)
            self.ledger.at[idx, "last_check_date"] = closes.index[-1].strftime("%Y-%m-%d")
            self.ledger.at[idx, "last_price"] = float(closes.iloc[-1])
            self.ledger.at[idx, "current_value_eur"] = value
            self.ledger.at[idx, "unrealized_return_pct"] = tr - 1

    def position_value(self, ticker=None, bucket=None) -> float:
        lots = self.open_lots()
        if ticker is not None:
            lots = lots[lots["ticker"] == ticker]
        if bucket is not None:
            lots = lots[lots["bucket"] == bucket]
        return float(pd.to_numeric(lots["current_value_eur"], errors="coerce").fillna(0).sum())

    def equity(self) -> float:
        return sum(self.state["cash"].values()) + self.position_value()

    # ---- orders -----------------------------------------------------------------------------
    def add_order(self, side, ticker, bucket, reason, decided_at, **kw):
        order = {"side": side, "ticker": ticker, "bucket": bucket, "reason": reason,
                 "decided_at": decided_at.strftime("%Y-%m-%dT%H:%M:%SZ"), **kw}
        self.state["pending"].append(order)
        what = (f"{kw.get('amount_eur', 0):.2f} EUR" if side == "buy" else f"{kw.get('fraction', 1.0):.0%} de la ligne")
        print(f"  ORDRE {side.upper()} {ticker} ({what}, {reason}) -- execute a la cloture de la prochaine seance")

    def execute_pending(self, min_position_eur: float = 0.0, max_whole_share_eur=None):
        now = now_utc()
        remaining = []
        # sales first, so their proceeds can fund buys booked in the same run
        for order in sorted(self.state["pending"], key=lambda o: o["side"] != "sell"):
            decided = pd.Timestamp(order["decided_at"])
            closes = self._series(self.close, order["ticker"])
            day = execution_day(order["ticker"], decided, closes, now)
            if day is None:
                if order["side"] == "buy" and now - decided > ORDER_EXPIRY:
                    print(f"  ORDRE ABANDONNE {order['ticker']} : pas de cours depuis {decided.date()}")
                    self.state.setdefault("dropped", []).append(order["ticker"])
                else:
                    remaining.append(order)
                continue
            done = (self._sell(order, day) if order["side"] == "sell"
                    else self._buy(order, day, closes.loc[day], min_position_eur, max_whole_share_eur))
            if done is None:
                remaining.append(order)
        self.state["pending"] = remaining

    def _buy(self, order, day, close_price, min_position_eur, max_whole_share_eur):
        ticker, bucket = order["ticker"], order["bucket"]
        price_eur = to_eur(float(close_price), order.get("currency"), self.fx)
        if price_eur is None or price_eur <= 0:
            return None  # FX missing this run, retry
        cash = self.state["cash"].get(bucket, 0.0)
        amount = min(order["amount_eur"], cash)
        if amount < min(min_position_eur, order["amount_eur"]) or amount <= 0:
            if any(o["side"] == "sell" and o["bucket"] == bucket for o in self.state["pending"]):
                return None  # wait for a pending sale to free the cash
            print(f"  ACHAT ANNULE {ticker} : {cash:.2f} EUR disponibles pour {order['amount_eur']:.2f} demandes")
            self.state.setdefault("dropped", []).append(ticker)
            return False
        if order.get("fractional", True):
            cost = amount
        else:
            if max_whole_share_eur is not None and price_eur > max_whole_share_eur:
                print(f"  ACHAT ANNULE {ticker} : une action vaut {price_eur:.2f} EUR (> {max_whole_share_eur:.2f})")
                self.state.setdefault("dropped", []).append(ticker)
                return False
            # nearest whole number of shares (at least 1), not the floor: a 18 EUR share for a 33 EUR
            # slot buys 2 (37 EUR) rather than leaving 15 EUR idle
            shares = min(max(1, round(order["amount_eur"] / price_eur)), int(cash // price_eur))
            if shares < 1:
                return None if any(o["side"] == "sell" for o in self.state["pending"]) else False
            cost = shares * price_eur
        cost_local = from_eur(cost, order.get("currency"), self.fx)
        shares = cost_local / float(close_price)
        row = {c: None for c in LEDGER_COLUMNS}
        row.update({
            "ticker": ticker, "name": order.get("name"), "sector": order.get("sector"),
            "country": order.get("country"), "status": "open", "currency": order.get("currency"),
            "fractional": bool(order.get("fractional", True)), "bucket": bucket,
            "entry_decided_at": order["decided_at"], "entry_date": day.strftime("%Y-%m-%d"),
            "entry_price": float(close_price), "shares": shares, "entry_value_eur": cost,
            "entry_cost_local": cost_local, "entry_score": order.get("score"),
            "last_check_date": day.strftime("%Y-%m-%d"), "last_price": float(close_price),
            "current_value_eur": cost, "unrealized_return_pct": 0.0,
        })
        self.ledger = pd.concat([self.ledger, pd.DataFrame([row], dtype=object)], ignore_index=True)
        self.state["cash"][bucket] = cash - cost
        print(f"  ACHAT {ticker} : {cost:.2f} EUR a la cloture du {day.date()} ({float(close_price):g} "
              f"{order.get('currency') or ''}), {order['reason']}")
        return True

    def _sell(self, order, day):
        ticker, fraction = order["ticker"], float(order.get("fraction", 1.0))
        lots = self.open_lots()
        lots = lots[(lots["ticker"] == ticker) & (lots["bucket"] == order["bucket"])]
        if not len(lots):
            return False  # nothing left to sell (already sold) -- drop the order
        closes = self._series(self.close, ticker)
        proceeds, fee_left, new_rows = 0.0, TRADE_FEE_EUR, []
        for idx in lots.index:
            tr = self.lot_tr(idx, day)
            if tr is None:
                return None
            gross = to_eur(float(self.ledger.at[idx, "entry_cost_local"]) * tr * fraction,
                           self.ledger.at[idx, "currency"], self.fx)
            if gross is None:
                return None
            entry_value = float(self.ledger.at[idx, "entry_value_eur"]) * fraction  # cost of the part sold
            target = idx
            if fraction < 1.0:  # split the lot: the sold part becomes its own closed row
                part = self.ledger.loc[idx].copy()
                for col in ("shares", "entry_value_eur", "entry_cost_local", "current_value_eur"):
                    self.ledger.at[idx, col] = float(self.ledger.at[idx, col]) * (1 - fraction)
                    part[col] = float(part[col]) * fraction
                new_rows.append(part)
                target = None
            fee = min(fee_left, gross)
            fee_left -= fee
            closed = {
                "status": "closed", "exit_decided_at": order["decided_at"], "exit_date": day.strftime("%Y-%m-%d"),
                "exit_price": float(closes.loc[day]), "exit_reason": order["reason"],
                "exit_value_eur": gross - fee, "return_pct": tr - 1 - fee / entry_value,
                "holding_days": (day - pd.Timestamp(self.ledger.at[idx, "entry_date"])).days,
                "current_value_eur": gross, "unrealized_return_pct": tr - 1,
            }
            if target is None:
                new_rows[-1].update(closed)
            else:
                for col, val in closed.items():
                    self.ledger.at[target, col] = val
            proceeds += gross - fee
        if new_rows:
            self.ledger = pd.concat([self.ledger, pd.DataFrame(new_rows, dtype=object)], ignore_index=True)
        self.state["cash"][order["bucket"]] = self.state["cash"].get(order["bucket"], 0.0) + proceeds
        print(f"  VENTE {ticker} ({fraction:.0%}) : {proceeds:.2f} EUR nets a la cloture du {day.date()}, {order['reason']}")
        return True

    # ---- output -----------------------------------------------------------------------------
    def save(self, extra_summary=None):
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger.to_csv(self.ledger_path, index=False)
        self.state_path.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        closed = self.ledger[self.ledger["status"] == "closed"]
        rets = pd.to_numeric(closed["return_pct"], errors="coerce")
        cash = sum(self.state["cash"].values())
        total = self.equity()
        summary = {
            "strategy": self.label,
            "cash_eur": cash,
            "total_equity_eur": total,
            "total_return_pct": total / self.capital - 1,
            "nb_open": int(self.open_lots()["ticker"].nunique()),
            "nb_closed": len(closed),
            "win_rate_closed": float((rets > 0).mean()) if len(closed) else None,
            "avg_return_closed": float(rets.mean()) if len(closed) else None,
            "pending_orders": len(self.state["pending"]),
            "launched_at": self.state.get("launched_at"),
            **(extra_summary or {}),
        }
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        row = {"timestamp": now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"), "cash_eur": cash, "total_equity_eur": total,
               "n_open": summary["nb_open"], "n_closed": summary["nb_closed"]}
        pd.DataFrame([row]).to_csv(self.curve_path, mode="a", header=not self.curve_path.exists(), index=False)
        print(f"\n=== {self.label} : {summary['nb_open']} lignes, {cash:.2f} EUR cash, {len(self.state['pending'])} "
              f"ordre(s) en attente, valeur {total:.2f} EUR ({summary['total_return_pct']:+.1%}) ===")


def quarter_key(ts: pd.Timestamp) -> str:
    return f"{ts.year}-T{(ts.month - 1) // 3 + 1}"
