"""Per-exchange trading-hours lookup (2026-09-02): fetch_cache.py's refresh rotation used to
spread evenly across every hour of a flat 168-slot calendar week regardless of whether any
given ticker's market was even open (commit 068d52d) -- but fetching a closed market just
re-reads the same last close, so a US ticker refreshed at 3am New York time isn't fresher, it's
wasted rate-limit budget. Per the user's direction: refresh each ticker only during ITS market's
own trading hours, spread over a BUSINESS week (5 trading days) rather than a calendar week.

MARKET_HOURS keys on the ticker's exchange suffix (same convention as
simulate_constrained_portfolio.CANADA_SUFFIX / select_top_picks.ticker_region -- "" means no
dot in the ticker, i.e. US). Hours are whole local hours, not exact minutes: fetch_cache runs
on the cron's own hourly schedule, so sub-hour precision buys nothing. E.g. NYSE's real
9:30-16:00 session is represented as open_hour=9, close_hour=16 ("eligible during the hour
that starts at local H, for open_hour <= H < close_hour"). Deliberately ignores mid-session
lunch breaks (Tokyo, Hong Kong) and exchange-specific holiday calendars -- a ticker fetched
during a lunch break or a market holiday just returns the same still-valid last price, same
non-harm as fetching slightly before/after the exact bell; adding a full holiday calendar per
exchange would be real complexity for a case that costs nothing when missed.

Coverage matches this project's actual universe (data/universe/trending_universe.csv exchange
suffixes, checked 2026-09-02) -- an unmapped suffix falls back to DEFAULT_MARKET (all-day,
weekdays only) rather than being silently starved of any refresh at all.

Uses zoneinfo (stdlib) rather than fixed UTC offsets so DST transitions are handled correctly
without extra logic; tzdata is pinned in requirements.txt since Windows has no OS-level IANA
database for zoneinfo to fall back on (not a concern for the Linux CI runner, which ships
system tzdata, but keeps local development/testing consistent).
"""
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

# suffix -> (IANA tz, open_hour, close_hour, label)
MARKET_HOURS = {
    "":      ("America/New_York",     9, 16, "US (NYSE/NASDAQ)"),
    "TO":    ("America/Toronto",      9, 16, "Canada (TSX)"),
    "T":     ("Asia/Tokyo",           9, 15, "Japon (TSE)"),
    "TW":    ("Asia/Taipei",          9, 14, "Taiwan (TWSE)"),
    "HK":    ("Asia/Hong_Kong",       9, 16, "Hong Kong (HKEX)"),
    "L":     ("Europe/London",        8, 16, "UK (LSE)"),
    "PA":    ("Europe/Paris",         9, 17, "France (Euronext Paris)"),
    "DE":    ("Europe/Berlin",        9, 17, "Allemagne (Xetra)"),
    "SW":    ("Europe/Zurich",        9, 17, "Suisse (SIX)"),
    "MI":    ("Europe/Rome",          9, 17, "Italie (Borsa Italiana)"),
    "AS":    ("Europe/Amsterdam",     9, 17, "Pays-Bas (Euronext Amsterdam)"),
    "MC":    ("Europe/Madrid",        9, 17, "Espagne (BME)"),
    "ST":    ("Europe/Stockholm",     9, 17, "Suede (Nasdaq Stockholm)"),
    "OL":    ("Europe/Oslo",          9, 16, "Norvege (Oslo Bors)"),
    "BR":    ("Europe/Brussels",      9, 17, "Belgique (Euronext Brussels)"),
    "IR":    ("Europe/Dublin",        8, 16, "Irlande (Euronext Dublin)"),
    "LS":    ("Europe/Lisbon",        8, 16, "Portugal (Euronext Lisbon)"),
    "KS":    ("Asia/Seoul",           9, 15, "Coree du Sud (KRX)"),
    "AX":    ("Australia/Sydney",    10, 16, "Australie (ASX)"),
    "SA":    ("America/Sao_Paulo",   10, 17, "Bresil (B3)"),
    "MX":    ("America/Mexico_City",  8, 15, "Mexique (BMV)"),
    "NS":    ("Asia/Kolkata",         9, 15, "Inde (NSE)"),
    "BO":    ("Asia/Kolkata",         9, 15, "Inde (BSE)"),
    # multi-class Canadian listings observed in the universe (e.g. "RCI.B.TO") -- same market
    "UN.TO": ("America/Toronto",      9, 16, "Canada (TSX)"),
    "B.TO":  ("America/Toronto",      9, 16, "Canada (TSX)"),
    "A.TO":  ("America/Toronto",      9, 16, "Canada (TSX)"),
    "X.TO":  ("America/Toronto",      9, 16, "Canada (TSX)"),
    "A.L":   ("Europe/London",        8, 16, "UK (LSE)"),
}
DEFAULT_MARKET = ("UTC", 0, 24, "marche non reconnu -- eligible toute la journee (jours ouvres)")
# fail open: an unmapped suffix is still weekday-only (see current_slot_for_market's own
# weekend check) but never time-restricted -- better to refresh it more than strictly
# necessary than to silently never refresh it because its exchange wasn't in the table.


def ticker_suffix(ticker: str) -> str:
    return ticker.split(".", 1)[1] if "." in ticker else ""


def market_of(ticker: str) -> tuple:
    """Returns (tz_name, open_hour, close_hour, label) for this ticker's exchange."""
    return MARKET_HOURS.get(ticker_suffix(ticker), DEFAULT_MARKET)


def current_slot_for_market(tz_name: str, open_hour: int, close_hour: int):
    """Returns (slot, total_slots) for a market's session RIGHT NOW if it's open (weekday,
    local time within [open_hour, close_hour)), else None. Purely a function of wall-clock
    time -- no persisted state needed, same reasoning as fetch_cache's old flat rotation
    (see project memory 'Fetch rotation schedule'): Monday's first open hour is slot 0, and
    the sequence runs through Friday's last open hour before wrapping back to Monday."""
    now_local = pd.Timestamp.now(tz=ZoneInfo(tz_name))
    if now_local.weekday() >= 5:  # Saturday/Sunday
        return None
    if not (open_hour <= now_local.hour < close_hour):
        return None
    session_hours = close_hour - open_hour
    slot = now_local.weekday() * session_hours + (now_local.hour - open_hour)
    return slot, 5 * session_hours


# One answer per ticker per process: a bot run lasts minutes, the session state doesn't change.
_TRADABLE_CACHE = {}


def tradable_now(ticker: str) -> bool:
    """True when an order on this ticker would really execute right now (2026-10-12).

    Before this, the bots #1-40 bought and sold at whatever price Yahoo returned, market open or
    not: Bot #40 bought B3SA3.SA on Sunday 2026-10-04 at Friday's close (19.00 BRL) and booked the
    +16% Monday gap (open 22.10) as its own gain -- a fill no real order could have had, and the
    reason D10 looked like it had "an edge". Every buy, conviction top-up and sale now waits for a
    run during the ticker's own session (the hourly CI gets several per session), at that run's
    live price; a closed market defers the trade, it doesn't cancel it.

    Two checks, cheapest first: the exchange's weekday hours (MARKET_HOURS, no request), then at
    least one 1-minute Yahoo bar dated today in the exchange's time zone -- which catches what
    MARKET_HOURS deliberately ignores (holidays: Tokyo on 2026-10-12, Sports Day; a session not
    open yet at :05 past the hour). Tested 2026-10-12 on US, Paris, Xetra, London, Zurich,
    Stockholm, Oslo, Taipei, Tokyo, Hong Kong, Sao Paulo, Mexico, Sydney, Mumbai, Seoul, Toronto:
    every one serves 1-minute bars (15-min delayed for most). A ticker without 1-minute data falls
    back to today's daily bar, so an exotic listing can't be locked out of its own exits.
    """
    tz_name, open_hour, close_hour, _ = market_of(ticker)
    if current_slot_for_market(tz_name, open_hour, close_hour) is None:
        return False
    if ticker not in _TRADABLE_CACHE:
        tz = ZoneInfo(tz_name)
        today = pd.Timestamp.now(tz=tz).date()
        ok = False
        try:
            t = yf.Ticker(ticker)
            bars = t.history(period="1d", interval="1m")
            if not len(bars):
                bars = t.history(period="5d", interval="1d")
            ok = bool(len(bars)) and bars.index[-1].tz_convert(tz).date() == today
        except Exception as e:
            print(f"  echec verification de seance pour {ticker}: {e}", file=sys.stderr)
        _TRADABLE_CACHE[ticker] = ok
    return _TRADABLE_CACHE[ticker]
