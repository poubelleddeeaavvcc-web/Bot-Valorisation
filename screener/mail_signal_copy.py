"""Bot #52 Courrier -- COPIE DE TOUS LES AVIS (2026-10-20, demande de l'utilisateur : la strategie
reelle du Bot #33, screener/mail_signal_real.py, attend qu'une newsletter ait fait ses preuves et
reste a 0 position -- "je veux des donnees pour savoir si ce bot pourrait degager un edge").

Une seule regle, sans selection : chaque avis d'ACHAT verifie du journal (results/screener/
mail_signal_journal.csv, ecrit par mail_signal_bot.py) sur une place accessible depuis un compte
francais (mail_signal_real.REAL_EXCHANGES) achete POSITION_EUR a la cloture de la seance qui suit
l'avis, et la ligne est revendue HOLD_BDAYS seances plus tard -- l'horizon "20 jours" du classement
des newsletters. Un nouvel avis sur une valeur deja detenue prolonge sa detention au lieu de racheter.
Frais : 1 EUR a l'achat et 1 EUR a la vente (TRADE_FEE_EUR). Long uniquement, pas de stop : on mesure
ce que rapporte le fait de suivre les newsletters, pas une mecanique de sortie.

Moteur : sleeve_book.Book (comme #41-51) -- remplissage a la cloture de la seance suivante,
valorisation en rendement total. Au premier run (aucun etat), tout le journal depuis START est
REJOUE seance par seance aux vrais cours de cloture (cours tronques au jour rejoue, jamais de cours
futur) : la courbe, les achats et les ventes passes existent tout de suite ; ensuite le bot avance en
direct, apres chaque passage de mail_signal_bot.py. Moment de decision d'un avis : la date du mail
(ce qu'un lecteur aurait pu faire), sauf pour un avis lu en direct et enregistre plus tard que son
mail, ou c'est l'heure d'enregistrement (jamais d'achat a un cours deja connu au moment de la lecture).

Le resume compare chaque ligne cloturee a l'ETF de reference de son marche sur les memes seances
(colonne "benchmark" du journal, SPY / EXSA.DE..., rendement total) : c'est l'ecart moyen au marche
qui dit si suivre les newsletters rapporte quelque chose.
"""
import pathlib
import sys

import pandas as pd

HERE = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

import screener.sleeve_book as sb  # noqa: E402
from screener.sleeve_book import Book, fetch_daily  # noqa: E402
from screener.mail_signal_real import REAL_EXCHANGES  # noqa: E402
from screener.simulate_constrained_portfolio import TRADE_FEE_EUR, fractional_eligible  # noqa: E402

JOURNAL_PATH = HERE / "results/screener/mail_signal_journal.csv"
STEM = "mail_signal_copy"   # results/simulation/mail_signal_copy_* -- deja dans le git add du workflow
LABEL = "Bot #52 Courrier -- copie de tous les avis"
START = pd.Timestamp("2026-09-04", tz="UTC")   # premier avis du journal (rattrapage compris)
STARTING_CAPITAL = 20000.0  # ~140 lignes simultanees sur le journal rejoue (a 15 000 EUR, des avis etaient deja ignores)
POSITION_EUR = 100.0
HOLD_BDAYS = 20
MAX_WHOLE_SHARE_EUR = 250.0  # action non fractionnable plus chere : avis ignore plutot que ligne x3
REPLAY_HOUR_UTC = 22         # instant rejoue chaque jour : apres la cloture US


def load_tips() -> pd.DataFrame:
    if not JOURNAL_PATH.exists():
        return pd.DataFrame()
    j = pd.read_csv(JOURNAL_PATH)
    j = j[(j["side"] == "long") & j["exchange"].isin(REAL_EXCHANGES)].copy()
    j["mail_date_utc"] = pd.to_datetime(j["mail_date_utc"], utc=True, format="mixed")
    j["recorded_at_utc"] = pd.to_datetime(j["recorded_at_utc"], utc=True, format="mixed")
    j = j[j["mail_date_utc"] >= START]
    live = ~j["backfill"].astype(str).str.lower().eq("true")
    j["decided_at"] = j["mail_date_utc"]
    j.loc[live, "decided_at"] = j.loc[live, ["mail_date_utc", "recorded_at_utc"]].max(axis=1)
    return j.sort_values("decided_at").reset_index(drop=True)


def step(book: Book, tips: pd.DataFrame, now: pd.Timestamp):
    """One pass of the bot as of `now`: fills/marking, exits due, then the new tips."""
    st = book.state
    book.execute_pending(min_position_eur=POSITION_EUR / 2, max_whole_share_eur=MAX_WHOLE_SHARE_EUR)
    book.mark_to_market()
    today = pd.Timestamp(now.date())

    held = set(book.open_lots()["ticker"])
    pending_sells = {o["ticker"] for o in st["pending"] if o["side"] == "sell"}
    pending_buys = {o["ticker"] for o in st["pending"] if o["side"] == "buy"}
    st["hold_until"] = {t: d for t, d in st["hold_until"].items() if t in held or t in pending_buys}
    for t in sorted(held - pending_sells):
        until = st["hold_until"].get(t)
        if until is None or pd.Timestamp(until) <= today:
            book.add_order("sell", t, "pool", f"detention de {HOLD_BDAYS} seances terminee", now, fraction=1.0)

    done = set(st["signals_done"])
    new = tips[(tips["decided_at"] <= now) & ~tips["signal_id"].astype(str).isin(done)]
    for r in new.itertuples():
        st["signals_done"].append(str(r.signal_id))
        st["counters"]["avis"] += 1
        # vente decidee HOLD_BDAYS seances apres l'avis, executee a la cloture suivante : ~HOLD_BDAYS
        # seances de detention apres l'achat (lui-meme a la cloture qui suit l'avis)
        until = (pd.Timestamp(r.decided_at.date()) + pd.offsets.BDay(HOLD_BDAYS)).strftime("%Y-%m-%d")
        pending_buys = {o["ticker"] for o in st["pending"] if o["side"] == "buy"}
        if r.ticker in set(book.open_lots()["ticker"]) | pending_buys:
            st["hold_until"][r.ticker] = max(st["hold_until"].get(r.ticker, until), until)
            st["counters"]["prolonges"] += 1
            continue
        free = st["cash"]["pool"] - sum(o["amount_eur"] + TRADE_FEE_EUR for o in st["pending"] if o["side"] == "buy")
        if free < POSITION_EUR + TRADE_FEE_EUR:
            st["counters"]["complet"] += 1
            continue
        decided = pd.Timestamp(r.decided_at)
        # "sector" porte la newsletter a l'origine de la ligne (lu par market_comparison)
        book.add_order("buy", r.ticker, "pool", f"avis {r.publication}", decided,
                       amount_eur=POSITION_EUR, name=r.name, sector=r.publication, currency=r.currency,
                       fractional=fractional_eligible(r.ticker, None, None))
        st["hold_until"][r.ticker] = until
        st["counters"]["achetes"] += 1


def market_comparison(book: Book, tips: pd.DataFrame) -> dict:
    """Ecart de chaque ligne cloturee a l'ETF de son marche, memes seances, rendement total."""
    closed = book.ledger[book.ledger["status"] == "closed"]
    if not len(closed):
        return {"n": 0}
    bench_of = tips.groupby("ticker")["benchmark"].last().to_dict()
    benches = sorted({bench_of.get(t) for t in closed["ticker"]} - {None})
    _, adj = fetch_daily(benches, pd.Timestamp(START.date()) - pd.Timedelta(days=5)) if benches else (None, pd.DataFrame())
    rows = []
    for r in closed.itertuples():
        b = bench_of.get(r.ticker)
        s = adj[b].dropna() if b in getattr(adj, "columns", []) else pd.Series(dtype=float)
        e, x = pd.Timestamp(r.entry_date), pd.Timestamp(r.exit_date)
        if not len(s.loc[:e]) or not len(s.loc[:x]):
            continue
        bench_ret = float(s.loc[:x].iloc[-1] / s.loc[:e].iloc[-1] - 1)
        rows.append({"publication": r.sector, "ret": float(r.return_pct), "excess": float(r.return_pct) - bench_ret})
    df = pd.DataFrame(rows)
    if not len(df):
        return {"n": 0}
    by_pub = (df.groupby("publication").agg(n=("ret", "size"), rendement_moyen=("ret", "mean"),
                                            ecart_moyen_marche=("excess", "mean"))
              .sort_values("n", ascending=False).round(4))
    se = df["excess"].std(ddof=1) / len(df) ** 0.5 if len(df) > 1 else None
    return {"n": int(len(df)), "rendement_moyen": round(float(df["ret"].mean()), 4),
            "ecart_moyen_marche": round(float(df["excess"].mean()), 4),
            "t_stat_ecart": round(float(df["excess"].mean() / se), 2) if se else None,
            "battent_le_marche": round(float((df["excess"] > 0).mean()), 3),
            "par_newsletter": by_pub.reset_index().to_dict(orient="records")}


def replay(book: Book, tips: pd.DataFrame, until: pd.Timestamp):
    """Premier run : rejoue le journal seance par seance, cours tronques a chaque jour rejoue."""
    tickers = set(tips["ticker"])
    close, adj = fetch_daily(tickers, pd.Timestamp(START.date()) - pd.Timedelta(days=5))
    real_now = sb.now_utc
    try:
        for day in pd.bdate_range(START.date(), until.date() - pd.Timedelta(days=1)):
            now = pd.Timestamp(day, tz="UTC") + pd.Timedelta(hours=REPLAY_HOUR_UTC)
            sb.now_utc = lambda now=now: now
            book.close, book.adj = close.loc[:day.normalize()], adj.loc[:day.normalize()]
            step(book, tips, now)
            book.mark_to_market()
            book.save(summary_extra(book))
    finally:
        sb.now_utc = real_now
    print(f"  rejeu termine : {book.state['counters']}")


def summary_extra(book: Book) -> dict:
    return {"counters": book.state["counters"], "capital_eur": STARTING_CAPITAL, "position_eur": POSITION_EUR,
            "hold_bdays": HOLD_BDAYS,
            "start": START.strftime("%Y-%m-%d"),
            "investi_eur": round(book.position_value(), 2)}


def main():
    tips = load_tips()
    fresh = not (HERE / f"results/simulation/{STEM}_state.json").exists()
    book = Book(STEM, LABEL, STARTING_CAPITAL, {"pool": STARTING_CAPITAL})
    book.buy_fee_eur = TRADE_FEE_EUR
    st = book.state
    for k, v in (("hold_until", {}), ("signals_done", []),
                 ("counters", {"avis": 0, "achetes": 0, "prolonges": 0, "complet": 0})):
        st.setdefault(k, v)
    now = sb.now_utc()
    if fresh:
        st["launched_at"] = START.strftime("%Y-%m-%dT%H:%M:%SZ")
        replay(book, tips, now)
    todo = tips[(tips["decided_at"] <= now) & ~tips["signal_id"].astype(str).isin(set(st["signals_done"]))]
    book.load_prices(extra_tickers=set(todo["ticker"]))
    book.mark_to_market()
    step(book, tips, now)
    book.mark_to_market()
    extra = summary_extra(book)
    extra["vs_marche"] = market_comparison(book, tips)
    book.save(extra)


if __name__ == "__main__":
    main()
