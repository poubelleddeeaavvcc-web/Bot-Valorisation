"""Does buying 'cheap' (undervalued, high book-to-market) stocks actually beat 'expensive'
(growth) stocks over the long run? Answered with real academic data (Kenneth French Data
Library), not a TradingView backtest -- this is the best publicly available long-run
evidence for or against the user's stated premise ("find undervalued stocks, hold for the
re-rating, sell, reinvest")."""
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
FF3 = HERE / "data/factors/ff3/F-F_Research_Data_Factors.csv"
BEME = HERE / "data/factors/beme/Portfolios_Formed_on_BE-ME.csv"


def load_ff3_monthly():
    df = pd.read_csv(FF3, skiprows=4, nrows=1206 - 5)
    df.columns = ["date", "Mkt-RF", "SMB", "HML", "RF"]
    df = df[df["date"].astype(str).str.match(r"^\d{6}$")]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    for c in ["Mkt-RF", "SMB", "HML", "RF"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df.set_index("date")


def load_beme_monthly():
    df = pd.read_csv(BEME, skiprows=23, nrows=1226 - 24)
    df = df.rename(columns={df.columns[0]: "date"})
    df = df[df["date"].astype(str).str.match(r"^\d{6}$")]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df.set_index("date")


def cagr(returns: pd.Series) -> float:
    growth = (1 + returns).prod()
    years = len(returns) / 12
    return growth ** (1 / years) - 1


def main():
    ff3 = load_ff3_monthly()
    beme = load_beme_monthly()

    print("=== Prime de valeur (HML) par décennie : value - growth, en rendement annualise ===")
    for start, end, label in [
        (1930, 1939, "1930s"), (1940, 1949, "1940s"), (1950, 1959, "1950s"),
        (1960, 1969, "1960s"), (1970, 1979, "1970s"), (1980, 1989, "1980s"),
        (1990, 1999, "1990s"), (2000, 2009, "2000s"), (2010, 2019, "2010s"),
        (2020, 2026, "2020-2026"),
    ]:
        window = ff3.loc[f"{start}":f"{end}", "HML"]
        if len(window) == 0:
            continue
        print(f"  {label}: HML (value - growth) = {cagr(window):+.1%}/an, sur {len(window)} mois")

    print("\n=== Rendement annualise Hi 10 (value/decote) vs Lo 10 (growth/glamour), par decennie ===")
    for start, end, label in [
        (1930, 1939, "1930s"), (1940, 1949, "1940s"), (1950, 1959, "1950s"),
        (1960, 1969, "1960s"), (1970, 1979, "1970s"), (1980, 1989, "1980s"),
        (1990, 1999, "1990s"), (2000, 2009, "2000s"), (2010, 2019, "2010s"),
        (2020, 2026, "2020-2026"),
    ]:
        w = beme.loc[f"{start}":f"{end}"]
        if len(w) == 0 or "Hi 10" not in w or "Lo 10" not in w:
            continue
        print(f"  {label}: Hi10(value)={cagr(w['Hi 10']):+.1%}/an  Lo10(growth)={cagr(w['Lo 10']):+.1%}/an  "
              f"ecart={cagr(w['Hi 10']) - cagr(w['Lo 10']):+.1%}")

    print(f"\n=== Vue d'ensemble (juillet 1926 -> aujourd'hui, {len(ff3)} mois) ===")
    print(f"  HML (value - growth) annualise : {cagr(ff3['HML']):+.2%}/an")
    print(f"  Hi10 (value) annualise          : {cagr(beme['Hi 10']):+.2%}/an")
    print(f"  Lo10 (growth) annualise         : {cagr(beme['Lo 10']):+.2%}/an")
    print(f"  Marche (Mkt-RF + RF) annualise  : {cagr(ff3['Mkt-RF'] + ff3['RF']):+.2%}/an")

    last15 = beme.loc["2011":"2026"]
    print(f"\n=== 15 dernieres annees (2011-2026, {len(last15)} mois) ===")
    print(f"  Hi10 (value) annualise  : {cagr(last15['Hi 10']):+.2%}/an")
    print(f"  Lo10 (growth) annualise : {cagr(last15['Lo 10']):+.2%}/an")


if __name__ == "__main__":
    main()
