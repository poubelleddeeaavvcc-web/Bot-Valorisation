"""Asness/Moskowitz/Pedersen ('Value and Momentum Everywhere', 2013) showed value and
momentum are negatively correlated and that blending them smooths the long losing
stretches either factor has alone. Replicated here with the same public Fama-French data
already used, to check whether this actually would have helped over the exact periods
where pure value (HML) got crushed -- before building anything on top of the idea."""
import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent
FF3 = HERE / "data/factors/ff3/F-F_Research_Data_Factors.csv"
MOM = HERE / "data/factors/mom/F-F_Momentum_Factor.csv"


def load_ff3():
    df = pd.read_csv(FF3, skiprows=4, nrows=1206 - 5)
    df.columns = ["date", "Mkt-RF", "SMB", "HML", "RF"]
    df = df[df["date"].astype(str).str.match(r"^\d{6}$")]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    for c in ["Mkt-RF", "SMB", "HML", "RF"]:
        df[c] = pd.to_numeric(df[c], errors="coerce") / 100.0
    return df.set_index("date")


def load_mom():
    df = pd.read_csv(MOM, skiprows=13, nrows=1209 - 14)
    df.columns = ["date", "Mom"]
    df = df[df["date"].astype(str).str.match(r"^\d{6}$")]
    df["date"] = pd.to_datetime(df["date"], format="%Y%m")
    df["Mom"] = pd.to_numeric(df["Mom"], errors="coerce") / 100.0
    return df.set_index("date")


def cagr(returns: pd.Series) -> float:
    growth = (1 + returns).prod()
    years = len(returns) / 12
    return growth ** (1 / years) - 1


def sharpe(returns: pd.Series) -> float:
    return returns.mean() / returns.std() * np.sqrt(12)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    return (equity / equity.cummax() - 1).min()


def main():
    ff3, mom = load_ff3(), load_mom()
    df = ff3.join(mom, how="inner")
    df["Blend"] = 0.5 * df["HML"] + 0.5 * df["Mom"]

    print(f"Correlation HML vs Mom (mensuelle, {len(df)} mois depuis {df.index[0]:%Y-%m}) : "
          f"{df['HML'].corr(df['Mom']):+.2f}")

    print("\n=== CAGR par decennie : HML seul vs Mom seul vs blend 50/50 ===")
    for start, end, label in [
        (1930, 1939, "1930s"), (1940, 1949, "1940s"), (1950, 1959, "1950s"),
        (1960, 1969, "1960s"), (1970, 1979, "1970s"), (1980, 1989, "1980s"),
        (1990, 1999, "1990s"), (2000, 2009, "2000s"), (2010, 2019, "2010s"),
        (2020, 2026, "2020-2026"),
    ]:
        w = df.loc[f"{start}":f"{end}"]
        if len(w) == 0:
            continue
        print(f"  {label}: HML={cagr(w['HML']):+.1%}  Mom={cagr(w['Mom']):+.1%}  Blend={cagr(w['Blend']):+.1%}")

    print(f"\n=== Vue d'ensemble (depuis {df.index[0]:%Y-%m}) ===")
    for name in ["HML", "Mom", "Blend"]:
        print(f"  {name:6s}: CAGR={cagr(df[name]):+.2%}  Sharpe={sharpe(df[name]):.2f}  "
              f"MaxDD={max_drawdown_from_returns(df[name]):.1%}  "
              f"pire decennie glissante={df[name].rolling(120).apply(lambda x: cagr(x)).min():+.1%}")

    last15 = df.loc["2011":"2026"]
    print(f"\n=== 15 dernieres annees (2011-2026), la periode ou le value pur perdait ===")
    for name in ["HML", "Mom", "Blend"]:
        print(f"  {name:6s}: CAGR={cagr(last15[name]):+.2%}  Sharpe={sharpe(last15[name]):.2f}  "
              f"MaxDD={max_drawdown_from_returns(last15[name]):.1%}")

    # rolling 10-year CAGR of HML alone vs blend, to see how often each spends a decade underwater
    roll_hml = df["HML"].rolling(120).apply(lambda x: cagr(x))
    roll_blend = df["Blend"].rolling(120).apply(lambda x: cagr(x))
    print(f"\n=== Part du temps ou le facteur a un CAGR glissant sur 10 ans negatif ===")
    print(f"  HML seul : {(roll_hml.dropna() < 0).mean():.0%} des mois")
    print(f"  Blend    : {(roll_blend.dropna() < 0).mean():.0%} des mois")


if __name__ == "__main__":
    main()
