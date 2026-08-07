"""Does 'cheap AND profitable' (value + quality) beat 'cheap alone', and does it avoid the
long value-underperformance stretches seen in run_value_premium_check.py? Real data again:
Fama-French 25 portfolios double-sorted on Book-to-Market x Operating Profitability."""
import pathlib

import pandas as pd

HERE = pathlib.Path(__file__).parent
PATH = HERE / "data/factors/bemeop/25_Portfolios_BEME_OP_5x5.csv"


def load():
    df = pd.read_csv(PATH, skiprows=23, nrows=782 - 24)
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
    df = load()
    groups = {
        "Cher + peu rentable (LoBM LoOP)": "LoBM LoOP",
        "Cher + rentable (LoBM HiOP)": "LoBM HiOP",
        "Decote + peu rentable = piege a valeur (HiBM LoOP)": "HiBM LoOP",
        "Decote + rentable = value+qualite (HiBM HiOP)": "HiBM HiOP",
    }

    print("=== Vue d'ensemble (1963-2026) ===")
    for label, col in groups.items():
        print(f"  {label:55s}: {cagr(df[col]):+.2%}/an")

    print("\n=== Par decennie ===")
    for start, end, label in [
        (1963, 1969, "1963-69"), (1970, 1979, "1970s"), (1980, 1989, "1980s"),
        (1990, 1999, "1990s"), (2000, 2009, "2000s"), (2010, 2019, "2010s"),
        (2020, 2026, "2020-2026"),
    ]:
        w = df.loc[f"{start}":f"{end}"]
        if len(w) == 0:
            continue
        vals = {label2: cagr(w[col]) for label2, col in groups.items()}
        print(f"  {label}: piege_valeur={vals['Decote + peu rentable = piege a valeur (HiBM LoOP)']:+.1%}  "
              f"value+qualite={vals['Decote + rentable = value+qualite (HiBM HiOP)']:+.1%}  "
              f"cher+rentable={vals['Cher + rentable (LoBM HiOP)']:+.1%}")

    print("\n=== 15 dernieres annees (2011-2026) : la ou le value pur perdait contre le growth ===")
    w = df.loc["2011":"2026"]
    for label, col in groups.items():
        print(f"  {label:55s}: {cagr(w[col]):+.2%}/an")


if __name__ == "__main__":
    main()
