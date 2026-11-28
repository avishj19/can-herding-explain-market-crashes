"""Are the saved market files in data/ right? Each one checked against an independent copy.

Independent copies (downloaded into scratch/crosscheck/, git-ignored, if not already there):
    FRED  SP500, DJIA, NASDAQ100 (S&P Dow Jones Indices / Nasdaq, via the St. Louis Fed;
          SP500 and DJIA cover only the last ~10 years), and DCOILBRENTEU (the saved Brent
          oil file was made from it, so for oil this is a check that nothing was lost or
          revised, not a second source)
    Yahoo ^DJI (1992-): a second source for the MeasuringWorth Dow
Plus two checks inside data/: the 1950- and 1928- S&P files, and the two Dow files, must
agree where they overlap.

For each pair, on the days both have (up to 2026-10-30, the end of the saved files):
common days, days in one but not the other, the largest and median relative difference
of closes, every day above 0.1%, and the same for daily log returns (what the model is
compared on). Last, the app's numbers (explore.summary: tail, kurtosis, big days) for
the saved returns and for the independent copy's, over the same days.

Run from the project folder:  python3 -I scripts/analysis/data_crosscheck.py
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd

from herding import explore, market_data

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "scratch" / "crosscheck"
END = pd.Timestamp("2026-10-30")
LIMIT = 0.001          # 0.1%: a difference above this is listed
FRED = ["SP500", "DJIA", "NASDAQ100", "DCOILBRENTEU"]
SHOW = 12              # at most this many dates are printed per list


def download():
    """Fetch any independent copy not yet in scratch/crosscheck/."""
    OUT.mkdir(parents=True, exist_ok=True)
    for series in FRED:
        path = OUT / f"fred_{series}.csv"
        if not path.exists():
            subprocess.run(["curl", "-sSL", "--connect-timeout", "30", "--retry", "3", "-o", str(path),
                            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"], check=True)
    path = OUT / "yahoo_DJI.csv"
    if not path.exists():
        import yfinance as yf
        close = yf.download("^DJI", start="1985-01-01", end="2026-10-31", auto_adjust=False,
                            progress=False)["Close"]
        close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
        close.rename("Close").rename_axis("Date").to_csv(path)


def fred(series):
    """A FRED series as numbers, blank days (holidays) dropped, up to END."""
    df = pd.read_csv(OUT / f"fred_{series}.csv", parse_dates=["observation_date"], index_col=0)
    s = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
    return s[s.index <= END].rename(series)


def saved(name):
    return market_data.load_prices(ROOT / "data" / name)["Close"]


def _dates(index):
    shown = ", ".join(d.strftime("%Y-%m-%d") for d in index[:SHOW])
    return shown + (f", ... ({len(index)} in all)" if len(index) > SHOW else "")


def compare(label, a, b, a_name="saved", b_name="other"):
    """Print how closely two daily series agree; return the dates whose close differs > LIMIT."""
    lo, hi = max(a.index[0], b.index[0]), min(a.index[-1], b.index[-1], END)
    a, b = a[(a.index >= lo) & (a.index <= hi)], b[(b.index >= lo) & (b.index <= hi)]
    both = a.index.intersection(b.index)
    only_a, only_b = a.index.difference(b.index), b.index.difference(a.index)
    rel = (a[both] / b[both] - 1).abs()
    bad = rel[rel > LIMIT]
    print(f"\n== {label}   ({lo:%Y-%m-%d} to {hi:%Y-%m-%d})")
    print(f"   common days {len(both)}; only in {a_name}: {len(only_a)}; only in {b_name}: {len(only_b)}")
    if len(only_a):
        print(f"     only in {a_name}: {_dates(only_a)}")
    if len(only_b):
        print(f"     only in {b_name}: {_dates(only_b)}")
    print(f"   closes: max |relative difference| {rel.max():.2e} ({rel.idxmax():%Y-%m-%d}), "
          f"median {rel.median():.2e}, days above 0.1%: {len(bad)}")
    for d in bad.sort_values(ascending=False).index[:SHOW]:
        print(f"     {d:%Y-%m-%d}  {a_name} {a[d]:.4f}  {b_name} {b[d]:.4f}  ({a[d] / b[d] - 1:+.3%})")
    ra, rb = np.log(a[both]).diff().dropna(), np.log(b[both]).diff().dropna()
    dr = (ra - rb).abs()
    big = dr[dr > LIMIT]
    print(f"   log returns (on common days): max |difference| {dr.max():.2e} ({dr.idxmax():%Y-%m-%d}), "
          f"median {dr.median():.2e}, days above 0.001: {len(big)}")
    for d in big.sort_values(ascending=False).index[:SHOW]:
        print(f"     {d:%Y-%m-%d}  {a_name} {ra[d]:+.5f}  {b_name} {rb[d]:+.5f}")
    return pd.DatetimeIndex(bad.index), both


def app_numbers(label, a, b, both, suspect=()):
    """explore.summary of each copy's returns on the common days (and of the saved one
    without the suspect closes, the way market_data.returns drops BAD_CLOSES)."""
    rows = {"saved": np.log(a[both]).diff().dropna(), "independent": np.log(b[both]).diff().dropna()}
    if len(suspect):
        rows["saved without suspects"] = np.log(a[both.difference(suspect)]).diff().dropna()
    print(f"   app numbers, {label}:")
    for name, r in rows.items():
        s = explore.summary(r.values)
        print(f"     {name:<23} days {s['days']:5d}  tail {s['tail']:.3f}  kurtosis {s['kurtosis']:6.2f}  "
              f"memory {s['memory']:.3f}  worst {s['worst']:6.2f} sd  big days {s['big_days']}")


def main(days=None, n=None):
    """Compare every pair and print the report. days and n are unused (nothing is simulated here):
    they are there so tests/test_analysis.py can call every analysis script the same way."""
    download()
    spx, spx28, ndx = saved("sp500_daily.csv"), saved("sp500_daily_from_1928.csv"), saved("nasdaq100_daily.csv")
    dow, dow_short = saved("dow_jones_daily_from_1896.csv"), saved("dow_jones_daily_1927_1952.csv")
    dji = pd.read_csv(OUT / "yahoo_DJI.csv", parse_dates=["Date"], index_col="Date")["Close"]

    print("Saved files in data/ against independent copies (differences above 0.1% listed)")
    pairs = [("S&P 500 1950- (Yahoo ^GSPC) vs FRED SP500", spx, fred("SP500"), "yahoo", "fred"),
             ("S&P 500 1928- (Yahoo ^GSPC) vs FRED SP500", spx28, fred("SP500"), "yahoo", "fred"),
             ("Nasdaq-100 (Yahoo ^NDX) vs FRED NASDAQ100", ndx, fred("NASDAQ100"), "yahoo", "fred"),
             ("Dow 1896- (MeasuringWorth) vs FRED DJIA", dow, fred("DJIA"), "mw", "fred"),
             ("Dow 1896- (MeasuringWorth) vs Yahoo ^DJI", dow, dji, "mw", "yahoo")]
    for label, a, b, an, bn in pairs:
        bad, both = compare(label, a, b, an, bn)
        app_numbers(label.split(" vs ")[0], a, b, both, bad)

    print("\nInside data/: the files must agree where they overlap")
    compare("S&P 500: 1950- file vs 1928- file", spx, spx28, "1950file", "1928file")
    compare("Dow: 1896- file vs 1927-1952 file", dow, dow_short, "1896file", "1927file")

    print("\nBrent oil (the site's Commodities group) against today's FRED (the source it was made from)")
    compare("Brent oil vs DCOILBRENTEU", saved("commodities/oil_brent.csv"), fred("DCOILBRENTEU"), "saved", "fred")
    print("\nNot checked here: gold and copper (Yahoo futures; FRED has no free daily series), "
          "and the 5 stocks.")


if __name__ == "__main__":
    main()
