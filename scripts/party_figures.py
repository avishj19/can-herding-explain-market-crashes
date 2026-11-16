"""Three plain pictures about the stock market and the president's party
(figures/fig13-fig15*.png), for a reader who knows no finance.

Run:  python3 scripts/party_figures.py        (a few seconds)

    fig13  the Dow Jones since 1896, and what $1 became under each party
    fig14  every presidency: yearly return, volatility, worst day
    fig15  is the Democrat-minus-Republican return gap more than luck?

Every number printed on a figure is computed here from the data, never typed in.
Each figure function takes the returns and a path, and gives back its numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the project folder, so `herding` imports

import numpy as np
import pandas as pd

from herding import experiments, market_data, plots, politics
from herding.plots import BLUE, GRID, INK, MUTED, ORANGE, SECOND, plt
from herding.plots import DEM_SHADE, REP_SHADE     # the same party colours as Fig 12

FIG = market_data.ROOT / "figures"
SHADE = {"D": DEM_SHADE, "R": REP_SHADE}           # light: background bands
STRONG = {"D": BLUE, "R": ORANGE}                  # strong: lines and bars
PARTY = {"D": "Democratic", "R": "Republican"}
# Events named on the figures: a label is used only when a day found by the code falls
# inside the window (first day, last day).
EVENTS = [("1929-10-24", "1929-11-13", "1929 crash"),
          ("1987-10-19", "1987-10-19", "Black Monday"),
          ("2008-09-15", "2008-12-31", "2008 financial crisis"),
          ("2020-03-01", "2020-03-31", "Covid panic")]
MARKET_NAMES = {"dow": "Dow Jones", "spx_1928": "S&P 500", "spx": "S&P 500"}


# ------------------------------------------------------------------ small helpers
def years_between(start, end):
    """Calendar years from the day `start` to the day `end`, both included."""
    return ((pd.Timestamp(end) - pd.Timestamp(start)).days + 1) / 365.25


def event_name(date):
    """The name of the famous event a date belongs to, or '' if it is not one of EVENTS."""
    date = pd.Timestamp(date)
    for first, last, name in EVENTS:
        if pd.Timestamp(first) <= date <= pd.Timestamp(last):
            return name
    return ""


def event_days(r):
    """For each of EVENTS inside the data: (the day of the biggest fall in it, its name)."""
    out = []
    for first, last, name in EVENTS:
        window = r.loc[first:last]
        if len(window):
            out.append((window.idxmin(), name))
    return out


def money(x):
    """Dollars as short text: $0.85, $7.40, $1,240 (the \\ stops matplotlib reading $ as maths)."""
    return rf"\${x:,.2f}" if x < 10 else rf"\${x:,.0f}"


def term_table(r, min_days=20):
    """One row per presidency in the returns `r` (a dated pandas Series of daily log
    returns): name, party, label, years in the data, and
        yearly      the price change per calendar year, compounded (0.07 = +7% a year)
        volatility  how much the market moves in a typical year (the standard deviation
                    of daily returns, scaled up to a year)
        worst       the biggest one-day fall (-0.12 = the market lost 12% that day)
        total       the whole term's log return (used to add terms up)."""
    rows = []
    for t in politics.terms(r.index, min_days):
        x = r.iloc[t["a"]:t["b"]]
        years = years_between(t["start"], t["end"])
        span = f"{t['start'].year}–{t['end'].strftime('%y')}"
        rows.append(dict(
            name=t["name"], party=t["party"], start=t["start"], end=t["end"], years=years,
            label=f"{t['name']} ({t['party']})  {span}",
            total=float(x.sum()), yearly=float(np.exp(x.sum() / years) - 1),
            volatility=float(x.std() * np.sqrt(len(x) / years)),
            worst=float(np.exp(x.min()) - 1), worst_date=x.idxmin()))
    return rows


def dollar_under(r, dem, party):
    """What $1 becomes, day by day, if it is in the market only on days when `party`
    ('D' or 'R') holds the presidency, and sits still (no interest) on all other days."""
    mine = np.asarray(dem, dtype=bool) == (party == "D")
    return np.exp(np.cumsum(np.where(mine, np.asarray(r, dtype=float), 0.0)))


def gap_percent(r, dem):
    """Average return under Democrats minus under Republicans, in % per year of 252
    trading days (the same measure as Fig 12b, from politics.party_gap)."""
    r = np.asarray(r, dtype=float)
    return 100 * politics.party_gap(r, dem)["mean"] * r.std()


def party_bands(ax, dates):
    """Shade the background by the president's party (light blue D, light orange R)."""
    plots.party_bands(ax, dates, politics.switches(dates))


# ------------------------------------------------------------------ Fig 13
def party_totals(r):
    """Per party: the years in office, what $1 became under that party only, the same
    as a yearly rate, and what it became without that party's most extreme presidency
    (the best Democratic one, the worst Republican one)."""
    table = term_table(r, min_days=1)
    out = {}
    for party, pick in (("D", max), ("R", min)):
        mine = [t for t in table if t["party"] == party]
        total, years = sum(t["total"] for t in mine), sum(t["years"] for t in mine)
        extreme = pick(mine, key=lambda t: t["total"])
        out[party] = dict(years=years, dollar=float(np.exp(total)),
                          yearly=float(np.exp(total / years) - 1),
                          extreme=extreme["name"], extreme_span=extreme["label"].split("  ")[1],
                          dollar_without=float(np.exp(total - extreme["total"])))
    return out


def _name_strip(ax, r):
    """A strip of party-coloured boxes above the chart, one per president, with the name."""
    for t in term_table(r, min_days=1):
        ax.axvspan(t["start"], t["end"], color=SHADE[t["party"]], lw=0)
        ax.axvline(t["start"], color="white", lw=1.2)
        middle = t["start"] + (t["end"] - t["start"]) / 2
        ax.text(middle, 0.5, f"{t['name']} ({t['party']})", rotation=90, ha="center", va="center",
                fontsize=8.5, color=INK)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _dollar_axis(ax, lowest, highest):
    """A log scale in plain dollars: $1, $10, $100 ..."""
    ax.set_yscale("log")
    ticks = 10.0 ** np.arange(np.floor(np.log10(lowest)), np.ceil(np.log10(highest)) + 1)
    ax.set_yticks(ticks, [rf"\${t:,g}" for t in ticks])
    ax.minorticks_off()
    ax.set_ylim(lowest / 1.5, highest * 1.6)


def fig13_long_view(r, path=FIG / "fig13_parties_130_years.png"):
    """The long view: the market since the data start with every president named, and
    what $1 became if invested only under Democratic or only under Republican presidents."""
    plots.style()
    dates, dem = r.index, politics.democrat(r.index)
    price = np.exp(np.cumsum(r.values))
    tot = party_totals(r)
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=(1.05, 3, 3), hspace=0.16)
    a = fig.add_subplot(gs[1])
    _name_strip(fig.add_subplot(gs[0], sharex=a), r)
    b = fig.add_subplot(gs[2], sharex=a)
    for ax in (a, b):
        party_bands(ax, dates)
        ax.grid(axis="x", visible=False)
    a.plot(dates, price, color=INK, lw=1.0)
    _dollar_axis(a, price.min(), price.max())
    for day, name in event_days(r):
        i = dates.get_loc(day)
        up = price[i] < np.sqrt(price.min() * price.max())      # label above the line in the lower half
        a.annotate(f"{name}\n{day.year}", (day, price[i]), xytext=(0, 38 if up else -38),
                   textcoords="offset points", ha="center", va="bottom" if up else "top", fontsize=8.5, color=INK,
                   arrowprops=dict(arrowstyle="-", color=SECOND, lw=0.8))
    a.set_title(f"a. $1 put into the Dow Jones in {dates[0].year} and left there became "
                f"{money(price[-1])} by {dates[-1].year}.  Background colour: the president's party",
                loc="left")
    a.set_ylabel("value of $1 (each step up = ×10)")
    lines = {}
    for party in "DR":
        lines[party] = dollar_under(r.values, dem, party)
        b.plot(dates, lines[party], color=STRONG[party], lw=2.2)
        t = tot[party]
        b.annotate(f"Only under {PARTY[party]}\npresidents ({t['years']:.0f} years):\n"
                   f"{money(t['dollar'])}  ({t['yearly']:+.1%} a year)",
                   (dates[-1], lines[party][-1]), xytext=(8, 0), textcoords="offset points",
                   va="center", fontsize=10, color=INK, fontweight="bold", annotation_clip=False)
    _dollar_axis(b, min(v.min() for v in lines.values()), max(v.max() for v in lines.values()))
    b.set_title("b. The same $1, but in the market only while one party holds the presidency "
                "(the line is flat while the other party is in office)", loc="left")
    b.set_ylabel("value of $1 (each step up = ×10)")
    d, g = tot["D"], tot["R"]
    b.text(0.012, 0.95,
           f"A few eras carry the gap:\nwithout {d['extreme']} ({d['extreme_span']}), the Democratic "
           f"$1 ends at {money(d['dollar_without'])}\nwithout {g['extreme']} ({g['extreme_span']}), "
           f"the Republican $1 ends at {money(g['dollar_without'])}",
           transform=b.transAxes, va="top", fontsize=9.5, color=INK,
           bbox=dict(facecolor="white", edgecolor=GRID, boxstyle="round,pad=0.5"))
    a.margins(x=0)
    a.tick_params(labelbottom=False)
    fig.subplots_adjust(top=0.93, bottom=0.11, left=0.06, right=0.84)
    plots.save(fig, path, f"Fig 13. {dates[-1].year - dates[0].year} years of the stock market and the "
               "president's party",
               f"The Dow Jones average (an index of large US stocks), {dates[0]:%d %b %Y} to {dates[-1]:%d %b %Y}; "
               "D = Democratic president, R = Republican. These are price changes only: no dividends "
               "(the cash companies pay shareholders) and no correction for inflation.\nSo the dollar amounts "
               "understate what an investor earned and overstate what it could buy. The gap is real in the "
               "data, but party is not shown to be the cause: it rests on a few eras (panel b, box), and Fig 15 "
               "asks whether luck could give a gap this big.")
    return dict(final=float(price[-1]), **{f"{k}_{p}": v for p, t in tot.items() for k, v in t.items()})


# ------------------------------------------------------------------ Fig 14
def _bars(ax, values, parties, title, fmt, notes=None):
    """One column of horizontal bars, one per presidency (top = earliest), each with its value."""
    y = np.arange(len(values))
    ax.barh(y, values, color=[STRONG[p] for p in parties], height=0.68)
    ax.axvline(0, color=SECOND, lw=0.8)
    pad = 0.012 * np.abs(values).max()
    for i, v in enumerate(values):
        text = fmt(v) + (f"   {notes[i]}" if notes else "")
        ax.text(v + (pad if v >= 0 else -pad), i, text, va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_axisbelow(True)
    ax.set_ylim(len(values) - 0.4, -0.6)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}%")


def fig14_every_presidency(r, path=FIG / "fig14_every_presidency.png"):
    """One row per presidency, in time order: yearly return, volatility and worst day.
    Good and bad spells do not sort by party."""
    plots.style()
    table = term_table(r)
    parties = [t["party"] for t in table]
    notes = [f"{t['worst_date']:%d %b %Y}" + (f" · {event_name(t['worst_date'])}"
                                              if event_name(t["worst_date"]) else "") for t in table]
    fig, ax = plt.subplots(1, 3, figsize=(16, 0.36 * len(table) + 2.6), sharey=True,
                           gridspec_kw=dict(width_ratios=(1.15, 0.9, 1.25), wspace=0.06))
    _bars(ax[0], 100 * np.array([t["yearly"] for t in table]), parties,
          "a. Yearly return\n(price change per year in office)", lambda v: f"{v:+.0f}%" if round(v) else "0%")
    _bars(ax[1], 100 * np.array([t["volatility"] for t in table]), parties,
          "b. Volatility\n(how much the market moves in a typical year)", lambda v: f"{v:.0f}%")
    _bars(ax[2], 100 * np.array([t["worst"] for t in table]), parties,
          "c. Worst single day\n(biggest one-day fall, and when)", lambda v: f"{v:.1f}%", notes)
    ax[0].set_yticks(np.arange(len(table)), [t["label"] for t in table], fontsize=9.5, color=INK)
    lo, hi = ax[0].get_xlim()
    ax[0].set_xlim(lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo))
    ax[1].set_xlim(0, ax[1].get_xlim()[1] * 1.15)
    ax[2].set_xticks(np.arange(10 * np.floor(ax[2].get_xlim()[0] / 10), 1, 10))
    ax[2].set_xlim(ax[2].get_xlim()[0] * 2.3, 0)
    for party, x in (("D", 0.30), ("R", 0.52)):
        fig.text(x, 0.915, "■", color=STRONG[party], fontsize=13, ha="right", va="center")
        fig.text(x + 0.004, 0.915, f"{PARTY[party]} president ({party})", fontsize=10, color=INK, va="center")
    fig.subplots_adjust(top=0.84, bottom=0.11, left=0.14, right=0.99)
    best, worst = max(table, key=lambda t: t["yearly"]), min(table, key=lambda t: t["yearly"])
    wild, calm = max(table, key=lambda t: t["volatility"]), min(table, key=lambda t: t["volatility"])
    plots.save(fig, path, "Fig 14. Every presidency since "
               f"{table[0]['start'].year}: booms, busts and crashes do not sort by party",
               f"Dow Jones average, {r.index[0]:%d %b %Y} to {r.index[-1]:%d %b %Y}, one row per president "
               "(top = earliest; the first and last rows are part-terms). Returns are price changes without "
               "dividends or inflation.\nVolatility: the standard deviation (typical size) of daily moves, "
               f"scaled to a year. Best yearly return: {best['label'].split('  ')[0]}; worst: "
               f"{worst['label'].split('  ')[0]}. Wildest: {wild['label'].split('  ')[0]}; calmest: "
               f"{calm['label'].split('  ')[0]}. Both parties have rows at both ends. A president takes office "
               "in the middle of whatever the market is already doing.")
    return table


# ------------------------------------------------------------------ Fig 15
def luck_test(r):
    """The real D-minus-R return gap (% a year), the gaps from every shifted party
    calendar (politics.shifted_gaps: what luck alone gives), and the real gap's
    percentile among them (the share of fake calendars with a smaller gap)."""
    dem = politics.democrat(r.index)
    sd = r.values.std()
    fake = np.array([100 * g["mean"] * sd for g in politics.shifted_gaps(r.values, dem)])
    real = gap_percent(r.values, dem)
    return dict(real=float(real), fake=fake, percentile=experiments.percentile(real, fake))


def rolling_gaps(r, window_years=40, min_days=500):
    """The D-minus-R gap (% a year) inside each `window_years`-long window ending on
    31 December of every year: a list of (end year, gap). Windows with fewer than
    min_days days under either party are skipped."""
    out = []
    for year in range(r.index[0].year + window_years, r.index[-1].year + 1):
        x = r.loc[f"{year - window_years + 1}-01-01":f"{year}-12-31"]
        dem = politics.democrat(x.index)
        if min(dem.sum(), (~dem).sum()) >= min_days:
            out.append((year, float(gap_percent(x.values, dem))))
    return out


def _luck_panel(ax, name, test, bins):
    """Grey histogram of the fake-calendar gaps with the real gap as one bold line."""
    ax.hist(test["fake"], bins=bins, color=MUTED, alpha=0.55, edgecolor="white", lw=0.6)
    ax.axvline(0, color=SECOND, lw=0.8)
    ax.axvline(test["real"], color=INK, lw=3)
    ax.annotate(f"real gap {test['real']:+.1f}% a year\nbigger than {test['percentile']:.0f}%\n"
                f"of {len(test['fake'])} fake calendars", (test["real"], 0.97), xytext=(-8, 0),
                textcoords="offset points", xycoords=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=9.5, color=INK, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.set_title(name, loc="left")
    ax.grid(axis="x", visible=False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+g}%" if v else "0")


def fig15_luck(markets, path=FIG / "fig15_is_the_gap_luck.png", window_years=40):
    """Is the party gap more than luck? markets: {name: dated Series of daily log returns}.
    Top: each market's real gap against the gaps from shifted (fake) party calendars.
    Bottom: the gap of the first market inside a moving window, to see if it is stable."""
    plots.style()
    tests = {name: luck_test(r) for name, r in markets.items()}
    everything = np.concatenate([np.append(t["fake"], t["real"]) for t in tests.values()])
    bins = np.linspace(np.floor(everything.min()), np.ceil(everything.max()), 31)
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, len(markets), height_ratios=(1, 0.85), hspace=0.5, wspace=0.08)
    first = None
    for j, (name, r) in enumerate(markets.items()):
        ax = fig.add_subplot(gs[0, j], sharex=first, sharey=first)
        first = first or ax
        _luck_panel(ax, name, tests[name], bins)
        ax.set_xlabel("return under Democrats minus under Republicans (% a year)")
        if j:
            ax.tick_params(labelleft=False)
    first.set_ylabel("number of fake calendars")
    first.set_ylim(0, first.get_ylim()[1] * 1.45)
    name, r = next(iter(markets.items()))
    rolling = rolling_gaps(r, window_years)
    b = fig.add_subplot(gs[1, :])
    years, gaps = np.array([y for y, _ in rolling]), np.array([g for _, g in rolling])
    b.axhline(0, color=SECOND, lw=0.9)
    b.fill_between(years, 0, gaps, where=gaps >= 0, color=BLUE, alpha=0.25, lw=0, interpolate=True)
    b.fill_between(years, 0, gaps, where=gaps <= 0, color=ORANGE, alpha=0.3, lw=0, interpolate=True)
    b.plot(years, gaps, color=INK, lw=2)
    whole = tests[name]["real"]
    b.axhline(whole, color=INK, lw=1, ls="--")
    b.text(years[-1], whole, f" whole period: {whole:+.1f}%", va="center", fontsize=9, color=INK, clip_on=False)
    lo, hi = min(gaps.min(), 0), max(gaps.max(), whole)
    b.set_ylim(lo - 0.3 * (hi - lo), hi + 0.3 * (hi - lo))
    b.text(0.008, 0.96, "above 0: stocks did better under Democrats", transform=b.transAxes,
           va="top", fontsize=9.5, color=INK)
    b.text(0.008, 0.04, "below 0: stocks did better under Republicans", transform=b.transAxes,
           va="bottom", fontsize=9.5, color=INK)
    b.set_xlim(years[0], years[-1])
    b.set_xlabel(f"last year of the {window_years}-year window")
    b.set_ylabel("gap, D minus R (% a year)")
    b.yaxis.set_major_formatter(lambda v, _: f"{v:+g}%" if v else "0")
    b.set_title(f"The same gap measured in every {window_years}-year stretch of the {name}: "
                f"from {gaps.min():+.1f}% to {gaps.max():+.1f}% a year, depending on which years you look at",
                loc="left")
    fig.subplots_adjust(top=0.9, bottom=0.14, left=0.05, right=0.93)
    plots.save(fig, path, "Fig 15. Stocks did better under Democratic presidents. Is that more than luck?",
               "The luck test: keep the market exactly as it was, but slide the calendar of presidents along it "
               "by a quarter, two quarters, ... (never closer than a year to the real one). Each fake calendar "
               "keeps the real lengths of the terms but has nothing to do with who governed, so the gap it "
               "gives is pure chance (grey bars).\nBlack line: the gap with the real calendar. A real gap "
               "beyond about 95% of the fake ones would be hard to call luck. Gaps are averages of daily "
               "price changes (no dividends), per 252 trading days.")
    return dict(tests=tests, rolling=rolling)


# ------------------------------------------------------------------ all
def main():
    dow = market_data.returns("dow")
    made = [fig13_long_view(dow), fig14_every_presidency(dow)]
    loaded = {"dow": dow, "spx_1928": market_data.returns("spx_1928"), "spx": market_data.returns("spx")}
    markets = {f"{MARKET_NAMES[m]}, {r.index[0].year}–{r.index[-1].year}": r for m, r in loaded.items()}
    made.append(fig15_luck(markets))
    for t in made[2]["tests"].items():
        print(f"{t[0]}: gap {t[1]['real']:+.2f}% a year, percentile {t[1]['percentile']:.1f}")
    print("saved fig13, fig14, fig15 in", FIG)


if __name__ == "__main__":
    main()
