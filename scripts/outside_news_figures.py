"""Outside news (extension): commodities, interest rates, the dollar and the Fed
(figures/fig16-fig18*.png), and the numbers behind them.

Run:  python3 scripts/outside_news_figures.py        (about half a minute on 8 cores)

    fig16  how the S&P 500 moves with oil, gold, copper, the dollar and interest rates
    fig17  take out what outside news explains: do the fat tails and memory survive?
    fig18  the four trading moments of the S&P 500, with error bars, against the model

The question (see herding/factors.py): some news reaches every trader at once and a
logical trader acts on it with no herding at all. Does such news, rather than herding,
make the S&P's fat tails and volatility clustering?
Every number printed on a figure is computed here from the data, never typed in.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # the project folder, so `herding` imports

from multiprocessing import Pool

import numpy as np

from herding import calibrate, experiments, factors, market_data, plots, stats
from herding.plots import BLUE, INK, MUTED, ORANGE, REAL, SECOND, SPX_MODEL, plt

FIG = market_data.ROOT / "figures"
ALL = ("oil", "gold", "copper", "dollar", "rates")
NAMES = {"oil": "oil (Brent)", "gold": "gold", "copper": "copper", "dollar": "US dollar",
         "rates": "10-year interest rate"}
WINDOW = ("oil", "gold", "copper", "rates")      # 2000-: longer than with the dollar (2006-)
SEEDS = calibrate.HELD_OUT_SEEDS[:40]
SCALE_FREE = ("kurtosis", "alpha_hill", "alpha_reg", "ac_abs_1", "ac_abs_50", "H_abs", "kurt_5", "kurt_25")
SHORT = {"kurtosis": "kurtosis", "alpha_hill": "tail α (Hill)", "alpha_reg": "tail α (paper's way)",
         "ac_abs_1": "clustering, next day", "ac_abs_50": "clustering, 50 days", "H_abs": "memory H of |r|",
         "kurt_5": "kurtosis, 5 days", "kurt_25": "kurtosis, 25 days"}
CORR_FACTORS = ("oil", "rates", "gold")          # the rolling-correlation panels


def model_runs(days, seeds=SEEDS, pool=None):
    """Held-out histories of the S&P-calibrated model, `days` long; runaways are dropped
    (and counted by the caller from the length)."""
    out = experiments.histories(calibrate.load("spx"), seeds, days, pool=pool)
    return [h["r"] for h in out if h is not None]


def percentiles(value_fp, sims):
    """The percentile of each scale-free measure of `value_fp` among the model histories."""
    return {k: experiments.percentile(value_fp[k], np.array([f[k] for f in sims])) for k in SCALE_FREE}


# ------------------------------------------------------------------ Fig 16
def fig16_comovement(spx, path=FIG / "fig16_sp500_with_commodities_and_rates.png"):
    """Correlations (all days vs crash days), rolling correlations, and the share of S&P
    moves the outside factors explain in calm and wild markets."""
    b = factors.betas(spx, ALL)
    crash = {n: factors.crash_corr(spx, factors.changes(n)) for n in ALL}
    regimes = factors.r2_by_regime(spx, ALL)
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 1), hspace=0.42, wspace=0.28)

    a = fig.add_subplot(gs[0, 0])
    y = np.arange(len(ALL))
    a.barh(y + 0.2, [crash[n][1] for n in ALL], 0.38, color=MUTED, label="other days")
    a.barh(y - 0.2, [crash[n][0] for n in ALL], 0.38, color=INK, label="the S&P's worst 5% of days")
    a.set_yticks(y, [NAMES[n] for n in ALL])
    a.invert_yaxis()
    a.axvline(0, color=SECOND, lw=0.9)
    a.set_xlabel("correlation with the S&P 500 (same day)")
    a.set_title("a. Moving together, and in crashes", loc="left")
    a.legend(loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=2, fontsize=8)
    a.grid(axis="y", visible=False)

    for i, n in enumerate(CORR_FACTORS):
        a = fig.add_subplot(gs[0, 1 + i] if i < 2 else gs[1, 0])
        rc = factors.rolling_corr(spx, factors.changes(n))
        a.axhline(0, color=SECOND, lw=0.9)
        a.fill_between(rc.index, 0, rc.values, where=rc.values >= 0, color=BLUE, alpha=0.25, lw=0, interpolate=True)
        a.fill_between(rc.index, 0, rc.values, where=rc.values <= 0, color=ORANGE, alpha=0.3, lw=0, interpolate=True)
        a.plot(rc.index, rc.values, color=INK, lw=1)
        a.set_ylim(-0.8, 0.8)
        a.set_ylabel("1-year correlation")
        pos = float((rc.values > 0).mean())
        a.set_title(f"{'bcd'[i]}. S&P with {NAMES[n]}\n(positive in {pos:.0%} of years)", loc="left")

    a = fig.add_subplot(gs[1, 1])
    keys = ("calm", "middle", "wild")
    bars = a.bar(keys, [100 * regimes[k] for k in keys], color=[MUTED, SECOND, INK], width=0.6)
    a.bar_label(bars, fmt="%.0f%%", color=INK, fontsize=9)
    a.set_ylabel("share of the S&P's daily variance explained")
    a.set_xlabel("market, by its volatility over the 20 days before")
    a.set_title("e. Outside markets explain more in wild times", loc="left")
    a.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    a.grid(axis="x", visible=False)

    a = fig.add_subplot(gs[1, 2])
    a.axis("off")
    rows = [f"{NAMES[n]:<24}{b[n]['corr']:+.2f}   {b[n]['spx_beta']:+.2f}" for n in ALL]
    a.text(0, 1, "f. One number per factor (2006-2026)\n\n" + f"{'':<24}corr   S&P beta\n" + "\n".join(rows)
           + "\n\nS&P beta: the S&P's move per 1 unit of\nthe factor's move (per 1 point of yield\n"
             "for rates, per 1% for the others).",
           family="DejaVu Sans Mono", fontsize=9, va="top", color=INK)
    plots.save(fig, path, "Fig 16. Commodities, interest rates and the dollar: how the S&P 500 moves with them",
               "Daily changes on days all markets traded (2006-2026 for a, e, f; each factor's own full history in "
               "b-d). Copper and oil rise with stocks; the dollar falls when stocks rise; gold hardly moves with "
               "them, even in crashes (a 'safe haven', Baur & Lucey 2010). The stock-oil and stock-interest-rate "
               "links changed sign over the years, so no fixed beta describes them.\nIn e, the factors (each on the "
               "day before, the day itself and the day after; Dimson 1979) explain more of the S&P's moves in wild "
               "markets. That does not mean news causes the crashes: in a panic the S&P also moves oil and rates.")
    return dict(betas=b, crash=crash, regimes=regimes)


# ------------------------------------------------------------------ Fig 17
def fig17_without_outside_news(spx, pool, path=FIG / "fig17_without_outside_news.png", seeds=SEEDS):
    """Remove every move the factors explain, and Fed decision days: are the fat tails and
    the memory still there, and is the S&P still a typical model history?"""
    resid, r2 = factors.remove(spx, WINDOW)
    raw = spx.loc[resid.index]
    sims = [stats.fingerprint(r) for r in model_runs(len(resid), seeds, pool)]
    fp_raw, fp_res = stats.fingerprint(raw.values), stats.fingerprint(resid.values)
    pct_raw, pct_res = percentiles(fp_raw, sims), percentiles(fp_res, sims)
    since94 = spx[spx.index >= "1994-02-01"]
    fed = {k: factors.event_days(since94, factors.on_days(since94.index, factors.decisions(k)["date"]))
           for k in ("scheduled", "unscheduled")}
    fed["all"] = factors.event_days(since94, factors.on_days(since94.index, factors.decisions()["date"]))

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(2, 3, width_ratios=(1, 1.25, 1), hspace=0.45, wspace=0.3)
    a = fig.add_subplot(gs[:, 0])
    plots.ccdf(a, raw.values, "S&P 500", REAL, points=40)
    plots.ccdf(a, resid.values, "S&P with outside news removed", REAL, points=40, mfc="white")
    plots.gaussian_ccdf(a)
    a.set_xlim(0.3, 20)
    a.set_ylim(1e-4, 1.2)
    a.set_xlabel("|return| / standard deviation")
    a.set_ylabel("share of days with a bigger move")
    a.set_title(f"a. Outside news explains {r2:.0%} of the\nS&P's daily variance; the tail stays fat", loc="left")
    a.legend(loc="lower left")

    a = fig.add_subplot(gs[:, 1])
    y = np.arange(len(SCALE_FREE))
    a.axvspan(5, 95, color=SPX_MODEL, alpha=0.1, lw=0)
    for k, yy in zip(SCALE_FREE, y):
        a.plot([pct_raw[k], pct_res[k]], [yy, yy], color=MUTED, lw=1, zorder=1)
    a.plot([pct_raw[k] for k in SCALE_FREE], y, "o", color=REAL, ms=8, label="S&P 500", zorder=2)
    a.plot([pct_res[k] for k in SCALE_FREE], y, "o", color=REAL, mfc="white", ms=8,
           label="outside news removed", zorder=3)
    a.set_yticks(y, [SHORT[k] for k in SCALE_FREE])
    a.invert_yaxis()
    a.set_xlim(-3, 103)
    a.set_xlabel(f"percentile among {len(sims)} model histories (blue band: typical, 5-95%)")
    ok = (sum(5 <= pct_raw[k] <= 95 for k in SCALE_FREE), sum(5 <= pct_res[k] <= 95 for k in SCALE_FREE))
    a.set_title(f"b. Still a typical history of the herding model?\n"
                f"typical on {ok[0]} of {len(SCALE_FREE)} measures, then {ok[1]} of {len(SCALE_FREE)}", loc="left")
    a.legend(loc="lower right", fontsize=8.5)
    a.grid(axis="y", visible=False)

    a = fig.add_subplot(gs[0, 2])
    kinds = ("scheduled", "unscheduled")
    bars = a.bar([f"scheduled\n({fed['scheduled']['days']} days)", f"surprise\n({fed['unscheduled']['days']} days)"],
                 [fed[k]["size_ratio"] for k in kinds], color=[MUTED, INK], width=0.55)
    a.bar_label(bars, fmt="%.1f×", color=INK, fontsize=9)
    a.axhline(1, color=SECOND, lw=0.9, ls="--")
    a.set_ylabel("move size vs. other days")
    a.set_title("c. Fed decision days, 1994-2026", loc="left")
    a.grid(axis="x", visible=False)

    a = fig.add_subplot(gs[1, 2])
    a.axis("off")
    f = fed["all"]
    a.text(0, 1, f"d. Leaving out all {f['days']} Fed days\n\n"
                 f"{'':<16}with   without\n"
                 f"{'kurtosis':<16}{f['kurtosis_all']:5.1f}   {f['kurtosis_without']:5.1f}\n"
                 f"{'tail α (Hill)':<16}{f['alpha_all']:5.2f}   {f['alpha_without']:5.2f}\n\n"
                 f"They are {f['chance_share']:.0%} of days and {f['big_share']:.0%}\nof the biggest 1% of days.",
           family="DejaVu Sans Mono", fontsize=9.5, va="top", color=INK)
    plots.save(fig, path, "Fig 17. Take out the outside news: the fat tails and the memory survive",
               f"{resid.index[0]:%Y}-{resid.index[-1]:%Y} ({len(resid):,} days). 'Outside news removed': the S&P's "
               f"daily returns minus the part explained by oil, gold, copper and the 10-year interest rate on the "
               "day before, the day itself and the day after. This removes too much rather than too little, because "
               "part of what moves with those markets is the stock market's own panic.\nIn b each dot is the "
               "S&P's percentile among held-out histories of the S&P-calibrated model of the same length. "
               "Surprise Fed moves are huge days, but there are only 9; scheduled decisions are ordinary days.")
    return dict(r2=r2, days=len(resid), raw=fp_raw, resid=fp_res, pct_raw=pct_raw, pct_res=pct_res,
                histories=len(sims), fed=fed)


# ------------------------------------------------------------------ Fig 18
def fig18_moments(spx, pool, path=FIG / "fig18_trading_moments_with_error_bars.png", seeds=SEEDS, draws=1000):
    """Mean, spread, skewness and kurtosis of the S&P with 95% block-bootstrap intervals,
    against the same four numbers in model histories of the same length."""
    m = stats.moments(spx.values, draws=draws)
    runs = model_runs(len(spx), seeds, pool)
    sims = {k: np.array([f(r) for r in runs]) for k, f in
            (("mean", np.mean), ("sd", np.std), ("skew", stats.skewness), ("kurtosis", stats.kurtosis))}
    fig, ax = plt.subplots(1, 4, figsize=(15, 4.6))
    pct = {}
    scale = {"mean": 1e4, "sd": 100, "skew": 1, "kurtosis": 1}
    unit = {"mean": "basis points a day", "sd": "% a day", "skew": "", "kurtosis": ""}
    for a, k in zip(ax, stats.MOMENTS):
        v, lo, hi = (x * scale[k] for x in m[k])
        s = sims[k] * scale[k]
        pct[k] = experiments.percentile(m[k][0], sims[k])
        jitter = np.random.default_rng(0).uniform(-1, 1, len(s))
        a.plot(s, jitter, "o", color=SPX_MODEL, ms=4, alpha=0.7, label="model histories")
        a.axvspan(lo, hi, color=REAL, alpha=0.12, lw=0, label="S&P 95% interval")
        a.axvline(v, color=REAL, lw=2, label="S&P 500")
        if k == "kurtosis":
            a.set_xscale("symlog", linthresh=1)
            a.set_xlim(-0.5, 1.5 * max(s.max(), hi))
        else:                                  # a freak history (near runaway) would squash the rest
            left = min(lo, np.percentile(s, 2.5))
            right = max(hi, np.percentile(s, 97.5))
            pad = 0.15 * (right - left)
            a.set_xlim(left - pad, right + pad)
            off = int(np.sum((s < left - pad) | (s > right + pad)))
            if off:
                a.text(0.99, 0.02, f"{off} history off the scale" if off == 1 else f"{off} histories off the scale",
                       transform=a.transAxes, ha="right", fontsize=7.5, color=SECOND)
        a.set_yticks([])
        a.set_ylim(-1.6, 1.6)
        a.grid(axis="y", visible=False)
        a.set_xlabel(unit[k])
        a.set_title(f"{k}: S&P {v:.2f} ({lo:.2f} to {hi:.2f})\nat {pct[k]:.0f}% of the model", loc="left",
                    fontsize=9.5)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0.1), ncol=3,
               fontsize=8.5)
    fig.subplots_adjust(top=0.78, bottom=0.36, left=0.02, right=0.99, wspace=0.12)
    plots.save(fig, path, "Fig 18. The four trading moments, with error bars: the model is symmetric, the S&P is not",
               f"S&P 500, {spx.index[0]:%Y}-{spx.index[-1]:%Y}. Shaded: 95% interval from a block bootstrap (resampled "
               "one-year blocks, so wild spells stay together). Dots: held-out histories of the S&P-calibrated model, "
               "same length. The S&P falls harder than it rises (negative skew); the model's traders behave the "
               "same in both directions. The model's value has no long-run growth, so its mean is 0 by design. "
               "The kurtosis interval is very wide: one day (Black Monday) dominates it.")
    return dict(moments=m, percentile=pct, histories=len(runs))


# ------------------------------------------------------------------ all
def main():
    plots.style()
    spx = market_data.spx_returns()
    with Pool() as pool:
        out = dict(fig16=fig16_comovement(spx), fig17=fig17_without_outside_news(spx, pool),
                   fig18=fig18_moments(spx, pool))
    f17 = out["fig17"]
    print(f"Outside news explains {f17['r2']:.1%} of the S&P's daily variance ({f17['days']:,} days)")
    print(f"{'measure':<22}{'S&P':>8}{'removed':>9}{'pct':>6}{'pct rm':>8}")
    for k in SCALE_FREE:
        print(f"{SHORT[k]:<22}{f17['raw'][k]:8.3f}{f17['resid'][k]:9.3f}{f17['pct_raw'][k]:6.0f}{f17['pct_res'][k]:8.0f}")
    for k, v in f17["fed"].items():
        print(f"Fed {k}: {v}")
    print("R^2 by regime:", {k: round(v, 3) for k, v in out["fig16"]["regimes"].items()})
    print("moments:", out["fig18"]["moments"], out["fig18"]["percentile"])
    print("saved fig16, fig17, fig18 in", FIG)
    return out


if __name__ == "__main__":
    main()
