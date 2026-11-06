"""Make every results figure (figures/fig1-11*.png), explained in figures/RESULTS.md.

Run:  python3 make_figures.py        (a few minutes with numba, using every core)

Two fitted settings are used (both from python3 -m herding.calibrate):
    paper-matched  settings/paper.json  reproduces Lux & Marchesi (1999)'s numbers
    S&P-calibrated settings/spx.json    reproduces the S&P 500's fingerprint
Every model check uses held-out seeds that neither fit ever saw.
"""
from multiprocessing import Pool

import numpy as np

from herding import calibrate, experiments, kirman, market_data, model, plots, policy, stats
from herding.plots import (BLUE, BLUE_LIGHT, GRID, INK, MUTED, NEWS, ORANGE,
                           ORANGE_LIGHT, PAPER_MODEL, REAL, SECOND, SPX_MODEL, plt)

FIG = market_data.ROOT / "figures"
HELD_OUT = calibrate.HELD_OUT_SEEDS            # 40 seeds, never used for fitting
LONG = 100_000                                 # days in a long run (the paper's Figs 2-3)
BREAKER_SEEDS = tuple(range(3000, 3200))       # policy experiments: their own seeds
SURPRISE_SEEDS = tuple(range(4000, 4400))
TAUS = (1, 5, 15, 25)                          # the paper's aggregation horizons
LABEL = {k: k for k in stats.FINGERPRINT}
LABEL.update(sd="daily sd", kurtosis="kurtosis", alpha_hill="tail α (Hill)",
             alpha_reg="tail α (paper's fit)", ac_r_1="next-day corr.",
             ac_abs_1="clustering, 1 day", ac_abs_50="clustering, 50 days",
             H="H of returns", H_abs="H of |returns|", kurt_5="kurtosis, 5 days",
             kurt_25="kurtosis, 25 days")


def shade(hex_color, amount):
    """Mix a colour with white: amount 0 = the colour, 1 = white."""
    c = np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    return tuple(c + (1 - c) * amount)


# ------------------------------------------------------------------ Fig 1
def fig1_snapshot(paper_run, spx_run, spx, path=FIG / "fig1_snapshot.png", days=4000):
    """The paper's Fig 1: price tracks its value, but returns are wilder than the news.
    Paper conventions: 4,000 days; price solid, value dotted and shifted down for
    visibility; output (b) and input (c) on the same y range. Added: the chartist
    share (d) and the last 4,000 days of the real S&P 500 (right)."""
    fig, ax = plt.subplots(4, 3, figsize=(17, 10), sharex=True)
    t = np.arange(days + 1)
    for col, (run, color, name) in enumerate([(paper_run, PAPER_MODEL, "paper-matched model"),
                                              (spx_run, SPX_MODEL, "S&P-calibrated model")]):
        lp, lf, z = run["lp"][:days + 1], run["lf"][:days + 1], run["z"][:days + 1]
        a = ax[0, col]
        price = 100 * np.exp(lp - lp[0])
        gap = 5 * np.ceil(0.1 * np.ptp(price))        # the paper's vertical shift, in whole 5s
        a.plot(t, price, color=color, lw=1.0, label="price p")
        a.plot(t, 100 * np.exp(lf - lp[0]) - gap, ":", color=INK, lw=1.0,
               label=f"fundamental value p_f (drawn {gap:.0f} lower)")
        a.set_title(f"a. Price and value ({name})")
        lo_, hi_ = min(price.min(), (100 * np.exp(lf - lp[0]) - gap).min()), price.max()
        a.set_ylim(lo_ - 0.05 * (hi_ - lo_), hi_ + 0.4 * (hi_ - lo_))     # room for the legend
        a.legend(loc="upper left", frameon=True, facecolor="white", edgecolor=GRID, framealpha=0.9)
        r, eps = np.diff(lp), np.diff(lf)
        lim = 1.1 * max(np.abs(r).max(), np.abs(eps).max())
        ax[1, col].plot(t[1:], r, color=color, lw=0.4)
        ax[1, col].set_title("b. The output: daily returns ln p_t − ln p_t−1")
        ax[2, col].plot(t[1:], eps, color=NEWS, lw=0.4)
        ax[2, col].set_title("c. The input: news ε_t = ln p_f,t − ln p_f,t−1 (Gaussian)")
        for a, x in ((ax[1, col], r), (ax[2, col], eps)):
            a.set_ylim(-lim, lim)
            a.text(0.01, 0.92, f"kurtosis {stats.kurtosis(x):.1f}", transform=a.transAxes,
                   fontsize=8.5, color=INK, va="top")
        ax[3, col].fill_between(t, 0, z, color=shade(color, 0.35), lw=0)
        ax[3, col].set_ylim(0, 1)
        ax[3, col].set_title("d. Share of traders who are chartists")
        ax[3, col].set_xlabel("day")
    r = np.asarray(spx[-days:])
    ax[0, 2].plot(t, 100 * np.exp(np.concatenate([[0], np.cumsum(r)])), color=REAL, lw=1.0, label="price")
    ax[0, 2].set_title(f"a. Real S&P 500, last {days:,} trading days")
    ax[0, 2].legend(loc="upper left")
    ax[1, 2].plot(t[1:], r, color=REAL, lw=0.4)
    ax[1, 2].set_ylim(-1.1 * np.abs(r).max(), 1.1 * np.abs(r).max())
    ax[1, 2].set_title("b. Daily returns (its own y scale)")
    ax[1, 2].text(0.01, 0.92, f"kurtosis {stats.kurtosis(r):.1f}", transform=ax[1, 2].transAxes,
                  fontsize=8.5, color=INK, va="top")
    ax[2, 2].text(0.02, 0.5, "A real market's news and its share of\nchartists cannot be measured,\n"
                  "so panels c and d exist only\nfor the model.", fontsize=10.5, color=SECOND, va="center")
    ax[2, 2].axis("off")
    ax[3, 2].axis("off")
    ax[1, 2].tick_params(labelbottom=True)
    ax[1, 2].set_xlabel("trading day")
    for row, label in enumerate(["level (start = 100)", "log change", "log change", "share"]):
        ax[row, 0].set_ylabel(label)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return plots.save(fig, path, "Fig 1. The price follows its value, but returns are far wilder than the news",
                      f"A {days:,}-day stretch of one held-out history for each setting, drawn as in Lux & Marchesi's "
                      "Fig 1 (value dotted and shifted down so both lines show; b and c on the same scale). The news "
                      "(c) is a plain bell curve with no calm or wild spells; the returns (b) have bursts of large "
                      "moves, and the bursts come when many traders are chartists (d, not in the paper). The "
                      "S&P-calibrated model keeps few chartists most of the time, so its bursts are rarer (Fig 5).")


# ------------------------------------------------------------------ Fig 2
TAU_MARKERS = ("^", "+", "D", "v")              # the paper's Fig 2 symbols for tau = 1, 5, 15, 25
PAPER_ALPHA = 2.64                               # Lux & Marchesi (1999) Fig 2, tau = 1


def fig2_aggregation(paper_r, paper_news, spx_model_runs, spx, path=FIG / "fig2_tails.png",
                     paper_runs=None):
    """The paper's Fig 2: fat tails at 1 day, a cross-over to the bell curve at longer horizons.
    Paper conventions: log-log share of |ret|/sd beyond x for tau = 1, 5, 15, 25 as markers,
    the bell curve as a solid line, alpha from a log-log fit to the largest 30% at tau = 1."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4), sharey=True)
    pooled = np.concatenate(spx_model_runs)
    panels = [(paper_r, paper_runs or [paper_r], PAPER_MODEL, f"a. Paper-matched model (one run, {len(paper_r):,} days)"),
              (pooled, spx_model_runs, SPX_MODEL, f"b. S&P-calibrated model ({len(spx_model_runs)} histories pooled)"),
              (spx, [spx], REAL, f"c. Real S&P 500, 1950–2026 ({len(spx):,} days)")]
    for a, (r, runs, color, title) in zip(ax, panels):
        shown = [r] if r is paper_r else runs        # kurtosis of the run drawn, or the median of histories
        for i, tau in enumerate(TAUS):
            k = np.median([stats.kurtosis(stats.aggregate(h, tau)) for h in shown])
            word = "median kurtosis" if len(shown) > 1 else "kurtosis"
            plots.ccdf(a, stats.aggregate(r, tau), f"τ = {tau:>2}: {word} {k:5.1f}", shade(color, 0.5 * i / 3),
                       points=45, marker=TAU_MARKERS[i], ms=5 if TAU_MARKERS[i] == "+" else 4, mew=1.2)
        if r is paper_r:
            plots.ccdf(a, paper_news, "news, τ = 1", NEWS, lw=1.2, ls="-.")
        plots.gaussian_ccdf(a, ls="-")
        top = a.legend(loc="upper right", fontsize=8)
        alpha, c, lo, hi = stats.tail_fit(r)
        fit = plots.power_law(a, lo, c * lo ** -alpha, -alpha, hi, f"fit to largest 30%: α = {alpha:.2f}",
                              color=INK, ls=":", lw=1.6)
        ref = plots.power_law(a, lo, c * lo ** -alpha, -PAPER_ALPHA, hi, f"Lux & Marchesi: α = {PAPER_ALPHA}")
        head = "τ = 1 tail"
        if len(runs) > 1:
            alphas = [stats.alpha_regression(h) for h in runs]
            head += (f" (α, mean of {len(runs)} runs: {np.mean(alphas):.2f})" if r is paper_r
                     else f" (α, median of {len(runs)} histories: {np.median(alphas):.2f})")
        a.legend(handles=[fit, ref], loc="lower left", fontsize=8, title=head, title_fontsize=8, alignment="left")
        a.add_artist(top)
        a.set_xlim(0.3, 40)
        a.set_ylim(1e-5, 1.5)
        a.set_title(title)
        a.set_xlabel("|return| / standard deviation")
    ax[0].set_ylabel("share of periods with a bigger move")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    return plots.save(fig, path, "Fig 2. Fat tails over one day, closer to a bell curve over 25 days",
                      "Drawn as in Lux & Marchesi's Fig 2: markers show the share of τ-day periods whose move (in "
                      "standard deviations) beats x; solid grey = bell curve. Dotted: the paper's way of measuring the "
                      "tail, a straight-line fit to the largest 30% of daily moves (tail α = minus its slope); dashed: "
                      "the slope of the paper's α = 2.64, for comparison. As τ grows the curves cross over to the bell "
                      "curve. The news (dash-dot, not drawn in the paper) is a bell curve.")


# ------------------------------------------------------------------ Fig 3
PAPER_H = dict(r=0.48, abs_r=0.85, news=0.49, abs_news=0.51)     # Lux & Marchesi (1999) Fig 3


def fig3_memory(paper_r, paper_news, spx_model_r, spx, path=FIG / "fig3_memory.png"):
    """The paper's Fig 3: DFA. Returns have no memory (H ~ 0.5); their sizes do (H ~ 0.85).
    Paper conventions: log-log F(n) against window n; output as triangles with a solid fit,
    news as plus signs with a dotted (a) or dashed (b) fit, news shifted in b. Added: a
    grey reference line with the paper's slope, and panel c (the S&P 500)."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.2))
    s_long = stats.dfa_scales(len(paper_r))
    s_spx = stats.dfa_scales(len(spx))
    for a, f, key, title in [(ax[0], np.abs, "abs_", "a. Size of moves: long memory in the output only"),
                             (ax[1], lambda x: x, "", "b. Direction of moves: no memory in either")]:
        _, n0, F0 = plots.dfa(a, f(paper_r), s_long, "|returns|" if key else "returns", PAPER_MODEL, marker="^")
        plots.dfa(a, f(paper_news), s_long, f"{'|news|' if key else 'news'} (paper {PAPER_H[key + 'news']})",
                  NEWS, marker="+", fit_ls=":" if key else "--", shift=1.0 if key else 3.0)
        plots.power_law(a, n0, F0, PAPER_H[key + "r"], s_long[-1], f"Lux & Marchesi: H = {PAPER_H[key + 'r']}")
        a.set_title(title)
    a = ax[2]
    _, n0, F0 = plots.dfa(a, np.abs(spx), s_spx, "S&P |returns|", REAL)
    plots.dfa(a, np.abs(spx_model_r), s_spx, "model |returns|", SPX_MODEL, marker="^")
    plots.power_law(a, n0, F0, PAPER_H["abs_r"], s_spx[-1], f"Lux & Marchesi's model: H = {PAPER_H['abs_r']}")
    _, n0, F0 = plots.dfa(a, spx, s_spx, "S&P returns", REAL, marker="D", shift=0.2)
    plots.dfa(a, spx_model_r, s_spx, "model returns", SPX_MODEL, marker="v", shift=0.2)
    plots.power_law(a, n0, F0, 0.5, s_spx[-1], "no memory: H = 0.5", ls=":")
    a.set_title("c. The S&P 500 and the S&P-calibrated model")
    for a in ax:
        a.set_xlabel("window n (days)")
        a.legend(loc="upper left", fontsize=8)
    ax[0].set_ylabel("fluctuation F(n)")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return plots.save(fig, path, "Fig 3. Moves have no memory of direction, but a long memory of size",
                      "Detrended fluctuation analysis (Peng et al. 1994), drawn as in Lux & Marchesi's Fig 3: F(n) grows "
                      "like n^H, so H is the slope. H = 0.5 means no memory; H near 1 means calm and wild spells that "
                      "last. Grey dashed: the paper's slope, for comparison. Panels a-b: one long paper-matched run (the "
                      "output) and its news (the input; shifted up in b). Panel c: one held-out history the length of "
                      "the S&P (returns curves shifted down).")


# ------------------------------------------------------------------ Fig 4
def fig4_checks(no_chart, spx_runs, paper_runs, spx, path=FIG / "fig4_checks.png"):
    """The proposal's checks: no chartists -> bell curve; herding -> alpha ~ 3 like the S&P."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))
    a = ax[0]
    plots.ccdf(a, np.concatenate(no_chart), "no chartists", NEWS)
    plots.ccdf(a, np.concatenate(spx_runs), "with chartists (S&P settings)", SPX_MODEL)
    plots.ccdf(a, spx, "S&P 500", REAL)
    plots.gaussian_ccdf(a)
    a.set_xlim(0.05, 40)
    a.set_ylim(1e-6, 1.5)
    a.set_xlabel("|return| / standard deviation")
    a.set_ylabel("share of days with a bigger move")
    a.set_title("a. Without chartists the tails are a bell curve")
    a.legend(loc="lower left")
    groups = [("no chartists", no_chart, NEWS), ("paper-matched", paper_runs, PAPER_MODEL),
              ("S&P-calibrated", spx_runs, SPX_MODEL)]
    for a, measure, title, ref in [
            (ax[1], lambda r: stats.kurtosis(r), "b. Excess kurtosis (0 = bell curve)", 0.0),
            (ax[2], lambda r: stats.hill_alpha(r), "c. Tail exponent α (Hill, top 5%)", 3.0)]:
        ticks = []
        for i, (name, runs, color) in enumerate(groups):
            v = np.array([measure(r) for r in runs])
            jitter = np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
            a.plot(i + jitter, v, "o", color=color, ms=4, alpha=0.75)
            a.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=INK, lw=1.6)
            ticks.append(f"{name}\nmedian {np.median(v):.2f}")
        real = measure(spx)
        a.axhline(real, color=REAL, lw=1.2, label=f"S&P 500: {real:.2f}")
        a.axhline(ref, color=MUTED, ls="--", lw=1,
                  label="bell curve: 0" if ref == 0 else "inverse cubic law: α = 3")
        a.set_xticks(range(3), ticks)
        a.set_xlim(-0.5, 2.5)
        a.set_title(title)
        a.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID)
    ax[1].set_yscale("symlog", linthresh=1)
    ax[1].set_ylim(-0.5, None)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return plots.save(fig, path, "Fig 4. The proposal's checks: no herding, no fat tails; with herding, α ≈ 3",
                      "Dots: held-out histories the length of the S&P 500 (40 per group; 8 long runs for "
                      "paper-matched); bars: medians. With no chartists the returns are just news, so kurtosis is "
                      "about 0 and α is large (thin tails). With chartists both fitted models have fat tails.")


# ------------------------------------------------------------------ Fig 5
def fig5_mechanism(run, path=FIG / "fig5_mechanism.png", window=1500):
    """Where bursts come from: volatility rises with the chartist share."""
    r, z = run["r"], run["z"][1:]
    vol = np.abs(r) / r.std()
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.6, 1, 1))
    peak = int(np.argmax(np.convolve(vol, np.ones(50), "same")))
    lo = max(0, min(peak - window // 2, len(r) - window))
    t = np.arange(lo, lo + window)
    a = fig.add_subplot(gs[0, 0])
    a.plot(t, r[t], color=SPX_MODEL, lw=0.5)
    a.set_title("a. The wildest stretch of one history: returns")
    a.set_ylabel("daily return")
    b = fig.add_subplot(gs[1, 0], sharex=a)
    b.fill_between(t, 0, z[t], color=BLUE_LIGHT, lw=0)
    b.plot(t, z[t], color=SPX_MODEL, lw=0.8)
    b.set_ylim(0, 1)
    b.set_title("…and the share of chartists at the same time")
    b.set_xlabel("day")
    edges = np.linspace(0, 1, 21)
    mid = (edges[:-1] + edges[1:]) / 2
    which = np.clip(np.digitize(z, edges) - 1, 0, 19)
    count = np.bincount(which, minlength=20)
    mean_vol = np.array([vol[which == i].mean() if count[i] >= 30 else np.nan for i in range(20)])
    c = fig.add_subplot(gs[:, 1])
    c.plot(mid, mean_vol, "o-", color=SPX_MODEL)
    c.axhline(1, color=MUTED, ls="--", lw=1)
    c.set_xlabel("share of chartists")
    c.set_ylabel("average |return| / overall sd")
    c.set_title("b. More chartists, wilder days")
    d = fig.add_subplot(gs[:, 2])
    d.bar(mid, count / count.sum(), width=0.045, color=BLUE_LIGHT, edgecolor=SPX_MODEL, lw=0.8)
    d.set_xlabel("share of chartists")
    d.set_ylabel("share of days")
    d.set_title("c. Crowded markets are rare")
    top = min(1.0, edges[np.nonzero(count)[0].max() + 1] + 0.05)
    for axis in (c, d):
        axis.set_xlim(0, top)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    return plots.save(fig, path, "Fig 5. The mechanism: bursts happen when the crowd turns to chartism",
                      "S&P-calibrated settings, one long held-out run. Lux & Marchesi: volatility is above average "
                      "when the share of noise traders nears a critical value, and the bursts end on their own when "
                      "large mispricing lures traders back to fundamentals (a: the share falls again after the burst).")


# ------------------------------------------------------------------ Fig 6
def fig6_kirman(path=FIG / "fig6_kirman.png"):
    """Kirman's ants: the herding mechanism, checked against his exact solution."""
    cases = [("a. Herding: ε < (1−δ)/(N−1)", dict(N=100, eps=0.002, delta=0.1)),
             ("b. No herding: ε > (1−δ)/(N−1)", dict(N=100, eps=0.05, delta=0.1))]
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    for col, (name, kw) in enumerate(cases):
        k = kirman.simulate(**kw, steps=40_000_000, seed=1)
        show = k[:5_000_000:2_000]
        ax[0, col].plot(np.arange(len(show)) * 2_000 / 1e6, show, color=BLUE, lw=0.8)
        ax[0, col].set_ylim(0, 100)
        ax[0, col].set_ylabel("ants at source A")
        ax[0, col].set_xlabel("steps (millions; first 5 of 40 shown)")
        ax[0, col].set_title(name)
        hist = np.bincount(k, minlength=101) / len(k)
        exact = kirman.stationary(**kw)
        ax[1, col].bar(np.arange(101), hist, width=0.9, color=BLUE_LIGHT, label="simulation (40M steps)")
        ax[1, col].plot(np.arange(101), exact, color=INK, lw=1.6, label="Kirman's exact answer")
        tv = 0.5 * np.abs(hist - exact).sum()
        ax[1, col].set_title(f"mismatch (total variation distance) = {tv:.3f}", fontsize=9.5)
        ax[1, col].set_xlabel("ants at source A")
        ax[1, col].set_ylabel("share of time")
        ax[1, col].set_ylim(0, 1.35 * max(hist.max(), exact.max()))     # room for the legend
        ax[1, col].legend(loc="upper center")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    return plots.save(fig, path, "Fig 6. Kirman's ants herd exactly as his math predicts",
                      "100 ants choose between two identical food sources (δ = 0.1). Rarely switching on their own "
                      "(a), they crowd at one source and then flip all at once; otherwise (b) they split about 50/50. "
                      "Bottom: time spent in each state (bars) vs the exact long-run answer (line). Lux & Marchesi "
                      "use the same recruitment rule for traders joining or leaving the chartists.")


# ------------------------------------------------------------------ Fig 7
def fig7_typical(spx_runs, spx, path=FIG / "fig7_typical.png"):
    """Is the S&P 500 a typical history of the calibrated model, measure by measure?"""
    target = stats.fingerprint(spx)
    sims = [stats.fingerprint(r) for r in spx_runs]
    keys = list(stats.FINGERPRINT)
    fig, ax = plt.subplots(3, 4, figsize=(15, 8.5))
    pct = {}
    for a, k in zip(ax.flat, keys):
        v = np.array([f[k] for f in sims])
        pct[k] = experiments.percentile(target[k], v)
        jitter = np.random.default_rng(0).uniform(-1, 1, len(v))
        a.plot(v, jitter, "o", color=SPX_MODEL, ms=4, alpha=0.7)
        a.axvline(target[k], color=REAL, lw=2)
        a.set_yticks([])
        a.set_ylim(-1.6, 1.6)
        if k.startswith("kurt"):
            a.set_xscale("symlog", linthresh=1)           # one freak history would squash the rest
            a.set_xlim(-0.5, 1.5 * max(v.max(), target[k]))
        unusual = pct[k] < 5 or pct[k] > 95
        a.set_title(f"{LABEL[k]}: S&P at {pct[k]:.0f}%" + ("  ← unusual" if unusual else ""),
                    fontsize=9.5, color=INK)
        a.grid(axis="y", visible=False)
    ax.flat[-1].axis("off")
    ok = sum(5 <= p <= 95 for p in pct.values())
    ax.flat[-1].text(0, 0.5, f"{ok} of {len(keys)} measures:\nthe S&P sits inside the\nmiddle 90% of model\nhistories",
                     fontsize=12, color=INK, va="center")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    plots.save(fig, path, "Fig 7. Does the real S&P 500 look like one more history of the model?",
               f"Dots: {len(sims)} held-out histories of the S&P-calibrated model, each {len(spx):,} days like the "
               "S&P 500 sample; black line: the S&P 500. The percentage is the share of histories below the S&P; "
               "below 5% or above 95% marks a measure where the S&P is unusual for the model. Kurtosis axes are "
               "logarithmic above 1, because one history has a freak event (kurtosis in the thousands).")
    return pct


# ------------------------------------------------------------------ Fig 8
def fig8_out_of_sample(pool, p, path=FIG / "fig8_out_of_sample.png", n=20):
    """Markets the calibration never saw: percentile of each among model histories."""
    keys = [k for k in stats.FINGERPRINT if k != "sd"]     # shape only: volatility differs by asset
    if not 1 <= n <= len(HELD_OUT) // 2:
        raise ValueError(f"n must be between 1 and {len(HELD_OUT) // 2}, got {n}")
    # (name, returns, seeds): the indexes use held-out seeds 1000-1019, the stocks
    # their own, 1020-1039, so the stocks are not compared with shorter copies of
    # the very histories the indexes were compared with.
    series = [("S&P 500 1928–", market_data.returns("spx_1928").values, HELD_OUT[:n]),
              ("Nasdaq-100 1985–", market_data.returns("ndx").values, HELD_OUT[:n])]
    series += [(t, market_data.stock_returns(t).values, HELD_OUT[n:2 * n])
               for t in market_data.stock_tickers()]
    cache, rows = {}, []
    for name, r, seeds in series:
        if (len(r), seeds) not in cache:
            cache[len(r), seeds] = [stats.fingerprint(x)
                                    for x in experiments.returns(p, seeds, len(r), pool=pool)]
        f = stats.fingerprint(r)
        rows.append([experiments.percentile(f[k], [s[k] for s in cache[len(r), seeds]]) for k in keys])
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(13, 9))
    cmap = matplotlib_diverging()
    im = ax.imshow(rows, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            v = rows[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v < 10 or v > 90 else INK)
    ax.set_xticks(range(len(keys)), [LABEL[k] for k in keys], rotation=30, ha="right")
    ax.set_yticks(range(len(series)), [s[0] for s in series], fontsize=8.5)
    ax.grid(False)
    ax.axhline(1.5, color="white", lw=3)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("percentile among model histories (50 = typical)")
    unusual = int(np.sum((rows < 5) | (rows > 95)))
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    plots.save(fig, path, "Fig 8. Markets the model never saw: two indexes and 20 stocks",
               f"Each cell: where a real market sits among {n} S&P-calibrated histories of the same length (0-100). "
               f"Middle colours = typical; dark ends = unusual (below 5 or above 95): {unusual} of {rows.size} cells. "
               "Volatility (sd) is left out: stocks move more than the index by nature. Stocks are price-only "
               "(no dividends) and are today's large companies (survivorship bias).")
    return rows, keys, [s[0] for s in series]


def matplotlib_diverging():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("pct", [BLUE, "#f2f1ec", ORANGE])


# ------------------------------------------------------------------ Fig 9
def _robust(args):
    i, days = args
    p = calibrate.draw(np.random.default_rng(50_000 + i), calibrate.RANGES)
    try:
        r = model.simulate(p, days + calibrate.BURN, seed=7)["r"][calibrate.BURN:]
    except model.RunawayError:
        return None
    return stats.kurtosis(r), stats.alpha_regression(r), stats.hurst(np.abs(r), stats.dfa_scales(days))


def _robust_base(args):
    """The no-chartist yardstick for Fig 9: kurtosis and H of |returns|."""
    i, days = args
    p = calibrate.draw(np.random.default_rng(60_000 + i), calibrate.RANGES)
    r = model.simulate(p, days, seed=7, chartists=False)["r"]
    return stats.kurtosis(r), stats.hurst(np.abs(r), stats.dfa_scales(days))


def fig9_robustness(pool, n=400, days=20_000, path=FIG / "fig9_robustness.png"):
    """The paper: results hold 'for an open set of parameter values'. Do they?"""
    # chunksize=1: run times differ a lot (runaways stop early), so hand out one at a time
    out = pool.map(_robust, [(i, days) for i in range(n)], chunksize=1)
    ok = np.array([o for o in out if o is not None])
    base = np.array(pool.map(_robust_base, [(i, days) for i in range(40)], chunksize=1))
    base_k, base_h = base[:, 0], base[:, 1]
    fat = ok[:, 0] > base_k.max()
    memory = ok[:, 2] > base_h.max()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    a = ax[0]
    a.scatter(ok[~memory, 1], ok[~memory, 2], s=12, color=ORANGE_LIGHT, label="no volatility memory")
    a.scatter(ok[memory, 1], ok[memory, 2], s=12, color=PAPER_MODEL, label="volatility memory")
    a.plot(PAPER_ALPHA, PAPER_H["abs_r"], "*", ms=16, color=INK,
           label=f"Lux & Marchesi (1999): α {PAPER_ALPHA}, H {PAPER_H['abs_r']}")
    a.axhline(base_h.max(), color=NEWS, ls="--", lw=1.2, label="largest value with no chartists")
    a.set_xlabel("tail α (paper's method)")
    a.set_ylabel("H of |returns|")
    a.set_xlim(1, 5.5)      # room on the right for the legend
    a.set_title("a. Every setting that did not run away")
    a.legend(loc="upper right", frameon=True, facecolor="white", edgecolor=GRID)
    b = ax[1]
    counts = [n - len(ok), int(np.sum(~fat)), int(np.sum(fat & ~memory)), int(np.sum(fat & memory))]
    names = ["ran away\n(endless bubble)", "thin tails", "fat tails,\nno memory", "fat tails\nand memory"]
    colors = [MUTED, NEWS, ORANGE_LIGHT, PAPER_MODEL]
    b.bar(range(4), np.array(counts) / n * 100, color=colors, width=0.6)
    for i, c in enumerate(counts):
        b.text(i, c / n * 100, f"{c / n:.0%}", ha="center", va="bottom", fontsize=9.5, color=INK)
    b.set_xticks(range(4), names)
    b.set_ylabel(f"share of {n} random settings (%)")
    b.set_title("b. What random settings produce")
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    plots.save(fig, path, "Fig 9. Most stable settings give fat tails and memory, but many settings are unstable",
               f"{n} settings drawn at random from the fitting ranges, one {days:,}-day run each. 'Fat tails' = "
               "kurtosis above every no-chartist run; 'memory' = H of |returns| above every no-chartist run "
               "(dashed line). Many random settings end in an endless bubble, so the paper's 'open set' of good "
               "parameters is real but far from all of them.")
    return dict(n=n, runaway=counts[0], thin=counts[1], fat_only=counts[2], fat_memory=counts[3])


# ------------------------------------------------------------------ Fig 10
MIN_SURVIVORS = 5     # Fig 10: a median of fewer surviving histories is not shown


def _dial(pool, p, a1s, days, n):
    """For each a1: (share of histories that did not run away, then the median
    kurtosis, Hill alpha and worst day in sd over the survivors). The medians are
    NaN (not drawn) when fewer than min(MIN_SURVIVORS, n) histories survived."""
    rows = []
    for a1 in a1s:
        out = experiments.histories(dict(p, a1=float(a1)), HELD_OUT[:n], days, pool=pool)
        ok = [h["r"] for h in out if h is not None]
        enough = len(ok) >= min(MIN_SURVIVORS, n)
        rows.append((len(ok) / n,
                     np.median([stats.kurtosis(r) for r in ok]) if enough else np.nan,
                     np.median([stats.hill_alpha(r) for r in ok]) if enough else np.nan,
                     np.median([np.abs(r).max() / r.std() for r in ok]) if enough else np.nan))
    return np.array(rows)


def fig10_herding(pool, spx_p, paper_p, days, path=FIG / "fig10_herding.png", n=12):
    """The project question as a dial: what does stronger mood herding (a1) do?"""
    a1s = np.round(np.linspace(0, 1.4, 8), 2)
    dials = {"S&P-calibrated": (_dial(pool, spx_p, a1s, days, n), spx_p, SPX_MODEL, "o"),
             "paper-matched": (_dial(pool, paper_p, a1s, days, n), paper_p, PAPER_MODEL, "s")}
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.6))
    titles = ["a. Histories without a runaway", "b. Excess kurtosis (median)",
              "c. Tail α, Hill (median)", "d. Worst day, in sd (median)"]
    for name, (rows, p, color, marker) in dials.items():
        for j, a in enumerate(ax):
            a.plot(a1s, rows[:, j], marker + "-", color=color, label=name)
            a.axvline(p["a1"], color=color, ls=":", lw=1.2)
    for j, a in enumerate(ax):
        a.set_xlabel("herding strength a1")
        a.set_title(titles[j])
    ax[0].set_ylim(-0.05, 1.05)
    ax[0].legend(loc="lower right")
    ax[2].axhline(3, color=MUTED, ls="--", lw=1)
    fig.tight_layout(rect=(0, 0.09, 1, 0.93))
    plots.save(fig, path, "Fig 10. Turning the herding dial",
               f"Each fitted setting with only a1 (how strongly chartists copy the majority mood) changed; dotted "
               f"lines mark the fitted a1. {n} held-out histories of {days:,} days at each value; a runaway is an "
               f"endless bubble. Missing points in b-d: fewer than {MIN_SURVIVORS} histories survived, too few for a median. "
               "This is the dial the web app will expose.")
    return a1s, {k: v[0] for k, v in dials.items()}


# ------------------------------------------------------------------ Fig 11
def fig11_policy(pool, p, spx, path=FIG / "fig11_policy.png", breaker_seeds=BREAKER_SEEDS,
                 surprise_seeds=SURPRISE_SEEDS):
    """Extension: circuit breakers and a policy surprise, in S&P-calibrated markets."""
    days = len(spx)
    sd = np.median([r.std() for r in experiments.returns(p, breaker_seeds[:5], days, pool=pool)])
    rules, scale = policy.scaled_breakers(sd, np.std(spx))
    whole, events, bad_b = policy.breaker_experiment(pool, p, breaker_seeds, days, rules)
    surprise, bad_s = policy.surprise_experiment(pool, p, surprise_seeds)
    # (fewer than 2 histories with a pause, e.g. in a short test run: nothing to compare)
    diffs = {k: policy.paired_difference(events["off"][k], events["on"][k])
             if len(events["on"][k]) >= 2 else (np.nan, np.nan, np.nan) for k in policy.EVENT_DAYS}
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))
    a = ax[0]
    off, on = 100 * events["off"][1], 100 * events["on"][1]
    lim = (min(off.min(), on.min()) - 1, max(off.max(), on.max()) + 1) if len(on) else (-10, 0)
    a.plot(lim, lim, color=MUTED, ls="--", lw=1, label="no difference")
    a.scatter(off, on, s=14, color=SPX_MODEL, label="one history's first halt")
    a.axvline(-100 * rules["levels"][0], color=GRID, lw=1)
    a.axhline(-100 * rules["levels"][0], color=GRID, lw=1)
    a.set_xlim(lim)
    a.set_ylim(lim)
    a.set_xlabel("day's change without breakers (%)")
    a.set_ylabel("same day with breakers (%)")
    a.set_title("a. The day a breaker first fires")
    a.legend(loc="upper left")
    b = ax[1]
    for i, k in enumerate(policy.EVENT_DAYS):
        m, lo, hi = (100 * x for x in diffs[k])
        b.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=8, color=SPX_MODEL, capsize=5, lw=1.6)
        if np.isfinite(m):
            b.text(i + 0.08, m, f"{m:+.1f}", va="center", fontsize=9.5, color=INK)
    b.axhline(0, color=MUTED, ls="--", lw=1)
    b.set_xticks(range(len(policy.EVENT_DAYS)),
                 ["the halt day" if k == 1 else f"after {k} days\n(counting the halt day)" for k in policy.EVENT_DAYS])
    b.set_xlim(-0.5, len(policy.EVENT_DAYS) - 0.5)
    b.set_ylabel("with minus without breakers (% points)")
    b.set_title("b. Change since the previous close (mean, 95% interval)")
    c = ax[2]
    z = 100 * surprise["chartists"]
    c.scatter(z, surprise["placebo_peak"] + 1, s=10, color=MUTED, alpha=0.5,
              label="same history, no surprise (placebo)")
    c.scatter(z, surprise["peak"], s=12, color=SPX_MODEL, label="with a 10% surprise")
    edges = np.array([0, 5, 10, 20, 100])
    mids, med, med0 = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi)
        if m.sum() >= 5:
            mids.append(np.median(z[m]))
            med.append(np.median(surprise["peak"][m]))
            med0.append(np.median(surprise["placebo_peak"][m] + 1))
    c.plot(mids, med, "o-", color=INK, lw=2, label="median, with the surprise")
    c.plot(mids, med0, "o--", color=SECOND, lw=1.6, label="median, placebo")
    c.axhline(1, color=MUTED, ls="--", lw=1)
    c.set_xlabel("chartists the day before the surprise (% of traders)")
    c.set_ylabel("largest fall within 10 days, in units of the surprise")
    c.set_title("c. A policy surprise: the value drops 10% at once")
    c.legend(loc="upper right")
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    plots.save(fig, path, "Fig 11. Policy experiments (extension): circuit breakers and a policy surprise",
               f"S&P-calibrated model. Breakers: the US rules (15-minute pauses at -7% and -13% before 3:25 pm, "
               f"closed at -20%), "
               f"scaled x{scale:.2f} to the model's volatility; {len(breaker_seeds)} paired histories of {days:,} days "
               f"with and without breakers, compared at each history's first halt (identical until then). "
               f"Surprise: {len(surprise_seeds)} histories, one 10% drop in the value on a random day; "
               "1 = the price fell exactly as much as the value; the placebo is the same measure without the "
               "surprise (ordinary wobble); the gap between the two medians shows the overshoot roughly "
               "(paired numbers in figures/RESULTS.md).")
    summary = dict(scale=scale, levels=rules["levels"], breaker_runaways=bad_b, events=len(events["on"][1]),
                   halt_days=float(whole["on"]["halt_days"].mean()),
                   closed_days=float(whole["on"]["closed_days"].mean()),
                   worse_day1=int(np.sum(events["on"][1] < events["off"][1])),
                   better_day1=int(np.sum(events["on"][1] > events["off"][1])),
                   diffs=diffs, surprise_runaways=bad_s)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi)
        if m.any():
            summary[f"surprise_{lo}-{hi}%"] = dict(n=int(m.sum()), peak=float(np.median(surprise["peak"][m])),
                                                   placebo=float(np.median(surprise["placebo_peak"][m] + 1)),
                                                   end=float(np.median(surprise["end"][m])))
            if m.sum() >= 2:      # overshoot paired with each history's own placebo
                summary[f"surprise_{lo}-{hi}%"]["overshoot"] = policy.median_overshoot(
                    surprise["peak"][m], surprise["placebo_peak"][m])
    return summary


# ------------------------------------------------------------------ all
def main():
    plots.style()
    paper, spx_p = calibrate.load("paper"), calibrate.load("spx")
    spx = market_data.spx_returns().values
    with Pool() as pool:
        paper_long = experiments.histories(paper, HELD_OUT[:8], LONG, keep=("r", "lp", "lf", "z"), pool=pool)
        spx_hist = experiments.histories(spx_p, HELD_OUT, len(spx), keep=("r", "lp", "lf", "z"), pool=pool)
        spx_long = experiments.histories(spx_p, HELD_OUT[:1], LONG, keep=("r", "z"), pool=pool)[0]
        no_chart = experiments.returns(spx_p, HELD_OUT, len(spx), chartists=False, pool=pool)
        for name, runs in [("paper-matched", paper_long), ("S&P-calibrated", spx_hist)]:
            bad = sum(h is None for h in runs)
            if bad:
                raise model.RunawayError(f"{bad} held-out {name} histories ran away", None)
        paper_r = paper_long[0]["r"]
        paper_news = np.diff(paper_long[0]["lf"])
        spx_runs = [h["r"] for h in spx_hist]
        print("saved", fig1_snapshot(paper_long[0], spx_hist[0], spx))
        print("saved", fig2_aggregation(paper_r, paper_news, spx_runs, spx,
                                        paper_runs=[h["r"] for h in paper_long]))
        print("saved", fig3_memory(paper_r, paper_news, spx_runs[0], spx))
        print("saved", fig4_checks(no_chart, spx_runs, [h["r"] for h in paper_long], spx))
        print("saved", fig5_mechanism(spx_long))
        print("saved", fig6_kirman())
        pct = fig7_typical(spx_runs, spx)
        print("fig7 percentiles", {k: round(v) for k, v in pct.items()})
        rows, keys, names = fig8_out_of_sample(pool, spx_p)
        print("fig8 unusual cells", int(np.sum((rows < 5) | (rows > 95))), "of", rows.size)
        print("fig9", fig9_robustness(pool))
        a1s, dials = fig10_herding(pool, spx_p, paper, len(spx))
        for name, rows in dials.items():
            print("fig10", name, dict(zip(a1s.tolist(), np.round(rows, 2).tolist())))
        print("fig11", fig11_policy(pool, spx_p, spx))


if __name__ == "__main__":
    main()
