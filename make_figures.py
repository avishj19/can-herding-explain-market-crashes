"""Make every results figure (figures/fig*.png) for the project write-up.

Run:  python3 make_figures.py      (~1-2 min with numba)
All model checks use held-out seeds the calibration never saw.
Colors (validated colorblind-safe): S&P/real data = ink, calibrated = blue,
paper settings = orange, no chartists = aqua, references = muted gray dashes.
"""
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from herding import calibrate, experiments, kirman, market_data, model, stats
from herding.experiments import BURN, DAYS, worst_fall
from herding.plots import AQUA, BLUE, BLUE_LIGHT, GRID, INK, MUTED, ORANGE, RED, SECOND

FIG = market_data.ROOT / "figures"
HELD_OUT = range(600, 620)                       # never used by the calibration
LABELS = dict(alpha_5pct="tail exponent α (5% tail)", alpha_1pct="tail exponent α (1% tail)",
              share_4sd="share of days beyond 4 std", ac_r_1="return correlation, 1 day",
              ac_r_2="return correlation, 2 days", ac_r_2to10="return correlation, days 2-10",
              ac_abs_1="volatility memory, 1 day", ac_abs_20="volatility memory, 20 days",
              ac_abs_50="volatility memory, 50 days", ac_abs_100="volatility memory, 100 days")

STYLE = {   # applied only while a figure is drawn (see styled), never globally
    "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.edgecolor": "#c3c2b7", "axes.labelcolor": SECOND,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False,
    "lines.linewidth": 2, "savefig.dpi": 150, "savefig.bbox": "tight"}


def styled(fn):
    """Draw a figure with STYLE without changing matplotlib's global settings."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context(STYLE):
            return fn(*args, **kwargs)
    return wrapper


def save(fig, name, caption):
    fig.text(0.01, -0.02, caption, ha="left", va="top", fontsize=8.5, color=SECOND, wrap=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print("saved", name)


def unusual(pct):
    """A real value outside the model's 5-95% range."""
    return not 5 <= pct <= 95


def ensemble(pool, p, seeds=HELD_OUT, days=DAYS, herding=True):
    return experiments.ensemble(p, seeds, days, herding, pool=pool)


# ----------------------------------------------------------------- figure 1
@styled
def fig_fingerprint(spx, cal_runs, paper_runs):
    target = stats.stylized_facts(spx)
    cal = [stats.stylized_facts(r) for r in cal_runs]
    paper = [stats.stylized_facts(r) for r in paper_runs]
    pct = calibrate.percentiles(target, cal, calibrate.TARGETS)
    fig, axes = plt.subplots(2, 5, figsize=(15, 5.6))
    for ax, k in zip(axes.flat, calibrate.TARGETS):
        v = np.array([f[k] for f in cal])
        lo, med, hi = np.percentile(v, [5, 50, 95])
        ax.plot([0, 0], [lo, hi], color=BLUE, lw=6, solid_capstyle="round", alpha=0.35)
        ax.plot(0, med, "o", color=BLUE, ms=8, label="calibrated model (median, 5-95%)")
        ax.plot(0.25, np.median([f[k] for f in paper]), "X", color=ORANGE, ms=9,
                label="paper settings (median)")
        ax.plot(-0.25, target[k], "D", color=INK, ms=8, label="S&P 500 1950-2026")
        ax.set_xlim(-0.6, 0.6); ax.set_xticks([])
        flag = "UNUSUAL" if unusual(pct[k]) else "typical"
        ax.set_title(f"{LABELS[k]}\nS&P percentile {pct[k]:.0f}% ({flag})", fontsize=9.5,
                     color=INK if flag == "typical" else RED)
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo - 0.06 * (yhi - ylo), yhi + 0.06 * (yhi - ylo))
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Fig 1. The S&P 500's fingerprint vs 20 held-out histories of the calibrated model",
                 y=1.12, fontsize=13, fontweight="bold")
    fig.tight_layout()
    n_typ = sum(not unusual(v) for v in pct.values())
    save(fig, "fig1_fingerprint.png",
         f"Each panel is one measure. Blue bar: the range covering 90% of 20 simulated 77-year histories "
         f"(seeds 600-619, never used in calibration); blue dot: their median. Black diamond: the real "
         f"S&P 500. Orange X: the original paper settings. The S&P falls inside the model's typical "
         f"range on {n_typ} of 10 measures; the paper settings miss volatility memory badly.")
    return pct


# ----------------------------------------------------------------- figure 2
@styled
def fig_tails(spx, cal_runs, base):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pooled = np.concatenate([(r - r.mean()) / r.std() for r in cal_runs])
    for r, lab, c, ls in [(spx, "S&P 500 (real)", INK, "-"), (pooled, "calibrated model (20 histories)", BLUE, "-"),
                          (base, "no chartists (bell-curve news only)", AQUA, "-")]:
        z, cc = stats.ccdf(r)
        ax.loglog(z, cc, ls, color=c, label=lab)
    g = np.linspace(1, 9, 100)
    ax.loglog(g, 2 * norm.sf(g), "--", color=MUTED, lw=1.5, label="Gaussian (bell curve)")
    ax.loglog(g, 0.6 * g ** -3.0, ":", color=MUTED, lw=1.8, label="slope −3: inverse cubic law")
    ax.set_xlim(0.8, 40); ax.set_ylim(1e-5, 1)
    ax.set_xlabel("size of daily move, in standard deviations (log scale)")
    ax.set_ylabel("share of days with a bigger move (log scale)")
    ax.set_title("Fig 2. Fat tails: big days are far more common than a bell curve predicts")
    ax.legend(loc="lower left", fontsize=8.5)
    z, cc = stats.ccdf(spx)                          # Black Monday is the S&P's largest move
    ax.plot(z[-1], cc[-1], "o", color=INK, ms=6)
    ax.annotate(f"Black Monday 1987\n({z[-1]:.0f} std)", (z[-1], cc[-1]), xytext=(0, 32),
                textcoords="offset points", ha="center", fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=SECOND))
    save(fig, "fig2_tails.png",
         "On log-log axes a power law is a straight line. The real S&P and the calibrated model both "
         "follow the slope −3 line (the inverse cubic law, Gopikrishnan et al. 1998) from about 2 to 8 "
         "standard deviations; beyond that the model's tail bends down (its biggest days are about 12 std, "
         "Black Monday was 23). With no chartists, the model's tail bends down exactly like the Gaussian: "
         "herding is what creates the fat tail.")


# ----------------------------------------------------------------- figure 3
@styled
def fig_memory(spx, cal_runs, paper_runs):
    lags = np.arange(1, 101)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[2, 1]))
    for ax, lg, f, title in [(axes[0], lags, np.abs, "Volatility memory: does a big move predict big moves later?"),
                             (axes[1], lags[:10], lambda x: x, "Return correlation: does today's direction predict tomorrow's?")]:
        cal = np.array([stats.acf(f(r), lg) for r in cal_runs])
        lo, med, hi = np.percentile(cal, [5, 50, 95], axis=0)
        ax.fill_between(lg, lo, hi, color=BLUE_LIGHT, alpha=0.6, lw=0, label="calibrated model, 5-95%")
        ax.plot(lg, med, color=BLUE, label="calibrated model, median")
        ax.plot(lg, np.median([stats.acf(f(r), lg) for r in paper_runs], axis=0), color=ORANGE,
                label="paper settings, median")
        ax.plot(lg, stats.acf(f(spx), lg), color=INK, label="S&P 500 (real)")
        ax.axhline(0, color="#c3c2b7", lw=1)
        ax.set_xlabel("days apart (lag)"); ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("correlation")
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Fig 3. Volatility clustering: the calibrated model remembers wild periods as long as the market does",
                 y=1.03, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_memory.png",
         "Left: correlation between the SIZE of moves k days apart. The S&P still remembers a wild "
         "period after 100 days; the paper settings forget within ~30 days; the calibrated model tracks "
         "the S&P. Right: correlation of DIRECTION. The real market is unpredictable (≈0). The model "
         "has a small 2-day reversal (the S&P is outside its typical range here: its known weak spot).")


# ----------------------------------------------------------------- figure 4
@styled
def fig_mechanism(p, seed):
    """One typical history (pass the median-mismatch seed, see main)."""
    o = model.simulate_full(DAYS + BURN, p=p, seed=seed)
    r, frac, mis = o["r"][BURN:], o["frac"][BURN:], (o["lp"] - o["lf"])[BURN + 1:]
    i = int(np.argmin(r))
    w = slice(max(0, i - 400), min(len(r), i + 300))
    d = np.arange(len(r))[w] - i
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(d, r[w] / r.std(), color=INK, lw=0.8)
    axes[0].set_ylabel("daily return\n(std units)")
    axes[0].set_title("① Returns: calm, then a burst of huge moves (the crash)", loc="left")
    axes[1].fill_between(d, 0, frac[w] * 100, color=BLUE, alpha=0.8, lw=0)
    axes[1].set_ylabel("chartists\n(% of traders)")
    axes[1].set_title("② Chartist share: traders crowd into trend-following, recruiting each other", loc="left")
    axes[2].plot(d, mis[w] * 100, color=ORANGE)
    axes[2].axhline(0, color="#c3c2b7", lw=1)
    axes[2].set_ylabel("price vs value\n(% mispricing)")
    axes[2].set_title("③ Mispricing: with few fundamentalists left, nothing anchors the price", loc="left")
    axes[2].set_xlabel("days relative to the worst day of this history")
    for ax in axes:
        ax.axvline(0, color=RED, lw=1, ls="--")
    fig.suptitle(f"Fig 4. How the calibrated model makes a crash (a typical held-out history, seed {seed})",
                 y=1.0, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_mechanism.png",
         "The same days in three panels. The crash (red line) comes when the share of chartists surges: "
         "each chartist makes it easier to recruit more (Kirman-style crowding), so fundamentalists "
         "dwindle and the price swings far from its value. As mispricing grows, betting on the price "
         "returning looks profitable, traders switch back to fundamentalist, and calm returns.")


# ----------------------------------------------------------------- figure 5
@styled
def fig_kirman():
    cases = [("herding: ε < (1−δ)/(N−1)", dict(N=100, eps=0.002, delta=0.01)),
             ("no herding: ε > (1−δ)/(N−1)", dict(N=100, eps=0.05, delta=0.01))]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for col, (name, kw) in enumerate(cases):
        k = kirman.simulate(**kw, steps=40_000_000, seed=1)
        show = k[:5_000_000:2_000]                   # first 5M steps, for readability
        axes[0, col].plot(np.arange(len(show)) * 2_000 / 1e6, show, color=BLUE, lw=0.8)
        axes[0, col].set_ylim(0, 100); axes[0, col].set_ylabel("ants at source A")
        axes[0, col].set_xlabel("steps (millions; first 5M of 40M shown)"); axes[0, col].set_title(name)
        hist = np.bincount(k, minlength=101) / len(k)
        axes[1, col].bar(np.arange(101), hist, width=0.9, color=BLUE_LIGHT, label="simulation (40M steps)")
        axes[1, col].plot(np.arange(101), kirman.stationary(**kw), color=INK, lw=1.8,
                          label="Kirman's exact answer")
        tv = 0.5 * np.abs(hist - kirman.stationary(**kw)).sum()
        axes[1, col].set_xlabel("ants at source A"); axes[1, col].set_ylabel("share of time")
        axes[1, col].text(0.5, 0.55 if col == 0 else 0.08, f"mismatch (total variation) = {tv:.3f}",
                          transform=axes[1, col].transAxes, ha="center", fontsize=8.5, color=INK)
        axes[1, col].legend(fontsize=8.5, loc="upper center" if col == 0 else "upper left")
    fig.suptitle("Fig 5. Kirman's ant model reproduces his exact math: herding makes the crowd flip all at once",
                 y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_kirman.png",
         "100 ants choose between two identical food sources (N=100, δ=0.01). Left: when spontaneous "
         "switching ε is rarer than recruitment, ants herd at one source and then flip en masse. Right: "
         "otherwise they split about 50/50. Bottom: time spent at each state (bars) matches the exact "
         "long-run answer from detailed balance (line). This is the herding mechanism inside Lux-Marchesi.")


# ----------------------------------------------------------------- figure 6
@styled
def fig_out_of_sample(pool, p, cal_runs):
    """Percentile of each real series among 20 model histories of the same length.
    The 1950- S&P row reuses Fig 1's held-out histories so the two figures agree;
    the 20 stocks are summarized as one row (median percentile, count unusual)."""
    def facts(n, seeds):
        return [stats.stylized_facts(x) for x in ensemble(pool, p, seeds, n)]

    def row(r, sims):
        pct = calibrate.percentiles(stats.stylized_facts(r), sims, calibrate.TARGETS)
        return [pct[k] for k in calibrate.TARGETS]

    rows = [row(market_data.spx_returns().values, [stats.stylized_facts(r) for r in cal_runs])]
    for i, name in enumerate(["spx_1928", "ndx"]):
        r = market_data.returns(name).values
        rows.append(row(r, facts(len(r), range(700 + 20 * i, 720 + 20 * i))))
    stocks = [market_data.stock_returns(t).values for t in market_data.stock_tickers()]
    sims = facts(int(np.median([len(r) for r in stocks])), range(760, 780))
    stock_pct = np.array([row(r, sims) for r in stocks])
    rows.append(np.median(stock_pct, axis=0))
    n_bad = np.sum((stock_pct < 5) | (stock_pct > 95), axis=0)
    m = np.array(rows)
    names = ["S&P 500 1950-2026\n(fitted data, for reference)", "S&P 500 1928-2026\n(incl. Great Depression)",
             "Nasdaq-100 1985-2026", f"{len(stocks)} stocks, market beta ≈ 1\n(median; # unusual of {len(stocks)})"]
    fig, ax = plt.subplots(figsize=(13, 4.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", ["#256abf", "#9ec5f4", "#f0efec", "#f2a3a2", "#c53a39"])
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for (i, j), v in np.ndenumerate(m):
        text = f"{v:.0f}" + (f"\n({n_bad[j]} unusual)" if i == len(m) - 1 else "")
        ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=INK,
                fontweight="bold" if i < len(m) - 1 and unusual(v) else "normal")
    ax.set_xticks(range(len(calibrate.TARGETS)), [LABELS[k] for k in calibrate.TARGETS],
                  rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(names)), names, fontsize=8.5); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label("percentile among 20 model histories\n(50 = typical; <5 or >95 = unusual)", fontsize=8.5)
    ax.set_title("Fig 6. Out-of-sample: real data the calibration never saw, scored against the model\n"
                 "(top row: the S&P 500 data it was fitted to, same 20 histories as Fig 1)", fontsize=10.5)
    fig.tight_layout()                               # keeps the tilted labels above the caption
    index_bad = int(sum(unusual(v) for v in m[:3].flat))
    save(fig, "fig6_out_of_sample.png",
         f"Each cell is where the real value falls among 20 model histories of the same length (bold = "
         f"unusual). With 10 measures and a 5-95% band, about 1 unusual cell per row is expected by chance. "
         f"The three indexes are unusual on {index_bad} of 30 cells (top row = fitted data). Stocks: {int(n_bad.sum())} of "
         f"{n_bad.size * len(stocks)} stock-measure pairs are unusual; single companies also get "
         f"company-specific news (earnings surprises) that a whole-market model does not have.")
    return m, n_bad


# ----------------------------------------------------------------- figure 7
def _breaker_pair(args):
    p, seed, levels = args
    out = []
    for b in (None, dict(levels=levels, halt_minutes=15)):
        r = model.simulate_full(DAYS + BURN, p=p, seed=seed, breakers=b)["r"][BURN:]
        out.append([worst_fall(r, k) for k in (1, 5, 20)])
    return out


def _shock_pair(args):
    p, seed = args
    a = model.simulate_full(1300, True, p, seed=seed)
    b = model.simulate_full(1300, True, p, seed=seed, shocks={1000: -0.10})
    return a["frac"][999], (b["lp"] - a["lp"])[1000:1061]


@styled
def fig_policy(pool, p, cal_runs, spx):
    """Two policy experiments, each with paired runs (identical randomness with and
    without the policy): circuit breakers, and a one-time 10% policy surprise."""
    rules, scale = experiments.scaled_breakers(np.median([r.std() for r in cal_runs]), spx.std())
    res = np.array(pool.map(_breaker_pair, [(p, s, rules["levels"]) for s in range(3000, 3020)]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, j, name in [(axes[0], 0, "worst single day"), (axes[1], 2, "worst 20-day fall")]:
        off, on = np.expm1(-res[:, 0, j]) * -100, np.expm1(-res[:, 1, j]) * -100
        top = max(off.max(), on.max()) * 1.08
        ax.plot([0, top], [0, top], color=MUTED, ls="--", lw=1.2)
        ax.scatter(off, on, s=60, color=BLUE, edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        lower, higher = int(np.sum(on < off - 1e-9)), int(np.sum(on > off + 1e-9))
        ax.set_title(f"Circuit breakers: {name}\nlower in {lower}/20 histories, higher in {higher}/20",
                     fontsize=10)
        ax.set_xlabel("fall without breakers (%)"); ax.set_ylabel("fall with breakers (%)")
        ax.set_xlim(0, top); ax.set_ylim(0, top)
        ax.text(0.05 * top, 0.9 * top, "below the line =\nbreakers helped", fontsize=8, color=SECOND)

    shocks = pool.map(_shock_pair, [(p, s) for s in range(5000, 5200)])
    frac = np.array([x[0] for x in shocks])
    resp = np.array([x[1] for x in shocks]) / np.log(0.9)     # 1.0 = moved exactly by the shock
    bins, names = [(0, 0.1), (0.1, 0.3), (0.3, 1.01)], ["<10%", "10-30%", ">30%"]
    over = [resp[(frac >= lo) & (frac < hi)].max(axis=1) for lo, hi in bins]
    ax = axes[2]
    ax.bar(names, [o.mean() for o in over], color=["#86b6ef", "#3987e5", "#184f95"], width=0.55)
    top = min(max(o.max() for o in over) + 0.12, 1.6)   # rare extreme overshoots go off-scale
    for i, o in enumerate(over):
        ax.plot([i] * len(o), np.minimum(o, top - 0.02), "o", color=INK, ms=3, alpha=0.35)
        if o.max() > top - 0.02:
            ax.annotate(f"{o.max():.2f}×", (i, top - 0.02), xytext=(14, -2), textcoords="offset points",
                        fontsize=8, color=SECOND, va="center")
        label = f"n={len(o)}\nmean {o.mean():.2f}×\nmax {o.max():.2f}×"
        if o.max() > top - 0.02:                 # off-scale bin: put its label beside the dots
            ax.text(i - 0.32, 1.0 + 0.6 * (top - 1.0), label, ha="right", fontsize=8.5)
        else:
            ax.text(i, o.max() + 0.01, label, ha="center", fontsize=8.5)
    ax.axhline(1, color=MUTED, ls="--", lw=1.2)
    ax.set_ylim(0.95, top)
    ax.set_xlabel("chartist share when the surprise hits")
    ax.set_ylabel("largest price fall ÷ size of the surprise")
    ax.set_title("A 10% policy surprise: how far the price overshoots\n(1.0 = falls exactly 10%)",
                 fontsize=10)
    fig.suptitle("Fig 7. Policy: breakers soften single days, and surprises hurt most in a crowded market",
                 y=1.05, fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig7_policy.png",
         f"Left and middle: each dot is one 77-year history run twice with identical randomness, without "
         f"and with US-style circuit breakers (15-min pauses at −7% and −13%, close at −20%, scaled "
         f"×{scale:.2f} to the model's volatility); dots on the line are histories where no breaker fired. "
         f"Breakers soften the worst single day but barely change 20-day falls: they spread a crash out "
         f"rather than prevent it. Right: 200 histories run with and without a one-time surprise that cuts "
         f"the value 10% (like an unexpected rate hike). It is amplified only when many chartists are active.")
    return res, frac, over


def main():
    FIG.mkdir(exist_ok=True)
    p = calibrate.load()
    spx = market_data.spx_returns().values
    with Pool() as pool:
        cal_runs = ensemble(pool, p)
        paper_runs = ensemble(pool, model.P)
        base = experiments.history(p, seed=600, herding=False)
        fig_fingerprint(spx, cal_runs, paper_runs)
        fig_tails(spx, cal_runs, base)
        fig_memory(spx, cal_runs, paper_runs)
        target = stats.stylized_facts(spx)
        typical = HELD_OUT[calibrate.representative([stats.stylized_facts(r) for r in cal_runs], target)]
        fig_mechanism(p, typical)
        fig_kirman()
        fig_out_of_sample(pool, p, cal_runs)
        fig_policy(pool, p, cal_runs, spx)


if __name__ == "__main__":
    main()
