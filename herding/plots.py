"""Drawing helpers shared by main.py, make_figures.py and (later) the web app.

One palette everywhere, checked for colour-blind safety (the dataviz validator:
blue, orange and aqua pass as a set of three in light mode):
    real market data         INK (near-black)
    S&P-calibrated model     BLUE
    paper-matched model      ORANGE
    no chartists / news      AQUA
Aqua is low-contrast on white, so it is always labelled (legend or text), never
identified by colour alone.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from . import stats

INK, SECOND, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_LIGHT, ORANGE_LIGHT = "#b7d3f6", "#f7c4ae"
REAL, SPX_MODEL, PAPER_MODEL, NEWS = INK, BLUE, ORANGE, AQUA

STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": MUTED, "axes.labelcolor": SECOND, "axes.titlecolor": INK,
    "axes.titlesize": 10.5, "axes.labelsize": 9.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": SECOND, "ytick.color": SECOND, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False, "lines.linewidth": 1.6,
    "font.family": "DejaVu Sans",
}


def style():
    plt.rcParams.update(STYLE)


def save(fig, path, title, caption):
    """Title on top, a plain-language caption underneath, save, close."""
    path = Path(path)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, caption, ha="left", va="bottom", fontsize=8.6, color=SECOND, wrap=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=130, bbox_inches="tight")
    finally:
        plt.close(fig)
    return path


def ccdf(ax, r, label, color, **kw):
    """Log-log tail plot: share of days whose |standardized return| exceeds x."""
    z, cc = stats.ccdf(r)
    keep = z > 0
    ax.loglog(z[keep], cc[keep], color=color, label=label, **kw)


def gaussian_ccdf(ax, label="bell curve N(0,1)"):
    g = np.geomspace(0.05, 6, 200)
    ax.loglog(g, 2 * norm.sf(g), "--", color=MUTED, lw=1.2, label=label)


def dfa(ax, x, scales, label, color, marker="o", shift=1.0):
    """DFA fluctuation F(n) vs window n (log-log), with the fitted slope H in the label."""
    s, F = stats.dfa(x, scales)
    H = np.polyfit(np.log(s), np.log(F), 1)[0]
    ax.loglog(s, F * shift, marker, color=color, ms=4.5, lw=0,
              label=f"{label}: H = {H:.2f}")
    fit = np.exp(np.polyval(np.polyfit(np.log(s), np.log(F), 1), np.log(s)))
    ax.loglog(s, fit * shift, "-", color=color, lw=1.0)
    return H


def standardized(r):
    r = np.asarray(r, dtype=float)
    return (r - r.mean()) / r.std()


def comparison(real, model_r, z, no_chartists, path):
    """main.py's quick 2x2 overview: S&P vs model returns, tails, chartist share."""
    style()
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    n = min(len(real), len(model_r))
    for a, r, c, name in [(ax[0, 0], real[-n:], REAL, "S&P 500"),
                          (ax[0, 1], model_r[-n:], SPX_MODEL, "model (S&P-calibrated)")]:
        a.plot(standardized(r), lw=0.35, color=c)
        a.set_ylim(-25, 25)
        a.set_title(f"{name}: daily returns ({n:,} days)")
        a.set_ylabel("return / standard deviation")
    a = ax[1, 0]
    ccdf(a, real, "S&P 500", REAL)
    ccdf(a, model_r, "model", SPX_MODEL)
    ccdf(a, no_chartists, "model, no chartists", NEWS)
    gaussian_ccdf(a)
    a.set_xlim(0.3, 30)
    a.set_ylim(1e-5, 1.2)
    a.set_xlabel("|return| / standard deviation")
    a.set_ylabel("share of days with a bigger move")
    a.set_title("Tails: fat for the S&P and the model, thin without chartists")
    a.legend(loc="lower left")
    a = ax[1, 1]
    a.fill_between(np.arange(len(z)), 0, z, color=SPX_MODEL, lw=0)
    a.set_ylim(0, 1)
    a.set_title("Share of traders who are chartists (model)")
    a.set_xlabel("day")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return save(fig, path, "S&P 500 vs the herding model",
                "Model settings fitted to the S&P 500 (settings/spx.json); one held-out history shown.")
