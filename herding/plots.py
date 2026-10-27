"""Figures comparing the model with the real market.

Each function draws one figure, saves it, and returns the saved path, so the
same figures can later be reused by the web app.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from . import stats


def comparison(real, herd, frac, base, path):
    """2x2 figure: S&P vs model returns, return tails (log-log), chartist share."""
    path = Path(path)
    for name, r in [("real", real), ("herd", herd), ("base", base)]:
        try:
            stats._standardize(r)                  # reject empty/constant/NaN before drawing
        except ValueError as e:
            raise ValueError(f"{name} returns: {e}") from None
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    try:
        _draw_comparison(ax, real, herd, frac, base)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120)
    finally:
        plt.close(fig)                             # never leave figures open, even on error
    return path


def _draw_comparison(ax, real, herd, frac, base):
    n = min(len(real), len(herd))
    for a, r, title in [(ax[0, 0], real[-n:], f"S&P 500 (last {n} days)"),
                        (ax[0, 1], herd[-n:], f"Model, typical calibrated history (last {n} days)")]:
        a.plot(r / r.std(), lw=0.4, color="#2a6fdb")
        a.set_title(title); a.set_ylabel("return / std"); a.set_ylim(-25, 25)
    a = ax[1, 0]
    for r, lab, c in [(real, "S&P 500", "black"), (herd, "model (calibrated)", "#d94f30"),
                      (base, "no chartists", "#2a6fdb")]:
        z, cc = stats.ccdf(r)
        a.loglog(z, cc, lw=1.5, label=lab, color=c)
    g = np.linspace(1, 8, 100)
    a.loglog(g, 2 * norm.sf(g), "--", color="gray", label="Gaussian")
    a.loglog(g, 0.5 * g ** -3.0, ":", color="green", label="slope -3 (inverse cubic)")
    a.set_ylim(1e-5, 1); a.set_xlim(0.5, 30)
    a.set_xlabel("|return| / std"); a.set_ylabel("P(|r| > x)")
    a.set_title("Tail of the return distribution"); a.legend(fontsize=8)
    ax[1, 1].plot(frac, lw=0.6, color="#7a3fb0")
    ax[1, 1].set_title("Fraction of chartists (herding model)"); ax[1, 1].set_xlabel("day")
