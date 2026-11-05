# Can Herding Explain Market Crashes?

A CMU 02-120 (Programming for Scientists, Fall 2026) final project.

Real stock markets crash far more often than a bell curve allows: under a bell
curve, Black Monday (−20% in one day, 1987) should essentially never happen.
This project asks whether **herding** (traders copying each other) can produce
those crashes by itself, even when the news hitting the market is perfectly
ordinary bell-curve noise.

To answer it, the program rebuilds the **Lux–Marchesi (1999)** agent-based market
model. Traders switch between *fundamentalists* (who buy when the price is below
its true value) and *chartists* (who follow the crowd, as optimists or pessimists,
like Kirman's ants). The model is calibrated to the S&P 500 and compared with it
on fat tails, volatility clustering and extreme moves.

**New to the project? Start with [`KEY_POINTS.md`](KEY_POINTS.md).** It explains
the ideas, the figures and the findings without assuming any finance background.
Detailed results for every figure are in [`figures/RESULTS.md`](figures/RESULTS.md).

## How to run it

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
python3 main.py                      # S&P 500 vs the model -> figures/draft.png
python3 make_figures.py              # all figures fig1-fig9 (about 4 minutes)
python3 -m pytest tests/ -q -m "not slow"   # quick tests
python3 -m pytest tests/ -q          # all tests (about 7 minutes)
python3 -m herding.calibrate         # refit the model to the S&P 500 (about an hour)
```

## What is where

| Path | What it holds |
|---|---|
| `herding/` | The program: the model, statistics, data loading, Kirman's ant model, calibration, plots |
| `main.py`, `make_figures.py` | Scripts that produce the figures |
| `figures/` | Figures and `RESULTS.md`, which explains them |
| `tests/` | One test file per module |
| `analysis/` | Exploration scripts behind individual findings |
| `data/` | S&P 500, Nasdaq-100, 20 large stocks, VIX (Yahoo Finance) and US recessions (FRED) |
| `paper/`, `notes/` | Sources and my study notes |
| `proposal.pdf` | My project proposal |
| `journal/` | Build journal (Claude Code session logs, copied automatically) |

The web app (adjust herding strength and watch crashes) and the video demo are
coming; links will be added here.

## Main source

Lux, T. & Marchesi, M. (1999). Scaling and criticality in a stochastic
multi-agent model of a financial market. *Nature* 397, 498–500.
doi:10.1038/17290. (Not included in this repository because it is copyrighted.)
