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

**New to the project? Start with [`docs/KEY_POINTS.md`](docs/KEY_POINTS.md).** It explains
the ideas, the figures and the findings without assuming any finance background.
To see every figure with a short caption, open [`figures/GALLERY.md`](figures/GALLERY.md).
Detailed results for every figure are in [`figures/RESULTS.md`](figures/RESULTS.md).

## How to run it

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
python3 scripts/main.py              # S&P 500 vs the model -> figures/fig00_quick_check_sp500_vs_model.png
python3 scripts/make_figures.py      # all figures fig01-fig12 (about 2 minutes on 8 cores)
python3 -m pytest tests/ -q -m "not slow"   # quick tests
python3 -m pytest tests/ -q          # all tests
python3 -m herding.calibrate paper   # refit to the paper's numbers (add --refine to polish)
python3 -m herding.calibrate spx     # refit to the S&P 500 (1.5-2.5 hours; rerun to resume if stopped)
python3 data/download_market_data.py             # re-download every data file (ends 2026-10-30)
```

The model's equations, each with its source, are in [`docs/SPEC.md`](docs/SPEC.md).

## Where everything is

Each folder has one job:

```
02-120 Project/
├── README.md        you are here
├── app.py           the website (streamlit run app.py); its pages are in web/
├── web/             the website's pages, shared style, and saved model results (numbers.json)
├── herding/         the simulation code (the model, statistics, fitting, plots)
├── scripts/         things you RUN
│   ├── main.py          quick check: S&P 500 vs the model
│   ├── make_figures.py  draws every figure in figures/
│   └── analysis/        smaller scripts behind single findings
├── figures/         things you LOOK at: GALLERY.md (every figure + caption), RESULTS.md (details)
├── docs/            things you READ
│   ├── KEY_POINTS.md    the whole project explained for a newcomer
│   ├── SPEC.md          the model's equations, each with its source
│   ├── proposal.pdf     my project proposal
│   └── papers/          source papers
├── data/            real market data (S&P 500, Nasdaq-100, 20 stocks, VIX, recessions)
├── settings/        the two fitted settings (paper-matched, S&P-calibrated)
├── tests/           one test file per module in herding/
├── journal/         build journal (Claude Code session logs, copied automatically)
└── scratch/         temporary files, not in the repository
```

## The website

The project's website lets anyone pick a stock, stock index or commodity, turn the herding
strength up and down, and watch how the simulated market's crashes change.

```bash
pip install -r requirements.txt
streamlit run app.py                 # then open http://localhost:8501
```

| Page | What it does |
|---|---|
| Start here | What the project is, how to use the site, how to read each chart |
| Story | The argument in five steps, one chart each |
| Test program | Any market (saved, or any ticker live from Yahoo) next to the model; ranges 1D to All; run the model and see each run on the herding chart |
| Compare | All 26 saved markets on one map, next to the model |
| Checks | The proposal's checks, each with its rule and result |
| Findings | Every figure from `figures/`, with captions |

The site's code is `app.py` and `web/`; every number it shows comes from
`herding/explore.py` (tested in `tests/test_explore.py`). The model results it shows are the
same 40 held-out histories as the figures, saved in `web/numbers.json` by
`python3 scripts/web_numbers.py`. `python3 scripts/analysis/web_stress_test.py` checks the
site on all the data, including that the numbers on each page match the code and the docs.

**Live site: https://herding-crashes.streamlit.app** (Streamlit Community Cloud; the first visit after a quiet
spell takes a minute while the server wakes up). Video demo: coming.

## Main source

Lux, T. & Marchesi, M. (1999). Scaling and criticality in a stochastic
multi-agent model of a financial market. *Nature* 397, 498–500.
doi:10.1038/17290. (Not included in this repository because it is copyrighted.)
