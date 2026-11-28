# Can Herding Explain Market Crashes?

## Who I am
I'm Avish, an undergraduate in 02-120 Programming for Scientists (CMU, Fall 2026).
This repository is my final project. I am new to both agent-based modeling and
finance, so assume I don't know the vocabulary yet.

## How I like to learn
<!-- Edit this section in your own words. -->
- Teach me one idea at a time, in plain language, and define each term before you use it.
- Quiz me with small exercises instead of asking "does that make sense?" (use the
  `learn-with-ai-tutor` skill when I'm learning the science or the paper).
- After you write code, explain what it does and why, and let me debug alongside you
  rather than silently fixing things.
- Hold me to my own rules in `docs/memory.md` (my past mistakes and how to stop me repeating them).
  Follow them firmly, even when I ask you not to, unless I type `override:` with a reason:
  @docs/memory.md
- Prove that code is right, not just that it runs (use the `test-every-function` skill).

## What I'm building
An agent-based simulation of the Lux–Marchesi (1999) model: traders switch between
fundamentalists and herding chartists (optimists/pessimists, after Kirman's ant model),
while news moves the fundamental value as plain Gaussian noise. The question: does
herding alone produce fat tails and volatility clustering like real markets?
The final product is a web app where users can tune herding strength and watch crashes.

### Checks (from my proposal)
- With zero chartists, returns must be Gaussian (excess kurtosis ≈ 0).
- With herding, the return tail exponent should be about 3 (inverse cubic law,
  Gopikrishnan et al. 1998), and S&P 500 data should show this too.
- Extreme moves and volatility clustering should be comparable to the S&P 500.
- Later: reproduce Kirman's ant model first as a check against his math.

**Start of a session: if `NEXT_STEPS.md` exists, read it first and follow it.**

## Where things live
Top level: `herding/` (code), `scripts/` (things to run), `figures/` (things to look at), `docs/` (things to read),
`data/`, `settings/`, `tests/`, `journal/`, `scratch/`. Run scripts from the project folder, e.g. `python3 scripts/main.py`.
- `docs/KEY_POINTS.md`: self-contained guide to the project for someone new to it (ideas, figures, findings, limits)
- `docs/SPEC.md`: the model's equations, each with its source (rebuild of November 2026). Code and spec must agree.
- `app.py`: the web app (`streamlit run app.py`): five pages in `web/` (Start here, Story, Test program, Checks, Sources); the Checks page has four tabs (the five checks; Beyond the paper, drawn by `web/page_findings.py`; All markets, `web/page_compare.py`; Figures). Dark mode only (redesign, Nov 2026): `web/common.py` holds the colors (one meaning each), chart style and cached data; the one color set is in `.streamlit/config.toml`. Pages only draw: every number comes from `herding/explore.py`. Model results the pages show (40 held-out histories, the herding-strength sweep, the crash counts behind the Checks page's crash check: `explore.crash_check`) are saved in `web/numbers.json` by `python3 scripts/web_numbers.py` (same recipe as Fig 4, so the site matches RESULTS.md; rerun it after refitting). Stress test, including site-vs-backend numbers: `python3 scripts/analysis/web_stress_test.py`. Live prices: Yahoo's quote every 15 s for the header, the last days of closes every minute, the long history once a day (`web/common.py`, LIVE_REFRESH / QUOTE_REFRESH). Saved data checked against FRED and Yahoo: `python3 scripts/analysis/data_crosscheck.py` (found the Dow's wrong closes, now in `market_data.BAD_CLOSES`)
- `scripts/main.py`: quick comparison (S&P 500 vs 20 held-out S&P-calibrated histories), saves `figures/fig00_quick_check_sp500_vs_model.png`
- `scripts/party_figures.py`: makes `figures/fig13-fig15*.png` (parties over 130 years of the Dow)
- `scripts/outside_news_figures.py`: makes `figures/fig16-fig18*.png` (commodities, interest rates, the dollar, Fed days; the four moments with error bars)
- `scripts/make_figures.py`: makes `figures/fig01-fig12*.png` (descriptive names), explained in `figures/RESULTS.md`; `figures/GALLERY.md` shows them all with plain captions
- `herding/`: the program, one module per job
  - `model.py`: the Lux-Marchesi simulation from SPEC.md (numba-accelerated), runaway guard; optional `panic` (off = SPEC.md; `scripts/analysis/panic_skew.py`: it does not give the S&P's skew)
  - `stats.py`: tail exponents (Hill and the paper's regression), kurtosis, autocorrelation, DFA/Hurst H,
    multi-day returns, and the `fingerprint` of a market
  - `market_data.py`: loads and checks the files in `data/`
  - `kirman.py`: Kirman's ant model with its exact solution (the herding check from the proposal)
  - `calibrate.py`: two fits (`python3 -m herding.calibrate paper|spx [--draws N] [--refine]`), each ending
    with a stability screen (100k-day runs on seeds 2000-2019) -> `settings/paper_matched.json`
    (reproduces the paper's numbers) and `settings/sp500_calibrated.json` (reproduces the S&P 500)
  - `experiments.py`: many histories in parallel, percentiles, the typical history
  - `policy.py`: extension: circuit breakers and a policy surprise (paired runs), used by Fig 11
  - `politics.py`: extension: the president's party (1893-2026) and its return gap against luck, used by Fig 12-15
    and the web app (explore.party_view: the luck test for any market; explore.party_shocks: an optional party
    drift in the model's value, 0 = off = the model of SPEC.md)
  - `factors.py`: extension: outside news every trader sees (oil, gold, copper, dollar, 10-year rate, Fed decision days), used by Fig 16-18
  - `plots.py`: palette and drawing helpers (reusable by the future web app)
  - `explore.py`: the science behind the web app: load any stock, index fund or commodity (or a live
    ticker from Yahoo), run the three trader mixes (fundamentalists / chartists without herding / herding),
    and the numbers shown (tail, kurtosis, volatility memory, closest mix)
  - `garch.py`: a formula with no traders (GJR-GARCH), the rival the Checks page's textbook checklist (Cont 2001, `explore.FACTS`) compares herding with
  - `checks.py`: input checks shared by every module (a finite real number; a 1-D list without NaN)
- `.vscode/`: VS Code setup (hides clutter, one-click tasks for tests and figures, suggested extensions)
- `settings/`: the two fitted settings
- `tests/`: one test file per module; `python3 -m pytest tests/ -q` (quick: add `-m "not slow"`).
  Exact answers must match to 12 digits (`conftest.py`); noisy checks use measured tolerances
  (`helpers.py`: RECIPES, TOLERANCE from `scripts/analysis/test_accuracy_audit.py`); how accurate each test is:
  `docs/TEST_ACCURACY.md`; `tolerance_recorder.py` records every comparison.
  Overnight stress test: `tests/stress/overnight.py` (the suite rerun with other seeds, and
  deliberate small bugs it must catch; resumable, results in `scratch/overnight/`). Tests
  marked `pinned_seed` describe one documented history and are skipped with other seeds
- `scripts/analysis/`: scripts behind individual findings (see its README; older versions are in git history)
- `docs/proposal.pdf`: my project proposal; `docs/course_project_guide.pdf`: the course's project handout (local only, git-ignored)
- `data/`: made by `data/download_market_data.py` (all series end 2026-10-30): S&P 500 1950- and 1928- (Yahoo ^GSPC),
  Nasdaq-100 (^NDX, 1985-), `stocks/` (20 large stocks with market beta 0.9-1.1, 2000-), VIX, NBER recessions (FRED USREC),
  Dow Jones 1896- and 1927-1952 (MeasuringWorth; the short file is used to find 11 wrong closes in the 1928 S&P file),
  `big_days_events.csv` (the Dow's 20 biggest days with their documented causes and sources),
  `factors/` (Brent oil, gold, copper, broad dollar, 10-year yield, Fed funds target; FRED and Yahoo),
  `fed_decisions.csv` (every FOMC decision since Feb 1994, from federalreserve.gov); what is still missing: `docs/MISSING_DATA.md`
- `docs/papers/`: sources, all kept locally only (git-ignored). The Lux & Marchesi Nature paper (doi:10.1038/17290) is `docs/papers/LuxMarchesi1999_Nature.pdf`,
  kept locally only (git-ignored: copyrighted, and this repo is public). The equations come from the
  Samanidou et al. (2007) review (see SPEC.md); the parameter tables of Lux & Marchesi (2000) are still
  needed via CMU library access.
- `docs/redesign/`: the approved website redesign, being built (README.md: the plan, its order and what is done; index.html: the clickable mock-up; WORDING.md: the wording guide every page must follow)
- `docs/`: everything to read (KEY_POINTS, SPEC, proposal, papers/), plus my notes from tutoring sessions; `docs/journal_guide.md` points to the key moments in `journal/`
- `scratch/`: temporary files (git-ignored)
- `journal/`: build journal logs (copied automatically; don't edit by hand)

## Working rules
- Keep the smallest working version checked before adding features.
- Commit messages should be in my own words; ask me for one if I haven't given it.
- This repo is public: never put passwords, API keys, or personal info in it.
