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

## Where things live
- `KEY_POINTS.md`: self-contained guide to the project for someone new to it (ideas, figures, findings, limits)
- `notes/SPEC.md`: the model's equations, each with its source (rebuild of November 2026). Code and spec must agree.
- `main.py`: quick comparison (S&P 500 vs 20 held-out S&P-calibrated histories), saves `figures/fig00_quick_check_sp500_vs_model.png`
- `make_figures.py`: makes `figures/fig01-fig11*.png` (descriptive names), explained in `figures/RESULTS.md`; `figures/GALLERY.md` shows them all with plain captions
- `herding/`: the program, one module per job
  - `model.py`: the Lux-Marchesi simulation from SPEC.md (numba-accelerated), runaway guard
  - `stats.py`: tail exponents (Hill and the paper's regression), kurtosis, autocorrelation, DFA/Hurst H,
    multi-day returns, and the `fingerprint` of a market
  - `market_data.py`: loads and checks the files in `data/`
  - `kirman.py`: Kirman's ant model with its exact solution (the herding check from the proposal)
  - `calibrate.py`: two fits (`python3 -m herding.calibrate paper|spx [--draws N] [--refine]`), each ending
    with a stability screen (100k-day runs on seeds 2000-2019) -> `settings/paper.json`
    (reproduces the paper's numbers) and `settings/spx.json` (reproduces the S&P 500)
  - `experiments.py`: many histories in parallel, percentiles, the typical history
  - `policy.py`: extension: circuit breakers and a policy surprise (paired runs), used by Fig 11
  - `plots.py`: palette and drawing helpers (reusable by the future web app)
- `.vscode/`: VS Code setup (hides clutter, one-click tasks for tests and figures, suggested extensions)
- `settings/`: the two fitted settings
- `tests/`: one test file per module; `python3 -m pytest tests/ -q` (quick: add `-m "not slow"`)
- `analysis/`: scripts behind individual findings (see its README; older versions are in git history)
- `notes/proposal.pdf`: my project proposal; `notes/course_project_guide.pdf`: the course's project handout
- `data/`: made by `data/download.py` (all series end 2026-10-30): S&P 500 1950- and 1928- (Yahoo ^GSPC),
  Nasdaq-100 (^NDX, 1985-), `stocks/` (20 large stocks with market beta 0.9-1.1, 2000-), VIX, NBER recessions (FRED USREC)
- `paper/`: sources. The Lux & Marchesi Nature paper (doi:10.1038/17290) is `paper/LuxMarchesi1999_Nature.pdf`,
  kept locally only (git-ignored: copyrighted, and this repo is public). The equations come from the
  Samanidou et al. (2007) review (see SPEC.md); the parameter tables of Lux & Marchesi (2000) are still
  needed via CMU library access.
- `notes/`: my own notes from tutoring sessions; `notes/journal_guide.md` points to the key moments in `journal/`
- `scratch/`: temporary files (git-ignored)
- `journal/`: build journal logs (copied automatically; don't edit by hand)

## Working rules
- Keep the smallest working version checked before adding features.
- Commit messages should be in my own words; ask me for one if I haven't given it.
- This repo is public: never put passwords, API keys, or personal info in it.
