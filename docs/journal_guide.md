# Guide to my build journal

The logs in `journal/` are complete and unedited (copied automatically). This page points to the
moments that matter. Line numbers are in `journal/readable/<session>.md`.

## Sessions

| Session | What it is |
|---|---|
| `1769bda2` | **The main session** (Oct 27 – Nov 5): everything from the first draft to the rebuild |
| `f7f70605` | An earlier copy of the start of `1769bda2` (same text up to about line 2250) |
| `3715dd0d` | Side question: the VS Code run button |
| `50651951`, `2457780e`, `d1f0e881`, `be5bed11` | Overnight Nov 5–6, sessions I set up to run while I slept: independent check, figures (stages A and B), final review |
| `43cd33d8` | Morning of Nov 6: my decisions on the overnight questions |
| `6e56466b` | Nov 6–9: tidying the folders so VS Code shows the work clearly |
| `562e93e7` | Nov 9–14: "100,000 tests" debugging run with a PDF report; the parties extension |
| `bdb43497` | Nov 16–17: outside news (commodities, rates, Fed days), test precision, the overnight stress test |

## Key moments in `1769bda2`

**Learning the science** (tutoring, with quizzes)
- 426–570: the Lux–Marchesi model, one idea at a time. My answers show what I got right and wrong
  (for example, 478: I first thought returns follow a bell curve because the news does).
- 1435–1707: how the calibrated model works (kurtosis; why one run is not enough, 1695).
- 4131: Lesson 1 of the tutoring on the paper, after the rebuild.

**Checking the work**
- 644–747: first full test of every function; the list of problems.
- 845–1079: `/test-every-function`, then guardrails.
- 1006: finding: stronger herding *between* chartists makes tails thinner. This changed how I state my question.
- 1902–1951: I asked for every decision made without me and every bug: 4 bugs found, then fixed (1910).
- 2867–3136: stress test; the `dt` bug and the Kirman bug (2844); recalibration.
- 3162–3569: checked the project against the paper's PDF (figures included); `KEY_POINTS.md` made.
- 3764–4622: full rebuild from the paper's equations (`docs/SPEC.md`). Findings and bugs at 4595.
- 5018–5442: independent review by a subagent, and a recalibration that I **rejected** because it
  dropped the fat tails (5406).

**Extensions I asked for**
- 1741–1902: more data (1928–, Nasdaq-100, stocks, recessions) and circuit breakers / government policy.

## Overnight Nov 5–6, and the morning after (`43cd33d8`)
The overnight sessions refit the model, checked everything by different methods, and redrew the
figures in the paper's style. Summary: `figures/RESULTS.md`, "Overnight recalibration". In the morning I:
- kept the current settings, following the rule set before seeing results, even though a rival fit
  had a lower held-out loss;
- dropped a hand-picked 3× weight on tail measures after checking published papers (Winker et al. 2007,
  Franke 2009): they weight each measure by its own uncertainty, which our fit already does;
- changed Fig 6 to δ = 0.1, so Kirman's ant story holds literally;
- refused a speed-up that was not bit-identical (reproducibility first);
- corrected H of |returns| to 0.81 ± 0.03.

## Outside news, test precision and the overnight stress test (`bdb43497`, Nov 16–17)
- 3: my request: missing data, news every trader acts on, commodities' beta and correlation.
- 229–268: findings: outside news explains 16% of the S&P's daily moves, and the fat tails and
  memory survive once it is removed (Figs 16–18, `docs/MISSING_DATA.md`).
- 287: "error accuracy of each test": exact tests now demand 12 digits, and noisy ones use
  |bias| + 5 sd from 200 seeds (`docs/TEST_ACCURACY.md`).
- After 583: the overnight stress test (`tests/stress/overnight.py`). The whole suite was
  rerun with 39 other seed sets and 2,638 small deliberate bugs were planted one at a time.
  Found: 25 tests that passed only with their own seeds (now fixed or marked as describing one
  documented history), a model property (chartists gain about 0.0016 of the market a day
  when profits are equal, a time-step effect), a process-pool hang on runaways (fixed), and
  the S&P settings' runaway rate measured on 400 histories (2.75%). Details:
  `docs/TEST_ACCURACY.md`, section 4.

## Git history
Commit messages are in my own words: `git log --oneline`.
