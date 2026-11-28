# Next steps (updated 28 Nov 2026, paused to move to the new MacBook)

Claude: read this first, then do the steps in order. Hold Avish to docs/memory.md (quizzes, no "just fix it",
own commit messages, finish lines).

## Where things stand
The website rebuild (docs/redesign/README.md, mock-up docs/redesign/index.html, wording docs/redesign/WORDING.md):
- **Done:** Sources page; audit dark colors; dark mode only (no light colors, no button, one theme in
  .streamlit/config.toml); five pages (Compare and Findings are tabs of Checks: The five checks / Beyond the
  paper / All markets / Figures); scoreboard links that jump to each check; the 8-step Story with a picture or
  chart per step (tests/test_story_steps.py).
- **Partly done:** the Test program regroup (web/page_explore.py). An Opus subagent rebuilt most of it (chart
  tabs, presets Wilder / Best fit / Calmer, lower tabs, sidebar settings) but was stopped before its report.
  Open the page, compare with the mock-up's Test program (#lab), list what is missing, finish it. Avish also
  wants its top bar sleeker.
- All 345 site tests passed at the pause (scratch/handoff_tests.txt). Everything is still **uncommitted**.

## Steps
1. **Moving the project.** Uncommitted work only moves with Migration Assistant or by copying the whole
   folder. If cloning from GitHub instead, commit and push first on the old Mac (Avish's own message).
   Private files that are never on GitHub: docs/papers/, docs/course_project_guide.pdf,
   .claude/hooks/private_terms.local.txt, .streamlit/secrets.toml (if any).
2. **Check the new Mac:** `python3 -m pip install -r requirements.txt`, then
   `python3 -m pytest tests/ -q -m "not slow"`.
3. **Two open exercises for Avish (rule 1: answer before new work):** (a) why delete `[theme.light]` in
   .streamlit/config.toml instead of leaving it unused? (b) how to fix the Dow typos below so the fix is
   visible and checkable?
4. **Finish the Test program** (above), then the **wording pass** on all five pages (WORDING.md; in
   herding/explore.py the scorecard's "Crash days" becomes "Big days (per 1,000 days)" and "Wild days bunch up"
   becomes "Bursts of wild days").
5. **Opus review** of everything the subagents changed (web/page_story.py, web/page_explore.py, their tests):
   no feature lost, every number from the backend, no test weakened. Then look at every page in the browser.
6. **Cleanup, only after Avish confirms this finish line:** remove code the rebuild left unused (vulture, then
   by hand); all site tests pass; show Avish the list; page load times (stress test) no slower; site-vs-backend
   numbers unchanged.
7. **Final checks:** full suite with slow tests; `python3 scripts/analysis/web_stress_test.py`; the Lux &
   Marchesi comparison (Checks → Beyond the paper). Data cross-check already ran (scratch/final_crosscheck.txt).
8. **Commit** in Avish's own words (suggest splitting: science / site rebuild / docs).

## Found, not fixed (rule 2: Avish guesses the fix first)
- The MeasuringWorth Dow has digit typos against Yahoo ^DJI: 2002-09-25 7,481.82 vs 7,841.82 (two digits
  swapped; the Dow rose that day), and about 8 one-digit slips (e.g. 1999-01-26 9,234.58 vs 9,324.58).
  Dow tail 2.79 vs 2.81, big days 62 vs 60. Only the party study uses the Dow.
- Nasdaq-100: Yahoo and FRED differ by up to 2.4% on some 1995-2004 days; tail 3.15 vs 3.19. Unknown which is right.
- Open question for Avish: "according to the Lux-Marchesi paper only" — keep the beyond-the-paper parts
  (GARCH, party study, other markets), which are labeled as such?

## Held back (Dec 1 rule: new features need Avish's one sentence on how they serve the question)
The animated crowd on Start, motion effects, ⌘K search.

## Later, optional
Regenerate the president's-party numbers in figures/RESULTS.md (`python3 scripts/party_figures.py`).
