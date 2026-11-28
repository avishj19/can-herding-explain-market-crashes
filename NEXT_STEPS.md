# Next steps (updated 28 Nov 2026, evening, new MacBook). READ THIS FIRST.

Claude: hold Avish to docs/memory.md. Keep sessions lean (weekly usage is limited). Run tests with
`env -u PYTHONDONTWRITEBYTECODE python3 -m pytest ...` (this shell's PYTHONDONTWRITEBYTECODE=1 makes app.py reload
modules in every test process and causes false failures).

## Already pushed (28 Nov)
"the crash check", "site improvements in UI", "redesign commit for the plans we have made": the redesign, the Opus
review's 7 fixes. Live site: https://herding-crashes.streamlit.app (sleeps when idle; wake it a few minutes before a demo).

## Phase A (done, not committed): independent re-check, reports in testing_checks/
- a1_tail_range: proposal's 2.6-3 is fine; Gopikrishnan's S&P 500 gives 2.93 / 3.02; "herding strength" is the right term.
- a2_sweep: saved numbers reproduce exactly; fresh seeds (700000+) reproduce every measure; Black Monday wait very uncertain.
- a4_engine: no safe speed-up worth applying; engine unchanged.
- a6_data: data inventory, delete lists, stock set (REPORT.md saved by the main session).
- Opus review of all four: tail match is partly by design (fit targets: stats.FINGERPRINT); 2.5-3.5 was chosen after
  the results; app.py SITE_MODULES must drop politics/factors when they are deleted.

## Phase B
1. DONE (not committed): check 3 judged against the proposal's 2.6-3.0 (explore.PROPOSAL_TAIL, tail_check(close=...),
   tail_interval = S&P's own 95% range ~2.57-3.34) -> scoreboard "🟡 Close". Honesty lines with tests
   (tests/test_review_fixes.py): partly-fitted tail, Black Monday wait with its Poisson range (explore.wait_interval,
   common.wait_range), excess kurtosis without Black Monday 11.4. Quick suite 1,487 passed; stress test 0 failures.
2. DONE (not committed; an Opus agent): steps 3-6, in order, tests after each:
   3 remove outside news (factors.py, Fig 16-18, 4 data files; oil/gold/copper moved to data/commodities/);
   4 remove the party study (politics.py, Fig 12-15, the `party` parameter end to end, app.py SITE_MODULES);
   5 stocks -> MSFT, DIS, HD, NKE, HON (rule: beta 0.9-1.1, household name, 5 sectors; NO kurtosis filter), rerun Fig 8;
   6 delete data/vix_fear_index_daily.csv and data/us_recessions_monthly.csv; then docs.
   Gates passed: quick suite green after each step; stress test 0 failures after step 4 and at the end;
   web/numbers.json unchanged (rerun into scratch/removals/: checks, sweep, crashes and facts identical; the
   paper comparison differs only in the 15th-16th digit, from the statistics, not the model). Fig 8 regenerated
   (11 of 70 cells unusual) and its RESULTS.md/GALLERY.md text rewritten. Docs updated. Suggested commits, one per
   step (3 outside news, 4 party study, 5 stocks, 6 VIX/recessions, docs), messages in Avish's words.
3. TODO after that: web/page_sources.py: one line that the proposal planned Stooq, but Stooq now blocks programs, so
   Yahoo with an independent FRED cross-check replaced it; add Dewachter & Lyrio (technical trading rules, cited in the
   proposal). Avish's own reflection on this belongs in Avish's write-up/video in Avish's own words, not on the site.
4. DONE: Sources note (Stooq -> Yahoo + FRED, within 0.12%) and Dewachter & Lyrio. Final Opus review (testing_checks/
   review_b/): fixed its findings (like-for-like kurtosis via explore.kurtosis_without_worst and numbers.json
   checks.herding.kurtosis_wo_worst, only that field added; fit-target wording; tail_interval block 250; README).
   Full suite 1,351 passed before these fixes; quick suite green after; stress test 0 failures.
   OPEN, FOR AVISH (rule 2, guess first): seed 1025's "92.6-sd fall" is not a crash but an overshoot: 99% of traders
   turn trend chasers, the price goes 413 -> 74 -> 286 in two days (-82%, +285%). It is one of the two
   Black-Monday-sized falls behind check 4; seed 1004's -26% is a real crash. Not fixed; the engine is unchanged.
   Also: docs/redesign/more.js (mock-up data) is stale; rerun docs/redesign/make_more.py if the mock-up matters.
5. TODO: commits, one per step, messages in Avish's own words (rule 5; ask Avish). Then push.

## Open, for Avish
- Journal file journal/1769bda2-....jsonl is 95 MB; GitHub refuses files over 100 MB -> Git LFS for journal/*.jsonl.
- Motion and ⌘K search: held back by the Dec 1 rule.
- Before Dec 3 (rule 9): explain herding/model.py, stats.hill_alpha and every proposal check out loud.
