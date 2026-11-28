# The website redesign: plan and mock-up (November 2026)

Approved by Avish on 27 Nov 2026 ("build it"). Built so far: the Sources page and the new dark colors (27 Nov);
dark mode only (item 1), five pages (item 2) and the Checks tabs with scoreboard links (item 6) (28 Nov). The Story and the
Test program are being rebuilt; the wording pass follows. Held back by the Dec 1 "no new features" rule until
Avish says how they help the project's question: the animated crowd (item 3), motion (item 9) and ⌘K search (item 10).

## See the mock-up

```bash
python3 docs/redesign/make_more.py      # once: copies Plotly's library and the figures here (about a minute)
python3 -m http.server 8765 --directory docs/redesign
```

Then open http://localhost:8765. (In Claude Code's browser pane: the `redesign-mockup` entry in
`.claude/launch.json`.) Every number in it is real: `make_data.py` and `make_more.py` write `data.js` and
`more.js` from the saved data and the fitted model.

## What to build (the mock-up shows each piece)

1. **Dark mode only.** Remove the light colors and the light/dark button. New dark colors from the
   readability audit (every line at least 3:1 against the chart, every colored label 4.5:1, checked for
   colorblindness): BELL #737A87, GUIDE #8A8D99, GRID #2E2E2E, LINE #4A4A4A, MUTED #A29F98,
   SHADE_LOW #7A8496, value traders #3D9BE0, trend chasers #E66A1F, herding #F2A7DA,
   HERD_FILL rgba(242,167,218,0.10), DOWN_FILL rgba(255,122,107,0.10). Main lines at least 1.6 px,
   chart text at least 12 px, the ±4 sd guides drawn above the data, both Bursts panels on one scale.
2. **Five pages:** Start · Story · Test program · Checks · Sources. Compare and Findings move inside
   Checks as tabs (nothing deleted).
3. **Start:** the question as the headline; the animated crowd of 500 traders (the model's own run
   through its biggest crash, drawn in a small script with a Pause button; still for reduced motion);
   "Why build this" with the three wait cards (bell curve 10⁸⁶ years, S&P 500 77 years, model 1,534
   years for a Black-Monday-sized fall); "How the program answers it" in three illustrated steps;
   a glossary.
4. **Story:** 8 guided steps (Next / Back, a step list, a progress bar): the puzzle, the idea (ants),
   the market, experiments 1-3, "how much herding?" (guess first), the verdict.
5. **Test program:** everything the current page does, regrouped: the three-tasks card; presets
   (Wilder / Best fit / Calmer); four chart tabs (Price with range buttons and drawdown; Watch a crash
   with trader-mix and zoom pickers; How rare big days are with Tail / Histogram; Bursts of wild days
   with Bursts / Memory); "Try this" and a one-line insight under each; the scorecard; four tabs below
   (Model and S&P 500, Big days with the reader, History log, Every history on one chart). Other
   markets, live prices, chart switches, a2 / v1 / days / seed, the party study and the share link go
   in the sidebar ("Settings"). Runs are automatic (no Run button).
6. **Checks:** a table of the 5 checks; click a row to open its rule and chart (ants with the ε
   slider, the two dot charts, the crash chart, the comparison table). Tabs: Beyond the paper (the
   textbook checklist vs GARCH, the paper comparison, the party study as "extra study"), All markets
   (today's Compare map), Figures (today's gallery).
7. **Sources:** "For educational purposes only" at the top; 15 references in groups with what each
   was used for; data and tools.
8. **Wording:** follow `WORDING.md` everywhere: one name per idea (value traders, trend chasers,
   herding strength, sd, big day, bursts of wild days, simulated history, true value), numbers as in
   its number style, 10⁸⁶ not 10^86. In `herding/explore.py` the scorecard's "Crash days" becomes
   "Big days (per 1,000 days)" and "Wild days bunch up" becomes "Bursts of wild days".
9. **Motion, kept snappy** (0.14-0.42 s, ease cubic-bezier(.2,.8,.2,1)), and none for reduced motion:
   a sliding tab highlight, hover lift on cards, a quick fade when a chart changes, a scrolling tape
   that pauses on hover, a pinned tape and navigation. Tested: Plotly's smooth morph between herding
   settings does **not** survive a Streamlit rerun (the chart swaps in one jump), so use a fade.
10. **⌘K search and the S key for Settings** need a small injected script; build them last, with a
    plain fallback, since a Streamlit update could break them.

## Suggested order and time (about 3-5 hours)

Theme and colors (30 min) → five pages and Sources (20) → Start (30) → Story (40) → Test program
(1.5 h) → Checks (40) → wording, full tests, browser check (45). Keep the tests passing after each
stage; where a test checks something changed on purpose (light mode, the Compare page), update it to
check the new version.

A backup of the site before the rebuild was in `scratch/backup_before_rebuild/` on the old Mac
(git-ignored, so not on GitHub; the last commit has the same code once this work is committed).
