# Wording guide for the site

One question, said the same way everywhere (hero, Story step 1 and Checks):

> **If the news is plain bell-curve noise, can traders copying each other (herding) alone cause the fat tails, crashes and bursts of wild days that real markets show?**

## Sources checked

Loaded: GOV.UK *Clear language* and *A to Z style guide* (numbers, abbreviations), digital.gov plain-language *Writing* guide (it replaced plainlanguage.gov), Mailchimp *Voice and tone* and *Grammar and mechanics*, and ncase.me (only the home page loaded).
Could not load: Investopedia (blocked), Robinhood Learn (404), Our World in Data chart guidance (the FAQ and About pages have none; the OWID rules below come from how their charts are written, not from a page I read).

## Rules (12)

1. **One idea per sentence. Split any sentence over 25 words.** Keep paragraphs to 3 sentences or fewer (GOV.UK sets the limit at 5).
2. **Talk to the visitor as "you" and use the active voice.** Say who does what: "Trend chasers copy the crowd", not "the crowd is copied". Use "we" only for "you and this site, together" (Nicky Case's style).
3. **Define a term once, in place, the first time it appears on each page. After that, use exactly that word.** Never swap in a synonym for variety (GOV.UK, digital.gov).
4. **Prefer a plain word. Keep the technical name in brackets once,** so visitors can look it up: "value traders (economists call them fundamentalists)".
5. **Do not use negative contractions** (don't, can't, doesn't). People misread them (GOV.UK). Positive contractions such as "it's" are fine.
6. **Headings and Story titles state the finding**, not the topic: "With only value traders, daily moves follow a bell curve", not "Test 1" (Our World in Data).
7. **Chart text follows a set pattern.** The caption says what is drawn and gives units ("Across: ... Up: ..."). "What to notice" points to one thing to look at. "Try this" asks for one action.
8. **Write numbers as numerals, with a comma in thousands** (19,329), a leading zero (0.05), % with no space (20%), a real minus sign (−6) and "to" for ranges (2.6 to 3; tables may use a dash). Spell out a number that starts a sentence (GOV.UK, Mailchimp).
9. **Write units in full in prose: "years", never "yrs".** "sd" is the only abbreviation, and you define it first.
10. **Write huge numbers as powers of ten with a real superscript, then say what they mean once:** 10⁸⁶ years ("a 1 followed by 86 zeros"). Never write 10^86 or 1e86.
11. **Buttons start with a verb that says what happens:** Start, Open, Show, Set, Switch, Go to, Copy, Reset. Use the same verb for the same action on every page.
12. **No hype and no advice.** Do not write "proves", "predicts", "secret" or "beat the market". Keep "For educational purposes only" word for word.

## Term list (one name per idea)

| Use this | Plain definition (first use) | Never call it |
|---|---|---|
| **value traders** | Traders who buy when the price is below the true value and sell when it is above. Technical name, once: fundamentalists. | fundamentalists (except once), value investors |
| **trend chasers** | Traders who follow the price trend and copy the crowd's mood. Technical name, once: chartists. | chartists (except once), trend-followers, chasers, speculators |
| **optimists / pessimists** | Trend chasers who expect a rise (they buy) or a fall (they sell). | bulls/bears, buyers/sellers |
| **the crowd's mood** | Whether most trend chasers are optimists or pessimists right now. | sentiment, opinion index |
| **herding** | Trend chasers copying each other's mood. Verb: "copy". | contagion, imitation, mimicry |
| **herding strength (a1)** | How strongly trend chasers copy each other. Show "a1" only next to the slider and in Settings. | copying strength, a1 on its own |
| **reaction to price trends (a2)** | How strongly trend chasers react to recent price moves (Settings only). | trend following (it sounds like the trader type) |
| **true value** | What the stock is really worth. Only the news moves it. | fundamental value, fundamental, "value" on its own |
| **bell-curve news** | News that moves the true value by bell-curve amounts: mostly small, never extreme. | calm news, white noise |
| **bell curve** | The usual "most days are average" shape. Technical name, once: normal distribution. | Gaussian, normal curve |
| **standard deviation (sd)** | The usual size of a day's move. A 4-sd day moves four times as much as usual. | SD, σ, sigma, "typical days" |
| **big day / big rise / big fall** | A day that moves more than 4 sd (up or down / up / down). | extreme move, outlier, event |
| **crash** | A sudden, very large market-wide fall. For numbers, give the size: "an 8-sd fall", "a fall the size of Black Monday". | collapse, meltdown, "8-sd crash" |
| **fat tails** | Big days come far more often than a bell curve predicts. | heavy tails, leptokurtosis |
| **tail exponent** | A score for how common big days are. Lower means more big days. S&P 500 about 3; bell curve about 6. Keep this real, searchable name everywhere and add "(lower = more big days)" on first use per page. Do not add a second name such as "tail score". | tail, tail index, alpha, "TAIL" on its own |
| **excess kurtosis** | 0 for a bell curve; bigger means more very calm and very wild days (fat tails). Always "excess": the site's number is kurtosis minus 3, and plain kurtosis is 3 for a bell curve (corrected 28 Nov after the Opus review). | kurtosis (alone, next to a number) |
| **wild day** | A day that moves much more than usual (any size well above 1 sd). | volatile day |
| **bursts of wild days** | Wild days come in bunches, not alone. Technical name, once: volatility clustering. | clustering, volatility memory, "bunch up", clumps |
| **runs away / ran away** (verb), **runaway** (adjective) | The price leaves its true value and never comes back. | explodes, diverges, blows up |
| **simulated history** | One run of the model from its first day to its last. Plural: simulated histories. | run, simulation, seed, path |
| **seed** | Settings only: "the number that picks one simulated history". | (nothing else) |
| **the model** | The simulated market of 500 traders. Chart legends: "Model", or "Model, value traders only". | herding model, simulated market, simulation |
| **the S&P 500** | First use: "the S&P 500 (the 500 biggest US companies' stocks)". Write "real" only when you contrast it with the model in the same sentence. | real market, the real thing, S&P on its own |
| **simulated trading day** | One day in the model. 77 years = 19,329 days. | step, tick, simulated day |
| **check** | One of the proposal's 5 tests on the Checks page. | test (taken), criterion |
| **experiment** | Story steps 4 to 6 (they were "Test 1 to 3"). The new name avoids a clash with "Test program" and "Checks". | test |
| **extra study** | The work beyond the proposal (the president's party, outside news). | extension |

**Number style:** years: "35 years". Tail exponents: 2 decimals (2.92). Expected counts: 1 decimal (0.8). Waits in years and counts of days: whole numbers with commas. sd sizes: whole numbers ("4 sd", "8-sd fall"). Gaps and distances: as the data gives them (0.012). Write ranges with "to". Write "per 1,000 days", never "/1k". Dates look like "19 October 1987".

## Rewrites

`${...}` parts are left as they are. Strings marked † live outside index.html (data.js / `herding/explore.py`).

| Current (verbatim snippet) | New |
|---|---|
| `The traders copy each other. Is that enough` | Traders copy each other. Is that enough |
| `to test one idea: that` / `crashes come from traders copying each other, not from the news.` | to test one idea: that crashes can come from traders copying each other, even when the news is plain bell-curve noise. |
| `Open the test program` | (no change: this is the standard button) |
| `Right: the model's 500 traders, day by day through its biggest simulated crash.` | The model's 500 traders, day by day, through its biggest crash. |
| `<span style="margin-left:auto;color:var(--muted)">day` | Simulated day |
| `about once in <span class="mono" style="color:var(--accent)">10^86</span> years` | about once in 10<sup>86</sup> years (a 1 followed by 86 zeros) |
| `if the news itself is calm, bell-curve noise, can herding alone create the fat tails and bursts of` / `wild days that real markets show?` | if the news is plain bell-curve noise, can traders copying each other (herding) alone cause the fat tails, crashes and bursts of wild days that real markets show? |
| `"Bell curve · wait for a 20-sd fall", "10^86 yrs", "Black-Monday-sized. Longer than the age of the universe."` | "Bell curve: one 20-sd fall every", "10⁸⁶ years", "A fall the size of Black Monday. Far longer than the universe has existed." |
| `"Real S&P 500 · its wait", fmt(c.spx_years, 0) + " yrs", "Black Monday came within the record since 1950."` | "S&P 500: one such fall in", `fmt(c.spx_years, 0) + " years"`, "Black Monday (1987) came within its record since 1950." |
| `"Herding model · 8-sd fall every", fmt(c.model8, 0) + " yrs", "Black-Monday-sized: every "` … `" yrs. Closer, still too rare."` | "Model: one 8-sd fall every", `… + " years"`, "A fall the size of Black Monday: every " … " years. Closer, but still too rare." (Better still: make `bm_model` the big number so all 3 cards compare the same size of fall.) |
| `Ants copy each other between two food sources.` | Ants copy each other when they choose between two food sources. |
| `Value traders buy cheap and sell dear. Chartists chase the trend and copy the crowd's mood. Traders switch to whichever strategy is paying.` | Value traders buy below the true value and sell above it. Trend chasers follow the trend and copy the crowd's mood. Each trader switches to whichever kind is making more money. |
| `how rare big days are, how wild days bunch up, how long until a crash.` | how rare big days are, whether wild days come in bursts, and how long between crashes. |
| `A 4-sd day is four typical days at once.` | A <b>standard deviation</b> (sd) is the usual size of a day's move. A 4-sd day moves four times as much as usual. (replaces the whole first sentence) |
| `<p>If daily moves followed a <b>bell curve</b>, days that big would almost never come. The real S&amp;P 500 has had dozens.</p>` | `<p>If daily moves followed a <b>bell curve</b> (a "normal distribution"), days that big would almost never come. The S&amp;P 500 has had dozens.</p>` |
| `` `beyond 4 sd since 1950 · a bell curve expects ${fmt(M.spx.bell_big, 1)}` `` | `` `beyond 4 sd since 1950. A bell curve expects ${fmt(M.spx.bell_big, 1)}.` `` |
| `Right of the dotted 4: the bell curve falls off a cliff. The real market's white line keeps going.` | Right of the dotted line at 4 sd, the bell curve drops away. The white S&P 500 line keeps going. |
| `Each step down the side is 10 times rarer. Up = common, down = rare.` | Each step down the left side is 10 times rarer. Higher means more common. |
| `The same rule, copying the crowd, is the "herding" this project puts into a market.` | Copying the crowd like this is called <b>herding</b>. It is the one rule this project adds to a market. |
| `gap between our ant simulation and Kirman's exact math (0 = perfect; must be under 0.05)` | gap between the simulated ants and Kirman's exact formula (0 = perfect match; the check needs under 0.05) |
| `So we build a market of 500 traders, fed only calm news` | So we build a market of 500 traders and feed it bell-curve news |
| `(changes in what the stock is really worth)` | (changes in the stock's <b>true value</b>) |
| `simulated traders, every day for 77 years` | simulated traders, trading every day for 77 years |
| `Only one kind copies the crowd: the trend chasers. Turning that copying up or down is the experiment.` | Only trend chasers (economists call them chartists) copy the crowd. Turning that copying up or down is the experiment. |
| `Buy when the price is below what the stock is worth, sell when it is above.` | Economists call them fundamentalists. They buy when the price is below the true value and sell when it is above. |
| `Trend chasers who think it will rise: they buy. They copy the crowd's mood.` | Trend chasers who expect a rise, so they buy. They copy the crowd's mood. |
| `Trend chasers who think it will fall: they sell. They copy the crowd's mood too.` | Trend chasers who expect a fall, so they sell. They copy the crowd's mood too. |
| `short:"Test 1"` / `"Test 2"` / `"Test 3"` | Experiment 1 / Experiment 2 / Experiment 3 |
| `Test 1: with only value traders, the market is a calm bell curve` | Experiment 1: with only value traders, daily moves follow a bell curve |
| `So the news alone does not make crashes.` | So bell-curve news alone does not make big days. |
| `` `beyond 4 sd in 77 simulated years · the real market had ${M.spx.big}` `` | `` `beyond 4 sd in 77 simulated years. The S&P 500 had ${M.spx.big} since 1950.` `` |
| `Test 2: trend chasers who don't copy each other make the market run away` | Experiment 2: without herding, trend chasers make the price run away |
| `Real markets don't do that.` | Real markets do not do that. |
| `title:{text:"Chasers"}` / `title:{text:"Mispricing"}` / `title:{text:"Trading day"}` | Trend chasers / Price above true value / Simulated trading day |
| `name:"Price above value"` | Price above true value |
| `Test 3: trend chasers who copy each other give the real market's fat tails` | Experiment 3: with herding, the model gets the S&P 500's fat tails |
| `Big days now come about as often as in the real S&amp;P 500, and in <b>bursts</b>, like the real thing.` | Big days now come about as often as in the S&amp;P 500, and they come in <b>bursts</b>, as they do there. |
| `` `tail exponent of the model (median of 40 runs) · real S&P 500: ${fmt(M.spx.tail)} · a bell curve: about 6` `` | `` `tail exponent of the model (lower = more big days), middle value of 40 simulated histories. S&P 500: ${fmt(M.spx.tail)}. Bell curve: about 6.` `` |
| `The pink herding line follows the white S&P line far past 4 sd. The bell curve gave up long before.` | The pink model line follows the white S&P 500 line far past 4 sd. The bell curve drops away long before. |
| `Copying trend chasers (pink) against the real market (white).` | The model with herding (pink) against the S&P 500 (white). |
| `How much copying? Guess first` | How much herding? Guess first |
| `Too little copying and the market runs away; more copying calms it down. The real market's tails show up only near the edge between the two.` | With too little herding the price runs away. More herding calms it. The S&P 500's fat tails appear only near the edge between the two. |
| `"the herding strength that matches the S&P 500: right at the edge" : "make a guess"` | "the herding strength that best matches the S&P 500, right at that edge" : "guess to see the answer" |
| `(10 runs each). Bars: how many runs ran away.` | (10 simulated histories each). Bars: how many of the 10 ran away. |
| `Tail exponent (higher = calmer)` | Tail exponent (lower = more big days) |
| `a market of copying traders reproduces most of what makes real markets wild. Two things it still misses.` | a market with herding reproduces most of what makes real markets wild. It still misses two things. |
| `"between 8-sd crashes in the model · a bell curve: trillions of years"` | "between 8-sd falls in the model. A bell curve: trillions of years." |
| `` Big days as common as in the real S&amp;P 500 (tail ${fmt(M.checks.tail_median)} vs ${fmt(M.spx.tail)}). `` | `` Big days as common as in the S&amp;P 500 (tail exponent ${fmt(M.checks.tail_median)}; S&amp;P 500 ${fmt(M.spx.tail)}). `` |
| `<b>Wild days in bursts</b>, like real markets.` | <b>Bursts of wild days</b>, like the S&amp;P 500's. |
| `<b>Black Monday's size</b> comes about every` … `the real market had one in ${fmt(c.spx_years, 0)}.` | A <b>fall the size of Black Monday</b> comes about every … the S&amp;P 500 had one in ${fmt(c.spx_years, 0)} years. |
| `Real crashes are bigger than real rallies; the model's are mirror images.` | In the S&amp;P 500 the biggest falls are bigger than the biggest rises. In the model they are the same size. |
| `Try it yourself in the Test program</a> · <a href="#checks">See every check` | Open the test program</a> · <a href="#checks">See all the checks |
| `"Try it yourself →" : "Next →"` | "Open the test program →" : "Next →" |
| `"Wild-day bursts"` (VIEWS, TRY, CAP, ACTIONS) | "Bursts of wild days" |
| `Count the triangles: days beyond 4 standard deviations. A bell curve expects about one in this whole history.` | Count the triangles: each is a big day (beyond 4 sd). A bell curve expects about one in all these years. |
| `Look just left of the fall: chartists pile in, the crowd leans one way, then it breaks.` | Look just before the fall: trend chasers pile in, most turn optimist, then they flip together. |
| `watch the pink line swing: too little herding is wild, too much is calm. Where does it sit on the white S&P line?` | watch the pink model line move: less herding is wilder, more is calmer. Where does it match the white S&P 500 line? |
| `Do the tall spikes come alone or in clumps? Real market on top, model below, same scale.` | Do the tall spikes come alone or in bursts? S&P 500 on top, model below, same scale. |
| `The S&P 500 since 1950, log scale. ▲ big rises, ▼ big falls.` | The S&P 500 since 1950. Log scale: equal heights mean equal % changes. ▲ big rise, ▼ big fall (beyond 4 sd). |
| `The herding model's biggest crash, a year around it: price and true value, share of chartists, optimists − pessimists.` | The model's biggest crash, with about a year around it. Top: price and true value. Middle: share of traders who are trend chasers. Bottom: optimists minus pessimists. |
| `A straight line means fat tails.` | A line that stays straight instead of dropping away means fat tails. |
| `Each day's move in standard deviations; both panels share one scale.` | Each day's move, in sd. Both panels use the same scale. |
| `"Move size (standard deviations)"` and `"Size of the day's move (standard deviations)"` | Size of the day's move (sd) |
| `name:"Fundamentalists only"` | Model, value traders only |
| `name:"Herding model"` (3 places) | Model |
| `title:{text:"Chartists"}` / `title:{text:"Crowd"}` / `"S&P (sd)"` / `"Simulated day"` | Trend chasers / Mood / S&P 500 (sd) / Simulated trading day |
| `Demo a price tick` | Show a sample price change |
| `<span>wild</span><span>fit</span><span>calm</span>` | wilder / best fit / calmer |
| `Runs as you slide; the chart crossfades to the new run.` | The model reruns as you drag. The chart switches to the new simulated history. |
| `{match:"✓ MATCH", close:"≈ CLOSE", off:"✗ OFF"}` | ✓ MATCHES / ≈ CLOSE / ✗ FAR OFF |
| `` model ${fmt(r.model)} · S&amp;P ${fmt(r.market)} `` | `` Model ${fmt(r.model)} · S&amp;P 500 ${fmt(r.market)} `` |
| `This history ran away: the price left its value for good.` | This simulated history ran away: the price left its true value for good. |
| `a2, v1, days, seed, other markets and the run log:` | More model settings, other markets and the history log are in |
| † `"Crash days"` / `"Wild days bunch up"` | Big days (per 1,000 days; it counts days beyond 4 sd) / Bursts of wild days |
| `<th>PROPOSAL SAID</th>` | THE PROPOSAL EXPECTED |
| `` `Distance ${M.kirman.tv.toFixed(4)} (limit 0.05)` `` | `` `Gap ${M.kirman.tv.toFixed(4)} (needs under 0.05)` `` |
| `"2. No chartists", "A bell curve", `` `Kurtosis ${fmt(C.kurt_median, 3)}` `` | "2. No trend chasers", "A bell curve (kurtosis near 0)", `` `Kurtosis ${fmt(C.kurt_median, 3)} (0 = bell curve)` `` |
| `"Within decades, not 10^86 years"` | "Within decades, not 10⁸⁶ years" |
| `` `8-sd falls every ${fmt(c.model8, 0)} yrs; Black Monday size every ${fmt(c.bm_model, 0)}` `` | `` `An 8-sd fall every ${fmt(c.model8, 0)} years; a fall the size of Black Monday every ${fmt(c.bm_model, 0)} years` `` |
| `"✓ passes"` / `"≈ partly"` / `"no pass line"` | ✓ Passes / ≈ Partly / No pass mark set |
| `"5. Extremes, clustering", "Comparable to the S&P 500", "See the table"` | "5. Big days and bursts of wild days", "Comparable to the S&P 500", "Compared in the table below" |
| `Each check opens below its row on the real site, with its rule, its chart and how the rule is computed.` | On the finished site, each check opens below its row with its rule, its chart and how it is calculated. |
| `Everything from today's site, kept. Nothing removed: it moved here.` | Every option from the current site is here. |
| `Any ticker (Yahoo)` | Any stock symbol (from Yahoo Finance) |
| `Live prices and 1D / 7D within the day` | Live prices, with 1-day and 7-day charts |
| `Mark big days (over 4 sd)` | Mark big days (beyond 4 sd) |
| `Drawdown panel` | Show the fall from the last peak (drawdown) |
| `Log price scale` | Log price scale (equal heights = equal % changes) |
| `President's party shading (extension)` | Shade by the president's party (extra study) |
| `MODEL, ADVANCED` | MORE MODEL SETTINGS |
| `Trend following (a2)` | Reaction to price trends (a2) |
| `Mood changes (v1)` | How often trend chasers change mood (v1) |
| `Days · Seed` | Simulated trading days · Seed (picks the history) |
| `Party drift (extension)` / `0 %/yr` | Party effect on true value (extra study) / 0% a year |
| `Run log · every run on one chart` | History log: every history you ran, on one chart |
| `"Chart: Price"` (and the 3 other `"Chart: …"`) | "Show chart: Price" (same pattern for the others) |
| `"Herding: wild (0.9)"` / `"Herding: fitted to the S&P 500"` / `"Herding: calm (1.4)"` | "Set herding: 0.9 (wilder)" / "Set herding: best fit to the S&P 500" / "Set herding: 1.4 (calmer)" |
| `"Market: S&P 500"` (and every `"Market: …"`) | "Switch market: S&P 500" (same pattern) |
| `"Type any ticker"` | "Switch market: type any stock symbol" |
| `"Live prices on/off"` | "Turn live prices on or off" |
| `"Advanced model settings (a2, v1, days, seed)"` | "Open more model settings" |
| `"President's party (extension)"` | "Shade by the president's party (extra study)" |
| `["Run log", "setting"` | "Open the history log" |
| `"Go to: Sources and citations"` | "Go to: Sources" |
| `"view", () => pick` / `"model", () => setA1` (category tags) | "chart" / "herding" |
| `CITATIONS · PAPERS · DATA · TOOLS` | PAPERS · DATA · TOOLS |
| `The paper's settings tables (not in hand; the site fits its own).` | The paper's tables of settings (not yet obtained, so this site fits its own). |
| `The target for the tail exponent.` | The target for the tail exponent: about 3. |
| `The DFA method behind the Hurst H.` | The method that measures how long wild spells last (DFA, Hurst H). |
| `GARCH, the formula with no traders.` | GARCH: a rival formula for bursts of wild days, with no traders in it. |
| `The GARCH term for falls.` | The part of GARCH that makes falls bigger than rises. |
| `The "panic" idea: tried, and it does not give the skew.` | The "panic" idea: tested here. It does not make falls bigger than rises. |
| `The president's-party analysis.` | The extra study on the president's party. |
| `` `TAIL <b>${fmt(s.tail)}</b>` `` / `` `BIG DAYS <b>${s.big}</b> (bell ${fmt(s.bell_big, 1)})` `` | `` `TAIL EXPONENT <b>${fmt(s.tail)}</b>` `` / `` `BIG DAYS <b>${s.big}</b> (BELL CURVE ${fmt(s.bell_big, 1)})` `` |
| `` `MODEL a1 <b>${r.a1}</b> · TAIL <b>${r.tail ? fmt(r.tail) : "ran away"}</b>` `` | `` `MODEL HERDING <b>${r.a1}</b> · TAIL EXPONENT <b>${r.tail ? fmt(r.tail) : "RAN AWAY"}</b>` `` |
| `` `BELL-CURVE WAIT, 20 SD <b>10^86 yrs</b>` `` / `` `MODEL WAIT, 8 SD <b>${fmt(M.crash_years.model8, 0)} yrs</b>` `` | `` `BELL CURVE: ONE 20-sd FALL EVERY <b>10⁸⁶ YEARS</b>` `` / `` `MODEL: ONE 8-sd FALL EVERY <b>${fmt(M.crash_years.model8, 0)} YEARS</b>` `` |

Unchanged on purpose: "For educational purposes only" (both places), the page names (Start, Story, Test program, Checks, Sources), "Start the story", "Open the test program", "Next →" and "← Back", "Kirman's ants", the eyebrow "THE ARGUMENT IN 8 STEPS · ABOUT 3 MINUTES", and every number.
