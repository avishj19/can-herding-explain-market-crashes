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
- `main.py`: current draft (simulation + comparison with S&P 500, saves `figures/draft.png`)
- `data/spx_daily.csv`: S&P 500 daily closes, 1950 to today (Yahoo Finance, ^GSPC)
- `paper/`: sources. The Lux & Marchesi Nature paper (doi:10.1038/17290) is paywalled
  and still needs to be added via CMU library access.
- `journal/`: build journal logs (copied automatically; don't edit by hand)

## Working rules
- Keep the smallest working version checked before adding features.
- Commit messages should be in my own words; ask me for one if I haven't given it.
- This repo is public: never put passwords, API keys, or personal info in it.
