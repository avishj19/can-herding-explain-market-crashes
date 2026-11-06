"""Can herding explain market crashes? The Lux-Marchesi model and its checks.

Modules (the equations are in docs/SPEC.md):
    model        the Lux-Marchesi market simulation
    stats        tail exponents, kurtosis, autocorrelation, DFA, the fingerprint
    market_data  the S&P 500 and other market data, loaded with checks
    kirman       Kirman's ant model and its exact solution
    calibrate    fit the settings to the paper or to the S&P 500 (python3 -m herding.calibrate)
    experiments  run many histories in parallel; percentiles; typical history
    policy       extension: circuit breakers and a policy surprise (Fig 11)
    politics     extension: Democratic vs Republican presidents, against luck (Fig 12)
    plots        palette and drawing helpers shared by every figure
"""
