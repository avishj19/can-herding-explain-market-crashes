"""Can herding explain market crashes? The Lux-Marchesi model and its checks.

Modules (the equations are in notes/SPEC.md):
    model        the Lux-Marchesi market simulation
    stats        tail exponents, kurtosis, autocorrelation, DFA, the fingerprint
    market_data  the S&P 500 and other market data, loaded with checks
    kirman       Kirman's ant model and its exact solution
    calibrate    fit the settings to the paper or to the S&P 500 (python3 -m herding.calibrate)
    experiments  run many histories in parallel; percentiles; typical history
    plots        palette and drawing helpers shared by every figure
"""
