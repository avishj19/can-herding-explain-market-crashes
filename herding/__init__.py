"""Can herding explain market crashes? The Lux-Marchesi model and its checks.

Modules:
    model        the Lux-Marchesi market simulation, with guardrails
    stats        tail exponent, kurtosis, autocorrelation, stylized-facts fingerprint
    market_data  the S&P 500 data, loaded with checks
    kirman       Kirman's ant model and its exact solution
    calibrate    fit the model's settings to the S&P 500 (python3 -m herding.calibrate)
    plots        figures comparing the model with the real market
"""
