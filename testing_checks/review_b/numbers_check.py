import json, numpy as np
from multiprocessing import Pool
from herding import explore as ex, stats
from herding import market_data
d = json.load(open("web/numbers.json"))
p, burn = ex.fitted_settings()
days = d["recipe"]["days"]; seeds = d["recipe"]["seeds"]
def one(s):
    r, e = ex.run_mix(p, "herding", p["a1"], days, s, burn)
    if r is None: return s, None, None
    sm = ex.summary(r)
    return s, sm, float(stats.kurtosis(np.delete(r, int(np.argmin(r))))), float(stats.kurtosis(np.delete(r, int(np.argmax(np.abs(r))))))
if __name__ == "__main__":
    with Pool() as pool: res = pool.map(one, seeds)
    mism = 0
    for i,(s, sm, kw, kwa) in enumerate(res):
        for k in ("tail","kurtosis","memory","worst"):
            if sm[k] != d["checks"]["herding"][k][i]: mism += 1; print("MISMATCH", s, k, sm[k], d["checks"]["herding"][k][i])
    print("exact mismatches:", mism, "of", 4*len(seeds))
    kws = np.array([r[2] for r in res]); kk = np.array(d["checks"]["herding"]["kurtosis"])
    print("model kurt all days: p5 %.2f med %.2f p95 %.2f" % tuple(np.percentile(kk,[5,50,95])))
    print("model kurt w/o worst fall: p5 %.2f med %.2f p95 %.2f" % tuple(np.percentile(kws,[5,50,95])))
    kwa = np.array([r[3] for r in res])
    print("model kurt w/o biggest |move|: p5 %.2f med %.2f p95 %.2f" % tuple(np.percentile(kwa,[5,50,95])))
    spx = ex.log_returns(ex.saved_prices("S&P 500")).values
    print("spx kurt", stats.kurtosis(spx), "w/o worst", stats.kurtosis(np.delete(spx, int(np.argmin(spx)))))
    print("spread", ex.spread(d["checks"]["herding"]["kurtosis"]))
