"""Development-only: compare feature sets / scores by SR AUC (uses labels -> NOT an unbiased result)."""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "src")
import c_anomaly as c

df = c.load()
sample = c.make_sample(df, 1000, 0)
in_sr = sample.mjj.between(*c.SR)
in_sb = sample.mjj.between(*c.FULL) & ~in_sr
sb, sr = sample[in_sb], sample[in_sr]

def variants(d):
    o = {}
    o["raw6"] = d[c.FEATURES].to_numpy()
    lg = d[c.FEATURES].copy()
    lg["m_lo"], lg["m_hi"] = np.log1p(lg.m_lo), np.log1p(lg.m_hi)
    o["log-mass6"] = lg.to_numpy()
    lg2 = lg.copy(); lg2["t21_lo"] = np.log(lg.t21_lo.clip(1e-3)); lg2["t21_hi"] = np.log(lg.t21_hi.clip(1e-3))
    lg2["t32_lo"] = np.log(lg.t32_lo.clip(1e-3)); lg2["t32_hi"] = np.log(lg.t32_hi.clip(1e-3))
    o["all-log"] = lg2.to_numpy()
    return o

vs, vr = variants(sb), variants(sr)
rng = np.random.default_rng(0); idx = rng.permutation(len(sb)); nv = len(sb) // 5
for name in vs:
    mu, sd = vs[name][idx[nv:]].mean(0), vs[name][idx[nv:]].std(0)
    z = lambda a: (a - mu) / sd
    model = c.train_ae(z(vs[name][idx[nv:]]), z(vs[name][idx[:nv]]), 0, epochs=25)
    s = c.score(model, z(vr[name]))
    print(f"{name:10s} AE AUC in SR = {roc_auc_score(sr.label, s):.3f}", flush=True)
