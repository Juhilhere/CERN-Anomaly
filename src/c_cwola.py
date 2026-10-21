"""Part C (method 2 and reference): CWoLa-style hunt and a supervised upper bound, same data and bump hunt.

CWoLa ("classification without labels"): train a classifier to tell signal-region events from sideband events.
If a signal exists, the only systematic difference between the two samples is the signal's jet substructure,
so the classifier learns it without ever seeing a label. Every event is scored by a model that did NOT train on
it (K-fold cross-fitting), otherwise the classifier would memorise individual events and fake a bump.
The supervised upper bound uses true labels (not available in real data) just to show what is achievable.
"""
import json
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, "src")
import c_anomaly as c

K = 5


def crossfit(X, y, seed):
    s = np.zeros(len(X))
    for tr, te in KFold(K, shuffle=True, random_state=seed).split(X):
        clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=15, random_state=seed)
        clf.fit(X[tr], y[tr])
        s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def evaluate(window, scores, tag, seed):
    in_sr = window.mjj.between(*c.SR).to_numpy()
    thr = np.quantile(scores[~in_sr], 1 - c.KEEP)       # label-free: from sideband scores only
    keep = scores > thr
    kept = window[keep]
    sr = window[in_sr]
    tpr = (keep & in_sr & (window.label.to_numpy() == 1)).sum() / max((sr.label == 1).sum(), 1)
    fpr = (keep & in_sr & (window.label.to_numpy() == 0)).sum() / max((sr.label == 0).sum(), 1)
    bins = np.arange(c.FULL[0], c.FULL[1] + 1, 100.0)
    b = c.bump_hunt(window.mjj.to_numpy(), bins)
    a = c.bump_hunt(kept.mjj.to_numpy(), bins)
    return dict(method=tag, seed=seed, auc_sr=float(roc_auc_score(sr.label, scores[in_sr])) if sr.label.nunique() > 1 else None,
                tpr=float(tpr), fpr=float(fpr), sic=float(tpr / np.sqrt(fpr)) if fpr > 0 else None,
                z_before=b[3], z_after=a[3], obs_after=a[0], exp_after=a[1])


def main():
    df = c.load()
    results = []
    for n_sig, seeds in ((c.N_SIGNAL_INJECTED, range(5)), (0, range(3))):
        for seed in seeds:
            sample = c.make_sample(df, n_sig, seed)
            window = sample[sample.mjj.between(*c.FULL)].reset_index(drop=True)
            X = window[c.FEATURES].to_numpy()
            in_sr = window.mjj.between(*c.SR).to_numpy()
            r = evaluate(window, crossfit(X, in_sr.astype(int), seed), "cwola", seed)
            r["n_signal"] = n_sig
            results.append(r)
            print(r, flush=True)
            if n_sig:
                r = evaluate(window, crossfit(X, window.label.to_numpy(), seed), "supervised_upper_bound", seed)
                r["n_signal"] = n_sig
                results.append(r)
                print(r, flush=True)
    (c.ROOT / "results" / "c_cwola.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
