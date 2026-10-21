"""Part C: label-free anomaly search for a dijet resonance (LHC Olympics 2020 R&D data).

Idea (the same logic as a real bump hunt):
  * the dijet invariant mass m_jj defines a signal region (SR) and sidebands (SB);
  * an autoencoder is trained ONLY on sideband events, which are almost pure QCD, so it learns what ordinary
    jets look like and reconstructs them well;
  * in the SR, events it reconstructs badly are "anomalous" (a different jet substructure from QCD);
  * we keep only the worst-reconstructed fraction and look for a bump in m_jj on top of a smooth background
    that is fitted in the sidebands.

Labels are never used for training or for choosing the cut. They are used afterwards, only to score the method.
The m_jj value is never an input feature, so the network cannot simply learn "the SR is anomalous".
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from iminuit import Minuit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SR = (3300.0, 3700.0)      # signal region in m_jj [GeV]
FULL = (2900.0, 4100.0)    # fit range; sidebands = FULL minus SR
N_SIGNAL_INJECTED = 1000   # signal events mixed into 1M background (a realistic ~0.1% contamination)
KEEP = 0.01                # keep the 1% of SR events with the highest anomaly score


def load():
    df = pd.read_hdf(ROOT / "data" / "events_anomalydetection_v2.features.h5")
    E = 0.0
    P = np.zeros((len(df), 3))
    for j in "12":
        p = df[[f"pxj{j}", f"pyj{j}", f"pzj{j}"]].to_numpy()
        m = df[f"mj{j}"].to_numpy()
        E = E + np.sqrt((p**2).sum(1) + m**2)
        P = P + p
    df["mjj"] = np.sqrt(np.clip(E**2 - (P**2).sum(1), 0, None))
    for j in "12":
        df[f"t21j{j}"] = df[f"tau2j{j}"] / df[f"tau1j{j}"].clip(lower=1e-6)
        df[f"t32j{j}"] = df[f"tau3j{j}"] / df[f"tau2j{j}"].clip(lower=1e-6)
    lo = df.mj1 <= df.mj2  # order the two jets by mass so the features do not depend on pT ordering
    out = pd.DataFrame(index=df.index)
    for name in ("m", "t21", "t32"):
        a, b = df[f"{name}j1" if name != "m" else "mj1"], df[f"{name}j2" if name != "m" else "mj2"]
        out[f"{name}_lo"] = np.where(lo, a, b)
        out[f"{name}_hi"] = np.where(lo, b, a)
    # Heavy-tailed positive quantities are log-transformed (chosen on the labelled R&D data: development choice,
    # see REPORT.md). m_jj itself is NOT a feature.
    out["m_lo"], out["m_hi"] = np.log1p(out.m_lo), np.log1p(out.m_hi)
    for name in ("t21_lo", "t21_hi", "t32_lo", "t32_hi"):
        out[name] = np.log(out[name].clip(lower=1e-3))
    out["mjj"] = df.mjj
    out["label"] = df.label.astype(int)
    return out


FEATURES = ["m_lo", "m_hi", "t21_lo", "t21_hi", "t32_lo", "t32_hi"]


def make_sample(df, n_signal, seed):
    rng = np.random.default_rng(seed)
    bkg = df[df.label == 0]
    sig = df[df.label == 1]
    pick = sig.iloc[rng.permutation(len(sig))[:n_signal]]
    return pd.concat([bkg, pick], ignore_index=True)


class AutoEncoder(torch.nn.Module):
    def __init__(self, n_in=6, latent=3):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(n_in, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16),
                                       torch.nn.ReLU(), torch.nn.Linear(16, latent))
        self.dec = torch.nn.Sequential(torch.nn.Linear(latent, 16), torch.nn.ReLU(), torch.nn.Linear(16, 32),
                                       torch.nn.ReLU(), torch.nn.Linear(32, n_in))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_ae(x_train, x_val, seed, epochs=30):
    torch.manual_seed(seed)
    model = AutoEncoder(x_train.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    xt = torch.tensor(x_train, dtype=torch.float32)
    xv = torch.tensor(x_val, dtype=torch.float32)
    best, best_state = np.inf, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), 512):
            b = xt[perm[i:i + 512]]
            opt.zero_grad()
            loss = ((model(b) - b) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = ((model(xv) - xv) ** 2).mean().item()
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.eval()


def score(model, x):
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32)
        return ((model(xt) - xt) ** 2).mean(1).numpy()


def _design(c):
    u = c / 13000.0
    return np.c_[np.ones_like(u), np.log(1 - u), -np.log(u)]  # log f = log a + p1 log(1-u) - p2 log u


def bump_hunt(mjj, bins):
    """Fit the 3-parameter dijet function to the sidebands only (Poisson likelihood), extrapolate under the SR.

    Returns (observed, expected, sigma_expected_from_fit, Z, bin_centres, counts, fit_curve).
    Z = (N - B) / sqrt(B + sigma_B^2): the fit uncertainty on B is included.
    """
    counts, edges = np.histogram(mjj, bins=bins)
    c = 0.5 * (edges[1:] + edges[:-1])
    in_sr = (c > SR[0]) & (c < SR[1])
    X = _design(c)
    y = counts.astype(float)
    w = np.sqrt(np.maximum(y, 1.0))
    beta0 = np.linalg.lstsq((X * w[:, None])[~in_sr], (np.log(np.maximum(y, 1.0)) * w)[~in_sr], rcond=None)[0]

    def nll(*p):
        eta = X[~in_sr] @ np.array(p)
        return float(np.sum(np.exp(eta) - y[~in_sr] * eta))

    m = Minuit(nll, *beta0)
    m.errordef = Minuit.LIKELIHOOD
    m.migrad()
    m.hesse()
    beta = np.array(m.values)
    fit = np.exp(X @ beta)
    expected = float(fit[in_sr].sum())
    grad = (X[in_sr] * fit[in_sr][:, None]).sum(0)  # d(expected)/d(beta)
    sigma_b = float(np.sqrt(grad @ np.array(m.covariance) @ grad))
    observed = int(y[in_sr].sum())
    z = float((observed - expected) / np.sqrt(expected + sigma_b**2))
    return observed, expected, sigma_b, z, c, counts, fit


def run(df, n_signal, seed, make_plot=False):
    sample = make_sample(df, n_signal, seed)
    in_sr = (sample.mjj > SR[0]) & (sample.mjj < SR[1])
    in_sb = (sample.mjj > FULL[0]) & (sample.mjj < FULL[1]) & ~in_sr
    sb = sample[in_sb]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(sb))
    n_val = len(sb) // 5
    mu, sd = sb[FEATURES].iloc[idx[n_val:]].mean(), sb[FEATURES].iloc[idx[n_val:]].std()
    z = lambda d: ((d[FEATURES] - mu) / sd).to_numpy()
    model = train_ae(z(sb.iloc[idx[n_val:]]), z(sb.iloc[idx[:n_val]]), seed)

    window = sample[(sample.mjj > FULL[0]) & (sample.mjj < FULL[1])]
    s_all = score(model, z(window))
    s_sb = score(model, z(sb.iloc[idx[:n_val]]))
    thr = np.quantile(s_sb, 1 - KEEP)  # threshold from sideband validation scores only: label-free
    kept = window[s_all > thr]

    sr = window[(window.mjj > SR[0]) & (window.mjj < SR[1])]
    s_sr = s_all[((window.mjj > SR[0]) & (window.mjj < SR[1])).to_numpy()]
    auc = roc_auc_score(sr.label, s_sr)
    tpr = (kept.mjj.between(*SR) & (kept.label == 1)).sum() / max((sr.label == 1).sum(), 1)
    fpr = (kept.mjj.between(*SR) & (kept.label == 0)).sum() / max((sr.label == 0).sum(), 1)
    bins = np.arange(FULL[0], FULL[1] + 1, 100.0)
    before = bump_hunt(window.mjj.to_numpy(), bins)
    after = bump_hunt(kept.mjj.to_numpy(), bins)
    # sculpting check: does the score correlate with m_jj for pure background?
    bg = window[window.label == 0]
    corr = float(np.corrcoef(score(model, z(bg)), bg.mjj)[0, 1])
    res = dict(seed=seed, n_signal=n_signal, auc_sr=float(auc), tpr=float(tpr), fpr=float(fpr),
               sic=float(tpr / np.sqrt(fpr)) if fpr > 0 else float("nan"),
               z_before=before[3], z_after=after[3], obs_after=after[0], exp_after=after[1],
               score_mjj_corr_bkg=corr)
    if make_plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, r, title in ((axes[0], before, "no anomaly cut"), (axes[1], after, f"top {KEEP:.0%} anomaly score")):
            _, _, _, z_, c, counts, fitv = r
            ax.errorbar(c, counts, np.sqrt(counts), fmt="k.", label="data (blind to labels)")
            ax.plot(c, fitv, "r-", label="sideband fit")
            ax.axvspan(*SR, color="orange", alpha=0.15, label="signal region")
            ax.set(title=f"{title}: Z = {z_:.1f}σ", xlabel="m_jj [GeV]", ylabel="events / 100 GeV")
            ax.legend()
        fig.savefig(ROOT / "figures" / "c_bump_hunt.png", dpi=150, bbox_inches="tight")
    return res


if __name__ == "__main__":
    df = load()
    results = []
    for n_sig, seeds in ((N_SIGNAL_INJECTED, range(5)), (0, range(3))):
        for s in seeds:
            r = run(df, n_sig, s, make_plot=(n_sig > 0 and s == 0))
            print(r)
            results.append(r)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "c_anomaly.json").write_text(json.dumps(results, indent=1))
