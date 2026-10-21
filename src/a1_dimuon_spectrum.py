"""Part A: rediscover the dimuon resonances in CMS 2011 open data.

Recomputes the invariant mass from the muon four-vectors (instead of trusting
the provided M column), checks the two agree, plots the spectrum on log axes,
and fits the J/psi peak with a Gaussian signal plus a linear background.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "Dimuon_SingleMu.csv")
print(df.shape)
print(df.columns.tolist())

MU = 0.1056584  # muon mass [GeV]


def four_vector(i):
    pt, eta, phi = df[f"pt{i}"], df[f"eta{i}"], df[f"phi{i}"]
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return np.sqrt(px**2 + py**2 + pz**2 + MU**2), px, py, pz


v1, v2 = four_vector(1), four_vector(2)
e, px, py, pz = (a + b for a, b in zip(v1, v2))
mass = np.sqrt(np.clip(e**2 - px**2 - py**2 - pz**2, 0, None))

# Data-quality checks on the released CSV
mismatch = (np.hypot(df.px1, df.py1) - df.pt1).abs() > 0.01
print(f"muon-1 rows where hypot(px,py) != pt: {mismatch.sum()} of {len(df)} ({mismatch.mean():.1%})")
print("max |m(pt,eta,phi) - provided M| [GeV]:", float(np.abs(mass - df.M).max()))
print("opposite-sign fraction (file is pre-selected):", float((df.Q1 * df.Q2 < 0).mean()))

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.logspace(np.log10(0.3), np.log10(120), 400)
ax.hist(mass, bins=bins, histtype="step")
ax.set(xscale="log", yscale="log", xlabel="m(μμ) [GeV]", ylabel="events / bin")
fig.savefig(ROOT / "figures" / "a1_dimuon_spectrum.png", dpi=150, bbox_inches="tight")

lo, hi = 2.8, 3.4
x = mass[(mass > lo) & (mass < hi)].to_numpy()


def pdf_cdf(xx, n_sig, n_bkg, mu, sigma, slope):
    from scipy.stats import norm

    sig_norm = norm.cdf(hi, mu, sigma) - norm.cdf(lo, mu, sigma)
    sig = n_sig * norm.pdf(xx, mu, sigma) / sig_norm
    c = 1.0 + slope * (xx - (lo + hi) / 2)
    area = (hi - lo) + slope * 0.0
    bkg = n_bkg * c / area
    return n_sig + n_bkg, sig + bkg


cost = ExtendedUnbinnedNLL(x, pdf_cdf)
fit = Minuit(cost, n_sig=len(x) * 0.5, n_bkg=len(x) * 0.5, mu=3.097, sigma=0.04, slope=0.0)
fit.limits["sigma"] = (0.005, 0.2)
fit.limits["slope"] = (-1.5, 1.5)
fit.limits["n_sig"] = (0, None)
fit.limits["n_bkg"] = (0, None)
fit.migrad()
fit.hesse()
print(fit)
print(f"J/psi mass = {fit.values['mu']:.4f} +- {fit.errors['mu']:.4f} GeV (PDG 3.0969)")
print(f"width sigma = {fit.values['sigma']*1000:.1f} MeV (detector resolution)")

fig, ax = plt.subplots(figsize=(8, 5))
counts, edges = np.histogram(x, bins=120, range=(lo, hi))
centres = 0.5 * (edges[1:] + edges[:-1])
ax.errorbar(centres, counts, np.sqrt(counts), fmt="k.", label="CMS 2011")
w = edges[1] - edges[0]
ax.plot(centres, pdf_cdf(centres, *fit.values)[1] * w, "r-", label="Gauss + linear")
ax.set(xlabel="m(μμ) [GeV]", ylabel=f"events / {w*1000:.0f} MeV")
ax.legend()
fig.savefig(ROOT / "figures" / "a1_jpsi_fit.png", dpi=150, bbox_inches="tight")
