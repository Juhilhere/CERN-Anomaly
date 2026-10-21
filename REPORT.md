# CERN-Anomaly: report

Author: Juhil Modi. Data: CMS 2011 open data (CERN Open Data record 545, CC0) and LHC Olympics 2020 R&D dataset
(Zenodo 4536377). All numbers below come from the scripts in `src/`; `results/*.json` hold the raw outputs.

## Part A. Dimuon resonances (CMS 2011)
`src/a1_dimuon_spectrum.py`, figures `a1_*.png`

- 83,761 opposite-sign dimuon events. The J/psi (3.1 GeV), Upsilon (~9.5 GeV) and Z (~91 GeV) peaks are visible.
- **Data-quality finding:** for 3,360 of 83,761 events (4.0%) the released `px, py, pz` columns of muon 1 do not match
  `pt, eta, phi` (`hypot(px,py) != pt`). The provided invariant mass `M` agrees with `pt, eta, phi`. I therefore build
  four-vectors from `pt, eta, phi` (max |m - M| < 1 GeV over all events, which is rounding at 4 decimals).
- J/psi fit (unbinned extended likelihood, Gaussian + linear background, 2.8-3.4 GeV): mass 3.0940 +- 0.0009 GeV,
  resolution 32.5 MeV. The world average is 3.0969 GeV, so this is **3 sigma low**. I do not claim a precision measurement:
  the model ignores the radiative tail and momentum-scale effects.

## Part B. C++ re-implementation
`cpp/invmass.cpp`, check `src/b_check_cpp.py`: same four-vector maths in C++17 (g++ -Wall -Wextra, no warnings).
Result: maximum difference to Python 5e-10 GeV over all 83,761 events (PASS).

## Part D. ROOT file round trip
`src/d_root_io.py`: events written with uproot (it writes the newer **RNTuple** format, not a classic TTree), read
back as awkward arrays, four-vectors combined with `vector`. 11,142 events fall in an 80-100 GeV Z window.
Lesson: storing momenta as float32 gives a mass error of up to 1.1 GeV for 3 events with unphysical momenta
(up to 46 TeV), because E^2 - p^2 cancels catastrophically. Use float64 when recomputing masses.

## Part C. Dijet resonance search (LHCO 2020 R&D, 1M QCD + 1000 injected signal events, ~0.1%)
`src/c_anomaly.py`, `src/c_cwola.py`, `src/c_summary.py`, figure `c_summary.png`

Setup. Signal region (SR) m_jj in 3300-3700 GeV, sidebands inside 2900-4100 GeV. Six jet-substructure features
(masses, tau21, tau32 of the two jets, ordered by mass; m_jj is not a feature). Background in m_jj: 3-parameter
dijet function fitted to the sidebands with a Poisson likelihood; significance
Z = (N - B) / sqrt(B + sigma_B^2) includes the fit uncertainty. On pure background the fit predicts 121,353 SR
events against 121,352 observed (Z = 0.0). Working point: keep the top 1% by score; the threshold is set from
sideband scores only, so no label is used.

| method | mean Z in SR (5 signal draws) | range | background-only control (3 runs) |
|---|---|---|---|
| no cut | 1.3 | 1.2-1.3 | 0.0 |
| autoencoder trained on sidebands | 1.2 | -0.3 to 1.8 | -0.2 |
| CWoLa (SR vs sideband classifier, 5-fold cross-fit) | 5.0 | 3.7-7.3 | **+1.3** (0.8-1.6) |
| supervised upper bound (uses labels) | 6.6 | 5.1-7.7 | n/a |

Findings, stated plainly:
1. **The autoencoder did not help.** Its significance is no better than applying no cut (mean SIC 0.9, below 1).
   It introduced no fake bump (control about 0) and its score is almost uncorrelated with m_jj (r = 0.02).
2. **CWoLa recovered most of the available sensitivity** without labels (5.0 vs 6.6 for the supervised bound).
3. **CWoLa has a measurable bias.** On background only it produces a +0.8 to +1.6 sigma false excess, because
   its score depends mildly on m_jj (it keeps 1.5% of SR background instead of 1.0%). A real analysis would need
   a decorrelation or a sideband-interpolation step before quoting a significance.

Limits of this study:
- Feature transforms (logs) were picked by looking at labelled AUC on this same dataset. This is a development
  result, not an unbiased test.
- The 8 runs share one fixed background sample; "seeds" change the injected signal and the fold split only, so the
  control spread understates real statistical fluctuations.
- One signal model (W' -> X(qq) Y(qq), m_X = 500, m_Y = 100 GeV) and one window; no window scan, so no look-elsewhere
  correction. Simulated data only, no detector systematics.
- The autoencoder and the training details were not tuned beyond three feature sets; a better architecture or
  score (e.g. normalising-flow density) may do better. I do not claim anything about autoencoders in general.
