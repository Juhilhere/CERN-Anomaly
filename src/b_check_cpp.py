"""Part B check: the C++ masses must equal the Python masses event by event."""
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
exe = ROOT / "cpp" / "invmass.exe"
subprocess.run(["g++", "-O2", "-std=c++17", "-Wall", "-Wextra", "-o", str(exe), str(ROOT / "cpp" / "invmass.cpp")], check=True)
out = ROOT / "cpp" / "masses_cpp.txt"
subprocess.run([str(exe), str(ROOT / "data" / "Dimuon_SingleMu.csv"), str(out)], check=True)

df = pd.read_csv(ROOT / "data" / "Dimuon_SingleMu.csv")
MU = 0.1056584
tot = [0, 0, 0, 0]
for i in "12":
    pt, eta, phi = df[f"pt{i}"], df[f"eta{i}"], df[f"phi{i}"]
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    for k, v in enumerate((np.sqrt(px**2 + py**2 + pz**2 + MU**2), px, py, pz)):
        tot[k] = tot[k] + v
m_py = np.sqrt(np.clip(tot[0] ** 2 - tot[1] ** 2 - tot[2] ** 2 - tot[3] ** 2, 0, None)).to_numpy()
m_cpp = np.loadtxt(out)
assert len(m_py) == len(m_cpp)
print("events:", len(m_cpp))
print("max |py - cpp| [GeV]:", np.abs(m_py - m_cpp).max())
print("max relative difference:", np.max(np.abs(m_py - m_cpp) / np.maximum(m_py, 1e-9)))
assert np.allclose(m_py, m_cpp, rtol=1e-9, atol=1e-9)
print("PASS")
