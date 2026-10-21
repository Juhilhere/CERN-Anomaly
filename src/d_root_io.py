"""Part D: write the dimuon events to a ROOT file with uproot (it writes an RNTuple, the successor of TTree), read them back lazily with awkward arrays,
select a mass window with a boolean mask, and check nothing was lost in the round trip."""
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot
import vector

vector.register_awkward()
ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "Dimuon_SingleMu.csv")
MU = 0.1056584

branches = {
    "run": df.Run.to_numpy(np.int32),
    "event": df.Event.to_numpy(np.int64),
    "pt": df[["pt1", "pt2"]].to_numpy(np.float64),
    "eta": df[["eta1", "eta2"]].to_numpy(np.float64),
    "phi": df[["phi1", "phi2"]].to_numpy(np.float64),
    "charge": df[["Q1", "Q2"]].to_numpy(np.int8),
}
path = ROOT / "data" / "dimuon.root"
with uproot.recreate(path) as f:
    f["Events"] = {
        "run": branches["run"],
        "event": branches["event"],
        "muon_pt": ak.Array(branches["pt"]),
        "muon_eta": ak.Array(branches["eta"]),
        "muon_phi": ak.Array(branches["phi"]),
        "muon_charge": ak.Array(branches["charge"]),
    }

with uproot.open(path) as f:
    tree = f["Events"]
    print(tree.show())
    arr = tree.arrays(["muon_pt", "muon_eta", "muon_phi"])

mu = ak.zip(
    {"pt": arr.muon_pt, "eta": arr.muon_eta, "phi": arr.muon_phi, "mass": ak.full_like(arr.muon_pt, MU)},
    with_name="Momentum4D",
)
pair = mu[:, 0] + mu[:, 1]
m = ak.to_numpy(pair.mass)
in_z = (m > 80) & (m < 100)
print("events in Z window 80-100 GeV:", int(in_z.sum()), "of", len(m))
print("median m in Z window:", float(np.median(m[in_z])))
print("float64 storage: max |m - csv M| [GeV]:", float(np.abs(m - df.M.to_numpy()).max()))
