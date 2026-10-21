import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ae = json.load(open(ROOT / "results" / "c_anomaly.json"))
cw = json.load(open(ROOT / "results" / "c_cwola.json"))
rows = {
    "no cut": [r["z_before"] for r in ae if r["n_signal"] > 0],
    "autoencoder": [r["z_after"] for r in ae if r["n_signal"] > 0],
    "CWoLa": [r["z_after"] for r in cw if r["method"] == "cwola" and r["n_signal"] > 0],
    "supervised\n(upper bound)": [r["z_after"] for r in cw if r["method"] != "cwola"],
}
ctrl = {
    "autoencoder": [r["z_after"] for r in ae if r["n_signal"] == 0],
    "CWoLa": [r["z_after"] for r in cw if r["method"] == "cwola" and r["n_signal"] == 0],
}
fig, ax = plt.subplots(figsize=(8, 4.5))
for i, (k, v) in enumerate(rows.items()):
    ax.scatter([i] * len(v), v, c="C0", label="1000 signal events injected" if i == 0 else None)
    ax.hlines(np.mean(v), i - 0.25, i + 0.25, colors="C0")
for k, v in ctrl.items():
    i = list(rows).index(k)
    ax.scatter([i + 0.15] * len(v), v, c="C3", marker="x", label="background only (control)" if k == "autoencoder" else None)
ax.axhline(0, c="gray", lw=0.5)
ax.set_xticks(range(len(rows)), list(rows))
ax.set(ylabel="local significance in SR [σ]", title="Dijet bump hunt, 1% working point (5 signal draws, 3 controls)")
ax.legend()
fig.savefig(ROOT / "figures" / "c_summary.png", dpi=150, bbox_inches="tight")
for k, v in rows.items(): print(k.replace("\n", " "), "mean", round(np.mean(v), 2), "min", round(min(v), 2), "max", round(max(v), 2))
for k, v in ctrl.items(): print("control", k, [round(x, 2) for x in v], "mean", round(np.mean(v), 2))
