# Data (not stored in git)

Put these files in `data/`:

| file | source | licence |
|---|---|---|
| `Dimuon_SingleMu.csv` | CERN Open Data record 545: https://opendata.cern.ch/record/545 | CC0 |
| `events_anomalydetection_v2.features.h5` | LHC Olympics 2020 R&D dataset, Zenodo 4536377: https://zenodo.org/records/4536377 | see Zenodo record |

`data/dimuon.root` is produced by `src/d_root_io.py`.

## Run order
```
python -m venv .venv && .venv/Scripts/pip install uproot awkward vector numpy pandas scipy scikit-learn h5py tables matplotlib iminuit torch
python src/a1_dimuon_spectrum.py
python src/b_check_cpp.py          # needs g++
python src/d_root_io.py
python src/c_anomaly.py && python src/c_cwola.py && python src/c_summary.py
```
