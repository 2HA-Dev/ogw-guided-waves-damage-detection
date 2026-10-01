"""Extraction of one frequency from the OGW dataset no. 2 (infrastructure tool).

Usage: python3 tools/extract_frequency.py 120
Writes data/ogw_f<F>kHz.h5 containing, for the 966 measurements sorted by class
then by date:
  catch       (N, 66, 13108) float32   received signal, full resolution (10 MHz)
  label       (N,)  str               udam, dam_D04, dam_D12, dam_D16, dam_D24
  damaged     (N,)  int8              0 healthy, 1 damaged
  cycle       (N,)  int8              1 or 2 for healthy (chronological order), 1 otherwise
  T_ctc, T_meas_1, T_meas_2, humidity, timestamp  (N,) float64
  folder      (N,)  str               original timestamped folder
and as attributes: fs, f_kHz, channels (66, 2), excitation (n,), the excitation
signal copied as stored in the first measurement file; its length n depends on
the frequency (5001 points at 40 kHz, 2001 at 100 kHz, 1667 at 120 kHz).
No preprocessing here: decimation and normalization are done downstream (ogw.data).
"""
import csv
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

F = int(sys.argv[1]) if len(sys.argv) > 1 else 120
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / f"ogw_f{F}kHz.h5"

rows = list(csv.DictReader(open(ROOT / "metadata" / "ogw_index.csv")))
by = defaultdict(list)
for r in rows:
    by[r["label"]].append(r)
order = ["udam", "dam_D04", "dam_D12", "dam_D16", "dam_D24"]
meas = []
for lab in order:
    v = sorted(by[lab], key=lambda r: float(r["timestamp"]))
    for i, r in enumerate(v):
        r["cycle"] = 1 + i // 161 if lab == "udam" else 1
        meas.append(r)
N = len(meas)

zips = {lab: zipfile.ZipFile(RAW / f"OGW_CFRP_Temperature_{lab}.zip") for lab in order}
with h5py.File(OUT, "w") as out:
    catch = out.create_dataset("catch", (N, 66, 13108), dtype="f4", chunks=(1, 66, 13108))
    for i, r in enumerate(meas):
        name = f"OGW_CFRP_Temperature_{r['label']}/{r['folder']}/pc_f{F}kHz.h5"
        with zips[r["label"]].open(name) as fh, h5py.File(fh, "r") as h:
            catch[i] = h["pitchcatch/catch"][()].astype("f4")
            if i == 0:
                out.attrs["fs"] = float(h["command/pitchcatch/sampling_frequency"][()][0])
                out.attrs["channels"] = h["command/pitchcatch/channels"][()]
                out.attrs["excitation"] = h["command/pitchcatch/signal_data"][()].ravel()
    out.attrs["f_kHz"] = F
    s = h5py.string_dtype()
    out["label"] = np.array([r["label"] for r in meas], dtype=object).astype(s)
    out["folder"] = np.array([r["folder"] for r in meas], dtype=object).astype(s)
    out["damaged"] = np.array([int(r["damaged"]) for r in meas], dtype="i1")
    out["cycle"] = np.array([r["cycle"] for r in meas], dtype="i1")
    for k in ["T_ctc", "T_meas_1", "T_meas_2", "humidity", "timestamp"]:
        out[k] = np.array([float(r[k]) for r in meas])
print("written:", OUT, N, "measurements")
