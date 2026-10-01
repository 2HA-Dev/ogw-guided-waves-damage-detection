"""Index of the OGW dataset no. 2 measurements (infrastructure tool).

Reads, without extraction, one .h5 file per measurement folder (the zips are
stored uncompressed) and writes metadata/ogw_index.csv: class, folder,
timestamp, temperature (setpoint and measured), humidity.
"""
import csv
import zipfile
from pathlib import Path

import h5py

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "metadata" / "ogw_index.csv"
PROBE_FREQ = "pc_f100kHz.h5"


def scalar(ds):
    v = ds[()]
    return [float(x) for x in v.ravel()] if hasattr(v, "ravel") else [float(v)]


rows = []
for zpath in sorted(RAW.glob("OGW_CFRP_Temperature_*.zip")):
    label = zpath.stem.replace("OGW_CFRP_Temperature_", "")  # udam, dam_D04...
    with zipfile.ZipFile(zpath) as z:
        members = sorted(n for n in z.namelist() if n.endswith("/" + PROBE_FREQ))
        for name in members:
            folder = name.split("/")[1]
            with z.open(name) as fh, h5py.File(fh, "r") as f:
                temp_ctc = scalar(f["CTC/Temperature"])[0]
                hum = scalar(f["CTC/Humidity"])[0]
                temps = scalar(f["Temperature/values"])
                ts = scalar(f["timestamp"])[0]
            rows.append({
                "label": label,
                "damaged": int(label != "udam"),
                "folder": folder,
                "timestamp": ts,
                "T_ctc": round(temp_ctc, 3),
                "T_meas_1": round(temps[0], 3),
                "T_meas_2": round(temps[1], 3) if len(temps) > 1 else "",
                "humidity": round(hum, 3),
            })
    print(zpath.name, len(members), "measurements")

OUT.parent.mkdir(exist_ok=True)
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("written:", OUT, len(rows), "rows")
