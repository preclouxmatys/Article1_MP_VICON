"""r01b_extra_markers.py : extrait les marqueurs d'orteils (orteil_D/G) de chaque CSV
Vicon, absents des tableaux r01, pour enrichir la calibration des cameras."""
from pathlib import Path
import sys, numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from r01_prepare_arrays import read_vicon, vicon_marker, CONDS
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/intermediate/revision/arrays_extra"; OUT.mkdir(parents=True, exist_ok=True)
for vpath in sorted((ROOT / "data/raw/vicon_csv").glob("*.csv")):
    cond = next(c for c in CONDS if vpath.stem.upper().startswith(c)); dyad = vpath.stem[len(cond):]
    if all((OUT / f"{dyad}_{cond}_{p}.npz").exists() for p in ["P1", "P2"]):
        continue
    dv = read_vicon(vpath)
    for pid, pre in [("P1", ""), ("P2", "2")]:
        ex = {}
        for m in ["orteil_D", "orteil_G"]:
            a = vicon_marker(dv, pre + m)
            ex[f"V_{m}"] = a if a is not None else np.full((len(dv), 3), np.nan)
        np.savez_compressed(OUT / f"{dyad}_{cond}_{pid}.npz", **ex)
    print(vpath.stem, flush=True)
