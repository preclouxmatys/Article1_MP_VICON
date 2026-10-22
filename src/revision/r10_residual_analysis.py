"""r10_residual_analysis.py : analyse residuelle de Winter (2009) pour choisir la
frequence de coupure du filtre passe-bas (Vicon 100 Hz ; MediaPipe en pixels)."""
from pathlib import Path
import json, numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
ROOT = Path(__file__).resolve().parents[2]; ARR = ROOT / "data/intermediate/revision/arrays"
FC = np.arange(1, 12.01, 0.5)  # < Nyquist a 25 i/s
def fill(x): return pd.DataFrame(x).interpolate(limit_direction="both").to_numpy()
def winter(x, fs):
    res = []
    for fc in FC:
        b, a = butter(4, fc / (fs / 2)); y = filtfilt(b, a, x, axis=0)
        res.append(np.sqrt(np.mean((x - y) ** 2)))
    res = np.array(res); hi = FC >= 0.6 * FC.max()        # zone de bruit : regression lineaire
    p = np.polyfit(FC[hi], res[hi], 1); a0 = p[1]              # intercept = residu du bruit
    k = np.argmax(res <= a0); return float(FC[k])
out = []
for f in sorted(ARR.glob("*.npz")):
    d = np.load(f); names = list(d["MP_names"]); fps = float(d["fps"])
    row = {"rec": f.stem}
    for m in ["poignet_D", "poignet_G", "Tempe_D"]:
        x = fill(d[f"V_{m}"]); row[f"V_{m}"] = np.median([winter(x[:, j], 100.0) for j in range(3)])
    for nm in ["RIGHT_WRIST", "LEFT_WRIST", "NOSE"]:
        a = d["MP"][:, names.index(nm), :2].copy(); a[:, 0] *= float(d["W"]); a[:, 1] *= float(d["H"])
        a = fill(a); row[f"MP_{nm}"] = np.median([winter(a[:, j], fps) for j in range(2)])
    out.append(row)
r = pd.DataFrame(out); r.to_csv(ROOT / "data/final/revision/T_residual_analysis.csv", index=False)
print(r.drop(columns="rec").describe().loc[["50%", "25%", "75%"]].round(1).to_string())
