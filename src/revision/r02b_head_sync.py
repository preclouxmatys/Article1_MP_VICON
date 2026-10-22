"""r02b_head_sync.py : decalage MP <-> Vicon estime sur la TETE (vitesse du nez
MediaPipe vs vitesse du milieu des tempes Vicon), independant des bras. Sert a
synchroniser les analyses des bras ; le decalage 'poignets' (r02) sert pour la tete."""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
ROOT = Path(__file__).resolve().parents[2]
ARR = ROOT / "data/intermediate/revision/arrays"
SWAP = {("D14", "SEMI"): {"P1": "P2", "P2": "P1"}}
def fill(x): return pd.DataFrame(x).interpolate(limit_direction="both").to_numpy()
def lp(x, fs): b, a = butter(4, 6 / (fs / 2)); return filtfilt(b, a, x, axis=0)
def spd(p, fs): v = np.linalg.norm(np.diff(p, axis=0), axis=1) * fs; return np.r_[v, v[-1]]
def xc(a, b, L):
    n = min(len(a), len(b)); a = a[:n] - a[:n].mean(); b = b[:n] - b[:n].mean()
    nf = 1 << int(np.ceil(np.log2(2 * n))); cc = np.fft.irfft(np.fft.rfft(a, nf) * np.conj(np.fft.rfft(b, nf)), nf)
    lags = np.r_[np.arange(0, L + 1), np.arange(-L, 0)]; v = np.r_[cc[:L + 1], cc[-L:]] / (n - np.abs(lags)) / (a.std() * b.std())
    k = int(np.argmax(v)); return float(v[k]), int(lags[k])
rows = []
for f in sorted(ARR.glob("*.npz")):
    dy, co, pi = f.stem.split("_"); dv = np.load(f)
    dm = np.load(ARR / f"{dy}_{co}_{SWAP.get((dy, co), {}).get(pi, pi)}.npz")
    fps = float(dm["fps"]); nv = int(dm["n_video"]); names = list(dm["MP_names"])
    nose = dm["MP"][:, names.index("NOSE"), :2].copy(); nose[:, 0] *= float(dm["W"]); nose[:, 1] *= float(dm["H"])
    sm = spd(lp(fill(nose), fps), fps)
    tm = (fill(dv["V_Tempe_D"]) + fill(dv["V_Tempe_G"])) / 2
    tv = np.arange(int(dv["n_vicon"])) / 100.0; t = np.arange(nv) / fps
    sv = np.interp(t, tv, spd(lp(tm, 100.0), 100.0), right=np.nan)
    pk, lg = xc(np.nan_to_num(sv), sm, int(20 * fps))
    rows.append(dict(dyad=dy, condition=co, pid=pi, head_lag_frames=lg, head_lag_s=lg / fps, head_peak=pk))
pd.DataFrame(rows).to_csv(ROOT / "data/intermediate/revision/r02b_head_sync.csv", index=False)
