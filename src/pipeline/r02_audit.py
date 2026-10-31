"""
r02_audit.py: quality check of the raw data, per recording.
  - Vicon: % of missing samples and longest gap per arm marker; aberrant
    speeds (jumps / label swaps); wrist-to-shoulder-centre distance outside
    the physiological range.
  - Synchronisation: MediaPipe <-> Vicon time lag estimated by cross-correlation
    of wrist speeds (left + right), +/- 20 s. This wrist-based lag is used for
    the head time-series analyses.
  - Laterality: correlation of MediaPipe RIGHT_WRIST speed with Vicon right vs left wrist.
Output: data/intermediate/pipeline/r02_audit.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[2]
ARR = ROOT / "data/intermediate/pipeline/arrays"
OUT = ROOT / "data/intermediate/pipeline/r02_audit.csv"


def fill(x):
    s = pd.DataFrame(x).interpolate(limit_direction="both")
    return s.to_numpy()


def lp(x, fs, fc=6.0):
    b, a = butter(4, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)


def maxgap(m):
    run = best = 0
    for v in m:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def mp_xy(d, name):
    names = list(d["MP_names"]); j = names.index(name)
    xy = d["MP"][:, j, :2].copy()
    xy[:, 0] *= float(d["W"]); xy[:, 1] *= float(d["H"])
    return xy


def speed(p, fs):
    v = np.linalg.norm(np.diff(p, axis=0), axis=1) * fs
    return np.r_[v, v[-1]]


import sys
rows = []
for f in sorted(ARR.glob("*.npz")):
    d = np.load(f)
    dyad, cond, pid = f.stem.split("_")
    r = {"dyad": dyad, "condition": cond, "pid": pid, "W": int(d["W"]), "H": int(d["H"]),
         "fps": float(d["fps"]), "n_vicon": int(d["n_vicon"]), "n_video": int(d["n_video"]),
         "mp_detect_pct": 100 * float(d["n_detected"]) / int(d["n_video"])}
    for m in ["epaule_D", "epaule_G", "coude_D", "coude_G", "poignet_D", "poignet_G", "Tempe_D", "Tempe_G", "AR_Tete_D", "AR_Tete_G"]:
        nan = np.isnan(d[f"V_{m}"][:, 0])
        r[f"nanpct_{m}"] = 100 * nan.mean(); r[f"maxgap_{m}"] = maxgap(nan)
    V = {m: fill(d[f"V_{m}"]) for m in ["epaule_D", "epaule_G", "poignet_D", "poignet_G", "coude_D", "coude_G"]}
    C = (V["epaule_D"] + V["epaule_G"]) / 2
    for s in "DG":
        dist = np.linalg.norm(V[f"poignet_{s}"] - C, axis=1)
        r[f"dist_{s}_p1"], r[f"dist_{s}_p99"], r[f"dist_{s}_max"] = np.nanpercentile(dist, 1), np.nanpercentile(dist, 99), np.nanmax(dist)
        r[f"dist_{s}_pct_gt900"] = 100 * np.mean(dist > 900)
        vraw = speed(V[f"poignet_{s}"], 100.0)
        r[f"wrist_{s}_frames_gt5ms"] = int(np.sum(vraw > 5000))
        ua = np.linalg.norm(V[f"epaule_{s}"] - V[f"coude_{s}"], axis=1)
        r[f"upperarm_{s}_med"], r[f"upperarm_{s}_iqr"] = np.median(ua), np.subtract(*np.percentile(ua, [75, 25]))
    sw = np.linalg.norm(V["epaule_D"] - V["epaule_G"], axis=1)
    r["sw_med"], r["sw_p5"], r["sw_p95"] = np.median(sw), np.percentile(sw, 5), np.percentile(sw, 95)
    # --- synchronisation: Vicon wrist speed resampled at the video frame times
    fps = float(d["fps"]); nv = int(d["n_video"])
    t_vid = np.arange(nv) / fps
    t_vic = np.arange(int(d["n_vicon"])) / 100.0
    sp_v = {s: np.interp(t_vid, t_vic, speed(lp(V[f"poignet_{s}"], 100.0), 100.0), right=np.nan) for s in "DG"}
    sp_m = {}
    for s, nm in [("D", "RIGHT_WRIST"), ("G", "LEFT_WRIST")]:
        xy = fill(mp_xy(d, nm)); sp_m[s] = speed(lp(xy, fps), fps)
    a = np.nan_to_num(sp_v["D"] + sp_v["G"]); b = sp_m["D"] + sp_m["G"]
    n = min(len(a), len(b)); a, b = a[:n] - a[:n].mean(), b[:n] - b[:n].mean()
    L = int(20 * fps)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    cc = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    lags = np.r_[np.arange(0, L + 1), np.arange(-L, 0)]
    vals = np.r_[cc[:L + 1], cc[-L:]]
    ov = n - np.abs(lags)
    vals = vals / ov / (a.std() * b.std())
    k = int(np.argmax(vals)); best = (float(vals[k]), int(lags[k]))
    c0 = np.corrcoef(a, b)[0, 1]
    r["xcorr_lag_frames"], r["xcorr_lag_s"], r["xcorr_peak"], r["xcorr_at0"] = best[1], best[1] / fps, best[0], c0
    # --- laterality
    for sm in "DG":
        for sv in "DG":
            x, y = sp_m[sm][:n], np.nan_to_num(sp_v[sv][:n])
            r[f"lat_MP{sm}_V{sv}"] = np.corrcoef(x, y)[0, 1]
    rows.append(r); print(f.stem, f"lag={best[1]/fps:+.2f}s peak={best[0]:.2f} at0={c0:.2f}", flush=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
