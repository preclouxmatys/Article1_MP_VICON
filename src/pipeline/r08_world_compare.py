"""r08_world_compare.py: MediaPipe world landmarks (3D, metres, origin at the
hip centre) vs Vicon 3D, on the 20-recording subsample.
Variables directly comparable in 3D and in metric units:
  - 3D elbow angle (arm near / far from the camera)
  - wrist amplitude P95-P5 (wrist to shoulder-centre distance, mm)
  - wrist QoM relative to the shoulder centre (mm; the global translation of
    the body is absent from the world landmarks by construction)
Run from src/pipeline (imports r04_process)."""
from pathlib import Path
import json, numpy as np, pandas as pd
import r04_process as R
ROOT = Path(__file__).resolve().parents[2]; REV = ROOT / "data/intermediate/pipeline"; OUT = ROOT / "data/results"
NAMES = [l.name for l in __import__("mediapipe").solutions.pose.PoseLandmark]
rows = []
for f in sorted((REV / "world").glob("D??_*_P?.npz")):
    rec = f.stem; dy, co, pi = rec.split("_")
    js = json.load(open(REV / f"per_recording/{rec}.json"))
    Wd = np.load(f)["W"]; fps, lag = js["fps"], js["lag_s"] * js["fps"]
    dv = np.load(REV / f"arrays/{rec}.npz")
    V = {k[2:]: dv[k].astype(float).copy() for k in dv.files if k.startswith("V_")}
    R.clean_segments(V)
    tv = np.arange(int(dv["n_vicon"])) / 100.0; t = (np.arange(len(Wd)) + lag) / fps
    def vic(k):
        a = R.lowpass(R.interp_nan(V[k], maxgap=200), 100.0)
        return np.column_stack([np.interp(t, tv, a[:, j], left=np.nan, right=np.nan) for j in range(3)])
    def mpw(nm):
        a = Wd[:, NAMES.index(nm), :3] * 1000.0
        return R.lowpass(R.interp_nan(a, maxgap=5), fps)
    out = dict(dyad=dy, condition=co, pid=pi, near_side=js["near_side"])
    for s, side in [("D", "RIGHT"), ("G", "LEFT")]:
        am = R.angle(mpw(f"{side}_SHOULDER"), mpw(f"{side}_ELBOW"), mpw(f"{side}_WRIST"))
        av = R.angle(vic(f"epaule_{s}"), vic(f"coude_{s}"), vic(f"poignet_{s}"))
        g = R.agree(am, av); lab = "near" if s == js["near_side"] else "far"
        for k in ["r", "rmse", "bias", "sdd"]:
            out[f"ELB3D_{lab}_{k}"] = g[k]
            out[f"ELB2Dpx_{lab}_{k}"] = js[f"ELB_{s}_MP_vs_V3D_{k}"]
    Cm = (mpw("RIGHT_SHOULDER") + mpw("LEFT_SHOULDER")) / 2; Cv = (vic("epaule_D") + vic("epaule_G")) / 2
    out["AMP_world_mm"] = sum(R.amp_robust(np.linalg.norm(mpw(f"{w}_WRIST") - Cm, axis=1))[0] for w in ["RIGHT", "LEFT"])
    out["AMP_V3D_mm"] = sum(R.amp_robust(np.linalg.norm(vic(f"poignet_{s}") - Cv, axis=1))[0] for s in "DG")
    Hv = Cv  # shoulder centre (no hip markers in most Vicon recordings)
    out["QOM_world_mm"] = sum(R.path_length(mpw(f"{w}_WRIST") - Cm) for w in ["RIGHT", "LEFT"])
    out["QOM_V3D_relhip_mm"] = sum(R.path_length(vic(f"poignet_{s}") - Hv) for s in "DG")
    out["QOM_MP_px"] = js["QOM_WRIST_MP_px"]; out["QOM_VIC_px"] = js["QOM_WRIST_VIC_px"]
    out["sw_world_mm"] = float(np.nanmedian(np.linalg.norm(mpw("RIGHT_SHOULDER") - mpw("LEFT_SHOULDER"), axis=1)))
    out["sw_V3D_mm"] = js["sw3_med_mm"]
    rows.append(out); print(rec, round(out["ELB3D_near_r"], 2), round(out["ELB3D_far_r"], 2), flush=True)
d = pd.DataFrame(rows); d.to_csv(OUT / "world_landmarks_subset.csv", index=False)
fz = lambda v: float(np.tanh(np.nanmean(np.arctanh(np.clip(v, -.999, .999)))))
summ = []
for arm in ["near", "far"]:
    for ref in ["ELB3D", "ELB2Dpx"]:
        rr = d[f"{ref}_{arm}_r"].to_numpy(); rm = d[f"{ref}_{arm}_rmse"].to_numpy()
        rng = np.random.default_rng(42); bs = [rng.integers(0, len(d), len(d)) for _ in range(2000)]
        rb = [fz(rr[i]) for i in bs]; mb = [np.nanmedian(rm[i]) for i in bs]
        summ.append(dict(variable=f"coude {arm} {ref}", n=len(d), r_fisher=fz(rr), r_lo=np.percentile(rb, 2.5),
                         r_hi=np.percentile(rb, 97.5), rmse_median=np.nanmedian(rm), rmse_lo=np.percentile(mb, 2.5),
                         rmse_hi=np.percentile(mb, 97.5), bias_mean=d[f"{ref}_{arm}_bias"].mean(),
                         sdd_mean=d[f"{ref}_{arm}_sdd"].mean()))
def icc21(x, y):
    Y = np.c_[x, y]; n, k = Y.shape; gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)
from scipy import stats
for a, b, lab in [("AMP_world_mm", "AMP_V3D_mm", "amplitude mm"), ("QOM_world_mm", "QOM_V3D_relhip_mm", "QoM rel. centre epaules mm"),
                  ("QOM_MP_px", "QOM_VIC_px", "QoM px (meme sous-echantillon)"), ("sw_world_mm", "sw_V3D_mm", "largeur epaule mm")]:
    x, y = d[a].to_numpy(), d[b].to_numpy()
    summ.append(dict(variable=lab, n=len(d), r=stats.pearsonr(x, y)[0], ICC=icc21(x, y),
                     bias_pct=100 * np.mean(x - y) / np.mean(y), mean_MP=x.mean(), mean_V=y.mean()))
pd.DataFrame(summ).to_csv(OUT / "T_world_landmarks.csv", index=False)
print(pd.DataFrame(summ).round(3).to_string())
