"""r03_match_check.py : pour les enregistrements dont la synchro echoue, cherche
quel enregistrement Vicon (meme dyade, toutes conditions, P1/P2) correspond
le mieux a la video MediaPipe (correlation croisee des vitesses de poignet)."""
from pathlib import Path
import numpy as np, pandas as pd, sys
from scipy.signal import butter, filtfilt
ROOT = Path(__file__).resolve().parents[2]
ARR = ROOT / "data/intermediate/revision/arrays"
def fill(x): return pd.DataFrame(x).interpolate(limit_direction="both").to_numpy()
def lp(x, fs): b, a = butter(4, 6/(fs/2)); return filtfilt(b, a, x, axis=0)
def spd(p, fs): v = np.linalg.norm(np.diff(p, axis=0), axis=1)*fs; return np.r_[v, v[-1]]
def vic_speed(d, fps, n):
    t = np.arange(n)/fps; tv = np.arange(int(d["n_vicon"]))/100
    return sum(np.interp(t, tv, spd(lp(fill(d[f"V_poignet_{s}"]),100),100)) for s in "DG")
def mp_speed(d):
    names=list(d["MP_names"]); fps=float(d["fps"]); out=0
    for nm in ["RIGHT_WRIST","LEFT_WRIST"]:
        xy=d["MP"][:,names.index(nm),:2].copy(); xy[:,0]*=float(d["W"]); xy[:,1]*=float(d["H"])
        out = out + spd(lp(fill(xy),fps),fps)
    return out
def xc(a,b,L):
    n=min(len(a),len(b)); a=a[:n]-a[:n].mean(); b=b[:n]-b[:n].mean()
    nf=1<<int(np.ceil(np.log2(2*n))); cc=np.fft.irfft(np.fft.rfft(a,nf)*np.conj(np.fft.rfft(b,nf)),nf)
    lags=np.r_[np.arange(0,L+1),np.arange(-L,0)]; v=np.r_[cc[:L+1],cc[-L:]]/(n-np.abs(lags))/(a.std()*b.std())
    k=np.argmax(v); return v[k], lags[k]
targets = [a for a in sys.argv[1:] if not a.startswith("--")]
for t in targets:
    dyad = t.split("_")[0]
    dm = np.load(ARR/f"{t}.npz"); fps=float(dm["fps"]); n=int(dm["n_video"]); m=mp_speed(dm)
    res=[]
    for f in sorted(ARR.glob(("*" if "--all" in sys.argv else dyad)+"_*.npz")):
        dv=np.load(f); c,l=xc(vic_speed(dv,fps,n), m, int(60*fps)); res.append((round(c,2), round(l/fps,2), f.stem))
    res.sort(reverse=True); print(t, "->", res[:3], flush=True)
