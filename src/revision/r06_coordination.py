"""
r06_coordination.py : validite au niveau du construit interactionnel.
Pour chaque dyade x condition, indice de coordination interpersonnelle par
correlation croisee fenetree (WCC ; fenetres de 5 s, pas de 1 s, decalages
+/- 2 s ; indice = moyenne, en z de Fisher, du pic de correlation par fenetre)
entre les vitesses des deux partenaires, calcule avec chaque systeme :
  MP (pixels, video de chaque partenaire), VIC (Vicon projete dans chaque
  video, pixels), V3D (Vicon 3D, mm). Signaux : poignets (G+D) et tete.
Pseudo-dyades (P1 d'une dyade x P2 d'une autre, meme condition) = reference
de hasard. Sorties : data/final/revision/T_coordination*.csv
"""
from pathlib import Path
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as swv

ROOT = Path(__file__).resolve().parents[2]
TS = ROOT / "data/intermediate/revision/timeseries"
OUT = ROOT / "data/final/revision"
FS = 25.0; WIN = int(5 * FS); STEP = int(1 * FS); LAG = int(2 * FS)
RNG = np.random.default_rng(42)
SIG = {"wrist": {"MP": "wrist_mp", "VIC": "wrist_vic", "V3D": "wrist_v3"},
       "head": {"MP": "head_mp", "VIC": "headtm_vic", "V3D": "headtm_v3"}}


def load(dyad, cond, pid):
    d = np.load(TS / f"{dyad}_{cond}_{pid}.npz")
    return {k: d[k] for k in d.files}


def on_grid(d, key, grid):
    v = d[key] * float(d["fps"])  # pas par image -> vitesse par seconde
    t = d["t"]; m = np.isfinite(v)
    return np.interp(grid, t[m], v[m], left=np.nan, right=np.nan)


def wcc(x, y):
    n = min(len(x), len(y)); x, y = x[:n], y[:n]
    ok = np.isfinite(x) & np.isfinite(y)
    x = np.where(ok, x, np.nanmean(x)); y = np.where(ok, y, np.nanmean(y))
    peaks = []
    for s in range(LAG, n - WIN - LAG, STEP):
        xs = x[s:s + WIN]
        Y = swv(y[s - LAG:s + WIN + LAG], WIN)  # (2*LAG+1, WIN)
        xz = (xs - xs.mean()) / (xs.std() + 1e-12)
        Yz = (Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-12)
        r = Yz @ xz / WIN
        peaks.append(np.clip(r.max(), -.999, .999))
    return float(np.tanh(np.mean(np.arctanh(peaks))))


def pair_index(a, b):
    t0 = max(a["t"][0], b["t"][0]); t1 = min(a["t"][-1], b["t"][-1])
    grid = np.arange(t0, t1, 1 / FS)
    out = {}
    for sig, sysk in SIG.items():
        for sysn, key in sysk.items():
            out[f"{sig}_{sysn}"] = wcc(on_grid(a, key, grid), on_grid(b, key, grid))
    return out


recs = sorted({tuple(p.stem.split("_")[:2]) for p in TS.glob("*.npz")})
real = []
cache = {}
for dyad, cond in recs:
    a, b = load(dyad, cond, "P1"), load(dyad, cond, "P2")
    cache[(dyad, cond)] = (a, b)
    real.append(dict(dyad=dyad, condition=cond, kind="real", **pair_index(a, b)))
    print(dyad, cond, {k: round(v, 3) for k, v in real[-1].items() if isinstance(v, float)}, flush=True)
pseudo = []
for cond in ["SEATED", "SEMI", "STANDING"]:
    dy = sorted(d for d, c in recs if c == cond)
    for d1 in dy:
        others = [d for d in dy if d != d1]
        for d2 in RNG.choice(others, 5, replace=False):
            pseudo.append(dict(dyad=f"{d1}x{d2}", condition=cond, kind="pseudo",
                               **pair_index(cache[(d1, cond)][0], cache[(d2, cond)][1])))
res = pd.DataFrame(real + pseudo)
res.to_csv(OUT / "coordination_indices.csv", index=False)
