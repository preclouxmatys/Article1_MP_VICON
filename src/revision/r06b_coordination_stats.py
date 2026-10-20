"""r06b : statistiques de l'analyse de coordination (accord inter-systemes,
dyades reelles vs pseudo-dyades)."""
from pathlib import Path
import numpy as np, pandas as pd, pingouin as pg
from scipy import stats
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "data/final/revision"
d = pd.read_csv(OUT / "coordination_indices.csv")
real, pseudo = d[d.kind == "real"], d[d.kind == "pseudo"]
rows = []
for sig in ["wrist", "head"]:
    for ref in ["VIC", "V3D"]:
        x, y = real[f"{sig}_MP"].to_numpy(), real[f"{sig}_{ref}"].to_numpy(); n = len(x)
        icc = pg.intraclass_corr(data=pd.DataFrame({"t": np.r_[np.arange(n), np.arange(n)], "r": ["a"] * n + ["b"] * n,
                                 "s": np.r_[x, y]}), targets="t", raters="r", ratings="s").set_index("Type").loc["ICC(A,1)"]
        rows.append(dict(signal=sig, comparison=f"MP vs {ref}", n=n, r=stats.pearsonr(x, y)[0],
                         ICC=icc["ICC"], ICC_lo=icc["CI95"][0], ICC_hi=icc["CI95"][1], bias=np.mean(x - y)))
agree = pd.DataFrame(rows); agree.to_csv(OUT / "T_coordination_agreement.csv", index=False)
rows = []
for sig in ["wrist", "head"]:
    for sysn in ["MP", "VIC", "V3D"]:
        a, b = real[f"{sig}_{sysn}"], pseudo[f"{sig}_{sysn}"]
        # test par permutation des etiquettes reel/pseudo (les pseudo-dyades ne sont pas independantes)
        obs = a.mean() - b.mean(); allv = np.r_[a, b]; na = len(a); rng = np.random.default_rng(1); c = 0
        for _ in range(5000):
            p = rng.permutation(allv); c += (p[:na].mean() - p[na:].mean()) >= obs
        sd_p = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append(dict(signal=sig, system=sysn, real_mean=a.mean(), real_sd=a.std(ddof=1), pseudo_mean=b.mean(),
                         pseudo_sd=b.std(ddof=1), diff=obs, cohen_d=obs / sd_p, p_perm=(c + 1) / 5001))
rp = pd.DataFrame(rows); rp.to_csv(OUT / "T_coordination_real_vs_pseudo.csv", index=False)
print(agree.round(3).to_string()); print(rp.round(3).to_string())
