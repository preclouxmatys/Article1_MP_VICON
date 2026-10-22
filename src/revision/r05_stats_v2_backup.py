"""
r05_stats.py : statistiques de validite (pipeline de revision).
Entree : data/intermediate/revision/r04_all_recordings.csv (une ligne par enregistrement)
Sorties : data/final/revision/*.csv (tableaux) ; chiffres cles en JSON.
"""
from pathlib import Path
import json, glob
import numpy as np, pandas as pd
import pingouin as pg
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "data/intermediate/revision"
OUT = ROOT / "data/final/revision"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(42); NBOOT = 5000

df = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob(str(REV / "per_recording/*.json")))])
df.to_csv(REV / "r04_all_recordings.csv", index=False)
df["subject"] = df.dyad + "_" + df.pid


def icc21(x, y):
    Y = np.c_[x, y]; n, k = Y.shape; gm = Y.mean()
    msr = k * ((Y.mean(1) - gm) ** 2).sum() / (n - 1); msc = n * ((Y.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((Y - Y.mean(1, keepdims=True) - Y.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def validity(d, mp, ref, label, boot=True):
    d = d[[mp, ref, "dyad", "condition", "subject"]].dropna()
    x, y = d[mp].to_numpy(), d[ref].to_numpy()
    n = len(d); r = np.corrcoef(x, y)[0, 1]
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3)
    icc_tab = pg.intraclass_corr(data=pd.DataFrame({"t": np.r_[np.arange(n), np.arange(n)],
                                 "r": ["MP"] * n + ["V"] * n, "s": np.r_[x, y]}), targets="t", raters="r", ratings="s")
    row2 = icc_tab.set_index("Type").loc["ICC(A,1)"]
    dd = x - y; mref = y.mean()
    lr = np.log(x / y) if (x > 0).all() and (y > 0).all() else np.full(n, np.nan)
    mean_s = (x + y) / 2
    slope = np.polyfit(mean_s, dd, 1)[0]
    from scipy import stats
    sl = stats.linregress(mean_s, dd)
    out = dict(label=label, n=n, mean_MP=x.mean(), sd_MP=x.std(ddof=1), mean_ref=mref, sd_ref=y.std(ddof=1),
               r=r, r_lo=np.tanh(z - 1.96 * se), r_hi=np.tanh(z + 1.96 * se), R2=r ** 2,
               ICC=row2["ICC"], ICC_lo=row2["CI95"][0], ICC_hi=row2["CI95"][1],
               RMSE=np.sqrt(np.mean(dd ** 2)), bias=dd.mean(), bias_pct=100 * dd.mean() / mref,
               LoA_lo=dd.mean() - 1.96 * dd.std(ddof=1), LoA_hi=dd.mean() + 1.96 * dd.std(ddof=1),
               BA_slope=sl.slope, BA_slope_p=sl.pvalue,
               ratio_gmean=np.exp(np.nanmean(lr)), ratio_LoA_lo=np.exp(np.nanmean(lr) - 1.96 * np.nanstd(lr, ddof=1)),
               ratio_LoA_hi=np.exp(np.nanmean(lr) + 1.96 * np.nanstd(lr, ddof=1)))
    if boot:
        dy = d.dyad.unique(); idx = {k: np.where(d.dyad.values == k)[0] for k in dy}
        rb, ib = [], []
        for _ in range(NBOOT):
            s = np.concatenate([idx[k] for k in RNG.choice(dy, len(dy))])
            rb.append(np.corrcoef(x[s], y[s])[0, 1]); ib.append(icc21(x[s], y[s]))
        out.update(r_boot_lo=np.percentile(rb, 2.5), r_boot_hi=np.percentile(rb, 97.5),
                   ICC_boot_lo=np.percentile(ib, 2.5), ICC_boot_hi=np.percentile(ib, 97.5))
    return out


def by_condition(d, mp, ref, label):
    rows = []
    for c, g in d.groupby("condition"):
        v = validity(g, mp, ref, label, boot=False)
        rows.append(dict(label=label, condition=c, n=v["n"], r=v["r"], ICC=v["ICC"], ICC_lo=v["ICC_lo"],
                         ICC_hi=v["ICC_hi"], bias_pct=v["bias_pct"]))
    rm = pg.rm_corr(data=d.dropna(subset=[mp, ref]), x=ref, y=mp, subject="subject")
    m = smf.mixedlm(f"{mp} ~ {ref} + C(condition)", d.dropna(subset=[mp, ref]), groups="dyad",
                    re_formula="1", vc_formula={"subject": "0 + C(subject)"}).fit(reml=True)
    return rows, dict(label=label, rmcorr_r=rm["r"].iloc[0], rmcorr_lo=rm["CI95"].iloc[0][0],
                      rmcorr_hi=rm["CI95"].iloc[0][1], rmcorr_p=rm["pval"].iloc[0], rmcorr_dof=rm["dof"].iloc[0],
                      mixed_slope=m.params[ref], mixed_slope_lo=m.conf_int().loc[ref, 0],
                      mixed_slope_hi=m.conf_int().loc[ref, 1], mixed_p=m.pvalues[ref])


MAIN = [  # (colonne MP, colonne reference, etiquette)
    ("QOM_WRIST_MP_px", "QOM_WRIST_VIC_px", "QoM poignet (px, plan image)"),
    ("QOM_WRIST_MP_norm", "QOM_WRIST_V3D25_norm", "QoM poignet (largeurs d'epaule, Vicon 3D)"),
    ("QOM_HEAD_MP_px", "QOM_HEAD_VIC_px", "QoM tete : nez MP vs nez virtuel (px)"),
    ("QOM_HEAD_MP_px", "QOM_HEAD_tempmid_VIC_px", "QoM tete : nez MP vs milieu des tempes (px)"),
    ("AMP_p95p5_MP_px", "AMP_p95p5_VIC_px", "Amplitude poignet P95-P5 (px, plan image)"),
]
SENS = [
    ("QOM_WRIST_MPnormcoord_norm", "QOM_WRIST_V3D100_norm", "Poignet - definition d'origine (coord. normalisees)"),
    ("QOM_WRIST_MP_norm", "QOM_WRIST_V3D100_norm", "Poignet - pixels vs Vicon 100 Hz"),
    ("QOM_WRIST_MP_norm", "QOM_WRIST_VIC_norm", "Poignet - pixels, Vicon projete, normalises"),
    ("QOM_HEAD_MPnormcoord_norm", "QOM_HEAD_tempmid_V3D100_norm", "Tete - definition d'origine (coord. norm., tempes)"),
    ("QOM_HEAD_MP_norm", "QOM_HEAD_tempmid_V3D25_norm", "Tete - pixels vs tempes 3D"),
    ("QOM_HEAD_MP_norm", "QOM_HEAD_V3D25_norm", "Tete - pixels vs nez virtuel 3D"),
    ("AMP_maxmin_MP_px", "AMP_maxmin_VIC_px", "Amplitude max-min (px, plan image)"),
    ("AMP_p95p5_MP_mm_shoulderscale", "AMP_p95p5_V3D_mm", "Amplitude P95-P5 mm (echelle epaule) vs 3D"),
    ("AMP_maxmin_MP_mm_shoulderscale", "AMP_maxmin_V3D_mm", "Amplitude max-min mm (echelle epaule) vs 3D"),
    ("AMP_p95p5_MP_mm_shoulderscale", "AMP_p95p5_VXY_mm", "Amplitude P95-P5 mm vs plan XY"),
    ("AMP_p95p5_MP_mm_shoulderscale", "AMP_p95p5_VXZ_mm", "Amplitude P95-P5 mm vs plan XZ"),
    ("AMP_p95p5_MP_mm_shoulderscale", "AMP_p95p5_VYZ_mm", "Amplitude P95-P5 mm vs plan YZ"),
]
main = pd.DataFrame([validity(df, *a) for a in MAIN]); main.to_csv(OUT / "T_main_validity.csv", index=False)
sens = pd.DataFrame([validity(df, *a, boot=False) for a in SENS]); sens.to_csv(OUT / "T_sensitivity.csv", index=False)
cond_rows, mixed = [], []
for a in MAIN:
    rws, mx = by_condition(df, *a); cond_rows += rws; mixed.append(mx)
pd.DataFrame(cond_rows).to_csv(OUT / "T_by_condition.csv", index=False)
pd.DataFrame(mixed).to_csv(OUT / "T_rmcorr_mixed.csv", index=False)

# descriptives par condition
desc_cols = {"QOM_WRIST_MP_px": "QoM poignet MP (px)", "QOM_WRIST_VIC_px": "QoM poignet Vicon proj. (px)",
             "QOM_WRIST_V3D25_norm": "QoM poignet Vicon 3D (larg. epaule)", "QOM_WRIST_MP_norm": "QoM poignet MP (larg. epaule)",
             "QOM_HEAD_MP_px": "QoM tete MP (px)", "QOM_HEAD_VIC_px": "QoM tete Vicon nez virtuel (px)",
             "QOM_HEAD_tempmid_VIC_px": "QoM tete Vicon tempes (px)",
             "AMP_p95p5_MP_px": "Amplitude MP (px)", "AMP_p95p5_VIC_px": "Amplitude Vicon proj. (px)",
             "AMP_p95p5_V3D_mm": "Amplitude Vicon 3D (mm)", "sw_mp_px_med": "Largeur epaule MP (px)",
             "sw3_med_mm": "Largeur epaule Vicon (mm)", "px_per_mm": "Echelle (px/mm)"}
desc = df.groupby("condition")[list(desc_cols)].agg(["mean", "std"])
alld = df[list(desc_cols)].agg(["mean", "std"]).T.unstack() if False else None
rows = []
for c in desc_cols:
    row = {"variable": desc_cols[c]}
    for cond in ["SEATED", "SEMI", "STANDING"]:
        g = df.loc[df.condition == cond, c]; row[cond] = f"{g.mean():.1f} ({g.std():.1f})"
    row["Total"] = f"{df[c].mean():.1f} ({df[c].std():.1f})"; rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "T_descriptives.csv", index=False)

# ---------------- angle du coude
rows = []
for _, r in df.iterrows():
    for s in "DG":
        base = dict(dyad=r.dyad, condition=r.condition, pid=r.pid, side=s, near=(s == r.near_side),
                    vis_ok_pct=r[f"vis_ok_pct_{s}"])
        for ref in ["V3D", "VIC"]:
            for pre in ["MP", "MPvis"]:
                for k in ["r", "rmse", "bias", "sdd"]:
                    base[f"{pre}_{ref}_{k}"] = r[f"ELB_{s}_{pre}_vs_{ref}_{k}"]
        for k in ["r", "rmse", "bias", "sdd"]:
            base[f"MPnorm_V3D_{k}"] = r[f"ELB_{s}_MPnormcoord_vs_V3D_{k}"]
            base[f"VIC_V3D_{k}"] = r[f"ELB_{s}_VIC_vs_V3D_{k}"]
        for pl in ["XY", "XZ", "YZ"]:
            base[f"MP_{pl}_r"] = r[f"ELB_{s}_MP_vs_{pl}_r"]
        rows.append(base)
el = pd.DataFrame(rows); el.to_csv(OUT / "elbow_per_recording_side.csv", index=False)
fz = lambda v: float(np.tanh(np.nanmean(np.arctanh(np.clip(v, -.999, .999)))))


def elbow_summary(g, pre, ref):
    rr = g[f"{pre}_{ref}_r"].to_numpy(); dy = g.dyad.to_numpy(); u = np.unique(dy)
    idx = {k: np.where(dy == k)[0] for k in u}
    bs = [fz(rr[np.concatenate([idx[k] for k in RNG.choice(u, len(u))])]) for _ in range(2000)]
    rm = g[f"{pre}_{ref}_rmse"]; bi = g[f"{pre}_{ref}_bias"]; sd = g[f"{pre}_{ref}_sdd"]
    return dict(n=len(g), r_fisher=fz(rr), r_boot_lo=np.percentile(bs, 2.5), r_boot_hi=np.percentile(bs, 97.5),
                r_median=np.nanmedian(rr), r_q1=np.nanpercentile(rr, 25), r_q3=np.nanpercentile(rr, 75),
                RMSE_mean=rm.mean(), RMSE_sd=rm.std(), RMSE_median=rm.median(),
                bias_mean=bi.mean(), bias_sd=bi.std(), absbias_mean=bi.abs().mean(), sdd_mean=sd.mean(), sdd_sd=sd.std())


es = []
for lab, g in [("Bras proche", el[el.near]), ("Bras eloigne", el[~el.near]), ("Tous", el)]:
    for pre, ref in [("MP", "V3D"), ("MP", "VIC"), ("MPvis", "V3D"), ("MPnorm", "V3D"), ("VIC", "V3D")]:
        es.append(dict(arm=lab, comparison=f"{pre} vs {ref}", **elbow_summary(g, pre, ref)))
    for pl in ["XY", "XZ", "YZ"]:
        es.append(dict(arm=lab, comparison=f"MP vs plan {pl}", n=len(g), r_fisher=fz(g[f"MP_{pl}_r"]),
                       r_median=g[f"MP_{pl}_r"].median()))
pd.DataFrame(es).to_csv(OUT / "T_elbow.csv", index=False)
ec = []
for (c, nr), g in el.groupby(["condition", "near"]):
    ec.append(dict(condition=c, arm="proche" if nr else "eloigne", **elbow_summary(g, "MP", "V3D")))
pd.DataFrame(ec).to_csv(OUT / "T_elbow_by_condition.csv", index=False)

# ---------------- divers chiffres
from scipy import stats
rel_bias = 100 * (df.QOM_WRIST_MP_px - df.QOM_WRIST_VIC_px) / df.QOM_WRIST_VIC_px
rel_bias_h = 100 * (df.QOM_HEAD_MP_px - df.QOM_HEAD_tempmid_VIC_px) / df.QOM_HEAD_tempmid_VIC_px
key = dict(
    n_rec=len(df), fps_counts=df.fps.round(2).value_counts().to_dict(),
    res_counts=(df.W.astype(int).astype(str) + "x" + df.H.astype(int).astype(str)).value_counts().to_dict(),
    lag_abs_median=df.lag_s.abs().median(), lag_abs_q1=df.lag_s.abs().quantile(.25), lag_abs_q3=df.lag_s.abs().quantile(.75),
    lag_abs_max=df.lag_s.abs().max(), n_lag_gt_1s=int((df.lag_s.abs() > 1).sum()),
    n_lag_gt_04s=int((df.lag_s.abs() > .4).sum()),
    camera_projective=int((df.camera_model == "projective").sum()),
    reproj_calib_median=df.reproj_calib_med_px.median(), reproj_calib_q1=df.reproj_calib_med_px.quantile(.25),
    reproj_calib_q3=df.reproj_calib_med_px.quantile(.75), reproj_calib_max=df.reproj_calib_med_px.max(),
    view_az_abs_median=df.view_azimuth_vs_partner_deg.abs().median(),
    view_az_abs_q1=df.view_azimuth_vs_partner_deg.abs().quantile(.25),
    view_az_abs_q3=df.view_azimuth_vs_partner_deg.abs().quantile(.75),
    view_elev_median=df.view_elevation_deg.median(), view_elev_q1=df.view_elevation_deg.quantile(.25),
    view_elev_q3=df.view_elevation_deg.quantile(.75),
    interpersonal_mm_mean=df.interpersonal_dist_mm.mean(), interpersonal_mm_sd=df.interpersonal_dist_mm.std(),
    sw3_mean=df.sw3_med_mm.mean(), sw3_sd=df.sw3_med_mm.std(),
    px_per_mm_median=df.px_per_mm.median(), px_per_mm_q1=df.px_per_mm.quantile(.25), px_per_mm_q3=df.px_per_mm.quantile(.75),
    mp_detect_pct_min=None,
    ratio_100_25_mean=(df.QOM_WRIST_V3D100_norm / df.QOM_WRIST_V3D25_norm).mean(),
    ratio_100_25_max=(df.QOM_WRIST_V3D100_norm / df.QOM_WRIST_V3D25_norm).max(),
    qomwin_wrist_median=df.QOMwin_WRIST_r.median(), qomwin_wrist_q1=df.QOMwin_WRIST_r.quantile(.25),
    qomwin_wrist_q3=df.QOMwin_WRIST_r.quantile(.75),
    qomwin_head_median=df.QOMwin_HEAD_tempmid_r.median(), qomwin_head_q1=df.QOMwin_HEAD_tempmid_r.quantile(.25),
    qomwin_head_q3=df.QOMwin_HEAD_tempmid_r.quantile(.75),
    qomwin_headnose_median=df.QOMwin_HEAD_r.median(),
    jitter_wrist_median_px=df.MP_jitter_WRIST_px.median(),
    jitter_vs_relbias_r=stats.pearsonr(df.MP_jitter_WRIST_px, rel_bias)[0],
    jitter_vs_relbias_p=stats.pearsonr(df.MP_jitter_WRIST_px, rel_bias)[1],
    jitter_vs_relbias_spearman=stats.spearmanr(df.MP_jitter_WRIST_px, rel_bias)[0],
    jitter_vs_relbias_spearman_p=stats.spearmanr(df.MP_jitter_WRIST_px, rel_bias)[1],
    head_relbias_median=rel_bias_h.median(),
    vicon_labelfix_recordings=int((df.vicon_labelfix_frames > 0).sum()),
    head_all4_pct_median=df.head_all4_pct.median(), head_all4_pct_min=df.head_all4_pct.min(),
    near_side_D=int((df.near_side == "D").sum()),
    holdout_near_wrist_median=np.nanmedian([r[f"reproj_holdout_{'RIGHT' if r.near_side=='D' else 'LEFT'}_WRIST_px"] for _, r in df.iterrows()]),
    holdout_far_wrist_median=np.nanmedian([r[f"reproj_holdout_{'LEFT' if r.near_side=='D' else 'RIGHT'}_WRIST_px"] for _, r in df.iterrows()]),
    holdout_near_elbow_median=np.nanmedian([r[f"reproj_holdout_{'RIGHT' if r.near_side=='D' else 'LEFT'}_ELBOW_px"] for _, r in df.iterrows()]),
    holdout_far_elbow_median=np.nanmedian([r[f"reproj_holdout_{'LEFT' if r.near_side=='D' else 'RIGHT'}_ELBOW_px"] for _, r in df.iterrows()]),
    vis_ok_pct_median=el.vis_ok_pct.median(), vis_ok_pct_far_median=el.loc[~el.near, "vis_ok_pct"].median(),
)
audit = pd.read_csv(REV / "r02_audit.csv")
key["mp_detect_pct_min"] = audit.mp_detect_pct.min(); key["mp_detect_pct_mean"] = audit.mp_detect_pct.mean()
key["vicon_nan_arm_max_pct"] = float(audit.filter(like="nanpct_poignet").max().max())
# permutation (supplementaire) : P1/P2 echanges au sein de chaque dyade x condition
def perm(d, mp, ref):
    x = d[mp].to_numpy(); y = d[ref].to_numpy(); r0 = np.corrcoef(x, y)[0, 1]
    gi = list(d.groupby(["dyad", "condition"]).indices.values()); c = 0
    for _ in range(NBOOT):
        ys = y.copy()
        for g in gi:
            ys[g] = y[RNG.permutation(g)]
        c += np.corrcoef(x, ys)[0, 1] >= r0
    return (c + 1) / (NBOOT + 1)
key["perm_p_wrist"] = perm(df, "QOM_WRIST_MP_px", "QOM_WRIST_VIC_px")
key["perm_p_head"] = perm(df, "QOM_HEAD_MP_px", "QOM_HEAD_tempmid_VIC_px")
json.dump({k: (float(v) if isinstance(v, (np.floating, float, np.integer)) else v) for k, v in key.items()},
          open(OUT / "key_numbers.json", "w"), indent=1, default=str)
print(main[["label", "n", "r", "ICC", "ICC_lo", "ICC_hi", "ICC_boot_lo", "ICC_boot_hi", "bias_pct", "ratio_gmean", "ratio_LoA_lo", "ratio_LoA_hi", "BA_slope_p"]].round(3).to_string())
