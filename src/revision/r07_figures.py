"""r07_figures.py : figures de l'article (version revisee)."""
from pathlib import Path
import json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Wedge

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / "data/intermediate/revision"; OUT = ROOT / "data/final/revision"
FIG = OUT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(REV / "r04_all_recordings.csv")
for k in ["p95p5", "maxmin"]:
    df[f"AMP_{k}_MP_norm"] = df[f"AMP_{k}_MP_px"] / df.sw_mp_px_med
    df[f"AMP_{k}_V3D_norm"] = df[f"AMP_{k}_V3D_mm"] / df.sw3_med_mm
df["calib_ok"] = (df.AMP_p95p5_VIC_px / (df.AMP_p95p5_V3D_mm * df.px_per_mm)) <= 3
DF_ALL = df
key = json.load(open(OUT / "key_numbers.json"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titlesize": 11, "axes.titleweight": "bold"})
COND = {"SEATED": ("Assis", "#3b6ea8"), "SEMI": ("Semi-debout", "#d08a2e"), "STANDING": ("Debout", "#4f9d69")}


def scatter_ba(ax1, ax2, mp, ref, xlabel, ylabel, unit, title, data=None):
    global df
    df_save = df
    if data is not None:
        df = data
    for c, (lab, col) in COND.items():
        g = df[df.condition == c]
        ax1.scatter(g[ref], g[mp], s=22, alpha=.8, color=col, edgecolor="white", linewidth=.4, label=lab)
    lim = [0, max(df[mp].max(), df[ref].max()) * 1.05]
    ax1.plot(lim, lim, ls="--", color="0.4", lw=1); ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel(xlabel); ax1.set_ylabel(ylabel); ax1.set_title(title, loc="left")
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    lr = np.log(df[mp] / df[ref]); gm = (df[mp] * df[ref]) ** .5
    m, s = lr.mean(), lr.std(ddof=1)
    for c, (lab, col) in COND.items():
        k = df.condition == c
        ax2.scatter(gm[k], 100 * (np.exp(lr[k]) - 1), s=22, alpha=.8, color=col, edgecolor="white", linewidth=.4)
    for v, ls in [(m, "-"), (m - 1.96 * s, "--"), (m + 1.96 * s, "--")]:
        ax2.axhline(100 * (np.exp(v) - 1), color="0.25", ls=ls, lw=1)
        ax2.text(gm.max() * 1.01, 100 * (np.exp(v) - 1), f"{100*(np.exp(v)-1):+.0f} %", va="center", fontsize=8)
    ax2.axhline(0, color="0.7", lw=.8)
    pct = 100 * (np.exp(lr) - 1); top = 250
    if (pct > top).any():
        ax2.set_ylim(min(-100, pct.min() - 10), top)
        for j, (gx, pv) in enumerate(sorted(zip(gm[pct > top], pct[pct > top]))):
            ax2.annotate(f"{pv:+.0f} %", (gx, top), xytext=(-18 + 36 * j, -14), textcoords="offset points", ha="center", fontsize=7,
                         arrowprops=dict(arrowstyle="-|>", color="0.3", lw=.6))
    ax2.set_xscale("log"); ax2.set_xlabel(f"Moyenne géométrique des deux systèmes ({unit})")
    ax2.set_ylabel("Écart MediaPipe / Vicon (%)"); ax2.set_title("Bland-Altman (log-ratio)", loc="left")
    df = df_save


# Figure 1 : dispositif (vue de dessus, valeurs mesurees)
fig, ax = plt.subplots(figsize=(6.4, 3.6))
D = key["interpersonal_mm_mean"] / 1000; az = np.radians(key["view_az_abs_median"])
p1, p2 = np.array([0, 0]), np.array([D, 0])
for p, lab, ang in [(p1, "P1", 0), (p2, "P2", 180)]:
    ax.add_patch(Circle(p, .17, color="#cfd8e3", ec="#41536b", lw=1.2, zorder=3))
    ax.add_patch(Wedge(p, .17, ang - 25, ang + 25, color="#41536b", zorder=4))
    ax.text(p[0], p[1] - .3, lab, ha="center", fontsize=10, fontweight="bold")
dist_cam = 3.1  # schematique (la distance absolue n'est pas identifiable avec un modele quasi affine)
for p, s, col, lab in [(p1, 1, "#3b6ea8", "Caméra P1"), (p2, -1, "#d08a2e", "Caméra P2")]:
    base = 0 if s == 1 else np.pi
    cam = p + dist_cam * np.array([np.cos(base + s * az), np.sin(base + s * az)])
    ax.plot(*cam, marker="s", ms=11, color=col, zorder=5)
    ax.add_patch(FancyArrowPatch(cam, p + (cam - p) * .08, arrowstyle="-|>", mutation_scale=10, color=col, lw=1.3, ls="--"))
    ax.text(cam[0], cam[1] + .22, lab, ha="center", color=col, fontsize=9)
ax.annotate("", p2, p1, arrowprops=dict(arrowstyle="<->", color="0.4"))
ax.text(D / 2, -.12, f"{D:.2f} m (DS {key['interpersonal_mm_sd']/1000:.2f})", ha="center", fontsize=8, color="0.3")
ax.text(.12, .62, f"{key['view_az_abs_median']:.0f}°\n(IQR {key['view_az_abs_q1']:.0f}–{key['view_az_abs_q3']:.0f}°)",
        fontsize=8, color="#3b6ea8")
ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.5, D + 1.5); ax.set_ylim(-.6, 2.3)
ax.set_title("Dispositif vu de dessus (angles mesurés par calibration)", loc="center", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "fig1_dispositif.png", dpi=300); plt.close(fig)

# Figures 2-4 : comparaisons principales, SANS calibration (normalisees par la largeur d'epaule)
for fname, mp, ref, title in [
        ("fig2_qom_poignet.png", "QOM_WRIST_MP_norm", "QOM_WRIST_V3D25_norm", "QoM poignets"),
        ("fig3_qom_tete.png", "QOM_HEAD_MP_norm", "QOM_HEAD_tempmid_V3D25_norm", "QoM tête (nez MP vs milieu des tempes)"),
        ("fig4_amplitude.png", "AMP_p95p5_MP_norm", "AMP_p95p5_V3D_norm", "Amplitude des poignets (P95–P5)")]:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
    scatter_ba(a1, a2, mp, ref, "Vicon 3D (largeurs d'épaule)", "MediaPipe 2D (largeurs d'épaule)", "larg. d'épaule", title)
    fig.tight_layout(); fig.savefig(FIG / fname, dpi=300); plt.close(fig)
# Figure S3 : comparaisons dans le plan image (calibrations valides)
fig, axs = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, mp, ref, title in [(axs[0], "QOM_WRIST_MP_px", "QOM_WRIST_VIC_px", "QoM poignets"),
                           (axs[1], "QOM_HEAD_MP_px", "QOM_HEAD_tempmid_VIC_px", "QoM tête"),
                           (axs[2], "AMP_p95p5_MP_px", "AMP_p95p5_VIC_px", "Amplitude poignets")]:
    g = df[df.calib_ok]
    for c, (lab, col) in COND.items():
        h = g[g.condition == c]
        ax.scatter(h[ref], h[mp], s=16, alpha=.8, color=col, edgecolor="white", linewidth=.3, label=lab)
    lim = [0, max(g[mp].max(), g[ref].max()) * 1.05]
    ax.plot(lim, lim, "--", color="0.4", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Vicon projeté (px)"); ax.set_ylabel("MediaPipe (px)"); ax.set_title(title, loc="left")
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "figS3_plan_image.png", dpi=300); plt.close(fig)

# Figure 5 : angle du coude, bras proche vs eloigne
el = pd.read_csv(OUT / "elbow_per_recording_side.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8))
groups = [("Bras proche", el[el.near]), ("Bras éloigné", el[~el.near])]
cols = ["#3b6ea8", "#b5523b"]
for j, (metric, ax, ylab) in enumerate([("MP_V3D_r", a1, "r (série temporelle, par enregistrement)"),
                                        ("MP_V3D_rmse", a2, "RMSE (°)")]):
    data = [g[metric].dropna().to_numpy() for _, g in groups]
    bp = ax.boxplot(data, widths=.5, showfliers=False, patch_artist=True, medianprops=dict(color="black"))
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(.35)
    rng = np.random.default_rng(0)
    for i, (dd, c) in enumerate(zip(data, cols)):
        ax.scatter(i + 1 + rng.uniform(-.15, .15, len(dd)), dd, s=8, color=c, alpha=.6)
    ax.set_xticks([1, 2]); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylabel(ylab)
a1.set_title("Angle du coude : MediaPipe 2D vs Vicon 3D", loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig5_coude_proche_eloigne.png", dpi=300); plt.close(fig)

# Figure 6 : serie temporelle (bras proche droit, enregistrement median)
cand = el[(el.near) & (el.side == "D") & (el.calib_ok)].copy()
med = cand.MP_V3D_r.median(); row = cand.iloc[(cand.MP_V3D_r - med).abs().argmin()]
ts = np.load(REV / f"timeseries/{row.dyad}_{row.condition}_{row.pid}.npz")
t = ts["t"] - ts["t"][0]
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(t, ts["elb_v3"], color="#b5523b", lw=1, label="Vicon 3D")
ax.plot(t, ts["elb_vic"], color="0.55", lw=.8, ls="--", label="Vicon projeté (plan image)")
ax.plot(t, ts["elb_mp"], color="#3b6ea8", lw=1, label="MediaPipe 2D (pixels)")
ax.set_xlabel("Temps (s)"); ax.set_ylabel("Angle du coude (°)"); ax.legend(frameon=False, ncol=3, fontsize=8, loc="lower left")
ax.set_title(f"Bras proche de la caméra, enregistrement médian ({row.dyad}, {COND[row.condition][0].lower()}, r = {row.MP_V3D_r:.2f})", loc="left")
fig.tight_layout(); fig.savefig(FIG / "fig6_coude_serie.png", dpi=300); plt.close(fig)
json.dump(dict(dyad=row.dyad, condition=row.condition, pid=row.pid, r=row.MP_V3D_r, rmse=row.MP_V3D_rmse,
               r_vic=row.MP_VIC_r), open(OUT / "fig6_example.json", "w"))

# Figure S1 : effet de la correction des coordonnees normalisees
fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, mp, ref, ttl in [(axs[0], "QOM_WRIST_MPnormcoord_norm", "QOM_WRIST_V3D25_norm", "Coordonnées normalisées"),
                         (axs[1], "QOM_WRIST_MP_norm", "QOM_WRIST_V3D25_norm", "Pixels")]:
    ax.scatter(df[ref], df[mp], s=18, color="#3b6ea8", alpha=.7)
    lim = [0, max(df["QOM_WRIST_MPnormcoord_norm"].max(), df[ref].max()) * 1.05]
    ax.plot(lim, lim, "--", color="0.4", lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Vicon 3D (largeurs d'épaule)"); ax.set_ylabel("MediaPipe (largeurs d'épaule)"); ax.set_title(ttl, loc="left")
fig.tight_layout(); fig.savefig(FIG / "figS1_correction_pixels.png", dpi=300); plt.close(fig)

# Figure S2 : decalages temporels
fig, ax = plt.subplots(figsize=(6, 3))
ax.hist(df.lag_s.clip(-2, 2), bins=40, color="#3b6ea8", alpha=.8)
ax.set_xlabel("Décalage MediaPipe – Vicon estimé (s, borné à ±2 s)"); ax.set_ylabel("Enregistrements")
fig.tight_layout(); fig.savefig(FIG / "figS2_decalages.png", dpi=300); plt.close(fig)
print("figures ok")
