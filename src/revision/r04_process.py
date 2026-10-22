"""
r04_process.py  (pipeline de revision)

Traitement complet d'un enregistrement (participant x condition), en corrigeant
les problemes identifies par l'audit (r02/r03) :

 1. Coordonnees MediaPipe converties en PIXELS (x*W, y*H) avant tout calcul
    de distance ou d'angle (les coordonnees normalisees sont anisotropes, et
    les videos ont des recadrages differents : 840x720 ... 1280x720, 1080x1920).
 2. Timeline video complete (images non detectees = NaN, interpolees si <= 5
    images), et appariement sur le NUMERO d'image (et non l'indice de ligne).
 3. Inversion de fichiers corrigee : en D14 SEMI, la video etiquetee P1 est
    celle de P2 et inversement (verifie par correlation croisee, r = .96).
 4. Synchronisation : decalage MP<->Vicon estime par correlation croisee des
    vitesses de la TETE (r02b ; nez MP vs milieu des tempes Vicon), puis affine
    en minimisant l'erreur de reprojection des epaules et chevilles (cf. 5).
    Les analyses temporelles de la tete utilisent le decalage estime sur les poignets.
 5. Calibration camera (DLT, 11 parametres) par enregistrement, a partir des
    correspondances marqueurs Vicon <-> reperes MediaPipe du TRONC et des
    JAMBES uniquement (epaules, hanches, genoux, chevilles). Les bras (coudes,
    poignets) ne servent PAS a la calibration : ils sont donc evalues sur des
    points independants. Le Vicon est ensuite projete dans le vrai plan image.
 6. Vicon : inversions breves de labels detectees (longueur d'avant-bras ou de
    bras aberrante) -> NaN puis interpolation ; filtre Butterworth 6 Hz.
 7. Tete : point unique = milieu des deux marqueurs des tempes (Vicon), compare
    au repere du nez (MediaPipe).
Sorties : ligne de resultats (dict) + series temporelles (npz).
"""
from pathlib import Path
import sys, json
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
ARR = ROOT / "data/intermediate/revision/arrays"
TS = ROOT / "data/intermediate/revision/timeseries"; TS.mkdir(parents=True, exist_ok=True)
RES = ROOT / "data/intermediate/revision/per_recording"; RES.mkdir(parents=True, exist_ok=True)
AUDIT = ROOT / "data/intermediate/revision/r02_audit.csv"
HEADSYNC = ROOT / "data/intermediate/revision/r02b_head_sync.csv"
FC, ORDER, FSV = 6.0, 4, 100.0
VIDEO_SWAP = {("D14", "SEMI"): {"P1": "P2", "P2": "P1"}}
CALIB = [("epaule_D", "RIGHT_SHOULDER"), ("epaule_G", "LEFT_SHOULDER"), ("hanche_D", "RIGHT_HIP"),
         ("hanche_G", "LEFT_HIP"), ("genou_D", "RIGHT_KNEE"), ("genou_G", "LEFT_KNEE"),
         ("cheville_D", "RIGHT_ANKLE"), ("cheville_G", "LEFT_ANKLE")]
# Note : l'ajout des orteils (orteil_D/G <-> FOOT_INDEX) a ete teste et degrade la
# calibration (pieds hors champ ou mal detectes dans les videos recadrees) ; non retenu.
ARM = [("coude_D", "RIGHT_ELBOW"), ("coude_G", "LEFT_ELBOW"), ("poignet_D", "RIGHT_WRIST"), ("poignet_G", "LEFT_WRIST")]


# ---------------------------------------------------------------- utilitaires
def interp_nan(x, maxgap=None):
    df = pd.DataFrame(x)
    if maxgap is None:
        return df.interpolate(limit_direction="both").to_numpy()
    out = df.interpolate(limit_direction="both", limit=maxgap, limit_area=None).to_numpy()
    # remettre NaN dans les trous plus longs que maxgap
    isn = np.isnan(np.asarray(x, float)).any(axis=1) if np.ndim(x) > 1 else np.isnan(x)
    run = np.zeros(len(isn), int); c = 0
    for i, v in enumerate(isn):
        c = c + 1 if v else 0; run[i] = c
    longgap = np.zeros(len(isn), bool); i = len(isn) - 1
    while i >= 0:
        if run[i] > maxgap:
            L = run[i]; longgap[i - L + 1:i + 1] = True; i -= L
        else:
            i -= 1
    out[longgap] = np.nan
    return out


def lowpass(x, fs):
    x = np.asarray(x, float)
    mask = np.isnan(x).any(axis=1)
    xf = interp_nan(x)
    b, a = butter(ORDER, FC / (fs / 2))
    y = filtfilt(b, a, xf, axis=0)
    y[mask] = np.nan
    return y


def path_length(p):
    s = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return float(np.nansum(s))


def step(p):
    s = np.linalg.norm(np.diff(p, axis=0), axis=1)
    return np.r_[s, np.nan]


def angle(A, B, C):
    u, v = A - B, C - B
    c = np.sum(u * v, 1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def amp_robust(dist):
    d = dist[np.isfinite(dist)]
    return (np.percentile(d, 95) - np.percentile(d, 5), d.max() - d.min()) if d.size else (np.nan, np.nan)


def agree(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 50:
        return dict(r=np.nan, rmse=np.nan, bias=np.nan, sdd=np.nan, n=int(m.sum()))
    x, y = x[m], y[m]; dd = x - y
    return dict(r=float(np.corrcoef(x, y)[0, 1]), rmse=float(np.sqrt(np.mean(dd ** 2))),
                bias=float(dd.mean()), sdd=float(dd.std()), n=int(m.sum()))


# ---------------------------------------------------------------- DLT
def dlt_linear(X, u):
    m = np.isfinite(X).all(1) & np.isfinite(u).all(1)
    X, u = X[m], u[m]
    Xm, Xs = X.mean(0), X.std()
    um, us = u.mean(0), u.std()
    Xn, un = (X - Xm) / Xs, (u - um) / us
    n = len(X); A = np.zeros((2 * n, 12))
    Xh = np.c_[Xn, np.ones(n)]
    A[0::2, 0:4] = Xh; A[0::2, 8:12] = -un[:, [0]] * Xh
    A[1::2, 4:8] = Xh; A[1::2, 8:12] = -un[:, [1]] * Xh
    P = np.linalg.svd(A, full_matrices=False)[2][-1].reshape(3, 4)
    T = np.diag([us, us, 1.0]); T[:2, 2] = um
    U = np.eye(4) / Xs; U[3, 3] = 1; U[:3, 3] = -Xm / Xs
    P = T @ P @ U
    return P / np.linalg.norm(P[2, :3])


def project(P, X):
    Xh = np.c_[X, np.ones(len(X))]
    q = Xh @ P.T
    return q[:, :2] / q[:, [2]]


def dlt_refine(P, X, u):
    m = np.isfinite(X).all(1) & np.isfinite(u).all(1)
    X, u = X[m], u[m]
    def res(p):
        return (project(np.r_[p, 1.0].reshape(3, 4) / 1.0, X) - u).ravel()
    p0 = (P / P[2, 3]).ravel()[:11]
    sol = least_squares(res, p0, loss="soft_l1", f_scale=10.0, max_nfev=200)
    P2 = np.r_[sol.x, 1.0].reshape(3, 4)
    return P2 / np.linalg.norm(P2[2, :3])


def affine_fit(X, u):
    """Camera affine (8 parametres) : u = A X + b. Utilisee quand la DLT
    projective est instable (singularite w ~ 0), cas frequent ici car le
    participant occupe une faible profondeur a ~3-4 m de la camera."""
    m = np.isfinite(X).all(1) & np.isfinite(u).all(1)
    Xh = np.c_[X[m], np.ones(m.sum())]
    sol = np.linalg.lstsq(Xh, u[m], rcond=None)[0]  # (4, 2)
    for _ in range(3):  # IRLS robuste (Huber-like)
        e = np.linalg.norm(Xh @ sol - u[m], axis=1)
        w = 1 / np.maximum(e, 10.0)
        sol = np.linalg.lstsq(Xh * w[:, None], u[m] * w[:, None], rcond=None)[0]
    P = np.zeros((3, 4)); P[:2] = sol.T; P[2, 3] = 1.0
    return P


def camera_center(P):
    return -np.linalg.inv(P[:, :3]) @ P[:, 3]


# ---------------------------------------------------------------- tete
# ---------------------------------------------------------------- Vicon nettoyage
def clean_segments(V):
    """Detecte les inversions breves de labels : longueur bras/avant-bras
    s'ecartant de plus de 25 % de sa mediane -> NaN pour le marqueur distal."""
    n_bad = 0
    for s in "DG":
        for prox, dist in [(f"epaule_{s}", f"coude_{s}"), (f"coude_{s}", f"poignet_{s}")]:
            L = np.linalg.norm(V[dist] - V[prox], axis=1)
            med = np.nanmedian(L)
            bad = np.abs(L - med) > 0.25 * med
            # dilatation de 5 images de part et d'autre
            bad = np.convolve(bad.astype(int), np.ones(11), "same") > 0
            V[dist][bad] = np.nan; n_bad += int(bad.sum())
    return n_bad


# ---------------------------------------------------------------- principal
def process(dyad, cond, pid, audit):
    dv = np.load(ARR / f"{dyad}_{cond}_{pid}.npz")
    pid_vid = VIDEO_SWAP.get((dyad, cond), {}).get(pid, pid)
    dm = np.load(ARR / f"{dyad}_{cond}_{pid_vid}.npz")
    names = list(dm["MP_names"]); W, H, fps = float(dm["W"]), float(dm["H"]), float(dm["fps"])
    MP = dm["MP"]; nvid = MP.shape[0]
    def mp(nm, px=True):
        a = MP[:, names.index(nm), :2].copy()
        if px:
            a[:, 0] *= W; a[:, 1] *= H
        return interp_nan(a, maxgap=5)
    def vis(nm):
        return MP[:, names.index(nm), 3]

    V = {k[2:]: dv[k].astype(float).copy() for k in dv.files if k.startswith("V_")}
    ex = ARR.parent / "arrays_extra" / f"{dyad}_{cond}_{pid}.npz"
    if ex.exists():
        de = np.load(ex); V.update({k[2:]: de[k].astype(float).copy() for k in de.files})
    n_bad = clean_segments(V)
    nvic = int(dv["n_vicon"]); tv = np.arange(nvic) / FSV
    Vf = {}
    for k, a in V.items():
        if np.isfinite(a).mean() < 0.5:
            Vf[k] = np.full_like(a, np.nan); continue
        Vf[k] = lowpass(interp_nan(a, maxgap=200), FSV)
    Vf["tempmid"] = (Vf["Tempe_D"] + Vf["Tempe_G"]) / 2

    # MediaPipe filtre (pixels)
    M = {nm: lowpass(mp(nm), fps) for nm in set([b for _, b in CALIB + ARM] + ["NOSE"])}

    # ---- synchro : la correlation croisee des poignets (r02) n'est conservee qu'a titre
    # de controle ; le decalage est estime UNIQUEMENT a partir des points de calibration
    # (epaules, chevilles, orteils), sans aucun point des bras ni de la tete.
    # Decalage principal : correlation croisee sur la TETE (r02b), independante des bras ;
    # affine ensuite par reprojection des points de calibration (tronc, pieds).
    # Le decalage estime sur les POIGNETS (r02) sert pour les analyses temporelles de la tete.
    a = audit.set_index(["dyad", "condition", "pid"])
    row = a.loc[(dyad, cond, pid)]
    peak = float(row.xcorr_peak) if (dyad, cond) not in VIDEO_SWAP else np.nan
    lag_xcorr = float(row.xcorr_lag_frames) if (dyad, cond) not in VIDEO_SWAP else np.nan
    hs = pd.read_csv(HEADSYNC).set_index(["dyad", "condition", "pid"]).loc[(dyad, cond, pid)]
    lag_head, head_peak = float(hs.head_lag_frames), float(hs.head_peak)

    def vic_at(lag, keys):
        t = (np.arange(nvid) + lag) / fps
        out = {}
        for k in keys:
            arr = Vf[k]
            out[k] = np.column_stack([np.interp(t, tv, arr[:, j], left=np.nan, right=np.nan) for j in range(3)])
        return out

    calib_keys = [v for v, _ in CALIB]
    sub = np.arange(0, nvid, 3)
    def calib_data(lag):
        VV = vic_at(lag, calib_keys)
        X = np.vstack([VV[v][sub] for v, m in CALIB]); u = np.vstack([M[m][sub] for v, m in CALIB])
        vv = np.concatenate([vis(m)[sub] for v, m in CALIB])
        ok = np.isfinite(X).all(1) & np.isfinite(u).all(1) & (vv > 0.5)
        return X[ok], u[ok]
    best = None
    for l2 in np.arange(lag_head - 12, lag_head + 12.01, 0.25):
        X, u = calib_data(l2)
        if len(X) < 200:
            continue
        P = dlt_linear(X, u)
        e = np.median(np.linalg.norm(project(P, X) - u, axis=1))
        if best is None or e < best[0]:
            best = (e, l2, P)
    lag = best[1]
    # decalage pour la tete : meme affinage, centre sur l'estimation 'poignets' si fiable
    lag_for_head = lag + ((lag_xcorr - lag_head) if (np.isfinite(lag_xcorr) and peak >= 0.6) else 0.0)
    X, u = calib_data(lag)
    P_proj = dlt_refine(best[2], X, u)
    P_aff = affine_fit(X, u)
    allX = np.vstack([Vf[k][::10] for k in ["epaule_D", "epaule_G", "coude_D", "coude_G", "poignet_D",
                                            "poignet_G", "hanche_D", "hanche_G", "genou_D", "genou_G",
                                            "cheville_D", "cheville_G", "tempmid"]])
    allX = allX[np.isfinite(allX).all(1)]
    w = np.c_[allX, np.ones(len(allX))] @ P_proj[2]
    w = w * np.sign(np.median(w))
    stable = (np.min(w) / np.median(w)) > 0.5
    e_proj = np.median(np.linalg.norm(project(P_proj, X) - u, axis=1))
    e_aff = np.median(np.linalg.norm(project(P_aff, X) - u, axis=1))
    use_proj = bool(stable and e_proj <= e_aff)
    P = P_proj if use_proj else P_aff
    err_cal = np.linalg.norm(project(P, X) - u, axis=1)
    keys = list(Vf.keys())
    VV = vic_at(lag, keys)
    proj = {k: project(P, VV[k]) for k in keys}

    # erreur de reprojection sur points NON utilises (bras)
    err_arm = {}
    for v, m in ARM:
        e = np.linalg.norm(proj[v] - M[m], axis=1); e = e[np.isfinite(e) & (vis(m) > 0.5)]
        err_arm[m] = float(np.median(e)) if e.size else np.nan

    # geometrie camera : direction de visee (robuste meme si la DLT est quasi
    # affine, ce qui rend la position 3D du centre optique mal determinee)
    shC3 = np.nanmedian((Vf["epaule_D"] + Vf["epaule_G"]) / 2, 0)
    if use_proj and np.median(np.c_[X, np.ones(len(X))] @ P[2]) < 0:
        P = -P
    w0 = P[2] @ np.r_[shC3, 1.0]
    p0 = (P[:2] @ np.r_[shC3, 1.0]) / w0
    J = (P[:2, :3] - np.outer(p0, P[2, :3])) / w0
    nview = np.linalg.svd(J)[2][-1]
    # sens : chaque camera est placee derriere le partenaire (dispositif), donc du
    # cote vers lequel le participant est tourne. Avant du participant = perpendiculaire
    # a la ligne des epaules, orientee vers le partenaire (Vicon uniquement).
    partner_ = "P2" if pid == "P1" else "P1"
    dp_ = np.load(ARR / f"{dyad}_{cond}_{partner_}.npz")
    pC_ = np.nanmedian((dp_["V_epaule_D"] + dp_["V_epaule_G"]) / 2, 0)
    shl_ = np.nanmedian(Vf["epaule_D"] - Vf["epaule_G"], 0)
    fwd_ = np.array([-shl_[1], shl_[0], 0.0])
    if np.dot(fwd_[:2], (pC_ - shC3)[:2]) < 0:
        fwd_ = -fwd_
    if nview @ fwd_ < 0:
        nview = -nview  # vecteur unitaire du participant vers la camera
    partner = "P2" if pid == "P1" else "P1"
    dp = np.load(ARR / f"{dyad}_{cond}_{partner}.npz")
    pC = np.nanmedian((dp["V_epaule_D"] + dp["V_epaule_G"]) / 2, 0)
    to_par = pC - shC3
    az = lambda v: np.degrees(np.arctan2(v[1], v[0]))
    shl = np.nanmedian(Vf["epaule_D"] - Vf["epaule_G"], 0)
    fwd = np.array([-shl[1], shl[0], 0.0])
    if np.dot(fwd[:2], to_par[:2]) < 0:
        fwd = -fwd
    near_D = float(np.nanmedian((Vf["epaule_D"] - Vf["epaule_G"]) @ nview))
    geo = dict(view_azimuth_vs_partner_deg=float((az(nview) - az(to_par) + 180) % 360 - 180),
               view_azimuth_vs_facing_deg=float((az(nview) - az(fwd) + 180) % 360 - 180),
               view_elevation_deg=float(np.degrees(np.arcsin(np.clip(nview[2], -1, 1)))),
               near_side="D" if near_D > 0 else "G", near_shoulder_depth_diff_mm=near_D,
               interpersonal_dist_mm=float(np.linalg.norm(to_par[:2])))

    out = dict(dyad=dyad, condition=cond, pid=pid, video_file_pid=pid_vid, W=W, H=H, fps=fps,
               lag_xcorr_s=lag_xcorr / fps, xcorr_peak=peak, lag_s=lag / fps, lag_head_xcorr_s=lag_head / fps, head_peak=head_peak, lag_for_head_s=lag_for_head / fps, camera_model="projective" if use_proj else "affine",
               reproj_calib_proj_px=float(e_proj), reproj_calib_aff_px=float(e_aff), reproj_calib_med_px=float(np.median(err_cal)),
               reproj_calib_p90_px=float(np.percentile(err_cal, 90)), n_calib=len(X),
               vicon_labelfix_frames=n_bad, **geo,
               **{f"reproj_holdout_{m}_px": e for m, e in err_arm.items()})

    # échelle px/mm locale : largeur d'epaule projetee vs 3D
    sw3 = np.linalg.norm(VV["epaule_D"] - VV["epaule_G"], axis=1)
    sw_mp_px = np.linalg.norm(M["RIGHT_SHOULDER"] - M["LEFT_SHOULDER"], axis=1)
    sw_vic_px = np.linalg.norm(proj["epaule_D"] - proj["epaule_G"], axis=1)
    out.update(sw3_med_mm=float(np.nanmedian(sw3)), sw_mp_px_med=float(np.nanmedian(sw_mp_px)),
               sw_vicproj_px_med=float(np.nanmedian(sw_vic_px)),
               sw_mp_norm_med=float(np.nanmedian(np.linalg.norm(
                   (MP[:, names.index("RIGHT_SHOULDER"), :2] - MP[:, names.index("LEFT_SHOULDER"), :2]), axis=1))))
    # pixels par mm a la profondeur du participant (pour exprimer des erreurs en mm)
    px_per_mm = np.nanmedian(sw_vic_px / sw3)
    out["px_per_mm"] = float(px_per_mm)

    # ------------------------------------------------ QoM
    win = int(round(fps))
    VVh = vic_at(lag_for_head, ["tempmid"])
    projh = {"tempmid": project(P, VVh["tempmid"])}
    for lbl, vk, mk in [("WRIST", ["poignet_D", "poignet_G"], ["RIGHT_WRIST", "LEFT_WRIST"]),
                        ("HEAD", ["tempmid"], ["NOSE"])]:
        # (a) Vicon 3D a 100 Hz / largeur d'epaule 3D (controle de l'effet de cadence)
        out[f"QOM_{lbl}_V3D100_norm"] = sum(path_length(Vf[k]) for k in vk) / np.nanmedian(
            np.linalg.norm(Vf["epaule_D"] - Vf["epaule_G"], axis=1))
        # (b) Vicon 3D re-echantillonne a la cadence video (25 Hz), synchronise
        out[f"QOM_{lbl}_V3D25_norm"] = sum(path_length(VV[k]) for k in vk) / np.nanmedian(sw3)
        # (c) Vicon projete dans l'image (px) et MediaPipe (px)
        out[f"QOM_{lbl}_VIC_px"] = sum(path_length(proj[k]) for k in vk)
        out[f"QOM_{lbl}_MP_px"] = sum(path_length(M[m]) for m in mk)
        # (d) normalises par la largeur d'epaule (px) de chaque systeme
        out[f"QOM_{lbl}_VIC_norm"] = out[f"QOM_{lbl}_VIC_px"] / np.nanmedian(sw_vic_px)
        out[f"QOM_{lbl}_MP_norm"] = out[f"QOM_{lbl}_MP_px"] / np.nanmedian(sw_mp_px)
        # (e) MediaPipe en coordonnees normalisees (analyse de sensibilite)
        raw = sum(path_length(lowpass(mp(m, px=False), fps)) for m in mk)
        out[f"QOM_{lbl}_MPnormcoord_norm"] = raw / out["sw_mp_norm_med"]
        # QoM fenetre (1 s) : correlation des series temporelles
        srcp = proj if lbl == "WRIST" else projh  # tete : decalage estime sur les poignets
        sv = sum(step(srcp[k]) for k in vk); sm = sum(step(M[m]) for m in mk)
        nw = len(sv) // win
        qv = np.nansum(sv[:nw * win].reshape(nw, win), 1); qm = np.nansum(sm[:nw * win].reshape(nw, win), 1)
        g = agree(qm, qv)
        out[f"QOMwin_{lbl}_r"] = g["r"]
        # bruit MediaPipe : pas inter-images quand le Vicon projete est quasi immobile
        still = sv < np.nanpercentile(sv, 10)
        out[f"MP_jitter_{lbl}_px"] = float(np.nanmedian(sm[still]))
    # QoM tete : milieu des tempes (definition principale)
    out["QOM_HEAD_tempmid_V3D100_norm"] = path_length(Vf["tempmid"]) / np.nanmedian(
        np.linalg.norm(Vf["epaule_D"] - Vf["epaule_G"], axis=1))
    out["QOM_HEAD_tempmid_VIC_px"] = path_length(proj["tempmid"])
    out["QOM_HEAD_tempmid_V3D25_norm"] = path_length(VV["tempmid"]) / np.nanmedian(sw3)
    out["QOM_HEAD_tempmid_VIC_norm"] = out["QOM_HEAD_tempmid_VIC_px"] / np.nanmedian(sw_vic_px)
    sv = step(projh["tempmid"]); sm = step(M["NOSE"]); nw = len(sv) // win
    out["QOMwin_HEAD_tempmid_r"] = agree(np.nansum(sm[:nw * win].reshape(nw, win), 1),
                                         np.nansum(sv[:nw * win].reshape(nw, win), 1))["r"]

    # ------------------------------------------------ amplitude poignet
    Cm = (M["RIGHT_SHOULDER"] + M["LEFT_SHOULDER"]) / 2
    Cv = (proj["epaule_D"] + proj["epaule_G"]) / 2
    C3 = (VV["epaule_D"] + VV["epaule_G"]) / 2
    scale_old = out["sw3_med_mm"] / out["sw_mp_px_med"]
    for kind in ["p95p5", "maxmin"]:
        j = 0 if kind == "p95p5" else 1
        out[f"AMP_{kind}_MP_px"] = sum(amp_robust(np.linalg.norm(M[m] - Cm, axis=1))[j] for m in ["RIGHT_WRIST", "LEFT_WRIST"])
        out[f"AMP_{kind}_VIC_px"] = sum(amp_robust(np.linalg.norm(proj[k] - Cv, axis=1))[j] for k in ["poignet_D", "poignet_G"])
        out[f"AMP_{kind}_V3D_mm"] = sum(amp_robust(np.linalg.norm(VV[k] - C3, axis=1))[j] for k in ["poignet_D", "poignet_G"])
        out[f"AMP_{kind}_MP_mm_shoulderscale"] = out[f"AMP_{kind}_MP_px"] * scale_old
        for pl, ix in {"XY": [0, 1], "XZ": [0, 2], "YZ": [1, 2]}.items():
            out[f"AMP_{kind}_V{pl}_mm"] = sum(amp_robust(np.linalg.norm(VV[k][:, ix] - C3[:, ix], axis=1))[j]
                                              for k in ["poignet_D", "poignet_G"])

    # ------------------------------------------------ angle du coude
    for s, side in [("D", "RIGHT"), ("G", "LEFT")]:
        a_mp = angle(M[f"{side}_SHOULDER"], M[f"{side}_ELBOW"], M[f"{side}_WRIST"])
        a_mpn = angle(*[lowpass(mp(f"{side}_{j}", px=False), fps) for j in ["SHOULDER", "ELBOW", "WRIST"]])
        a_vic = angle(proj[f"epaule_{s}"], proj[f"coude_{s}"], proj[f"poignet_{s}"])
        a_v3 = angle(VV[f"epaule_{s}"], VV[f"coude_{s}"], VV[f"poignet_{s}"])
        vmask = (vis(f"{side}_SHOULDER") > 0.5) & (vis(f"{side}_ELBOW") > 0.5) & (vis(f"{side}_WRIST") > 0.5)
        out[f"vis_ok_pct_{s}"] = 100 * float(np.mean(vmask))
        for ref, aref in [("VIC", a_vic), ("V3D", a_v3)]:
            for k, v in agree(a_mp, aref).items():
                out[f"ELB_{s}_MP_vs_{ref}_{k}"] = v
            a_f = a_mp.copy(); a_f[~vmask] = np.nan
            for k, v in agree(a_f, aref).items():
                out[f"ELB_{s}_MPvis_vs_{ref}_{k}"] = v
        for k, v in agree(a_mpn, a_v3).items():
            out[f"ELB_{s}_MPnormcoord_vs_V3D_{k}"] = v
        for k, v in agree(a_vic, a_v3).items():
            out[f"ELB_{s}_VIC_vs_V3D_{k}"] = v
        for pl, ix in {"XY": [0, 1], "XZ": [0, 2], "YZ": [1, 2]}.items():
            a_pl = angle(VV[f"epaule_{s}"][:, ix], VV[f"coude_{s}"][:, ix], VV[f"poignet_{s}"][:, ix])
            out[f"ELB_{s}_MP_vs_{pl}_r"] = agree(a_mp, a_pl)["r"]
        out[f"ELB_{s}_V3D_min"] = float(np.nanpercentile(a_v3, 1))
        out[f"ELB_{s}_MP_min"] = float(np.nanpercentile(a_mp, 1))
        out[f"ELB_{s}_VIC_min"] = float(np.nanpercentile(a_vic, 1))
        if s == "D":
            ts_elb = dict(elb_mp=a_mp, elb_vic=a_vic, elb_v3=a_v3)

    # ------------------------------------------------ series temporelles (temps Vicon, cadence video)
    t_vic = (np.arange(nvid) + lag) / fps
    np.savez_compressed(TS / f"{dyad}_{cond}_{pid}.npz", t=t_vic, fps=fps,
                        wrist_mp=sum(step(M[m]) for m in ["RIGHT_WRIST", "LEFT_WRIST"]),
                        wrist_vic=sum(step(proj[k]) for k in ["poignet_D", "poignet_G"]),
                        wrist_v3=sum(step(VV[k]) for k in ["poignet_D", "poignet_G"]),
                        head_mp=step(M["NOSE"]), head_vic=step(proj["tempmid"]), head_v3=step(VV["tempmid"]),
                        headtm_vic=step(proj["tempmid"]), headtm_v3=step(VV["tempmid"]),
                        sw_mp_px=np.nanmedian(sw_mp_px), sw_vic_px=np.nanmedian(sw_vic_px), sw3=np.nanmedian(sw3),
                        **ts_elb)
    return out


if __name__ == "__main__":
    audit = pd.read_csv(AUDIT)
    recs = sorted(p.stem for p in ARR.glob("*.npz"))
    force = "--force" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    todo = [r for r in recs if force or not (RES / f"{r}.json").exists()]
    if args:
        todo = [r for r in todo if any(r.startswith(a) for a in args)]
    for r in todo:
        d, c, p = r.split("_")
        try:
            o = process(d, c, p, audit)
        except Exception as e:
            import traceback; traceback.print_exc(); o = dict(dyad=d, condition=c, pid=p, error=str(e))
        (RES / f"{r}.json").write_text(json.dumps(o, default=float))
        print(r, "lag=%.2fs reproj=%.1fpx holdout_wrist=%s" % (o.get("lag_s", np.nan), o.get("reproj_calib_med_px", np.nan),
              round(o.get("reproj_holdout_RIGHT_WRIST_px", np.nan), 1)), flush=True)
