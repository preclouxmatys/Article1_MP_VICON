"""
TEST EXPLORATOIRE (non integre au pipeline officiel) :
Recalcule le QoM tete Vicon en moyennant les deux marqueurs de tempe
(point milieu, frame par frame) AVANT filtrage + calcul du deplacement
cumule, au lieu de sommer deux QoM calcules separement par marqueur.
Objectif : tester si cela ameliore l'accord avec MediaPipe (qui utilise
un seul point, NOSE), en supprimant l'asymetrie de definition.

Ne modifie AUCUN fichier de la pipeline officielle (02_..., 04_...,
data/final/validity_results_all.xlsx). Ecrit uniquement un fichier
xlsx a part, prefixe TEST_, et imprime les stats sur stdout.
"""
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy import stats as sstats

import config

VICON_CSV_DIR = Path(config.VICON_CSV_DIR)
INTERMEDIATE_DIR = Path(config.INTERMEDIATE_DIR)
FINAL_DIR = Path(config.FINAL_DIR)

CONDITIONS = ["SEATED", "SEMI", "STANDING"]

SUBJECT_TOKENS = {
    "P1": {"WR_D": "poignet_D", "WR_G": "poignet_G",
           "TP_D": "Tempe_D", "TP_G": "Tempe_G",
           "SH_D": "epaule_D", "SH_G": "epaule_G"},
    "P2": {"WR_D": "2poignet_D", "WR_G": "2poignet_G",
           "TP_D": "2Tempe_D", "TP_G": "2Temps_G",
           "SH_D": "2epaule_D", "SH_G": "2epaule_G"},
}


def parse_vicon_filename(stem: str):
    upper = stem.upper()
    for cond in CONDITIONS:
        if upper.startswith(cond):
            rest = stem[len(cond):].lstrip("_")
            return cond, rest
    raise ValueError(f"Condition non reconnue pour le fichier : {stem}")


def read_vicon_csv(csv_path: Path) -> pd.DataFrame:
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header_lines = [next(reader) for _ in range(5)]

    marker_row = header_lines[2]
    axis_row = header_lines[3]

    n = max(len(marker_row), len(axis_row))
    marker_row += [""] * (n - len(marker_row))
    axis_row += [""] * (n - len(axis_row))

    filled = []
    last = ""
    for m in marker_row:
        m = (m or "").strip()
        if m == "":
            filled.append(last)
        else:
            last = m
            filled.append(last)

    colnames = []
    for m, a in zip(filled, axis_row):
        m = (m or "").strip()
        a = (a or "").strip()
        if a in ["Frame", "Sub Frame"]:
            colnames.append(a)
        elif a in ["X", "Y", "Z"]:
            colnames.append(f"{m}_{a}")
        else:
            colnames.append(m if m else a)

    df = pd.read_csv(csv_path, skiprows=5, header=None, names=colnames, engine="python")
    df = df.dropna(axis=1, how="all")
    return df


def find_xyz_cols(cols, token):
    pat = re.compile(rf"(?:^|:)\s*{re.escape(token)}_([XYZ])\b", re.IGNORECASE)
    found = {}
    for c in cols:
        c2 = c.replace(" ", "")
        m = pat.search(c2)
        if m:
            axis = m.group(1).upper()
            if axis not in found or len(c) < len(found[axis]):
                found[axis] = c
    return found.get("X"), found.get("Y"), found.get("Z")


def butter_lowpass_filt(x, fs=config.FS_VICON, cutoff=config.CUTOFF_HZ, order=config.BUTTER_ORDER):
    x = np.asarray(x, dtype=float)
    if len(x) < (order * 3 + 1):
        return x
    nyq = 0.5 * fs
    w = min(cutoff / nyq, 0.99)
    b, a = butter(order, w, btype="low", analog=False)
    return filtfilt(b, a, x, method="pad")


def motion_quantity_point_filtered(df_or_arr, X=None, Y=None, Z=None):
    if isinstance(df_or_arr, pd.DataFrame):
        arr = df_or_arr[[X, Y, Z]].to_numpy(dtype=float)
    else:
        arr = df_or_arr
    arr_filt = np.column_stack([
        butter_lowpass_filt(arr[:, 0]),
        butter_lowpass_filt(arr[:, 1]),
        butter_lowpass_filt(arr[:, 2]),
    ])
    d = np.diff(arr_filt, axis=0)
    step = np.sqrt((d**2).sum(axis=1))
    step = step[np.isfinite(step)]
    return float(step.sum()), int(step.size)


def shoulder_width(df, cols, tok):
    RX, RY, RZ = find_xyz_cols(cols, tok["SH_D"])
    LX, LY, LZ = find_xyz_cols(cols, tok["SH_G"])
    if None in [RX, RY, RZ, LX, LY, LZ]:
        return None
    sw = np.sqrt((df[RX] - df[LX])**2 + (df[RY] - df[LY])**2 + (df[RZ] - df[LZ])**2)
    sw = sw.replace([np.inf, -np.inf], np.nan).dropna()
    return float(sw.median()) if len(sw) else None


def qdm_head_midpoint(df, cols, token_d, token_g):
    """QoM tete VERSION TEST : point milieu des deux tempes, calcule
    frame par frame, PUIS filtre + deplacement cumule (UNE seule fois),
    au lieu de sommer deux QoM calcules separement (version officielle)."""
    Xd, Yd, Zd = find_xyz_cols(cols, token_d)
    Xg, Yg, Zg = find_xyz_cols(cols, token_g)
    if None in [Xd, Yd, Zd, Xg, Yg, Zg]:
        return None
    arr_d = df[[Xd, Yd, Zd]].to_numpy(dtype=float)
    arr_g = df[[Xg, Yg, Zg]].to_numpy(dtype=float)
    midpoint = (arr_d + arr_g) / 2.0
    q_mid, _ = motion_quantity_point_filtered(midpoint)
    return q_mid


def qdm_for_pair_ORIGINAL(df, cols, token_d, token_g):
    """Reprise EXACTE de la version officielle (02_...), pour verifier
    que ce script reproduit bien les memes chiffres que la pipeline
    avant de comparer a la version TEST."""
    Xd, Yd, Zd = find_xyz_cols(cols, token_d)
    Xg, Yg, Zg = find_xyz_cols(cols, token_g)
    if None in [Xd, Yd, Zd, Xg, Yg, Zg]:
        return None
    q_d, _ = motion_quantity_point_filtered(df, Xd, Yd, Zd)
    q_g, _ = motion_quantity_point_filtered(df, Xg, Yg, Zg)
    return q_d + q_g


def process_one_csv(csv_path: Path):
    condition, dyad = parse_vicon_filename(csv_path.stem)
    df = read_vicon_csv(csv_path)
    cols = list(df.columns)

    rows = []
    for pid, tok in SUBJECT_TOKENS.items():
        sw_mm = shoulder_width(df, cols, tok)
        qdm_head_midpoint_mm = qdm_head_midpoint(df, cols, tok["TP_D"], tok["TP_G"])
        qdm_head_original_mm = qdm_for_pair_ORIGINAL(df, cols, tok["TP_D"], tok["TP_G"])

        rows.append({
            "dyad": dyad,
            "condition": condition,
            "pid": pid,
            "shoulder_width_mm": sw_mm,
            "QDM_HEAD_VICON_MIDPOINT_mm": qdm_head_midpoint_mm,
            "QDM_HEAD_VICON_ORIGINAL_mm": qdm_head_original_mm,
        })
    return rows


def icc_2_1(ratings):
    """ICC(2,1) - modele a deux facteurs aleatoires, mesure unique,
    accord absolu (Shrout & Fleiss, 1979 ; McGraw & Wong, 1996).
    ratings : array (n_sujets, k_juges=2) -> ici colonne0=VICON, colonne1=MP.
    Reimplementation manuelle (pas de psych::ICC disponible ici, R absent
    de cet environnement) - meme formule sous-jacente que psych::ICC type ICC2."""
    ratings = np.asarray(ratings, dtype=float)
    n, k = ratings.shape
    mean_rows = ratings.mean(axis=1)
    mean_cols = ratings.mean(axis=0)
    grand_mean = ratings.mean()

    SSR = k * np.sum((mean_rows - grand_mean) ** 2)          # sujets
    SSC = n * np.sum((mean_cols - grand_mean) ** 2)          # juges/systemes
    SST = np.sum((ratings - grand_mean) ** 2)                # total
    SSE = SST - SSR - SSC                                    # residuelle

    MSR = SSR / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))

    icc = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
    return icc


def summarize(vicon_col, mp_col, label):
    d = pd.concat([vicon_col, mp_col], axis=1).dropna()
    d.columns = ["VICON", "MP"]
    n = len(d)
    r, p = sstats.pearsonr(d["VICON"], d["MP"])
    r2 = r ** 2
    icc = icc_2_1(d[["VICON", "MP"]].to_numpy())
    diff = d["MP"] - d["VICON"]
    bias = diff.mean()
    sd_diff = diff.std(ddof=1)
    loa_lo = bias - 1.96 * sd_diff
    loa_hi = bias + 1.96 * sd_diff
    rmse = np.sqrt((diff ** 2).mean())
    mean_sys = (d["MP"] + d["VICON"]) / 2
    slope, intercept, r_ba, p_ba, se_ba = sstats.linregress(mean_sys, diff)

    print(f"\n=== {label} (n={n}) ===")
    print(f"  r    = {r:.6f}   (p = {p:.3e})   R2 = {r2:.6f}")
    print(f"  ICC(2,1) = {icc:.6f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  Biais (MP - VICON) = {bias:.4f}   LoA = [{loa_lo:.4f} ; {loa_hi:.4f}]")
    print(f"  Pente biais proportionnel = {slope:.6f}  (p = {p_ba:.3e})")
    return dict(label=label, n=n, r=r, p=p, R2=r2, ICC_2_1=icc, RMSE=rmse,
                bias=bias, LoA_lower=loa_lo, LoA_upper=loa_hi,
                BA_slope=slope, BA_slope_p=p_ba)


def main():
    csv_files = sorted(VICON_CSV_DIR.rglob("*.csv"))
    print(f"{len(csv_files)} CSV Vicon trouve(s) dans {VICON_CSV_DIR}")

    all_rows = []
    for csv_path in csv_files:
        all_rows.extend(process_one_csv(csv_path))
    vicon_test = pd.DataFrame(all_rows)

    # Verification : la colonne "ORIGINAL" recalculee ici doit etre
    # identique (aux erreurs d'arrondi pres) a la colonne officielle
    # QDM_HEAD_VICON_mm de vicon_qom_shoulder.xlsx
    vicon_official = pd.read_excel(INTERMEDIATE_DIR / "vicon_qom_shoulder.xlsx")
    check = vicon_test.merge(
        vicon_official[["dyad", "condition", "pid", "QDM_HEAD_VICON_mm"]],
        on=["dyad", "condition", "pid"], how="left"
    )
    max_diff = (check["QDM_HEAD_VICON_ORIGINAL_mm"] - check["QDM_HEAD_VICON_mm"]).abs().max()
    print(f"\n[Verification] Ecart max entre la colonne ORIGINAL recalculee ici "
          f"et le fichier officiel vicon_qom_shoulder.xlsx : {max_diff:.10f}")
    if max_diff > 1e-6:
        print("  ATTENTION : ecart non negligeable, a investiguer avant de faire confiance "
              "aux resultats TEST ci-dessous.")
    else:
        print("  OK - ce script reproduit exactement la pipeline officielle sur la version "
              "ORIGINAL, donc la version MIDPOINT ci-dessous est fiable.")

    # Normalisation par largeur d'epaule
    vicon_test["QDM_HEAD_VICON_MIDPOINT_norm"] = (
        vicon_test["QDM_HEAD_VICON_MIDPOINT_mm"] / vicon_test["shoulder_width_mm"]
    )

    # Fusion avec MediaPipe (donnees officielles, non modifiees)
    mp = pd.read_excel(INTERMEDIATE_DIR / "mediapipe_qom_shoulder.xlsx")
    df = vicon_test.merge(mp[["dyad", "condition", "pid", "QDM_HEAD_MP_norm"]],
                           on=["dyad", "condition", "pid"], how="inner")
    print(f"\nTable fusionnee : {len(df)} lignes (attendu 120)")

    # Stats : version officielle (deux points, somme) vs version TEST (point milieu)
    official_official = vicon_official.merge(mp[["dyad", "condition", "pid", "QDM_HEAD_MP_norm"]],
                                              on=["dyad", "condition", "pid"], how="inner")
    res_official = summarize(official_official["QDM_HEAD_VICON_norm"],
                              official_official["QDM_HEAD_MP_norm"],
                              "TETE - VERSION OFFICIELLE (somme de 2 QoM, 2 points)")
    res_test = summarize(df["QDM_HEAD_VICON_MIDPOINT_norm"], df["QDM_HEAD_MP_norm"],
                          "TETE - VERSION TEST (point milieu des 2 tempes, 1 seul point)")

    out_path = FINAL_DIR / "TEST_head_midpoint_comparison.xlsx"
    with pd.ExcelWriter(out_path) as writer:
        df.to_excel(writer, sheet_name="data_midpoint", index=False)
        pd.DataFrame([res_official, res_test]).to_excel(writer, sheet_name="summary", index=False)
    print(f"\nSauvegarde (TEST uniquement, pipeline officielle non touchee) : {out_path}")


if __name__ == "__main__":
    main()
