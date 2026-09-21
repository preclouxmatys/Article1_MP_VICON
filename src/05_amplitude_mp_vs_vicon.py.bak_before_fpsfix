"""
05_amplitude_mp_vs_vicon.py

Amplitude pic-a-pic des poignets par rapport au centre des epaules :
  - MediaPipe : distance 2D brute (unites image), sommee gauche+droite
  - Vicon     : distance 3D projetee sur les 3 plans (XY, XZ, YZ), sommee
                gauche+droite, evaluee aux frames correspondant a la
                timeline MediaPipe (appariement par plus-proche-voisin)

L'amplitude MediaPipe est ensuite mise a l'echelle en mm via le ratio des
largeurs d'epaules (Vicon / MediaPipe), deja calculees aux etapes 2 et 3.

Sortie : data/final/comparison_amplitude_long.xlsx
"""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config

VICON_CSV_DIR = Path(config.VICON_CSV_DIR)
POSE_DIR = Path(config.INTERMEDIATE_DIR) / "mediapipe_pose"
VICON_QOM_PATH = Path(config.INTERMEDIATE_DIR) / "vicon_qom_shoulder.xlsx"
MP_QOM_PATH = Path(config.INTERMEDIATE_DIR) / "mediapipe_qom_shoulder.xlsx"
OUTPUT_PATH = Path(config.FINAL_DIR) / "comparison_amplitude_long.xlsx"

CONDITIONS = ["SEATED", "SEMI", "STANDING"]

SUBJECT_TOKENS = {
    "P1": {"WR_D": "poignet_D", "WR_G": "poignet_G", "SH_D": "epaule_D", "SH_G": "epaule_G"},
    "P2": {"WR_D": "2poignet_D", "WR_G": "2poignet_G", "SH_D": "2epaule_D", "SH_G": "2epaule_G"},
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
    marker_row, axis_row = header_lines[2], header_lines[3]
    n = max(len(marker_row), len(axis_row))
    marker_row += [""] * (n - len(marker_row))
    axis_row += [""] * (n - len(axis_row))
    filled, last = [], ""
    for m in marker_row:
        m = (m or "").strip()
        last = m if m else last
        filled.append(last)
    colnames = []
    for m, a in zip(filled, axis_row):
        m, a = (m or "").strip(), (a or "").strip()
        if a in ["Frame", "Sub Frame"]:
            colnames.append(a)
        elif a in ["X", "Y", "Z"]:
            colnames.append(f"{m}_{a}")
        else:
            colnames.append(m if m else a)
    df = pd.read_csv(csv_path, skiprows=5, header=None, names=colnames, engine="python")
    return df.dropna(axis=1, how="all")


def find_xyz_cols(cols, token):
    pat = re.compile(rf"(?:^|:)\s*{re.escape(token)}_([XYZ])\b", re.IGNORECASE)
    found = {}
    for c in cols:
        m = pat.search(c.replace(" ", ""))
        if m:
            axis = m.group(1).upper()
            if axis not in found or len(c) < len(found[axis]):
                found[axis] = c
    return found.get("X"), found.get("Y"), found.get("Z")


def peak_to_peak_dist(w, ls, rs):
    """w, ls, rs : arrays (n, k). Retourne (min, max, peak-to-peak) de la
    distance poignet <-> centre des epaules."""
    C = (ls + rs) / 2.0
    d = np.sqrt(((w - C) ** 2).sum(axis=1))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return np.nan, np.nan, np.nan
    return float(d.min()), float(d.max()), float(d.max() - d.min())


def map_mp_to_vicon_idx(n_mp, n_vicon, fps_mp, fps_vicon):
    idx = np.rint(np.arange(n_mp) * (fps_vicon / fps_mp)).astype(int)
    return np.clip(idx, 0, n_vicon - 1)


def find_mp_pose_file(dyad, pid, condition):
    folder = POSE_DIR / dyad / pid / condition
    candidates = list(folder.glob("*_pose.xlsx"))
    return candidates[0] if len(candidates) == 1 else None


def vicon_wrist_shoulder_arrays(df_v, cols, tok, side):
    wtoken = tok[f"WR_{side}"]
    wX, wY, wZ = find_xyz_cols(cols, wtoken)
    dX, dY, dZ = find_xyz_cols(cols, tok["SH_D"])
    gX, gY, gZ = find_xyz_cols(cols, tok["SH_G"])
    if None in [wX, wY, wZ, dX, dY, dZ, gX, gY, gZ]:
        return None
    w = df_v[[wX, wY, wZ]].to_numpy(float)
    rs = df_v[[dX, dY, dZ]].to_numpy(float)
    ls = df_v[[gX, gY, gZ]].to_numpy(float)
    return w, ls, rs


def process_one_pair(csv_path, dyad, condition, pid):
    df_v = read_vicon_csv(csv_path)
    cols = list(df_v.columns)
    n_v = len(df_v)
    tok = SUBJECT_TOKENS[pid]

    mp_path = find_mp_pose_file(dyad, pid, condition)
    if mp_path is None:
        return {"dyad": dyad, "condition": condition, "pid": pid, "error": "fichier pose MP introuvable"}

    df_mp = pd.read_excel(mp_path)
    n_mp = len(df_mp)
    idx_v = map_mp_to_vicon_idx(n_mp, n_v, config.FS_MP_NOMINAL, config.FS_VICON)

    # --- MediaPipe (2D, unites brutes) ---
    ls_mp = df_mp[["LEFT_SHOULDER_x", "LEFT_SHOULDER_y"]].to_numpy(float)
    rs_mp = df_mp[["RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y"]].to_numpy(float)
    wR_mp = df_mp[["RIGHT_WRIST_x", "RIGHT_WRIST_y"]].to_numpy(float)
    wL_mp = df_mp[["LEFT_WRIST_x", "LEFT_WRIST_y"]].to_numpy(float)

    _, _, ppR_mp = peak_to_peak_dist(wR_mp, ls_mp, rs_mp)
    _, _, ppL_mp = peak_to_peak_dist(wL_mp, ls_mp, rs_mp)
    mp_amp_raw = ppR_mp + ppL_mp

    # --- Vicon (3D projete sur 3 plans, aux frames appariees a la timeline MP) ---
    arrays_D = vicon_wrist_shoulder_arrays(df_v, cols, tok, "D")
    arrays_G = vicon_wrist_shoulder_arrays(df_v, cols, tok, "G")
    if arrays_D is None or arrays_G is None:
        return {"dyad": dyad, "condition": condition, "pid": pid, "error": "colonnes Vicon manquantes"}

    wD, lsD, rsD = arrays_D
    wG, lsG, rsG = arrays_G
    wD, lsD, rsD = wD[idx_v], lsD[idx_v], rsD[idx_v]
    wG, lsG, rsG = wG[idx_v], lsG[idx_v], rsG[idx_v]

    plane_indices = {"XY": [0, 1], "XZ": [0, 2], "YZ": [1, 2]}
    row = {"dyad": dyad, "condition": condition, "pid": pid, "MP_ampWRISTS_pp_raw": mp_amp_raw}

    for plane, idx in plane_indices.items():
        _, _, ppD = peak_to_peak_dist(wD[:, idx], lsD[:, idx], rsD[:, idx])
        _, _, ppG = peak_to_peak_dist(wG[:, idx], lsG[:, idx], rsG[:, idx])
        row[f"VICON_{plane}_ampWRISTS_pp_mm"] = ppD + ppG

    return row


def main():
    rows = []
    for csv_path in sorted(VICON_CSV_DIR.rglob("*.csv")):
        condition, dyad = parse_vicon_filename(csv_path.stem)
        for pid in ["P1", "P2"]:
            rows.append(process_one_pair(csv_path, dyad, condition, pid))

    df = pd.DataFrame(rows)
    n_errors = df["error"].notna().sum() if "error" in df.columns else 0
    print(f"Lignes en erreur : {n_errors}/{len(df)}")
    if n_errors:
        print(df.loc[df["error"].notna(), ["dyad", "condition", "pid", "error"]])

    # Mise a l'echelle MP -> mm via les largeurs d'epaules deja calculees (etapes 2 et 3)
    vicon_sw = pd.read_excel(VICON_QOM_PATH)[["dyad", "condition", "pid", "shoulder_width_mm"]]
    mp_sw = pd.read_excel(MP_QOM_PATH)[["dyad", "condition", "pid", "shoulder_width_mp"]]

    df = df.merge(vicon_sw, on=["dyad", "condition", "pid"], how="left")
    df = df.merge(mp_sw, on=["dyad", "condition", "pid"], how="left")

    df["scale_mp_to_mm"] = df["shoulder_width_mm"] / df["shoulder_width_mp"]
    df["MP_ampWRISTS_pp_mm"] = df["MP_ampWRISTS_pp_raw"] * df["scale_mp_to_mm"]

    cols_final = ["dyad", "condition", "pid",
                  "MP_ampWRISTS_pp_mm",
                  "VICON_XY_ampWRISTS_pp_mm", "VICON_XZ_ampWRISTS_pp_mm", "VICON_YZ_ampWRISTS_pp_mm"]
    df_final = df[[c for c in cols_final if c in df.columns]]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_excel(OUTPUT_PATH, index=False)
    print(f"Sauvegarde : {OUTPUT_PATH} ({len(df_final)} lignes)")


if __name__ == "__main__":
    main()
