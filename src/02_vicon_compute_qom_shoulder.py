"""
02_vicon_compute_qom_shoulder.py

Pour chaque CSV Vicon (un par dyade x condition) :
  - calcule la largeur d'epaule de P1 et P2
  - calcule le QoM (Quantity of Movement) des poignets et de la tete,
    avec filtrage Butterworth (zero-phase) applique aux trajectoires
    avant le calcul de deplacement frame-a-frame
  - normalise le QoM par la largeur d'epaule (ratio sans dimension)

Sortie : data/intermediate/vicon_qom_shoulder.xlsx, format long
         (une ligne par dyade x condition x participant, soit 120 lignes)
"""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import config

VICON_CSV_DIR = Path(config.VICON_CSV_DIR)
OUTPUT_PATH = Path(config.INTERMEDIATE_DIR) / "vicon_qom_shoulder.xlsx"

CONDITIONS = ["SEATED", "SEMI", "STANDING"]

# Tokens des marqueurs Vicon pour P1 et P2.
# Note : "2Temps_G" (sans le "e") est bien le nom exact du marqueur dans les
# CSV bruts pour P2 - c'est une coquille presente dans les donnees d'origine,
# pas une erreur de ce script. Ne pas "corriger" sans verifier le CSV brut.
SUBJECT_TOKENS = {
    "P1": {"WR_D": "poignet_D", "WR_G": "poignet_G",
           "TP_D": "Tempe_D", "TP_G": "Tempe_G",
           "SH_D": "epaule_D", "SH_G": "epaule_G"},
    "P2": {"WR_D": "2poignet_D", "WR_G": "2poignet_G",
           "TP_D": "2Tempe_D", "TP_G": "2Temps_G",
           "SH_D": "2epaule_D", "SH_G": "2epaule_G"},
}


# --------------------------
# Parsing du nom de fichier -> (condition, dyade)
# --------------------------
def parse_vicon_filename(stem: str):
    upper = stem.upper()
    for cond in CONDITIONS:
        if upper.startswith(cond):
            rest = stem[len(cond):].lstrip("_")
            return cond, rest  # rest = "D01", "D11", etc.
    raise ValueError(f"Condition non reconnue pour le fichier : {stem}")


# --------------------------
# Lecture d'un CSV Vicon
# --------------------------
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


# --------------------------
# Filtrage + QoM
# --------------------------
def butter_lowpass_filt(x, fs=config.FS_VICON, cutoff=config.CUTOFF_HZ, order=config.BUTTER_ORDER):
    x = np.asarray(x, dtype=float)
    if len(x) < (order * 3 + 1):
        return x
    nyq = 0.5 * fs
    w = min(cutoff / nyq, 0.99)
    b, a = butter(order, w, btype="low", analog=False)
    return filtfilt(b, a, x, method="pad")


def motion_quantity_point_filtered(df, X, Y, Z):
    arr = df[[X, Y, Z]].to_numpy(dtype=float)
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


def qdm_for_pair(df, cols, token_d, token_g):
    Xd, Yd, Zd = find_xyz_cols(cols, token_d)
    Xg, Yg, Zg = find_xyz_cols(cols, token_g)
    if None in [Xd, Yd, Zd, Xg, Yg, Zg]:
        return None
    q_d, _ = motion_quantity_point_filtered(df, Xd, Yd, Zd)
    q_g, _ = motion_quantity_point_filtered(df, Xg, Yg, Zg)
    return q_d + q_g


# --------------------------
# Traitement d'un CSV -> 2 lignes (P1, P2)
# --------------------------
def process_one_csv(csv_path: Path):
    condition, dyad = parse_vicon_filename(csv_path.stem)
    df = read_vicon_csv(csv_path)
    cols = list(df.columns)

    rows = []
    for pid, tok in SUBJECT_TOKENS.items():
        sw_mm = shoulder_width(df, cols, tok)
        qdm_wrist_mm = qdm_for_pair(df, cols, tok["WR_D"], tok["WR_G"])
        qdm_head_mm = qdm_for_pair(df, cols, tok["TP_D"], tok["TP_G"])

        row = {
            "dyad": dyad,
            "condition": condition,
            "pid": pid,
            "csv_file": csv_path.name,
            "shoulder_width_mm": sw_mm,
            "QDM_WRIST_VICON_mm": qdm_wrist_mm,
            "QDM_HEAD_VICON_mm": qdm_head_mm,
        }
        row["QDM_WRIST_VICON_norm"] = (
            qdm_wrist_mm / sw_mm if qdm_wrist_mm is not None and sw_mm else None
        )
        row["QDM_HEAD_VICON_norm"] = (
            qdm_head_mm / sw_mm if qdm_head_mm is not None and sw_mm else None
        )
        rows.append(row)

    return rows


def main():
    csv_files = sorted(VICON_CSV_DIR.rglob("*.csv"))
    print(f"{len(csv_files)} CSV Vicon trouve(s) dans {VICON_CSV_DIR}")

    all_rows = []
    for csv_path in csv_files:
        all_rows.extend(process_one_csv(csv_path))

    df_out = pd.DataFrame(all_rows)

    n_missing = df_out[["shoulder_width_mm", "QDM_WRIST_VICON_mm", "QDM_HEAD_VICON_mm"]].isna().any(axis=1).sum()
    print(f"Lignes avec au moins une valeur manquante : {n_missing}/{len(df_out)}")
    if n_missing:
        print(df_out.loc[
            df_out[["shoulder_width_mm", "QDM_WRIST_VICON_mm", "QDM_HEAD_VICON_mm"]].isna().any(axis=1),
            ["csv_file", "pid"]
        ])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(OUTPUT_PATH, index=False)
    print(f"Sauvegarde : {OUTPUT_PATH} ({len(df_out)} lignes)")


if __name__ == "__main__":
    main()
