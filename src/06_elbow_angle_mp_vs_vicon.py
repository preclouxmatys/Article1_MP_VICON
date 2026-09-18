"""
06_elbow_angle_mp_vs_vicon.py

Angle au coude (epaule-coude-poignet), calcule dans un repere local
centre sur les epaules :
  - MediaPipe : 2D (unique, l'image est deja en 2D)
  - Vicon VI3 : 3D complet, non projete (reference "haute fidelite")
  - Vicon VI2 : projete sur chacun des 3 plans (XY, XZ, YZ), pour tester
                si le choix du plan de projection explique les ecarts

Sortie : data/final/comparison_elbow_angle_long.xlsx
         (une ligne par dyade x condition x participant x plan de projection)
"""

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

import config

VICON_CSV_DIR = Path(config.VICON_CSV_DIR)
POSE_DIR = Path(config.INTERMEDIATE_DIR) / "mediapipe_pose"
OUTPUT_PATH = Path(config.FINAL_DIR) / "comparison_elbow_angle_long.xlsx"

CONDITIONS = ["SEATED", "SEMI", "STANDING"]
PLANES = ["XY", "XZ", "YZ"]


# --------------------------
# Lecture / parsing (identique aux etapes precedentes)
# --------------------------
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
        m = pat.search(str(c).replace(" ", ""))
        if m:
            axis = m.group(1).upper()
            if axis not in found or len(str(c)) < len(str(found[axis])):
                found[axis] = c
    return found.get("X"), found.get("Y"), found.get("Z")


def find_mp_pose_file(dyad, pid, condition):
    folder = POSE_DIR / dyad / pid / condition
    candidates = list(folder.glob("*_pose.xlsx"))
    return candidates[0] if len(candidates) == 1 else None


def map_mp_to_vicon_idx(n_mp, n_vicon, fps_mp, fps_vicon):
    idx = np.rint(np.arange(n_mp) * (fps_vicon / fps_mp)).astype(int)
    return np.clip(idx, 0, n_vicon - 1)


# --------------------------
# Geometrie
# --------------------------
def pick_plane(arr3, plane):
    idx = {"XY": [0, 1], "XZ": [0, 2], "YZ": [1, 2]}[plane]
    return arr3[:, idx]


def angle_at_joint(A, B, C):
    BA, BC = A - B, C - B
    num = np.sum(BA * BC, axis=1)
    den = np.linalg.norm(BA, axis=1) * np.linalg.norm(BC, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = num / den
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.degrees(np.arccos(cosang))
    ang[~np.isfinite(ang)] = np.nan
    return ang


def corr_rmse(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return np.nan, np.nan
    xx, yy = x[m], y[m]
    return float(np.corrcoef(xx, yy)[0, 1]), float(np.sqrt(np.mean((xx - yy) ** 2)))


def _norm2(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = np.nan
    return v / n


def build_shoulder_frame_2d(Ls, Rs):
    C = (Ls + Rs) / 2.0
    u = _norm2(Rs - Ls)
    v = np.stack([-u[:, 1], u[:, 0]], axis=1)
    return C, u, v


def project_to_frame_2d(P, C, u, v):
    Q = P - C
    return np.stack([np.sum(Q * u, axis=1), np.sum(Q * v, axis=1)], axis=1)


def mp_triplet_in_shoulder_frame(df_pose, side):
    required = [
        "LEFT_SHOULDER_x", "LEFT_SHOULDER_y", "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y",
        f"{side}_WRIST_x", f"{side}_WRIST_y", f"{side}_ELBOW_x", f"{side}_ELBOW_y",
        f"{side}_SHOULDER_x", f"{side}_SHOULDER_y",
    ]
    if any(c not in df_pose.columns for c in required):
        return None
    Ls = df_pose[["LEFT_SHOULDER_x", "LEFT_SHOULDER_y"]].to_numpy(float)
    Rs = df_pose[["RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y"]].to_numpy(float)
    C, u, v = build_shoulder_frame_2d(Ls, Rs)
    W = df_pose[[f"{side}_WRIST_x", f"{side}_WRIST_y"]].to_numpy(float)
    E = df_pose[[f"{side}_ELBOW_x", f"{side}_ELBOW_y"]].to_numpy(float)
    S = df_pose[[f"{side}_SHOULDER_x", f"{side}_SHOULDER_y"]].to_numpy(float)
    return project_to_frame_2d(W, C, u, v), project_to_frame_2d(E, C, u, v), project_to_frame_2d(S, C, u, v)


def vicon_triplet(df_v, cols, pid, side):
    if pid == "P1":
        wtoken, etoken, stoken = f"poignet_{side}", f"coude_{side}", f"epaule_{side}"
    else:
        wtoken, stoken = f"2poignet_{side}", f"2epaule_{side}"
        etoken = None
        for cand in [f"2coude_{side}", f"2coudes_{side}", f"2elbow_{side}", f"elbow2_{side}"]:
            if None not in find_xyz_cols(cols, cand):
                etoken = cand
                break
        etoken = etoken or f"2coude_{side}"

    wX, wY, wZ = find_xyz_cols(cols, wtoken)
    eX, eY, eZ = find_xyz_cols(cols, etoken)
    sX, sY, sZ = find_xyz_cols(cols, stoken)
    if None in [wX, wY, wZ, eX, eY, eZ, sX, sY, sZ]:
        return None
    W = df_v[[wX, wY, wZ]].to_numpy(float)
    E = df_v[[eX, eY, eZ]].to_numpy(float)
    S = df_v[[sX, sY, sZ]].to_numpy(float)
    return W, E, S


def vicon_shoulders_3d(df_v, cols, pid):
    tok_L, tok_R = ("epaule_G", "epaule_D") if pid == "P1" else ("2epaule_G", "2epaule_D")
    LX, LY, LZ = find_xyz_cols(cols, tok_L)
    RX, RY, RZ = find_xyz_cols(cols, tok_R)
    if None in [LX, LY, LZ, RX, RY, RZ]:
        return None
    return df_v[[LX, LY, LZ]].to_numpy(float), df_v[[RX, RY, RZ]].to_numpy(float)


def vicon_triplet_2d(df_v, cols, pid, side, plane):
    trip = vicon_triplet(df_v, cols, pid, side)
    sh = vicon_shoulders_3d(df_v, cols, pid)
    if trip is None or sh is None:
        return None
    W3, E3, S3 = trip
    L3, R3 = sh
    W2, E2, S2 = pick_plane(W3, plane), pick_plane(E3, plane), pick_plane(S3, plane)
    L2, R2 = pick_plane(L3, plane), pick_plane(R3, plane)
    C, u, v = build_shoulder_frame_2d(L2, R2)
    return project_to_frame_2d(W2, C, u, v), project_to_frame_2d(E2, C, u, v), project_to_frame_2d(S2, C, u, v)


# --------------------------
# Traitement d'une paire (csv, pid)
# --------------------------
def process_one_pair(csv_path, dyad, condition, pid):
    df_v = read_vicon_csv(csv_path)
    cols = list(df_v.columns)
    n_v = len(df_v)

    mp_path = find_mp_pose_file(dyad, pid, condition)
    if mp_path is None:
        return [{"dyad": dyad, "condition": condition, "pid": pid, "plane": p,
                  "error": "fichier pose MP introuvable"} for p in PLANES]

    df_mp = pd.read_excel(mp_path)
    n_mp = len(df_mp)
    idx_v = map_mp_to_vicon_idx(n_mp, n_v, config.FS_MP_NOMINAL, config.FS_VICON)

    mpR = mp_triplet_in_shoulder_frame(df_mp, "RIGHT")
    mpL = mp_triplet_in_shoulder_frame(df_mp, "LEFT")
    tripR = vicon_triplet(df_v, cols, pid, "D")
    tripL = vicon_triplet(df_v, cols, pid, "G")

    if mpR is None or mpL is None or tripR is None or tripL is None:
        return [{"dyad": dyad, "condition": condition, "pid": pid, "plane": p,
                  "error": "colonnes manquantes (MP ou Vicon)"} for p in PLANES]

    ang_mp_R = angle_at_joint(mpR[2], mpR[1], mpR[0])
    ang_mp_L = angle_at_joint(mpL[2], mpL[1], mpL[0])

    W3R, E3R, S3R = [a[idx_v] for a in tripR]
    W3L, E3L, S3L = [a[idx_v] for a in tripL]
    ang_vi3_R = angle_at_joint(S3R, E3R, W3R)
    ang_vi3_L = angle_at_joint(S3L, E3L, W3L)
    corr_vi3_R, rmse_vi3_R = corr_rmse(ang_mp_R, ang_vi3_R)
    corr_vi3_L, rmse_vi3_L = corr_rmse(ang_mp_L, ang_vi3_L)

    rows = []
    for plane in PLANES:
        vi2R = vicon_triplet_2d(df_v, cols, pid, "D", plane)
        vi2L = vicon_triplet_2d(df_v, cols, pid, "G", plane)
        row = {"dyad": dyad, "condition": condition, "pid": pid, "plane": plane,
               "corr_MP_vs_VI3_R": corr_vi3_R, "rmse_MP_vs_VI3_R": rmse_vi3_R,
               "corr_MP_vs_VI3_L": corr_vi3_L, "rmse_MP_vs_VI3_L": rmse_vi3_L}
        if vi2R is None or vi2L is None:
            row["error"] = "colonnes Vicon manquantes (repere epaule 2D)"
        else:
            W2R, E2R, S2R = [a[idx_v] for a in vi2R]
            W2L, E2L, S2L = [a[idx_v] for a in vi2L]
            ang_vi2_R = angle_at_joint(S2R, E2R, W2R)
            ang_vi2_L = angle_at_joint(S2L, E2L, W2L)
            row["corr_MP_vs_VI2_R"], row["rmse_MP_vs_VI2_R"] = corr_rmse(ang_mp_R, ang_vi2_R)
            row["corr_MP_vs_VI2_L"], row["rmse_MP_vs_VI2_L"] = corr_rmse(ang_mp_L, ang_vi2_L)
            row["error"] = None
        rows.append(row)
    return rows


def main():
    all_rows = []
    for csv_path in sorted(VICON_CSV_DIR.rglob("*.csv")):
        condition, dyad = parse_vicon_filename(csv_path.stem)
        for pid in ["P1", "P2"]:
            all_rows.extend(process_one_pair(csv_path, dyad, condition, pid))

    df = pd.DataFrame(all_rows)
    n_errors = df["error"].notna().sum() if "error" in df.columns else 0
    print(f"Lignes en erreur : {n_errors}/{len(df)} (attendu : 0 une fois les 120 videos MP traitees)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"Sauvegarde : {OUTPUT_PATH} ({len(df)} lignes)")


if __name__ == "__main__":
    main()
