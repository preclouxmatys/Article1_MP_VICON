"""
03_mediapipe_compute_qom_shoulder.py

Pour chaque fichier de landmarks MediaPipe (_pose.xlsx, produit par l'etape 1) :
  - calcule la largeur d'epaule (distance 2D LEFT_SHOULDER / RIGHT_SHOULDER)
  - filtre (Butterworth zero-phase) les trajectoires 2D des poignets et du nez
  - calcule le QoM (deplacement cumule frame-a-frame) pour les poignets (G+D)
    et pour la tete (nez uniquement, cote MediaPipe)
  - normalise par la largeur d'epaule (ratio sans dimension)

Entree : data/intermediate/mediapipe_pose/D../P./condition/*_pose.xlsx
Sortie : data/intermediate/mediapipe_qom_shoulder.xlsx, format long
         (une ligne par dyade x condition x participant, soit 120 lignes)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

import config

POSE_DIR = Path(config.INTERMEDIATE_DIR) / "mediapipe_pose"
OUTPUT_PATH = Path(config.INTERMEDIATE_DIR) / "mediapipe_qom_shoulder.xlsx"


def butter_lowpass_filt(x, fs, cutoff=config.CUTOFF_HZ, order=config.BUTTER_ORDER):
    x = np.asarray(x, dtype=float)
    if len(x) < (order * 3 + 1):
        return x
    nyq = 0.5 * fs
    w = min(cutoff / nyq, 0.99)
    b, a = butter(order, w, btype="low", analog=False)
    return filtfilt(b, a, x, method="pad")


def qdm_from_xy(df, prefix, fs):
    """QoM 2D (deplacement cumule) pour un landmark MediaPipe donne (ex: 'NOSE', 'LEFT_WRIST')."""
    x = butter_lowpass_filt(df[f"{prefix}_x"].to_numpy(dtype=float), fs)
    y = butter_lowpass_filt(df[f"{prefix}_y"].to_numpy(dtype=float), fs)
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    d = d[np.isfinite(d)]
    return float(d.sum())


def shoulder_width_mp(df):
    sw = np.sqrt(
        (df["LEFT_SHOULDER_x"] - df["RIGHT_SHOULDER_x"]) ** 2
        + (df["LEFT_SHOULDER_y"] - df["RIGHT_SHOULDER_y"]) ** 2
    )
    sw = sw.replace([np.inf, -np.inf], np.nan).dropna()
    return float(sw.median()) if len(sw) else None


def estimate_fps(df):
    """Estime la frequence d'echantillonnage reelle a partir des timestamps (plus fiable
    que la metadonnee fps de la video, qui peut etre imprecise en VFR)."""
    dt_ms = df["t_ms"].diff().dropna()
    dt_ms = dt_ms[dt_ms > 0]
    if len(dt_ms) == 0:
        return config.FS_MP_NOMINAL
    return 1000.0 / dt_ms.median()


def process_one_pose_file(pose_path: Path, dyad: str, pid: str, condition: str):
    df = pd.read_excel(pose_path)

    needed = [
        "LEFT_SHOULDER_x", "LEFT_SHOULDER_y", "RIGHT_SHOULDER_x", "RIGHT_SHOULDER_y",
        "LEFT_WRIST_x", "LEFT_WRIST_y", "RIGHT_WRIST_x", "RIGHT_WRIST_y",
        "NOSE_x", "NOSE_y",
    ]
    if not all(c in df.columns for c in needed):
        return {"dyad": dyad, "condition": condition, "pid": pid, "error": "colonnes manquantes"}

    fs = estimate_fps(df)
    sw = shoulder_width_mp(df)

    qdm_left_wrist = qdm_from_xy(df, "LEFT_WRIST", fs)
    qdm_right_wrist = qdm_from_xy(df, "RIGHT_WRIST", fs)
    qdm_wrist_raw = qdm_left_wrist + qdm_right_wrist
    qdm_head_raw = qdm_from_xy(df, "NOSE", fs)

    return {
        "dyad": dyad,
        "condition": condition,
        "pid": pid,
        "fs_est": fs,
        "shoulder_width_mp": sw,
        "QDM_WRIST_MP_raw": qdm_wrist_raw,
        "QDM_HEAD_MP_raw": qdm_head_raw,
        "QDM_WRIST_MP_norm": qdm_wrist_raw / sw if sw else None,
        "QDM_HEAD_MP_norm": qdm_head_raw / sw if sw else None,
    }


def main():
    pose_files = sorted(POSE_DIR.rglob("*_pose.xlsx"))
    print(f"{len(pose_files)} fichier(s) pose trouve(s) dans {POSE_DIR}")

    rows = []
    for f in pose_files:
        rel = f.relative_to(POSE_DIR)
        parts = rel.parts  # ("D01", "P1", "SEATED", "xxx_pose.xlsx")
        if len(parts) < 3:
            print(f"Ignore (structure de dossier inattendue) : {rel}")
            continue
        dyad, pid, condition = parts[0], parts[1], parts[2]
        rows.append(process_one_pose_file(f, dyad, pid, condition))

    df_out = pd.DataFrame(rows)

    n_errors = df_out["error"].notna().sum() if "error" in df_out.columns else 0
    print(f"Lignes en erreur (colonnes manquantes) : {n_errors}/{len(df_out)}")
    if n_errors:
        print(df_out.loc[df_out["error"].notna(), ["dyad", "pid", "condition"]])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(OUTPUT_PATH, index=False)
    print(f"Sauvegarde : {OUTPUT_PATH} ({len(df_out)} lignes)")


if __name__ == "__main__":
    main()
