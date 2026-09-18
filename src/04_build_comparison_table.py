"""
04_build_comparison_table.py

Fusionne les sorties Vicon (etape 2) et MediaPipe (etape 3) sur
(dyad, condition, pid) pour produire la table finale de comparaison,
en format long : 120 lignes (20 dyades x 3 conditions x 2 participants).

Entree : data/intermediate/vicon_qom_shoulder.xlsx
         data/intermediate/mediapipe_qom_shoulder.xlsx
Sortie : data/final/comparison_qom_long.xlsx
"""

from pathlib import Path

import pandas as pd

import config

VICON_PATH = Path(config.INTERMEDIATE_DIR) / "vicon_qom_shoulder.xlsx"
MP_PATH = Path(config.INTERMEDIATE_DIR) / "mediapipe_qom_shoulder.xlsx"
OUTPUT_PATH = Path(config.FINAL_DIR) / "comparison_qom_long.xlsx"

KEYS = ["dyad", "condition", "pid"]


def main():
    vicon = pd.read_excel(VICON_PATH)
    mp = pd.read_excel(MP_PATH)

    print(f"Vicon     : {len(vicon)} lignes")
    print(f"MediaPipe : {len(mp)} lignes")

    # Fusion "outer" d'abord, pour detecter tout desaccord entre les deux cotes
    merged_outer = vicon.merge(mp, on=KEYS, how="outer", indicator=True)
    only_vicon = merged_outer[merged_outer["_merge"] == "left_only"]
    only_mp = merged_outer[merged_outer["_merge"] == "right_only"]

    if len(only_vicon) or len(only_mp) > 0:
        print(f"\nATTENTION : {len(only_vicon)} ligne(s) presente(s) uniquement cote Vicon :")
        print(only_vicon[KEYS])
        print(f"\nATTENTION : {len(only_mp)} ligne(s) presente(s) uniquement cote MediaPipe :")
        print(only_mp[KEYS])
    else:
        print("\nToutes les lignes correspondent des deux cotes (aucune ligne orpheline).")

    # Fusion finale (inner = seulement les lignes presentes des deux cotes)
    df = vicon.merge(mp, on=KEYS, how="inner")

    print(f"\nTable finale : {len(df)} lignes (attendu : 120)")

    cols = KEYS + [
        "shoulder_width_mm", "shoulder_width_mp",
        "QDM_WRIST_VICON_mm", "QDM_HEAD_VICON_mm",
        "QDM_WRIST_MP_raw", "QDM_HEAD_MP_raw",
        "QDM_WRIST_VICON_norm", "QDM_WRIST_MP_norm",
        "QDM_HEAD_VICON_norm", "QDM_HEAD_MP_norm",
    ]
    df = df[cols].rename(columns={"shoulder_width_mm": "shoulder_width_vicon_mm"})

    n_missing = df[["QDM_WRIST_VICON_norm", "QDM_WRIST_MP_norm",
                     "QDM_HEAD_VICON_norm", "QDM_HEAD_MP_norm"]].isna().any(axis=1).sum()
    print(f"Lignes avec une valeur normalisee manquante : {n_missing}/{len(df)}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_PATH, index=False)
    print(f"\nSauvegarde : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
