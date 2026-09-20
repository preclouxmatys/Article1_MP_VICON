"""
08_permutation_test.py

Test de permutation contraint par dyade x condition (label-swap du cote
Vicon a l'interieur de chaque groupe de 2 [P1, P2]), pour le QoM poignet
et tete. Complete par un controle de robustesse sur une seule condition
posturale (n=40, participants strictement independants).
"""

from pathlib import Path

import numpy as np
import pandas as pd

import config

QOM_PATH = Path(config.FINAL_DIR) / "comparison_qom_long.xlsx"


def corr_and_perm(df, mp_col, vicon_col, group_cols=("dyad", "condition"),
                   n_perm=config.N_PERMUTATIONS, seed=config.RANDOM_SEED):
    d = df[[*group_cols, mp_col, vicon_col]].dropna().reset_index(drop=True)
    x = d[mp_col].to_numpy()
    y = d[vicon_col].to_numpy()

    r_obs = np.corrcoef(x, y)[0, 1]

    groups = d.groupby(list(group_cols)).indices
    rng = np.random.default_rng(seed)

    perm_rs = np.empty(n_perm)
    for i in range(n_perm):
        y_shuf = y.copy()
        for idx in groups.values():
            idx = np.asarray(idx)
            y_shuf[idx] = y[rng.permutation(idx)]
        perm_rs[i] = np.corrcoef(x, y_shuf)[0, 1]

    p_value = (np.sum(np.abs(perm_rs) >= np.abs(r_obs)) + 1) / (n_perm + 1)
    return r_obs, p_value, perm_rs, len(d)


def report(label, df, mp_col, vicon_col):
    r_obs, p_value, perm_rs, n = corr_and_perm(df, mp_col, vicon_col)
    print(f"\n{label} (n={n})")
    print(f"  r observe        = {r_obs:.4f}")
    print(f"  R2               = {r_obs**2:.4f}")
    print(f"  p (permutation)  = {p_value:.6f}")
    print(f"  null: moyenne={perm_rs.mean():.4f}, sd={perm_rs.std():.4f}")
    return r_obs, p_value


def main():
    df = pd.read_excel(QOM_PATH)

    print("=== Analyse principale (120 observations, toutes conditions poolees) ===")
    report("QoM Poignet", df, "QDM_WRIST_MP_norm", "QDM_WRIST_VICON_norm")
    report("QoM Tete", df, "QDM_HEAD_MP_norm", "QDM_HEAD_VICON_norm")

    print("\n=== Controle de robustesse : une seule condition (n=40, participants independants) ===")
    for condition in ["SEATED", "SEMI", "STANDING"]:
        df_one = df[df["condition"] == condition].reset_index(drop=True)
        print(f"\n--- Condition : {condition} ---")
        report("QoM Poignet", df_one, "QDM_WRIST_MP_norm", "QDM_WRIST_VICON_norm")
        report("QoM Tete", df_one, "QDM_HEAD_MP_norm", "QDM_HEAD_VICON_norm")


if __name__ == "__main__":
    main()
