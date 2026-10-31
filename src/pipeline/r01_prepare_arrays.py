"""
r01_prepare_arrays.py

Converts each recording (participant x condition) into one compact .npz file:
  - Vicon: all markers of interest (mm, 100 Hz, NaN = gap)
  - MediaPipe: 33 landmarks (normalised x, y, z, visibility) on the FULL video
    timeline (one row per video frame, NaN when no pose was detected), plus
    image width/height and video frame rate.

The _pose.xlsx files only contain frames in which a pose was detected. Each row
is therefore placed back on its video frame number ('frame' column), so that
MediaPipe and Vicon are matched by frame number and not by row index.
Requires ffprobe (FFmpeg) to read video metadata.
"""
import csv, re, subprocess, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[2]
VICON = ROOT / "data/raw/vicon_csv"
VIDEOS = ROOT / "data/raw/videos"
POSE = ROOT / "data/intermediate/mediapipe_pose"
OUT = ROOT / "data/intermediate/pipeline/arrays"
OUT.mkdir(parents=True, exist_ok=True)
CONDS = ["SEATED", "SEMI", "STANDING"]
MARKERS = ["Tempe_D", "Tempe_G", "AR_Tete_D", "AR_Tete_G", "epaule_D", "epaule_G",
           "coude_D", "coude_G", "poignet_D", "poignet_G", "majeur_D", "majeur_G",
           "entre_homoplate", "hanche_D", "hanche_G", "genou_D", "genou_G",
           "cheville_D", "cheville_G"]
P2_ALIAS = {"Tempe_G": "Temps_G"}  # marker label typo in the raw CSV files for P2


def read_vicon(path):
    """Reads both Vicon CSV export formats present in the data
    (5 header lines 'Frame, Sub Frame' + a units line, or 4 header lines
    'Field #' without Sub Frame or units: dyads 13 to 15)."""
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        rd = csv.reader(f)
        rows = [next(rd) for _ in range(6)]
    ax_i = next(i for i, r in enumerate(rows) if r and r[0].strip() in ("Frame", "Field #"))
    mk, ax = rows[ax_i - 1], rows[ax_i]
    skip = ax_i + 1 + (1 if any(c.strip() == "mm" for c in rows[ax_i + 1]) else 0)
    names, last = [], ""
    mk = mk + [""] * (len(ax) - len(mk))
    for m, a in zip(mk, ax):
        last = m.strip() or last
        a = a.strip()
        if a in ("X", "Y", "Z"):
            names.append(f"{last.split(':')[-1]}_{a}")
        else:
            names.append("Frame" if a in ("Frame", "Field #") else (a or f"_c{len(names)}"))
    df = pd.read_csv(path, skiprows=skip, header=None, engine="c", low_memory=False)
    df = df.iloc[:, :len(names)]
    df.columns = names[:df.shape[1]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df[df["Frame"].notna()].reset_index(drop=True)
    return df


def vicon_marker(df, name):
    cols = [f"{name}_{a}" for a in "XYZ"]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].to_numpy(float)


def video_meta(path):
    o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
                        "-of", "json", str(path)], capture_output=True, text=True).stdout
    s = json.loads(o)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    rot = 0
    for sd in s.get("side_data_list", []) or []:
        if "rotation" in sd: rot = int(sd["rotation"])
    if "tags" in s and "rotate" in s["tags"]: rot = int(s["tags"]["rotate"])
    w, h = int(s["width"]), int(s["height"])
    if abs(rot) in (90, 270): w, h = h, w
    return w, h, float(num) / float(den), int(s.get("nb_frames", 0)), rot


def main():
    log = []
    for vpath in sorted(VICON.glob("*.csv")):
        stem = vpath.stem
        cond = next(c for c in CONDS if stem.upper().startswith(c))
        dyad = stem[len(cond):]
        dv = read_vicon(vpath)
        for pid in ["P1", "P2"]:
            out = OUT / f"{dyad}_{cond}_{pid}.npz"
            if out.exists():
                try:
                    np.load(out)["MP"]; continue
                except Exception:
                    pass  # incomplete file (interrupted write): regenerate it
            pre = "" if pid == "P1" else "2"
            vic = {}
            for m in MARKERS:
                name = pre + (P2_ALIAS.get(m, m) if pid == "P2" else m)
                arr = vicon_marker(dv, name)
                if arr is None and pid == "P2":
                    arr = vicon_marker(dv, pre + m)
                vic[m] = arr if arr is not None else np.full((len(dv), 3), np.nan)
            vids = list((VIDEOS / dyad / pid / cond).glob("*.mp4"))
            poses = list((POSE / dyad / pid / cond).glob("*_pose.xlsx"))
            if len(vids) != 1 or len(poses) != 1:
                log.append(f"{dyad} {cond} {pid}: video/pose file not found"); continue
            W, H, fps, nb, rot = video_meta(vids[0])
            dp = pd.read_excel(poses[0], engine="calamine")
            n = int(max(nb, dp["frame"].max() + 1))
            lm_names = sorted({c[:-2] for c in dp.columns if c.endswith("_x")})
            mp = np.full((n, len(lm_names), 4), np.nan)
            fr = dp["frame"].to_numpy(int)
            for j, l in enumerate(lm_names):
                for k, s in enumerate("xyzv"):
                    mp[fr, j, k] = dp[f"{l}_{s}"].to_numpy(float)
            np.savez_compressed(out, **{f"V_{k}": v for k, v in vic.items()},
                                MP=mp, MP_names=np.array(lm_names), W=W, H=H, fps=fps, rot=rot,
                                n_vicon=len(dv), n_video=n, n_detected=len(dp))
            log.append(f"{dyad} {cond} {pid}: vicon={len(dv)} video={n} detect={len(dp)} ({100*len(dp)/n:.1f}%) {W}x{H}@{fps:g} rot={rot}")
            print(log[-1], flush=True)
    (OUT.parent / "r01_log.txt").write_text("\n".join(log))


if __name__ == "__main__":
    main()
