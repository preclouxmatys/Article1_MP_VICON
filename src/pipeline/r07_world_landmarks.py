"""r07_world_landmarks.py: MediaPipe extraction (same version 0.10.14,
model_complexity=1) keeping the pose_world_landmarks (metric 3D coordinates,
origin at the hip centre), on a balanced subsample of 20 recordings (one
recording per dyad; conditions and participants in rotation; the list is in
data/results/world_landmarks_subset.csv).
Usage: python r07_world_landmarks.py D01_SEATED_P1 [...]   (--part=k/K splits long videos)"""
import sys, cv2, numpy as np, mediapipe as mp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/intermediate/pipeline/world"; OUT.mkdir(parents=True, exist_ok=True)
SWAP = {("D14", "SEMI"): {"P1": "P2", "P2": "P1"}}
part = [a for a in sys.argv[1:] if a.startswith("--part=")]
part = tuple(map(int, part[0][7:].split("/"))) if part else None  # e.g. --part=1/2 (long videos)
for rec in [a for a in sys.argv[1:] if not a.startswith("--")]:
    dy, co, pi = rec.split("_"); piv = SWAP.get((dy, co), {}).get(pi, pi)
    suffix = f"_part{part[0]}" if part else ""
    if (OUT / f"{rec}{suffix}.npz").exists() or (OUT / f"{rec}.npz").exists():
        continue
    v = list((ROOT / "data/raw/videos" / dy / piv / co).glob("*.mp4"))[0]
    cap = cv2.VideoCapture(str(v)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = np.full((n + 50, 33, 4), np.nan); i = 0; stop = n + 50
    if part:
        k, K = part; i = (k - 1) * n // K; stop = k * n // K
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    with mp.solutions.pose.Pose(model_complexity=1) as pose:
        while True:
            ok, f = cap.read()
            if not ok or i >= stop:
                break
            r = pose.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if r.pose_world_landmarks:
                W[i] = [[l.x, l.y, l.z, l.visibility] for l in r.pose_world_landmarks.landmark]
            i += 1
    if part:
        np.savez_compressed(OUT / f"{rec}{suffix}.npz", W=W[:i], start=(part[0] - 1) * n // part[1], stop=i)
        parts = sorted(OUT.glob(f"{rec}_part*.npz"))
        if len(parts) == part[1]:
            Wall = np.full((n + 50, 33, 4), np.nan); last = 0
            for pf in parts:
                d = np.load(pf); a, b = int(d["start"]), int(d["stop"]); Wall[a:b] = d["W"][a:b]; last = max(last, b)
            np.savez_compressed(OUT / f"{rec}.npz", W=Wall[:last])
    else:
        np.savez_compressed(OUT / f"{rec}.npz", W=W[:i])
    print(rec, i, flush=True)
