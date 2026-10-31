"""
01_mediapipe_extract_landmarks.py

Extracts the 33 MediaPipe Pose landmarks from each raw video, frame by frame,
and saves one Excel file per video.

Input  : config.VIDEO_DIR (raw videos, organised by dyad / participant / condition)
Output : data/intermediate/mediapipe_pose/<same subfolders>/<name>_pose.xlsx
"""

import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import config

# --------------------------
# Parameters
# --------------------------
VIDEO_DIR = Path(config.VIDEO_DIR)
POSE_OUTPUT_DIR = Path(config.INTERMEDIATE_DIR) / "mediapipe_pose"

MODEL_COMPLEXITY = 1
FORCE_REPROCESS = False   # True = reprocess even if the output file already exists
MAX_VIDEOS = None         # set a small number (e.g. 2) to test before processing all 120 videos

mp_pose = mp.solutions.pose
POSE_LANDMARKS = [lm.name for lm in mp_pose.PoseLandmark]


def process_video(video_path: Path, output_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []

    with mp_pose.Pose(model_complexity=MODEL_COMPLEXITY) as model:
        for i in tqdm(range(nframes), desc=video_path.name, leave=False):
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = model.process(rgb)

            if res.pose_landmarks:
                data = {"frame": i, "t_ms": (i / fps) * 1000}
                for j, lm in enumerate(res.pose_landmarks.landmark):
                    name = POSE_LANDMARKS[j]
                    data[f"{name}_x"] = lm.x
                    data[f"{name}_y"] = lm.y
                    data[f"{name}_z"] = lm.z
                    data[f"{name}_v"] = getattr(lm, "visibility", 0)
                rows.append(data)

    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_excel(output_path, index=False)
    print(f"  -> {len(rows)} frames saved to {output_path}")


def main():
    videos = sorted(
        v for v in VIDEO_DIR.rglob("*")
        if v.suffix.lower() in [".mp4", ".mov", ".avi"]
    )
    print(f"{len(videos)} video(s) found in {VIDEO_DIR}")

    if MAX_VIDEOS is not None:
        videos = videos[:MAX_VIDEOS]
        print(f"MAX_VIDEOS set -> processing limited to {len(videos)} video(s)")

    n_done, n_skipped = 0, 0
    for video_path in videos:
        relative = video_path.relative_to(VIDEO_DIR)
        output_path = POSE_OUTPUT_DIR / relative.with_name(f"{video_path.stem}_pose.xlsx")

        if output_path.exists() and not FORCE_REPROCESS:
            print(f"Already processed, skipped: {relative}")
            n_skipped += 1
            continue

        print(f"Processing: {relative}")
        process_video(video_path, output_path)
        n_done += 1

    print(f"\nDone. {n_done} video(s) processed, {n_skipped} skipped (already present).")


if __name__ == "__main__":
    main()
