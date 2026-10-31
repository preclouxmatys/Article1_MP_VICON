# MediaPipe vs Vicon during spontaneous dyadic conversation

Analysis code for the article *Measuring movement in conversation with a single camera:
a comparison of MediaPipe and Vicon* (Precloux, Guyot, Pla & Marin).

Forty participants (20 dyads) conversed face-to-face in three postures (seated,
semi-standing, standing). Each participant was recorded at the same time by a Vicon
system (100 Hz) and by a video camera placed at a three-quarter view, processed with
MediaPipe Pose 0.10.14. The pipeline compares the two systems on wrist and head
quantity of movement (QoM), wrist amplitude and elbow angle.

## Repository content

```
src/
├── config.py                         paths and acquisition parameters
├── 01_mediapipe_extract_landmarks.py video -> MediaPipe Pose landmarks
└── revision/                         analysis pipeline of the article
    ├── r01_prepare_arrays.py         Vicon CSV + MediaPipe landmarks -> one .npz per recording
    ├── r01b_extra_markers.py         extra Vicon markers (toes; tested for calibration, not retained)
    ├── r02_audit.py                  data quality audit; wrist-based time lag
    ├── r02b_head_sync.py             head-based time lag (nose vs temple midpoint)
    ├── r03_match_check.py            checks that each video matches its Vicon file
    ├── r04_process.py                synchronisation, filtering, camera calibration, all variables
    ├── r05_stats.py                  agreement statistics (tables of the article)
    ├── r07_figures.py                figures of the article
    ├── r08_world_landmarks.py        MediaPipe 3D world landmarks (20-recording subsample)
    ├── r09_world_compare.py          world landmarks vs Vicon 3D
    └── r10_residual_analysis.py      Winter residual analysis (filter cut-off)
data/final/revision/                  result tables (.csv), figures, key numbers
requirements.txt
```

## Data availability

Videos and Vicon trajectories are not shared: they contain identifiable participant data
(ethics approval CER-UM, UM 2025-101bis). The result tables in `data/final/revision/`
(one line per recording or per arm) are provided. To run the pipeline on equivalent data,
place the raw files as follows:

```
data/raw/videos/D<dyad>/P<participant>/<condition>/*.mp4
data/raw/vicon_csv/<condition>D<dyad>.csv        <condition> = SEATED, SEMI or STANDING
```

## Setup

Python 3.11.

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
cd src
python 01_mediapipe_extract_landmarks.py     # several hours for 120 videos

cd revision
python r01_prepare_arrays.py
python r01b_extra_markers.py
python r02_audit.py
python r02b_head_sync.py
python r03_match_check.py                     # optional check
python r04_process.py                         # --force to recompute existing recordings
python r05_stats.py
python r07_figures.py
python r08_world_landmarks.py <recordings>    # e.g. D01_SEATED_P1 ; list in world_landmarks_subset.csv
python r09_world_compare.py
python r10_residual_analysis.py
```

Intermediate files are written to `data/intermediate/` (not versioned). Tables, figures
and `key_numbers.json` are written to `data/final/revision/`.

## Main methodological choices

- **Pixels.** MediaPipe normalised coordinates (x / width, y / height) are converted to
  pixels before any distance or angle is computed.
- **Filtering.** 4th-order zero-lag Butterworth low-pass filter at 6 Hz for both systems
  (checked with a residual analysis, `r10`).
- **Head.** The MediaPipe nose is compared with the midpoint of the two Vicon temple markers.
- **Synchronisation.** No common trigger. The lag is estimated by cross-correlating head
  speed in both systems, then refined by aligning shoulders and ankles. Arm analyses are
  therefore never synchronised on the arms; head time-series analyses use the wrist-based lag.
- **Primary comparison (no calibration).** QoM and amplitude are divided by the shoulder
  width measured by the same system: MediaPipe in pixels, Vicon in millimetres. No MediaPipe
  data is used to build the Vicon reference.
- **Image-plane comparison (interpretation only).** Each camera is calibrated from Vicon
  shoulder and ankle markers matched to MediaPipe landmarks (DLT, or affine model when
  unstable). The Vicon is then projected into the image. Calibration is checked without
  MediaPipe; 5 failed calibrations are excluded from this comparison only. The calibration
  also defines the arm nearest to the camera ("near" arm).
- **Data checks.** One video swap between partners (D14, semi-standing) is corrected in
  `r04_process.py` (`VIDEO_SWAP`). Two Vicon CSV export formats are handled in `r01`.

## Citation

To be completed upon publication.

## Contact

Matys Precloux, EuroMov Digital Health in Motion, Univ Montpellier, IMT Mines Alès —
matys.precloux@umontpellier.fr
