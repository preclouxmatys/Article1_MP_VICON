# MediaPipe vs Vicon during spontaneous dyadic conversation

Analysis code for the article *Measuring movement in conversation with a single camera:
a comparison of MediaPipe and Vicon* (Precloux, Guyot, Pla & Marin).

Forty participants (20 dyads) conversed face-to-face in three postures (seated,
semi-standing, standing), giving 120 recordings (participant x posture). Each participant
was recorded at the same time by a Vicon system (8 cameras, 100 Hz) and by a video camera
placed behind the partner (three-quarter view), processed with MediaPipe Pose 0.10.14.
The pipeline compares the two systems on wrist and head quantity of movement (QoM),
wrist amplitude and elbow angle.

## Repository content

```
src/
├── config.py                          paths used by script 01
├── 01_mediapipe_extract_landmarks.py  video -> MediaPipe Pose landmarks (one .xlsx per video)
└── pipeline/
    ├── r01_prepare_arrays.py          Vicon CSV + MediaPipe landmarks -> one .npz per recording
    ├── r02_audit.py                   data quality checks; wrist-based time lag
    ├── r02b_head_sync.py              head-based time lag (MediaPipe nose vs Vicon temple midpoint)
    ├── r03_match_check.py             checks which Vicon file matches a given video
    ├── r04_process.py                 synchronisation, filtering, camera calibration, all variables
    ├── r05_stats.py                   agreement statistics (tables of the article)
    ├── r06_figures.py                 figures of the article
    ├── r07_world_landmarks.py         MediaPipe 3D world landmarks (20-recording subsample)
    ├── r08_world_compare.py           world landmarks vs Vicon 3D
    └── r09_residual_analysis.py       Winter residual analysis (filter cut-off)
data/results/                          result tables (.csv), figures, key numbers
                                       (see data/results/README.md)
```

## Data availability

Videos and Vicon trajectories are not shared: they contain identifiable participant data
(ethics approval CER-UM, UM 2025-101bis). All results are provided in `data/results/`,
including one line per recording (`per_recording_results.csv`). To run the pipeline on
equivalent data, place the raw files as follows:

```
data/raw/videos/D<dyad>/P<participant>/<condition>/*.mp4
data/raw/vicon_csv/<condition>D<dyad>.csv          <condition> = SEATED, SEMI or STANDING
```

Vicon marker labels expected in the CSV files: `Tempe_D/G` (temples), `AR_Tete_D/G`,
`epaule_D/G` (acromion), `coude_D/G` (lateral epicondyle), `poignet_D/G` (wrist),
`majeur_D/G`, `entre_homoplate`, `cheville_D/G` (D = right, G = left; partner P2 labels
are prefixed with `2`).

## Setup

Python 3.10 (MediaPipe 0.10.14, legacy Solutions API) and FFmpeg (`ffprobe`, used by `r01`).

```bash
pip install -r requirements.txt
```

## Running the pipeline

```bash
cd src
python 01_mediapipe_extract_landmarks.py   # several hours for 120 videos

cd pipeline
python r01_prepare_arrays.py
python r02_audit.py
python r02b_head_sync.py
python r03_match_check.py D14_SEMI_P1       # optional check (see "Data checks" below)
python r04_process.py                       # --force to recompute existing recordings
python r05_stats.py
python r06_figures.py
python r07_world_landmarks.py <recordings>  # the 20 recordings listed in data/results/world_landmarks_subset.csv
python r08_world_compare.py
python r09_residual_analysis.py
```

Intermediate files are written to `data/intermediate/` (not versioned). Tables, figures and
`key_numbers.json` are written to `data/results/`. Random seeds are fixed (bootstrap and
permutation tests), so the tables are reproduced exactly.

## Main methodological choices

- **Pixels.** MediaPipe normalised coordinates (x / width, y / height) are converted to
  pixels before any distance or angle is computed.
- **Filtering.** 4th-order zero-lag Butterworth low-pass filter at 6 Hz for both systems
  (checked with a residual analysis, `r09`).
- **Head.** The MediaPipe nose is compared with the midpoint of the two Vicon temple markers.
- **Synchronisation.** No common trigger. The lag is estimated by cross-correlating head
  speed in both systems, then refined by aligning shoulders and ankles. Arm analyses are
  therefore never synchronised on the arms; head time-series analyses use the wrist-based lag.
- **Main comparison (no calibration).** QoM and amplitude are divided by the shoulder width
  measured by the same system: MediaPipe in pixels, Vicon in millimetres. No MediaPipe data
  is used to build the Vicon reference.
- **Image-plane comparison (interpretation only).** Each camera is calibrated from Vicon
  shoulder and ankle markers matched to MediaPipe landmarks (DLT, or affine model when
  unstable); the Vicon markers are then projected into the image. Each calibration is
  checked without MediaPipe (ratio of projected to scaled 3D wrist amplitude > 3 = failure);
  5 failed calibrations are excluded from this comparison only. The calibration also gives
  the camera viewing direction, which defines the arm nearest to the camera ("near" arm).
- **Data checks.** The two partners' videos are swapped for D14 semi-standing (found with
  `r03`, corrected in `r02b`, `r04` and `r07`). Two Vicon CSV export formats are handled in `r01`.

## License

MIT (see `LICENSE`).

## Citation

To be completed upon publication.

## Contact

Matys Precloux, EuroMov Digital Health in Motion, Univ Montpellier, IMT Mines Alès —
matys.precloux@umontpellier.fr
