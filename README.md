# MediaPipe vs Vicon — Concurrent Validity for Movement Dynamics in Spontaneous Dyadic Conversation

This repository contains the full, reproducible analysis pipeline for the concurrent
validity study comparing MediaPipe BlazePose (markerless, monocular RGB) against a
Vicon marker-based motion capture system (gold standard), for quantifying global
movement dynamics (wrist and head Quantity of Movement, wrist amplitude, elbow angle)
during spontaneous face-to-face dyadic conversation.

Twenty dyads (N = 40 participants) were recorded simultaneously with both systems
across three postural conditions (seated, semi-standing, standing), giving 120
participant x condition observations.

## Repository structure

```
Article1_MP_VICON/
├── src/
│   ├── config.py                          # paths and processing parameters
│   ├── 01_mediapipe_extract_landmarks.py  # raw video -> MediaPipe Pose landmarks
│   ├── 02_vicon_compute_qom_shoulder.py   # Vicon CSV -> QoM (wrist, head) + shoulder width
│   ├── 03_mediapipe_compute_qom_shoulder.py # MediaPipe pose -> QoM (wrist, head) + shoulder width
│   ├── 04_build_comparison_table.py       # merges Vicon + MediaPipe QoM into the final long-format table
│   ├── 05_amplitude_mp_vs_vicon.py        # wrist peak-to-peak amplitude, MediaPipe vs Vicon (3 projection planes)
│   ├── 06_elbow_angle_mp_vs_vicon.py      # elbow angle, MediaPipe (2D) vs Vicon (3D and projected 2D, 3 planes)
│   ├── 07_validity_stats.R                # r, R^2, ICC(2,1), RMSE, bias, Bland-Altman, figures
│   └── 08_permutation_test.py             # dyad x condition constrained permutation test
├── data/
│   ├── raw/            # NOT included in this repo (see Data availability below)
│   ├── intermediate/   # generated automatically, not versioned
│   └── final/           # final comparison tables, statistics, and figures (versioned)
├── requirements.txt
└── README.md
```

## Setup

The pipeline requires Python 3.11 (MediaPipe's legacy Solutions API, used here, is not
available in more recent MediaPipe releases) and R (>= 4.0) for the statistics step.

```bash
conda create -n syncogest python=3.11 -y
conda activate syncogest
pip install -r requirements.txt
```

R packages required for `07_validity_stats.R`: `readxl`, `dplyr`, `tidyr`, `ggplot2`,
`psych`, `writexl`, `scales`, `here`.

## Data availability

Raw video recordings and Vicon marker trajectories are not included in this
repository, as they contain identifiable participant data collected under ethics
approval (CER-UM, avis n° UM 2025-101bis). Researchers wishing to reproduce this
analysis with their own equivalent data should place it as follows before running
the pipeline:

data/raw/videos/D<dyad>/P<participant>/<condition>/*.mp4
data/raw/vicon_csv/<condition>D<dyad>.csv


where `<condition>` is one of `SEATED`, `SEMI`, `STANDING`.

## Reproducing the analysis

Run the scripts in order from the `src/` directory:

```bash
conda activate syncogest
cd src

python3 01_mediapipe_extract_landmarks.py   # ~2-4h for 120 videos
python3 02_vicon_compute_qom_shoulder.py
python3 03_mediapipe_compute_qom_shoulder.py
python3 04_build_comparison_table.py
python3 05_amplitude_mp_vs_vicon.py
python3 06_elbow_angle_mp_vs_vicon.py
Rscript 07_validity_stats.R
python3 08_permutation_test.py
```

Each script prints diagnostic checks (row counts, missing values) so that any
mismatch between the two systems' outputs is caught immediately rather than
silently propagating downstream.

## Main results

| Segment       | n   | r    | R^2  | ICC(2,1) | Permutation p |
|---------------|-----|------|------|----------|---------------|
| QoM Wrist     | 120 | .923 | .853 | .842     | < .001        |
| QoM Head      | 120 | .879 | .772 | .260     | < .001        |

Full results (amplitude, elbow angle, Bland-Altman statistics) are in
`data/final/validity_results_all.xlsx`, with figures in `data/final/figures/`.

## Known limitations

- Head QoM for MediaPipe relies on a single landmark (nose), versus two bilateral
  temple markers for Vicon — a genuine anatomical-correspondence limitation.
- MediaPipe wrist amplitude is scaled to millimeters via a per-participant
  shoulder-width ratio; it is not a direct metric measurement.
- The three postural conditions are non-independent within participant (repeated
  measures); the permutation test corrects for the dyad-level non-independence
  between the two participants of a pair, and a single-condition robustness check
  (n = 40, fully independent participants) is provided in `08_permutation_test.py`.

## Citation

*To be completed upon publication.*

## Contact

Matys Precloux — PhD candidate, EuroMov, University of Montpellier.

# Article1_MP_VICON
