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

Frame correspondence between MediaPipe and Vicon (100 Hz) is established using each
recording's own estimated frame rate (`estimate_fps()` in scripts 05 and 06), not a
fixed nominal value — actual MediaPipe throughput is closer to 25 fps for most
recordings despite a nominal target of 30 fps, and this matters in particular for the
frame-by-frame elbow-angle comparison. `config.FS_MP_NOMINAL` is used only as a
fallback when the estimation itself fails.

## Main results

**Global movement (Quantity of Movement, wrist-width-normalized units)**

| Segment       | n   | r     | R^2   | ICC(2,1) | Permutation p |
|---------------|-----|-------|-------|----------|---------------|
| QoM Wrist     | 120 | .929  | .862  | .859     | < .001        |
| QoM Head      | 120 | .894  | .799  | .885     | < .001        |

QoM Head uses the midpoint of Vicon's two bilateral temple markers as a single point
(`qdm_head_midpoint` in `02_vicon_compute_qom_shoulder.py`), matching the single NOSE
landmark used on the MediaPipe side; an earlier version that summed the two temple
markers separately produced a much lower ICC due to a scale mismatch, not a genuine
validity difference.

**Wrist peak-to-peak amplitude (mm, MediaPipe vs Vicon projected onto each anatomical plane)**

| Plane | n   | r    | R^2  | ICC(2,1) |
|-------|-----|------|------|----------|
| XY    | 120 | .539 | .290 | .143     |
| XZ    | 120 | .737 | .543 | .397     |
| YZ    | 120 | .609 | .371 | .244     |

**Elbow angle (bilateral mean per recording, r Fisher-z–averaged across the 120 recordings x 2 sides)**

| Reference Vicon | n   | r moyen | RMSE (deg) |
|------------------|-----|---------|------------|
| VI3 (raw 3D)     | 120 | .808    | 37.2       |
| XZ (projected)   | 120 | .804    | 36.9       |
| YZ (projected)   | 120 | .768    | 30.4       |
| XY (projected)   | 120 | .142    | 71.8       |

A strong correlation for the elbow angle reflects good temporal correspondence
(MediaPipe and Vicon rise and fall together), not necessarily an accurate absolute
angle: RMSE stays substantial (30–37°) even in the best-performing planes. Note that
`07_validity_stats.R` currently reports the *arithmetic* mean of per-recording
correlations in its `angle_summary` output; the Fisher z-average shown above (the
statistically appropriate way to aggregate correlation coefficients) is computed as a
downstream step from the same per-recording values in
`data/final/comparison_elbow_angle_long.xlsx`, and is not yet built into the R script.

Full results (including Bland-Altman bias/limits-of-agreement, proportional-bias
regression, and dyad-cluster bootstrap confidence intervals) are in
`data/final/validity_results_all.xlsx`, with figures in `data/final/figures/`.

## Known limitations

- MediaPipe wrist amplitude is scaled to millimeters via a per-participant
  shoulder-width ratio; it is not a direct metric measurement.
- The permutation-test robustness check repeated within each single posture condition
  (`08_permutation_test.py`) uses n = 40 participants (20 dyads) per condition — these
  are 40 distinct people, but not 40 statistically independent observations, since the
  two members of a dyad remain non-independent of each other. Confidence intervals for
  r and ICC(2,1) that account for this dyad-level clustering (dyad-cluster bootstrap,
  5,000 resamples) are reported alongside the classical analytic ones in the article.
- Classical analytic confidence intervals (r, ICC) assume independent observations;
  the 120 participant x condition rows are not fully independent (20 dyads x 2
  participants x 3 conditions), which the permutation test and dyad-cluster bootstrap
  are both designed to address.

## Citation

*To be completed upon publication.*

## Contact

Matys Precloux — PhD candidate, EuroMov, University of Montpellier.
