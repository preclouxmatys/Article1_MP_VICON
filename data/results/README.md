# Results

All files are produced by the scripts in `src/pipeline/`. Labels inside the tables and
figures are in French, as in the manuscript. MP = MediaPipe.

## Tables and their place in the article

| File | Content | Article |
|---|---|---|
| `T_main_validity.csv` | Main comparison (no calibration): wrist QoM, head QoM, wrist amplitude, in shoulder widths | Table 2, Figures 2–4 |
| `T_by_condition.csv` | Same comparison per posture | Table 3 |
| `T_rmcorr_mixed.csv` | Repeated-measures correlation (and mixed model) across postures | Table 3 |
| `T_elbow.csv` | Elbow angle, per arm (near / far) and per comparison | Table 4, Tables S4–S5, Figure 5 |
| `T_world_landmarks.csv` | MediaPipe world landmarks vs Vicon 3D | Table 5 |
| `T_sensitivity.csv`, `T_planes.csv` | Sensitivity analyses (normalised coordinates, Vicon at 100 Hz, max–min amplitude, laboratory planes) | Table S1, Figure S1 |
| `T_image_plane_validity.csv` | MediaPipe vs Vicon projected into the image (valid calibrations) | Table S2, Figure S3 |
| `T_elbow_by_condition.csv` | Elbow angle (MP 2D vs Vicon 3D) per posture | Table S3 |
| `T_descriptives.csv` | Means (SD) per posture | — |
| `T_residual_analysis.csv` | Optimal cut-off frequency per recording (Winter residual analysis) | Methods (filtering) |
| `key_numbers.json` | Other numbers reported in the text (lags, calibration, detection rate, permutation test, …) | Methods, Results, S2–S3 |
| `per_recording_results.csv` | All variables, one line per recording (120 lines) | — |
| `elbow_per_recording_side.csv` | Elbow-angle agreement, one line per recording and arm (240 lines) | Figure 5 |
| `world_landmarks_subset.csv` | World-landmark variables for the 20-recording subsample | Table 5 |
| `fig6_example.json` | Recording shown in Figure 6 | Figure 6 |

## Columns of the agreement tables

`n` number of recordings; `mean_MP`, `sd_MP`, `mean_ref`, `sd_ref` means and SDs;
`r` Pearson correlation with analytic 95% CI (`r_lo`, `r_hi`) and dyad-cluster bootstrap CI
(`r_boot_lo`, `r_boot_hi`); `ICC` = ICC(A,1) with its 95% CI (`ICC_lo`, `ICC_hi`) and
bootstrap CI (`ICC_boot_lo`, `ICC_boot_hi`); `bias` and `bias_pct` mean difference
(MP − reference) in units and in % of the reference mean; `RMSE`; `LoA_lo`, `LoA_hi`
limits of agreement of the raw differences; `ratio_gmean`, `ratio_LoA_lo`, `ratio_LoA_hi`
geometric mean ratio MP / reference and its 95% limits of agreement (log-ratio
Bland-Altman); `BA_slope`, `BA_slope_p` proportional bias (regression of the log-ratio on
the log geometric mean).

Elbow tables: `r_fisher` mean of per-recording correlations after Fisher z transform,
with bootstrap CI (`r_boot_lo`, `r_boot_hi`); `r_median`, `r_q1`, `r_q3`; `RMSE_*`, `bias_*`,
`sdd_*` (variable error = SD of the differences) in degrees. Comparisons: `MP vs V3D`
(MediaPipe 2D vs Vicon 3D), `VIC vs V3D` (Vicon projected into the image vs Vicon 3D),
`MP vs VIC` (MediaPipe vs projected Vicon), `MPvis` (frames with visibility > 0.5 only),
`MPnorm` (angle computed in MediaPipe normalised coordinates), `MP vs plan XY/XZ/YZ`
(Vicon angle projected on a laboratory plane).

## Column names in `per_recording_results.csv`

Identification: `dyad`, `condition` (SEATED, SEMI, STANDING), `pid` (P1, P2),
`video_file_pid` (video actually used, see the D14 swap), `W`, `H`, `fps`.

Variables follow the pattern `<VARIABLE>_<SEGMENT>_<SYSTEM>_<UNIT>`:

- `QOM` path length summed over both wrists (`WRIST`) or of the head point (`HEAD`;
  the head point is the temple midpoint, so `QOM_HEAD_*` and `QOM_HEAD_tempmid_*` are the same).
- `AMP_p95p5` / `AMP_maxmin` wrist-to-shoulder-centre distance range (P95–P5 or max–min),
  summed over both wrists.
- Systems: `MP` MediaPipe (pixels); `MPnormcoord` MediaPipe in normalised coordinates;
  `VIC` Vicon projected into the image (pixels); `V3D25` / `V3D` Vicon 3D at the video
  frame rate; `V3D100` Vicon 3D at 100 Hz; `VXY`, `VXZ`, `VYZ` Vicon on a laboratory plane.
- Units: `_px` pixels, `_mm` millimetres, `_norm` divided by the shoulder width of the same
  system; `_mm_shoulderscale` MediaPipe pixels converted with the shoulder-width scale.
- `QOMwin_*_r` correlation of QoM computed in 1-s windows; `MP_jitter_*_px` MediaPipe noise
  (median frame-to-frame step when the Vicon point is almost still).
- `ELB_<D|G>_<A>_vs_<B>_<stat>` elbow-angle agreement for the right (D) or left (G) arm,
  `stat` = `r`, `rmse`, `bias`, `sdd`, `n`; `vis_ok_pct_<D|G>` % of frames with visibility > 0.5.

Synchronisation and calibration: `lag_s` final lag (s), `lag_head_xcorr_s` head-based lag,
`lag_xcorr_s` wrist-based lag, `lag_for_head_s` lag used for head time series,
`xcorr_peak`, `head_peak` cross-correlation peaks; `camera_model` (projective or affine),
`reproj_calib_*_px` calibration reprojection error, `reproj_holdout_*_px` reprojection
error on the arms (not used for calibration), `px_per_mm` image scale at the participant;
`view_azimuth_vs_partner_deg`, `view_elevation_deg` camera direction; `near_side` arm
nearest to the camera (D or G); `interpersonal_dist_mm`; `sw3_med_mm`, `sw_mp_px_med`,
`sw_vicproj_px_med` shoulder widths; `vicon_labelfix_frames` Vicon frames corrected for
label swaps.
