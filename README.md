# SPI+CBU - Remaining-Useful-Life prediction

Self-contained code, data of **every RUL figure and data table** in the manuscript. The method, **SPI+CBU**, is a statistical, online-updatable predictor: a sparse-prior-information (SPI) log-linear degradation model whose coefficients and population prior are learned by sparse Bayesian learning (SBL), updated online by a constrained Bayesian update (CBU) that keeps every prediction inside a historical-lives band via a posterior-predictive check (PPC), with the constraint enforced by Gaussian mutation (GM). The 90% intervals are recalibrated by split-conformal prediction. It is validated on 56 screening trajectories and, in triplicate, on a KCoHCF electrode (5000 cycles) and PePurease urease beads (300 cycles), and benchmarked against five baselines (SPI+TBU, Linear, Log-Bayes, Wiener process, Weibull degradation).


## Code components

| Step | Script | Output |
|---|---|---|
| 1 | `add_logbayes.py`   | adds the Log-Bayes baseline to the Task 2/3 per-cycle CSVs |
| 2 | `add_baselines.py`  | adds Wiener + Weibull baselines; prints the check that SPI+CBU has the lowest MAE in every case |
| 3 | `make_main.py`      | main **Fig. 7A, 7B** |
| 4 | `make_ED.py`        | **Extended Data Fig. 8, 9** |
| 5 | `make_SI.py`        | **Supplementary Figs 18-30** (GM panels, tracking, projection, strict-vs-soft) and **Extended Data Fig. 10** (averaged GM ablation) |
| 6 | `make_error_figs.py`| **absolute-error vs observed-fraction**, averaged (**Fig. 7C / Supplementary Fig. 25**) |
| 7 | `calib_per_task.py` | **calibration** reliability, averaged across tasks (raw vs conformal, +/-1 STD): **Supplementary Fig. 31** |
| 8 | `export_new_data.py`| `method_comparison_mae.csv`, error-vs-fraction CSVs |
| 9 | `export_metrics.py` | `per_panel_metrics.csv`, `per_task_metrics_summary.csv` (RMSE & MAE) |
| 10 | `export_source_data.py` | one tidy `SourceData_<FigID>.csv` per figure (Nature source-data requirement) |

## Outputs (created/updated under `./outputs/`)

```
outputs/
├── fig_main/        Fig7A, Fig7B, Fig7C                                   (png + svg)
├── fig_ED/          ExtendedDataFig8, ExtendedDataFig9, ExtendedDataFig10, forward_projection_log_perrep
├── fig_SI/          SupplementaryFig18-31, forward_projection_{KCoHCF,Urea}_log    (png + svg)
├── source_data/     SourceData_Fig7A/B/C, ExtendedDataFig8/9/10, SupplementaryFig18-31 (one CSV per figure)
├── additional_data/
      ├── method_comparison_mae.csv      RUL MAE, all tasks x all 6 methods
      ├── per_panel_metrics.csv          RMSE & MAE per panel per method (RUL + tracking)
      ├── per_task_metrics_summary.csv   the same, averaged per task
      ├── calibration_per_task.csv       per-task raw vs conformal coverage (+/-STD)
      ├── task1_curves.csv, task2_curves.csv, task3_curves.csv   per-cycle predictions, all 6 methods
      ├── task4_observed.csv, task4_projection_curve.csv, task4_projection_triplicate.csv
└── figs/, *coverage*.csv, gm_effect_*.csv, calibration.csv   precomputed result inputs
```

The output file names match the manuscript display items directly:

| File | Manuscript item |
|---|---|
| `fig_main/Fig7A` | Fig. 7A - short-term RUL validation (all tasks) |
| `fig_main/Fig7B` | Fig. 7B - capacity/activity forward projection |
| `fig_main/Fig7C` | Fig. 7C - absolute RUL error vs observed fraction (averaged) |
| `fig_ED/ExtendedDataFig8` | Extended Data Fig. 8 - RUL on the 56 screening trajectories |
| `fig_ED/ExtendedDataFig9` | Extended Data Fig. 9 - triplicate deployed-material RUL |
| `fig_ED/ExtendedDataFig10` | Extended Data Fig. 10 - GM ablation, averaged across tasks |
| `fig_SI/SupplementaryFig18` | RUL prediction to the 80% threshold (reaching set) |
| `fig_SI/SupplementaryFig19` | Capacity tracking to the 80% threshold (reaching set) |
| `fig_SI/SupplementaryFig20`, `21` | Forward projection to 80%, KCoHCF and PePurease (linear axis) |
| `fig_SI/SupplementaryFig22` | Capacity tracking across the 56 screening trajectories |
| `fig_SI/SupplementaryFig23`, `24` | Triplicate tracking, KCoHCF and PePurease |
| `fig_SI/SupplementaryFig25` | Absolute RUL error vs observed fraction (averaged; copy of Fig. 7C) |
| `fig_SI/SupplementaryFig26`, `27` | Per-trajectory GM effect, screening set |
| `fig_SI/SupplementaryFig28`, `29` | Per-replicate GM effect, KCoHCF and PePurease |
| `fig_SI/SupplementaryFig30` | Soft (triggered) vs strict GM |
| `fig_SI/SupplementaryFig31` | Conformal calibration reliability, averaged across tasks |

Source-data CSVs in `source_data/` carry the matching `SourceData_<item>.csv` name for every figure above.

## Files

| File | Role |
|---|---|
| `rul_style.py` | palette, plotting helpers, projection engine (imported by the `make_*` scripts) |
| `task3_triplicate.py`, `task4_triplicate.py` | core RUL engine (basis, SBL, online update, PPC, GM, projection) |
| `add_logbayes.py`, `add_baselines.py` | append the comparison baselines to the per-cycle CSVs |
| `make_main.py`, `make_ED.py`, `make_SI.py`, `make_error_figs.py`, `calib_per_task.py` | figure generators |
| `export_new_data.py`, `export_metrics.py`, `export_source_data.py` | data / metric / source-data tables |
| `export_all_results.py` | upstream: rebuilds the per-cycle result CSVs from raw data (see note below) |
| `Electrode_RUL_Colab.py` | complete original analysis notebook (full upstream pipeline, reference) |
| `Electrode_RUL_Data/` | raw cycling data (3 CSVs) |

## Method summary

- **SPI model.** g(t) is a sparse combination over a log-linear basis {1, log(1+t), t}; for the decelerating capacity/activity fade, g(t) = w0 + w1 log(1+t) + w2 t.
- **SBL.** Learns the coefficients and an informative population prior from historical run-to-failure data.
- **CBU + PPC.** Online prequential update (observe -> update -> predict -> repeat); the plausible RUL is constrained to a historical-lives band (10th-90th percentile of reference lives for the screen, min-max of the three replicates for each deployed material). The PPC index is the fraction of posterior particles whose predicted life falls inside the band.
- **GM.** When the PPC index drops below the trigger, Gaussian mutation relocates the band-violating particles toward the consistent ones (covariance scaled by 1/lambda^2, lambda = 3).
- **Calibration.** The 90% interval is recalibrated by split-conformal prediction so empirical coverage matches the nominal level.
- **Metrics.** RUL accuracy is reported as MAE (cycles); capacity/activity tracking as RMSE (%).

## Key results

- Screening set (56 trajectories): RUL MAE 6.3 cycles (SPI+CBU) vs 29.5 / 25.2 / 197 (SPI+TBU / Linear / Log-Bayes).
- Reaching-to-80% set (10 trajectories): RUL MAE 5.1 cycles, 93% interval coverage; baselines 19-73.
- Forward projection to 80%: KCoHCF electrode ~1.7-2.0 x 10^5 cycles (conservative floor ~7 x 10^3); PePurease beads ~1.3-2.3 x 10^3 cycles.
- SPI+CBU has the lowest RUL MAE in every task and case (printed by `add_baselines.py`; tabulated in `outputs/method_comparison_mae.csv`).

## Citation

```
A durable membrane-free platform for green hydrogen and high-purity ammonium recovery powered by human urine remediation.
├── DOI:
├── Duc Anh Nguyen#, Kim Anh Nguyen Thi, Taehun Kim, Hongrae Im, Am Jang* (#first-author, *corresponding author)
├── Department of Global Smart City, Sungkyunkwan University (SKKU), 2066, Seobu-ro, Jangan-gu, Suwon, Gyeonggi-do, 16419, Republic of Korea.
├── Contact information: nguyenducanh@g.skku.edu (+8210-2816-9711)
```

## Best regards

```
We know that we may have made mistakes. However, if you have read this far, you have already made our day.
We are deeply grateful for the support that has carried this study to this point, and we appreciate it more than words can express.
We sincerely hope to be given the opportunity to address your valuable questions and comments.
```
