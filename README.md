# Breast Cancer Biomarker Discovery via LASSO Regression

A machine learning project that identifies gene biomarkers capable of distinguishing breast cancer tissue from normal breast tissue using L1-regularised (LASSO) regression on microarray transcriptomic data

## Project Overview

This project trains a LASSO model on the **GSE15852** dataset to select a compact panel of discriminatory gene probes, then validates the model's generalisability on an independent external dataset (**GSE42568**).

## Repository Structure

| File / Directory | Description |
|---|---|
| `analysis.ipynb` | **Main analysis notebook** training pipeline from raw GEO data to a validated LASSO biomarker panel |
| `externalvalidation.ipynb` | **External validation notebook** loads the exported model and evaluates it on an unseen dataset |
| `data/` | Raw GEO series-matrix files used by the notebooks |
| `exported_model/` | Saved model artifacts (scaler, Lasso model, feature list) produced by the analysis notebook |

## Key Files in Detail

### `analysis.ipynb` — Training Pipeline

This notebook performs end-to-end biomarker discovery on **GSE15852** (43 cancer + 43 normal, Affymetrix HG-U133A):

1. **Data Parsing & Alignment** — Extracts sample labels (`Normal` / `Tumor`) and the expression matrix from the GEO series-matrix file, then transposes to sample-major format (86 samples × 22 283 probes).
2. **Log2 Transform & ComBat Batch Correction** — Applies a `log2(x+1)` transform to stabilise distributions, detects a latent batch effect via PCA, and removes it with **pyComBat** while preserving the biological (tissue-type) signal.
3. **Unsupervised Variance Pre-Filter** — Removes the bottom 50 % of probes by variance (an unsupervised filter that never examines labels), reducing the feature space to ~11 142 probes.
4. **Standardisation** — Fits a `StandardScaler` on the filtered matrix so LASSO regularisation penalises features on the same scale.
5. **PCA Diagnostic** — Verifies that the post-processing data still requires ~21 components for 80 % variance (unchanged from earlier runs), confirming discriminatory signal is preserved.
6. **LassoCV with 1-SE Rule** — Runs 10-fold cross-validated LASSO to choose `alpha_min`, then applies a 1-standard-error rule to select a larger, more parsimonious `alpha`. This yields **15 non-zero probes**, the final biomarker panel.
7. **In-Sample Evaluation** — Reports R² ≈ 0.73, training accuracy ≈ 98.8 %, and a per-class classification report (precision/recall/F1).
8. **Model Export** — Saves the scaler, the LASSO model, and the list of retained feature names to `exported_model/` via `joblib`.
9. **Probe-ID → Gene Mapping** — Maps the 15 retained Affymetrix probe IDs to human gene symbols using the GPL96 annotation file (e.g., `RGS1`, `CD24`, `KRT19`, `LPL`, `S100B`, …).

### `externalvalidation.ipynb` — External Validation

This notebook evaluates the trained model on **GSE42568** (104 cancer + 17 normal, Affymetrix HG-U133 Plus 2.0 / GPL570):

1. **Load Training Artifacts** — Reads the saved scaler, LASSO model, and retained feature list from `exported_model/`.
2. **Parse External Dataset** — Extracts labels and the expression matrix from the GSE42568 series-matrix file (121 samples × 54 675 probes).
3. **Log2 Scale Guard** — Applies the same rule as training (`log2(x+1)` only if max > 50). GSE42568 is already log2 (max ≈ 16), so no transform is applied.
4. **Feature Alignment** — Builds a matrix aligned to all scaler features (11 142 columns), filling any missing probes with 0. All 15 retained probes are found in the external data.
5. **Evaluation** — Scores the frozen LASSO model two ways and applies a 0.5 decision threshold:
   - **Option A (primary):** standardises the 15 panel probes with GSE42568's own per-probe mean/SD (no labels used), removing the MAS5 vs RMA baseline offset.
   - **Option B:** the original approach, the *training* scaler (`.transform()`) on the log-aligned data. The panel probes still sit 1–5.5 log2 units away from the training means here.

| Metric | Option A (per-dataset) | Option B (training scaler) |
|---|---|---|
| Accuracy | 75.2 % | 97.5 % |
| ROC-AUC | 0.880 | 0.901 |
| Sensitivity (Tumor recall) | 74.0 % | 100 % |
| Specificity (Normal recall) | 82.4 % | 82.4 % |
| Confusion matrix (TN/FP/FN/TP) | 14/3/27/77 | 14/3/0/104 |

Ranking quality holds up across both (AUC ≈ 0.88–0.90), so the panel does separate tumour from normal tissue on an independent cohort. The fixed 0.5 threshold does not transfer cleanly between platforms, though. Option B's 100 % sensitivity is inflated by the normalisation offset. Option A is conservative on sensitivity because centring on a 86 %-tumour cohort shifts tumour scores towards the threshold.

## Datasets

| GEO Accession | Role | Platform | Samples |
|---|---|---|---|
| GSE15852 | Training | GPL96 (HG-U133A) | 43 Normal + 43 Tumor |
| GSE42568 | External Validation | GPL570 (HG-U133 Plus 2.0) | 17 Normal + 104 Tumor |

> GPL570 is a superset of GPL96, so the 15 biomarker probes are present in both platforms.
>
> GSE15852 is raw MAS5 intensity (max ≈ 19 000) while GSE42568 is already log2 RMA (max ≈ 16). The validation notebook applies the same log2 guard as training (no-op here) and then standardises the panel probes per dataset (Option A) to remove the remaining MAS5 vs RMA offset.

## Requirements

- Python ≥ 3.10
- NumPy, Pandas, Matplotlib, scikit-learn, joblib
- [pyComBat](https://github.com/epigenelabs/pyComBat) (`combat` package)

## Quick Start

```bash
# 1. Run the training pipeline
jupyter notebook analysis.ipynb

# 2. Run external validation
jupyter notebook externalvalidation.ipynb
```
