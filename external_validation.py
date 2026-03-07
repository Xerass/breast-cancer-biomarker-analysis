"""
External Validation of the LASSO Breast Cancer Biomarker Signature
===================================================================

Validation dataset:  GSE42568  (104 breast cancer + 17 normal, 121 total)
Class balance:  IMBALANCED — 86 % tumor.  Standard accuracy is misleading;
see Sensitivity / Specificity / ROC-AUC for a reliable picture.

This script loads the trained model, scaler, and feature list exported by
lasso_pipeline_refactored.py (now trained on the balanced GSE15852 set),
then evaluates the signature on this independent imbalanced dataset.

Steps
-----
1. Load exported assets (model, scaler, feature list) via joblib.
2. Parse target labels from the GSE42568 !Sample_characteristics_ch1 row.
3. Load and transpose the expression matrix.
4. Strict feature alignment — subset, zero-fill missing, and reorder.
5. Scale using the TRAINING scaler only (.transform, never .fit_transform).
6. Evaluate — confusion matrix, ROC-AUC, Sensitivity, Specificity.
"""

# ── Imports ──────────────────────────────────────────────────────────────
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

# ── Configuration ────────────────────────────────────────────────────────
EXTERNAL_DATA_PATH = "data/GSE42568_series_matrix.txt"
EXPORT_DIR         = "exported_model"

# =========================================================================
# Step 1 — Load exported training assets
# =========================================================================
scaler   = joblib.load(os.path.join(EXPORT_DIR, "training_scaler.joblib"))
model    = joblib.load(os.path.join(EXPORT_DIR, "lasso_1se_model.joblib"))
features = joblib.load(os.path.join(EXPORT_DIR, "retained_features.joblib"))

print("=" * 70)
print("  LOADED TRAINING ARTIFACTS")
print("=" * 70)
print(f"  Model type         : {type(model).__name__}")
print(f"  Model alpha        : {model.alpha:.6f}")
print(f"  # retained probes  : {len(features)}")
print(f"  Scaler n_features  : {scaler.n_features_in_}")
print(f"  Retained features  : {features}")
print("=" * 70)
print()

# =========================================================================
# Step 2 — Extract labels (y_test) from the external GEO file
# =========================================================================
# GSE42568 stores tissue type in a !Sample_characteristics_ch1 row that
# contains "tissue:" entries.  We map:
#   "normal breast"  → 0
#   everything else  → 1  (breast cancer)
# -------------------------------------------------------------------------

y_test            = None
sample_ids        = None
matrix_start_line = None

with open(EXTERNAL_DATA_PATH, "r") as f:
    for line_num, line in enumerate(f, start=1):
        if line.startswith("!Sample_characteristics_ch1") and "tissue:" in line:
            fields = line.strip().split("\t")[1:]
            y_test = np.array(
                [0 if "normal breast" in v else 1 for v in fields],
                dtype=int,
            )
        if line.startswith("!Sample_geo_accession"):
            sample_ids = [
                v.strip().strip('"') for v in line.strip().split("\t")[1:]
            ]
        if line.startswith("!series_matrix_table_begin"):
            matrix_start_line = line_num
            break

# Sanity checks
if y_test is None:
    raise RuntimeError("Could not find the 'tissue:' row in the external file.")
if matrix_start_line is None:
    raise RuntimeError("Could not find !series_matrix_table_begin in the external file.")

n_normal = int((y_test == 0).sum())
n_tumor  = int((y_test == 1).sum())
print(f"External dataset labels parsed:  {len(y_test)} samples  "
      f"({n_normal} normal, {n_tumor} tumor)")
if n_tumor / len(y_test) > 0.7:
    print(f"  ⚠  Class imbalance detected: {n_tumor/len(y_test)*100:.0f}% tumor")
print()

# =========================================================================
# Step 3 — Load expression data (X_test)
# =========================================================================
# GEO series-matrix files are gene-major (rows = probes, cols = samples).
# We transpose so that rows = patients and columns = Probe_IDs to match
# scikit-learn conventions.
# -------------------------------------------------------------------------

expr = pd.read_csv(
    EXTERNAL_DATA_PATH,
    sep="\t",
    skiprows=matrix_start_line,
    index_col=0,
)

# Drop the trailing sentinel row
if expr.index[-1].startswith("!"):
    expr = expr.iloc[:-1]

X_test_df = expr.T
X_test_df = X_test_df.apply(pd.to_numeric, errors="coerce").fillna(0)

print(f"External expression matrix loaded: {X_test_df.shape}  (samples × probes)")
print()

# =========================================================================
# Step 4 — Strict feature alignment
# =========================================================================
# The external dataset may have a different set of Probe_IDs than the
# training set.  We must:
#   a) Keep ONLY the columns matching the retained feature list.
#   b) Zero-fill any retained features that are missing in the external data.
#   c) Reorder columns to exactly match the training feature order.
#
# This guarantees that X_test has the same shape and column semantics as
# the data the scaler and model were trained on.
# -------------------------------------------------------------------------

external_probes = set(X_test_df.columns)
retained_set    = set(features)

present  = retained_set & external_probes
missing  = retained_set - external_probes

print(f"Feature alignment:")
print(f"  Retained features     : {len(features)}")
print(f"  Found in external data: {len(present)}")
print(f"  Missing (zero-filled) : {len(missing)}")
if missing:
    print(f"    → {sorted(missing)}")
print()

# ── Important: we must provide the FULL scaler feature set, not just the
#    retained ones.  The training scaler was fit on ALL variance-filtered
#    features (~27 000+).  We first need to build a matrix with ALL those
#    columns (filling missing ones with 0) so that .transform() works.
#    Then we subset to the retained features AFTER scaling.
# -------------------------------------------------------------------------

# Get the full feature list the scaler was trained on
scaler_features = list(scaler.feature_names_in_) if hasattr(scaler, "feature_names_in_") else None

if scaler_features is not None:
    # Build a DataFrame aligned to ALL scaler features
    X_test_full = pd.DataFrame(0.0, index=X_test_df.index, columns=scaler_features)

    # Fill in values for probes that exist in both datasets.
    # We iterate column-by-column to avoid alignment issues between
    # the scaler-ordered and external-ordered DataFrames.
    common_full = list(set(scaler_features) & external_probes)
    for col in common_full:
        X_test_full[col] = X_test_df[col].values

    print(f"  Full scaler alignment : {len(scaler_features)} columns "
          f"({len(common_full)} found, {len(scaler_features) - len(common_full)} zero-filled)")
    print()

    # =====================================================================
    # Step 5 — Scale (NO data leakage)
    # =====================================================================
    # We use .transform() — NOT .fit_transform() — so the centering and
    # scaling parameters come exclusively from the training data.
    # -----------------------------------------------------------------
    X_test_scaled_full = scaler.transform(X_test_full)

    # Keep the full 27 338-column matrix — the Lasso model was fitted on
    # all variance-filtered features and expects the same input shape.
    # Features with zero coefficients are handled internally by the model.

else:
    # Fallback: if the scaler doesn't store feature_names_in_ (older sklearn),
    # we align only to the retained features and scale them.
    # This is slightly less precise but still valid.
    X_test_aligned = pd.DataFrame(0.0, index=X_test_df.index, columns=features)
    for col in present:
        X_test_aligned[col] = X_test_df[col].values

    # For this path we can't use the full scaler — we'd need to know all
    # the training features.  Instead, we manually centre and scale using
    # the scaler's mean_ and scale_ at the correct indices.
    # Since the scaler may have many more features, we find the indices.
    raise RuntimeError(
        "Your scikit-learn version does not store feature_names_in_ on "
        "StandardScaler.  Please upgrade to sklearn >= 1.0."
    )

print(f"Scaled test matrix shape: {X_test_scaled_full.shape}")
print()

# =========================================================================
# Step 6 — Evaluate
# =========================================================================
# The Lasso model outputs continuous predictions in [0, 1] (approximately).
# We threshold at 0.5 for binary classification.
# -------------------------------------------------------------------------

y_scores = model.predict(X_test_scaled_full)
y_pred   = (y_scores > 0.5).astype(int)

acc     = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_scores)
cm      = confusion_matrix(y_test, y_pred)

# Derive Sensitivity and Specificity from the confusion matrix
TN, FP, FN, TP = cm.ravel()
sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0   # recall for Tumor
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0   # recall for Normal

print("=" * 70)
print("  EXTERNAL VALIDATION RESULTS  (GSE42568 — imbalanced)")
print("=" * 70)
print()
print(f"  Accuracy    : {acc:.4f}   (⚠ may be inflated by class imbalance)")
print(f"  ROC-AUC     : {roc_auc:.4f}")
print(f"  Sensitivity : {sensitivity:.4f}   (Tumor recall = TP / [TP + FN])")
print(f"  Specificity : {specificity:.4f}   (Normal recall = TN / [TN + FP])")
print()
print("  Classification Report:")
print("  " + "-" * 60)
print(
    classification_report(
        y_test, y_pred,
        target_names=["Normal", "Tumor"],
        zero_division=0,
    )
)

print("  Confusion Matrix:")
print("  " + "-" * 60)
print(f"                  Predicted Normal   Predicted Tumor")
print(f"  Actual Normal         {TN:>5}             {FP:>5}")
print(f"  Actual Tumor          {FN:>5}             {TP:>5}")
print()
print("=" * 70)
