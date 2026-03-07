"""
Breast Cancer Biomarker Discovery via LASSO Regression — Refactored Pipeline
=============================================================================

Training dataset:  GSE15852  (43 breast cancer + 43 normal breast tissue)
Samples:  86 total  (perfectly balanced 50/50 class split)

We train on the balanced external dataset to avoid the "always guess
tumor" trap that arises from the original imbalanced GSE42568 set.

Pipeline improvements (unchanged from the previous version):
1. Unsupervised variance pre-filtering  (drop bottom 50 % of probes)
2. 10-fold cross-validation instead of 5-fold
3. Dynamic SE-multiplier fallback for 1-SE alpha selection
"""

# ── Imports ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV, Lasso
from sklearn.metrics import accuracy_score, classification_report
import joblib
from combat.pycombat import pycombat

# ── Configuration ────────────────────────────────────────────────────────
DATA_PATH = "data/GSE15852_series_matrix.txt"

# =========================================================================
# Section 1 — Parse metadata  (GSE15852-specific: !Sample_source_name_ch1)
# =========================================================================
# GSE15852 stores tissue type in !Sample_source_name_ch1.
# We map:
#   "normal breast tissue"  → 0
#   "breast tumor tissue"   → 1
# -------------------------------------------------------------------------

y                 = None
matrix_start_line = None

with open(DATA_PATH, "r") as f:
    for line_num, line in enumerate(f, start=1):

        # ── Label row ─────────────────────────────────────────────────
        if line.startswith("!Sample_source_name_ch1"):
            fields = line.strip().split("\t")[1:]          # skip row name
            labels = []
            for v in fields:
                v_clean = v.strip().strip('"').lower()
                if "normal" in v_clean:
                    labels.append(0)
                elif "tumor" in v_clean or "tumour" in v_clean or "cancer" in v_clean:
                    labels.append(1)
                else:
                    raise ValueError(
                        f"Unrecognised tissue label: '{v}'. "
                        "Expected 'normal breast tissue' or 'breast tumor tissue'."
                    )
            y = np.array(labels, dtype=int)

        # ── Matrix start marker ───────────────────────────────────────
        if line.startswith("!series_matrix_table_begin"):
            matrix_start_line = line_num
            break

# Sanity checks
if y is None:
    raise RuntimeError("Could not find !Sample_source_name_ch1 in the training file.")
if matrix_start_line is None:
    raise RuntimeError("Could not find !series_matrix_table_begin in the training file.")

n_normal = int((y == 0).sum())
n_cancer = int((y == 1).sum())
print(f"Training dataset labels parsed:  {len(y)} samples  "
      f"({n_normal} normal, {n_cancer} tumor)")
print(f"  Matrix data starts after line {matrix_start_line}")

# =========================================================================
# Section 2 — Load & transpose the expression matrix  (unchanged)
# =========================================================================
expr = pd.read_csv(
    DATA_PATH,
    sep="\t",
    skiprows=matrix_start_line,
    index_col=0,
)
if expr.index[-1].startswith("!"):
    expr = expr.iloc[:-1]

print("BEFORE transpose (gene-major):")
print(f"  Shape: {expr.shape}  (genes x samples)")
print()

X_df = expr.T
X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0)

print("AFTER transpose (sample-major):")
print(f"  Shape: {X_df.shape}  (samples x probes)")
print()

# =========================================================================
# Section 2-pre — LOG2 TRANSFORM GUARD & ComBat BATCH CORRECTION  **NEW**
# =========================================================================
# -------------------------------------------------------------------------
# Rationale
# -------------------------------------------------------------------------
# The PCA of this dataset shows a dominant ~40 % variance split on PC1 that
# is not aligned with the Normal/Cancer biological signal.  This is a
# hallmark of a latent batch effect.  After scanning every metadata row in
# the GEO file (submission date, treatment/growth/extraction/hybridisation
# protocols, scanner) we found NO explicit batch variable — all 86 samples
# share identical metadata.  We therefore infer batch membership from the
# PCA itself: samples with PC1 < 0 are assigned to Batch_0, and those with
# PC1 > 0 to Batch_1.  pyComBat is then used to remove this technical
# effect while preserving the Tissue (Normal vs Cancer) biological signal
# via its `mod` covariate argument.
#
# Log2 note: ComBat assumes approximately normally-distributed data.
# Raw microarray intensities can be heavily right-skewed; a log2 transform
# before ComBat is standard practice whenever values exceed ~50.
# -------------------------------------------------------------------------

# ── Log2 transform guard ─────────────────────────────────────────────────
max_val = X_df.max().max()
if max_val > 50:
    print("=" * 65)
    print("  LOG2 TRANSFORM")
    print("=" * 65)
    print(f"  Max value before transform: {max_val:.2f}")
    X_df = np.log2(X_df + 1)
    print(f"  Max value after  transform: {X_df.max().max():.2f}")
    print(f"  Log2(x+1) applied to stabilise distribution for ComBat.")
    print()
else:
    print(f"  Log2 guard: max value = {max_val:.2f} (≤ 50) — no transform needed.")
    print()

# ── Initial PCA for batch detection ──────────────────────────────────────
# We run a quick PCA on temporarily-standardised data to extract PC1
# coordinates for heuristic batch assignment.
_scaler_tmp = StandardScaler()
_X_tmp = _scaler_tmp.fit_transform(X_df)
_pca_tmp = PCA(n_components=2)
_pca_scores = _pca_tmp.fit_transform(_X_tmp)

batch = [0 if pc1 < 0 else 1 for pc1 in _pca_scores[:, 0]]
n_b0 = batch.count(0)
n_b1 = batch.count(1)

print("=" * 65)
print("  PCA-HEURISTIC BATCH ASSIGNMENT")
print("=" * 65)
print(f"  Batch_0 (PC1 < 0): {n_b0} samples")
print(f"  Batch_1 (PC1 > 0): {n_b1} samples")
print()

# Save pre-correction PCA scores for the before/after comparison plot
_pre_combat_pca = _pca_scores.copy()
_pre_combat_var = _pca_tmp.explained_variance_ratio_.copy()

# ── Apply pyComBat ────────────────────────────────────────────────────────
# pyComBat expects a gene-major DataFrame (genes × samples).
# The `mod` argument accepts a list-of-lists of covariates to PRESERVE.
# We pass the tissue labels so that the biological signal is not removed.
print("  Running pyComBat ...")
X_corrected_gene_major = pycombat(X_df.T, batch, mod=[y.tolist()])
X_df = X_corrected_gene_major.T  # back to sample-major
print(f"  ComBat-corrected matrix shape: {X_df.shape}")
print()

# ── Post-correction PCA (for comparison plot) ────────────────────────────
_scaler_post = StandardScaler()
_X_post = _scaler_post.fit_transform(X_df)
_pca_post = PCA(n_components=2)
_post_combat_pca = _pca_post.fit_transform(_X_post)
_post_combat_var = _pca_post.explained_variance_ratio_.copy()

# ── Before / After ComBat PCA scatter ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

labels_tissue = np.where(y == 0, 'Normal', 'Cancer')
batch_arr = np.array(batch)
marker_map = {0: 'o', 1: 's'}  # circle = Batch_0, square = Batch_1
color_map  = {'Normal': '#2ecc71', 'Cancer': '#e74c3c'}

for ax_i, (scores, var_ratio, title_tag) in enumerate([
    (_pre_combat_pca,  _pre_combat_var,  'BEFORE ComBat'),
    (_post_combat_pca, _post_combat_var, 'AFTER ComBat'),
]):
    ax = axes[ax_i]
    for tissue in ['Normal', 'Cancer']:
        for b in [0, 1]:
            mask = (labels_tissue == tissue) & (batch_arr == b)
            if mask.sum() == 0:
                continue
            ax.scatter(
                scores[mask, 0], scores[mask, 1],
                c=color_map[tissue],
                marker=marker_map[b],
                label=f'{tissue} (Batch {b})',
                s=60, alpha=0.75, edgecolors='white', lw=0.5,
            )
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
    ax.set_title(f'PCA — {title_tag}')
    ax.legend(fontsize=7, frameon=True)
    ax.grid(True, alpha=0.3)

fig.suptitle('ComBat Batch Correction Effect on GSE15852', fontsize=13, y=1.02)
fig.tight_layout()
plt.show()

# Clean up temporary variables
del _scaler_tmp, _X_tmp, _pca_tmp, _pca_scores
del _scaler_post, _X_post, _pca_post
del _pre_combat_pca, _pre_combat_var, _post_combat_pca, _post_combat_var

# =========================================================================
# Section 2a — UNSUPERVISED VARIANCE PRE-FILTER  **NEW**
# =========================================================================
# -------------------------------------------------------------------------
# Statistical reasoning
# -------------------------------------------------------------------------
# In microarray data the vast majority of ~54 000 probes show little to no
# variation across samples.  These "flat" genes carry almost no
# discriminatory information, yet they massively inflate the feature-space
# dimensionality.  When p >> n the Lasso's cross-validated MSE becomes
# extremely noisy because each fold trains on an underdetermined system
# where the solution is dominated by random noise in the fold split.
#
# By removing the bottom 50 % of probes by variance we:
#   • halve the feature space, lowering the p/n ratio;
#   • eliminate technical noise from non-informative probes;
#   • stabilise the Lasso's cross-validated MSE, allowing the 1-SE rule
#     to operate without collapsing to 0 features.
#
# The filter is *unsupervised*: it uses only probe variance across samples
# and never looks at the labels y, so it introduces no target leakage.
# -------------------------------------------------------------------------

feature_variances = X_df.var(axis=0)            # variance per probe
variance_threshold = feature_variances.median() # median = 50th percentile

# Keep only probes whose variance exceeds the median
keep_mask = feature_variances >= variance_threshold
X_filtered = X_df.loc[:, keep_mask]

print("=" * 65)
print("  VARIANCE PRE-FILTER (unsupervised)")
print("=" * 65)
print(f"  Features before filter : {X_df.shape[1]:,}")
print(f"  Variance threshold     : {variance_threshold:.6f}  (median)")
print(f"  Features after filter  : {X_filtered.shape[1]:,}")
print(f"  Features removed       : {(~keep_mask).sum():,}  (bottom 50 %)")
print()

# =========================================================================
# Section 2b — Standardise  (now operates on the filtered matrix)
# =========================================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

print(f"Standardised matrix shape: {X_scaled.shape}")
print(f"Mean ~ 0 check (first 5 cols): {X_scaled.mean(axis=0)[:5].round(6)}")
print(f"Std  ~ 1 check (first 5 cols): {X_scaled.std(axis=0)[:5].round(6)}")
print()

# =========================================================================
# Section 2c — PCA Diagnostic  (unchanged logic, runs on filtered data)
# =========================================================================
pca_full = PCA(n_components=min(20, X_scaled.shape[0]))
X_pca = pca_full.fit_transform(X_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"PC1 explains {explained[0]*100:.1f}% of variance")
print(f"PC2 explains {explained[1]*100:.1f}% of variance")
print(f"PC1+PC2 cumulative: {cumulative[1]*100:.1f}%")
print(f"Components needed for 80% variance: {np.searchsorted(cumulative, 0.80) + 1}")
print()

# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].bar(range(1, len(explained)+1), explained * 100,
            color='steelblue', edgecolor='white')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained (%)')
axes[0].set_title('Scree Plot')

axes[1].plot(range(1, len(cumulative)+1), cumulative * 100,
             'o-', color='steelblue', lw=2, markersize=5)
axes[1].axhline(80, ls='--', color='grey', lw=1, label='80% threshold')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()
fig.tight_layout()
plt.show()

# PCA scatter
labels_str = np.where(y == 0, 'Normal', 'Cancer')
colors_map = {'Normal': '#2ecc71', 'Cancer': '#e74c3c'}

fig, ax = plt.subplots(figsize=(8, 6))
for label in ['Normal', 'Cancer']:
    mask = labels_str == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=colors_map[label], label=label, s=60,
               alpha=0.75, edgecolors='white', lw=0.5)
ax.set_xlabel(f'PC1 ({explained[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({explained[1]*100:.1f}%)')
ax.set_title('PCA — Cancer vs Normal Tissue (Unsupervised)')
ax.legend(title='Tissue', frameon=True)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.show()

# =========================================================================
# Section 3 — LassoCV  **MODIFIED: cv=10**
# =========================================================================
# -------------------------------------------------------------------------
# Using cv=10 instead of cv=5 stabilises the error estimates:
#   • Each training fold contains ~109 samples  (vs ~97 with cv=5),
#     giving the Lasso a better-conditioned training set.
#   • The standard error of the mean MSE is computed over 10 folds
#     instead of 5, shrinking the SE bands and making the 1-SE rule
#     less likely to over-penalise.
# -------------------------------------------------------------------------

lasso = LassoCV(cv=10, random_state=42, max_iter=10_000)
lasso.fit(X_scaled, y)

n_features_alpha_min = int((lasso.coef_ != 0).sum())
print(f"Optimal alpha (alpha_min): {lasso.alpha_:.6f}")
print(f"Non-zero features at alpha_min: {n_features_alpha_min} out of {len(lasso.coef_)}")
print()

# =========================================================================
# Section 3b — DYNAMIC 1-SE FALLBACK  **NEW — replaces old binary search**
# =========================================================================
# -------------------------------------------------------------------------
# The 1-Standard-Error (1-SE) rule
# -------------------------------------------------------------------------
# The idea: instead of picking the alpha with the absolute lowest mean MSE
# (alpha_min), pick the *largest* alpha whose mean MSE is still within one
# standard error of the minimum.  A larger alpha means a sparser model
# (fewer features), which is more interpretable and less prone to overfit.
#
# threshold = mse_min + (multiplier × se_min)
#
# In high-dimensional settings with few samples, the SE bands can be so
# wide that the 1-SE alpha produces 0 features.  To handle this we use a
# *dynamic multiplier*: start at 1.0 and progressively reduce to 0.75,
# 0.50, 0.25 until the selected alpha yields a clinically viable signature
# (between 15 and 30 features).  If none work, fall back to alpha_min.
# -------------------------------------------------------------------------

# Extract the MSE path from LassoCV  (shape: n_alphas × n_folds)
mean_mse = np.mean(lasso.mse_path_, axis=1)
std_mse  = np.std(lasso.mse_path_, axis=1)
n_folds  = lasso.mse_path_.shape[1]

# Standard error = std / sqrt(n_folds)
se_mse = std_mse / np.sqrt(n_folds)

# Identify the minimum MSE and its SE
idx_min   = np.argmin(mean_mse)
mse_min   = mean_mse[idx_min]
se_min    = se_mse[idx_min]
alpha_min = lasso.alphas_[idx_min]

# Targets for a clinically viable signature
MIN_FEATURES = 15
MAX_FEATURES = 30
MIN_FEATURES_HARD = 10    # absolute floor before reducing multiplier

# Dynamic SE-multiplier cascade
multipliers = [1.0, 0.75, 0.50, 0.25]
chosen_alpha      = None
chosen_n_features = None
chosen_multiplier = None

for mult in multipliers:
    threshold = mse_min + mult * se_min

    # Find the largest alpha whose mean MSE ≤ threshold.
    # lasso.alphas_ is sorted descending (largest first), so the first
    # alpha that satisfies the condition is the most parsimonious.
    candidate_alpha = None
    for i, a in enumerate(lasso.alphas_):
        if mean_mse[i] <= threshold:
            candidate_alpha = a
            break               # first hit = largest qualifying alpha

    if candidate_alpha is None:
        # No alpha satisfies this threshold — try a smaller multiplier
        continue

    # Fit a Lasso at the candidate alpha and count features
    model_1se = Lasso(alpha=candidate_alpha, max_iter=10_000)
    model_1se.fit(X_scaled, y)
    n_feat = int((model_1se.coef_ != 0).sum())

    # Check against the hard floor
    if n_feat >= MIN_FEATURES_HARD:
        chosen_alpha      = candidate_alpha
        chosen_n_features = n_feat
        chosen_multiplier = mult
        # If within the ideal range, stop immediately
        if MIN_FEATURES <= n_feat <= MAX_FEATURES:
            break
        # Otherwise, keep searching with a smaller multiplier for a
        # better (more features) result

# Fallback: if no multiplier produced ≥ MIN_FEATURES_HARD features,
# use alpha_min itself
if chosen_alpha is None:
    chosen_alpha      = alpha_min
    chosen_multiplier = 0.0   # indicates fallback
    model_1se = Lasso(alpha=chosen_alpha, max_iter=10_000)
    model_1se.fit(X_scaled, y)
    chosen_n_features = int((model_1se.coef_ != 0).sum())

adjusted_1se_alpha = chosen_alpha

# ── Summary ──────────────────────────────────────────────────────────────
print("=" * 70)
print("  DYNAMIC 1-SE ALPHA SELECTION SUMMARY")
print("=" * 70)
print(f"  MSE at alpha_min         : {mse_min:.6f}")
print(f"  SE  at alpha_min         : {se_min:.6f}")
print(f"  SE multiplier used       : {chosen_multiplier}")
print(f"  1-SE threshold (MSE)     : {mse_min + chosen_multiplier * se_min:.6f}")
print()
print(f"  alpha_min                : {alpha_min:.6f}  →  {n_features_alpha_min} features")
print(f"  adjusted_1se_alpha       : {adjusted_1se_alpha:.6f}  →  {chosen_n_features} features")
print(f"  Alpha multiplier (ratio) : {adjusted_1se_alpha / alpha_min:.2f}x")
print("=" * 70)
print()

# =========================================================================
# Section 4 — Biomarker Panel (from the adjusted 1-SE model)
# =========================================================================
coef_1se = pd.Series(model_1se.coef_, index=X_filtered.columns, name="coefficient")
nonzero_1se = coef_1se[coef_1se != 0].copy()
nonzero_1se = nonzero_1se.reindex(nonzero_1se.abs().sort_values(ascending=False).index)

panel = nonzero_1se.to_frame()
panel.index.name = "Probe_ID"
panel["abs_coefficient"] = panel["coefficient"].abs()
panel["direction"] = np.where(panel["coefficient"] > 0, "up in cancer", "down in cancer")
panel["rank"] = range(1, len(panel) + 1)

y_pred_1se = (model_1se.predict(X_scaled) > 0.5).astype(int)
acc_1se = accuracy_score(y, y_pred_1se)

print(f"Adjusted 1-SE LASSO selected {len(panel)} biomarker probes")
print(f"Training accuracy:  {acc_1se:.4f}")
print(f"R² (in-sample):     {model_1se.score(X_scaled, y):.4f}")
print()
print("=" * 70)
print(f"  BIOMARKER PANEL ({len(panel)} probes,  alpha = {adjusted_1se_alpha:.6f})")
print("=" * 70)
print(panel.to_string())
print()

# =========================================================================
# Section 4b — For reference: alpha_min biomarker panel (top 20)
# =========================================================================
coef_series = pd.Series(lasso.coef_, index=X_filtered.columns, name="coefficient")
nonzero = coef_series[coef_series != 0].copy()
nonzero = nonzero.reindex(nonzero.abs().sort_values(ascending=False).index)

print(f"Total probes:           {len(coef_series):,}")
print(f"Non-zero (alpha_min):   {len(nonzero):,}")
print(f"Zeroed-out (excluded):  {(coef_series == 0).sum():,}")
print()

top20 = nonzero.head(20).to_frame()
top20.index.name = "Probe_ID"
top20["abs_coefficient"] = top20["coefficient"].abs()
top20["direction"] = np.where(top20["coefficient"] > 0, "up in cancer", "down in cancer")

print("=" * 65)
print("  TOP 20 alpha_min BIOMARKER PROBES (for comparison)")
print("=" * 65)
print(top20.to_string())
print()

# =========================================================================
# Section 5 — Performance Verification  (uses adjusted 1-SE model)
# =========================================================================
r2 = model_1se.score(X_scaled, y)
y_pred = (model_1se.predict(X_scaled) > 0.5).astype(int)
acc = accuracy_score(y, y_pred)

print(f"R² score (in-sample):       {r2:.4f}")
print(f"Training accuracy:          {acc:.4f}")
print(f"Adjusted 1-SE alpha:        {adjusted_1se_alpha:.6f}")
print(f"Non-zero / Total features:  {chosen_n_features} / {len(model_1se.coef_)}")
print()
print(classification_report(y, y_pred, target_names=['Normal', 'Cancer']))

# ── Alpha-path MSE curve ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.5))

ax.semilogx(lasso.alphas_, mean_mse, color='steelblue', lw=2,
            label='Mean CV MSE')
ax.fill_between(lasso.alphas_, mean_mse - std_mse, mean_mse + std_mse,
                alpha=0.2, color='steelblue', label='± 1 std')

# Mark alpha_min
ax.axvline(alpha_min, color='crimson', ls='--', lw=1.5,
           label=f'alpha_min = {alpha_min:.4f}  ({n_features_alpha_min} feat)')

# Mark adjusted 1-SE alpha
ax.axvline(adjusted_1se_alpha, color='orange', ls='--', lw=1.5,
           label=f'adj 1-SE alpha = {adjusted_1se_alpha:.4f}  ({chosen_n_features} feat)')

# Horizontal threshold line
threshold_val = mse_min + chosen_multiplier * se_min
ax.axhline(threshold_val, color='green', ls=':', lw=1.2,
           label=f'1-SE threshold (mult={chosen_multiplier})')

ax.set_xlabel('Alpha (log scale)')
ax.set_ylabel('Mean Squared Error')
ax.set_title('LassoCV — Alpha Selection Path  (cv=10, variance-filtered)')
ax.legend(fontsize=8, loc='upper left')
ax.invert_xaxis()
fig.tight_layout()
plt.show()

# =========================================================================
# Section 6 — EXPORT TRAINING ARTIFACTS  **NEW**
# =========================================================================
# Export the three assets needed for external validation:
#   1. The fitted StandardScaler   — ensures test data undergoes the exact
#      same centering/scaling transform learned from the training set.
#      Using .transform() (not .fit_transform()) on new data prevents
#      data leakage.
#   2. The adjusted 1-SE Lasso model — the final sparse classifier.
#   3. The ordered list of Probe_IDs the model retained — needed to
#      subset and align external datasets to the correct feature columns.
# -------------------------------------------------------------------------

# Build the list of retained Probe_IDs (non-zero coefficients)
retained_features = list(X_filtered.columns[model_1se.coef_ != 0])

# Save to disk
EXPORT_DIR = "exported_model"
import os
os.makedirs(EXPORT_DIR, exist_ok=True)

joblib.dump(scaler,             os.path.join(EXPORT_DIR, "training_scaler.joblib"))
joblib.dump(model_1se,          os.path.join(EXPORT_DIR, "lasso_1se_model.joblib"))
joblib.dump(retained_features,  os.path.join(EXPORT_DIR, "retained_features.joblib"))

print()
print("=" * 70)
print("  EXPORTED TRAINING ARTIFACTS")
print("=" * 70)
print(f"  Scaler       → {os.path.join(EXPORT_DIR, 'training_scaler.joblib')}")
print(f"  Lasso model  → {os.path.join(EXPORT_DIR, 'lasso_1se_model.joblib')}")
print(f"  Feature list → {os.path.join(EXPORT_DIR, 'retained_features.joblib')}")
print(f"  # of features: {len(retained_features)}")
print(f"  Features     : {retained_features}")
print("=" * 70)
