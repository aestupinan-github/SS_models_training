# ============================================================
# PLOT C++ PREDICTION RESULTS
# ------------------------------------------------------------
# Reads:
#   predictions.csv
#
# Reconstructs:
#   1. Predicted vs True overlap
#   2. Error histogram
#   3. Relative error histogram
#   4. Surrogate vs analytical timing comparison (NEW)
#
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============================================================
# LOAD CSV
# ============================================================
CASE_DIR = Path(sys.argv[1])
data_file = CASE_DIR / "predictions.csv"
print("\nReading: ", data_file)
df = pd.read_csv(data_file)
print(df.info())

# ============================================================
# TRUE AND PREDICTED VALUES
# ============================================================

TRUE_COLUMN = "exact_overlap"
PREDICTION_COLUMN = "pred_overlap"

required_columns = [
    TRUE_COLUMN,
    PREDICTION_COLUMN,
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise KeyError(
        f"Missing required column(s): {missing_columns}\n"
        f"Available columns are: {list(df.columns)}"
    )

# Convert to numeric in case the CSV contains empty strings
# or values stored as text.
df[TRUE_COLUMN] = pd.to_numeric(
    df[TRUE_COLUMN],
    errors="coerce"
)

df[PREDICTION_COLUMN] = pd.to_numeric(
    df[PREDICTION_COLUMN],
    errors="coerce"
)

# Remove rows for which either overlap value is missing
valid_mask = (
    np.isfinite(df[TRUE_COLUMN]) &
    np.isfinite(df[PREDICTION_COLUMN])
)

df_valid = df.loc[valid_mask].copy()

if df_valid.empty:
    raise ValueError(
        "No valid overlap rows found in the CSV file."
    )

print(
    f"\nUsing {len(df_valid)} valid overlap rows "
    f"out of {len(df)} total rows."
)

true_overlap = df_valid[TRUE_COLUMN].to_numpy()
pred_overlap = df_valid[PREDICTION_COLUMN].to_numpy()

# ============================================================
# ABSOLUTE ERROR
# ============================================================
abs_error = np.abs(
    pred_overlap - true_overlap
)

# ============================================================
# RELATIVE ERROR
# ============================================================
relative_error = (
    abs_error /
    (np.abs(true_overlap) + 1e-15)
)

# ============================================================
# RMSE
# ============================================================
rmse = np.sqrt(
    np.mean(
        (pred_overlap - true_overlap) ** 2
    )
)
relative_rmse = (
    rmse /
    (np.mean(np.abs(true_overlap)) + 1e-15)
)

# ============================================================
# PRINT METRICS
# ============================================================
print("\n================================================")
print("OVERLAP RESULTS")
print("================================================")
print(f"RMSE = {rmse:.6e} m")
print(
    f"Relative RMSE = "
    f"{100 * relative_rmse:.4f} %"
)
print(
    f"Mean Relative Error = "
    f"{100 * relative_error.mean():.4f} %"
)
print(
    f"Median Relative Error = "
    f"{100 * np.median(relative_error):.4f} %"
)

# ============================================================
# PREDICTED VS TRUE OVERLAP
# ============================================================
plt.figure(figsize=(7, 7))
plt.scatter(
    true_overlap,
    pred_overlap,
    s=8,
    alpha=0.5
)
vmin = min(
    true_overlap.min(),
    pred_overlap.min()
)
vmax = max(
    true_overlap.max(),
    pred_overlap.max()
)
plt.plot(
    [vmin, vmax],
    [vmin, vmax],
    'k--'
)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("True overlap")
plt.ylabel("Predicted overlap")
plt.title("Predicted vs True Overlap")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "cpp_predicted_vs_true_overlap.png",
    dpi=300
)
plt.show()

# ============================================================
# ABSOLUTE ERROR HISTOGRAM
# ============================================================
plt.figure(figsize=(8, 5))
plt.hist(
    abs_error,
    bins=100,
    alpha=0.7
)
plt.xlabel("Absolute Error [m]")
plt.ylabel("Count")
plt.title("Overlap Absolute Error")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "cpp_overlap_absolute_error_histogram.png",
    dpi=300
)
plt.show()

# ============================================================
# RELATIVE ERROR HISTOGRAM
# ============================================================
plt.figure(figsize=(8, 5))
plt.hist(
    100 * relative_error,
    bins=100,
    alpha=0.7
)
plt.xlabel("Relative Error [%]")
plt.ylabel("Count")
plt.title("Overlap Relative Error")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    "cpp_overlap_relative_error_histogram.png",
    dpi=300
)
plt.show()

# ============================================================
# TIMING COMPARISON: SURROGATE (pred_time_us) VS
# ANALYTICAL (exact_time_us)
# ------------------------------------------------------------
# Only runs if both timing columns are present in predictions.csv
# (i.e. produced by the timed C++ benchmark version).
# ============================================================
if "pred_time_us" in df.columns and "exact_time_us" in df.columns:

    pred_time = df["pred_time_us"].values
    exact_time = df["exact_time_us"].values

    sum_pred = pred_time.sum()
    sum_exact = exact_time.sum()
    avg_pred = pred_time.mean()
    avg_exact = exact_time.mean()
    speedup = sum_exact / sum_pred if sum_pred > 0 else float("nan")

    print("\n================================================")
    print("TIMING RESULTS (surrogate vs analytical)")
    print("================================================")
    print(f"Surrogate  - sum : {sum_pred:.3f} us | avg : {avg_pred:.6f} us")
    print(f"Analytical - sum : {sum_exact:.3f} us | avg : {avg_exact:.6f} us")
    print(f"Speed-up (analytical/surrogate) : {speedup:.2f}x")

    idx = np.arange(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- left panel: per-row timings over the dataset ---
    axes[0].scatter(idx, pred_time, s=4, alpha=0.4, label="Surrogate (predict)")
    axes[0].scatter(idx, exact_time, s=4, alpha=0.4, label="Analytical (exact)")
    axes[0].axhline(avg_pred, color="tab:blue", linestyle="--", linewidth=1.5)
    axes[0].axhline(avg_exact, color="tab:orange", linestyle="--", linewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Time per evaluation (µs)")
    axes[0].set_title("Per-row timing: surrogate vs analytical")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[0].text(
        0.02, 0.95,
        f"avg surrogate  = {avg_pred:.4f} µs\n"
        f"avg analytical = {avg_exact:.4f} µs\n"
        f"speed-up       = {speedup:.1f}x",
        transform=axes[0].transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    # --- right panel: direct pred_time vs exact_time scatter ---
    axes[1].scatter(exact_time, pred_time, s=6, alpha=0.4, color="tab:green")
    lims = [
        min(pred_time.min(), exact_time.min()) * 0.8,
        max(pred_time.max(), exact_time.max()) * 1.2
    ]
    axes[1].plot(lims, lims, "k--", linewidth=1, label="y = x")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    axes[1].set_xlabel("Analytical time (µs)")
    axes[1].set_ylabel("Surrogate time (µs)")
    axes[1].set_title("Surrogate vs analytical time (per row)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(loc="upper left")

    axes[1].text(
        0.02, 0.95,
        f"avg surrogate  = {avg_pred:.4f} µs\n"
        f"avg analytical = {avg_exact:.4f} µs\n"
        f"speed-up       = {speedup:.1f}x",
        transform=axes[1].transAxes,
        va="top", ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.savefig("cpp_timing_comparison.png", dpi=300)
    plt.show()

else:
    print(
        "\n[INFO] 'pred_time_us'/'exact_time_us' columns not found in "
        "predictions.csv - skipping timing comparison plot. "
        "Re-run the timed C++ benchmark to generate them."
    )
