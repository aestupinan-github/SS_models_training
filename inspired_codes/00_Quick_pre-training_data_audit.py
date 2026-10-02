"""
Quick pre-training data audit
-------------------------------
Checks whether the new sweep dataset actually has:
  1. Per-row Z_POS jitter (not clustered at exact repeated anchor values)
  2. No empty gaps in Z_POS coverage across [HALF, DIAG_HALF]
  3. A sane contact-ratio trend across the domain
"""
import numpy as np
import pandas as pd

DATASET_PATH = "training_data_sweep_v2.csv"  # adjust if named differently
HALF = 0.5
DIAG_HALF = np.sqrt(3.0) / 2.0

df = pd.read_csv(DATASET_PATH)
n_total = len(df)
n_unique_z = df["position_z"].nunique()

print(f"Total rows: {n_total:,}")
print(f"Unique position_z values: {n_unique_z:,}")
print(f"Ratio unique/total: {n_unique_z/n_total*100:.4f}%")

if n_unique_z < 0.01 * n_total:
    print("-> WARNING: very few unique Z_POS values relative to row count.")
    print("   This suggests jitter is NOT active (still clustered at exact anchors).")
else:
    print("-> Looks like per-row jitter is active (many distinct Z_POS values).")

# Gap check: bin Z_POS into small windows and flag any empty bins
bins = np.linspace(HALF, DIAG_HALF, 200)
counts, _ = np.histogram(df["position_z"], bins=bins)
empty_bins = np.where(counts == 0)[0]
print(f"\nEmpty bins out of {len(counts)}: {len(empty_bins)}")
if len(empty_bins) > 0:
    for idx in empty_bins:
        print(f"  Empty region: [{bins[idx]:.4f}, {bins[idx+1]:.4f})")
else:
    print("-> No empty regions found across the domain.")

# Contact ratio trend, coarse bins
coarse_edges = np.linspace(HALF, DIAG_HALF, 15)
print(f"\n{'Z_POS bin':<20}{'N':>10}{'Contact %':>12}")
for lo, hi in zip(coarse_edges[:-1], coarse_edges[1:]):
    m = (df["position_z"] >= lo) & (df["position_z"] < hi)
    n = m.sum()
    if n == 0:
        print(f"[{lo:.4f},{hi:.4f}){'':<4}{0:>10}   (empty)")
        continue
    contact_pct = (df.loc[m, "overlap(m)"] > 0).mean() * 100
    print(f"[{lo:.4f},{hi:.4f}){'':<4}{n:>10}{contact_pct:>12.2f}")
