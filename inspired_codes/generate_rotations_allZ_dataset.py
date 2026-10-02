"""
============================================================
DENSE, MULTI-Z_POS TRAINING DATASET GENERATOR — FULL SWEEP
(Sobol-augmented orientation sampling)
------------------------------------------------------------
v2: adds three fixes motivated by stratified-error diagnostics
on the previous sweep dataset:
  1. Per-row Z_POS jitter: each row's height is perturbed by a
     small random amount around its anchor (recomputing overlap/
     clearance from the jittered height, not just relabeling),
     so the network sees a genuinely continuous height variable
     instead of ~17 discrete clusters. Directly targets the
     slope-check failure (mean d(overlap)/d(Z_POS) = -0.42 instead
     of -1.0, with huge variance) found via off-anchor validation.
  2. Anchor gap-closing: any consecutive anchor gap wider than
     MAX_ANCHOR_GAP is automatically subdivided. Fixes the empty
     [0.80, 0.83) test bin found in the stratified-error check.
  3. Adaptive clearance floor near Z_POS=HALF: anchors very close
     to HALF get a much lower MIN_CLEARANCE_FLOOR, so the contact
     classifier finally sees small-clearance examples right where
     it was previously undersupplied (90.71% accuracy region).
============================================================
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rsc
from scipy.stats import qmc
from scipy.spatial import cKDTree

# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_PATH = "training_data_sweep_v2.csv"
CUBE_SIZE = 1.0
HALF = CUBE_SIZE / 2.0
DIAG_HALF = np.sqrt(3.0) / 2.0
SEED = 7

# ------------------------------------------------------------
# ANCHOR LIST CONSTRUCTION
# ------------------------------------------------------------
N_BACKBONE = 11
t = np.linspace(0.0, 1.0, N_BACKBONE)
edge_weight = 0.5 - 0.5 * np.cos(np.pi * t)
backbone = HALF + edge_weight * (DIAG_HALF - HALF)

# Fix #3 groundwork: several very-close-to-HALF anchors (extended
# down to 1e-5 from the previous [1e-4,1e-3,1e-2] set), mirrored
# near DIAG_HALF for symmetry.
near_half = HALF + np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
near_diag = DIAG_HALF - np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2])

raw_anchors = np.concatenate([[HALF], near_half, backbone, near_diag, [DIAG_HALF]])
raw_anchors = np.unique(np.round(raw_anchors, 8))
raw_anchors = raw_anchors[(raw_anchors >= HALF) & (raw_anchors <= DIAG_HALF)]

# ------------------------------------------------------------
# FIX #2: GAP-CLOSING — subdivide any anchor gap > MAX_ANCHOR_GAP
# ------------------------------------------------------------
MAX_ANCHOR_GAP = 0.02   # tune down for finer mid-domain coverage

def close_gaps(anchors, max_gap):
    anchors = sorted(anchors)
    result = [anchors[0]]
    for i in range(1, len(anchors)):
        lo, hi = anchors[i - 1], anchors[i]
        gap = hi - lo
        if gap > max_gap:
            n_extra = int(np.ceil(gap / max_gap)) - 1
            extra = np.linspace(lo, hi, n_extra + 2)[1:-1]
            result.extend(extra.tolist())
        result.append(hi)
    return np.array(sorted(set(np.round(result, 8))))

Z_POS_VALUES = close_gaps(raw_anchors, MAX_ANCHOR_GAP)

# ------------------------------------------------------------
# FIX #1: PER-ANCHOR JITTER WIDTH (adaptive to local spacing)
# ------------------------------------------------------------
JITTER_FRACTION = 0.4   # fraction of nearest-neighbor anchor spacing
MAX_JITTER = 5e-3       # absolute cap, even in sparse mid-domain regions

def compute_jitter_widths(anchors, fraction, max_jitter):
    n = len(anchors)
    nn_dist = np.zeros(n)
    for i in range(n):
        d = []
        if i > 0:
            d.append(anchors[i] - anchors[i - 1])
        if i < n - 1:
            d.append(anchors[i + 1] - anchors[i])
        nn_dist[i] = min(d) if d else 0.0
    return np.clip(fraction * nn_dist, 0.0, max_jitter)

JITTER_HALF_WIDTHS = compute_jitter_widths(Z_POS_VALUES, JITTER_FRACTION, MAX_JITTER)

# ------------------------------------------------------------
# ROW BUDGET CONTROL
# ------------------------------------------------------------
SWEEP_SCALE = 0.3

MIN_OVERLAP_FLOOR = 1e-4
MIN_CLEARANCE_FLOOR_DEFAULT = 1e-4
# FIX #3: anchors within this distance of HALF get a much lower floor
EDGE_ZONE_WIDTH = 1e-2
MIN_CLEARANCE_FLOOR_EDGE = 1e-6

def get_clearance_floor(z_pos):
    if (z_pos - HALF) < EDGE_ZONE_WIDTH:
        return MIN_CLEARANCE_FLOOR_EDGE
    return MIN_CLEARANCE_FLOOR_DEFAULT

decade_specs = [
    (1e-4, 1e-3, 400),
    (1e-3, 1e-2, 250),
    (1e-2, 1.0,  150),
]
SAMPLES_PER_LEVEL = max(1, int(round(500 * SWEEP_SCALE)))

clearance_decade_specs = [
    (1e-6, 1e-5, 100),   # new fine bottom decade, only reachable near HALF
    (1e-5, 1e-4, 120),
    (1e-4, 1e-3, 150),
    (1e-3, 1e-2, 100),
]
CLEARANCE_SAMPLES_PER_LEVEL = max(1, int(round(500 * SWEEP_SCALE)))

S_MIN, S_MAX = 1.0, np.sqrt(3.0)

# ============================================================
# PROJECTED ROW COUNT (dry estimate)
# ------------------------------------------------------------
# Note: jitter can shift a small fraction of rows across the
# contact/clearance boundary near edge anchors, so actual counts
# may differ very slightly from this estimate. Not significant.
# ============================================================
def estimate_rows():
    total = 0
    per_anchor = []
    for Z_POS in Z_POS_VALUES:
        overlap_at_Smax = HALF * S_MAX - Z_POS
        max_clear = max(0.0, Z_POS - HALF)
        clear_floor = get_clearance_floor(Z_POS)
        n_contact = 0
        for lo, hi, n_levels in decade_specs:
            hi_c = min(hi, overlap_at_Smax)
            lo_c = max(lo, MIN_OVERLAP_FLOOR)
            if lo_c < hi_c:
                n_contact += n_levels * SAMPLES_PER_LEVEL
        n_clear = 0
        for lo, hi, n_levels in clearance_decade_specs:
            hi_c = min(hi, max_clear)
            lo_c = max(lo, clear_floor)
            if lo_c < hi_c:
                n_clear += n_levels * CLEARANCE_SAMPLES_PER_LEVEL
        per_anchor.append((Z_POS, n_contact, n_clear))
        total += n_contact + n_clear
    return total, per_anchor

_est_total, _est_per_anchor = estimate_rows()
print("================================================")
print("PROJECTED ROW COUNT (before generating)")
print("================================================")
print(f"Anchors: {len(Z_POS_VALUES)} | SWEEP_SCALE = {SWEEP_SCALE} | "
      f"MAX_ANCHOR_GAP = {MAX_ANCHOR_GAP}")
for (z, nc, ncl), jw in zip(_est_per_anchor, JITTER_HALF_WIDTHS):
    print(f"  Z_POS={z:.6f} (jitter=±{jw:.2e}) -> contact={nc:>8d}  "
          f"clearance={ncl:>7d}  total={nc+ncl:>8d}")
print(f"\nProjected TOTAL rows: {_est_total:,}\n")

# ============================================================
# SOBOL ENGINE
# ============================================================
sobol_engine = qmc.Sobol(d=5, scramble=True, seed=SEED)
PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

def next_pow2(n):
    return 1 << (n - 1).bit_length()

def draw_sobol(n):
    n_pad = next_pow2(n)
    return sobol_engine.random(n_pad)[:n]

# ============================================================
# CORE CONSTRUCTION (unchanged, validated)
# ============================================================
def sample_abc_on_circle(S, rng, max_iter=200):
    N = len(S)
    center = np.repeat(S[:, None] / 3.0, 3, axis=1)
    radius_sq = np.maximum(1.0 - (S ** 2) / 3.0, 0.0)
    radius = np.sqrt(radius_sq)
    u = np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0)
    w = np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0)
    abc = np.zeros((N, 3))
    unresolved = np.ones(N, dtype=bool)
    for _ in range(max_iter):
        idx = np.where(unresolved)[0]
        if len(idx) == 0:
            break
        tt = rng.uniform(0.0, 2.0 * np.pi, size=len(idx))
        pts = (center[idx]
               + radius[idx, None] * np.cos(tt)[:, None] * u[None, :]
               + radius[idx, None] * np.sin(tt)[:, None] * w[None, :])
        valid = np.all(pts >= -1e-9, axis=1)
        good_idx = idx[valid]
        abc[good_idx] = np.clip(pts[valid], 0.0, None)
        unresolved[good_idx] = False
    if unresolved.any():
        abc[unresolved] = center[unresolved]
    return abc

def construct_quaternions(S_required, rng):
    N = len(S_required)
    abc = sample_abc_on_circle(S_required, rng)
    sobol_pts = draw_sobol(N)
    spin_phi = sobol_pts[:, 0] * 2.0 * np.pi
    perm_choice = np.minimum((sobol_pts[:, 1] * 6).astype(int), 5)
    sign_x = np.where(sobol_pts[:, 2] < 0.5, 1.0, -1.0)
    sign_y = np.where(sobol_pts[:, 3] < 0.5, 1.0, -1.0)
    sign_z = np.where(sobol_pts[:, 4] < 0.5, 1.0, -1.0)

    abc_perm = np.zeros_like(abc)
    for p_idx, perm in enumerate(PERMS):
        sel = perm_choice == p_idx
        if sel.any():
            abc_perm[sel] = abc[sel][:, perm]

    signs = np.stack([sign_x, sign_y, sign_z], axis=1)
    r_vec = abc_perm * signs
    r_vec = r_vec / np.linalg.norm(r_vec, axis=1, keepdims=True)

    ref = np.tile(np.array([1.0, 0.0, 0.0]), (N, 1))
    close_to_ref = np.abs(np.sum(ref * r_vec, axis=1)) > 0.9
    ref[close_to_ref] = np.array([0.0, 1.0, 0.0])
    row1 = ref - r_vec * np.sum(ref * r_vec, axis=1, keepdims=True)
    row1 = row1 / np.linalg.norm(row1, axis=1, keepdims=True)
    row2 = np.cross(r_vec, row1)

    cphi, sphi = np.cos(spin_phi), np.sin(spin_phi)
    row1_rot = cphi[:, None] * row1 + sphi[:, None] * row2
    row2_rot = -sphi[:, None] * row1 + cphi[:, None] * row2

    M = np.stack([row1_rot, row2_rot, r_vec], axis=1)
    quats = Rsc.from_matrix(M).as_quat()
    qx, qy, qz, qw = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]

    flip = qw < 0.0
    qw[flip] *= -1.0
    qx[flip] *= -1.0
    qy[flip] *= -1.0
    qz[flip] *= -1.0

    rcx = 2 * (qx * qz - qw * qy)
    rcy = 2 * (qy * qz + qw * qx)
    rcz = 1 - 2 * (qx ** 2 + qy ** 2)
    S_check = np.abs(rcx) + np.abs(rcy) + np.abs(rcz)
    return qw, qx, qy, qz, S_check

# ============================================================
# FIX #1: JITTER + RELABEL — recompute overlap/clearance from the
# jittered Z_POS, not the anchor value, so every row stays
# geometrically self-consistent.
# ============================================================
def apply_jitter_and_label(Z_POS_anchor, S_check, jitter_half_width, rng):
    N = len(S_check)
    if jitter_half_width > 0:
        z_jitter = rng.uniform(-jitter_half_width, jitter_half_width, size=N)
    else:
        z_jitter = np.zeros(N)
    Z_row = np.clip(Z_POS_anchor + z_jitter, HALF, DIAG_HALF)
    overlap_actual = HALF * S_check - Z_row
    is_contact = overlap_actual > 0.0
    overlap_col = np.where(is_contact, overlap_actual, 0.0)
    clearance_col = np.where(~is_contact, -overlap_actual, 0.0)
    return (Z_row.astype(np.float32),
            overlap_col.astype(np.float32),
            clearance_col.astype(np.float32))

# ============================================================
# GENERATE ACROSS ALL ANCHORS
# ============================================================
rng = np.random.default_rng(SEED)
all_qw, all_qx, all_qy, all_qz = [], [], [], []
all_overlap, all_clearance, all_posz = [], [], []
anchor_summary = []

print("================================================")
print("GEOMETRY / Z_POS SWEEP (v2: jitter + gap-closing + edge floor)")
print("================================================")
print(f"CUBE_SIZE = {CUBE_SIZE}")
print(f"Z_POS values ({len(Z_POS_VALUES)}): {np.round(Z_POS_VALUES, 6)}")

for anchor_idx, Z_POS in enumerate(Z_POS_VALUES):
    jitter_hw = JITTER_HALF_WIDTHS[anchor_idx]
    overlap_at_Smin = HALF * S_MIN - Z_POS
    overlap_at_Smax = HALF * S_MAX - Z_POS
    max_achievable_clearance = max(0.0, -overlap_at_Smin)
    clear_floor = get_clearance_floor(Z_POS)

    print(f"\n--- Z_POS = {Z_POS:.6f} (jitter=±{jitter_hw:.2e}) | "
          f"achievable overlap: [{overlap_at_Smin:.6f}, {overlap_at_Smax:.6f}] | "
          f"max clearance: {max_achievable_clearance:.6f} | "
          f"clearance floor: {clear_floor:.0e} ---")

    actual_contact_n = 0
    actual_clear_n = 0

    print("Generating contact samples (deterministic overlap grid)...")
    for lo, hi, n_levels in decade_specs:
        hi_clipped = min(hi, overlap_at_Smax)
        lo_clipped = max(lo, MIN_OVERLAP_FLOOR)
        if lo_clipped >= hi_clipped:
            print(f"  Decade [{lo:.0e}, {hi:.0e}): UNREACHABLE -> skipped")
            continue
        overlap_levels = np.exp(np.linspace(np.log(lo_clipped), np.log(hi_clipped), n_levels))
        for target_overlap in overlap_levels:
            S_req = np.full(SAMPLES_PER_LEVEL, (target_overlap + Z_POS) / HALF)
            S_req = np.clip(S_req, S_MIN, S_MAX)
            qw, qx, qy, qz, S_check = construct_quaternions(S_req, rng)
            Z_row, overlap_col, clearance_col = apply_jitter_and_label(
                Z_POS, S_check, jitter_hw, rng)
            all_qw.append(qw); all_qx.append(qx); all_qy.append(qy); all_qz.append(qz)
            all_overlap.append(overlap_col)
            all_clearance.append(clearance_col)
            all_posz.append(Z_row)
            actual_contact_n += int((overlap_col > 0).sum())
            actual_clear_n += int((clearance_col > 0).sum())
        print(f"  Decade [{lo:.0e}, {hi_clipped:.2e}): {n_levels} x {SAMPLES_PER_LEVEL} "
              f"= {n_levels * SAMPLES_PER_LEVEL} samples")

    print("Generating zero-overlap samples (deterministic clearance grid)...")
    for lo, hi, n_levels in clearance_decade_specs:
        hi_clipped = min(hi, max_achievable_clearance)
        lo_clipped = max(lo, clear_floor)
        if lo_clipped >= hi_clipped:
            print(f"  Clearance decade [{lo:.0e}, {hi:.0e}): UNREACHABLE -> skipped")
            continue
        clearance_levels = np.exp(np.linspace(np.log(lo_clipped), np.log(hi_clipped), n_levels))
        for target_clearance in clearance_levels:
            target_overlap_neg = -target_clearance
            S_req = np.full(CLEARANCE_SAMPLES_PER_LEVEL, (target_overlap_neg + Z_POS) / HALF)
            S_req = np.clip(S_req, S_MIN, S_MAX)
            qw, qx, qy, qz, S_check = construct_quaternions(S_req, rng)
            Z_row, overlap_col, clearance_col = apply_jitter_and_label(
                Z_POS, S_check, jitter_hw, rng)
            all_qw.append(qw); all_qx.append(qx); all_qy.append(qy); all_qz.append(qz)
            all_overlap.append(overlap_col)
            all_clearance.append(clearance_col)
            all_posz.append(Z_row)
            actual_contact_n += int((overlap_col > 0).sum())
            actual_clear_n += int((clearance_col > 0).sum())
        print(f"  Clearance decade [{lo:.0e}, {hi_clipped:.2e}): {n_levels} x "
              f"{CLEARANCE_SAMPLES_PER_LEVEL} = {n_levels * CLEARANCE_SAMPLES_PER_LEVEL} samples")

    n_this = actual_contact_n + actual_clear_n
    ratio = (actual_contact_n / n_this * 100) if n_this > 0 else float("nan")
    anchor_summary.append((Z_POS, n_this, ratio))

# ============================================================
# ASSEMBLE + SAVE
# ============================================================
qw_all = np.concatenate(all_qw)
qx_all = np.concatenate(all_qx)
qy_all = np.concatenate(all_qy)
qz_all = np.concatenate(all_qz)
overlap_all = np.concatenate(all_overlap)
clearance_all = np.concatenate(all_clearance)
posz_all = np.concatenate(all_posz)

perm_shuffle = rng.permutation(len(qw_all))
df_out = pd.DataFrame({
    "position_z": posz_all[perm_shuffle],
    "Q_orientation.w()": qw_all[perm_shuffle],
    "Q_orientation.x()": qx_all[perm_shuffle],
    "Q_orientation.y()": qy_all[perm_shuffle],
    "Q_orientation.z()": qz_all[perm_shuffle],
    "overlap(m)": overlap_all[perm_shuffle],
    "clearance(m)": clearance_all[perm_shuffle],
})
df_out.to_csv(OUTPUT_PATH, index=False)

print("\n================================================")
print("PER-ANCHOR SUMMARY (post-jitter actual counts)")
print("================================================")
print(f"{'Z_POS':>10}{'N rows':>12}{'Contact %':>12}")
for z, n, cr in anchor_summary:
    print(f"{z:>10.6f}{n:>12d}{cr:>12.2f}")

print("\n================================================")
print("FINAL DATASET")
print("================================================")
print(f"Total samples: {len(df_out)}")
print(f"Number of anchors: {len(Z_POS_VALUES)}")
print(f"Overall contact ratio: {(df_out['overlap(m)'] > 0).mean() * 100:.2f}%")
print(f"Saved to: {OUTPUT_PATH}")

# ============================================================
# GLOBAL DIAGNOSTICS (unchanged — jitter doesn't affect quaternion
# distribution, only position_z / overlap / clearance columns)
# ============================================================
quat_norms = np.sqrt(qw_all**2 + qx_all**2 + qy_all**2 + qz_all**2)
print(f"\nQuaternion norm check: mean={quat_norms.mean():.6f}, std={quat_norms.std():.6e}")

rx_all = 2 * (qx_all * qz_all - qw_all * qy_all)
ry_all = 2 * (qy_all * qz_all + qw_all * qx_all)
rz_all = 1 - 2 * (qx_all**2 + qy_all**2)

abs_r = np.stack([np.abs(rx_all), np.abs(ry_all), np.abs(rz_all)], axis=1)
TOL = 1e-3
sorted_abs = np.sort(abs_r, axis=1)
near_tie = (sorted_abs[:, 2] - sorted_abs[:, 1]) < TOL
dominant_axis = np.where(near_tie, -1, np.argmax(abs_r, axis=1))
axis_names = {-1: "tied (near-corner)", 0: "x", 1: "y", 2: "z"}
unique_dom, dom_counts = np.unique(dominant_axis, return_counts=True)
print("\nDominant-axis distribution (tie-aware):")
for d, c in zip(unique_dom, dom_counts):
    print(f"  {axis_names[d]}: {c} rows ({c / len(dominant_axis) * 100:.2f}%)")

phi_all = np.arctan2(ry_all, rx_all)
print(f"\nAzimuthal angle (phi) std: {phi_all.std():.3f} (uniform -> ~1.81)")

SUBSAMPLE = min(50_000, len(df_out))
sub_idx = rng.choice(len(df_out), size=SUBSAMPLE, replace=False)
X_full = np.column_stack([
    df_out["position_z"].values, df_out["Q_orientation.w()"].values,
    df_out["Q_orientation.x()"].values, df_out["Q_orientation.y()"].values,
    df_out["Q_orientation.z()"].values,
])
tree = cKDTree(X_full)
dist, _ = tree.query(X_full[sub_idx], k=2)
nn_dist = dist[:, 1]
print(f"\nNN-distance (subsample {SUBSAMPLE}): mean={nn_dist.mean():.4e}, "
      f"median={np.median(nn_dist):.4e}, max={nn_dist.max():.4e}")
