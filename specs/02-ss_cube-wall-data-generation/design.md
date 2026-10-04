# Spec 02 — Design

## Status

**Revision 2, 2026-10-03. APPROVED by the user (2026-10-03).** Follows `requirements.md`
revision 2 (approved). Supersedes design revision 1 (git history).

### Decisions delegated to the agent (user: "you decide, but record this", 2026-10-03)

- **D02.1 Extra columns `sd_part` and `delta`: adopted.** They are stored
  after the R02.6 columns, whose order is unchanged. Reason: the coverage
  table (R02.5) and later per-part error analysis need them, and
  recomputing `delta` later would duplicate logic.
- **D02.2 `.gitignore`: add `interactions/*/data/dataset_*/`.** Datasets are
  never committed; their provenance and README listing are the record. The
  other `.gitignore`/`.gitkeep` gaps (review item A5) stay a separate change.
- **D02.3 Exclusion by body-frame direction `u`, not by full SO(3)
  orientation.** Reason: `sd` depends only on `u`, and every sampled
  orientation gets random spin and yaw, so excluding one quaternion would
  still leave the same `u` (same `sd`) in training. Rule: a sampled
  orientation is rejected if the angle between its `u` and the `u` of any
  excluded orientation is below `exclude_angle_deg`. *Confirmed* (measured,
  20000 pairs): the `u` angle never exceeds the SO(3) angle, so this rule
  removes a superset of the per-orientation rule. Symmetry classes are not
  handled by the generator, because they are shape-specific (unknown for an
  arbitrary STL); spec 04 can put the symmetric images of a held-out `u`
  into the exclusion list itself.

### Decision made during implementation (T02.5, agent, within the approved design)

- **D02.4 Stratified vs random `sd` parts, and rounding.** The design says
  "`round(fraction × K)` rows per part, remainder to the largest part ...
  for small `K`, parts are assigned by a seeded random draw". "Small" was not
  defined. Measured: with the default fractions, `K = 8` gives
  `near_contact = 0` and `exact_contact = 0` per orientation, and Python's
  `round` (half to even) gives `exact_contact = 0` at `K = 50`. Decision:
  rounding is half up; stratified allocation is used only if every part
  with a positive fraction gets at least one row per orientation; otherwise
  every row's part is a seeded random draw with the fractions as
  probabilities. The same `allocate_counts` rule splits `n_orient` over the
  orientation components. With the default fractions, stratified needs
  `K >= 50`. Recorded here because the user asked that delegated choices be
  recorded.
- **D02.5 Acceptance-check details (T02.6, agent, within R02.12).**
  (1) An extra check `h_consistent` (`|h - h_ref| <= 1e-12`) is added next to
  `label`: if a row's `h` and `sd` were wrong by the same amount,
  `position_z - h` would still equal `sd`, so `h_ref` is always recomputed
  from the stored quaternion. (2) "Inside the R02.2 ranges" is checked per
  `sd` part (e.g. a gap row in the near-contact band fails), plus the global
  bound `sd >= -0.1`; exact-contact rows must be exactly 0. (3) Coverage
  decades: lower bound inclusive, upper exclusive, the last decade includes
  0.1; `n_fail` of `coverage` counts failing cells (13 cells: 4 + 4 + 5).
  (4) Checks are pure and return JSON-serializable results;
  `require_acceptance` raises `GenerationError` naming each failed check.
- **D02.6 Writer and seeding details (T02.7, agent, within R02.7-R02.10).**
  (1) **Two seed streams, not four.** The design listed "orientations,
  near_kink, sd, yaw". Spin and yaw are drawn inside each orientation
  sampler, so separate streams would only add coupling. Implemented:
  `SeedSequence(seed).spawn(2)` -> `orientations`, `sd`; entropy and spawn
  keys recorded, which reproduces each stream. (2) **Safe publish.**
  Measured: `os.rename` onto an existing *empty* directory silently
  replaces it on Linux. The target is therefore first claimed with
  `os.mkdir` (fails if it exists), then renamed; an existing target, even
  empty, raises `GenerationError` and nothing is replaced. (3) **CSV.**
  Floats `%.17g` (measured: bitwise float64 round trip on 1e5 values),
  `orientation_id` integer, `component` and `sd_part` as names. CSV above
  1e4 rows raises `ConfigError` (R02.7). (4) **Provenance.** Exclusion list
  recorded by count and sha256 of its float64 bytes; shape recorded with its
  hull vertices and constants; timestamp UTC with microseconds;
  `git_commit` and `git_dirty` are `null` if git is unavailable.
  (5) `.gitignore` also ignores `.tmp_dataset_*/` build folders.

## Requirements traceability

| Design element | Requirements |
|---|---|
| Shape description from a point set (hull, facets, edges, constants) | R02.1, R02.3 |
| `sd` sampler (parts and fractions) | R02.2, R02.4 |
| Orientation sampler (`uniform`, `near_kink`, `special`, yaw, exclusion) | R02.3 |
| Row assembly and label recomputation | R02.1, R02.4, R02.6 |
| Coverage table | R02.5 |
| Writer (`.npz`, CSV export, folder versioning) | R02.6, R02.7, R02.8 |
| Seeding | R02.9 |
| Provenance record | R02.10 |
| Configuration validation and failure handling | R02.11, R02.13 |
| Acceptance checks | R02.12 |
| Summary and README listing | R02.14 |

## Existing code context

- `interactions/ss_cube-wall/python/cube_wall.py` (spec 01 rev 3):
  `CUBE_VERTICES`, `canonicalize_quaternion`, `support_height`, `z_touch`,
  `compute_signed_distance`, constants `R_CIRCUMSCRIBED`, `H_MIN`,
  `Z_PREFILTER`, `SD_COVER_MAX`, `SD_GAP_MAX`, `SD_PENETRATION_MAX`.
- Tests: `tests/`, run with `../env_folder/bin/python -B -m unittest discover
  tests` from the project root.
- Environment (*Confirmed*): Python env `../env_folder`, NumPy 2.3.1,
  SciPy 1.17.1. No new dependency is introduced.

## Files and responsibilities

| File | Responsibility | Change |
|---|---|---|
| `interactions/ss_cube-wall/python/generate_dataset.py` | Generator: config, samplers, rows, checks, writer, provenance, CLI | New |
| `interactions/ss_cube-wall/python/cube_wall.py` | Reference solution (spec 01) | Unchanged; imported |
| `tests/test_generate_dataset.py` | Unit tests of the generator | New |
| `interactions/ss_cube-wall/README.md` | Dataset listing | Updated per dataset |
| `.gitignore` | Ignore generated dataset folders | Extended: `interactions/*/data/dataset_*/` |

The generator lives in the interaction folder (R00.3, R00.5). It is
shape-general internally (takes a point set); future interactions copy it.

## Interfaces

```
@dataclass(frozen=True)
class GenConfig:
    n_orient: int                    # orientations
    k_per_orient: int                # rows per orientation
    seed: int                        # master seed
    sd_fractions: dict               # penetration .40, gap .30, near_contact .05,
                                     # exact_contact .01, gap_safety .24
    orient_fractions: dict           # uniform .70, near_kink .29, special .01
    sd_log_min: float = 1e-5         # accuracy zone lower bound of |sd|
    delta_range: tuple = (1e-6, 1e-1)
    exclude_quats: ndarray | None = None   # (M, 4) held-out orientations
    exclude_angle_deg: float = 0.0
    m_min: int = 1                   # coverage minimum per decade cell
    allow_full_scale: bool = False   # override for > 1e4 rows

validate_config(cfg) -> None                 # raises ConfigError (R02.11)
shape_from_points(points) -> Shape           # hull facets, edges, R_CIRC, H_MIN
sample_orientations(cfg, shape, rng) -> (quats (n,4), component (n,))
sample_sd(cfg, h, rng) -> sd (n*k,)          # per orientation h
build_rows(cfg, shape, quats, component, sd) -> dict of column arrays
coverage_table(rows, shape) -> dict          # R02.5
acceptance_checks(rows, cfg, shape) -> dict  # R02.12, each check pass/fail + worst value
write_dataset(rows, cfg, out_root, csv=False) -> Path   # new dataset_XXX folder
generate(cfg, out_root, csv=False) -> Path   # whole pipeline
```

CLI: `python generate_dataset.py --n-orient N --k K --seed S [--csv]
[--allow-full-scale] [--exclude FILE --exclude-angle DEG]`.

## Data and units

- All values dimensionless canonical units (`L = 1`), float64.
- Quaternion `(qw, qx, qy, qz)`, body to world, canonical per R01.4.
- `u = -R(q)^T e_z`: body-frame direction towards the wall. `h = max_v v·u`.
- Columns (R02.6): `position_z, qw, qx, qy, qz, sd, h` (float64),
  `orientation_id` (int64), `component` (int8 code: 0 `uniform`, 1
  `near_kink`, 2 `special`; names stored in the provenance and in the CSV as
  strings).
- Also stored per row (needed for the coverage table, R02.5): `sd_part`
  (int8 code: 0 penetration, 1 gap, 2 near_contact, 3 exact_contact,
  4 gap_safety) and `delta` (float64, support gap). *Proposal:* these two
  extra columns are added after the R02.6 columns; they do not change the
  R02.6 order.

## Algorithm

### Shape description (shape-general)

```
hull = scipy ConvexHull(points)
facets: merge coplanar hull simplices (normals equal within 1e-12)
        -> unit outward normals n_f and vertex sets
edges:  pairs of facets sharing exactly two hull vertices
R_CIRC = max |v|;  H_MIN = min_f offset_f  (distance centre -> facet plane)
```

Cube (*Confirmed*, measured): 6 facets, 12 edges, 8 vertices, `H_MIN = 0.5`.
`shape.H_MIN` and `shape.R_CIRC` must equal `cube_wall.H_MIN` and
`cube_wall.R_CIRCUMSCRIBED` (checked).

### Orientation sampling (R02.3)

Each component draws body-frame directions `u` or rotations, then:

```
uniform:   rot = scipy Rotation.random(n, rng)              # Haar on SO(3)
special:   u in {facet normals} ∪ {edge midpoint directions} ∪ {hull vertex directions},
           drawn uniformly from that finite set
near_kink: target Δ* log-uniform in delta_range;
           half of the rows: face mode   u0 = facet normal n_f, w ⟂ u0 random
           other half:      edge mode   u0 on the great arc between the two
                            facet normals of an edge (uniform in arc angle),
                            w = ±(n_i × n_j) normalised
           u(θ) = cos θ u0 + sin θ w;  bisection on θ in [0, 0.5] (60 steps)
           so that Δ(u(θ)) = Δ*
```

For `special` and `near_kink`, the rotation is built from `u`:
`rot0` = minimal rotation mapping body `u` to world `-e_z`, then a uniform
random rotation about body `u` (spin), then for all components a uniform
random yaw about world `z`: `rot = Rz(ψ) · rot0`. Finally `q` is canonicalized
(R01.4).

*Confirmed* (measured on the cube, 50000 samples, 1.8 s): the bisection hits
the target `Δ` within relative error `2e-10`, about 1e4 rows per `Δ` decade
over `[1e-6, 1e-1]`. In 1.7% of samples the bracket `θ ≤ 0.5` does not reach
`Δ*` (large targets); those samples are redrawn.

*Confirmed* (T02.4, measured, cube, 20000 samples): including the full
quaternion chain (spin, yaw, canonicalization), worst relative `Δ` error is
6.5e-10, absolute below 1e-15. Edge-mode starts are uniform along the whole
arc between two face normals, so they also land near faces: 55% of
`near_kink` rows are within `|u|_max > 0.99` of a face and 45% lie on a kink
circle away from faces (uniform SO(3): 3% and 28% near a circle). For a
shape larger than the cube (tetrahedron, `r_circ = 1.73`) the same absolute
round-off gives up to 1.5e-9 relative at `Δ ≈ 1e-6`.

Exclusion (D02.3): for each excluded quaternion compute its `u`. A sampled
orientation is redrawn if `arccos(u · u_excl) < exclude_angle_deg` for any
excluded `u_excl`. Symmetric images, if wanted, are listed explicitly by
spec 04.

### `sd` sampling (R02.2, R02.4)

For each orientation, `K` rows. Each row is assigned an `sd` part by
stratified allocation: `round(fraction × K)` rows per part, remainder to the
largest part, so each orientation gets every part when `K` is large enough;
for small `K`, parts are assigned by a seeded random draw with the fractions
as probabilities. "Small `K`" and the rounding rule are fixed by D02.4.

```
penetration:   sd = -10^U(log10 1e-5, log10 0.1)
gap:           sd = +10^U(log10 1e-5, log10 0.1)
near_contact:  sd = U(-1e-5, 1e-5)
exact_contact: sd = 0
gap_safety:    hi = Z_PREFILTER - h;  if hi > 0.1: sd = U(0.1, hi)  else  draw as "gap"
position_z = h + sd
```

### Row assembly and labels (R02.1)

```
h_ref  = support height recomputed from stored q (float64)
sd     = position_z - h_ref        # stored label, recomputed (R02.1)
```

`sd` differs from the drawn target by rounding only (order `1e-16`).

### Acceptance checks (R02.12)

Vectorized over all rows: label tolerance `1e-12`, ranges, finiteness, unit
norm within `1e-12`, canonical form, coverage `>= m_min` per decade cell,
exclusion distance. Each check records pass/fail, number of failing rows and
the worst value. Any failure aborts the write (R02.11). Details: D02.5.

### Writer and versioning (R02.7, R02.8)

```
root = interactions/ss_cube-wall/data
next = 1 + max existing dataset_XXX number (0 if none)
tmp  = root/.tmp_dataset_XXX_<pid>   # build here
write data.npz (np.savez, uncompressed), optional data.csv, provenance.json
os.rename(tmp, root/dataset_XXX)     # atomic on the same filesystem
```

If `dataset_XXX` appears between counting and renaming, the rename fails and
the generator stops without overwriting. A failed run leaves only a
`.tmp_*` folder, never a folder that looks complete.

`np.savez` output is byte-identical across runs for the same arrays
(*Confirmed*, measured), so the data checksum is reproducible.

### Seeding (R02.9)

```
ss = numpy SeedSequence(seed)
children = ss.spawn(4) -> streams: orientations, near_kink, sd, yaw
```

The spawn keys are recorded in the provenance. Implemented with two streams
(`orientations`, `sd`), see D02.6.

### Provenance (R02.10)

`provenance.json`, written last:

```
{ "spec": {"01": "rev 3", "02": "rev 2"},
  "generator": "generate_dataset.py", "git_commit": ..., "git_dirty": bool,
  "versions": {"python": ..., "numpy": ..., "scipy": ...},
  "config": {...all GenConfig fields...},
  "seed": S, "spawn_keys": {...},
  "shape": {"points": "cube_wall.CUBE_VERTICES", "R_CIRC": ..., "H_MIN": ...},
  "n_rows": ..., "columns": [...], "codes": {"component": ..., "sd_part": ...},
  "timestamp_utc": "...Z",
  "sha256": {"data.npz": ..., "data.csv": ...},
  "coverage": {...}, "acceptance": {...} }
```

The data file is not timestamped internally; only the provenance carries the
time.

## Design decisions

- **Bisection on tilt angle for `near_kink`**, rather than oversampling
  uniform orientations and filtering: filtering needs about `1e6/18` draws per
  row at `Δ ~ 1e-6`. Bisection is exact to `2e-10` and uses only hull
  facets and edges, which exist for any polyhedral hull (R01.16).
- **Spin about `u` and yaw about world `z`**: `sd` depends only on `u`, but
  the model sees the full quaternion, so all quaternions with the same `u`
  must appear.
- **Stratified `sd` allocation per orientation**: every orientation sees
  penetration and gap rows, which helps an orientation-based split.
- **`.npz` uncompressed**: byte-reproducible and fast. Compressed `.npz`
  was not chosen: reproducibility of zlib output across versions was not
  checked.
- **Generation-time checks abort the write**: an invalid dataset never lands
  in `data/`.

## Error handling and diagnostics

- `ConfigError` (subclass of `ValueError`) for invalid configuration, raised
  before any file is created, naming the parameter.
- `GenerationError` for non-finite values, reference-solution errors, failed
  checks, or write failures; the temporary folder is left for inspection and
  the message names it.
- Size above `1e4` rows without `allow_full_scale`: `ConfigError`.
- Summary printed at the end: size, per-component and per-part counts,
  coverage table, acceptance results, output path.

## Testing and validation strategy

| Test | Covers |
|---|---|
| `shape_from_points(CUBE_VERTICES)`: 6 facets, 12 edges, `R_CIRC`, `H_MIN` match `cube_wall` | R02.1 |
| `special` orientations have `Δ = 0` for faces and edges; corner-down `h = sqrt(3)/2` | R02.3 |
| `near_kink`: achieved `Δ` within `1e-9` relative of target; decades covered | R02.3, R02.5 |
| `uniform`: canonical quaternions, yaw uniformity (mean of `cos ψ`, `sin ψ` near 0) | R02.3 |
| Exclusion: no row whose `u` is within the angle of an excluded `u` (D02.3) | R02.3 |
| `sd` parts: ranges per part; gap safety falls back when `Z_PREFILTER - h <= 0.1` | R02.2 |
| Labels: `sd = position_z - h_ref` within `1e-12` | R02.1, R02.12 |
| Columns, dtypes, order; CSV header and 17-digit round trip | R02.6, R02.7 |
| Same seed twice: identical arrays and identical `data.npz` sha256 | R02.9 |
| Different seed: different arrays | R02.9 |
| Invalid configs (each R02.11 case) raise `ConfigError` and create no folder | R02.11 |
| Existing `dataset_XXX` never overwritten; counter increments | R02.8 |
| Size above `1e4` without override refused | R02.13 |
| Acceptance check fails on a corrupted row (mutation) | R02.12 |
| Smoke dataset end to end: `.npz` + CSV + provenance; all checks pass | completion criteria |

Tests write into a temporary directory, not into `data/`.

## Risks and limitations

- The `special` set for the cube includes 6 face, 12 edge and 8 vertex
  directions; for non-polyhedral shapes (smooth STL with many facets) the set
  becomes large and `near_kink` concentrates on many short edges. To be
  revisited when a second shape is added.
- Exclusion is by body-frame direction `u` (D02.3); symmetric images are not
  added automatically.
- Bitwise reproducibility holds only for the same NumPy/SciPy versions.
