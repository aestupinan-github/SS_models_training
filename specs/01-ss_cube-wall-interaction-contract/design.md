# Spec 01 — Design

## Status

Revision 3, 2026-10-03. Follows `requirements.md` revision 3 (amendment
approved by the user). Revision 2 approved by the user (2026-10-03).

## Requirements traceability

| Design element | Requirements |
|---|---|
| Geometry and frames | R01.1, R01.2, R01.3 |
| Quaternion normalization and canonicalization | R01.4 |
| Reference solution `g(q)`, `sd` | R01.5, R01.14 |
| Model I/O and rescaling | R01.6, R01.7 |
| Shape constants, zones, pre-filter | R01.11, R01.12, R01.13 |
| Generic support function `h(q)` from a point set | R01.16 |
| Contract smoke test | R01.4, R01.5, R01.7, R01.11, R01.13, R01.14, R01.16 |
| Sign safety and continuity metrics | R01.17, R01.18 (implemented in spec 04) |

## Existing code context

- `interactions/ss_cube-wall/python/cube_wall.py`: `CUBE_VERTICES`,
  `canonicalize_quaternion`, `z_touch`, `compute_signed_distance`.
- `tests/test_overlap_contract.py`: unit tests of the above (25 tests across
  the repo at revision 1).
- Run tests with `../env_folder/bin/python -B -m unittest discover tests`
  from the project root.

## Files and responsibilities

| File | Responsibility | Expected change |
|---|---|---|
| `interactions/ss_cube-wall/python/cube_wall.py` | Reference solution and contract helpers | Rework `canonicalize_quaternion`; add `rescale_*`, `classify_sd`, `is_far`; rev 3: shape constants, `classify_zone`, `support_height` |
| `tests/test_overlap_contract.py` | Unit tests for the contract | Add tests for R01.4, R01.7, R01.13, R01.14 |
| `interactions/ss_cube-wall/python/smoke_contract.py` | Contract smoke test (randomized, reported as smoke test) | New |

## Interfaces (reference solution, `float64`)

```
canonicalize_quaternion(q) -> list[4]      # R01.4
z_touch(q) -> float                        # g(q), R01.5
compute_signed_distance(q, z_center) -> float   # sd, R01.5
to_canonical(position_z, L_actual) -> float     # position_z / L_actual
from_canonical(sd_canonical, L_actual) -> float # sd_canonical * L_actual
classify_sd(sd) -> "far" | "in_range" | "too_deep"   # accuracy-zone test (rev 2)
is_far(position_z_canonical) -> bool       # R01.13 pre-filter
# rev 3
support_height(points, q) -> float         # h(q) = max_v v . u, u = -R(q)^T e_z (R01.16)
classify_zone(sd) -> "too_deep" | "accuracy" | "gap_safety" | "beyond"   # R01.13
R_CIRCUMSCRIBED, H_MIN, Z_PREFILTER, SD_COVER_MAX   # shape constants (R01.13)
```

`classify_sd` keeps its rev 2 meaning (accuracy zone vs outside); `far`
means `sd > 0.1`, which now includes the gap safety zone. `classify_zone`
is the rev 3 classification. Shape constants are derived from the point set
(`R_CIRCUMSCRIBED = max |v|`, `H_MIN` from the cube face-down value 0.5),
not hard-coded separately.

Errors: `canonicalize_quaternion` raises `ValueError` for a zero norm or a
non-finite component. `to_canonical` and `from_canonical` raise `ValueError`
for `L_actual <= 0` or non-finite values.

## Data and units

- All quantities dimensionless, canonical `L = 1`.
- Quaternion `(qw, qx, qy, qz)`, scalar first, body to world. Converted to
  scipy's `(x, y, z, w)` order inside `_rotate_vertices` only.
- Rescaling:
  `position_z_canonical = position_z / L_actual`;
  `sd_actual = sd_canonical * L_actual`.

## Algorithm

Quaternion canonicalization:

```
q = float64 array of 4
if not all finite or norm == 0: raise ValueError
q = q / norm
if q[0] < 0: q = -q
elif q[0] == 0:
    first nonzero of q[1], q[2], q[3]; if it is negative: q = -q
return q
```

Reference solution:

```
R = rotation(q)                        # body -> world
g = -min_over_vertices (R v)_z
sd = position_z - g
```

Range classification: `sd > 0.1` is `far`, `sd < -0.1` is `too_deep`,
otherwise `in_range`. Pre-filter: `position_z_canonical > Z_PREFILTER`
(`= sqrt(3)/2 + 0.1`) implies `far`.

Zones (rev 3): `sd < -0.1` too_deep; `<= 0.1` accuracy; `<= SD_COVER_MAX`
gap_safety; else beyond. `SD_COVER_MAX = Z_PREFILTER - H_MIN`.

Generic support function (rev 3):

```
u = -R(q)^T e_z            # body-frame direction towards the wall
h = max over points v of (v . u)
sd = position_z - h
```

`H_MIN` for an arbitrary point set is the minimum of `h` over SO(3), which
is the smallest distance from the centre to a hull face plane. For the cube
it is 0.5 (exact). For later shapes it is computed from the hull, not by
sampling.

## Design decisions

- **Pre-filter uses the circumscribed radius**, not the cube's max `g`. Both
  equal `sqrt(3)/2` for the cube, but the circumscribed-sphere radius exists
  for every convex shape (R01.16, testbed status).
- **No clamp.** The old clamp hid out-of-range queries. The alternative,
  clamping to the trained range, was rejected because it returns a plausible
  but wrong overlap.
- **Tie-break at `qw == 0`** uses the first nonzero vector component. The
  alternative, choosing by largest magnitude, was rejected because it is
  less stable when components are nearly equal.
- **Exact `qw == 0` test** after normalization: inputs with `qw` of order
  `1e-17` are a different numeric branch but the same orientation. This is a
  known limitation. The surrogate is insensitive to the sign flip only if its
  input layer is sign-invariant (spec 03).

## Error handling

Invalid quaternions and sizes raise `ValueError` with a message naming the
argument. No silent NaN propagation.

## Testing and validation strategy

| Test | Covers |
|---|---|
| Quaternion: unit input unchanged; negative `qw` flipped; `qw == 0` sign pair maps to one output; non-unit input normalized; zero and NaN rejected | R01.4 |
| Signed distance: face, edge, corner contact is zero; penetration negative; gap positive | R01.5 |
| Rescale round trip; canonical cube with `L_actual = 1` unchanged; `L_actual = 2` scales `sd` by 2 | R01.7 |
| Classification boundaries and pre-filter consistency with exact `sd` | R01.13 |
| Slope, yaw, 24 symmetries, `q`/`-q`, range of `g` on random poses (fixed seed) | R01.14 |
| Smoke test script: the above on a few thousand random poses, printed as pass/fail counts | all |

The smoke test checks software behavior against the exact solution. It is
not scientific validation of a surrogate.

## Testing additions (rev 3)

| Test | Covers |
|---|---|
| `support_height` equals `z_touch` for the cube on random poses; works for a non-convex point set (interior points do not change `h`) | R01.16 |
| Shape constants: `R_CIRCUMSCRIBED = sqrt(3)/2`, `H_MIN = 0.5`, `SD_COVER_MAX = Z_PREFILTER - H_MIN` | R01.13 |
| Zone boundaries | R01.13 |
| No unseen inputs: every random query with `position_z <= Z_PREFILTER` has exact `sd <= SD_COVER_MAX`; face-down at `Z_PREFILTER` attains it | R01.11, R01.12 |

R01.17 and R01.18 concern the surrogate; their metrics are designed in spec 04.

## Risks and limitations

- The rotation sense (body to world) and the size parameter must still be
  verified in the integration project.
- Exact `qw == 0` branching is a discontinuity in the canonical form.
