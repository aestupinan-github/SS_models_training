"""Contract smoke test for the ss_cube-wall reference solution (spec 01).

This is a SMOKE TEST of the software against the exact solution, not a
scientific result and not a validation of any surrogate.

Usage (from the project root):
    ../env_folder/bin/python -B interactions/ss_cube-wall/python/smoke_contract.py [--n 5000] [--seed 20261003]

Exit code 0 if all checks pass, 1 otherwise.
"""

import argparse
import itertools
import math
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

import cube_wall as cw


def wxyz(rot):
    x, y, z, w = rot.as_quat()
    return [w, x, y, z]


def symmetries():
    out = []
    for p in itertools.permutations(range(3)):
        for sg in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3))
            for i in range(3):
                m[i, p[i]] = sg[i]
            if np.isclose(np.linalg.det(m), 1.0):
                out.append(R.from_matrix(m))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261003)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rots = R.random(args.n, random_state=args.seed)
    sym = symmetries()
    results = []  # (requirement, description, n_fail, n_total, worst)

    def record(req, desc, errs, tol):
        errs = np.asarray(errs, dtype=float)
        results.append((req, desc, int((errs > tol).sum()), errs.size, float(errs.max()), tol))

    qs = [wxyz(r) for r in rots]
    g = np.array([cw.z_touch(q) for q in qs])

    # R01.5: sd = z - g, exact zero at z = g
    z = rng.uniform(0.0, 1.5, args.n)
    sd = np.array([cw.compute_signed_distance(q, zz) for q, zz in zip(qs, z)])
    record("R01.5", "sd == z - g(q)", np.abs(sd - (z - g)), 1e-12)
    record("R01.5", "sd(z = g(q)) == 0", [abs(cw.compute_signed_distance(q, gg)) for q, gg in zip(qs, g)], 1e-12)

    # R01.14
    record("R01.14", "slope d sd/dz == 1",
           [abs(cw.compute_signed_distance(q, 1.0) - cw.compute_signed_distance(q, 0.0) - 1.0) for q in qs], 1e-12)
    record("R01.14", "yaw invariance of g",
           [abs(cw.z_touch(wxyz(R.from_euler("z", a) * r)) - gg) for r, gg in zip(rots, g)
            for a in [rng.uniform(0, 2 * math.pi)]], 1e-12)
    n_sym = min(args.n, 500)
    record("R01.14", "24 cube symmetries invariance of g",
           [abs(cw.z_touch(wxyz(r * s)) - gg) for r, gg in zip(rots[:n_sym], g[:n_sym]) for s in sym], 1e-12)
    record("R01.14", "q and -q give the same g",
           [abs(cw.z_touch(q) - cw.z_touch([-c for c in q])) for q in qs], 0.0)
    record("R01.14", "g >= 0.5", np.maximum(0.5 - g, 0.0), 1e-12)
    record("R01.14", "g <= sqrt(3)/2", np.maximum(g - math.sqrt(3) / 2, 0.0), 1e-12)

    # R01.4: canonical form unique per orientation and sign-invariant
    def canon_diff(q):
        a = np.array(cw.canonicalize_quaternion(q))
        b = np.array(cw.canonicalize_quaternion([-c for c in q]))
        return np.abs(a - b).max()
    record("R01.4", "canonical(q) == canonical(-q)", [canon_diff(q) for q in qs], 0.0)
    record("R01.4", "canonical qw >= 0", [max(0.0, -cw.canonicalize_quaternion(q)[0]) for q in qs], 0.0)
    special = [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0.6, -0.8, 0], [0, 0, -0.6, 0.8]]
    record("R01.4", "qw == 0 cases map to one output", [canon_diff(q) for q in special], 0.0)
    bad_ok = 0
    bad = [[0, 0, 0, 0], [np.nan, 0, 0, 1], [1, np.inf, 0, 0]]
    for b in bad:
        try:
            cw.canonicalize_quaternion(b)
        except ValueError:
            bad_ok += 1
    record("R01.4", "invalid quaternions rejected", [float(len(bad) - bad_ok)], 0.0)

    # R01.7: rescaling
    Ls = rng.uniform(0.1, 5.0, args.n)
    record("R01.7", "L * sd_canonical(z / L) == rescaled",
           [abs(cw.from_canonical(cw.compute_signed_distance(q, cw.to_canonical(zz * L, L)), L)
                - L * (zz - gg)) for q, zz, L, gg in zip(qs, z, Ls, g)], 1e-11)

    # R01.13: classification and pre-filter
    zz = rng.uniform(0.0, 1.5, args.n)
    far_bad = sum(1 for q, z_ in zip(qs, zz) if cw.is_far(z_) and cw.classify_sd(cw.compute_signed_distance(q, z_)) != "far")
    record("R01.13", "pre-filter never contradicts exact sd", [float(far_bad)], 0.0)

    print("SMOKE TEST (not a scientific result): ss_cube-wall contract, n=%d, seed=%d" % (args.n, args.seed))
    failed = 0
    for req, desc, nf, nt, worst, tol in results:
        status = "PASS" if nf == 0 else "FAIL"
        failed += nf
        print(f"  [{status}] {req:7s} {desc:42s} fail {nf}/{nt}  worst {worst:.2e}  tol {tol:.0e}")
    print("RESULT:", "ALL PASS" if failed == 0 else f"{failed} FAILURES")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
