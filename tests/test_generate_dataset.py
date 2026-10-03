"""Tests for the ss_cube-wall dataset generator (spec 02).

T02.1: configuration and validation (R02.11, R02.13).
"""

import dataclasses
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interactions", "ss_cube-wall", "python"))

from generate_dataset import (  # noqa: E402
    ConfigError,
    DEFAULT_ORIENT_FRACTIONS,
    DEFAULT_SD_FRACTIONS,
    GenConfig,
    SMOKE_MAX_ROWS,
    validate_config,
)

SEED = 20261003


def cfg(**overrides):
    """Valid smoke-size config with approved default fractions, then overrides."""
    base = dict(n_orient=1250, k_per_orient=8, seed=SEED)
    base.update(overrides)
    return GenConfig(**base)


class TestDefaults(unittest.TestCase):
    """Approved defaults (spec 02 R02.2, R02.3, R02.13)."""

    def test_valid_default_config_passes(self):
        self.assertIsNone(validate_config(cfg()))

    def test_default_fractions_are_the_approved_values(self):
        self.assertEqual(DEFAULT_SD_FRACTIONS, {"penetration": 0.40, "gap": 0.30, "near_contact": 0.05,
                                                "exact_contact": 0.01, "gap_safety": 0.24})
        self.assertEqual(DEFAULT_ORIENT_FRACTIONS, {"uniform": 0.70, "near_kink": 0.29, "special": 0.01})
        c = cfg()
        self.assertEqual(c.sd_fractions, DEFAULT_SD_FRACTIONS)
        self.assertEqual(c.orient_fractions, DEFAULT_ORIENT_FRACTIONS)
        self.assertEqual(c.sd_log_min, 1e-5)
        self.assertEqual(c.delta_range, (1e-6, 1e-1))
        self.assertFalse(c.allow_full_scale)

    def test_default_dicts_are_not_shared(self):
        a, b = cfg(), cfg()
        self.assertIsNot(a.sd_fractions, b.sd_fractions)
        self.assertIsNot(a.sd_fractions, DEFAULT_SD_FRACTIONS)

    def test_config_is_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg().seed = 1

    def test_config_error_is_value_error_and_names_parameter(self):
        with self.assertRaises(ConfigError) as ctx:
            validate_config(cfg(n_orient=0))
        self.assertIsInstance(ctx.exception, ValueError)
        self.assertEqual(ctx.exception.param, "n_orient")
        self.assertIn("n_orient", str(ctx.exception))

    def test_rejects_non_config_object(self):
        with self.assertRaises(ConfigError):
            validate_config({"n_orient": 1})


class TestInvalidCases(unittest.TestCase):
    """Each invalid case of R02.11 raises ConfigError naming the parameter."""

    def assertRejects(self, param, **overrides):
        with self.assertRaises(ConfigError, msg=f"{overrides}") as ctx:
            validate_config(cfg(**overrides))
        self.assertEqual(ctx.exception.param, param, msg=f"{overrides}: {ctx.exception}")

    # non-positive counts
    def test_non_positive_counts(self):
        for name in ("n_orient", "k_per_orient", "m_min"):
            for bad in (0, -1):
                self.assertRejects(name, **{name: bad})

    def test_non_integer_counts(self):
        for name in ("n_orient", "k_per_orient", "m_min"):
            for bad in (1.5, 2.0, True, "8", None):
                self.assertRejects(name, **{name: bad})

    def test_numpy_integer_counts_accepted(self):
        self.assertIsNone(validate_config(cfg(n_orient=np.int64(1250), k_per_orient=np.int32(8))))

    # missing seed
    def test_missing_seed(self):
        with self.assertRaises(ConfigError) as ctx:
            validate_config(GenConfig(n_orient=1250, k_per_orient=8))
        self.assertEqual(ctx.exception.param, "seed")

    def test_invalid_seed(self):
        for bad in (None, -1, 1.0, True, "1"):
            self.assertRejects("seed", seed=bad)

    def test_seed_zero_is_valid(self):
        self.assertIsNone(validate_config(cfg(seed=0)))

    # fractions negative or not summing to 1
    def test_negative_fraction(self):
        sd = dict(DEFAULT_SD_FRACTIONS, penetration=-0.1, gap=0.80)
        self.assertRejects("sd_fractions", sd_fractions=sd)
        orient = dict(DEFAULT_ORIENT_FRACTIONS, uniform=-0.29, near_kink=1.28)
        self.assertRejects("orient_fractions", orient_fractions=orient)

    def test_fractions_not_summing_to_one(self):
        self.assertRejects("sd_fractions", sd_fractions=dict(DEFAULT_SD_FRACTIONS, gap=0.31))
        self.assertRejects("orient_fractions", orient_fractions=dict(DEFAULT_ORIENT_FRACTIONS, special=0.0))

    def test_non_finite_fraction(self):
        self.assertRejects("sd_fractions", sd_fractions=dict(DEFAULT_SD_FRACTIONS, gap=float("nan")))
        self.assertRejects("orient_fractions", orient_fractions=dict(DEFAULT_ORIENT_FRACTIONS, uniform=float("inf")))

    def test_missing_or_unknown_fraction_keys(self):
        missing = {k: v for k, v in DEFAULT_SD_FRACTIONS.items() if k != "gap_safety"}
        missing["penetration"] += 0.24
        self.assertRejects("sd_fractions", sd_fractions=missing)
        unknown = dict(DEFAULT_ORIENT_FRACTIONS, special=0.0, edge=0.01)
        self.assertRejects("orient_fractions", orient_fractions=unknown)

    def test_fractions_wrong_type(self):
        self.assertRejects("sd_fractions", sd_fractions=[0.4, 0.3, 0.05, 0.01, 0.24])
        self.assertRejects("orient_fractions", orient_fractions=dict(DEFAULT_ORIENT_FRACTIONS, uniform="0.70"))

    def test_zero_fraction_allowed(self):
        # disabling a component is allowed if the rest sums to 1
        orient = {"uniform": 0.71, "near_kink": 0.29, "special": 0.0}
        self.assertIsNone(validate_config(cfg(orient_fractions=orient)))

    # ranges outside spec 01; lower bound not below upper bound
    def test_sd_log_min_range(self):
        # must satisfy 0 < sd_log_min < 0.1 (spec 01 accuracy zone upper bound)
        for bad in (0.0, -1e-5, 0.1, 0.2, float("nan"), float("inf"), "1e-5", None):
            self.assertRejects("sd_log_min", sd_log_min=bad)
        self.assertIsNone(validate_config(cfg(sd_log_min=1e-8)))

    def test_delta_range(self):
        for bad in ((0.0, 1e-1), (-1e-6, 1e-1), (1e-1, 1e-6), (1e-3, 1e-3),
                    (1e-6, float("nan")), (1e-6, float("inf")), (1e-6,), (1e-6, 1e-3, 1e-1), None, "ab"):
            self.assertRejects("delta_range", delta_range=bad)
        self.assertIsNone(validate_config(cfg(delta_range=[1e-5, 1e-2])))

    def test_exclude_angle(self):
        for bad in (-1.0, 180.1, float("nan"), None, "5"):
            self.assertRejects("exclude_angle_deg", exclude_angle_deg=bad)
        self.assertIsNone(validate_config(cfg(exclude_angle_deg=180.0)))

    def test_exclude_quats(self):
        for bad in (np.zeros(4) + 1.0, np.ones((2, 3)), np.array([[1.0, np.nan, 0, 0]]),
                    np.array([[1.0, 0, 0, 0], [0.0, 0, 0, 0]]), "q"):
            self.assertRejects("exclude_quats", exclude_quats=bad, exclude_angle_deg=1.0)
        ok = np.array([[1.0, 0, 0, 0], [0.0, 2.0, 0, 0]])  # non-unit allowed, normalized later (R01.4)
        self.assertIsNone(validate_config(cfg(exclude_quats=ok, exclude_angle_deg=1.0)))

    def test_exclude_quats_without_angle(self):
        # a list without a positive angle would exclude nothing: reported, not ignored
        self.assertRejects("exclude_angle_deg", exclude_quats=np.array([[1.0, 0, 0, 0]]), exclude_angle_deg=0.0)

    def test_allow_full_scale_must_be_bool(self):
        for bad in (1, "yes", None):
            self.assertRejects("allow_full_scale", allow_full_scale=bad)


class TestSizeLimit(unittest.TestCase):
    """R02.13 / R02.11: above the smoke limit only with explicit override."""

    def test_smoke_limit_value(self):
        self.assertEqual(SMOKE_MAX_ROWS, 10_000)

    def test_at_limit_passes(self):
        self.assertIsNone(validate_config(cfg(n_orient=1250, k_per_orient=8)))

    def test_above_limit_refused(self):
        with self.assertRaises(ConfigError) as ctx:
            validate_config(cfg(n_orient=1251, k_per_orient=8))
        self.assertEqual(ctx.exception.param, "allow_full_scale")
        self.assertIn("10008", str(ctx.exception))

    def test_above_limit_with_override_passes(self):
        self.assertIsNone(validate_config(cfg(n_orient=10**6, k_per_orient=8, allow_full_scale=True)))


class TestNoSideEffects(unittest.TestCase):
    """R02.11: validation stops before writing anything."""

    def test_validation_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.getcwd()
            os.chdir(d)
            try:
                validate_config(cfg())
                with self.assertRaises(ConfigError):
                    validate_config(cfg(n_orient=0))
            finally:
                os.chdir(old)
            self.assertEqual(os.listdir(d), [])


if __name__ == "__main__":
    unittest.main()
