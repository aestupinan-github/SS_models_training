"""Dataset generator for the ss_cube-wall interaction (spec 02 rev 2).

Implemented so far: T02.1, configuration and validation (R02.11, R02.13).
Later tasks (T02.2-T02.8) add the shape description, samplers, rows,
checks, writer and CLI.

All quantities are dimensionless canonical units (L = 1), see spec 01.
"""

import math
import numbers
from dataclasses import dataclass, field

import numpy as np

# Largest dataset that may be generated without explicit approval (R02.13).
SMOKE_MAX_ROWS = 10_000

# Approved default fractions (spec 02 R02.2, R02.3; user approval 2026-10-03).
DEFAULT_SD_FRACTIONS = {
    "penetration": 0.40,
    "gap": 0.30,
    "near_contact": 0.05,
    "exact_contact": 0.01,
    "gap_safety": 0.24,
}
DEFAULT_ORIENT_FRACTIONS = {
    "uniform": 0.70,
    "near_kink": 0.29,
    "special": 0.01,
}

# Upper bound of the accuracy zone (spec 01 R01.11): |sd| <= 0.1.
SD_ACCURACY_MAX = 0.1
# Tolerance for "fractions sum to 1".
FRACTION_SUM_TOL = 1e-9


class ConfigError(ValueError):
    """Invalid generator configuration (R02.11). `param` names the parameter."""

    def __init__(self, param, message):
        super().__init__(f"invalid configuration parameter '{param}': {message}")
        self.param = param


@dataclass(frozen=True)
class GenConfig:
    """Generator configuration (spec 02 design, Interfaces).

    `seed` has no default on purpose: a missing seed is a configuration
    error (R02.11), so it defaults to None and is rejected by validation.
    """

    n_orient: int
    k_per_orient: int
    seed: int = None
    sd_fractions: dict = field(default_factory=lambda: dict(DEFAULT_SD_FRACTIONS))
    orient_fractions: dict = field(default_factory=lambda: dict(DEFAULT_ORIENT_FRACTIONS))
    sd_log_min: float = 1e-5
    delta_range: tuple = (1e-6, 1e-1)
    exclude_quats: object = None
    exclude_angle_deg: float = 0.0
    m_min: int = 1
    allow_full_scale: bool = False

    @property
    def n_rows(self):
        return self.n_orient * self.k_per_orient


def _is_int(x):
    return isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))


def _is_real(x):
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _check_positive_int(name, value):
    if not _is_int(value):
        raise ConfigError(name, f"must be an integer, got {value!r}")
    if value <= 0:
        raise ConfigError(name, f"must be positive, got {value}")


def _check_fractions(name, fractions, expected_keys):
    if not isinstance(fractions, dict):
        raise ConfigError(name, f"must be a dict, got {type(fractions).__name__}")
    keys = set(fractions)
    if keys != set(expected_keys):
        missing = sorted(set(expected_keys) - keys)
        unknown = sorted(keys - set(expected_keys))
        raise ConfigError(name, f"keys must be {sorted(expected_keys)}; missing {missing}, unknown {unknown}")
    for key, value in fractions.items():
        if not _is_real(value) or not math.isfinite(value):
            raise ConfigError(name, f"fraction '{key}' must be a finite number, got {value!r}")
        if value < 0.0:
            raise ConfigError(name, f"fraction '{key}' must not be negative, got {value}")
    total = math.fsum(fractions.values())
    if abs(total - 1.0) > FRACTION_SUM_TOL:
        raise ConfigError(name, f"fractions must sum to 1, got {total!r}")


def _check_exclusion(cfg):
    angle = cfg.exclude_angle_deg
    if not _is_real(angle) or not math.isfinite(angle):
        raise ConfigError("exclude_angle_deg", f"must be a finite number, got {angle!r}")
    if not 0.0 <= angle <= 180.0:
        raise ConfigError("exclude_angle_deg", f"must be in [0, 180] degrees, got {angle}")
    if cfg.exclude_quats is None:
        return
    try:
        q = np.asarray(cfg.exclude_quats, dtype=np.float64)
    except (TypeError, ValueError):
        raise ConfigError("exclude_quats", "must be an (M, 4) array of quaternions") from None
    if q.ndim != 2 or q.shape[1] != 4 or q.shape[0] == 0:
        raise ConfigError("exclude_quats", f"must have shape (M, 4) with M > 0, got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ConfigError("exclude_quats", "has non-finite components")
    if np.any(np.linalg.norm(q, axis=1) == 0.0):
        raise ConfigError("exclude_quats", "has a zero-norm quaternion")
    if angle == 0.0:
        raise ConfigError("exclude_angle_deg", "must be positive when exclude_quats is given "
                                               "(an angle of 0 would exclude nothing)")


def validate_config(cfg):
    """Validate a GenConfig. Raises ConfigError naming the parameter (R02.11).

    Pure check: never writes or creates anything. Also enforces the smoke
    size limit unless `allow_full_scale` is set (R02.13).
    """
    if not isinstance(cfg, GenConfig):
        raise ConfigError("cfg", f"must be a GenConfig, got {type(cfg).__name__}")

    _check_positive_int("n_orient", cfg.n_orient)
    _check_positive_int("k_per_orient", cfg.k_per_orient)
    _check_positive_int("m_min", cfg.m_min)

    if cfg.seed is None:
        raise ConfigError("seed", "is missing; a fixed, recorded seed is required (R02.9)")
    if not _is_int(cfg.seed) or cfg.seed < 0:
        raise ConfigError("seed", f"must be a non-negative integer, got {cfg.seed!r}")

    _check_fractions("sd_fractions", cfg.sd_fractions, DEFAULT_SD_FRACTIONS)
    _check_fractions("orient_fractions", cfg.orient_fractions, DEFAULT_ORIENT_FRACTIONS)

    lo = cfg.sd_log_min
    if not _is_real(lo) or not math.isfinite(lo) or not 0.0 < lo < SD_ACCURACY_MAX:
        raise ConfigError("sd_log_min", f"must satisfy 0 < sd_log_min < {SD_ACCURACY_MAX}, got {lo!r}")

    dr = cfg.delta_range
    if not isinstance(dr, (tuple, list)) or len(dr) != 2:
        raise ConfigError("delta_range", f"must be a pair (low, high), got {dr!r}")
    d_lo, d_hi = dr
    if not all(_is_real(v) and math.isfinite(v) for v in dr):
        raise ConfigError("delta_range", f"bounds must be finite numbers, got {dr!r}")
    if not 0.0 < d_lo < d_hi:
        raise ConfigError("delta_range", f"must satisfy 0 < low < high, got {dr!r}")

    _check_exclusion(cfg)

    if not isinstance(cfg.allow_full_scale, (bool, np.bool_)):
        raise ConfigError("allow_full_scale", f"must be a bool, got {cfg.allow_full_scale!r}")
    if cfg.n_rows > SMOKE_MAX_ROWS and not cfg.allow_full_scale:
        raise ConfigError("allow_full_scale",
                          f"{cfg.n_rows} rows exceed the smoke limit of {SMOKE_MAX_ROWS}; "
                          "full-scale generation needs user approval and allow_full_scale=True (R02.13)")
