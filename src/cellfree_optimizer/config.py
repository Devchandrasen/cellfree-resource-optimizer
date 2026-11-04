"""Validated configuration for independent network generation."""

from dataclasses import asdict, dataclass
import math
from numbers import Integral


@dataclass(frozen=True, slots=True)
class NetworkConfig:
    aps: int = 14
    antennas: int = 8
    users: int = 10
    max_aps: int = 8
    fronthaul: float = 4.0
    fronthaul_ratio: float = 7.0
    ap_power: float = 0.2
    pilot_power: float = 0.1
    pilot_length: int = 14
    coherence_symbols: int = 200
    side_length: float = 500.0
    height: float = 10.0
    angular_spread_degrees: float = 15.0
    quadrature_nodes: int = 20
    shadow_std_db: float = 4.0
    shadow_distance: float = 50.0
    pathloss_intercept_db: float = -30.5
    pathloss_slope: float = 36.7
    bandwidth_hz: float = 20e6
    temperature_kelvin: float = 290.0
    noise_figure_db: float = 9.0
    seed: int = 7

    def __post_init__(self):
        for name in [
            "aps",
            "antennas",
            "users",
            "max_aps",
            "pilot_length",
            "coherence_symbols",
            "quadrature_nodes",
            "seed",
        ]:
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < (0 if name == "seed" else 1)
            ):
                raise ValueError(f"{name} must be an integer {'>= 0' if name == 'seed' else '> 0'}")
        if self.max_aps > self.aps:
            raise ValueError("max_aps cannot exceed aps")
        if not self.users <= self.pilot_length < self.coherence_symbols:
            raise ValueError(
                "Require users <= pilot_length < coherence_symbols for orthogonal pilots"
            )
        for name in [
            "fronthaul",
            "ap_power",
            "pilot_power",
            "side_length",
            "height",
            "shadow_distance",
            "pathloss_slope",
            "bandwidth_hz",
            "temperature_kelvin",
        ]:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ["angular_spread_degrees", "shadow_std_db"]:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.fronthaul_ratio) or self.fronthaul_ratio < 1:
            raise ValueError("fronthaul_ratio must be finite and >= 1")
        if not all(math.isfinite(v) for v in [self.pathloss_intercept_db, self.noise_figure_db]):
            raise ValueError("Pathloss intercept and noise figure must be finite")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        return cls(**values)
