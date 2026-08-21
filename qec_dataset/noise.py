"""Shared circuit-level noise model.

Mirrors the four noise knobs stim's own `stim.Circuit.generated(...)`
exposes for surface codes, so the same `NoiseModel` object can drive both
the built-in surface-code generator and our generic CSS syndrome-extraction
circuit builder (circuits.py) used for BB codes.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseModel:
    after_clifford_depolarization: float = 0.0
    before_round_data_depolarization: float = 0.0
    after_reset_flip_probability: float = 0.0
    before_measure_flip_probability: float = 0.0

    @staticmethod
    def uniform(p: float) -> "NoiseModel":
        return NoiseModel(
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            after_reset_flip_probability=p,
            before_measure_flip_probability=p,
        )

    @property
    def is_noiseless(self) -> bool:
        return (
            self.after_clifford_depolarization == 0.0
            and self.before_round_data_depolarization == 0.0
            and self.after_reset_flip_probability == 0.0
            and self.before_measure_flip_probability == 0.0
        )

    def as_stim_generated_kwargs(self) -> dict:
        return {
            "after_clifford_depolarization": self.after_clifford_depolarization,
            "before_round_data_depolarization": self.before_round_data_depolarization,
            "after_reset_flip_probability": self.after_reset_flip_probability,
            "before_measure_flip_probability": self.before_measure_flip_probability,
        }
