"""Rotated surface code support, built on stim's built-in generator."""
from __future__ import annotations

import stim

from ..css_code import CSSCode
from ..extract import parity_checks_from_memory_circuit
from ..noise import NoiseModel

VALID_BASES = ("X", "Z")


def _task_name(basis: str) -> str:
    basis = basis.upper()
    if basis not in VALID_BASES:
        raise ValueError(f"basis must be one of {VALID_BASES}, got {basis!r}")
    return f"surface_code:rotated_memory_{basis.lower()}"


def rotated_surface_code(distance: int, basis: str = "Z") -> tuple[CSSCode, list[int]]:
    """Builds the Hx/Hz parity-check matrices of the distance-`d` rotated
    surface code by extracting them from stim's own generated circuit
    (one round is enough), so they are guaranteed consistent with the
    circuit `build_surface_code_circuit` below produces.
    """
    probe_circuit = stim.Circuit.generated(_task_name(basis), distance=distance, rounds=1)
    code, data_order = parity_checks_from_memory_circuit(
        probe_circuit, name=f"rotated_surface_d{distance}"
    )
    return code, data_order


def build_surface_code_circuit(
    distance: int,
    rounds: int,
    basis: str = "Z",
    noise: NoiseModel | None = None,
) -> stim.Circuit:
    noise = noise or NoiseModel.uniform(0.0)
    return stim.Circuit.generated(
        _task_name(basis),
        distance=distance,
        rounds=rounds,
        **noise.as_stim_generated_kwargs(),
    )
