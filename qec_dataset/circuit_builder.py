"""
QEDA Circuit Builder Bridge.
Maps declarative RunSpec instances from an EvaluationPlan to executable stim.Circuit objects.
Supports both Surface codes (via stim generator) and Bivariate Bicycle codes (via generic CSS builder).
"""
from __future__ import annotations

import re
from typing import Dict, Tuple, Optional
import stim

from m0_config.plan_schema import RunSpec, EvaluationPlan
from .codes.surface import build_surface_code_circuit
from .codes.bb import build_bb_preset
from .circuits import generic_css_memory_circuit
from .noise import NoiseModel

# In-memory cache for generated circuits to prevent redundant builds across decoders
# Key: (code_id, noise_p, rounds, basis)
_CIRCUIT_CACHE: Dict[Tuple[str, float, int, str], stim.Circuit] = {}


def clear_circuit_cache() -> None:
    """Clears the circuit build cache."""
    _CIRCUIT_CACHE.clear()


def build_surface_circuit_from_run_spec(run_spec: RunSpec, basis: str = "Z") -> stim.Circuit:
    """
    Builds a stim.Circuit for a Surface Code from a RunSpec.

    Args:
        run_spec: The immutable RunSpec containing distance, rounds, and noise_p.
        basis: Target memory basis ('Z' or 'X'). Defaults to 'Z'.

    Returns:
        stim.Circuit configured with rotated surface code memory experiment.
    """
    distance = run_spec.code_params.get("distance")
    if distance is None:
        raise ValueError(f"RunSpec '{run_spec.run_spec_id}' missing 'distance' in code_params: {run_spec.code_params}")

    distance = int(distance)
    rounds = int(run_spec.rounds)
    noise = NoiseModel.uniform(float(run_spec.noise_p))

    return build_surface_code_circuit(
        distance=distance,
        rounds=rounds,
        basis=basis,
        noise=noise,
    )


def build_bb_circuit_from_run_spec(run_spec: RunSpec, basis: str = "Z") -> stim.Circuit:
    """
    Builds a stim.Circuit for a Bivariate Bicycle (BB) Code from a RunSpec.

    Args:
        run_spec: The immutable RunSpec containing preset, rounds, and noise_p.
        basis: Target memory basis ('Z' or 'X'). Defaults to 'Z'.

    Returns:
        stim.Circuit configured with generic CSS memory experiment.
    """
    preset_val = run_spec.code_params.get("preset")
    if not preset_val:
        raise ValueError(f"RunSpec '{run_spec.run_spec_id}' missing 'preset' in code_params: {run_spec.code_params}")

    # Extract integer distance from preset string (e.g. 'd6' -> 6)
    if isinstance(preset_val, str):
        match = re.search(r"\d+", preset_val)
        if not match:
            raise ValueError(f"Invalid BB preset '{preset_val}' in RunSpec '{run_spec.run_spec_id}'. Expected 'd6', 'd10', etc.")
        distance = int(match.group())
    else:
        distance = int(preset_val)

    code = build_bb_preset(distance=distance)
    rounds = int(run_spec.rounds)
    noise = NoiseModel.uniform(float(run_spec.noise_p))

    return generic_css_memory_circuit(
        code=code,
        rounds=rounds,
        basis=basis,
        noise=noise,
    )


def build_circuit_from_run_spec(
    run_spec: RunSpec,
    basis: str = "Z",
    use_cache: bool = True
) -> stim.Circuit:
    """
    Universal factory routing RunSpec to appropriate circuit generator with caching.

    Args:
        run_spec: The immutable RunSpec specification.
        basis: Target memory experiment basis ('Z' or 'X'). Defaults to 'Z'.
        use_cache: Whether to reuse previously built circuits with identical parameters.

    Returns:
        stim.Circuit ready for sinter sampling or error analysis.
    """
    # code_params belongs in the key: `code_id` is a label a config author picks, and
    # editing `distance` while leaving the id alone (surface_d5 -> distance 7, say) is
    # an easy edit to make. Without the params here the cache would hand back the
    # previous distance's circuit under the unchanged id, silently.
    cache_key = (
        run_spec.code_id,
        tuple(sorted((k, str(v)) for k, v in run_spec.code_params.items())),
        float(run_spec.noise_p),
        int(run_spec.rounds),
        basis.upper(),
    )

    if use_cache and cache_key in _CIRCUIT_CACHE:
        return _CIRCUIT_CACHE[cache_key].copy()

    family = run_spec.code_family.lower()

    if family == "surface":
        circuit = build_surface_circuit_from_run_spec(run_spec, basis=basis)
    elif family == "bb":
        circuit = build_bb_circuit_from_run_spec(run_spec, basis=basis)
    else:
        raise ValueError(
            f"Unsupported code_family '{run_spec.code_family}' in RunSpec '{run_spec.run_spec_id}'."
        )

    if use_cache:
        _CIRCUIT_CACHE[cache_key] = circuit.copy()

    return circuit


def build_all_circuits_from_plan(
    plan: EvaluationPlan,
    basis: str = "Z"
) -> Dict[str, stim.Circuit]:
    """
    Builds stim.Circuit objects for all active RunSpecs in an EvaluationPlan.

    Args:
        plan: The compiled EvaluationPlan containing active run_specs.
        basis: Target memory experiment basis. Defaults to 'Z'.

    Returns:
        Mapping from run_spec_id to its stim.Circuit instance.
    """
    circuits: Dict[str, stim.Circuit] = {}
    for spec in plan.run_specs:
        circuits[spec.run_spec_id] = build_circuit_from_run_spec(spec, basis=basis)
    return circuits
