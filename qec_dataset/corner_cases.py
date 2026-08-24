"""Directed ("corner case") test vectors.

Unlike sampling.py (bulk random sampling using stim's probabilistic noise
channels), each case here injects a *specific, deterministic* Pauli error
into an otherwise noiseless circuit and samples it once — there's no
randomness left to average over, so the (detection events, observable
flip) pair is exactly reproducible. This is useful for exercising a
decoder on inputs that are easy to get wrong (a full logical error that
produces an all-zero syndrome) or that it must get right (a sub-threshold
error it must NOT "correct" into a logical flip).

The circuit passed in should be noiseless (e.g. built with
`NoiseModel.uniform(0.0)`) — the convenience `generate_*_corner_cases`
functions below build one for you.

Qubit indexing caveat: `CSSCode.logical_x`/`logical_z` columns are indices
into Hx/Hz, NOT necessarily the physical qubit ids used in a given
circuit. Our own `circuits.generic_css_memory_circuit` (used for BB codes)
happens to number data qubits 0..n-1, so columns equal physical ids there
— but `codes.surface.build_surface_code_circuit` uses stim's own numbering
(via `stim.Circuit.generated`), which is unrelated. Every function here
therefore takes an explicit `qubit_order` mapping (column index -> physical
qubit id, as returned by `codes.surface.rotated_surface_code` /
`codes.bb.build_bb_code`'s implicit identity order) and translates before
injecting anything into the circuit.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import stim

from .css_code import CSSCode

_VALID_PAULIS = ("X", "Y", "Z")


@dataclass
class CornerCase:
    name: str
    description: str
    injected_qubits: list[int]
    pauli: str
    detection_events: list[bool]
    observable_flips: list[bool]

    def to_dict(self) -> dict:
        return asdict(self)


def _insert_after_initial_reset(circuit: stim.Circuit, error_circuit: stim.Circuit) -> stim.Circuit:
    """Splices `error_circuit` in right after the circuit's initial reset
    block (i.e. right before the first TICK) — every circuit this package
    builds resets all qubits there, before round 1 begins.
    """
    flat = circuit.flattened()
    for i, instr in enumerate(flat):
        if isinstance(instr, stim.CircuitInstruction) and instr.name == "TICK":
            return flat[:i] + error_circuit + flat[i:]
    raise ValueError("circuit has no TICK instruction; cannot locate the initial reset boundary")


def _pauli_error_circuit(qubits: list[int], pauli: str) -> stim.Circuit:
    if pauli not in _VALID_PAULIS:
        raise ValueError(f"pauli must be one of {_VALID_PAULIS}, got {pauli!r}")
    c = stim.Circuit()
    if qubits:
        # probability 1.0 -> deterministic, not a random channel
        c.append(f"{pauli}_ERROR", qubits, 1.0)
    return c


def _to_physical_qubits(columns: list[int], qubit_order: list[int] | None) -> list[int]:
    if qubit_order is None:
        return list(columns)
    return [qubit_order[c] for c in columns]


def _sample_case(
    circuit: stim.Circuit,
    columns: list[int],
    pauli: str,
    name: str,
    description: str,
    qubit_order: list[int] | None = None,
) -> CornerCase:
    physical_qubits = _to_physical_qubits(columns, qubit_order)
    full_circuit = (
        _insert_after_initial_reset(circuit, _pauli_error_circuit(physical_qubits, pauli))
        if physical_qubits
        else circuit
    )
    dets, obs = full_circuit.compile_detector_sampler().sample(1, separate_observables=True)
    return CornerCase(
        name=name,
        description=description,
        injected_qubits=physical_qubits,
        pauli=pauli,
        detection_events=dets[0].tolist(),
        observable_flips=obs[0].tolist(),
    )


def all_zero_case(circuit: stim.Circuit) -> CornerCase:
    """No injected error at all. Every detector and observable should read 0."""
    return _sample_case(circuit, [], "X", name="all_zero", description="no injected error")


def _anticommuting_ops_for_basis(code: CSSCode, basis: str) -> tuple[np.ndarray, str]:
    basis = basis.upper()
    if basis == "Z":
        return code.logical_x, "X"
    if basis == "X":
        return code.logical_z, "Z"
    raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")


def full_logical_error_cases(
    code: CSSCode, circuit: stim.Circuit, basis: str, qubit_order: list[int] | None = None
) -> list[CornerCase]:
    """One case per logical qubit: inject the FULL support of the operator
    that anticommutes with that logical qubit's memory-basis observable.
    This is the worst case for a decoder — a genuine logical error with an
    all-zero syndrome (no detector fires), so nothing in the syndrome
    itself distinguishes it from "no error at all".
    """
    ops, pauli = _anticommuting_ops_for_basis(code, basis)
    cases = []
    for j, row in enumerate(ops):
        columns = np.nonzero(row)[0].tolist()
        cases.append(
            _sample_case(
                circuit,
                columns,
                pauli,
                name=f"full_logical_error_{j}",
                description=(
                    f"weight-{len(columns)} {pauli} error on logical qubit {j}'s "
                    "anticommuting operator support"
                ),
                qubit_order=qubit_order,
            )
        )
    return cases


def near_miss_logical_error_cases(
    code: CSSCode, circuit: stim.Circuit, basis: str, qubit_order: list[int] | None = None
) -> list[CornerCase]:
    """One case per logical qubit: take the FULL logical-error support
    (as in `full_logical_error_cases`) and remove exactly one qubit that
    also lies in the *paired* logical operator's support.

    `Lx_j . Lz_j = 1` by construction (see gf2.pair_logical_operators),
    i.e. the two supports overlap in an ODD number of qubits. Dropping any
    one qubit from that overlap flips the overlap parity to even, which
    provably un-flips observable j specifically — unlike naively dropping
    an arbitrary fraction of the support (which has no such guarantee: a
    qubit outside the overlap can be dropped without changing the parity
    at all).

    Caveat for codes with k > 1 (e.g. BB codes): only `observable_flips[j]`
    is provably False. `full_logical_error_cases`/`near_miss_*` use
    whatever representative `CSSCode.logical_x`/`logical_z` happened to
    compute (via gf2.py's quotient-basis + symplectic-pairing procedure),
    which is *not* guaranteed to be anywhere near minimum weight — BB code
    representatives observed in practice range up to ~26 qubits for a
    [[72,12,6]] code whose actual distance is 6. A weight-25 residual
    error crosses plenty of *other* logical qubits' supports, so
    `observable_flips[j' != j]` routinely also flip; this is a real
    property of the representative operator chosen, not a bug, but it
    means this case is a genuine "near miss for logical qubit j",
    not "no logical error at all". For a tight near-threshold case tied
    to the code's actual distance, you'd need a minimum-weight
    representative (e.g. from `distance.search_distance`'s error list),
    which this module does not compute.
    """
    ops, pauli = _anticommuting_ops_for_basis(code, basis)
    paired_ops = code.logical_z if basis.upper() == "Z" else code.logical_x
    cases = []
    for j, row in enumerate(ops):
        full_columns = np.nonzero(row)[0].tolist()
        paired_support = set(np.nonzero(paired_ops[j])[0].tolist())
        overlap = [c for c in full_columns if c in paired_support]
        if not overlap:
            raise ValueError(
                f"logical operator pairing invariant violated for qubit {j}: "
                "empty overlap with its paired operator"
            )
        drop = overlap[0]
        columns = [c for c in full_columns if c != drop]
        cases.append(
            _sample_case(
                circuit,
                columns,
                pauli,
                name=f"near_miss_logical_error_{j}",
                description=(
                    f"weight-{len(columns)} {pauli} error: logical qubit {j}'s full "
                    f"anticommuting support with column {drop} removed (breaks the "
                    "logical flip by construction — should not trip the observable)"
                ),
                qubit_order=qubit_order,
            )
        )
    return cases


def generate_corner_cases(
    code: CSSCode, circuit: stim.Circuit, basis: str, qubit_order: list[int] | None = None
) -> list[CornerCase]:
    return (
        [all_zero_case(circuit)]
        + full_logical_error_cases(code, circuit, basis, qubit_order=qubit_order)
        + near_miss_logical_error_cases(code, circuit, basis, qubit_order=qubit_order)
    )


def generate_surface_code_corner_cases(distance: int, rounds: int, basis: str = "Z") -> list[CornerCase]:
    from .codes.surface import build_surface_code_circuit, rotated_surface_code
    from .noise import NoiseModel

    code, order = rotated_surface_code(distance, basis=basis)
    circuit = build_surface_code_circuit(distance, rounds, basis=basis, noise=NoiseModel.uniform(0.0))
    return generate_corner_cases(code, circuit, basis, qubit_order=order)


def generate_bb_code_corner_cases(
    l: int, m: int, a_terms, b_terms, rounds: int, basis: str = "Z"
) -> list[CornerCase]:
    from .circuits import generic_css_memory_circuit
    from .codes.bb import build_bb_code
    from .noise import NoiseModel

    code = build_bb_code(l, m, a_terms, b_terms)
    circuit = generic_css_memory_circuit(code, rounds, basis=basis, noise=NoiseModel.uniform(0.0))
    # circuits.generic_css_memory_circuit numbers data qubits 0..n-1, i.e.
    # identity order -> qubit_order=None (the _to_physical_qubits default).
    return generate_corner_cases(code, circuit, basis, qubit_order=None)


def save_corner_cases(cases: list[CornerCase], path: str | Path) -> None:
    Path(path).write_text(json.dumps([c.to_dict() for c in cases], indent=2))


def load_corner_cases(path: str | Path) -> list[CornerCase]:
    data = json.loads(Path(path).read_text())
    return [CornerCase(**d) for d in data]
