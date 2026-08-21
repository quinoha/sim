"""Generic CSS memory-experiment circuit builder.

stim ships a built-in generator for surface codes
(`stim.Circuit.generated("surface_code:...")`) but has none for BB codes,
so this module builds a syndrome-extraction circuit directly from a
CSSCode's Hx/Hz matrices. It works for *any* CSS code, so it is also used
to cross-check the surface-code path (codes/surface.py builds surface
circuits via stim's generator; this module can build an equivalent one
from the same Hx/Hz and the two should agree on n/k/detector-determinism).

Conventions (must match extract.py):
  - data qubits are physical indices 0..n-1, reset once, measured once
    near the end.
  - X-check ancillas come right after (indices n..n+rank(Hx)-1); a check
    ancilla is the CX *control* into its data qubits (with an H sandwich,
    exactly like stim's own surface-code circuits).
  - Z-check ancillas follow (indices n+rank(Hx)..); a check ancilla is the
    CX *target* from its data qubits, no H.
  - all ancillas are measured+reset (MR) every round; data qubits are
    measured once at the very end (M for Z memory, MX for X memory).

CNOT scheduling: a simple greedy layering (assign each two-qubit gate to
the earliest round-local layer where neither qubit is already busy).
This is correct but not depth-optimal — no attempt is made to reproduce
the hardware-tuned schedule from the BB code literature.
"""
from __future__ import annotations

import numpy as np
import stim

from .css_code import CSSCode
from .noise import NoiseModel


def _schedule_layers(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    layers: list[list[tuple[int, int]]] = []
    busy: list[set[int]] = []
    for ctrl, tgt in edges:
        for layer, used in zip(layers, busy):
            if ctrl not in used and tgt not in used:
                layer.append((ctrl, tgt))
                used.add(ctrl)
                used.add(tgt)
                break
        else:
            layers.append([(ctrl, tgt)])
            busy.append({ctrl, tgt})
    return layers


def _row_supports(mat: np.ndarray) -> list[list[int]]:
    return [sorted(np.nonzero(row)[0].tolist()) for row in mat]


def generic_css_memory_circuit(
    code: CSSCode,
    rounds: int,
    basis: str = "Z",
    noise: NoiseModel | None = None,
) -> stim.Circuit:
    basis = basis.upper()
    if basis not in ("X", "Z"):
        raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    noise = noise or NoiseModel.uniform(0.0)

    n = code.n
    rx = code.Hx.shape[0]
    rz = code.Hz.shape[0]

    data_qubits = list(range(n))
    x_anc = list(range(n, n + rx))
    z_anc = list(range(n + rx, n + rx + rz))
    all_anc = x_anc + z_anc
    num_anc = len(all_anc)
    anc_position = {q: i for i, q in enumerate(all_anc)}

    x_supports = _row_supports(code.Hx)
    z_supports = _row_supports(code.Hz)

    def _term_ordered_edges(supports, make_edge):
        max_weight = max([len(s) for s in supports] + [0])
        edges = []
        for t in range(max_weight):
            for i, support in enumerate(supports):
                if t < len(support):
                    edges.append(make_edge(i, support[t]))
        return edges

    x_edges = _term_ordered_edges(x_supports, lambda i, col: (x_anc[i], data_qubits[col]))
    z_edges = _term_ordered_edges(z_supports, lambda i, col: (data_qubits[col], z_anc[i]))

    # X-check and Z-check gates must NOT be interleaved qubit-by-qubit: a data
    # qubit shared between an X-check and a Z-check that are scheduled into
    # the same batch of layers can end up entangled with a not-yet-measured
    # ancilla of the *other* type, which makes the check outcome genuinely
    # random even on a noiseless circuit (verified empirically). Scheduling
    # each type's edges into its own complete, separate block of layers
    # avoids this; the two blocks can go in either order.
    layers = _schedule_layers(x_edges) + _schedule_layers(z_edges)

    circuit = stim.Circuit()
    for q in data_qubits:
        circuit.append("QUBIT_COORDS", [q], [q, 0])
    for q in x_anc:
        circuit.append("QUBIT_COORDS", [q], [q, 1])
    for q in z_anc:
        circuit.append("QUBIT_COORDS", [q], [q, 2])

    data_flip_gate = "X_ERROR" if basis == "Z" else "Z_ERROR"

    circuit.append("R" if basis == "Z" else "RX", data_qubits)
    if noise.after_reset_flip_probability:
        circuit.append(data_flip_gate, data_qubits, noise.after_reset_flip_probability)
    circuit.append("R", all_anc)
    if noise.after_reset_flip_probability:
        circuit.append("X_ERROR", all_anc, noise.after_reset_flip_probability)
    circuit.append("TICK")

    for r in range(rounds):
        if noise.before_round_data_depolarization:
            circuit.append("DEPOLARIZE1", data_qubits, noise.before_round_data_depolarization)
        if x_anc:
            circuit.append("H", x_anc)
            if noise.after_clifford_depolarization:
                circuit.append("DEPOLARIZE1", x_anc, noise.after_clifford_depolarization)
        circuit.append("TICK")

        for layer in layers:
            flat_targets = [q for pair in layer for q in pair]
            circuit.append("CX", flat_targets)
            if noise.after_clifford_depolarization:
                circuit.append("DEPOLARIZE2", flat_targets, noise.after_clifford_depolarization)
            circuit.append("TICK")

        if x_anc:
            circuit.append("H", x_anc)
            if noise.after_clifford_depolarization:
                circuit.append("DEPOLARIZE1", x_anc, noise.after_clifford_depolarization)
            circuit.append("TICK")

        if noise.before_measure_flip_probability:
            circuit.append("X_ERROR", all_anc, noise.before_measure_flip_probability)
        circuit.append("MR", all_anc)
        if noise.after_reset_flip_probability:
            circuit.append("X_ERROR", all_anc, noise.after_reset_flip_probability)

        if r == 0:
            matching_anc = z_anc if basis == "Z" else x_anc
            for anc in matching_anc:
                p = anc_position[anc]
                circuit.append("DETECTOR", [stim.target_rec(-(num_anc - p))], [anc, 0])
        else:
            for anc in all_anc:
                p = anc_position[anc]
                this_rec = -(num_anc - p)
                prev_rec = this_rec - num_anc
                circuit.append(
                    "DETECTOR",
                    [stim.target_rec(this_rec), stim.target_rec(prev_rec)],
                    [anc, r],
                )

    final_measure_gate = "M" if basis == "Z" else "MX"
    if noise.before_measure_flip_probability:
        circuit.append(data_flip_gate, data_qubits, noise.before_measure_flip_probability)
    circuit.append(final_measure_gate, data_qubits)

    matching_anc = z_anc if basis == "Z" else x_anc
    matching_H = code.Hz if basis == "Z" else code.Hx
    for i, anc in enumerate(matching_anc):
        p = anc_position[anc]
        anc_rec = -(num_anc - p) - n
        data_cols = np.nonzero(matching_H[i])[0].tolist()
        targets = [stim.target_rec(anc_rec)] + [stim.target_rec(-(n - c)) for c in data_cols]
        circuit.append("DETECTOR", targets, [anc, rounds])

    logical_ops = code.logical_z if basis == "Z" else code.logical_x
    for j, row in enumerate(logical_ops):
        cols = np.nonzero(row)[0].tolist()
        targets = [stim.target_rec(-(n - c)) for c in cols]
        circuit.append("OBSERVABLE_INCLUDE", targets, j)

    return circuit
