"""Reconstruct a CSS code's Hx/Hz parity-check matrices directly from a
stim memory-experiment circuit.

Convention used by every circuit this package builds (both the stim
built-in surface-code generator and our own generic CSS syndrome-extraction
builder in circuits.py):
  - "data" qubits are reset once and measured once with a plain M/MX/MY/MZ
    near the end of the circuit.
  - "check/ancilla" qubits are measured+reset repeatedly with MR, once per
    round.
  - an ancilla is entangled with data qubits only via CX gates; if the
    ancilla is the CX *control* it is an X-type check, if it is the CX
    *target* it is a Z-type check.

Deriving Hx/Hz from the actual circuit (instead of re-deriving them by hand
from lattice geometry) guarantees the parity-check matrices we report are
exactly the ones the emitted circuit implements.
"""
from __future__ import annotations

import numpy as np
import stim

from .css_code import CSSCode

_TWO_QUBIT_GATES = {"CX", "CNOT", "ZCX"}
_PLAIN_MEASURE_GATES = {"M", "MX", "MY", "MZ"}


def parity_checks_from_memory_circuit(circuit: stim.Circuit, name: str = "extracted") -> tuple[CSSCode, list[int]]:
    """Returns (CSSCode, data_qubit_order) where data_qubit_order[i] is the
    physical qubit index (as used in the stim circuit) of Hx/Hz column i.
    """
    flat = circuit.flattened()

    mr_qubits: set[int] = set()
    m_qubits: set[int] = set()
    for instr in flat:
        if instr.name == "MR":
            mr_qubits.update(t.value for t in instr.targets_copy())
        elif instr.name in _PLAIN_MEASURE_GATES:
            m_qubits.update(t.value for t in instr.targets_copy())

    ancilla_qubits = mr_qubits
    data_qubits = m_qubits - mr_qubits
    if not data_qubits:
        raise ValueError("could not identify data qubits: no plain M/MX/MY/MZ found outside of MR")
    if not ancilla_qubits:
        raise ValueError("could not identify ancilla qubits: no MR instructions found")

    data_order = sorted(data_qubits)
    data_index = {q: i for i, q in enumerate(data_order)}

    support: dict[int, set[int]] = {q: set() for q in ancilla_qubits}
    check_type: dict[int, str] = {}
    frozen: set[int] = set()

    for instr in flat:
        if len(frozen) == len(ancilla_qubits):
            break
        if instr.name in _TWO_QUBIT_GATES:
            targets = [t.value for t in instr.targets_copy()]
            for i in range(0, len(targets), 2):
                a, b = targets[i], targets[i + 1]
                if a in ancilla_qubits and b in data_qubits:
                    anc, data_q, kind = a, b, "X"
                elif b in ancilla_qubits and a in data_qubits:
                    anc, data_q, kind = b, a, "Z"
                else:
                    continue
                if anc in frozen:
                    continue
                support[anc].add(data_q)
                prev = check_type.get(anc)
                if prev is not None and prev != kind:
                    raise ValueError(
                        f"ancilla qubit {anc} appears as both control and target "
                        "of data-qubit CX gates; cannot infer a single check type"
                    )
                check_type[anc] = kind
        elif instr.name == "MR":
            for t in instr.targets_copy():
                if t.value in ancilla_qubits:
                    frozen.add(t.value)

    x_rows, z_rows = [], []
    for anc in sorted(ancilla_qubits):
        kind = check_type.get(anc)
        if kind is None:
            continue  # ancilla never coupled to a data qubit before its first MR
        row = np.zeros(len(data_order), dtype=np.uint8)
        for data_q in support[anc]:
            row[data_index[data_q]] = 1
        (x_rows if kind == "X" else z_rows).append(row)

    n = len(data_order)
    Hx = np.array(x_rows, dtype=np.uint8) if x_rows else np.zeros((0, n), dtype=np.uint8)
    Hz = np.array(z_rows, dtype=np.uint8) if z_rows else np.zeros((0, n), dtype=np.uint8)
    return CSSCode(name=name, Hx=Hx, Hz=Hz), data_order
