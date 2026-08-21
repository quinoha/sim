from __future__ import annotations

from dataclasses import dataclass

import stim

from .css_code import CSSCode
from .distance import DistanceResult


@dataclass
class CodeReport:
    name: str
    n: int
    k: int
    distance: DistanceResult
    num_qubits: int
    num_detectors: int
    num_observables: int

    def __str__(self) -> str:
        return (
            f"{self.name}: [[{self.n}, {self.k}, {self.distance}]]\n"
            f"  circuit: {self.num_qubits} qubits, "
            f"{self.num_detectors} detectors, {self.num_observables} observables"
        )


def build_report(code: CSSCode, circuit: stim.Circuit, distance: DistanceResult) -> CodeReport:
    return CodeReport(
        name=code.name,
        n=code.n,
        k=code.k,
        distance=distance,
        num_qubits=circuit.num_qubits,
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
    )


def print_code_summary(code: CSSCode, circuit: stim.Circuit, distance: DistanceResult) -> CodeReport:
    report = build_report(code, circuit, distance)
    print(report)
    return report
