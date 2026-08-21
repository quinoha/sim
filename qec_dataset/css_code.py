"""CSS code representation: Hx/Hz parity-check matrices plus derived
[[n, k]] and logical operators. Code distance is *not* computed here —
see distance.py, which derives it from an actual stim circuit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from . import gf2


@dataclass
class CSSCode:
    name: str
    Hx: np.ndarray
    Hz: np.ndarray

    def __post_init__(self):
        self.Hx = np.array(self.Hx, dtype=np.uint8) % 2
        self.Hz = np.array(self.Hz, dtype=np.uint8) % 2
        if self.Hx.shape[1] != self.Hz.shape[1]:
            raise ValueError(
                f"Hx and Hz must have the same number of columns (n): "
                f"{self.Hx.shape[1]} vs {self.Hz.shape[1]}"
            )
        commutator = (self.Hx.astype(np.uint32) @ self.Hz.T.astype(np.uint32)) % 2
        if commutator.any():
            raise ValueError(
                f"CSS commutation condition failed for code '{self.name}': "
                "Hx @ Hz.T != 0 (mod 2)"
            )

    @property
    def n(self) -> int:
        return self.Hx.shape[1]

    @property
    def rank_x(self) -> int:
        return gf2.rank_mod2(self.Hx)

    @property
    def rank_z(self) -> int:
        return gf2.rank_mod2(self.Hz)

    @property
    def k(self) -> int:
        return self.n - self.rank_x - self.rank_z

    @cached_property
    def _logical_ops(self) -> tuple[np.ndarray, np.ndarray]:
        n, k = self.n, self.k
        if k == 0:
            empty = np.zeros((0, n), dtype=np.uint8)
            return empty, empty

        ker_hz = gf2.nullspace_mod2(self.Hz)
        ker_hx = gf2.nullspace_mod2(self.Hx)
        lx_candidates = gf2.quotient_basis(ker_hz, self.Hx, n)
        lz_candidates = gf2.quotient_basis(ker_hx, self.Hz, n)
        if len(lx_candidates) != k or len(lz_candidates) != k:
            raise ValueError(
                f"expected {k} logical operator candidates for '{self.name}', "
                f"got Lx={len(lx_candidates)} Lz={len(lz_candidates)}"
            )
        return gf2.pair_logical_operators(lx_candidates, lz_candidates)

    @property
    def logical_x(self) -> np.ndarray:
        """k x n binary matrix; row i is the qubit-support of logical X_i."""
        return self._logical_ops[0]

    @property
    def logical_z(self) -> np.ndarray:
        """k x n binary matrix; row i is the qubit-support of logical Z_i."""
        return self._logical_ops[1]

    def summary(self) -> str:
        return f"[[{self.n}, {self.k}]] CSS code '{self.name}' (rank Hx={self.rank_x}, rank Hz={self.rank_z})"
