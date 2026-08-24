"""Bivariate Bicycle (BB) code support — generic constructor only.

Given cyclic-group sizes l, m and two GF(2)[x,y]/(x^l-1, y^m-1) polynomials
A, B (each a small sum of x^a / y^b monomials), builds the CSS code with

    Hx = [A | B]        Hz = [B^T | A^T]

acting on n = 2*l*m qubits split into an "L" block and an "R" block of
l*m qubits each (the standard bivariate-bicycle construction, e.g. Bravyi
et al. 2024). x and y are represented as commuting cyclic-shift
permutation matrices, so Hx @ Hz.T = A B + B A = 0 (mod 2) automatically.

Literature presets are provided in `BB_PRESETS` / `build_bb_preset()`.
Their (l, m, A, B) parameters were cross-checked against the independent
implementation in `astra/bb_panq_functions.py:bb_code()` (which follows
`codes_q.create_bivariate_bicycle_codes`): for all six presets this
module's Hx/Hz come out **bit-identical** to that implementation's, so the
two can share trained models and parity-check matrices directly.
Callers can also supply (l, m, A-terms, B-terms) directly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from ..css_code import CSSCode

Term = tuple[str, int]

_TERM_RE = re.compile(r"^([xy])(\d+)$")


def shift_matrix(size: int) -> np.ndarray:
    """Cyclic shift permutation matrix representing multiplication by the
    generator of Z/size (i.e. "x" or "y") in the group ring."""
    mat = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        mat[i, (i + 1) % size] = 1
    return mat


def _matrix_power_mod2(mat: np.ndarray, power: int) -> np.ndarray:
    size = mat.shape[0]
    power = power % size if size else 0
    result = np.eye(size, dtype=np.uint8)
    if power == 0:
        return result
    base = mat.astype(np.uint32)
    acc = np.eye(size, dtype=np.uint32)
    for _ in range(power):
        acc = (acc @ base) % 2
    return acc.astype(np.uint8)


def poly_matrix(terms: list[Term], sx_full: np.ndarray, sy_full: np.ndarray) -> np.ndarray:
    """Sum (mod 2) of the monomials in `terms` (e.g. [("x", 3), ("y", 1)])
    evaluated as matrices, given the full lm x lm shift matrices for x, y."""
    size = sx_full.shape[0]
    result = np.zeros((size, size), dtype=np.uint8)
    for var, power in terms:
        if var == "x":
            term_mat = _matrix_power_mod2(sx_full, power)
        elif var == "y":
            term_mat = _matrix_power_mod2(sy_full, power)
        else:
            raise ValueError(f"unknown variable {var!r} in BB code term, expected 'x' or 'y'")
        result ^= term_mat
    return result


def parse_terms(spec: str) -> list[Term]:
    """Parses a comma-separated monomial spec like "x3,y1,y2" into
    [("x", 3), ("y", 1), ("y", 2)]."""
    terms = []
    for token in (t.strip() for t in spec.split(",") if t.strip()):
        m = _TERM_RE.match(token)
        if not m:
            raise ValueError(f"invalid BB code term {token!r}, expected e.g. 'x3' or 'y1'")
        terms.append((m.group(1), int(m.group(2))))
    if not terms:
        raise ValueError(f"no terms parsed from {spec!r}")
    return terms


def build_bb_code(l: int, m: int, a_terms: list[Term], b_terms: list[Term], name: str | None = None) -> CSSCode:
    if l <= 0 or m <= 0:
        raise ValueError("l and m must be positive")
    sx = shift_matrix(l)
    sy = shift_matrix(m)
    sx_full = np.kron(sx, np.eye(m, dtype=np.uint8)) % 2
    sy_full = np.kron(np.eye(l, dtype=np.uint8), sy) % 2
    sx_full = sx_full.astype(np.uint8)
    sy_full = sy_full.astype(np.uint8)

    a = poly_matrix(a_terms, sx_full, sy_full)
    b = poly_matrix(b_terms, sx_full, sy_full)

    Hx = np.concatenate([a, b], axis=1).astype(np.uint8) % 2
    Hz = np.concatenate([b.T, a.T], axis=1).astype(np.uint8) % 2

    if name is None:
        a_str = "+".join(f"{v}{p}" for v, p in a_terms)
        b_str = "+".join(f"{v}{p}" for v, p in b_terms)
        name = f"bb_l{l}_m{m}_A[{a_str}]_B[{b_str}]"
    return CSSCode(name=name, Hx=Hx, Hz=Hz)


@dataclass(frozen=True)
class BBPreset:
    """A known-good (l, m, A, B) parameter set for a BB code."""

    l: int
    m: int
    a_terms: tuple[Term, ...]
    b_terms: tuple[Term, ...]
    n: int
    k: int
    distance: int
    distance_is_upper_bound: bool = False

    @property
    def label(self) -> str:
        le = "<=" if self.distance_is_upper_bound else ""
        return f"[[{self.n},{self.k},{le}{self.distance}]]"


# Bivariate bicycle codes from the literature (Bravyi et al. 2024,
# "High-threshold and low-overhead fault-tolerant quantum memory").
# Keyed by nominal distance, matching astra/bb_panq_functions.py:bb_code(d).
# Distances marked `distance_is_upper_bound` are recorded in that source as
# upper bounds ("<=") rather than proven distances -- this module does not
# independently verify any of these distances; use distance.search_distance
# if you need to check one.
BB_PRESETS: dict[int, BBPreset] = {
    6: BBPreset(
        l=6, m=6,
        a_terms=(("x", 3), ("y", 1), ("y", 2)),
        b_terms=(("y", 3), ("x", 1), ("x", 2)),
        n=72, k=12, distance=6,
    ),
    10: BBPreset(
        l=15, m=3,
        a_terms=(("x", 9), ("y", 1), ("y", 2)),
        b_terms=(("y", 0), ("x", 2), ("x", 7)),
        n=90, k=8, distance=10,
    ),
    12: BBPreset(
        l=12, m=6,
        a_terms=(("x", 3), ("y", 1), ("y", 2)),
        b_terms=(("y", 3), ("x", 1), ("x", 2)),
        n=144, k=12, distance=12,
    ),
    18: BBPreset(
        l=12, m=12,
        a_terms=(("x", 3), ("y", 2), ("y", 7)),
        b_terms=(("y", 3), ("x", 1), ("x", 2)),
        n=288, k=12, distance=18,
    ),
    24: BBPreset(
        l=30, m=6,
        a_terms=(("x", 9), ("y", 1), ("y", 2)),
        b_terms=(("y", 3), ("x", 25), ("x", 26)),
        n=360, k=12, distance=24, distance_is_upper_bound=True,
    ),
    34: BBPreset(
        l=21, m=18,
        a_terms=(("x", 3), ("y", 10), ("y", 17)),
        b_terms=(("y", 5), ("x", 3), ("x", 19)),
        n=756, k=16, distance=34, distance_is_upper_bound=True,
    ),
}


def build_bb_preset(distance: int) -> CSSCode:
    """Builds a literature BB code preset by its nominal distance.

    Verifies the resulting n and k against the recorded values (the
    distance itself is NOT verified -- see BB_PRESETS).
    """
    if distance not in BB_PRESETS:
        raise ValueError(
            f"no BB preset for distance {distance}; available: {sorted(BB_PRESETS)}"
        )
    p = BB_PRESETS[distance]
    code = build_bb_code(
        p.l, p.m, list(p.a_terms), list(p.b_terms), name=f"bb_{p.n}_{p.k}_{p.distance}"
    )
    if code.n != p.n or code.k != p.k:
        raise ValueError(
            f"BB preset d={distance} mismatch: built [[{code.n},{code.k}]], expected [[{p.n},{p.k}]]"
        )
    return code
