"""Bivariate Bicycle (BB) code support — generic constructor only.

Given cyclic-group sizes l, m and two GF(2)[x,y]/(x^l-1, y^m-1) polynomials
A, B (each a small sum of x^a / y^b monomials), builds the CSS code with

    Hx = [A | B]        Hz = [B^T | A^T]

acting on n = 2*l*m qubits split into an "L" block and an "R" block of
l*m qubits each (the standard bivariate-bicycle construction, e.g. Bravyi
et al. 2024). x and y are represented as commuting cyclic-shift
permutation matrices, so Hx @ Hz.T = A B + B A = 0 (mod 2) automatically.

This module intentionally does NOT ship literature "preset" parameter
sets (e.g. the paper's [[72,12,6]] / [[144,12,12]] codes) — getting the
exact monomial exponents right requires checking them against the source
paper. Callers supply (l, m, A-terms, B-terms) directly; presets can be
added later once cross-checked against the paper.
"""
from __future__ import annotations

import re

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
