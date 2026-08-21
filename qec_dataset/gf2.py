"""GF(2) linear algebra helpers used to derive [[n, k, d]] and logical
operators for CSS codes.  Implemented with plain numpy (uint8, XOR-based
Gaussian elimination) so the package has no extra dependency beyond numpy.
"""
from __future__ import annotations

import numpy as np


def _to_u8(mat) -> np.ndarray:
    arr = np.array(mat, dtype=np.uint8)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr % 2


def rref_mod2(mat) -> tuple[np.ndarray, list[int]]:
    """Row-reduced echelon form over GF(2). Returns (rref, pivot_cols)."""
    m = _to_u8(mat).copy()
    rows, cols = m.shape
    pivot_cols: list[int] = []
    r = 0
    for c in range(cols):
        pivot_row = None
        for i in range(r, rows):
            if m[i, c]:
                pivot_row = i
                break
        if pivot_row is None:
            continue
        if pivot_row != r:
            m[[r, pivot_row]] = m[[pivot_row, r]]
        for i in range(rows):
            if i != r and m[i, c]:
                m[i, :] ^= m[r, :]
        pivot_cols.append(c)
        r += 1
        if r == rows:
            break
    return m, pivot_cols


def rank_mod2(mat) -> int:
    arr = _to_u8(mat)
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        return 0
    _, pivots = rref_mod2(arr)
    return len(pivots)


def nullspace_mod2(mat) -> np.ndarray:
    """Basis (as rows) of the right nullspace of `mat` over GF(2)."""
    arr = _to_u8(mat)
    n = arr.shape[1]
    if arr.shape[0] == 0:
        return np.eye(n, dtype=np.uint8)
    rref, pivot_cols = rref_mod2(arr)
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(n) if c not in pivot_set]
    basis = []
    for free_c in free_cols:
        vec = np.zeros(n, dtype=np.uint8)
        vec[free_c] = 1
        for row_idx, pc in enumerate(pivot_cols):
            if rref[row_idx, free_c]:
                vec[pc] = 1
        basis.append(vec)
    if not basis:
        return np.zeros((0, n), dtype=np.uint8)
    return np.array(basis, dtype=np.uint8)


def _reduce_by_basis(vec: np.ndarray, rows: list[np.ndarray], pivot_cols: list[int]) -> np.ndarray:
    vec = vec.copy()
    for row, pc in zip(rows, pivot_cols):
        if vec[pc]:
            vec ^= row
    return vec


def quotient_basis(v_basis, w_basis, n: int) -> np.ndarray:
    """Basis of span(v_basis) / (span(v_basis) ∩ span(w_basis)) over GF(2),
    assuming span(w_basis) ⊆ span(v_basis). Representatives are original
    vectors from `v_basis`.
    """
    v_basis = _to_u8(v_basis) if len(v_basis) else np.zeros((0, n), dtype=np.uint8)
    w_basis = _to_u8(w_basis) if len(w_basis) else np.zeros((0, n), dtype=np.uint8)

    rows: list[np.ndarray] = []
    pivot_cols: list[int] = []
    if len(w_basis):
        w_rref, w_pivots = rref_mod2(w_basis)
        for row_idx, pc in enumerate(w_pivots):
            rows.append(w_rref[row_idx].copy())
            pivot_cols.append(pc)

    quotient = []
    for v in v_basis:
        reduced = _reduce_by_basis(v, rows, pivot_cols)
        if not reduced.any():
            continue
        pc = int(np.argmax(reduced))
        rows.append(reduced)
        pivot_cols.append(pc)
        quotient.append(v.copy())
    if not quotient:
        return np.zeros((0, n), dtype=np.uint8)
    return np.array(quotient, dtype=np.uint8)


def pair_logical_operators(lx_candidates, lz_candidates) -> tuple[np.ndarray, np.ndarray]:
    """Symplectic Gram-Schmidt: given k candidate X-type and k candidate
    Z-type logical operators (each spanning the same quotient dimension),
    return paired (Lx, Lz) with Lx_i . Lz_j = delta_ij (mod 2).
    """
    lx = [np.array(v, dtype=np.uint8) % 2 for v in lx_candidates]
    lz = [np.array(v, dtype=np.uint8) % 2 for v in lz_candidates]
    k = len(lx)
    if k != len(lz):
        raise ValueError(f"logical X/Z candidate counts differ: {k} vs {len(lz)}")

    paired_x = []
    paired_z = []
    remaining_x = list(range(k))
    remaining_z = list(range(k))
    for _ in range(k):
        found = None
        for ix in remaining_x:
            for iz in remaining_z:
                if int(np.dot(lx[ix], lz[iz])) % 2 == 1:
                    found = (ix, iz)
                    break
            if found:
                break
        if found is None:
            raise ValueError("degenerate symplectic form: could not pair logical operators")
        ix, iz = found
        a, b = lx[ix], lz[iz]
        paired_x.append(a)
        paired_z.append(b)
        remaining_x.remove(ix)
        remaining_z.remove(iz)
        for jx in remaining_x:
            if int(np.dot(lx[jx], b)) % 2 == 1:
                lx[jx] = lx[jx] ^ a
        for jz in remaining_z:
            if int(np.dot(a, lz[jz])) % 2 == 1:
                lz[jz] = lz[jz] ^ b

    n = lx[0].shape[0] if lx else (lz[0].shape[0] if lz else 0)
    if k == 0:
        return np.zeros((0, n), dtype=np.uint8), np.zeros((0, n), dtype=np.uint8)
    return np.array(paired_x, dtype=np.uint8), np.array(paired_z, dtype=np.uint8)
