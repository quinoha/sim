"""Code-capacity (single-round, no circuit-level noise) dataset generation.

This is a different noise/measurement model from sampling.py (multi-round
circuit-level noise via a stim circuit) and corner_cases.py (deterministic
directed cases). Here, i.i.d. Pauli errors are sampled directly on the n
data qubits and the syndrome is computed with one perfect matrix multiply
against Hx/Hz — no stim circuit, no rounds, no measurement error. This
matches the "code capacity" noise model.

The output format matches an existing GNN decoder pipeline's
`generate_syndrome_error_volume` / `adapt_trainset` / Tanner-graph edge
builder (astra/panq_functions.py, astra/bb_panq_functions.py): each sample
is a 1D array

    [syndrome_z (Hz.rows, values 0/1), syndrome_x (Hx.rows, values 0/2),
     error (n, values 0=I, 1=X, 2=Z, 3=Y)]

i.e. Z-check outcomes first (unscaled), X-check outcomes second (scaled by
2 so the two check types stay distinguishable once concatenated), then a
per-qubit ground-truth error label. Hx/Hz come from this package's own
CSSCode builders (codes/surface.py, codes/bb.py); for the BB presets these
are bit-identical to codes_q's, and `tanner_graph_edges` below reproduces
that pipeline's edge set exactly (both verified in
tests/test_astra_integration.py).
"""
from __future__ import annotations

import numpy as np

from .css_code import CSSCode


def sample_iid_pauli_errors(
    n: int,
    shots: int,
    p: float,
    x_frac: float = 1 / 3,
    y_frac: float = 1 / 3,
    z_frac: float = 1 / 3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Samples i.i.d. Pauli errors on n qubits, `shots` times, at total
    error probability `p` split across X/Y/Z by `x_frac`/`y_frac`/`z_frac`
    (must sum to 1). Returns (err_x, err_z) bool arrays of shape
    (shots, n) in the standard symplectic representation: a Y error sets
    both err_x and err_z for that qubit.
    """
    if abs(x_frac + y_frac + z_frac - 1.0) > 1e-9:
        raise ValueError(f"x_frac + y_frac + z_frac must sum to 1, got {x_frac + y_frac + z_frac}")
    rng = rng if rng is not None else np.random.default_rng()
    u = rng.uniform(0.0, 1.0, size=(shots, n))
    px, py, pz = p * x_frac, p * y_frac, p * z_frac
    err_x = u < (px + py)
    err_z = np.logical_and(u >= px, u < (px + py + pz))
    return err_x, err_z


def encode_samples(code: CSSCode, err_x: np.ndarray, err_z: np.ndarray) -> np.ndarray:
    """Packs (err_x, err_z) into the [syndrome_z, syndrome_x, error]
    format described in this module's docstring."""
    syndrome_z = (err_x.astype(np.uint32) @ code.Hz.T.astype(np.uint32)) % 2  # Z-checks detect X errors
    syndrome_x = (err_z.astype(np.uint32) @ code.Hx.T.astype(np.uint32)) % 2  # X-checks detect Z errors
    error_label = err_x.astype(np.uint8) + 2 * err_z.astype(np.uint8)  # 0=I,1=X,2=Z,3=Y
    return np.concatenate(
        [syndrome_z.astype(np.uint8), (2 * syndrome_x).astype(np.uint8), error_label],
        axis=1,
    )


def generate_code_capacity_samples(
    code: CSSCode,
    shots: int,
    p: float,
    x_frac: float = 1 / 3,
    y_frac: float = 1 / 3,
    z_frac: float = 1 / 3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Returns an array of shape (shots, Hz.rows + Hx.rows + n): each row
    is one code-capacity sample in the [syndrome_z, syndrome_x, error]
    format. `error_index = Hz.rows + Hx.rows` marks where the syndrome
    part ends and the error part begins (mirrors the source pipeline's
    `error_index` convention).
    """
    err_x, err_z = sample_iid_pauli_errors(code.n, shots, p, x_frac, y_frac, z_frac, rng)
    return encode_samples(code, err_x, err_z)


def error_index(code: CSSCode) -> int:
    """Length of the syndrome portion of a sample (= total number of
    checks); the error portion starts here and has length code.n."""
    return code.Hz.shape[0] + code.Hx.shape[0]


def tanner_graph_edges(code: CSSCode) -> tuple[np.ndarray, np.ndarray]:
    """Generic CSS Tanner-graph edges, in the same style as the source
    pipeline's `surface_code_edges` (built directly from Hx/Hz nonzero
    entries, so it works for any CSS code, not just surface codes).

    Node ids: 0..rz-1 are Z-check nodes, rz..rz+rx-1 are X-check nodes,
    rz+rx..rz+rx+n-1 are qubit nodes (matching `error_index`/`encode_samples`
    ordering). Returns (src_ids, dst_ids) with edges in both directions.
    """
    rz, n = code.Hz.shape
    rx = code.Hx.shape[0]
    num_checks = rz + rx

    z_check_idx, z_qubit_idx = code.Hz.nonzero()
    x_check_idx, x_qubit_idx = code.Hx.nonzero()

    check_ids = np.concatenate([z_check_idx, rz + x_check_idx])
    qubit_ids = np.concatenate([z_qubit_idx, x_qubit_idx]) + num_checks

    src_ids = np.concatenate([check_ids, qubit_ids])
    dst_ids = np.concatenate([qubit_ids, check_ids])
    return src_ids, dst_ids


def generate_surface_code_capacity_samples(
    distance: int, shots: int, p: float, **kwargs
) -> tuple[np.ndarray, CSSCode]:
    from .codes.surface import rotated_surface_code

    code, _ = rotated_surface_code(distance)
    return generate_code_capacity_samples(code, shots, p, **kwargs), code


def generate_bb_code_capacity_samples(
    l: int, m: int, a_terms, b_terms, shots: int, p: float, **kwargs
) -> tuple[np.ndarray, CSSCode]:
    from .codes.bb import build_bb_code

    code = build_bb_code(l, m, a_terms, b_terms)
    return generate_code_capacity_samples(code, shots, p, **kwargs), code
