"""Surface-code grid masks used by the Cascade checkpoints.

Vendored from the external Cascade project (`cascade/geometry/surface_code.py`),
which the training scripts in `training/` import. It is kept here verbatim so
the checkpoints in `checkpoints/` remain reproducible from inside this repo
instead of depending on a path outside it.

These masks are a SYNTHETIC STAND-IN, and that matters for reading results:
`checkerboard_ancilla_mask` keeps only the `(i+j)` even sublattice of the
(d+1, d+1) grid, which in a rotated surface code corresponds to one check type.
Roughly 43% of the real stim detectors land on the other sublattice and are
encoded as NOT_A_CHECK_SITE -- the trained models therefore never see them.
That is a ceiling on what these checkpoints can do, not a property of the
architecture; `adapter.py`'s default "stim" mask mode feeds every detector and
is the right basis for retraining.
"""

import torch


def checkerboard_ancilla_mask(distance: int) -> torch.Tensor:
    """Placeholder ancilla (check-site) mask: True at (i+j) even sites on the
    (d+1, d+1) grid. NOT validated against real surface-code boundary
    combinatorics.
    """
    if distance < 1:
        raise ValueError(f"distance must be >= 1, got {distance}")
    grid = distance + 1
    i = torch.arange(grid).unsqueeze(1)
    j = torch.arange(grid).unsqueeze(0)
    return ((i + j) % 2 == 0).expand(grid, grid).clone()


def synthetic_data_qubit_mask(distance: int, ancilla_mask: torch.Tensor) -> torch.Tensor:
    """Placeholder data-qubit mask: complement of `ancilla_mask` on the same
    (d+1, d+1) grid. Shape-compatible stand-in only -- see module docstring
    for why this does not hold at real code distances.
    """
    grid = distance + 1
    if ancilla_mask.shape != (grid, grid):
        raise ValueError(f"ancilla_mask shape {tuple(ancilla_mask.shape)} != ({grid}, {grid})")
    return ~ancilla_mask


def synthetic_logical_masks(distance: int, data_qubit_mask: torch.Tensor) -> torch.Tensor:
    """Placeholder logical-operator support: one middle row and one middle
    column of `data_qubit_mask`, standing in for the boundary-to-boundary
    XL/ZL supports of a real surface code. Returns (2, d+1, d+1) bool.
    """
    grid = distance + 1
    if data_qubit_mask.shape != (grid, grid):
        raise ValueError(f"data_qubit_mask shape {tuple(data_qubit_mask.shape)} != ({grid}, {grid})")

    row_mask = torch.zeros(grid, grid, dtype=torch.bool)
    row_mask[grid // 2, :] = True
    row_mask &= data_qubit_mask

    col_mask = torch.zeros(grid, grid, dtype=torch.bool)
    col_mask[:, grid // 2] = True
    col_mask &= data_qubit_mask

    if not row_mask.any() or not col_mask.any():
        raise ValueError(
            "synthetic logical mask has empty support for this distance -- "
            "the middle-row/column placeholder needs a larger distance"
        )
    return torch.stack([row_mask, col_mask], dim=0)
