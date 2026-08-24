"""Adapter presenting a `CSSCode` under the attribute names the external
GNN decoder pipeline in `astra/` expects.

That pipeline (`astra/bb_panq_functions.py`, `astra/panq_functions.py`) was
written against `codes_q.css_code`, so its functions read `code.N`,
`code.hx`, `code.hz`, `code.hx_perp`, `code.hz_perp`, `code.D`, `code.K`,
`code.name` — lowercase `hx`/`hz`, uppercase `N`. This package uses
`code.n`, `code.Hx`, `code.Hz`, `code.k`. Rather than renaming ours (or
depending on `codes_q`), this module wraps ours in a read-only view.

`hx_perp`/`hz_perp` are the GF(2) kernels (nullspaces) of Hx/Hz, i.e. the
same thing `codes_q.css_code` computes with `utils.kernel(...)`. The
pipeline's `ler_loss` multiplies a (batch, N) residual by `hx_perp.T`, so
basis vectors must be ROWS of length N — which is what
`gf2.nullspace_mod2` returns.

The Hx/Hz themselves come from this package's own builders. For all six
BB presets in `codes/bb.py:BB_PRESETS` these were verified **bit-identical**
to `codes_q.create_bivariate_bicycle_codes`'s output (see
tests/test_astra_integration.py), so parity-check matrices and trained
checkpoints are interchangeable. Surface codes are built via stim's
generator rather than panqec's, so no such guarantee is claimed there.
"""
from __future__ import annotations

import numpy as np

from . import gf2
from .css_code import CSSCode


class AstraCodeView:
    """Read-only view of a CSSCode using astra/codes_q attribute names."""

    def __init__(self, code: CSSCode, distance: int | float | None = None):
        self._code = code
        self.N = code.n
        self.K = code.k
        self.D = np.nan if distance is None else distance
        self.name = code.name

        self.hx = np.asarray(code.Hx, dtype=int)
        self.hz = np.asarray(code.Hz, dtype=int)
        self.hx_perp = gf2.nullspace_mod2(code.Hx).astype(int)
        self.hz_perp = gf2.nullspace_mod2(code.Hz).astype(int)

        self.lx = np.asarray(code.logical_x, dtype=int)
        self.lz = np.asarray(code.logical_z, dtype=int)

    @property
    def css_code(self) -> CSSCode:
        """The underlying CSSCode this view wraps."""
        return self._code

    def __repr__(self) -> str:
        return f"AstraCodeView(name={self.name!r}, N={self.N}, K={self.K}, D={self.D})"
