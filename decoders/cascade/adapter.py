"""Bridge between stim detector data and the Cascade surface-code model.

`SurfaceCascade` consumes a `(B, T, G, G)` LongTensor of per-site syndrome
classes, but a stim memory experiment hands back a flat `(shots,
num_detectors)` bit matrix. This module derives the mapping between the two
directly from the circuit's own `DETECTOR` coordinates, so nothing about the
lattice is hardcoded.

For a rotated surface code stim emits detector coordinates `(x, y, t)` with
x, y on even integers `0, 2, ... 2d` and `t` one layer per measurement
round *plus one*: a distance-5, 5-round Z-memory circuit has 120 detectors
over t = 0..5, where t=0 and t=5 carry only the 12 Z-checks and the four
interior layers carry all 24 checks. Dividing the coordinates by two gives a
(d+1) x (d+1) grid -- exactly the grid `SyndromeEmbedding` expects, with the
24 real check sites and 12 structurally-absent ones.

Two consequences worth stating plainly:

1. **The model's `rounds` argument must be the detector layer count, not the
   syndrome round count.** `Readout`'s scatter convolution has a time kernel
   of exactly `rounds`, so it collapses T -> 1 only when they agree. Build
   the model with `rounds = T = circuit_rounds + 1`.

2. Sites that do not exist at a given layer (the X-checks at t=0 and t=T-1)
   fall out as `NOT_A_CHECK_SITE` for free, because the index tensor is
   pre-filled with that class and only real detectors overwrite it. That is
   the correct structural signal, and it is time-dependent -- which is why
   this module builds the index tensor itself instead of going through
   `syndrome_indices_from_detections`, whose mask is a single static (G, G).

Readout deviation: `readout.py` documents that data qubits and ancillas
cannot both live on the same (d+1, d+1) grid (d^2 data qubits do not fit in
the 2d+2 leftover sites), and leaves the resolution open. This adapter pools
each logical observable over *all real check sites* rather than scattering
to data qubits. That is a legitimate global-pool readout for a memory
experiment and keeps `forward()` untouched, but it is not the paper's
scatter-to-data-qubit picture; swapping in real stim-derived qubit
coordinates later only changes the masks passed here.
"""
from __future__ import annotations

import numpy as np
import stim
import torch

from .embedding import NOT_A_CHECK_SITE
from .geometry import (
    checkerboard_ancilla_mask,
    synthetic_data_qubit_mask,
    synthetic_logical_masks,
)
from .surface_cascade import SurfaceCascade


def checkpoint_architecture(state_dict: dict) -> dict:
    """Recovers (hidden_dim, depth, rounds, grid, num_logicals) from weights.

    The checkpoints carry no config, but the shapes pin every architectural
    choice: the embedding table's width is `hidden_dim`, the `blocks.<i>.`
    prefixes count `depth`, the readout's scatter convolution has a time
    kernel of exactly `rounds`, and its registered geometry buffers give the
    grid and the logical count.
    """
    sd = {
        (k[len("module.") :] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
        if torch.is_tensor(v)
    }
    try:
        hidden_dim = sd["embedding.embed.weight"].shape[1]
        scatter = sd["readout.scatter_conv.weight"]
        flat_idx = sd["readout.data_qubit_flat_indices"]
        pool = sd["readout.logical_pool_weights"]
    except KeyError as e:
        raise ValueError(f"not a SurfaceCascade state dict: missing {e}") from e

    depth = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    bottleneck = sd.get("blocks.0.message_passing.weight")
    grid = int(np.sqrt(int(flat_idx.max()) + 1))
    if grid * grid < int(flat_idx.max()) + 1:
        grid += 1

    return {
        "state_dict": sd,
        "hidden_dim": int(hidden_dim),
        "depth": depth,
        "rounds": int(scatter.shape[2]),
        "grid": grid,
        "num_logicals": int(pool.shape[0]),
        "bottleneck_ratio": int(hidden_dim // bottleneck.shape[0]) if bottleneck is not None else 4,
    }


def detector_grid_map(circuit: stim.Circuit) -> dict:
    """Derives the (T, G, G) grid layout from a circuit's DETECTOR coordinates.

    Returns a dict with the layer/row/column index arrays (one entry per
    detector, in detector order), the grid shape, and the union-over-time
    mask of real check sites.
    """
    coords = circuit.get_detector_coordinates()
    if not coords:
        raise ValueError("circuit has no detector coordinates; cannot build a grid map")

    first = next(iter(coords.values()))
    if len(first) < 3:
        raise ValueError(
            f"expected 3-D detector coordinates (x, y, t), got {len(first)}-D. "
            "This adapter targets stim's rotated surface-code memory circuits."
        )

    xs = sorted({c[0] for c in coords.values()})
    ys = sorted({c[1] for c in coords.values()})
    ts = sorted({c[2] for c in coords.values()})

    if len(xs) != len(ys):
        raise ValueError(f"detector grid is not square: {len(xs)} x-values vs {len(ys)} y-values")

    x_to_col = {x: i for i, x in enumerate(xs)}
    y_to_row = {y: i for i, y in enumerate(ys)}
    t_to_layer = {t: i for i, t in enumerate(ts)}

    num_detectors = circuit.num_detectors
    layer = np.empty(num_detectors, dtype=np.int64)
    row = np.empty(num_detectors, dtype=np.int64)
    col = np.empty(num_detectors, dtype=np.int64)

    for det_index, c in coords.items():
        layer[det_index] = t_to_layer[c[2]]
        row[det_index] = y_to_row[c[1]]
        col[det_index] = x_to_col[c[0]]

    grid = len(xs)
    valid_site_mask = np.zeros((grid, grid), dtype=bool)
    valid_site_mask[row, col] = True

    return {
        "layer": layer,
        "row": row,
        "col": col,
        "T": len(ts),
        "grid": grid,
        "valid_site_mask": valid_site_mask,
    }


class SurfaceCascadeDecoder:
    """Runs `SurfaceCascade` over stim detector data.

    Satisfies the runner's custom-decoder contract: an instance is callable
    with a `(shots, num_detectors)` array and returns `(shots,
    num_observables)` uint8 predictions.
    """

    def __init__(
        self,
        circuit: stim.Circuit,
        distance: int,
        hidden_dim: int = 32,
        depth: int = 5,
        bottleneck_ratio: int = 4,
        weights: str | None = None,
        device: str | None = None,
        batch_size: int = 1024,
        mask_mode: str = "stim",
    ):
        if mask_mode not in ("stim", "checkerboard"):
            raise ValueError(f"mask_mode must be 'stim' or 'checkerboard', got {mask_mode!r}")

        self.map = detector_grid_map(circuit)
        grid = self.map["grid"]
        self.T = self.map["T"]
        self.num_detectors = circuit.num_detectors
        self.num_observables = circuit.num_observables
        self.batch_size = batch_size
        self.mask_mode = mask_mode

        if grid != distance + 1:
            raise ValueError(
                f"detector grid is {grid}x{grid} but SurfaceCascade requires (d+1)x(d+1) "
                f"= {distance + 1}x{distance + 1} for distance={distance}"
            )

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        if mask_mode == "checkerboard":
            # Reproduces what the checkpoints in checkpoints/ were trained with.
            self.ancilla_mask = checkerboard_ancilla_mask(distance)
            data_mask = synthetic_data_qubit_mask(distance, self.ancilla_mask)
            logical_masks = synthetic_logical_masks(distance, data_mask)[1:2]
            if self.num_observables != logical_masks.shape[0]:
                raise ValueError(
                    f"checkerboard mode builds {logical_masks.shape[0]} logical mask(s) but the "
                    f"circuit has {self.num_observables} observables"
                )
        else:
            self.ancilla_mask = None
            data_mask = torch.from_numpy(self.map["valid_site_mask"])
            # See module docstring: global pool over real check sites, one row per observable.
            logical_masks = (
                data_mask.unsqueeze(0).expand(max(self.num_observables, 1), grid, grid).clone()
            )

        self.model = SurfaceCascade(
            distance=distance,
            rounds=self.T,  # detector layer count, NOT the syndrome round count
            hidden_dim=hidden_dim,
            depth=depth,
            data_qubit_mask=data_mask,
            logical_masks=logical_masks,
            bottleneck_ratio=bottleneck_ratio,
        ).to(self.device)

        if weights is not None:
            arch = checkpoint_architecture(torch.load(weights, map_location="cpu"))
            self.model.load_state_dict(arch["state_dict"])
            self.trained = True
        else:
            self.trained = False

        self.model.eval()
        if self.ancilla_mask is not None:
            self.ancilla_mask = self.ancilla_mask.to(self.device)

        self._layer = torch.from_numpy(self.map["layer"]).to(self.device)
        self._row = torch.from_numpy(self.map["row"]).to(self.device)
        self._col = torch.from_numpy(self.map["col"]).to(self.device)

    def to_syndrome_indices(self, dets: np.ndarray) -> torch.Tensor:
        """(shots, num_detectors) bits -> (shots, T, G, G) values in {0, 1, 2}."""
        bits = torch.as_tensor(np.ascontiguousarray(dets), device=self.device)
        if bits.ndim != 2 or bits.shape[1] != self.num_detectors:
            raise ValueError(
                f"expected detector array of shape (shots, {self.num_detectors}), "
                f"got {tuple(bits.shape)}"
            )
        b = bits.shape[0]
        grid = self.map["grid"]

        if self.mask_mode == "checkerboard":
            # Training convention: fill a dense grid, then blank every site outside the
            # static ancilla checkerboard. Detectors on the other sublattice are lost.
            dense = torch.zeros((b, self.T, grid, grid), dtype=torch.long, device=self.device)
            dense[:, self._layer, self._row, self._col] = bits.long()
            return torch.where(
                self.ancilla_mask,
                dense,
                torch.full_like(dense, NOT_A_CHECK_SITE),
            )

        idx = torch.full(
            (b, self.T, grid, grid), NOT_A_CHECK_SITE, dtype=torch.long, device=self.device
        )
        idx[:, self._layer, self._row, self._col] = bits.long()
        return idx

    @torch.no_grad()
    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        out = np.empty((dets.shape[0], self.num_observables), dtype=np.uint8)
        for start in range(0, dets.shape[0], self.batch_size):
            chunk = dets[start : start + self.batch_size]
            logits = self.model(self.to_syndrome_indices(chunk))
            out[start : start + chunk.shape[0]] = (logits > 0).to(torch.uint8).cpu().numpy()
        return out

    def __call__(self, dets: np.ndarray) -> np.ndarray:
        return self.decode_batch(dets)


def build_cascade_decoder(circuit: stim.Circuit, distance: int, **kwargs) -> SurfaceCascadeDecoder:
    """Module-level factory, safe to hand to a worker-process initializer."""
    return SurfaceCascadeDecoder(circuit, distance, **kwargs)


def load_pretrained(
    circuit: stim.Circuit, distance: int, weights: str, **kwargs
) -> SurfaceCascadeDecoder:
    """Loads one of `checkpoints/*.pth` with the geometry it was trained on.

    Architecture is read back off the weights, so the caller does not have to
    know that e.g. `cascade_d7_H256_ddp_ema.pth` is H=256, L=7. `mask_mode`
    defaults to "checkerboard" here because that is the convention
    `training/` used; loading a checkpoint under the "stim" mask feeds the
    model an input distribution it never saw.

    These checkpoints were trained at p=0.01 with rounds=distance. They do
    not transfer to much lower physical error rates -- at p=1e-3 the d=7
    model scores worse than predicting no logical flip at all.
    """
    arch = checkpoint_architecture(torch.load(weights, map_location="cpu"))
    kwargs.setdefault("mask_mode", "checkerboard")
    decoder = SurfaceCascadeDecoder(
        circuit,
        distance,
        hidden_dim=arch["hidden_dim"],
        depth=arch["depth"],
        bottleneck_ratio=arch["bottleneck_ratio"],
        weights=weights,
        **kwargs,
    )
    if decoder.T != arch["rounds"]:
        raise ValueError(
            f"checkpoint expects {arch['rounds']} detector layers but this circuit has "
            f"{decoder.T}; the checkpoints were trained with rounds=distance"
        )
    return decoder
