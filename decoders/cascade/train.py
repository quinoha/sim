"""Supervised training for `SurfaceCascade` on stim memory-experiment shots.

The label is the ground-truth observable flip stim already records (see
`qec_dataset/sampling.py`) -- no reference decoder is involved in producing
it, so the model is trained against the true logical error, not against
pymatching's opinion of it.

Following the paper's methodology, a model is trained per configuration:
`Readout`'s time kernel is sized for one specific detector-layer count, so a
checkpoint is only valid for the `(distance, rounds)` it was trained on. The
physical error rate is not baked into the architecture, but a model trained
at one `p` is not expected to transfer far from it.

Class imbalance is real here: at p=1e-3 a distance-5 memory experiment flips
the observable in roughly 5% of shots. `pos_weight` on the BCE loss keeps the
model from collapsing onto the majority class, which is exactly the failure
an untrained network already exhibits (it predicts one constant class for
every shot).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import stim
import torch
import torch.nn as nn

from .adapter import SurfaceCascadeDecoder


@dataclass
class TrainConfig:
    train_shots: int = 200_000
    val_shots: int = 20_000
    epochs: int = 8
    batch_size: int = 512
    lr: float = 3e-3
    weight_decay: float = 0.0
    seed: int = 1234
    pos_weight: float | None = None  # None -> derived from the training labels
    log_every: int = 0  # 0 = once per epoch


@dataclass
class TrainHistory:
    val_ler: list[float] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    epoch_seconds: list[float] = field(default_factory=list)

    @property
    def best_val_ler(self) -> float:
        return min(self.val_ler) if self.val_ler else float("nan")


def sample_labelled(circuit: stim.Circuit, shots: int, seed: int):
    """Returns (detection_events, observable_flips) for `shots` shots."""
    return circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True
    )


@torch.no_grad()
def evaluate_ler(decoder: SurfaceCascadeDecoder, dets: np.ndarray, obs: np.ndarray) -> float:
    """Fraction of shots where any predicted observable differs from the truth."""
    was_training = decoder.model.training
    decoder.model.eval()
    pred = decoder.decode_batch(dets)
    if was_training:
        decoder.model.train()
    return float(np.any(pred != obs, axis=1).mean())


def train_cascade(
    decoder: SurfaceCascadeDecoder,
    circuit: stim.Circuit,
    cfg: TrainConfig | None = None,
    save_to: str | None = None,
) -> TrainHistory:
    """Trains `decoder.model` in place and returns the per-epoch history."""
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)

    train_dets, train_obs = sample_labelled(circuit, cfg.train_shots, cfg.seed)
    val_dets, val_obs = sample_labelled(circuit, cfg.val_shots, cfg.seed + 1)

    y = torch.from_numpy(train_obs).float()
    if cfg.pos_weight is None:
        positives = y.sum().clamp(min=1.0)
        pos_weight = ((y.numel() - positives) / positives).to(decoder.device)
    else:
        pos_weight = torch.tensor(cfg.pos_weight, device=decoder.device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(
        decoder.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    history = TrainHistory()
    n = cfg.train_shots
    rng = np.random.default_rng(cfg.seed)

    for epoch in range(1, cfg.epochs + 1):
        decoder.model.train()
        order = rng.permutation(n)
        total, nb = 0.0, 0
        t0 = time.perf_counter()

        for start in range(0, n, cfg.batch_size):
            sel = order[start : start + cfg.batch_size]
            idx = decoder.to_syndrome_indices(train_dets[sel])
            target = torch.as_tensor(train_obs[sel], device=decoder.device).float()

            logits = decoder.model(idx)
            loss = loss_fn(logits, target)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()
            nb += 1

        secs = time.perf_counter() - t0
        ler = evaluate_ler(decoder, val_dets, val_obs)
        history.train_loss.append(total / max(nb, 1))
        history.val_ler.append(ler)
        history.epoch_seconds.append(secs)
        print(
            f"  epoch {epoch:>2}/{cfg.epochs}  loss={total / max(nb, 1):.4f}  "
            f"val LER={ler:.4f}  ({secs:.1f}s)"
        )

    decoder.model.eval()
    decoder.trained = True
    if save_to:
        torch.save(decoder.model.state_dict(), save_to)
        print(f"  saved weights -> {save_to}")
    return history
