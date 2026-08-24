"""Layer-A dataset generation: raw detection events plus the *ground truth*
logical observable flips, sampled directly from a stim circuit.

This layer is deliberately independent of any RTL interface or decoder
algorithm choice (see docs/rtl_test_vector_roadmap.md) — it only needs a
circuit (surface or BB code, any noise/rounds) and a shot count. The
observable flips written here are the *actual* logical error that occurred
in each shot, known exactly because stim tracks the noise it injected —
no decoder is involved in producing this label. A decoder's prediction can
later be compared against it, but is not required to define it.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import stim

VALID_FORMATS = ("01", "b8", "r8", "ptb64", "hits", "dets")


@dataclass
class DatasetMetadata:
    code_name: str
    n: int
    k: int
    distance: int | None
    distance_method: str | None
    basis: str
    rounds: int
    shots: int
    noise: dict
    num_detectors: int
    num_observables: int
    detector_format: str
    observable_format: str
    detectors_file: str
    observables_file: str

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def from_json(path: str | Path) -> "DatasetMetadata":
        return DatasetMetadata(**json.loads(Path(path).read_text()))


def sample_shots_to_files(
    circuit: stim.Circuit,
    shots: int,
    out_dir: str | Path,
    prefix: str = "shots",
    det_format: str = "b8",
    obs_format: str = "b8",
) -> tuple[Path, Path]:
    """Samples `shots` shots from `circuit`, writing detection events and
    ground-truth observable flips to two separate files. Streams directly
    to disk via stim's own writer, so shot counts that don't fit in memory
    are fine.
    """
    if det_format not in VALID_FORMATS or obs_format not in VALID_FORMATS:
        raise ValueError(f"format must be one of {VALID_FORMATS}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    det_path = out_dir / f"{prefix}.dets.{det_format}"
    obs_path = out_dir / f"{prefix}.obs.{obs_format}"

    sampler = circuit.compile_detector_sampler()
    sampler.sample_write(
        shots,
        filepath=str(det_path),
        format=det_format,
        obs_out_filepath=str(obs_path),
        obs_out_format=obs_format,
    )
    return det_path, obs_path


def load_dataset(meta: DatasetMetadata, out_dir: str | Path, bit_packed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Reads back a dataset written by `sample_shots_to_files` (+ metadata).
    Returns (detection_events, observable_flips) as bool arrays of shape
    (shots, num_detectors) / (shots, num_observables), or bit-packed uint8
    arrays if bit_packed=True.
    """
    out_dir = Path(out_dir)
    dets = stim.read_shot_data_file(
        path=str(out_dir / meta.detectors_file),
        format=meta.detector_format,
        num_detectors=meta.num_detectors,
        bit_packed=bit_packed,
    )
    obs = stim.read_shot_data_file(
        path=str(out_dir / meta.observables_file),
        format=meta.observable_format,
        num_observables=meta.num_observables,
        bit_packed=bit_packed,
    )
    return dets, obs


def _noise_to_dict(noise) -> dict:
    return dataclasses.asdict(noise) if dataclasses.is_dataclass(noise) else dict(noise)


def generate_surface_code_dataset(
    distance: int,
    rounds: int,
    basis: str,
    noise,
    shots: int,
    out_dir: str | Path,
    prefix: str | None = None,
) -> DatasetMetadata:
    from .codes.surface import build_surface_code_circuit, rotated_surface_code

    code, _ = rotated_surface_code(distance, basis=basis)
    circuit = build_surface_code_circuit(distance, rounds, basis=basis, noise=noise)
    prefix = prefix or f"surface_d{distance}_r{rounds}_{basis}"

    det_path, obs_path = sample_shots_to_files(circuit, shots, out_dir, prefix=prefix)
    meta = DatasetMetadata(
        code_name=code.name,
        n=code.n,
        k=code.k,
        distance=distance,
        distance_method="known_parameter",
        basis=basis,
        rounds=rounds,
        shots=shots,
        noise=_noise_to_dict(noise),
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
        detector_format="b8",
        observable_format="b8",
        detectors_file=det_path.name,
        observables_file=obs_path.name,
    )
    meta.to_json(Path(out_dir) / f"{prefix}.meta.json")
    return meta


def generate_bb_code_dataset(
    l: int,
    m: int,
    a_terms,
    b_terms,
    rounds: int,
    basis: str,
    noise,
    shots: int,
    out_dir: str | Path,
    prefix: str | None = None,
    compute_distance: bool = False,
) -> DatasetMetadata:
    from .circuits import generic_css_memory_circuit
    from .codes.bb import build_bb_code

    code = build_bb_code(l, m, a_terms, b_terms)
    circuit = generic_css_memory_circuit(code, rounds, basis=basis, noise=noise)
    prefix = prefix or f"bb_l{l}_m{m}_r{rounds}_{basis}"

    distance = None
    distance_method = None
    if compute_distance:
        from .distance import search_distance

        probe_circuit = generic_css_memory_circuit(code, rounds, basis=basis, noise=type(noise).uniform(1e-3))
        result = search_distance(probe_circuit)
        distance = result.distance
        distance_method = result.method

    det_path, obs_path = sample_shots_to_files(circuit, shots, out_dir, prefix=prefix)
    meta = DatasetMetadata(
        code_name=code.name,
        n=code.n,
        k=code.k,
        distance=distance,
        distance_method=distance_method,
        basis=basis,
        rounds=rounds,
        shots=shots,
        noise=_noise_to_dict(noise),
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
        detector_format="b8",
        observable_format="b8",
        detectors_file=det_path.name,
        observables_file=obs_path.name,
    )
    meta.to_json(Path(out_dir) / f"{prefix}.meta.json")
    return meta
