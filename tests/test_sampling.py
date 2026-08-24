import numpy as np

from qec_dataset.noise import NoiseModel
from qec_dataset.sampling import (
    DatasetMetadata,
    generate_bb_code_dataset,
    generate_surface_code_dataset,
    load_dataset,
)


def test_surface_code_dataset_roundtrip(tmp_path):
    meta = generate_surface_code_dataset(
        distance=3, rounds=3, basis="Z", noise=NoiseModel.uniform(0.02), shots=200, out_dir=tmp_path
    )
    assert meta.n == 9
    assert meta.k == 1
    assert meta.num_observables == 1

    dets, obs = load_dataset(meta, tmp_path)
    assert dets.shape == (200, meta.num_detectors)
    assert obs.shape == (200, meta.num_observables)
    assert dets.dtype == np.bool_
    # noise is nonzero, so *some* shots should show detection events / flips
    assert dets.any()

    reloaded = DatasetMetadata.from_json(tmp_path / f"surface_d3_r3_Z.meta.json")
    assert reloaded == meta


def test_surface_code_dataset_noiseless_is_clean(tmp_path):
    meta = generate_surface_code_dataset(
        distance=3, rounds=3, basis="Z", noise=NoiseModel.uniform(0.0), shots=100, out_dir=tmp_path
    )
    dets, obs = load_dataset(meta, tmp_path)
    assert not dets.any()
    assert not obs.any()


def test_bb_code_dataset_roundtrip(tmp_path):
    meta = generate_bb_code_dataset(
        l=6,
        m=6,
        a_terms=[("x", 3), ("y", 1), ("y", 2)],
        b_terms=[("y", 3), ("x", 1), ("x", 2)],
        rounds=2,
        basis="Z",
        noise=NoiseModel.uniform(0.01),
        shots=100,
        out_dir=tmp_path,
    )
    assert meta.n == 72
    assert meta.k == 12
    assert meta.distance is None  # compute_distance defaults to False

    dets, obs = load_dataset(meta, tmp_path)
    assert dets.shape == (100, meta.num_detectors)
    assert obs.shape == (100, meta.num_observables)
    assert dets.any()
