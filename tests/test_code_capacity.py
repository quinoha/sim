import numpy as np

from qec_dataset.code_capacity import (
    error_index,
    generate_bb_code_capacity_samples,
    generate_surface_code_capacity_samples,
    tanner_graph_edges,
)


def test_surface_code_capacity_shape_and_noiseless():
    samples, code = generate_surface_code_capacity_samples(distance=3, shots=200, p=0.1)
    ei = error_index(code)
    assert samples.shape == (200, ei + code.n)
    assert ei == code.Hz.shape[0] + code.Hx.shape[0]
    assert set(np.unique(samples[:, :ei]).tolist()) <= {0, 1, 2}
    assert set(np.unique(samples[:, ei:]).tolist()) <= {0, 1, 2, 3}

    zero_samples, _ = generate_surface_code_capacity_samples(distance=3, shots=50, p=0.0)
    assert not zero_samples.any()


def test_bb_code_capacity_shape_and_noiseless():
    samples, code = generate_bb_code_capacity_samples(
        l=6, m=6, a_terms=[("x", 3), ("y", 1), ("y", 2)], b_terms=[("y", 3), ("x", 1), ("x", 2)],
        shots=100, p=0.05,
    )
    ei = error_index(code)
    assert samples.shape == (100, ei + code.n)
    assert ei == code.n  # true for BB codes specifically: rz == rx == n/2

    zero_samples, _ = generate_bb_code_capacity_samples(
        l=6, m=6, a_terms=[("x", 3), ("y", 1), ("y", 2)], b_terms=[("y", 3), ("x", 1), ("x", 2)],
        shots=20, p=0.0,
    )
    assert not zero_samples.any()


def test_tanner_graph_edges_node_range():
    _, code = generate_surface_code_capacity_samples(distance=3, shots=1, p=0.0)
    ei = error_index(code)
    src, dst = tanner_graph_edges(code)
    assert src.min() >= 0
    assert max(src.max(), dst.max()) == ei + code.n - 1
