"""Integration smoke test against the external GNN decoder pipeline in astra/.

Skipped unless that pipeline is present. Point ASTRA_PATH at it if it
isn't at ~/astra.

These tests verify the three things that have to line up for our
code-capacity datasets to be usable there:
  1. hx_perp/hz_perp match what codes_q's kernel() computes,
  2. our Tanner-graph edge set matches its surface_code_edges(),
  3. our BB preset Hx/Hz are bit-identical to codes_q's,
and then run one real forward pass + loss through its GNNDecoder.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("MPLBACKEND", "Agg")  # astra imports matplotlib

ASTRA_PATH = os.environ.get("ASTRA_PATH", os.path.expanduser("~/astra"))

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ASTRA_PATH, "bb_panq_functions.py")),
    reason=f"astra pipeline not found at {ASTRA_PATH} (set ASTRA_PATH)",
)


@pytest.fixture(scope="module")
def astra():
    """Imports the astra pipeline modules, skipping if deps are missing."""
    if ASTRA_PATH not in sys.path:
        sys.path.insert(0, ASTRA_PATH)
    try:
        import bb_panq_functions
        import utils
    except Exception as e:  # missing torch/panqec/sympy/etc.
        pytest.skip(f"astra pipeline not importable: {e}")
    return bb_panq_functions, utils


@pytest.fixture(scope="module")
def preset_code():
    from qec_dataset.codes.bb import build_bb_preset

    return build_bb_preset(6)  # [[72, 12, 6]]


@pytest.fixture(scope="module")
def view(preset_code):
    from qec_dataset.astra_adapter import AstraCodeView

    return AstraCodeView(preset_code, distance=6)


def _rowspace_key(mat):
    """Canonical form of a GF(2) rowspace, so two different bases of the
    same space compare equal."""
    from qec_dataset.gf2 import rref_mod2

    mat = np.asarray(mat, dtype=np.uint8) % 2
    if mat.size == 0:
        return ()
    rref, pivots = rref_mod2(mat)
    return tuple(sorted(tuple(int(v) for v in rref[i]) for i in range(len(pivots))))


def test_perp_matrices_match_codes_q_kernel(astra, view):
    _, utils = astra
    assert _rowspace_key(view.hx_perp) == _rowspace_key(utils.kernel(view.hx)[0])
    assert _rowspace_key(view.hz_perp) == _rowspace_key(utils.kernel(view.hz)[0])
    # and they really are kernels
    assert not ((view.hx @ view.hx_perp.T) % 2).any()
    assert not ((view.hz @ view.hz_perp.T) % 2).any()


def test_bb_presets_are_bit_identical_to_codes_q(astra):
    bb_panq_functions, _ = astra
    from qec_dataset.codes.bb import BB_PRESETS, build_bb_preset

    for d, preset in BB_PRESETS.items():
        ours = build_bb_preset(d)
        theirs = bb_panq_functions.bb_code(d)
        assert (ours.n, ours.k) == (preset.n, preset.k) == (theirs.N, theirs.K)
        assert np.array_equal(np.asarray(ours.Hx, dtype=int), np.asarray(theirs.hx, dtype=int)), d
        assert np.array_equal(np.asarray(ours.Hz, dtype=int), np.asarray(theirs.hz, dtype=int)), d


def test_tanner_edges_match_astra(astra, preset_code, view):
    bb_panq_functions, _ = astra
    from qec_dataset.code_capacity import tanner_graph_edges

    our_src, our_dst = tanner_graph_edges(preset_code)
    their_src, their_dst = bb_panq_functions.surface_code_edges(view)
    ours = set(zip(our_src.tolist(), our_dst.tolist()))
    theirs = set(zip(np.asarray(their_src).tolist(), np.asarray(their_dst).tolist()))
    assert ours == theirs


def test_sample_layout_matches_astra_convention(preset_code, view):
    from qec_dataset.code_capacity import error_index, generate_code_capacity_samples

    samples = generate_code_capacity_samples(
        preset_code, shots=8, p=0.05, rng=np.random.default_rng(0)
    )
    # astra: size = 2 * code.N, error_index = code.N
    assert samples.shape[1] == 2 * view.N
    assert error_index(preset_code) == view.N


def test_full_forward_pass_and_loss(astra, preset_code, view):
    bb_panq_functions, _ = astra
    torch = pytest.importorskip("torch")
    from torch.utils.data import DataLoader

    from qec_dataset.code_capacity import generate_code_capacity_samples, tanner_graph_edges

    GNNDecoder = bb_panq_functions.GNNDecoder
    device = torch.device("cpu")
    n_node_inputs = n_node_outputs = 4  # syndrome vals 0..2, error labels 0..3
    n_iters = 2
    batch_size = 4

    src, dst = tanner_graph_edges(preset_code)
    gnn = GNNDecoder(
        dist=6, n_node_inputs=n_node_inputs, n_node_outputs=n_node_outputs,
        n_iters=n_iters, n_node_features=16, n_edge_features=16, msg_net_size=32,
    ).to(device)
    GNNDecoder.dist = 6
    GNNDecoder.surface_code_edges = (torch.LongTensor(src), torch.LongTensor(dst))
    GNNDecoder.hxperp = torch.FloatTensor(view.hx_perp).to(device)
    GNNDecoder.hzperp = torch.FloatTensor(view.hz_perp).to(device)
    GNNDecoder.device = device

    samples = generate_code_capacity_samples(
        preset_code, shots=16, p=0.05, rng=np.random.default_rng(0)
    )
    trainset = bb_panq_functions.adapt_trainset(samples, view, num_classes=n_node_inputs)
    loader = DataLoader(
        trainset, batch_size=batch_size, collate_fn=bb_panq_functions.collate, shuffle=False
    )
    inputs, targets, src_ids, dst_ids = next(iter(loader))

    expected_nodes = batch_size * 2 * view.N
    assert inputs.shape == (expected_nodes, n_node_inputs)
    assert targets.shape == (expected_nodes,)
    assert int(max(src_ids.max(), dst_ids.max())) < expected_nodes

    # no .float() cast -- exactly how astra's own training loop calls it
    outputs = gnn(inputs.to(device), src_ids.to(device), dst_ids.to(device))
    assert outputs.shape == (n_iters, expected_nodes, n_node_outputs)

    loss = bb_panq_functions.ler_loss(outputs[-1], targets.to(device), view)
    assert torch.isfinite(torch.as_tensor(loss)).all()
