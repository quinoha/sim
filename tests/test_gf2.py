import numpy as np
import pytest

from qec_dataset import gf2


def test_rank_mod2_basic():
    m = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.uint8)  # rows sum to 0 -> rank 2
    assert gf2.rank_mod2(m) == 2

    identity = np.eye(4, dtype=np.uint8)
    assert gf2.rank_mod2(identity) == 4

    zeros = np.zeros((3, 5), dtype=np.uint8)
    assert gf2.rank_mod2(zeros) == 0


def test_nullspace_mod2_dimension_and_correctness():
    m = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)
    ns = gf2.nullspace_mod2(m)
    assert ns.shape[1] == 4
    assert ns.shape[0] == 4 - gf2.rank_mod2(m)
    product = (m.astype(np.uint32) @ ns.T.astype(np.uint32)) % 2
    assert not product.any()


def test_quotient_basis_dimension():
    # W = rowspace of a 1x4 matrix, V = nullspace of a smaller constraint containing W
    w = np.array([[1, 1, 0, 0]], dtype=np.uint8)
    v = gf2.nullspace_mod2(np.zeros((0, 4), dtype=np.uint8))  # full space, dim 4
    q = gf2.quotient_basis(v, w, 4)
    assert q.shape == (3, 4)


def test_pair_logical_operators_symplectic():
    k = 2
    lx = np.array([[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=np.uint8)
    lz = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=np.uint8)
    px, pz = gf2.pair_logical_operators(lx, lz)
    prod = (px.astype(np.uint32) @ pz.T.astype(np.uint32)) % 2
    assert (prod == np.eye(k, dtype=np.uint32)).all()


def test_pair_logical_operators_degenerate_raises():
    lx = [np.array([1, 0], dtype=np.uint8)]
    lz = [np.array([0, 1], dtype=np.uint8)]  # dot product is 0 mod 2 -> cannot pair
    with pytest.raises(ValueError):
        gf2.pair_logical_operators(lx, lz)
