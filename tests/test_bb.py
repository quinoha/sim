import numpy as np
import pytest

from qec_dataset import gf2
from qec_dataset.circuits import generic_css_memory_circuit
from qec_dataset.codes.bb import build_bb_code, parse_terms
from qec_dataset.extract import parity_checks_from_memory_circuit
from qec_dataset.noise import NoiseModel

# l=m=6, A=x^3+y+y^2, B=y^3+x+x^2 -> the [[72, 12]] bivariate-bicycle code
# (used here purely as a size/shape fixture, not asserted against the
# literature's exact distance value; see docs/rtl_test_vector_roadmap.md).
L, M = 6, 6
A_TERMS = [("x", 3), ("y", 1), ("y", 2)]
B_TERMS = [("y", 3), ("x", 1), ("x", 2)]


def test_parse_terms():
    assert parse_terms("x3,y1,y2") == [("x", 3), ("y", 1), ("y", 2)]
    with pytest.raises(ValueError):
        parse_terms("z3")


def test_bb_presets_build_with_expected_nk():
    from qec_dataset.codes.bb import BB_PRESETS, build_bb_preset

    # the two largest presets are slow to build; n/k check on the rest is enough
    for d in sorted(BB_PRESETS):
        if BB_PRESETS[d].n > 300:
            continue
        preset = BB_PRESETS[d]
        code = build_bb_preset(d)
        assert (code.n, code.k) == (preset.n, preset.k), d
        assert code.n == 2 * preset.l * preset.m


def test_bb_preset_unknown_distance_raises():
    from qec_dataset.codes.bb import build_bb_preset

    with pytest.raises(ValueError):
        build_bb_preset(7)


def test_bb_code_shape_and_commutation():
    code = build_bb_code(L, M, A_TERMS, B_TERMS)
    assert code.n == 2 * L * M
    assert code.k == 12

    commutator = (code.Hx.astype(np.uint32) @ code.Hz.T.astype(np.uint32)) % 2
    assert not commutator.any()

    # k should match an independent GF(2) rank computation, not just the
    # CSSCode property under test.
    independent_k = code.n - gf2.rank_mod2(code.Hx) - gf2.rank_mod2(code.Hz)
    assert independent_k == code.k


def test_bb_logical_operators_are_symplectic_pairs():
    code = build_bb_code(L, M, A_TERMS, B_TERMS)
    lx, lz = code.logical_x, code.logical_z
    assert lx.shape == (code.k, code.n)
    assert lz.shape == (code.k, code.n)

    product = (lx.astype(np.uint32) @ lz.T.astype(np.uint32)) % 2
    assert (product == np.eye(code.k, dtype=np.uint32)).all()


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_noiseless_circuit_is_deterministic(basis):
    code = build_bb_code(L, M, A_TERMS, B_TERMS)
    circuit = generic_css_memory_circuit(code, rounds=3, basis=basis, noise=NoiseModel.uniform(0.0))
    dets, obs = circuit.compile_detector_sampler().sample(200, separate_observables=True)
    assert not dets.any()
    assert not obs.any()


def test_noisy_circuit_flags_detectors():
    code = build_bb_code(L, M, A_TERMS, B_TERMS)
    circuit = generic_css_memory_circuit(code, rounds=3, basis="Z", noise=NoiseModel.uniform(0.02))
    dets, obs = circuit.compile_detector_sampler().sample(300, separate_observables=True)
    assert dets.any()


def test_extraction_matches_construction():
    code = build_bb_code(L, M, A_TERMS, B_TERMS)
    circuit = generic_css_memory_circuit(code, rounds=2, basis="Z", noise=NoiseModel.uniform(0.0))
    extracted, order = parity_checks_from_memory_circuit(circuit, name="extracted")
    assert order == list(range(code.n))
    assert extracted.n == code.n
    assert extracted.k == code.k
    assert np.array_equal(extracted.Hx, code.Hx)
    assert np.array_equal(extracted.Hz, code.Hz)
