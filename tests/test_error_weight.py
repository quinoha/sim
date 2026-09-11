"""Tests for the weight-resolved decoder profile.

The load-bearing claim in `error_weight` is that XOR-ing the incidence rows of
the fault mechanisms that fired reproduces exactly the syndrome stim would have
reported. If that is wrong, every number the profile prints is wrong in a way
that looks like a decoder result rather than a bug, so it is tested against
stim's own sampler rather than against a hand-computed expectation.
"""
from __future__ import annotations

import numpy as np
import pytest

stim = pytest.importorskip("stim")

from qec_dataset.codes.surface import build_surface_code_circuit  # noqa: E402
from qec_dataset.noise import NoiseModel  # noqa: E402
from qec_dataset.error_weight import (  # noqa: E402
    FaultIncidence,
    bucket_stats,
    fault_incidence,
    natural_sample,
    random_guess_ler,
    sample_weight_sets,
    symptoms,
    verify_incidence,
    wilson_interval,
)


def surface_circuit(distance: int = 3, rounds: int = 3, p: float = 0.01) -> "stim.Circuit":
    """Built through the repo's own builder, so the tests exercise the path the
    profile actually uses rather than a parallel stim invocation."""
    return build_surface_code_circuit(
        distance=distance, rounds=rounds, basis="Z", noise=NoiseModel.uniform(p)
    )


# --- the incidence matrices ------------------------------------------------


def test_incidence_matches_stim_syndromes():
    """The whole instrument rests on this."""
    circuit = surface_circuit()
    dem = circuit.detector_error_model(decompose_errors=False)
    inc = fault_incidence(dem)
    verify_incidence(dem, inc, shots=512, seed=7)  # raises on disagreement


def test_incidence_shapes_match_the_dem():
    circuit = surface_circuit()
    dem = circuit.detector_error_model(decompose_errors=False)
    inc = fault_incidence(dem)

    assert inc.H.shape == (inc.num_mechanisms, dem.num_detectors)
    assert inc.L.shape == (inc.num_mechanisms, dem.num_observables)
    assert inc.probs.shape == (inc.num_mechanisms,)
    assert np.all((inc.H == 0) | (inc.H == 1))
    assert np.all((inc.probs > 0) & (inc.probs < 1))


def test_decomposed_dem_is_rejected():
    """A decomposed error would make one physical fault count as several units
    of weight, which is silently wrong rather than loudly wrong."""
    circuit = surface_circuit()
    dem = circuit.detector_error_model(decompose_errors=True)
    if not any("^" in str(inst) for inst in dem.flattened() if inst.type == "error"):
        pytest.skip("this stim did not decompose anything for this circuit")
    with pytest.raises(ValueError, match="decomposed"):
        fault_incidence(dem)


# --- weight-set sampling ---------------------------------------------------


@pytest.mark.parametrize("weight", [0, 1, 2, 5, 9])
def test_weight_sets_are_distinct_and_in_range(weight):
    rng = np.random.default_rng(0)
    sel = sample_weight_sets(200, weight, 500, rng)

    assert sel.shape == (500, weight)
    assert sel.min(initial=0) >= 0
    assert sel.max(initial=0) < 200
    for row in sel:
        assert len(set(row.tolist())) == weight


def test_weight_beyond_mechanism_count_is_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        sample_weight_sets(4, 5, 10, rng)


def test_weight_sets_are_uniform_enough():
    """Every mechanism should turn up about equally often. A biased sampler
    would quietly make the profile a statement about whichever mechanisms it
    favours instead of about the weight."""
    rng = np.random.default_rng(1)
    m = 20
    sel = sample_weight_sets(m, 3, 20_000, rng)
    counts = np.bincount(sel.reshape(-1), minlength=m)
    expected = sel.size / m
    assert np.all(np.abs(counts - expected) < 0.15 * expected)


# --- symptom construction --------------------------------------------------


def test_symptoms_xor_the_selected_rows():
    inc = FaultIncidence(
        H=np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]], dtype=np.uint8),
        L=np.array([[1], [0], [1]], dtype=np.uint8),
        probs=np.array([0.1, 0.1, 0.1]),
        num_detectors=3,
        num_observables=1,
    )
    # mechanisms 0 and 1 share detector 0, which must cancel.
    dets, obs = symptoms(inc, np.array([[0, 1], [0, 2]], dtype=np.int64))

    assert dets.tolist() == [[0, 1, 0], [1, 0, 1]]
    assert obs.tolist() == [[1], [0]]


def test_weight_zero_produces_a_clean_syndrome():
    inc = FaultIncidence(
        H=np.ones((4, 3), dtype=np.uint8),
        L=np.ones((4, 2), dtype=np.uint8),
        probs=np.full(4, 0.1),
        num_detectors=3,
        num_observables=2,
    )
    dets, obs = symptoms(inc, np.zeros((7, 0), dtype=np.int64))

    assert dets.shape == (7, 3) and not dets.any()
    assert obs.shape == (7, 2) and not obs.any()


def test_symptoms_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(2)
    inc = FaultIncidence(
        H=rng.integers(0, 2, size=(50, 8), dtype=np.uint8),
        L=rng.integers(0, 2, size=(50, 2), dtype=np.uint8),
        probs=np.full(50, 0.01),
        num_detectors=8,
        num_observables=2,
    )
    sel = sample_weight_sets(50, 4, 100, rng)

    whole = symptoms(inc, sel, chunk=10_000)
    split = symptoms(inc, sel, chunk=7)
    assert np.array_equal(whole[0], split[0])
    assert np.array_equal(whole[1], split[1])


# --- natural sampling ------------------------------------------------------


def test_natural_sample_weights_track_the_syndromes():
    circuit = surface_circuit()
    dem = circuit.detector_error_model(decompose_errors=False)
    dets, obs, weights = natural_sample(dem, 2000, seed=3)

    assert dets.shape == (2000, dem.num_detectors)
    assert obs.shape == (2000, dem.num_observables)
    assert weights.shape == (2000,)
    assert weights.min() >= 0
    # A zero-weight shot cannot fire a detector, and a clean syndrome is
    # overwhelmingly the zero-weight case.
    assert not dets[weights == 0].any()


# --- statistics ------------------------------------------------------------


def test_bucket_stats_separates_collapse_from_guessing():
    truth = np.array([[0], [0], [1], [1]], dtype=np.uint8)

    collapsed = bucket_stats(3, np.zeros_like(truth), truth)
    assert collapsed.miss == 1.0
    assert collapsed.false_alarm == 0.0
    assert collapsed.pred_flip_rate == 0.0
    assert collapsed.ler == 0.5

    perfect = bucket_stats(3, truth.copy(), truth)
    assert perfect.ler == 0.0
    assert perfect.miss == 0.0 and perfect.false_alarm == 0.0

    inverted = bucket_stats(3, 1 - truth, truth)
    assert inverted.ler == 1.0
    assert inverted.miss == 1.0 and inverted.false_alarm == 1.0


def test_bucket_stats_rejects_a_mismatched_shape():
    truth = np.zeros((4, 12), dtype=np.uint8)
    with pytest.raises(ValueError, match="broadcast"):
        bucket_stats(2, np.zeros((4, 1), dtype=np.uint8), truth)


@pytest.mark.parametrize("successes,trials", [(0, 100), (100, 100), (50, 100), (1, 3)])
def test_wilson_interval_stays_in_range_and_brackets_the_estimate(successes, trials):
    low, high = wilson_interval(successes, trials)
    assert 0.0 <= low <= high <= 1.0
    assert low <= successes / trials <= high
    assert isinstance(low, float) and isinstance(high, float)  # json-serializable


def test_wilson_interval_of_an_empty_bucket_is_nan():
    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)


def test_random_guess_ceiling():
    assert random_guess_ler(1) == 0.5
    assert random_guess_ler(12) == pytest.approx(1 - 2**-12)


# --- the property the whole exercise is about ------------------------------


def test_low_weight_errors_are_correctable_by_matching():
    """A distance-3 code corrects every weight-1 fault. If this fails, either
    the syndromes are wrong or the decoder is not being driven correctly --
    and it would be indistinguishable from a decoder result in the profile."""
    pymatching = pytest.importorskip("pymatching")

    circuit = surface_circuit(distance=3, rounds=3)
    dem = circuit.detector_error_model(decompose_errors=False)
    inc = fault_incidence(dem)
    rng = np.random.default_rng(11)

    sel = sample_weight_sets(inc.num_mechanisms, 1, 1000, rng)
    dets, truth = symptoms(inc, sel)

    matching = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    stats = bucket_stats(1, matching.decode_batch(dets), truth)
    assert stats.errors == 0, f"MWPM failed on {stats.errors}/1000 weight-1 faults"
