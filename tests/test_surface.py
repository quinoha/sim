import pytest

from qec_dataset.codes.surface import build_surface_code_circuit, rotated_surface_code
from qec_dataset.distance import shortest_graphlike_distance
from qec_dataset.noise import NoiseModel


@pytest.mark.parametrize("d", [3, 5])
def test_rotated_surface_code_nk(d):
    code, order = rotated_surface_code(d)
    assert code.n == d * d
    assert code.k == 1
    # stim.Circuit.generated assigns its own physical qubit indices (not
    # necessarily 0..n-1, unlike our own generic circuit builder) — just
    # check `order` names exactly n distinct qubits, in ascending order.
    assert len(order) == d * d
    assert order == sorted(order)
    assert len(set(order)) == d * d


@pytest.mark.parametrize("d", [3, 5])
def test_noiseless_circuit_is_deterministic(d):
    circuit = build_surface_code_circuit(d, rounds=d, noise=NoiseModel.uniform(0.0))
    dets, obs = circuit.compile_detector_sampler().sample(200, separate_observables=True)
    assert not dets.any()
    assert not obs.any()


@pytest.mark.parametrize("d", [3, 5])
def test_noisy_circuit_flags_detectors(d):
    circuit = build_surface_code_circuit(d, rounds=d, noise=NoiseModel.uniform(0.05))
    dets, obs = circuit.compile_detector_sampler().sample(500, separate_observables=True)
    assert dets.any()


@pytest.mark.parametrize("d", [3, 5])
def test_graphlike_distance_matches_parameter(d):
    circuit = build_surface_code_circuit(d, rounds=d, noise=NoiseModel.uniform(1e-3))
    result = shortest_graphlike_distance(circuit)
    assert result.distance == d
