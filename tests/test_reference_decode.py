from qec_dataset.circuits import generic_css_memory_circuit
from qec_dataset.codes.bb import build_bb_code
from qec_dataset.codes.surface import build_surface_code_circuit
from qec_dataset.noise import NoiseModel
from qec_dataset.reference_decode import reference_decode_stats


def test_pymatching_on_surface_code():
    circuit = build_surface_code_circuit(3, rounds=3, basis="Z", noise=NoiseModel.uniform(0.02))
    result = reference_decode_stats(circuit, shots=200, decoder="pymatching")
    assert result.shots == 200
    assert 0 <= result.errors <= result.shots


def test_pymatching_fails_on_non_graphlike_bb_code():
    code = build_bb_code(6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)])
    circuit = generic_css_memory_circuit(code, rounds=2, basis="Z", noise=NoiseModel.uniform(0.01))
    # MWPM is not applicable to BB codes' weight-6 (non-graphlike) checks;
    # this documents that the failure is expected, not silently wrong.
    try:
        reference_decode_stats(circuit, shots=20, decoder="pymatching")
        raised = False
    except Exception:
        raised = True
    assert raised


def test_bposd_fast_on_bb_code():
    code = build_bb_code(6, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)])
    circuit = generic_css_memory_circuit(code, rounds=2, basis="Z", noise=NoiseModel.uniform(0.02))
    result = reference_decode_stats(circuit, shots=10, decoder="bposd-fast")
    assert result.shots == 10
    assert 0 <= result.errors <= result.shots
