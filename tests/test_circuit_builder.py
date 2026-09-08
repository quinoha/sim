"""
Unit tests for QEDA Circuit Builder Bridge (qec_dataset.circuit_builder).
"""
import pytest
import stim

from m0_config import load_evaluation_plan
from m0_config.plan_schema import RunSpec
from qec_dataset import (
    build_circuit_from_run_spec,
    build_surface_circuit_from_run_spec,
    build_bb_circuit_from_run_spec,
    build_all_circuits_from_plan,
    clear_circuit_cache,
)


@pytest.fixture(autouse=True)
def clean_cache():
    clear_circuit_cache()
    yield
    clear_circuit_cache()


def test_build_surface_circuit_from_run_spec():
    # 1. Create a dummy RunSpec for surface_d5
    spec = RunSpec(
        run_spec_id="RS-TEST-SURFACE-D5",
        evaluation_id="test-eval",
        code_id="surface_d5",
        code_family="surface",
        code_params={"distance": 5},
        decoder_id="pymatching",
        decoder_kind="mwpm",
        noise_model="uniform_depolarizing",
        noise_p=0.001,
        rounds=3,
        shots=1000,
        seed=42,
        scenario="ONLINE",
        target_figure_ids=["C-F01"],
        required_field_ids=["accuracy.ler"],
    )

    # 2. Build stim.Circuit
    circuit = build_surface_circuit_from_run_spec(spec)

    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_detectors > 0
    assert circuit.num_observables == 1
    # For rotated surface code d=5, num_qubits = 2 * d^2 - 1 = 2 * 25 - 1 = 49 (or check > 0)
    assert circuit.num_qubits > 0


def test_build_bb_circuit_from_run_spec():
    # 1. Create a dummy RunSpec for bb_d6
    spec = RunSpec(
        run_spec_id="RS-TEST-BB-D6",
        evaluation_id="test-eval",
        code_id="bb_d6",
        code_family="bb",
        code_params={"preset": "d6"},
        decoder_id="bposd_fast",
        decoder_kind="bposd",
        noise_model="uniform_depolarizing",
        noise_p=0.002,
        rounds=3,
        shots=1000,
        seed=123,
        scenario="ONLINE",
        target_figure_ids=["C-F01"],
        required_field_ids=["accuracy.ler"],
    )

    # 2. Build stim.Circuit
    circuit = build_bb_circuit_from_run_spec(spec)

    assert isinstance(circuit, stim.Circuit)
    assert circuit.num_detectors > 0
    assert circuit.num_observables > 0
    # BB d6 has n=72 data qubits, plus ancillas
    assert circuit.num_qubits >= 72


def test_build_all_circuits_from_actual_plan():
    # Load actual 24-run evaluation plan
    plan = load_evaluation_plan("configs/experiment.yaml")
    assert len(plan.run_specs) == 24

    # Build all 24 circuits
    circuits = build_all_circuits_from_plan(plan)

    assert len(circuits) == 24
    for spec in plan.run_specs:
        assert spec.run_spec_id in circuits
        circ = circuits[spec.run_spec_id]
        assert isinstance(circ, stim.Circuit)
        assert circ.num_detectors > 0
        assert circ.num_observables > 0


def test_circuit_caching_behavior():
    spec1 = RunSpec(
        run_spec_id="RS-SURFACE-1",
        evaluation_id="test-eval",
        code_id="surface_d5",
        code_family="surface",
        code_params={"distance": 5},
        decoder_id="pymatching",
        decoder_kind="mwpm",
        noise_model="uniform_depolarizing",
        noise_p=0.001,
        rounds=3,
        shots=1000,
        seed=42,
        scenario="ONLINE",
        target_figure_ids=["C-F01"],
        required_field_ids=["accuracy.ler"],
    )
    spec2 = spec1.model_copy(update={"decoder_id": "bposd_fast", "decoder_kind": "bposd", "run_spec_id": "RS-SURFACE-2"})

    c1 = build_circuit_from_run_spec(spec1, use_cache=True)
    c2 = build_circuit_from_run_spec(spec2, use_cache=True)

    # They should have the same structure and detectors
    assert c1.num_detectors == c2.num_detectors
    assert c1.num_qubits == c2.num_qubits


def test_invalid_code_family_raises():
    invalid_spec = RunSpec(
        run_spec_id="RS-INVALID",
        evaluation_id="test-eval",
        code_id="unknown_code",
        code_family="color",
        code_params={},
        decoder_id="pymatching",
        decoder_kind="mwpm",
        noise_model="uniform_depolarizing",
        noise_p=0.001,
        rounds=3,
        shots=1000,
        seed=42,
        scenario="ONLINE",
        target_figure_ids=["C-F01"],
        required_field_ids=["accuracy.ler"],
    )

    with pytest.raises(ValueError, match="Unsupported code_family"):
        build_circuit_from_run_spec(invalid_spec)
