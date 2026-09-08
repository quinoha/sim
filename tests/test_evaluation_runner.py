"""
Unit tests for QEDA Stage 2 EvaluationRunner (runner.evaluation_runner).
"""
import pytest
from pathlib import Path

from m0_config.plan_schema import RunSpec, EvaluationPlan
from runner import EvaluationRunner, RunEvidence, EvidenceBundle


def test_runner_single_surface_spec(tmp_path: Path):
    runner = EvaluationRunner(output_root=tmp_path / "evidence")

    spec = RunSpec(
        run_spec_id="RS-TEST-SURFACE-SINGLE",
        evaluation_id="test-eval",
        code_id="surface_d3",
        code_family="surface",
        code_params={"distance": 3},
        decoder_id="pymatching",
        decoder_kind="mwpm",
        noise_model="uniform_depolarizing",
        noise_p=0.005,
        rounds=3,
        shots=100,
        seed=42,
        scenario="ONLINE",
        target_figure_ids=["C-F01"],
        required_field_ids=["accuracy.ler"],
    )

    evidence = runner.run_spec(spec, shots_override=50, num_workers=1)

    assert isinstance(evidence, RunEvidence)
    assert evidence.status == "SUCCESS"
    assert evidence.shots == 50
    assert 0.0 <= evidence.logical_error_rate <= 1.0
    assert evidence.elapsed_time_ns > 0
    assert Path(evidence.dets_path).exists()
    assert Path(evidence.obs_path).exists()
    assert Path(evidence.manifest_path).exists()


def test_runner_custom_decoder_registration(tmp_path: Path):
    runner = EvaluationRunner(output_root=tmp_path / "evidence")

    import numpy as np

    # Mock custom decoder: takes syndrome matrix, returns all-zero predictions
    def mock_custom_decoder(dets):
        return np.zeros((dets.shape[0], 1), dtype=np.uint8)

    runner.register_decoder("mock_gnn", mock_custom_decoder)

    spec = RunSpec(
        run_spec_id="RS-TEST-CUSTOM-DEC",
        evaluation_id="test-eval",
        code_id="surface_d3",
        code_family="surface",
        code_params={"distance": 3},
        decoder_id="mock_gnn",
        decoder_kind="custom",
        noise_model="uniform_depolarizing",
        noise_p=0.001,
        rounds=3,
        shots=30,
        seed=99,
        scenario="ONLINE",
        target_figure_ids=["C-F01"],
        required_field_ids=["accuracy.ler"],
    )

    evidence = runner.run_spec(spec, shots_override=30, num_workers=1)

    assert evidence.decoder_id == "mock_gnn"
    assert evidence.errors >= 0
    assert 0.0 <= evidence.logical_error_rate <= 1.0
    assert evidence.decode_time_ns > 0


def test_runner_run_mini_plan(tmp_path: Path):
    runner = EvaluationRunner(output_root=tmp_path / "evidence")

    spec1 = RunSpec(
        run_spec_id="RS-MINI-1",
        evaluation_id="test-eval",
        code_id="surface_d3",
        code_family="surface",
        code_params={"distance": 3},
        decoder_id="pymatching",
        decoder_kind="mwpm",
        noise_model="uniform_depolarizing",
        noise_p=0.002,
        rounds=3,
        shots=40,
        seed=1,
        scenario="ONLINE",
        target_figure_ids=["C-F01"],
        required_field_ids=["accuracy.ler"],
    )

    plan = EvaluationPlan(
        plan_id="PLAN-MINI-TEST",
        definition_snapshot={"experiment": {"id": "test"}},
        target_figure_ids=["C-F01"],
        required_field_ids=["accuracy.ler"],
        run_specs=[spec1],
    )

    bundle = runner.run_plan(plan, quick=True, quick_shots=20, num_workers=1)

    assert isinstance(bundle, EvidenceBundle)
    assert bundle.plan_id == "PLAN-MINI-TEST"
    assert bundle.total_run_specs == 1
    assert bundle.successful_runs == 1
    assert len(bundle.run_evidences) == 1
    assert bundle.run_evidences[0].shots == 20

    bundle_json_file = tmp_path / "evidence" / "evidence_bundle.json"
    assert bundle_json_file.exists()
