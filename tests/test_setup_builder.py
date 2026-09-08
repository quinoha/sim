"""
Unit tests for QEDA SetupBuilder and EvaluationPlan Generation (Stage 1 Core).
"""
import pytest
from m0_config import build_evaluation_plan, SetupBuilder, EvaluationPlan, RunSpec


def test_build_evaluation_plan_from_yaml():
    path = "configs/experiment.yaml"
    plan = build_evaluation_plan(path)

    assert isinstance(plan, EvaluationPlan)
    assert plan.plan_id == "PLAN-bb-vs-surface-2026-08"

    # Check RunSpec count:
    # 2 codes x 2 decoders x 4 p-values x 2 rounds = 32 combinations
    # Minus 8 denied (bb_d6 x pymatching) = 24 active RunSpecs
    assert len(plan.run_specs) == 24

    # Check compatibility results
    assert len(plan.compatibility_results) == 4
    denied = [c for c in plan.compatibility_results if c.status == "DENIED"]
    assert len(denied) == 1
    assert denied[0].code_id == "bb_d6"
    assert denied[0].decoder_id == "pymatching"


def test_run_specs_uniqueness_and_properties():
    path = "configs/experiment.yaml"
    plan = build_evaluation_plan(path)

    run_spec_ids = [rs.run_spec_id for rs in plan.run_specs]
    assert len(run_spec_ids) == len(set(run_spec_ids)), "All run_spec_ids must be unique"

    seeds = [rs.seed for rs in plan.run_specs]
    assert len(seeds) == len(set(seeds)), "All seeds must be uniquely assigned"

    # Verify no denied combination made it into run_specs
    for rs in plan.run_specs:
        assert not (rs.code_id == "bb_d6" and rs.decoder_id == "pymatching")
        assert rs.shots == 200000
        assert rs.noise_p in [0.001, 0.002, 0.003, 0.005]
        assert rs.rounds in [3, 5]


def test_figure_bindings_and_lookup():
    path = "configs/experiment.yaml"
    plan = build_evaluation_plan(path, target_figures=["C-F01", "C-F02"])

    assert plan.target_figure_ids == ["C-F01", "C-F02"]
    assert len(plan.figure_run_bindings) == 2

    # Lookup specs for C-F01
    cf01_specs = plan.get_specs_for_figure("C-F01")
    assert len(cf01_specs) == 24

    # Lookup individual RunSpec
    sample_id = plan.run_specs[0].run_spec_id
    found_spec = plan.get_run_spec(sample_id)
    assert found_spec is not None
    assert found_spec.run_spec_id == sample_id


def test_summary_and_string_representation():
    path = "configs/experiment.yaml"
    plan = build_evaluation_plan(path)

    summary_str = plan.summary()
    assert "PLAN-bb-vs-surface-2026-08" in summary_str
    assert "Total Active RunSpecs: 24" in summary_str
    assert "bb_d6 x pymatching" in summary_str
