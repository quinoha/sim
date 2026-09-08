"""
Unit tests for QEDA EvaluationPlan Loader (load_evaluation_plan).
"""
import pytest
from pathlib import Path
from m0_config import (
    load_evaluation_plan,
    build_evaluation_plan,
    EvaluationPlan
)


def test_load_plan_from_json_file(tmp_path: Path):
    # 1. Build and save a sample EvaluationPlan JSON
    yaml_path = "configs/experiment.yaml"
    original_plan = build_evaluation_plan(yaml_path)
    
    json_file = tmp_path / "test_plan.json"
    json_file.write_text(original_plan.model_dump_json(indent=2), encoding="utf-8")

    # 2. Load back from JSON file
    loaded_plan = load_evaluation_plan(json_file)

    assert isinstance(loaded_plan, EvaluationPlan)
    assert loaded_plan.plan_id == original_plan.plan_id
    assert len(loaded_plan.run_specs) == 24
    assert len(loaded_plan.target_figure_ids) == 5
    assert loaded_plan.run_specs[0].run_spec_id == original_plan.run_specs[0].run_spec_id


def test_load_plan_from_yaml_file():
    yaml_path = "configs/experiment.yaml"
    loaded_plan = load_evaluation_plan(yaml_path)

    assert isinstance(loaded_plan, EvaluationPlan)
    assert loaded_plan.plan_id == "PLAN-bb-vs-surface-2026-08"
    assert len(loaded_plan.run_specs) == 24


def test_load_plan_from_dict():
    yaml_path = "configs/experiment.yaml"
    original_plan = build_evaluation_plan(yaml_path)
    data_dict = original_plan.model_dump()

    loaded_plan = load_evaluation_plan(data_dict)
    assert isinstance(loaded_plan, EvaluationPlan)
    assert loaded_plan.plan_id == original_plan.plan_id
    assert len(loaded_plan.run_specs) == 24


def test_load_plan_passthrough():
    yaml_path = "configs/experiment.yaml"
    original_plan = build_evaluation_plan(yaml_path)

    passthrough_plan = load_evaluation_plan(original_plan)
    assert passthrough_plan is original_plan


def test_load_plan_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_evaluation_plan("non_existent_plan.json")
