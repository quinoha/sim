"""
QEDA EvaluationPlan Loader.
Loads, parses, and validates an EvaluationPlan from JSON files, YAML files, dicts, or existing objects.
"""
import json
from pathlib import Path
from typing import Union, Dict, Any

from .plan_schema import EvaluationPlan
from .setup_builder import SetupBuilder


def load_evaluation_plan(source: Union[str, Path, Dict[str, Any], EvaluationPlan]) -> EvaluationPlan:
    """
    Universal loader for EvaluationPlan.

    Args:
        source: A path to a .json or .yaml file, a dictionary, or an EvaluationPlan instance.

    Returns:
        Immutable, fully-validated EvaluationPlan instance.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        ValueError: If the file format is unsupported or JSON schema validation fails.
    """
    # 1. Already an EvaluationPlan instance
    if isinstance(source, EvaluationPlan):
        return source

    # 2. Raw dictionary
    if isinstance(source, dict):
        return EvaluationPlan.model_validate(source)

    # 3. File path (string or Path)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"EvaluationPlan source file not found: '{path.resolve()}'")

    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        # Compile YAML into EvaluationPlan on the fly
        return SetupBuilder.build_plan(path)

    elif suffix == ".json":
        # Load serialized EvaluationPlan JSON
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return EvaluationPlan.model_validate(data)

    else:
        # Attempt JSON parsing as fallback
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EvaluationPlan.model_validate(data)
        except Exception as e:
            raise ValueError(
                f"Unsupported file format '{suffix}' for EvaluationPlan at '{path}'. Expected .json or .yaml."
            ) from e
