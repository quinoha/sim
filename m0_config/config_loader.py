"""
QEDA M0 ConfigLoader.
Loads, validates, normalizes, and seals experiment configuration files.
"""
import os
import yaml
import json
from typing import Union, Dict, Any, Tuple, Optional, List
from pathlib import Path

from .schema import RawExperimentConfig, ConstraintSpec, WarmupSpec
from .validator import validate_experiment_semantics, ValidationIssue
from .unit_normalizer import normalize_time_to_ns, normalize_constraint
from .resolved_config import ResolvedConfig


class ConfigValidationError(Exception):
    """Raised when configuration fails schema or semantic validation."""
    def __init__(self, issues: List[ValidationIssue]):
        self.issues = issues
        messages = [f"[{issue.severity}] {issue.path}: {issue.message} ({issue.rule_id})" for issue in issues]
        super().__init__("\n" + "\n".join(messages))


def load_raw_dict(source: Union[str, Path, Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Load dictionary from file path or return existing dict with source tracking."""
    if isinstance(source, dict):
        return source, None
    
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    if path.suffix.lower() in [".yaml", ".yml"]:
        data = yaml.safe_load(content) or {}
    elif path.suffix.lower() == ".json":
        data = json.loads(content) or {}
    else:
        # Try YAML parser as universal parser for JSON/YAML
        data = yaml.safe_load(content) or {}
        
    return data, str(path.resolve())


def load_config(source: Union[str, Path, Dict[str, Any]], strict: bool = True) -> ResolvedConfig:
    """
    Main entry point for M0 Config Resolver.
    
    1. Loads raw YAML/JSON
    2. Validates schema and semantic rules
    3. Standardizes units (UnitNormalizer)
    4. Computes deterministic canonical SHA-256 hash (CanonicalHasher)
    5. Returns immutable ResolvedConfig contract object
    """
    raw_dict, source_path = load_raw_dict(source)
    
    # 1. Pydantic validation (SchemaValidator + DefaultOverlay)
    try:
        raw_config = RawExperimentConfig.model_validate(raw_dict)
    except Exception as e:
        raise ValueError(f"M0 Schema Validation Failed:\n{e}") from e
    
    # 2. Semantic Cross-Validation
    issues = validate_experiment_semantics(raw_config)
    error_issues = [i for i in issues if i.severity == "ERROR"]
    if strict and error_issues:
        raise ConfigValidationError(error_issues)
    
    # 3. Unit Normalization
    raw_data = raw_config.model_dump()
    
    # Normalize warmup timing
    if "protocol" in raw_data and "timing" in raw_data["protocol"] and raw_data["protocol"]["timing"]:
        warmup = raw_data["protocol"]["timing"].get("warmup")
        if warmup and "value" in warmup:
            val, _ = normalize_time_to_ns(warmup["value"], warmup.get("unit", "ms"))
            raw_data["protocol"]["timing"]["warmup"]["normalized_ns"] = val
            
    # Normalize qualification constraints
    if "qualification" in raw_data and raw_data["qualification"]:
        for c in raw_data["qualification"].get("constraints", []):
            norm_val, _ = normalize_constraint(c["metric"], c["value"], c.get("unit"))
            c["normalized_value"] = norm_val
            
    # 4. Attach metadata
    raw_data["source_path"] = source_path
    
    # 5. Build immutable ResolvedConfig from normalized dictionary
    resolved = ResolvedConfig.model_validate(raw_data)
    return resolved
