"""
Schema and Semantic Validator for QEDA M0.
Pure rule-based validator that does NOT mutate input objects, returning structured issues.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from .schema import RawExperimentConfig


class ValidationIssue(BaseModel):
    rule_id: str
    severity: str = Field(default="ERROR", description="ERROR | WARNING | INFO")
    path: str
    message: str


def validate_experiment_semantics(config: RawExperimentConfig) -> List[ValidationIssue]:
    """Perform deep semantic checks across configuration blocks."""
    issues: List[ValidationIssue] = []
    
    code_ids = {c.id for c in config.codes}
    decoder_ids = {d.id for d in config.decoders}
    
    # 1. Check duplicate Code IDs
    if len(code_ids) != len(config.codes):
        issues.append(ValidationIssue(
            rule_id="V-DUP-CODE-ID",
            severity="ERROR",
            path="codes",
            message="Duplicate code IDs found in 'codes' list."
        ))
        
    # 2. Check duplicate Decoder IDs
    if len(decoder_ids) != len(config.decoders):
        issues.append(ValidationIssue(
            rule_id="V-DUP-DEC-ID",
            severity="ERROR",
            path="decoders",
            message="Duplicate decoder IDs found in 'decoders' list."
        ))
        
    # 3. Validate Compatibility Deny Rules
    if config.compatibility and config.compatibility.deny:
        for idx, rule in enumerate(config.compatibility.deny):
            if rule.code not in code_ids:
                issues.append(ValidationIssue(
                    rule_id="V-UNKNOWN-DENY-CODE",
                    severity="WARNING",
                    path=f"compatibility.deny[{idx}].code",
                    message=f"Deny rule references unknown code ID '{rule.code}'."
                ))
            if rule.decoder not in decoder_ids:
                issues.append(ValidationIssue(
                    rule_id="V-UNKNOWN-DENY-DEC",
                    severity="WARNING",
                    path=f"compatibility.deny[{idx}].decoder",
                    message=f"Deny rule references unknown decoder ID '{rule.decoder}'."
                ))

    # 4. Check Code specific parameters
    for idx, c in enumerate(config.codes):
        if c.family == "surface":
            if not c.params or "distance" not in c.params:
                issues.append(ValidationIssue(
                    rule_id="V-SURFACE-NO-DISTANCE",
                    severity="ERROR",
                    path=f"codes[{idx}]",
                    message=f"Surface code '{c.id}' requires 'distance' parameter in params."
                ))
        elif c.family == "bb":
            if not c.preset and (not c.params or ("l" not in c.params or "m" not in c.params)):
                issues.append(ValidationIssue(
                    rule_id="V-BB-NO-PRESET-OR-PARAMS",
                    severity="ERROR",
                    path=f"codes[{idx}]",
                    message=f"BB code '{c.id}' requires either 'preset' (e.g. d6) or 'l', 'm', 'poly_a', 'poly_b' in params."
                ))

    # 5. Check Sweep parameter paths
    if config.sweep:
        allowed_sweep_prefixes = ["noise.p", "rounds", "shots", "noise.model", "mode"]
        for key in config.sweep:
            if key not in allowed_sweep_prefixes and not key.startswith("options."):
                issues.append(ValidationIssue(
                    rule_id="V-UNKNOWN-SWEEP-KEY",
                    severity="WARNING",
                    path=f"sweep.{key}",
                    message=f"Unrecognized sweep parameter '{key}'. Expected one of {allowed_sweep_prefixes} or options.*"
                ))

    return issues
