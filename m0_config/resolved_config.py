"""
Resolved and Sealed Configuration Contract for QEDA M0.
This immutable object is the authoritative input for M1 (Experiment Planner) and M3 (Artifact Builder).
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from .schema import (
    ExperimentMeta, DefaultsConfig, CodeSpec, DecoderSpec,
    ProtocolConfig, QualificationConfig, OutputConfig, CompatibilityConfig
)


class ResolvedConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    config_version: int
    source_path: Optional[str] = Field(default=None, description="Original YAML/JSON file path")

    experiment: ExperimentMeta
    defaults: DefaultsConfig
    codes: List[CodeSpec]
    decoders: List[DecoderSpec]
    sweep: Dict[str, List[Any]]
    protocol: ProtocolConfig
    qualification: QualificationConfig
    output: OutputConfig
    compatibility: CompatibilityConfig

    def summary(self) -> str:
        """Return a human-readable summary of the resolved experiment contract."""
        return (
            f"=== Experiment Contract [{self.experiment.id}] ===\n"
            f"Codes: {[c.id for c in self.codes]}\n"
            f"Decoders: {[d.id for d in self.decoders]}\n"
            f"Sweep Axes: {list(self.sweep.keys())}\n"
            f"Deny Rules: {len(self.compatibility.deny)} active\n"
            f"Constraints: {len(self.qualification.constraints)} gates\n"
        )
