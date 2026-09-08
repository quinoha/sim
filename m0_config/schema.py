"""
QEDA M0 Configuration Schemas and Data Models.
Defines both Raw input schemas and Resolved immutable structures.
"""
from typing import Dict, List, Literal, Optional, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ExperimentMeta(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str = Field(description="Unique experiment suite identifier")
    description: Optional[str] = Field(default="", description="Human-readable description")


class NoiseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    model: str = Field(default="uniform_depolarizing", description="Noise model type")
    p: float = Field(default=0.001, ge=0.0, le=1.0, description="Physical error rate")


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    mode: Literal["circuit_level", "code_capacity", "hybrid"] = Field(
        default="circuit_level", description="Execution mode"
    )
    rounds: int = Field(default=5, ge=1, description="Default syndrome extraction rounds")
    shots: int = Field(default=200000, ge=1, description="Default number of shots")
    seed: int = Field(default=12345, description="Default random seed")
    noise: NoiseConfig = Field(default_factory=NoiseConfig)


class CodeSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str = Field(description="Code instance identifier")
    family: Literal["surface", "bb", "color", "custom"] = Field(description="Code family")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom parameters, e.g. distance")
    preset: Optional[str] = Field(default=None, description="Preset identifier (e.g. d6, d10 for BB)")


class DecoderSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str = Field(description="Decoder instance identifier")
    kind: Literal["mwpm", "bposd", "astra_gnn", "rtl_sim", "custom"] = Field(description="Decoder algorithm/type")
    options: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Decoder hyperparameters")


class StoppingRule(BaseModel):
    model_config = ConfigDict(frozen=True)
    max_shots: int = Field(default=200000, ge=1)
    min_failures: int = Field(default=100, ge=1)


class CornerCaseConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    enable: bool = Field(default=True, description="Enable directed corner case injection")
    types: List[str] = Field(
        default_factory=lambda: ["all_zero", "full_logical_error", "near_miss"],
        description="Corner case types to evaluate"
    )


class AccuracyProtocol(BaseModel):
    model_config = ConfigDict(frozen=True)
    stopping_rule: StoppingRule = Field(default_factory=StoppingRule)
    corner_cases: Optional[CornerCaseConfig] = Field(default_factory=CornerCaseConfig)


class WarmupSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: float = Field(default=0.0, ge=0.0)
    unit: str = Field(default="ms", description="Time unit: ns, us, ms, s")
    normalized_ns: Optional[float] = Field(default=None, description="Normalized time in nanoseconds")


class TimingProtocol(BaseModel):
    model_config = ConfigDict(frozen=True)
    clock: str = Field(default="monotonic_ns")
    percentiles: List[int] = Field(default_factory=lambda: [50, 99])
    warmup: Optional[WarmupSpec] = Field(default_factory=WarmupSpec)


class ProtocolConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    accuracy: AccuracyProtocol = Field(default_factory=AccuracyProtocol)
    timing: Optional[TimingProtocol] = Field(default_factory=TimingProtocol)


class ConstraintSpec(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str = Field(description="Constraint identifier (e.g. C-LER, C-LAT)")
    metric: str = Field(description="Target metric name (e.g. logical_error_rate, latency_p99)")
    op: Literal["<=", "<", ">=", ">", "=="] = Field(description="Comparison operator")
    value: float = Field(description="Target threshold value")
    unit: Optional[str] = Field(default=None, description="Unit (e.g. us, ns, %)")
    normalized_value: Optional[float] = Field(default=None, description="Unit-normalized value")


class QualificationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    constraints: List[ConstraintSpec] = Field(default_factory=list)


class OutputConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    root: str = Field(default="./datasets", description="Root output directory")
    format: str = Field(default="dets_b8", description="Primary dataset format")
    formats: List[str] = Field(
        default_factory=lambda: ["dets_b8", "obs_b8", "meta_json"],
        description="List of export formats"
    )


class DenyRule(BaseModel):
    model_config = ConfigDict(frozen=True)
    code: str = Field(description="Code ID to deny")
    decoder: str = Field(description="Decoder ID to deny")
    reason: Optional[str] = Field(default="INCOMPATIBLE", description="Reason for incompatibility")


class CompatibilityConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    deny: List[DenyRule] = Field(default_factory=list)


class RawExperimentConfig(BaseModel):
    """Raw parsed configuration loaded directly from YAML/JSON."""
    model_config = ConfigDict(extra="ignore")
    
    config_version: int = Field(default=1, description="Config schema version")
    experiment: ExperimentMeta
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    codes: List[CodeSpec] = Field(min_length=1, description="Target QEC codes")
    decoders: List[DecoderSpec] = Field(min_length=1, description="Target decoders")
    sweep: Optional[Dict[str, List[Any]]] = Field(default_factory=dict, description="Parameter sweep axes")
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    qualification: Optional[QualificationConfig] = Field(default_factory=QualificationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    compatibility: Optional[CompatibilityConfig] = Field(default_factory=CompatibilityConfig)
