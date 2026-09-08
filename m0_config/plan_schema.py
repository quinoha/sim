"""
QEDA SetupBuilder EvaluationPlan and RunSpec Schemas.
Defines immutable data models for Stage 1 (SETUP) execution contracts according to QEDA Baseline 1.0.
"""
from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from .schema import StoppingRule, ConstraintSpec


class RunSpec(BaseModel):
    """
    Atomic execution specification for EvaluationRunner.
    Completely binds a single (Code x Decoder x Noise x Rounds x Scenario) point.
    """
    model_config = ConfigDict(frozen=True)

    run_spec_id: str = Field(description="Unique deterministic identifier for this run spec")
    evaluation_id: str = Field(description="Parent experiment / evaluation identifier")
    
    # Code workload specification
    code_id: str = Field(description="Code identifier (e.g. surface_d5, bb_d6)")
    code_family: Literal["surface", "bb", "color", "custom"] = Field(description="Code family")
    code_params: Dict[str, Any] = Field(default_factory=dict, description="Code parameters (distance, preset, etc)")
    
    # Decoder candidate specification
    decoder_id: str = Field(description="Decoder identifier (e.g. pymatching, bposd_fast)")
    decoder_kind: Literal["mwpm", "bposd", "astra_gnn", "rtl_sim", "custom"] = Field(description="Decoder algorithm/engine")
    decoder_options: Dict[str, Any] = Field(default_factory=dict, description="Hyperparameters for decoder")
    
    # Noise & physical error specification
    noise_model: str = Field(default="uniform_depolarizing", description="Noise model type")
    noise_p: float = Field(ge=0.0, le=1.0, description="Physical error rate")
    
    # Execution protocol & parameters
    rounds: int = Field(ge=1, description="Syndrome extraction rounds")
    shots: int = Field(ge=1, description="Target Monte-Carlo shots to sample")
    seed: int = Field(description="Deterministic random seed assigned to this run")
    scenario: Literal["ONLINE", "STREAM", "OFFLINE"] = Field(default="ONLINE", description="Execution scenario")
    
    # Measurement & stopping rules
    stopping_rule: Optional[StoppingRule] = Field(default=None, description="Early stopping criteria")
    
    # Data-to-Figure mapping
    target_figure_ids: List[str] = Field(default_factory=list, description="Figure IDs that consume this run spec")
    required_field_ids: List[str] = Field(default_factory=list, description="Canonical field IDs required by consumers")


class CompatibilityResult(BaseModel):
    """Result of compatibility and deny-rule evaluation for a (Code x Decoder) pair."""
    model_config = ConfigDict(frozen=True)
    code_id: str
    decoder_id: str
    status: Literal["COMPATIBLE", "DENIED", "UNSUPPORTED"]
    reason: Optional[str] = None


class FigureRunBinding(BaseModel):
    """Binding between a target Figure and the RunSpecs required to render it."""
    model_config = ConfigDict(frozen=True)
    figure_id: str
    run_spec_ids: List[str] = Field(default_factory=list)
    required_field_ids: List[str] = Field(default_factory=list)


class EvaluationPlan(BaseModel):
    """
    Immutable execution contract produced by SetupBuilder (Stage 1).
    Authoritative input for Stage 2 (EvaluationRunner).
    """
    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(description="Unique evaluation plan identifier")
    source_path: Optional[str] = Field(default=None, description="Original configuration file path")

    definition_snapshot: Dict[str, Any] = Field(description="Raw resolved configuration snapshot")
    catalog_versions: Dict[str, str] = Field(
        default_factory=lambda: {
            "field_catalog": "1.0",
            "figure_catalog": "1.0",
            "figure_field_matrix": "1.0"
        },
        description="Normative catalog version pins"
    )
    
    analysis_views: List[str] = Field(
        default_factory=lambda: ["ACCURACY", "TIMING", "RESOURCE"],
        description="Declared analysis views"
    )
    target_view_matrix: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Declared View x Candidate evaluation matrix"
    )
    
    # Core execution specification list
    run_specs: List[RunSpec] = Field(
        min_length=1,
        description="Deduplicated list of atomic run specifications"
    )
    
    target_figure_ids: List[str] = Field(
        default_factory=list,
        description="List of target Figure IDs to produce"
    )
    required_field_ids: List[str] = Field(
        default_factory=list,
        description="Deduplicated canonical field IDs required for target figures"
    )
    compatibility_results: List[CompatibilityResult] = Field(
        default_factory=list,
        description="Evaluation of compatibility and deny rules"
    )
    figure_run_bindings: List[FigureRunBinding] = Field(
        default_factory=list,
        description="Explicit mapping from Figure IDs to required RunSpec IDs"
    )
    qualification_policy_ref: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional qualification gate constraints"
    )

    def summary(self) -> str:
        """Return a structured human-readable summary of the evaluation plan."""
        lines = [
            f"=== EvaluationPlan [{self.plan_id}] ===",
            f"Target Figures ({len(self.target_figure_ids)}): {', '.join(self.target_figure_ids)}",
            f"Total Active RunSpecs: {len(self.run_specs)}",
            f"Compatibility Results: {len(self.compatibility_results)} pairs evaluated",
        ]
        
        # Breakdown by code x decoder
        code_counts = {}
        for rs in self.run_specs:
            key = f"{rs.code_id} x {rs.decoder_id}"
            code_counts[key] = code_counts.get(key, 0) + 1
        
        lines.append("  [RunSpec Breakdown by Workload]:")
        for pair, count in sorted(code_counts.items()):
            lines.append(f"    - {pair}: {count} points (sweep axes)")
            
        denied_pairs = [c for c in self.compatibility_results if c.status == "DENIED"]
        if denied_pairs:
            lines.append(f"  [Denied Combinations ({len(denied_pairs)})]:")
            for d in denied_pairs:
                lines.append(f"    - [DENIED] {d.code_id} x {d.decoder_id} (Reason: {d.reason})")
                
        return "\n".join(lines)

    def get_run_spec(self, run_spec_id: str) -> Optional[RunSpec]:
        """Lookup a RunSpec by its ID."""
        for rs in self.run_specs:
            if rs.run_spec_id == run_spec_id:
                return rs
        return None

    def get_specs_for_figure(self, figure_id: str) -> List[RunSpec]:
        """Return all RunSpecs bound to a given Figure ID."""
        for binding in self.figure_run_bindings:
            if binding.figure_id == figure_id:
                target_ids = set(binding.run_spec_ids)
                return [rs for rs in self.run_specs if rs.run_spec_id in target_ids]
        return []
