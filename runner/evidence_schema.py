"""
QEDA Stage 2 (EvaluationRunner) Output Schema.
Defines immutable evidence contracts (RunEvidence and EvidenceBundle) storing
raw physical observations, error counts, binary artifact paths, and timing facts.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field


class RunEvidence(BaseModel):
    """
    Evidence collected for a single RunSpec execution.
    Contains raw counts, binary artifact locations, and timing measurements.
    """
    model_config = ConfigDict(frozen=True)

    run_spec_id: str
    code_id: str
    decoder_id: str
    noise_p: float
    rounds: int
    shots: int
    errors: int
    logical_error_rate: float
    sampling_time_ns: int = 0
    decode_time_ns: int = 0
    elapsed_time_ns: int = 0
    avg_latency_per_shot_ns: float
    dets_path: str
    obs_path: str
    manifest_path: str
    status: str = "SUCCESS"
    error_message: Optional[str] = None

    # Why sampling stopped: "min_failures" once the declared failure count was
    # reached, "max_shots" when the budget ran out first, "fixed" when a caller
    # asked for an exact shot count. Without it a low `errors` reading is
    # ambiguous -- a genuinely good decoder and an under-powered run look alike.
    stopped_by: str = "fixed"
    # Shots this RunSpec's decoder actually consumed, against the shared pool it
    # was drawn from. Equal to `shots` unless the stopping rule ended it early.
    workload_id: Optional[str] = None


class EvidenceBundle(BaseModel):
    """
    Immutable bundle of evidence collected for all active RunSpecs in an EvaluationPlan.
    Passed downstream to Stage 3 (AnalysisEngine).
    """
    model_config = ConfigDict(frozen=True)

    bundle_id: str
    plan_id: str
    total_run_specs: int
    successful_runs: int
    failed_runs: int
    run_evidences: List[RunEvidence] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
