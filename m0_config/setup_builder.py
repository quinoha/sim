"""
QEDA SetupBuilder (Stage 1 Core Engine).
Translates resolved configuration contracts and target figure requirements into deduplicated EvaluationPlans.
"""
import hashlib
import itertools
from pathlib import Path
from typing import Union, Dict, Any, List, Optional, Tuple, Set

from .config_loader import load_config
from .resolved_config import ResolvedConfig
from .schema import CodeSpec, DecoderSpec, DenyRule
from .plan_schema import RunSpec, CompatibilityResult, FigureRunBinding, EvaluationPlan


# Normative Figure Catalog mapping to required canonical field categories
DEFAULT_FIGURE_FIELD_MAP: Dict[str, List[str]] = {
    "C-F01": ["workload.family", "noise.p", "accuracy.ler", "accuracy.ler_ci_95", "runtime.quality"],
    "C-F02": ["timing.service_time_ns", "timing.latency_p50_ns", "timing.latency_p99_ns", "traffic.scenario"],
    "C-F03": ["throughput.samples_per_sec", "resource.word_width_bits", "timing.clock_freq_mhz"],
    "C-F04": ["accuracy.threshold_p", "accuracy.suppression_factor", "workload.distance"],
    "C-F05": ["accuracy.ler", "timing.latency_p99_ns", "analysis.pareto_optimal"],
    "C-F07": ["corner.case_type", "corner.observable_flip", "corner.detector_count", "quality.status"],
    "C-F08": ["corner.best_case_ler", "corner.worst_case_ler", "analysis.sensitivity_penalty"],
    "C-F10": ["workload.rounds", "noise.p", "accuracy.ler", "timing.latency_p99_ns"]
}


class SetupBuilder:
    """
    Core Stage 1 Builder.
    Evaluates compatibility, expands orthogonal sweep grids, reverses figure field requirements,
    and seals the resulting immutable EvaluationPlan.
    """

    def __init__(self, catalog_versions: Optional[Dict[str, str]] = None):
        self.catalog_versions = catalog_versions or {
            "field_catalog": "1.0",
            "figure_catalog": "1.0",
            "figure_field_matrix": "1.0"
        }

    @classmethod
    def build_plan(
        cls,
        source: Union[str, Path, Dict[str, Any], ResolvedConfig],
        target_figures: Optional[List[str]] = None
    ) -> EvaluationPlan:
        """Convenience factory method to build an EvaluationPlan."""
        builder = cls()
        return builder.build(source, target_figures)

    def build(
        self,
        source: Union[str, Path, Dict[str, Any], ResolvedConfig],
        target_figures: Optional[List[str]] = None
    ) -> EvaluationPlan:
        """
        Main execution pipeline for SetupBuilder:
        1. Resolve and validate configuration
        2. Evaluate Code x Decoder compatibility matrix
        3. Reverse required fields from target figures
        4. Expand sweep axes into deduplicated RunSpecs
        5. Bind RunSpecs to target Figures
        6. Compute deterministic SHA-256 seal
        7. Return immutable EvaluationPlan
        """
        # 1. Ensure ResolvedConfig
        if isinstance(source, ResolvedConfig):
            resolved_config = source
        else:
            resolved_config = load_config(source)

        # 2. Determine target figures and required fields
        if target_figures:
            active_figure_ids = sorted(list(set(target_figures)))
        else:
            # Default to core cross-decoder figures
            active_figure_ids = ["C-F01", "C-F02", "C-F05", "C-F07", "C-F10"]

        required_field_ids_set: Set[str] = set()
        for fig_id in active_figure_ids:
            fields = DEFAULT_FIGURE_FIELD_MAP.get(fig_id, ["accuracy.ler", "timing.latency_p99_ns"])
            required_field_ids_set.update(fields)
        required_field_ids = sorted(list(required_field_ids_set))

        # 3. Evaluate Compatibility Matrix
        compatibility_results = self._evaluate_compatibility(resolved_config)
        denied_pairs = {
            (c.code_id, c.decoder_id)
            for c in compatibility_results
            if c.status == "DENIED"
        }

        # 4. Expand Sweep Axes into RunSpecs
        run_specs, figure_bindings = self._expand_run_specs(
            resolved_config,
            denied_pairs,
            active_figure_ids,
            required_field_ids
        )

        # 5. Assemble plan dictionary for hashing
        plan_dict = {
            "plan_id": f"PLAN-{resolved_config.experiment.id}",
            "catalog_versions": self.catalog_versions,
            "analysis_views": ["ACCURACY", "TIMING", "RESOURCE"],
            "target_view_matrix": [
                {"code_id": c.id, "decoder_id": d.id}
                for c in resolved_config.codes
                for d in resolved_config.decoders
                if (c.id, d.id) not in denied_pairs
            ],
            "run_specs": [rs.model_dump() for rs in run_specs],
            "target_figure_ids": active_figure_ids,
            "required_field_ids": required_field_ids,
            "compatibility_results": [cr.model_dump() for cr in compatibility_results],
            "figure_run_bindings": [fb.model_dump() for fb in figure_bindings],
            "qualification_policy_ref": (
                resolved_config.qualification.model_dump()
                if resolved_config.qualification
                else None
            ),
            "definition_snapshot": resolved_config.model_dump(exclude={"source_path"}),
            "source_path": resolved_config.source_path
        }

        # 6. Construct and return frozen EvaluationPlan
        return EvaluationPlan.model_validate(plan_dict)

    def _evaluate_compatibility(self, config: ResolvedConfig) -> List[CompatibilityResult]:
        """Evaluate explicit deny rules and inherent architectural compatibility."""
        results: List[CompatibilityResult] = []
        
        # Build lookup for declared deny rules
        declared_denies = {}
        if config.compatibility and config.compatibility.deny:
            for rule in config.compatibility.deny:
                declared_denies[(rule.code, rule.decoder)] = rule.reason or "DECLARED_INCOMPATIBLE"

        for code in config.codes:
            for decoder in config.decoders:
                pair = (code.id, decoder.id)
                
                # Check 1: Declared in config deny list
                if pair in declared_denies:
                    results.append(CompatibilityResult(
                        code_id=code.id,
                        decoder_id=decoder.id,
                        status="DENIED",
                        reason=declared_denies[pair]
                    ))
                # Check 2: Architectural check - Non-graphlike BB codes cannot use MWPM
                elif code.family == "bb" and decoder.kind == "mwpm":
                    results.append(CompatibilityResult(
                        code_id=code.id,
                        decoder_id=decoder.id,
                        status="DENIED",
                        reason="NON_GRAPHLIKE_WEIGHT6_CHECK"
                    ))
                else:
                    results.append(CompatibilityResult(
                        code_id=code.id,
                        decoder_id=decoder.id,
                        status="COMPATIBLE",
                        reason=None
                    ))

        return results

    def _expand_run_specs(
        self,
        config: ResolvedConfig,
        denied_pairs: Set[Tuple[str, str]],
        active_figure_ids: List[str],
        required_field_ids: List[str]
    ) -> Tuple[List[RunSpec], List[FigureRunBinding]]:
        """Cartesian expansion of sweep axes and generation of atomic RunSpecs."""
        # Extract sweep axes (fallback to defaults if not in sweep)
        sweep = config.sweep or {}
        
        noise_p_list = sweep.get("noise.p", [config.defaults.noise.p])
        rounds_list = sweep.get("rounds", [config.defaults.rounds])
        shots_list = sweep.get("shots", [config.defaults.shots])
        mode_list = sweep.get("mode", [config.defaults.mode])
        
        run_specs: List[RunSpec] = []
        figure_to_run_ids: Dict[str, List[str]] = {fig_id: [] for fig_id in active_figure_ids}
        
        # Base seed for deterministic derivation
        base_seed = config.defaults.seed

        for code in config.codes:
            for decoder in config.decoders:
                if (code.id, decoder.id) in denied_pairs:
                    continue  # Skip denied combination

                # Cartesian product across sweep axes
                for p_val, r_val, s_val, mode_val in itertools.product(
                    noise_p_list, rounds_list, shots_list, mode_list
                ):
                    # Deterministic RunSpec ID
                    p_float = float(p_val)
                    p_str = f"{p_float:.6f}".rstrip('0').rstrip('.')
                    if not p_str or p_str == "0":
                        p_str = "0.0"
                    rs_id = (
                        f"RS-{config.experiment.id}-{code.id.upper()}-{decoder.id.upper()}-"
                        f"P{p_str}-R{r_val}-S{s_val}"
                    )
                    
                    # Deterministic distinct seed per RunSpec point
                    seed_hash = int(hashlib.sha256(rs_id.encode('utf-8')).hexdigest()[:8], 16)
                    run_seed = (base_seed + seed_hash) % (2**31 - 1)

                    # Extract stopping rule if present
                    stopping_rule = None
                    if config.protocol and config.protocol.accuracy:
                        stopping_rule = config.protocol.accuracy.stopping_rule

                    # Extract scenario
                    scenario = "ONLINE"
                    if "scenario" in sweep:
                        scenario = sweep["scenario"][0]

                    run_spec = RunSpec(
                        run_spec_id=rs_id,
                        evaluation_id=config.experiment.id,
                        code_id=code.id,
                        code_family=code.family,
                        code_params=dict(code.params) if code.params else {"preset": code.preset},
                        decoder_id=decoder.id,
                        decoder_kind=decoder.kind,
                        decoder_options=dict(decoder.options) if decoder.options else {},
                        noise_model=config.defaults.noise.model,
                        noise_p=float(p_val),
                        rounds=int(r_val),
                        shots=int(s_val),
                        seed=run_seed,
                        scenario=scenario,
                        stopping_rule=stopping_rule,
                        target_figure_ids=active_figure_ids,
                        required_field_ids=required_field_ids
                    )
                    
                    run_specs.append(run_spec)
                    
                    # Bind to figures
                    for fig_id in active_figure_ids:
                        figure_to_run_ids[fig_id].append(rs_id)

        # Build figure bindings
        figure_bindings = [
            FigureRunBinding(
                figure_id=fig_id,
                run_spec_ids=run_ids,
                required_field_ids=DEFAULT_FIGURE_FIELD_MAP.get(fig_id, [])
            )
            for fig_id, run_ids in figure_to_run_ids.items()
        ]

        return run_specs, figure_bindings
