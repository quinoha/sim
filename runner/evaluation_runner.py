"""
QEDA Stage 2 EvaluationRunner Engine.
Executes an EvaluationPlan following a strict, clean unidirectional pipeline:
1. Sample binary shots to disk once (.dets.b8, .obs.b8)
2. Load .dets.b8 into memory as ground-truth test vectors
3. Run pure decoder batch inference and measure PURE decoding latency
4. Score predictions against .obs.b8 to compute Logical Error Rate (LER)
5. Save manifest.json and compile EvidenceBundle
"""
from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np
import stim
import pymatching
import stimbposd

from m0_config.plan_schema import EvaluationPlan, RunSpec
from qec_dataset.circuit_builder import build_circuit_from_run_spec
from qec_dataset.sampling import sample_shots_to_files
from .evidence_schema import RunEvidence, EvidenceBundle


class EvaluationRunner:
    """
    Orchestrates execution of an EvaluationPlan with true decoupled decoding and pure latency profiling.
    """

    def __init__(self, output_root: str | Path = "outputs/evidence"):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._custom_decoders: Dict[str, Callable[[np.ndarray], np.ndarray]] = {}

    def register_decoder(self, decoder_id: str, decode_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Register a custom user decoder callable:
            fn(syndrome_matrix: np.ndarray) -> prediction_matrix: np.ndarray
        """
        self._custom_decoders[decoder_id] = decode_fn

    def run_spec(
        self,
        spec: RunSpec,
        shots_override: Optional[int] = None,
        num_workers: int = 4
    ) -> RunEvidence:
        """
        Executes a single RunSpec:
        1. Builds Stim quantum circuit.
        2. Samples shots to disk (.dets.b8, .obs.b8) ONCE.
        3. Loads saved files and executes pure decoding.
        4. Scores predictions and profiles pure latency.
        """
        shots = shots_override if shots_override is not None else spec.shots
        spec_dir = self.output_root / spec.run_spec_id
        spec_dir.mkdir(parents=True, exist_ok=True)

        # 1. Quantum circuit construction via Stage 1 -> Stim bridge
        circuit = build_circuit_from_run_spec(spec)

        # 2. Step 1: Sample binary shots to disk (.dets.b8 and .obs.b8) ONCE
        t_samp_start = time.perf_counter_ns()
        det_path, obs_path = sample_shots_to_files(
            circuit=circuit,
            shots=shots,
            out_dir=spec_dir,
            prefix="shots",
            det_format="b8",
            obs_format="b8",
        )
        sampling_time_ns = time.perf_counter_ns() - t_samp_start

        # 3. Step 2: Load the saved problem sheet (.dets.b8) and ground-truth answers (.obs.b8)
        dets = stim.read_shot_data_file(
            path=str(det_path),
            format="b8",
            num_detectors=circuit.num_detectors,
            bit_packed=False,
        )
        actual_obs = stim.read_shot_data_file(
            path=str(obs_path),
            format="b8",
            num_observables=circuit.num_observables,
            bit_packed=False,
        )

        # 4. Step 3: Pure Decoder Batch Inference & Latency Stopwatch
        if spec.decoder_id in self._custom_decoders:
            decode_fn = self._custom_decoders[spec.decoder_id]
            t_dec_start = time.perf_counter_ns()
            predictions = decode_fn(dets)
            pure_decode_ns = time.perf_counter_ns() - t_dec_start

        elif spec.decoder_id == "pymatching":
            dem = circuit.detector_error_model()
            matcher = pymatching.Matching.from_detector_error_model(dem)

            t_dec_start = time.perf_counter_ns()
            predictions = matcher.decode_batch(dets)
            pure_decode_ns = time.perf_counter_ns() - t_dec_start

        elif spec.decoder_id in ("bposd_fast", "bposd"):
            osd_order = int(spec.decoder_options.get("osd_order", 10))
            dem = circuit.detector_error_model()
            decoder = stimbposd.BPOSD(dem, osd_order=osd_order)

            t_dec_start = time.perf_counter_ns()
            predictions = decoder.decode_batch(dets)
            pure_decode_ns = time.perf_counter_ns() - t_dec_start

        else:
            raise ValueError(f"Unknown or unsupported decoder_id '{spec.decoder_id}' for RunSpec '{spec.run_spec_id}'")

        # 5. Step 4: Grading against ground-truth observables (.obs.b8)
        # predictions and actual_obs shape: (shots, num_observables)
        errors = int(np.any(predictions != actual_obs, axis=1).sum())
        ler = errors / shots if shots > 0 else 0.0
        avg_latency_ns = pure_decode_ns / shots if shots > 0 else 0.0

        # 6. Step 5: Write manifest metadata sidecar
        manifest_data = {
            "run_spec_id": spec.run_spec_id,
            "code_id": spec.code_id,
            "decoder_id": spec.decoder_id,
            "noise_p": spec.noise_p,
            "rounds": spec.rounds,
            "shots": shots,
            "errors": errors,
            "logical_error_rate": ler,
            "sampling_time_ns": sampling_time_ns,
            "pure_decode_time_ns": pure_decode_ns,
            "pure_decode_time_ms": pure_decode_ns / 1e6,
            "avg_latency_per_shot_us": avg_latency_ns / 1e3,
            "num_detectors": circuit.num_detectors,
            "num_observables": circuit.num_observables,
            "detectors_file": det_path.name,
            "observables_file": obs_path.name,
        }
        manifest_path = spec_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        return RunEvidence(
            run_spec_id=spec.run_spec_id,
            code_id=spec.code_id,
            decoder_id=spec.decoder_id,
            noise_p=spec.noise_p,
            rounds=spec.rounds,
            shots=shots,
            errors=errors,
            logical_error_rate=ler,
            sampling_time_ns=sampling_time_ns,
            decode_time_ns=pure_decode_ns,
            elapsed_time_ns=pure_decode_ns,
            avg_latency_per_shot_ns=avg_latency_ns,
            dets_path=str(det_path),
            obs_path=str(obs_path),
            manifest_path=str(manifest_path),
            status="SUCCESS",
        )

    def run_plan(
        self,
        plan: EvaluationPlan,
        quick: bool = False,
        quick_shots: int = 1000,
        num_workers: int = 4,
        on_progress: Optional[Callable[[int, int, RunEvidence], None]] = None,
    ) -> EvidenceBundle:
        """
        Executes all active RunSpecs in an EvaluationPlan.
        """
        run_evidences = []
        total = len(plan.run_specs)

        for idx, spec in enumerate(plan.run_specs, start=1):
            shots = quick_shots if quick else spec.shots
            evidence = self.run_spec(spec, shots_override=shots, num_workers=num_workers)
            run_evidences.append(evidence)

            if on_progress:
                on_progress(idx, total, evidence)

        bundle = EvidenceBundle(
            bundle_id=f"BUNDLE-{plan.plan_id}-{int(time.time())}",
            plan_id=plan.plan_id,
            total_run_specs=total,
            successful_runs=len([e for e in run_evidences if e.status == "SUCCESS"]),
            failed_runs=len([e for e in run_evidences if e.status != "SUCCESS"]),
            run_evidences=run_evidences,
            metadata={"quick_mode": quick, "num_workers": num_workers},
        )

        bundle_path = self.output_root / "evidence_bundle.json"
        bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

        return bundle
