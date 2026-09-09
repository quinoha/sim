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
        self._decoder_factories: Dict[str, Callable[[Any, Any], Callable[[np.ndarray], np.ndarray]]] = {}
        self._factory_cache: Dict[tuple, Callable[[np.ndarray], np.ndarray]] = {}

    def register_decoder(self, decoder_id: str, decode_fn: Callable[[np.ndarray], np.ndarray]) -> None:
        """
        Register a custom user decoder callable:
            fn(syndrome_matrix: np.ndarray) -> prediction_matrix: np.ndarray
        """
        self._custom_decoders[decoder_id] = decode_fn

    def register_decoder_factory(
        self,
        decoder_id: str,
        factory: Callable[[Any, Any], Callable[[np.ndarray], np.ndarray]],
    ) -> None:
        """
        Register a decoder that has to be built per workload:
            factory(run_spec, circuit) -> fn(syndrome_matrix) -> prediction_matrix

        `register_decoder` is enough for a decoder that only ever sees the syndrome
        matrix, but some need the circuit itself -- a neural decoder derives its
        input grid from the circuit's DETECTOR coordinates, and its time kernel is
        sized for a specific round count, so one callable cannot serve every
        RunSpec in a plan. Built instances are cached per (decoder_id, code,
        rounds, code_params) so a sweep does not rebuild the same model repeatedly.
        """
        self._decoder_factories[decoder_id] = factory

    def _resolve_factory(self, spec: RunSpec, circuit) -> Optional[Callable[[np.ndarray], np.ndarray]]:
        factory = self._decoder_factories.get(spec.decoder_id)
        if factory is None:
            return None
        # Noise is deliberately absent: a decoder built from a circuit depends on its
        # detector geometry and round count, not on the error rate baked into it.
        # Keying on noise_p rebuilt (and re-loaded from disk) the same model once per
        # swept p value.
        key = (spec.decoder_id, spec.code_id, int(spec.rounds), str(sorted(spec.code_params.items())))
        if key not in self._factory_cache:
            self._factory_cache[key] = factory(spec, circuit)
        return self._factory_cache[key]

    def run_spec(
        self,
        spec: RunSpec,
        shots_override: Optional[int] = None,
        num_workers: int = 4,
        warmup_shots: int = 64,
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
        # Resolve the decoder first, so construction never lands inside the stopwatch
        # and every candidate is timed under the same contract.
        factory_fn = self._resolve_factory(spec, circuit)
        if factory_fn is not None:
            decode_fn = factory_fn
        elif spec.decoder_id in self._custom_decoders:
            decode_fn = self._custom_decoders[spec.decoder_id]
        elif spec.decoder_id == "pymatching":
            decode_fn = pymatching.Matching.from_detector_error_model(
                circuit.detector_error_model()
            ).decode_batch
        elif spec.decoder_id in ("bposd_fast", "bposd"):
            osd_order = int(spec.decoder_options.get("osd_order", 10))
            decode_fn = stimbposd.BPOSD(
                circuit.detector_error_model(), osd_order=osd_order
            ).decode_batch
        else:
            raise ValueError(f"Unknown or unsupported decoder_id '{spec.decoder_id}' for RunSpec '{spec.run_spec_id}'")

        # Warm up outside the stopwatch. The first call into a decoder pays costs that
        # have nothing to do with steady-state decoding -- for a GPU model that is
        # cuDNN algorithm selection, caching-allocator growth and lazy CUDA module
        # init, which measured ~860 ms on the first Cascade RunSpec of a run and made
        # it look ~40% slower than the identical RunSpecs that followed it.
        if warmup_shots > 0 and shots > 0:
            try:
                decode_fn(dets[: min(warmup_shots, shots)])
            except Exception:  # noqa: BLE001 - a decoder that dislikes a short batch
                pass           # is not a reason to abandon the measurement

        t_dec_start = time.perf_counter_ns()
        predictions = decode_fn(dets)
        pure_decode_ns = time.perf_counter_ns() - t_dec_start

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
            "warmup_shots": warmup_shots,
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
        warmup_shots: int = 64,
        on_progress: Optional[Callable[[int, int, RunEvidence], None]] = None,
    ) -> EvidenceBundle:
        """
        Executes all active RunSpecs in an EvaluationPlan.
        """
        run_evidences = []
        total = len(plan.run_specs)

        for idx, spec in enumerate(plan.run_specs, start=1):
            shots = quick_shots if quick else spec.shots
            try:
                evidence = self.run_spec(
                    spec, shots_override=shots, num_workers=num_workers,
                    warmup_shots=warmup_shots,
                )
            except Exception as exc:  # noqa: BLE001
                # One RunSpec must not take the matrix down with it. A plan can hold a
                # decoder that only covers part of its own sweep -- a neural decoder
                # trained at one round count, say -- and the rest of the evidence is
                # still worth keeping. The failure is recorded, not swallowed.
                evidence = RunEvidence(
                    run_spec_id=spec.run_spec_id,
                    code_id=spec.code_id,
                    decoder_id=spec.decoder_id,
                    noise_p=spec.noise_p,
                    rounds=spec.rounds,
                    shots=shots,
                    errors=0,
                    logical_error_rate=0.0,
                    avg_latency_per_shot_ns=0.0,
                    dets_path="",
                    obs_path="",
                    manifest_path="",
                    status="FAILED",
                    error_message=f"{type(exc).__name__}: {exc}",
                )
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
