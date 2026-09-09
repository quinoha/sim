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

    @staticmethod
    def workload_id(spec: RunSpec, basis: str = "Z") -> str:
        """Identifies the shots a RunSpec needs, with the decoder left out.

        Two RunSpecs that differ only by decoder are the same decoding problem, so
        they should be handed the same shots. Sampling them separately makes each
        decoder's error count independently noisy and throws away the paired
        comparison, which is the whole point of running them side by side.
        """
        params = "-".join(f"{k}{v}" for k, v in sorted(spec.code_params.items()))
        return f"WL-{spec.code_id}-{params}-P{spec.noise_p:g}-R{spec.rounds}-B{basis}"

    @staticmethod
    def _score(predictions: np.ndarray, truth: np.ndarray) -> int:
        """Counts shots where any observable was mispredicted, shape-checked.

        Without the check a decoder returning `(shots, 1)` against a k=12 code
        broadcasts against the `(shots, 12)` truth and yields a plausible but
        wrong count, with no error raised anywhere.
        """
        predictions = np.asarray(predictions)
        if predictions.shape != truth.shape:
            raise ValueError(
                f"decoder returned predictions of shape {predictions.shape}, expected "
                f"{truth.shape}; a mismatched shape would broadcast and silently "
                f"corrupt the logical error count"
            )
        return int(np.any(predictions != truth, axis=1).sum())

    def _decode_with_rule(
        self,
        decode_fn,
        dets: np.ndarray,
        truth: np.ndarray,
        min_failures: int,
        batch: int,
        warmup_shots: int,
    ) -> tuple[int, int, int, str]:
        """Decodes until `min_failures` failures accumulate or the pool runs out.

        Returns (errors, shots_used, decode_ns, stopped_by). With min_failures <= 0
        the whole pool is decoded in a single call, which keeps the timing
        characteristics a caller asking for an exact shot count would expect.
        """
        pool = dets.shape[0]
        if warmup_shots > 0 and pool > 0:
            try:
                decode_fn(dets[: min(warmup_shots, pool)])
            except Exception:  # noqa: BLE001 - a decoder that dislikes a short batch
                pass           # is not a reason to abandon the measurement

        if min_failures <= 0:
            t0 = time.perf_counter_ns()
            predictions = decode_fn(dets)
            elapsed = time.perf_counter_ns() - t0
            return self._score(predictions, truth), pool, elapsed, "fixed"

        errors = used = elapsed = 0
        while used < pool:
            take = min(batch, pool - used)
            chunk = dets[used : used + take]
            t0 = time.perf_counter_ns()
            predictions = decode_fn(chunk)
            elapsed += time.perf_counter_ns() - t0
            errors += self._score(predictions, truth[used : used + take])
            used += take
            if errors >= min_failures:
                return errors, used, elapsed, "min_failures"
        return errors, used, elapsed, "max_shots"

    def _build_decoder(self, spec: RunSpec, circuit):
        """Resolves a RunSpec's decoder. Never called inside a stopwatch."""
        factory_fn = self._resolve_factory(spec, circuit)
        if factory_fn is not None:
            return factory_fn
        if spec.decoder_id in self._custom_decoders:
            return self._custom_decoders[spec.decoder_id]
        if spec.decoder_id == "pymatching":
            return pymatching.Matching.from_detector_error_model(
                circuit.detector_error_model()
            ).decode_batch
        if spec.decoder_id in ("bposd_fast", "bposd"):
            osd_order = int(spec.decoder_options.get("osd_order", 10))
            return stimbposd.BPOSD(
                circuit.detector_error_model(), osd_order=osd_order
            ).decode_batch
        raise ValueError(
            f"Unknown or unsupported decoder_id '{spec.decoder_id}' for RunSpec '{spec.run_spec_id}'"
        )

    def _load_shots(self, circuit, out_dir: Path, shots: int, seed: int) -> tuple:
        """Samples `shots` to disk and reads them back. Reuses an adequate pool."""
        det_path = out_dir / "shots.dets.b8"
        obs_path = out_dir / "shots.obs.b8"
        meta_path = out_dir / "shots.meta.json"

        have = 0
        if meta_path.exists() and det_path.exists() and obs_path.exists():
            try:
                cached = json.loads(meta_path.read_text(encoding="utf-8"))
                if cached.get("seed") == seed and cached.get("num_detectors") == circuit.num_detectors:
                    have = int(cached.get("shots", 0))
            except (json.JSONDecodeError, OSError):
                have = 0

        t0 = time.perf_counter_ns()
        if have < shots:
            out_dir.mkdir(parents=True, exist_ok=True)
            circuit.compile_detector_sampler(seed=seed).sample_write(
                shots,
                filepath=str(det_path),
                format="b8",
                obs_out_filepath=str(obs_path),
                obs_out_format="b8",
            )
            meta_path.write_text(
                json.dumps({"seed": seed, "shots": shots, "num_detectors": circuit.num_detectors}),
                encoding="utf-8",
            )
            have = shots
        sampling_ns = time.perf_counter_ns() - t0

        dets = stim.read_shot_data_file(
            path=str(det_path), format="b8",
            num_detectors=circuit.num_detectors, bit_packed=False,
        )
        obs = stim.read_shot_data_file(
            path=str(obs_path), format="b8",
            num_observables=circuit.num_observables, bit_packed=False,
        )
        return dets[:shots], obs[:shots], det_path, obs_path, sampling_ns

    def run_spec(
        self,
        spec: RunSpec,
        shots_override: Optional[int] = None,
        num_workers: int = 4,
        warmup_shots: int = 64,
        min_failures: int = 0,
        stopping_batch: int = 25_000,
        shots_dir: Optional[Path] = None,
        preloaded: Optional[tuple] = None,
    ) -> RunEvidence:
        """
        Executes a single RunSpec:
        1. Builds Stim quantum circuit.
        2. Samples shots to disk (.dets.b8, .obs.b8) ONCE.
        3. Loads saved files and executes pure decoding.
        4. Scores predictions and profiles pure latency.

        `shots_dir` and `preloaded` let `run_plan` hand in a pool already sampled
        for the whole workload, so decoders sharing a workload face identical
        shots. Called on its own, the RunSpec samples its own pool as before.
        """
        shots = shots_override if shots_override is not None else spec.shots
        spec_dir = self.output_root / spec.run_spec_id
        spec_dir.mkdir(parents=True, exist_ok=True)

        # 1. Quantum circuit construction via Stage 1 -> Stim bridge
        circuit = build_circuit_from_run_spec(spec)

        # 2. Step 1: the shot pool. Shared when run_plan supplies one, otherwise
        #    sampled here into this RunSpec's own directory.
        if preloaded is not None:
            dets, actual_obs, det_path, obs_path, sampling_time_ns = preloaded
        else:
            dets, actual_obs, det_path, obs_path, sampling_time_ns = self._load_shots(
                circuit, shots_dir or spec_dir, shots, spec.seed
            )

        # 3. Step 2: resolve the decoder before any stopwatch starts
        decode_fn = self._build_decoder(spec, circuit)

        # 4. Step 3: decode under the stopping rule, warmed up outside the timer
        errors, shots_used, pure_decode_ns, stopped_by = self._decode_with_rule(
            decode_fn, dets, actual_obs, min_failures, stopping_batch, warmup_shots
        )

        # 5. Step 4: grading against ground-truth observables (.obs.b8)
        ler = errors / shots_used if shots_used > 0 else 0.0
        avg_latency_ns = pure_decode_ns / shots_used if shots_used > 0 else 0.0
        wl_id = self.workload_id(spec)

        # 6. Step 5: Write manifest metadata sidecar
        manifest_data = {
            "run_spec_id": spec.run_spec_id,
            "workload_id": wl_id,
            "code_id": spec.code_id,
            "decoder_id": spec.decoder_id,
            "noise_p": spec.noise_p,
            "rounds": spec.rounds,
            "seed": spec.seed,
            "shots": shots_used,
            "shots_available": int(dets.shape[0]),
            "stopped_by": stopped_by,
            "min_failures": min_failures,
            "errors": errors,
            "logical_error_rate": ler,
            "sampling_time_ns": sampling_time_ns,
            "pure_decode_time_ns": pure_decode_ns,
            "pure_decode_time_ms": pure_decode_ns / 1e6,
            "avg_latency_per_shot_us": avg_latency_ns / 1e3,
            "warmup_shots": warmup_shots,
            "num_detectors": circuit.num_detectors,
            "num_observables": circuit.num_observables,
            "detectors_file": str(det_path),
            "observables_file": str(obs_path),
        }
        manifest_path = spec_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        return RunEvidence(
            run_spec_id=spec.run_spec_id,
            code_id=spec.code_id,
            decoder_id=spec.decoder_id,
            noise_p=spec.noise_p,
            rounds=spec.rounds,
            shots=shots_used,
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
            stopped_by=stopped_by,
            workload_id=wl_id,
        )

    def run_plan(
        self,
        plan: EvaluationPlan,
        quick: bool = False,
        quick_shots: int = 1000,
        num_workers: int = 4,
        warmup_shots: int = 64,
        share_shots: bool = True,
        use_stopping_rule: bool = True,
        stopping_batch: int = 25_000,
        on_progress: Optional[Callable[[int, int, RunEvidence], None]] = None,
    ) -> EvidenceBundle:
        """
        Executes all active RunSpecs in an EvaluationPlan.

        RunSpecs are grouped by workload so every decoder facing the same
        (code, params, p, rounds) decodes a bit-identical pool of shots -- a
        difference in the error column is then attributable to the decoder and not
        to sampling noise. Within a workload the pool is sampled once.

        When the plan declares `protocol.accuracy.stopping_rule`, each RunSpec
        decodes only until it has accumulated `min_failures` logical failures.
        A fixed shot count under-powers exactly the interesting end of a sweep:
        at p=1e-3 on surface d=5, 200k shots yields around 33 failures against a
        declared minimum of 100. `quick` forces the fixed path, since a smoke test
        wants a bounded runtime rather than a statistically usable answer.
        """
        run_evidences: list[RunEvidence] = []
        total = len(plan.run_specs)

        rule = plan.run_specs[0].stopping_rule if plan.run_specs else None
        min_failures = 0
        if use_stopping_rule and not quick and rule is not None:
            min_failures = int(getattr(rule, "min_failures", 0) or 0)

        # Group while preserving the plan's own ordering, so progress output still
        # reads in RunSpec order within each workload.
        groups: Dict[str, list] = {}
        for spec in plan.run_specs:
            groups.setdefault(self.workload_id(spec) if share_shots else spec.run_spec_id, []).append(spec)

        idx = 0
        for wl_id, specs in groups.items():
            pool = None
            if share_shots:
                head = specs[0]
                shots = quick_shots if quick else head.shots
                try:
                    circuit = build_circuit_from_run_spec(head)
                    wl_dir = self.output_root / "_workloads" / wl_id
                    pool = self._load_shots(circuit, wl_dir, shots, head.seed)
                except Exception:  # noqa: BLE001 - fall back to per-RunSpec sampling
                    pool = None

            for spec in specs:
                idx += 1
                shots = quick_shots if quick else spec.shots
                try:
                    evidence = self.run_spec(
                        spec,
                        shots_override=shots,
                        num_workers=num_workers,
                        warmup_shots=warmup_shots,
                        min_failures=min_failures,
                        stopping_batch=stopping_batch,
                        preloaded=pool,
                    )
                except Exception as exc:  # noqa: BLE001
                    # One RunSpec must not take the matrix down with it. A plan can hold
                    # a decoder that only covers part of its own sweep -- a neural
                    # decoder trained at one round count, say -- and the rest of the
                    # evidence is still worth keeping. The failure is recorded.
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
                        workload_id=wl_id,
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
            metadata={
                "quick_mode": quick,
                "num_workers": num_workers,
                "shared_shots": share_shots,
                "min_failures": min_failures,
                "warmup_shots": warmup_shots,
            },
        )

        bundle_path = self.output_root / "evidence_bundle.json"
        bundle_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")

        return bundle
