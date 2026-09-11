#!/usr/bin/env python3
"""Per-RunSpec decoder comparison.

Expands `configs/experiment.yaml` into its RunSpec matrix, then decodes each
one and prints a compact table: how many logical failures each decoder made,
on which workload, and how fast.

Two things this does deliberately differently from `runner/evaluation_runner.py`:

1. **Shots are sampled once per workload, not once per RunSpec.** Every decoder
   facing the same (code, p, rounds, basis) sees a bit-identical problem sheet,
   so a difference in the error column is attributable to the decoder rather
   than to sampling noise. The runner currently re-samples per RunSpec, which
   makes any two decoder columns independently noisy.

2. **Predictions are shape-checked before scoring.** A decoder returning
   `(shots, 1)` against a k=12 code would otherwise broadcast against the
   `(shots, 12)` truth and silently produce a wrong-but-plausible error count.

Cascade is added on top of the plan's own decoder list: the checkpoints in
`decoders/cascade/checkpoints/` only cover rotated surface codes at
`rounds == distance`, so every other RunSpec gets an explicit skip reason
instead of a blank cell.

Usage:
    python compare_decoders.py --shots 500
    python compare_decoders.py --shots 200 --decoders pymatching,cascade
    python compare_decoders.py --shots 2000 --codes surface_d5 --rounds 5 --noise-p 0.01
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import stim

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from m0_config import build_evaluation_plan  # noqa: E402
from qec_dataset.circuit_builder import build_circuit_from_run_spec  # noqa: E402

# Best available checkpoint per code distance.
CASCADE_CHECKPOINTS = {
    5: "cascade_d5.pth",
    7: "cascade_d7_H256_ddp_ema.pth",
    9: "cascade_d9_H256.pth",
    11: "cascade_d11_H128.pth",
}
CASCADE_TRAINED_P = 0.01


def workload_key(spec) -> tuple:
    """Everything that determines the shots, with the decoder deliberately left out."""
    return (spec.code_id, float(spec.noise_p), int(spec.rounds), "Z")


def workload_seed(key: tuple, base: int) -> int:
    digest = hashlib.sha256("|".join(str(k) for k in key).encode()).hexdigest()[:8]
    return (base + int(digest, 16)) % (2**31 - 1)


def find_checkpoints(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_dir() else None
    candidates = [REPO / "decoders" / "cascade" / "checkpoints"]
    # A git worktree does not carry gitignored files; fall back to the main checkout.
    for parent in REPO.parents:
        if parent.name == ".claude":
            candidates.append(parent.parent / "decoders" / "cascade" / "checkpoints")
    for c in candidates:
        if c.is_dir() and any(c.glob("*.pth")):
            return c
    return None


def cascade_plan(spec, ckpt_dir: Path | None) -> tuple[str | None, str]:
    """Returns (checkpoint path, skip reason). Exactly one is meaningful."""
    if spec.code_family != "surface":
        return None, f"surface-only (this RunSpec is {spec.code_family})"
    if ckpt_dir is None:
        return None, "no checkpoint dir found (pass --checkpoints)"
    distance = int(spec.code_params.get("distance", 0))
    name = CASCADE_CHECKPOINTS.get(distance)
    if name is None:
        return None, f"no checkpoint for d={distance} (have {sorted(CASCADE_CHECKPOINTS)})"
    path = ckpt_dir / name
    if not path.exists():
        return None, f"{name} missing"
    if int(spec.rounds) != distance:
        return None, f"rounds={spec.rounds}, checkpoint trained at rounds=d={distance}"
    return str(path), ""


def warm_up(fn, dets: np.ndarray) -> None:
    """Pays the first-call cost before the stopwatch starts.

    A decoder that knows its own steady-state shape warms that shape itself: on a
    GPU, cuDNN benchmarks its algorithms per shape and the caching allocator
    reserves blocks per shape, so warming on a 32-shot slice tunes for a shape the
    timed calls never use -- which is the same as not warming up at all, except it
    looks like it worked. Everything else just gets the short slice.
    """
    own = getattr(getattr(fn, "__self__", None), "warmup", None)
    try:
        if callable(own):
            own(dets)
        else:
            fn(dets[:32])
    except Exception:  # noqa: BLE001 - a decoder that dislikes a short batch is not
        pass           # a reason to abandon the measurement


def build_decoder(kind: str, spec, circuit, ckpt: str | None, cascade_batch: int | None = None):
    """Returns (callable taking dets -> predictions, setup seconds)."""
    t0 = time.perf_counter()
    if kind == "pymatching":
        import pymatching

        fn = pymatching.Matching.from_detector_error_model(circuit.detector_error_model()).decode_batch
    elif kind in ("bposd_fast", "bposd"):
        import stimbposd

        order = int(spec.decoder_options.get("osd_order", 10))
        fn = stimbposd.BPOSD(circuit.detector_error_model(), osd_order=order).decode_batch
    elif kind == "cascade":
        from decoders.cascade.adapter import load_pretrained

        distance = int(spec.code_params["distance"])
        fn = load_pretrained(circuit, distance, ckpt, batch_size=cascade_batch).decode_batch
    else:
        raise ValueError(f"unknown decoder '{kind}'")
    return fn, time.perf_counter() - t0


def score(predictions: np.ndarray, truth: np.ndarray, shots: int, num_obs: int) -> int:
    if predictions.shape != (shots, num_obs):
        raise ValueError(
            f"decoder returned {predictions.shape}, expected ({shots}, {num_obs}) — "
            "a mismatched shape would broadcast and silently corrupt the error count"
        )
    return int(np.any(predictions != truth, axis=1).sum())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-RunSpec decoder comparison on identical shots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("config", nargs="?", default="configs/experiment.yaml")
    ap.add_argument(
        "--shots",
        type=int,
        default=None,
        help="FIXED shot count per RunSpec. Omit to use the plan's stopping rule "
        "(sample until min_failures or max_shots), which is what the config actually asks for.",
    )
    ap.add_argument(
        "--min-failures",
        type=int,
        default=-1,
        help="stop once every decoder has this many logical failures. -1 reads it from "
        "the plan's protocol.accuracy.stopping_rule (0 disables).",
    )
    ap.add_argument(
        "--max-shots",
        type=int,
        default=-1,
        help="ceiling for the stopping rule. -1 reads it from the plan.",
    )
    ap.add_argument(
        "--batch", type=int, default=25000, help="sampling increment in adaptive mode"
    )
    ap.add_argument(
        "--decoders",
        default="pymatching,bposd_fast,cascade",
        help="comma separated; cascade is added on top of the plan's own list",
    )
    ap.add_argument("--codes", default=None, help="filter by code_id (comma separated)")
    ap.add_argument("--rounds", default=None, help="filter by rounds (comma separated)")
    ap.add_argument("--noise-p", default=None, help="filter by noise_p (comma separated)")
    ap.add_argument(
        "--override-p",
        type=float,
        default=None,
        help="DIAGNOSTIC: force every selected RunSpec to this p, even if the plan "
        "does not sweep it. Use --override-p 0.01 to see Cascade at its training point.",
    )
    ap.add_argument(
        "--decoder-shots",
        default=None,
        help="per-decoder shot budget, e.g. 'cascade=20000,bposd_fast=2000'. A decoder "
        "given fewer shots decodes a PREFIX of the same sample, so the comparison "
        "stays paired -- its shots are a subset of the others'.",
    )
    ap.add_argument(
        "--cascade-batch",
        type=int,
        default=None,
        help="Cascade forward batch size. Defaults to 512 on a CPU and 8192 on a GPU. "
        "The model is tiny per shot (~41 kernel launches, 6x6x6 grid), so small "
        "batches leave a GPU launch-bound rather than compute-bound; 8k-64k is the "
        "useful range there, and worth tuning per card.",
    )
    ap.add_argument("--checkpoints", default=None, help="Cascade checkpoint directory")
    ap.add_argument("--seed", type=int, default=12345, help="base for workload seed derivation")
    return ap.parse_args()


def parse_decoder_shots(spec: str | None) -> dict[str, int]:
    if not spec:
        return {}
    out = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        name, _, value = item.partition("=")
        if not value:
            raise ValueError(f"--decoder-shots entry '{item}' must look like name=N")
        out[name.strip()] = int(value)
    return out


def main() -> int:
    args = parse_args()
    wanted = [d.strip() for d in args.decoders.split(",") if d.strip()]

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[error] config not found: {config_path.resolve()}", file=sys.stderr)
        return 1

    plan = build_evaluation_plan(config_path)
    specs = list(plan.run_specs)

    if args.codes:
        keep = {c.strip() for c in args.codes.split(",")}
        specs = [s for s in specs if s.code_id in keep]
    if args.rounds:
        keep = {int(r) for r in args.rounds.split(",")}
        specs = [s for s in specs if int(s.rounds) in keep]
    if args.noise_p:
        keep = {float(p) for p in args.noise_p.split(",")}
        specs = [s for s in specs if any(abs(s.noise_p - p) < 1e-12 for p in keep)]
    if not specs:
        print("[error] filters selected 0 RunSpecs.", file=sys.stderr)
        return 1
    if args.override_p is not None:
        specs = [s.model_copy(update={"noise_p": args.override_p}) for s in specs]

    if any(d.startswith("bposd") for d in wanted) and any(
        s.decoder_id.startswith("bposd") for s in specs
    ):
        n = sum(1 for s in specs if s.decoder_id.startswith("bposd"))
        est = n * args.shots * 0.156 / 60
        print(
            f"\n[warning] {n} BP-OSD RunSpec(s) x {args.shots} shots. On a BB code this measured "
            f"~156 ms/shot, so expect roughly {est:.0f} min. Drop --decoders to "
            f"'pymatching,cascade' or lower --shots for a quick pass."
        )

    per_decoder_shots = parse_decoder_shots(args.decoder_shots)
    ckpt_dir = find_checkpoints(args.checkpoints)
    groups: dict[tuple, list] = {}
    for s_ in specs:
        groups.setdefault(workload_key(s_), []).append(s_)

    rule = specs[0].stopping_rule
    min_failures = args.min_failures if args.min_failures >= 0 else getattr(rule, "min_failures", 100)
    max_shots = args.max_shots if args.max_shots > 0 else getattr(rule, "max_shots", 200_000)
    fixed = args.shots is not None
    if fixed:
        min_failures, max_shots = 0, args.shots

    print()
    print("=" * 107)
    print(f"QEDA decoder comparison  |  {config_path}  |  plan={plan.plan_id}")
    mode = f"FIXED {max_shots:,} shots" if fixed else (
        f"stopping rule: min_failures={min_failures}, max_shots={max_shots:,}, batch={args.batch:,}")
    print(f"{len(specs)} RunSpecs / {len(groups)} workloads  |  {mode}")
    print(f"cascade ckpts: {ckpt_dir if ckpt_dir else 'none'}")
    print("=" * 107)
    print(f"{'code':<11} {'p':>7} {'r':>2} | {'decoder':<11} {'errors/shots':>15} {'LER':>10} "
          f"{'us/shot':>9}  note")
    print("-" * 107)

    rows_ok = rows_skip = weak = 0
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2])):
        code_id, p_val, rounds, basis = key
        group = groups[key]
        circuit = build_circuit_from_run_spec(group[0], basis=basis)
        seed = workload_seed(key, args.seed)
        num_obs = circuit.num_observables

        seen: set[str] = set()
        jobs = []
        for sp in group:
            if sp.decoder_id in wanted and sp.decoder_id not in seen:
                seen.add(sp.decoder_id)
                jobs.append((sp.decoder_id, sp, None))
        if "cascade" in wanted:
            ck, why = cascade_plan(group[0], ckpt_dir)
            jobs.append(("cascade", group[0], ck if ck else why))

        live, skipped = {}, []
        for name, spec, extra in jobs:
            if name == "cascade" and not (extra and str(extra).endswith(".pth")):
                skipped.append((name, f"skipped: {extra}"))
                continue
            try:
                fn, setup_s = build_decoder(name, spec, circuit, extra, args.cascade_batch)
                live[name] = {"fn": fn, "err": 0, "n": 0, "t": 0.0,
                              "cap": per_decoder_shots.get(name, max_shots)}
            except Exception as e:  # noqa: BLE001
                skipped.append((name, f"failed: {type(e).__name__}: {e}"))

        # Adaptive sampling: one stream per workload, so every decoder sees the same
        # shots in the same order. A decoder that hits its own --decoder-shots cap
        # stops consuming batches; its shots stay a prefix of everyone else's.
        sampler = circuit.compile_detector_sampler(seed=seed)
        total = zeros = 0
        obs_flips = 0
        stop = "max_shots"
        while total < max_shots and live:
            take = min(args.batch if not fixed else max_shots, max_shots - total)
            dets, obs = sampler.sample(take, separate_observables=True)
            obs_flips += int(obs.any(axis=1).sum())
            for name, st in live.items():
                # A decoder that already satisfies the rule stops consuming batches.
                # Otherwise a cheap decoder still hunting for its 100th failure would
                # drag an expensive one along for hundreds of thousands of shots.
                if st["n"] >= st["cap"] or (min_failures and st["err"] >= min_failures):
                    continue
                room = min(take, st["cap"] - st["n"])
                if st["n"] == 0:
                    warm_up(st["fn"], dets[:room])
                t0 = time.perf_counter()
                pred = st["fn"](dets[:room])
                st["t"] += time.perf_counter() - t0
                st["err"] += score(np.asarray(pred), obs[:room], room, num_obs)
                st["n"] += room
            total += take
            if min_failures > 0 and all(
                st["err"] >= min_failures or st["n"] >= st["cap"] for st in live.values()
            ):
                stop = "min_failures"
                break
        zeros = total

        print(f"{code_id:<11} {p_val:>7.4f} {rounds:>2} | {'(workload)':<11} {'':>15} {'':>10} "
              f"{'':>9}  det={circuit.num_detectors} obs={num_obs} seed={seed} "
              f"all-zero LER={obs_flips / max(zeros, 1):.3e} stop={stop}@{total:,}")

        for name, st in live.items():
            notes = []
            if min_failures and st["err"] < min_failures:
                notes.append(f"WEAK: {st['err']}<{min_failures}")
                weak += 1
            if name == "cascade":
                notes.append(f"trained p={CASCADE_TRAINED_P}"
                             + ("" if abs(p_val - CASCADE_TRAINED_P) < 1e-9 else " (SHIFTED)"))
            print(f"{'':<11} {'':>7} {'':>2} | {name:<11} {st['err']:>6}/{st['n']:<8} "
                  f"{st['err'] / max(st['n'], 1):>10.3e} {st['t'] / max(st['n'], 1) * 1e6:>9.1f}"
                  f"  {'; '.join(notes)}")
            rows_ok += 1
        for name, why in skipped:
            print(f"{'':<11} {'':>7} {'':>2} | {name:<11} {'-':>15} {'-':>10} {'-':>9}  {why}")
            rows_skip += 1
        print("-" * 107)

    print(f"done: {rows_ok} decoded, {rows_skip} skipped/failed, {weak} below min_failures")
    print("us/shot is batch throughput, NOT p50/p99 latency. Compare within a workload only.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
