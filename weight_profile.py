#!/usr/bin/env python3
"""Weight-resolved decoder profile: find where a decoder stops correcting.

An LER sweep tells you how often a decoder fails at a given physical error
rate. It cannot tell you *why*, because every shot in it mixes error weights
together and the failures are dominated by whichever weight happens to carry
the probability mass. This script conditions on weight instead: it injects
error sets of exactly weight w, hands the resulting syndromes to every decoder
under test, and reports what each one did.

What the columns mean, and why LER alone is not enough. A decoder can fail at
high weight in three qualitatively different ways, and they are the difference
between "needs more training" and "is broken":

    graceful   LER climbs smoothly, false-alarm and miss both stay balanced.
               The decoder still carries information, it just runs out of
               distance. This is what a good decoder looks like past (d-1)/2.

    collapsed  miss -> 1, false-alarm -> 0, pred-flip -> 0. The decoder has
               quietly become "always predict no logical flip". Its LER then
               equals the truth-flip rate, and it is adding nothing.

    guessing   false-alarm and miss both -> 0.5, pred-flip -> 0.5. The decoder
               is emitting noise uncorrelated with the input.

The `trivial` row is the all-zeros decoder -- predict no logical flip, ever.
Any decoder whose LER meets or exceeds that row at weight w has stopped earning
its place at that weight, and the summary reports the first w where that
happens.

Weight counts fired DEM fault mechanisms, not flipped data qubits; see
qec_dataset/error_weight.py for why, and for the merged-mechanism caveat.

Usage:
    python weight_profile.py --distance 5 --rounds 5 --p 0.01
    python weight_profile.py --distance 5 --max-weight 12 --samples 4000
    python weight_profile.py --distance 5 --mode natural --shots 200000
    python weight_profile.py --distance 7 --decoders cascade --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import stim

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from qec_dataset.codes.surface import build_surface_code_circuit  # noqa: E402
from qec_dataset.noise import NoiseModel  # noqa: E402
from qec_dataset.error_weight import (  # noqa: E402
    bucket_stats,
    fault_incidence,
    natural_sample,
    random_guess_ler,
    sample_weight_sets,
    symptoms,
    verify_incidence,
)


def build_decoders(names: list[str], circuit: stim.Circuit, args) -> tuple[dict, list]:
    """Builds each requested decoder. Returns (live, skipped).

    Each decoder gets the detector error model it wants -- pymatching needs a
    decomposed one, since an undecomposed circuit-level DEM contains errors
    that fire more than two detectors and a matching graph cannot represent
    those. That is independent of the DEM used to define weight, which must
    stay undecomposed so one fault mechanism counts as one unit of weight.
    """
    live: dict = {}
    skipped: list[tuple[str, str]] = []

    for name in names:
        t0 = time.perf_counter()
        try:
            if name == "pymatching":
                import pymatching

                fn = pymatching.Matching.from_detector_error_model(
                    circuit.detector_error_model(decompose_errors=True)
                ).decode_batch
            elif name in ("bposd", "bposd_fast"):
                import stimbposd

                fn = stimbposd.BPOSD(
                    circuit.detector_error_model(decompose_errors=True),
                    osd_order=args.osd_order,
                ).decode_batch
            elif name == "cascade":
                from compare_decoders import CASCADE_CHECKPOINTS, find_checkpoints
                from decoders.cascade.adapter import load_pretrained

                ckpt_dir = find_checkpoints(args.checkpoints)
                if ckpt_dir is None:
                    raise FileNotFoundError("no checkpoint directory found (pass --checkpoints)")
                filename = CASCADE_CHECKPOINTS.get(args.distance)
                if filename is None:
                    raise ValueError(
                        f"no checkpoint for d={args.distance} "
                        f"(have {sorted(CASCADE_CHECKPOINTS)})"
                    )
                path = ckpt_dir / filename
                if not path.exists():
                    raise FileNotFoundError(f"{path} missing")
                if args.rounds != args.distance:
                    raise ValueError(
                        f"rounds={args.rounds} but the checkpoints were trained at "
                        f"rounds=d={args.distance}"
                    )
                fn = load_pretrained(
                    circuit, args.distance, str(path), batch_size=args.cascade_batch
                ).decode_batch
            else:
                raise ValueError(f"unknown decoder '{name}'")
        except Exception as e:  # noqa: BLE001
            skipped.append((name, f"{type(e).__name__}: {e}"))
            continue
        live[name] = {"fn": fn, "setup_s": time.perf_counter() - t0}

    return live, skipped


def warm_up(fn, dets: np.ndarray) -> None:
    """Pays the first-call cost outside the measurement, at the real shape."""
    own = getattr(getattr(fn, "__self__", None), "warmup", None)
    try:
        if callable(own):
            own(dets)
        else:
            fn(dets[:32])
    except Exception:  # noqa: BLE001
        pass


def trivial_stats(weight: int, truth: np.ndarray):
    """The all-zeros decoder, scored exactly like a real one."""
    return bucket_stats(weight, np.zeros_like(truth), truth)


def classify(stats) -> str:
    """Names the failure mode from the conditional splits. See module docstring."""
    if stats.n == 0 or not np.isfinite(stats.miss):
        return ""
    if stats.errors == 0:
        return "clean"
    fa = stats.false_alarm if np.isfinite(stats.false_alarm) else 0.0
    if stats.miss > 0.9 and fa < 0.1:
        return "COLLAPSED"
    if abs(stats.miss - 0.5) < 0.15 and abs(fa - 0.5) < 0.15:
        return "GUESSING"
    return "degrading"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Weight-resolved decoder profile",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--distance", type=int, default=5, help="rotated surface code distance")
    ap.add_argument("--rounds", type=int, default=None, help="syndrome rounds (default: = distance)")
    ap.add_argument(
        "--p",
        type=float,
        default=0.01,
        help="physical error rate the DEM is built at. In uniform mode this does NOT "
        "affect the injected error sets -- only how each decoder is calibrated "
        "(0.01 is where the Cascade checkpoints were trained).",
    )
    ap.add_argument("--basis", default="Z", choices=["Z", "X"])
    ap.add_argument(
        "--mode",
        default="uniform",
        choices=["uniform", "natural"],
        help="uniform: inject exactly-weight-w sets (reaches any weight). "
        "natural: sample the DEM at its own p and bin by realized weight "
        "(physically faithful, limited reach).",
    )
    ap.add_argument("--min-weight", type=int, default=1)
    ap.add_argument("--max-weight", type=int, default=10)
    ap.add_argument("--samples", type=int, default=2000, help="error sets per weight (uniform mode)")
    ap.add_argument("--shots", type=int, default=100_000, help="total shots (natural mode)")
    ap.add_argument(
        "--min-bucket",
        type=int,
        default=50,
        help="natural mode: skip weight buckets thinner than this, they carry no signal",
    )
    ap.add_argument("--decoders", default="cascade,pymatching")
    ap.add_argument("--osd-order", type=int, default=10)
    ap.add_argument("--checkpoints", default=None, help="Cascade checkpoint directory")
    ap.add_argument("--cascade-batch", type=int, default=None, help="default: 512 CPU / 8192 GPU")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--json", default=None, help="also write the full table here")
    ap.add_argument(
        "--no-verify",
        action="store_true",
        help="skip checking the incidence matrices against stim's own sampler",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.rounds is None:
        args.rounds = args.distance
    rng = np.random.default_rng(args.seed)

    circuit = build_surface_code_circuit(
        distance=args.distance,
        rounds=args.rounds,
        basis=args.basis,
        noise=NoiseModel.uniform(args.p),
    )
    # Undecomposed: one physical fault mechanism must count as one unit of weight.
    dem = circuit.detector_error_model(decompose_errors=False)
    inc = fault_incidence(dem)

    print()
    print("=" * 100)
    print(f"Weight-resolved decoder profile  |  surface d={args.distance} r={args.rounds} "
          f"basis={args.basis} p={args.p:g}")
    print(f"{inc.num_mechanisms} fault mechanisms, {inc.num_detectors} detectors, "
          f"{inc.num_observables} observable(s)")
    corrects = (args.distance - 1) // 2
    print(f"code should correct every weight <= {corrects}; first uncorrectable weight is "
          f"{corrects + 1}")
    print(f"random-guess LER ceiling = {random_guess_ler(inc.num_observables):.4f}")
    print("=" * 100)

    if not args.no_verify:
        t0 = time.perf_counter()
        verify_incidence(dem, inc, shots=256, seed=args.seed)
        print(f"[verify] incidence matrices reproduce stim's syndromes "
              f"({time.perf_counter() - t0:.2f}s)")

    names = [d.strip() for d in args.decoders.split(",") if d.strip()]
    live, skipped = build_decoders(names, circuit, args)
    for name, why in skipped:
        print(f"[skip] {name}: {why}")
    if not live:
        print("[error] no decoders available", file=sys.stderr)
        return 1
    built = ", ".join(f'{n} {st["setup_s"]:.1f}s' for n, st in live.items())
    print(f"[setup] {built}")

    # Build the per-weight problem sets. Every decoder sees bit-identical input.
    buckets: list[tuple[int, np.ndarray, np.ndarray]] = []
    if args.mode == "uniform":
        for w in range(args.min_weight, args.max_weight + 1):
            if w > inc.num_mechanisms:
                break
            sel = sample_weight_sets(inc.num_mechanisms, w, args.samples, rng)
            dets, truth = symptoms(inc, sel)
            buckets.append((w, dets, truth))
    else:
        dets_all, obs_all, weights = natural_sample(dem, args.shots, args.seed)
        for w in range(args.min_weight, args.max_weight + 1):
            mask = weights == w
            if int(mask.sum()) < args.min_bucket:
                continue
            buckets.append((w, dets_all[mask], obs_all[mask]))

    if not buckets:
        print("[error] no weight buckets with enough samples", file=sys.stderr)
        return 1

    for st in live.values():
        warm_up(st["fn"], buckets[0][1])

    print()
    print(f"{'w':>3} {'n':>7} {'decoder':<12} {'errors':>7} {'LER':>9} {'95% CI':>19} "
          f"{'FA':>6} {'miss':>6} {'pred':>6}  mode")
    print("-" * 100)

    table: list[dict] = []
    first_useless: dict[str, int] = {}

    for w, dets, truth in buckets:
        rows = [("trivial", trivial_stats(w, truth))]
        for name, st in live.items():
            rows.append((name, bucket_stats(w, st["fn"](dets), truth)))

        baseline = rows[0][1].ler
        for name, s in rows:
            tag = "" if name == "trivial" else classify(s)
            # "Useless" means it failed AND did no better than predicting all zeros.
            # Requiring an actual failure matters at low weight, where both the
            # decoder and the trivial baseline can be clean -- a tie at zero errors
            # is a decoder working perfectly, not one that has given up.
            if name != "trivial" and name not in first_useless and s.errors > 0 and s.ler >= baseline:
                first_useless[name] = w
            print(f"{w:>3} {s.n:>7} {name:<12} {s.errors:>7} {s.ler:>9.4f} "
                  f"[{s.ci[0]:.4f},{s.ci[1]:.4f}] {s.false_alarm:>6.3f} {s.miss:>6.3f} "
                  f"{s.pred_flip_rate:>6.3f}  {tag}")
            table.append({
                "weight": w, "decoder": name, "n": s.n, "errors": s.errors, "ler": s.ler,
                "ci_low": s.ci[0], "ci_high": s.ci[1], "truth_flip_rate": s.truth_flip_rate,
                "pred_flip_rate": s.pred_flip_rate, "false_alarm": s.false_alarm,
                "miss": s.miss, "mode": tag,
            })
        print("-" * 100)

    print()
    print("Summary")
    for name in live:
        w = first_useless.get(name)
        if w is None:
            print(f"  {name:<12} still beats the all-zeros decoder at every weight tested")
        else:
            print(f"  {name:<12} stops beating the all-zeros decoder at weight {w}")
    print()
    print("FA   = false alarm: predicted a flip when the truth had none")
    print("miss = missed a real flip. pred = fraction of shots predicted as flipped.")
    print("miss->1 with FA->0 is collapse to 'always no flip'; both ->0.5 is guessing.")
    if args.mode == "uniform":
        print("Uniform mode weights every mechanism equally, so these are NOT the rates a "
              "run at p would see -- it is a structural probe, not a prediction of LER.")
    print()

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "distance": args.distance, "rounds": args.rounds, "p": args.p,
                    "basis": args.basis, "mode": args.mode, "seed": args.seed,
                    "num_mechanisms": inc.num_mechanisms,
                    "num_detectors": inc.num_detectors,
                    "num_observables": inc.num_observables,
                    "random_guess_ler": random_guess_ler(inc.num_observables),
                    "rows": table,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[saved] {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
