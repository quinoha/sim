#!/usr/bin/env python3
"""
QEDA Simulator - Main Execution Entrypoint (Stage 1: SetupBuilder).

Usage:
    python main.py
    python main.py configs/experiment.yaml
    python main.py configs/experiment.yaml -o outputs/evaluation_plan.json
"""
import argparse
import sys
from pathlib import Path

from m0_config import build_evaluation_plan, ConfigValidationError, EvaluationPlan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QEDA QEC Simulator - SetupBuilder & Experiment Execution Entrypoint",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/experiment.yaml",
        help="Path to the experiment configuration YAML/JSON file (default: configs/experiment.yaml)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output path to save the resolved EvaluationPlan as JSON (e.g. outputs/evaluation_plan.json)"
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress the detailed RunSpec matrix table output"
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        default=None,
        help="Filter target figures to generate (e.g. --figures C-F01 C-F02)"
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Trigger Stage 2 (EvaluationRunner) to sample shots and execute parallel decoding"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke-test mode (samples 100 shots per RunSpec instead of full count)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes for Sinter sampling and decoding (default: 4)"
    )
    return parser.parse_args()


def print_banner(plan: EvaluationPlan) -> None:
    print("\n" + "=" * 80)
    print("QEDA Simulator | Stage 1: SetupBuilder (EvaluationPlan Generator)")
    print("=" * 80)
    print(f"Plan ID            : {plan.plan_id}")
    if plan.source_path:
        print(f"Source Config File : {plan.source_path}")
    print(f"Target Figures     : {', '.join(plan.target_figure_ids)}")
    print(f"Total Active Specs : {len(plan.run_specs)} RunSpecs (Orthogonal Sweep Grid)")
    print("=" * 80)


def print_runspec_table(plan: EvaluationPlan) -> None:
    print(f"\n[RunSpec Execution Matrix ({len(plan.run_specs)} Active Points)]")
    
    # Table headers
    header = (
        f"{'#':<3} | {'Code ID':<12} | {'Family':<8} | {'Decoder':<12} | "
        f"{'Noise p':<9} | {'Rounds':<6} | {'Shots':<8} | {'Scenario':<8} | {'Seed':<11}"
    )
    separator = "-" * len(header)
    
    print(separator)
    print(header)
    print(separator)

    current_group = None
    for idx, rs in enumerate(plan.run_specs, 1):
        group_key = (rs.code_id, rs.decoder_id)
        if current_group is not None and current_group != group_key:
            # Print subtle sub-separator between different (Code x Decoder) workloads
            print("-" * len(header))
        current_group = group_key

        print(
            f"{idx:<3} | "
            f"{rs.code_id:<12} | "
            f"{rs.code_family:<8} | "
            f"{rs.decoder_id:<12} | "
            f"{rs.noise_p:<9.4f} | "
            f"{rs.rounds:<6} | "
            f"{rs.shots:<8} | "
            f"{rs.scenario:<8} | "
            f"{rs.seed:<11}"
        )

    print(separator)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)

    if not config_path.exists():
        print(f"[Error] Configuration file not found at '{config_path.resolve()}'", file=sys.stderr)
        return 1

    try:
        # 1. Build EvaluationPlan via SetupBuilder
        plan = build_evaluation_plan(config_path, target_figures=args.figures)

        # 2. Print Summary Banner
        print_banner(plan)
        print("\n" + plan.summary())

        # 3. Print 24 RunSpec Table by default (unless --quiet)
        if not args.quiet:
            print_runspec_table(plan)

        # 4. Save to output JSON if requested
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(plan.model_dump_json(indent=2))
            print(f"\n[Saved] EvaluationPlan successfully written to: {out_path.resolve()}")

        # 5. Optional Stage 2: EvaluationRunner Execution
        if args.run:
            from runner import EvaluationRunner
            quick_shots = 100 if args.quick else None

            print("\n" + "=" * 80)
            print(f"QEDA Simulator | Stage 2: EvaluationRunner (Parallel Sinter Execution)")
            print(f"Mode               : {'QUICK SMOKE TEST (100 shots/spec)' if args.quick else 'FULL PRODUCTION RUN'}")
            print(f"Worker Processes   : {args.workers}")
            print("=" * 80)

            runner = EvaluationRunner(output_root="outputs/evidence")

            def on_progress(idx: int, total: int, evidence):
                print(
                    f"[{idx:>2}/{total}] {evidence.code_id:<12} x {evidence.decoder_id:<10} "
                    f"(p={evidence.noise_p:.4f}, r={evidence.rounds}) | "
                    f"Errors: {evidence.errors:>4}/{evidence.shots:<4} (LER: {evidence.logical_error_rate:.3e}) | "
                    f"Decode: {evidence.decode_time_ns / 1e6:>6.2f} ms ({evidence.avg_latency_per_shot_ns / 1e3:>5.1f} us/shot) | "
                    f"Samp: {evidence.sampling_time_ns / 1e6:>5.1f} ms"
                )

            bundle = runner.run_plan(
                plan,
                quick=args.quick,
                quick_shots=quick_shots or 1000,
                num_workers=args.workers,
                on_progress=on_progress,
            )

            print("=" * 80)
            print(f"[Complete] Stage 2 Execution Finished!")
            print(f"Bundle ID          : {bundle.bundle_id}")
            print(f"Successful Runs    : {bundle.successful_runs}/{bundle.total_run_specs}")
            print(f"Evidence Artifacts : outputs/evidence/evidence_bundle.json")
            print("=" * 80)
        else:
            print("\n[Complete] SetupBuilder execution finished. Ready for Stage 2 (EvaluationRunner).")
            print("Tip: Run with '--run --quick' to execute simulations for all 24 RunSpecs.")

        return 0

    except ConfigValidationError as e:
        print(f"\n[Validation Error] Configuration Validation Failed:\n{e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"\n[Error] Unexpected Error: {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
