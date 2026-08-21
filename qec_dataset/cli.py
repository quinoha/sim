from __future__ import annotations

import argparse

from .circuits import generic_css_memory_circuit
from .codes.bb import build_bb_code, parse_terms
from .codes.surface import build_surface_code_circuit, rotated_surface_code
from .distance import search_distance, shortest_graphlike_distance
from .noise import NoiseModel
from .report import print_code_summary


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rounds", type=int, default=None, help="syndrome-extraction rounds")
    p.add_argument("--basis", choices=["X", "Z"], default="Z")
    p.add_argument("--p", type=float, default=0.0, help="uniform circuit-level noise for the emitted circuit")
    p.add_argument("--out", type=str, default=None, help="path to write the emitted .stim circuit")


def cmd_surface(args: argparse.Namespace) -> None:
    rounds = args.rounds or args.distance
    code, _ = rotated_surface_code(args.distance, basis=args.basis)
    circuit = build_surface_code_circuit(args.distance, rounds, basis=args.basis, noise=NoiseModel.uniform(args.p))

    dist_circuit = build_surface_code_circuit(args.distance, rounds, basis=args.basis, noise=NoiseModel.uniform(1e-3))
    distance = shortest_graphlike_distance(dist_circuit)

    print_code_summary(code, circuit, distance)
    if args.out:
        circuit.to_file(args.out)
        print(f"circuit written to {args.out}")


def cmd_bb(args: argparse.Namespace) -> None:
    a_terms = parse_terms(args.A)
    b_terms = parse_terms(args.B)
    code = build_bb_code(args.l, args.m, a_terms, b_terms)
    rounds = args.rounds or 6

    circuit = generic_css_memory_circuit(code, rounds, basis=args.basis, noise=NoiseModel.uniform(args.p))

    dist_circuit = generic_css_memory_circuit(code, rounds, basis=args.basis, noise=NoiseModel.uniform(1e-3))
    distance = search_distance(dist_circuit)

    print_code_summary(code, circuit, distance)
    if args.out:
        circuit.to_file(args.out)
        print(f"circuit written to {args.out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qec_dataset",
        description="Emit stim circuits and report [[n, k, d]] for surface / BB codes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_surface = sub.add_parser("surface", help="rotated surface code (stim built-in generator)")
    p_surface.add_argument("--distance", type=int, required=True)
    _add_common_args(p_surface)
    p_surface.set_defaults(func=cmd_surface)

    p_bb = sub.add_parser("bb", help="bivariate bicycle code (generic constructor)")
    p_bb.add_argument("--l", type=int, required=True)
    p_bb.add_argument("--m", type=int, required=True)
    p_bb.add_argument("--A", type=str, required=True, help="e.g. 'x3,y1,y2'")
    p_bb.add_argument("--B", type=str, required=True, help="e.g. 'y3,x1,x2'")
    _add_common_args(p_bb)
    p_bb.set_defaults(func=cmd_bb)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
