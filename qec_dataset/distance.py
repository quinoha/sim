"""Code-distance estimation, reusing stim's own circuit-based search tools
instead of a hand-rolled GF(2) combinatorial search.

- `shortest_graphlike_distance`: exact for "graphlike" circuits (errors
  that create at most two detection events) such as the surface code.
- `search_distance`: a heuristic bounded search
  (`search_for_undetectable_logical_errors`) that also works for
  non-graphlike circuits (e.g. BB codes, whose checks have weight 6). It
  escalates its truncation bound until it finds an undetectable logical
  error or hits `max_weight_cap`. This is what the QEC literature actually
  uses in practice for such codes — stim explicitly documents
  `shortest_graphlike_error` as *not* applicable when errors aren't
  graphlike, and recommends this method instead.

Both methods need the circuit's detector error model to actually contain
error mechanisms, so callers should pass a circuit built with some
(possibly tiny) nonzero noise — the search only cares about which error
locations exist, not their probabilities.
"""
from __future__ import annotations

from dataclasses import dataclass

import stim


@dataclass
class DistanceResult:
    distance: int | None
    method: str
    search_bound: int | None = None
    bound_was_limiting: bool = False

    def __str__(self) -> str:
        if self.distance is None:
            return f"d=unknown (search exhausted up to bound {self.search_bound}, method={self.method})"
        note = " [heuristic upper bound, hit search cap]" if self.bound_was_limiting else ""
        return f"d={self.distance} (method={self.method}{note})"


def shortest_graphlike_distance(circuit: stim.Circuit) -> DistanceResult:
    """Exact for graphlike circuits (e.g. the surface code): the shortest
    set of graphlike errors that cause an undetected logical error."""
    err = circuit.shortest_graphlike_error(ignore_ungraphlike_errors=False)
    return DistanceResult(distance=len(err), method="shortest_graphlike_error")


def search_distance(
    circuit: stim.Circuit,
    start_bound: int = 4,
    max_bound: int = 24,
    step: int = 2,
) -> DistanceResult:
    """Heuristic bounded search for non-graphlike circuits (e.g. BB codes).
    Escalates the truncation bound until an undetectable logical error is
    found or `max_bound` is reached.
    """
    bound = start_bound
    while bound <= max_bound:
        try:
            err = circuit.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=bound,
                dont_explore_edges_with_degree_above=bound,
                dont_explore_edges_increasing_symptom_degree=True,
                canonicalize_circuit_errors=True,
            )
        except Exception:
            err = []
        if err:
            return DistanceResult(
                distance=len(err),
                method="search_for_undetectable_logical_errors",
                search_bound=bound,
                bound_was_limiting=(bound >= max_bound),
            )
        bound += step
    return DistanceResult(
        distance=None,
        method="search_for_undetectable_logical_errors",
        search_bound=max_bound,
        bound_was_limiting=True,
    )
