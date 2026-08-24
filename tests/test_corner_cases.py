from qec_dataset.corner_cases import (
    generate_bb_code_corner_cases,
    generate_surface_code_corner_cases,
    load_corner_cases,
    save_corner_cases,
)


def test_surface_code_corner_cases():
    cases = generate_surface_code_corner_cases(distance=3, rounds=3, basis="Z")
    by_name = {c.name: c for c in cases}

    all_zero = by_name["all_zero"]
    assert not any(all_zero.detection_events)
    assert not any(all_zero.observable_flips)

    full = by_name["full_logical_error_0"]
    assert not any(full.detection_events)  # silent logical error: no detector fires
    assert full.observable_flips == [True]

    near_miss = by_name["near_miss_logical_error_0"]
    assert any(near_miss.detection_events)  # disturbs the syndrome...
    assert near_miss.observable_flips == [False]  # ...but is provably not a logical error
    assert len(near_miss.injected_qubits) == len(full.injected_qubits) - 1


def test_bb_code_corner_cases_targeted_observable_guarantee():
    cases = generate_bb_code_corner_cases(
        l=6, m=6,
        a_terms=[("x", 3), ("y", 1), ("y", 2)],
        b_terms=[("y", 3), ("x", 1), ("x", 2)],
        rounds=2, basis="Z",
    )
    by_name = {c.name: c for c in cases}

    all_zero = by_name["all_zero"]
    assert not any(all_zero.detection_events)
    assert not any(all_zero.observable_flips)

    for j in range(12):
        full = by_name[f"full_logical_error_{j}"]
        assert not any(full.detection_events)
        assert full.observable_flips[j] is True

        near_miss = by_name[f"near_miss_logical_error_{j}"]
        # only the targeted observable is provably unflipped -- others may
        # legitimately flip too, since our logical operator representative
        # isn't minimum weight (see corner_cases.py docstring).
        assert near_miss.observable_flips[j] is False
        assert len(near_miss.injected_qubits) == len(full.injected_qubits) - 1


def test_corner_cases_roundtrip(tmp_path):
    cases = generate_surface_code_corner_cases(distance=3, rounds=3, basis="Z")
    path = tmp_path / "cases.json"
    save_corner_cases(cases, path)
    reloaded = load_corner_cases(path)
    assert reloaded == cases
