# QEC Test Vector Generator

![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)

**Built on Stim/Sinter.**

## Purpose

Build surface-code / BB-code (bivariate bicycle) circuits, inject a
circuit-level noise model, and sample the resulting syndrome (detection
event) data together with the ground-truth logical error — as a step
toward eventually feeding this data into an FPGA/ASIC decoder or an RTL
(SystemVerilog/UVM) test environment.

The RTL-specific piece (bit-packed `.mem`/`.hex` export tuned to a chosen
decoder algorithm and bus interface) is **not implemented yet** — see
[docs/rtl_test_vector_roadmap.md](docs/rtl_test_vector_roadmap.md) for
what's done, what's open, and why that part is intentionally deferred.

## Features

1. Customizable / scalable surface-code and BB-code circuits, with `[[n, k, d]]` reporting.
2. Adjustable circuit-level noise model (uniform depolarizing, 4 independently-tunable rates).
3. Golden-answer sampling: each shot's detection events come with the *actual* logical flip that occurred (no decoder needed to know this — it's true by construction from the noise stim injected).
4. Reference decoding (PyMatching for surface codes, BP+OSD for BB codes) as a sanity check on a dataset before scaling it up.

## Installation

This project was developed against the `gnn` conda environment.

```bash
conda activate gnn
cd sim
pip install -r requirements.txt
```

Installs: `stim`, `sinter`, `numpy`, `pytest`, `pymatching`, `stimbposd` (which pulls in `ldpc`).

> **Note:** if your environment also has `panqec` installed, it pins
> `pymatching~=2.0.1`, which conflicts with the `pymatching>=2.2` this
> project needs (sinter's decoder wrapper calls a `decode_batch` method
> that doesn't exist in pymatching 2.0.x). Upgrading pymatching is safe
> for this project but may affect other work relying on panqec's pinned
> version — check before doing so in a shared environment.

## Package layout

```
qec_dataset/
  gf2.py              GF(2) linear algebra (rank, nullspace, logical operator pairing)
  css_code.py         CSSCode: Hx/Hz -> n, k, logical_x, logical_z
  noise.py            NoiseModel (uniform depolarizing, 4 rates)
  extract.py          Recovers Hx/Hz directly from a stim circuit (cross-checks circuit construction)
  codes/
    surface.py        Rotated surface code (wraps stim.Circuit.generated)
    bb.py              BB code generic constructor (cyclic-shift-matrix based)
  circuits.py         Generic CSS syndrome-extraction circuit builder (used for BB codes)
  distance.py         Code distance: exact (surface) / heuristic search (BB)
  report.py           [[n, k, d]] + circuit stats reporting
  sampling.py         Dataset generation: detection events + ground-truth logical flips
  reference_decode.py Sanity-check decoding via sinter (pymatching / BP+OSD)
  corner_cases.py     Directed test vectors: deterministic Pauli errors (all-zero / full logical error / near-miss)
  code_capacity.py    Single-round (code-capacity) samples + per-qubit Pauli labels, for GNN decoder training
  astra_adapter.py    Presents a CSSCode under the attribute names the astra/ GNN pipeline expects
  cli.py              Command-line entry point
```

## CLI usage

Print `[[n, k, d]]` and circuit stats, optionally writing the circuit to a `.stim` file:

```bash
python -m qec_dataset.cli surface --distance 5 --rounds 5 --basis Z --p 0.001 --out surface_d5.stim

python -m qec_dataset.cli bb --l 6 --m 6 --A x3,y1,y2 --B y3,x1,x2 --rounds 6 --basis Z --p 0.005 --out bb_6_6.stim
```

`--A`/`--B` are comma-separated monomials (`x3` = x³, `y1` = y¹, ...). There
are no built-in literature presets for BB codes yet (see roadmap doc) — you
supply `l`, `m`, and the two polynomials directly. `l=m=6`,
`A=x3,y1,y2`, `B=y3,x1,x2` reproduces the `[[72, 12, 6]]` code.

## Python API

### Build a code and inspect [[n, k, d]]

```python
from qec_dataset.codes.surface import rotated_surface_code
from qec_dataset.codes.bb import build_bb_code

code, _ = rotated_surface_code(distance=5)
print(code.n, code.k)  # 25 1

bb = build_bb_code(l=6, m=6, a_terms=[("x", 3), ("y", 1), ("y", 2)],
                    b_terms=[("y", 3), ("x", 1), ("x", 2)])
print(bb.n, bb.k)  # 72 12
```

### Build a noisy circuit

```python
from qec_dataset.codes.surface import build_surface_code_circuit
from qec_dataset.circuits import generic_css_memory_circuit
from qec_dataset.noise import NoiseModel

noise = NoiseModel.uniform(0.01)  # or set the 4 rates independently
surface_circuit = build_surface_code_circuit(distance=5, rounds=5, basis="Z", noise=noise)
bb_circuit = generic_css_memory_circuit(bb, rounds=6, basis="Z", noise=noise)
```

### Generate a dataset (detection events + ground truth)

```python
from qec_dataset.sampling import generate_surface_code_dataset, generate_bb_code_dataset, load_dataset

meta = generate_surface_code_dataset(
    distance=5, rounds=5, basis="Z", noise=NoiseModel.uniform(0.01),
    shots=10_000, out_dir="datasets/surface_d5",
)
dets, obs = load_dataset(meta, "datasets/surface_d5")
# dets: bool array (shots, num_detectors) — the syndrome
# obs:  bool array (shots, num_observables) — the ACTUAL logical flip (ground truth)

meta_bb = generate_bb_code_dataset(
    l=6, m=6, a_terms=[("x", 3), ("y", 1), ("y", 2)], b_terms=[("y", 3), ("x", 1), ("x", 2)],
    rounds=6, basis="Z", noise=NoiseModel.uniform(0.005),
    shots=10_000, out_dir="datasets/bb_72_12",
)
```

Each call writes `<prefix>.dets.b8`, `<prefix>.obs.b8`, and a
`<prefix>.meta.json` sidecar (code/noise/format metadata) into `out_dir`.
`load_dataset` reads them back into numpy arrays via `stim.read_shot_data_file`.

### Sanity-check with a reference decoder

```python
from qec_dataset.reference_decode import reference_decode_stats

# surface codes: MWPM (fast, exact for graphlike circuits)
result = reference_decode_stats(surface_circuit, shots=5000, decoder="pymatching")

# BB codes: MWPM does NOT apply (weight-6 checks aren't graphlike — it will
# raise). Use BP+OSD instead:
result_bb = reference_decode_stats(bb_circuit, shots=200, decoder="bposd-fast")

print(result.logical_error_rate)
```

`decoder="bposd-fast"` uses `osd_order=10` for a quick sanity check.
`stimbposd`'s own presets (`"bposd"`, `"bposd-serial"`, ...) use
`osd_order=60`, which is far more accurate but can take well over a second
per shot on a code like `[[72, 12, 6]]` — expect minutes to hours, not
seconds, if you use those on more than a handful of shots.

This reference decoder is **not** the eventual RTL decoder (still undecided,
see the roadmap doc) and does **not** define the dataset's ground truth —
it only exists to catch "this circuit/noise combination doesn't produce a
sensible error rate" before spending time generating a large dataset.

### Directed corner cases

Deterministic (not randomly sampled) test vectors — a single reproducible
shot per case, since there's no randomness left once a specific error is
injected into a noiseless circuit:

```python
from qec_dataset.corner_cases import generate_surface_code_corner_cases, generate_bb_code_corner_cases

cases = generate_surface_code_corner_cases(distance=5, rounds=5, basis="Z")
for c in cases:
    print(c.name, c.detection_events, c.observable_flips)
```

Three kinds are generated per logical qubit `j`:
- `all_zero`: no injected error — everything reads 0.
- `full_logical_error_j`: the full support of the operator anticommuting
  with logical qubit `j`'s observable — a genuine logical error with an
  **all-zero syndrome** (no detector fires), the case a syndrome-only
  decoder can never catch.
- `near_miss_logical_error_j`: the same support with one qubit removed —
  provably does **not** flip observable `j` (the removed qubit breaks an
  odd-overlap invariant), while usually still tripping some detectors —
  a "don't overreact" input.

Caveat for k > 1 codes (BB codes): only `observable_flips[j]` is provably
unflipped in the near-miss case — other logical qubits' observables can
also flip, because our logical operator representative isn't guaranteed
to be minimum-weight. See
[docs/rtl_test_vector_roadmap.md](docs/rtl_test_vector_roadmap.md) for
details.

### Code-capacity datasets (for GNN decoder training)

The three sections above all use **circuit-level** noise (multi-round
syndrome extraction, measurement error). A supervised decoder that wants
per-qubit error labels on a single perfect syndrome round needs the
**code-capacity** model instead, which `code_capacity.py` provides
without going through a stim circuit at all:

```python
from qec_dataset.code_capacity import (
    generate_bb_code_capacity_samples, error_index, tanner_graph_edges,
)

samples, code = generate_bb_code_capacity_samples(
    l=6, m=6, a_terms=[("x", 3), ("y", 1), ("y", 2)], b_terms=[("y", 3), ("x", 1), ("x", 2)],
    shots=100_000, p=0.05,
)
# samples: (shots, num_checks + n) uint8
#   [:, :error_index(code)] -> syndrome (Z-checks as 1, X-checks as 2)
#   [:, error_index(code):] -> per-qubit ground truth (0=I, 1=X, 2=Z, 3=Y)
src_ids, dst_ids = tanner_graph_edges(code)  # Tanner graph, checks then qubits
```

Literature BB code parameters (Bravyi et al. 2024) are available as presets:

```python
from qec_dataset.codes.bb import build_bb_preset, BB_PRESETS
code = build_bb_preset(12)   # [[144, 12, 12]]
print({d: p.label for d, p in BB_PRESETS.items()})
```

`n` and `k` are verified when a preset is built; the distances are taken
from the source and **not** independently verified here (d=24 and d=34 are
recorded there as upper bounds).

### Handing off to the astra GNN pipeline

`astra_adapter.AstraCodeView` wraps a `CSSCode` in the attribute names
that pipeline reads (`N`, `hx`, `hz`, `hx_perp`, `hz_perp`, ...), where
`hx_perp`/`hz_perp` are the GF(2) kernels of Hx/Hz:

```python
from qec_dataset.astra_adapter import AstraCodeView
view = AstraCodeView(build_bb_preset(6), distance=6)
trainset = adapt_trainset(samples, view, num_classes=4)   # astra's function
```

Verified against that pipeline in `tests/test_astra_integration.py`: the
BB preset Hx/Hz are bit-identical to its `codes_q` implementation's, our
`hx_perp`/`hz_perp` span the same space as its `kernel()`, and
`tanner_graph_edges` reproduces its `surface_code_edges` edge set exactly —
so matrices and checkpoints are interchangeable.

## Testing

```bash
pytest tests/ -q
```

39 tests across `test_gf2.py`, `test_surface.py`, `test_bb.py`,
`test_sampling.py`, `test_reference_decode.py`, `test_corner_cases.py`,
`test_code_capacity.py`, and `test_astra_integration.py` (the last is
skipped unless the astra pipeline is present; set `ASTRA_PATH` if it isn't
at `~/astra`). See
[docs/testing_overview.md](docs/testing_overview.md) for what each test
actually checks.

## What's next

See [docs/rtl_test_vector_roadmap.md](docs/rtl_test_vector_roadmap.md) for
the RTL-facing gap analysis, the Layer A (done) / Layer B (blocked on
decoder + interface decisions) split, and open decisions.
