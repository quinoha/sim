"""Weight-resolved decoder profiling: where does a decoder stop correcting?

The LER sweeps answer "how often does this decoder fail at physical error rate
p". They cannot answer "at what error weight does it fail", because a sampled
run mixes every weight together and the failures are dominated by whatever
weight happens to carry the probability mass. This module separates the two by
conditioning on the weight of the error that actually occurred.

Weight here means **the number of independent fault mechanisms in the circuit's
detector error model that fired**, not the number of data qubits carrying a
Pauli. That choice is deliberate:

* It is native to everything already built. The DEM's detectors *are* the
  circuit's detectors, so a weight-w error set produces a syndrome that every
  decoder in this repo -- pymatching, BP-OSD, and the Cascade adapter with its
  (T, G, G) volume -- consumes unchanged. The code-capacity alternative
  (count flipped data qubits, one perfect round) would have to be embedded into
  Cascade's multi-layer input somehow, and the result would measure the
  embedding hack as much as the decoder.
* It keeps the intuition intact. For a distance-d rotated surface code memory
  experiment the circuit distance is also d, so a decoder worth the name
  corrects every weight <= (d-1)//2 and the first interesting weight is
  (d+1)//2 -- weight 3 at d=5, exactly where the question points.

The caveat to state plainly: stim merges fault mechanisms that produce
identical symptoms into a single DEM instruction, so "weight" counts merged
mechanisms, not raw circuit fault locations. For a surface-code memory circuit
the two agree closely; for a code with many symmetric fault paths they do not.

Two sampling modes, answering different questions:

`natural`
    Sample the DEM as physics would, ask stim which mechanisms fired, and bin
    the shots by realized weight. This is the exact conditional distribution at
    the circuit's own p -- it tells you which weights the real LER budget is
    actually spent on. It only reaches weights that carry probability mass; at
    p=1e-3 that runs out after a handful.

`uniform`
    Choose exactly w mechanisms uniformly at random and XOR their symptoms.
    This reaches any weight with any statistics you ask for, which is what a
    collapse curve needs. It is *not* the physical distribution -- it weights a
    rare mechanism the same as a common one. Its useful side effect is that the
    test set no longer depends on p at all, so decoders calibrated at different
    error rates can be compared on bit-identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import stim


@dataclass
class FaultIncidence:
    """A detector error model as two incidence matrices over its mechanisms.

    `H[i]` is the detector set of mechanism i, `L[i]` its observable set, both
    as 0/1 rows. An error set is then just a row selection, and its syndrome is
    the XOR of the selected rows.
    """

    H: np.ndarray  # (M, D) uint8
    L: np.ndarray  # (M, K) uint8
    probs: np.ndarray  # (M,) float64
    num_detectors: int
    num_observables: int

    @property
    def num_mechanisms(self) -> int:
        return int(self.H.shape[0])


def fault_incidence(dem: stim.DetectorErrorModel) -> FaultIncidence:
    """Flattens a DEM into per-mechanism detector/observable incidence rows.

    Pass a DEM built with `decompose_errors=False`: decomposition splits one
    mechanism into graphlike pieces separated by `^`, which would make a single
    physical fault count as several units of weight.
    """
    num_detectors = dem.num_detectors
    num_observables = dem.num_observables

    det_rows: list[np.ndarray] = []
    obs_rows: list[np.ndarray] = []
    probs: list[float] = []

    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        det = np.zeros(num_detectors, dtype=np.uint8)
        obs = np.zeros(num_observables, dtype=np.uint8)
        for target in instruction.targets_copy():
            if target.is_separator():
                raise ValueError(
                    "this DEM contains decomposed errors ('^' separators); build it with "
                    "decompose_errors=False so one fault mechanism counts as weight 1"
                )
            if target.is_relative_detector_id():
                det[target.val] ^= 1  # XOR, so a repeated target cancels as it should
            elif target.is_logical_observable_id():
                obs[target.val] ^= 1
        det_rows.append(det)
        obs_rows.append(obs)
        probs.append(float(instruction.args_copy()[0]))

    if not det_rows:
        raise ValueError("detector error model contains no error mechanisms")

    return FaultIncidence(
        H=np.array(det_rows, dtype=np.uint8),
        L=np.array(obs_rows, dtype=np.uint8),
        probs=np.array(probs, dtype=np.float64),
        num_detectors=num_detectors,
        num_observables=num_observables,
    )


def sample_weight_sets(
    num_mechanisms: int, weight: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    """`count` uniformly random distinct-`weight` subsets, as an (count, weight) index array.

    Drawn with replacement and resampled row-wise until every row is distinct.
    Conditioning i.i.d. draws on being all-distinct is uniform over ordered
    tuples of distinct elements, hence uniform over subsets -- and with
    weight << num_mechanisms the rejection rate is a few percent.
    """
    if weight < 0:
        raise ValueError(f"weight must be non-negative, got {weight}")
    if weight > num_mechanisms:
        raise ValueError(
            f"cannot draw {weight} distinct mechanisms from a model with {num_mechanisms}"
        )
    if weight == 0 or count == 0:
        return np.zeros((count, weight), dtype=np.int64)

    sel = rng.integers(0, num_mechanisms, size=(count, weight), dtype=np.int64)
    if weight == 1:
        return sel

    for _ in range(1000):
        ordered = np.sort(sel, axis=1)
        repeated = np.any(ordered[:, 1:] == ordered[:, :-1], axis=1)
        if not repeated.any():
            return sel
        sel[repeated] = rng.integers(
            0, num_mechanisms, size=(int(repeated.sum()), weight), dtype=np.int64
        )
    raise RuntimeError(
        f"could not draw {count} distinct {weight}-subsets from {num_mechanisms} mechanisms; "
        "the weight is probably too close to the mechanism count"
    )


def symptoms(inc: FaultIncidence, sel: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Syndromes and true observable flips for each error set in `sel`.

    Returns `(dets, obs)` shaped `(len(sel), D)` and `(len(sel), K)`, uint8.
    Chunked because the intermediate gather is (chunk, weight, D).
    """
    count, weight = sel.shape
    dets = np.zeros((count, inc.num_detectors), dtype=np.uint8)
    obs = np.zeros((count, inc.num_observables), dtype=np.uint8)
    if weight == 0:
        return dets, obs

    for start in range(0, count, chunk):
        rows = sel[start : start + chunk]
        dets[start : start + chunk] = np.bitwise_xor.reduce(inc.H[rows], axis=1)
        obs[start : start + chunk] = np.bitwise_xor.reduce(inc.L[rows], axis=1)
    return dets, obs


def sample_with_errors(
    dem: stim.DetectorErrorModel, shots: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Samples the DEM and asks which mechanisms fired on each shot.

    `return_errors` is what makes weight observable at all -- without it a shot
    is just a syndrome, with no record of what produced it.
    """
    sampler = dem.compile_sampler(seed=seed)
    try:
        dets, obs, errors = sampler.sample(shots, return_errors=True)
    except TypeError as e:
        raise RuntimeError(
            "this stim's CompiledDemSampler.sample does not accept return_errors, which "
            "this module needs to know each shot's error weight. Upgrade stim "
            f"(requirements.txt asks for >=1.16). Original error: {e}"
        ) from e
    if errors is None:
        raise RuntimeError("stim returned no per-mechanism error record despite return_errors=True")
    return dets, obs, errors


def natural_sample(
    dem: stim.DetectorErrorModel, shots: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Samples the DEM at its own probabilities. Returns (dets, obs, weights)."""
    dets, obs, errors = sample_with_errors(dem, shots, seed)
    return (
        dets.astype(np.uint8),
        obs.astype(np.uint8),
        errors.sum(axis=1).astype(np.int64),
    )


def verify_incidence(
    dem: stim.DetectorErrorModel, inc: FaultIncidence, shots: int = 256, seed: int = 0
) -> None:
    """Checks the incidence matrices against stim's own sampler.

    Everything downstream rests on the claim that XOR-ing the rows of `H` for
    the mechanisms that fired reproduces the syndrome stim would have reported.
    That claim is one target-parsing mistake away from being false, and a
    silently wrong syndrome would look like a decoder result rather than a bug,
    so it is checked rather than assumed.
    """
    dets, obs, errors = sample_with_errors(dem, shots, seed)
    fired = errors.astype(np.uint8)
    if fired.shape[1] != inc.num_mechanisms:
        raise AssertionError(
            f"stim reports {fired.shape[1]} mechanisms, incidence has {inc.num_mechanisms}"
        )
    rebuilt_dets = (fired @ inc.H.astype(np.int64)) % 2
    rebuilt_obs = (fired @ inc.L.astype(np.int64)) % 2

    if not np.array_equal(rebuilt_dets.astype(np.uint8), dets.astype(np.uint8)):
        bad = int(np.argmax(np.any(rebuilt_dets != dets, axis=1)))
        raise AssertionError(f"detector incidence disagrees with stim (first bad shot {bad})")
    if not np.array_equal(rebuilt_obs.astype(np.uint8), obs.astype(np.uint8)):
        bad = int(np.argmax(np.any(rebuilt_obs != obs, axis=1)))
        raise AssertionError(f"observable incidence disagrees with stim (first bad shot {bad})")


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Unlike the normal approximation it stays inside
    [0, 1] and remains meaningful at 0 or 100% failures, both of which this
    profile hits routinely at the ends of the weight range."""
    if trials == 0:
        return (float("nan"), float("nan"))
    phat = successes / trials
    denom = 1 + z * z / trials
    centre = (phat + z * z / (2 * trials)) / denom
    half = z * np.sqrt(phat * (1 - phat) / trials + z * z / (4 * trials * trials)) / denom
    # Plain floats, not np.float64: these end up in json.dumps, which cannot
    # serialize numpy scalars.
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


@dataclass
class WeightStats:
    """One decoder's behaviour on one weight bucket.

    `ler` alone cannot tell a decoder that has degraded gracefully from one that
    has started guessing, so the conditional splits are kept separately:

    * `false_alarm` -- among error sets that did NOT flip a logical observable,
      how often the decoder claimed one. Goes to ~0 for a decoder that has
      collapsed to always predicting "no flip", to ~0.5 for one guessing.
    * `miss` -- among error sets that DID flip one, how often the decoder missed
      it. Goes to ~1 for the collapsed decoder, ~0.5 for the guessing one.
    * `pred_flip_rate` vs `truth_flip_rate` -- a decoder still carrying
      information tracks the truth rate; one that has given up does not.
    """

    weight: int
    n: int
    errors: int
    ler: float
    ci: tuple[float, float]
    truth_flip_rate: float
    pred_flip_rate: float
    false_alarm: float
    miss: float
    extras: dict = field(default_factory=dict)


def bucket_stats(weight: int, predictions: np.ndarray, truth: np.ndarray) -> WeightStats:
    predictions = np.asarray(predictions)
    if predictions.shape != truth.shape:
        raise ValueError(
            f"decoder returned predictions of shape {predictions.shape}, expected "
            f"{truth.shape}; a mismatched shape would broadcast and silently corrupt "
            "the error count"
        )
    n = int(truth.shape[0])
    wrong = np.any(predictions != truth, axis=1)
    flipped = np.any(truth != 0, axis=1)
    predicted = np.any(predictions != 0, axis=1)
    errors = int(wrong.sum())

    return WeightStats(
        weight=weight,
        n=n,
        errors=errors,
        ler=errors / n if n else float("nan"),
        ci=wilson_interval(errors, n),
        truth_flip_rate=float(flipped.mean()) if n else float("nan"),
        pred_flip_rate=float(predicted.mean()) if n else float("nan"),
        false_alarm=float(wrong[~flipped].mean()) if (~flipped).any() else float("nan"),
        miss=float(wrong[flipped].mean()) if flipped.any() else float("nan"),
    )


def random_guess_ler(num_observables: int) -> float:
    """LER of a decoder that guesses every observable by coin flip: it has to
    get all K right, so 1 - 2^-K. The ceiling any real decoder must stay under."""
    return 1.0 - 2.0 ** (-num_observables)
