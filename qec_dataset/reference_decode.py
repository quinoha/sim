"""Sanity-check decoding via sinter, using pymatching (MWPM) for graphlike
circuits (surface codes) and BP+OSD (via the `stimbposd`/`ldpc` packages)
for non-graphlike circuits (BB codes, whose weight-6 checks don't reduce to
a plain graph-matching problem — MWPM simply fails to find a matching for
them).

This is NOT the RTL decoder (that choice is still open, see
docs/rtl_test_vector_roadmap.md) and it does not define the dataset's
ground truth (sampling.py's observable flips already are the ground
truth, known directly from the noise stim injected). This module exists
only to sanity-check that a generated circuit/noise combination produces
a sensible, nonzero logical error rate with a standard reference decoder,
before spending time turning it into a big dataset.
"""
from __future__ import annotations

from dataclasses import dataclass

import sinter
import stim
import stimbposd
from stimbposd.sinter_bp_osd import SinterDecoder_BPOSD

# "pymatching" is graphlike-only (surface codes); the "bposd" family (from
# stimbposd, backed by the `ldpc` package's belief-propagation + ordered
# statistics decoding) also handles the weight-6 non-graphlike checks BB
# codes produce.
#
# stimbposd's own "bposd" preset uses osd_order=60 (osd_cs method), which is
# fine for accuracy but measured at ~1s/shot even with osd_order=10 on a
# [[72,12,6]] code with 288 detectors — osd_order=60 was still not finished
# after 9 minutes for 500 shots in testing. "bposd-fast" trades order for
# speed and is the default for sanity-checking; use "bposd" (or build your
# own `SinterDecoder_BPOSD(osd_order=...)`) when decode quality matters more
# than turnaround time.
_CUSTOM_DECODERS = stimbposd.sinter_decoders()
_CUSTOM_DECODERS["bposd-fast"] = SinterDecoder_BPOSD(osd_order=10)


@dataclass
class ReferenceDecodeResult:
    shots: int
    errors: int
    decoder: str

    @property
    def logical_error_rate(self) -> float:
        return self.errors / self.shots if self.shots else float("nan")

    def __str__(self) -> str:
        return (
            f"decoder={self.decoder}: {self.errors}/{self.shots} logical errors "
            f"(rate={self.logical_error_rate:.3e})"
        )


def reference_decode_stats(
    circuit: stim.Circuit,
    shots: int,
    decoder: str = "pymatching",
    num_workers: int = 1,
) -> ReferenceDecodeResult:
    """Runs `decoder` (via sinter) on freshly-sampled shots from `circuit`
    and reports the logical error rate.

    `decoder="pymatching"`: minimum-weight matching. Only meaningful for
    graphlike circuits (surface codes) — raises on non-graphlike circuits
    (e.g. BB codes) instead of silently giving a nonsensical answer.

    `decoder="bposd-fast"` (osd_order=10): belief-propagation + ordered-
    statistics decoding via `stimbposd`/`ldpc`, tuned for a quick sanity
    check. Works for both graphlike and non-graphlike circuits; use this
    for BB codes. `"bposd"` (and its "-serial"/"-minsum" variants) use
    stimbposd's default osd_order=60, which is far more accurate but can
    take well over a second per shot on codes like [[72,12,6]] — expect
    minutes to hours, not seconds, if you reach for those on anything but
    a handful of shots.
    """
    task = sinter.Task(circuit=circuit, decoder=decoder)
    (stats,) = sinter.collect(
        num_workers=num_workers,
        tasks=[task],
        max_shots=shots,
        custom_decoders=_CUSTOM_DECODERS,
    )
    return ReferenceDecodeResult(shots=stats.shots, errors=stats.errors, decoder=decoder)
