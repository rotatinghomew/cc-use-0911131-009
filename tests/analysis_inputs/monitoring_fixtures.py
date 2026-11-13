from __future__ import annotations

"""Regenerates the committed sampling-shift-only example.

The fixture changes how records are sampled from 2026-03-19 onwards
(routine/modified 40/40 -> 10/70) while keeping latent review behaviour
constant: the same operators with the same per-stratum acceptance
propensities.  Raw proportions therefore fall, but the population rate does
not.  A test compares the committed file against this generator so the example
cannot silently drift.

Run manually with:
    python3 -m tests.analysis_inputs.monitoring_fixtures
"""

import json
import random
from pathlib import Path
from typing import Any

WEEKS = ["2026-03-05", "2026-03-12", "2026-03-19", "2026-03-26", "2026-04-02", "2026-04-09"]
REFERENCE_WEEKS = {"2026-03-05", "2026-03-12"}
OPERATORS = ["op0", "op1", "op2", "op3"]
SITES = {"op0": "site0", "op1": "site1", "op2": "site0", "op3": "site1"}
FRAME = {"routine": 800, "modified": 200}
BASE_ALLOCATION = {"routine": 40, "modified": 40}
SHIFT_ALLOCATION = {"routine": 10, "modified": 70}
LATENT = {
    "routine": {"op0": 0.9, "op1": 0.9, "op2": 0.8, "op3": 0.8},
    "modified": {"op0": 0.25, "op1": 0.25, "op2": 0.15, "op3": 0.15},
}
SEED = 43
FIXTURE_PATH = Path(__file__).resolve().parents[2] / "fixtures/analysis/sampling_shift_only.json"


def build() -> dict[str, Any]:
    rng = random.Random(SEED)
    samples: list[dict[str, Any]] = []
    frame: list[dict[str, Any]] = []
    for week in WEEKS:
        allocation = BASE_ALLOCATION if week in REFERENCE_WEEKS else SHIFT_ALLOCATION
        for stratum, population in FRAME.items():
            count = allocation[stratum]
            probability = count / population
            frame.append({"week": week, "stratum": stratum, "population": population})
            for index in range(count):
                operator = OPERATORS[index % len(OPERATORS)]
                accepted = rng.random() < LATENT[stratum][operator]
                samples.append(
                    {
                        "sample_id": f"{week}-{stratum}-{index:03d}",
                        "revision": 1,
                        "week": week,
                        "stratum": stratum,
                        "inclusion_probability": round(probability, 6),
                        "operator_id": operator,
                        "site_id": SITES[operator],
                        "version": "assist-v2",
                        "accepted": accepted,
                        "modified": (not accepted) if stratum == "modified" else False,
                        "review_complete": True,
                        "known_at": week,
                    }
                )
    return {
        "seed": SEED,
        "as_of": "2026-04-09",
        "reference_start": "2026-03-05",
        "reference_end": "2026-03-18",
        "reference_version": "ref-b",
        "note": "Sampling proportions change from 2026-03-19 (routine/modified 40/40 -> 10/70). "
        "Latent review behaviour is constant: the same operators with the same per-stratum "
        "acceptance propensities. Raw proportions fall because the mix shifts toward the "
        "low-acceptance stratum; design-weighted estimates must not follow.",
        "samples": samples,
        "sampling_frame": frame,
    }


def write(path: Path = FIXTURE_PATH) -> Path:
    path.write_text(
        json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


if __name__ == "__main__":
    print(write())
