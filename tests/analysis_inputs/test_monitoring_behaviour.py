from __future__ import annotations

"""Behavioural tests for the monitoring decisions.

These tests are the acceptance story from the quality meeting:

* more sampling of already-edited records must not look like a quality drop;
* a revised review replaces its earlier conclusion instead of adding evidence;
* re-running the same week must not keep minting alerts;
* a genuine sustained change in review behaviour is the only thing called a
  supported shift.
"""

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from alignment_observatory.monitoring import MonitoringStore, run_monitoring

REFERENCE = ["2026-01-01", "2026-01-08"]
POST = ["2026-01-15", "2026-01-22", "2026-01-29", "2026-02-05"]
WEEKS = REFERENCE + POST
OPERATORS = {
    f"op-{index:02d}": "site-1" if index % 2 == 0 else "site-2" for index in range(8)
}
FRAME = {"routine": 800, "modified": 200}
PROBABILITY = {"routine": 0.05, "modified": 0.2}


def make_row(
    sample_id,
    week,
    stratum,
    operator,
    accepted,
    *,
    complete=True,
    revision=1,
    version="ref-a",
    probability=None,
    known_at=None,
    modified=None,
):
    return {
        "sample_id": sample_id,
        "revision": revision,
        "week": week,
        "stratum": stratum,
        "inclusion_probability": probability if probability is not None else PROBABILITY[stratum],
        "operator_id": operator,
        "site_id": OPERATORS.get(operator, "site-1"),
        "version": version,
        "accepted": accepted,
        "modified": (not accepted) if modified is None else modified,
        "review_complete": complete,
        "known_at": known_at or week,
    }


def build(
    *,
    routine_accept=0.9,
    modified_accept=0.2,
    rate_overrides=None,
    allocation=None,
    probabilities=None,
    incomplete=None,
    weeks=WEEKS,
    seed=11,
    operator_effect=0.08,
):
    """Deterministic rows with per-stratum acceptance.

    ``allocation`` maps week -> {stratum: count}; it only changes the sample
    design.  Acceptance probabilities define latent behaviour and default to
    the same values in every week, i.e. no quality change at all.  A seeded
    RNG supplies the record-level variation, and ``operator_effect`` gives
    operators slightly different propensities so the operator-clustered
    variance is genuinely non-zero (as in real review data).
    """
    import random

    allocation = allocation or {}
    probabilities = probabilities or {}
    incomplete = incomplete or {}
    rate_overrides = rate_overrides or {}
    operators = sorted(OPERATORS)
    rng = random.Random(seed)
    operator_rates = {
        operator: operator_effect * ((position % 2) * 2 - 1)
        for position, operator in enumerate(operators)
    }
    rows = []
    for week in weeks:
        counts = allocation.get(week) or {"routine": 40, "modified": 40}
        for stratum, count in counts.items():
            base_rate = rate_overrides.get(
                (week, stratum),
                routine_accept if stratum == "routine" else modified_accept,
            )
            for index in range(count):
                operator = operators[index % len(operators)]
                rate = base_rate + operator_rates[operator]
                accepted = rng.random() < max(0.0, min(1.0, rate))
                probability = probabilities.get((week, stratum), PROBABILITY[stratum])
                rows.append(
                    make_row(
                        f"{week}-{stratum}-{index:04d}",
                        week,
                        stratum,
                        operator,
                        accepted,
                        probability=probability,
                        complete=index >= incomplete.get((week, stratum), 0),
                    )
                )
    frame = [
        {"week": week, "stratum": stratum, "population": population}
        for week in weeks
        for stratum, population in FRAME.items()
    ]
    return {
        "seed": seed,
        "as_of": weeks[-1],
        "reference_start": "2026-01-01",
        "reference_end": "2026-01-14",
        "reference_version": "ref-a",
        "samples": rows,
        "sampling_frame": frame,
    }


class MonitoringBehaviourTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = str(Path(self.directory.name) / "monitoring.sqlite3")

    def estimates(self, result, metric, weeks=None):
        return [
            estimate
            for estimate in result["estimates"]
            if estimate["metric"] == metric and (weeks is None or estimate["week"] in weeks)
        ]

    def test_sampling_proportion_change_alone_does_not_look_like_quality_drop(self):
        # Before the change every week is sampled 40/40.  From 2026-01-15 the
        # team deliberately over-samples the "modified" stratum: 10 routine and
        # 70 modified records per week.  Latent acceptance is identical
        # throughout, so the population rate must not move.
        # The declared design changes with the allocation: 10/800 and 70/200.
        allocation = {
            week: ({"routine": 40, "modified": 40} if week in REFERENCE else {"routine": 10, "modified": 70})
            for week in WEEKS
        }
        probabilities = {
            (week, stratum): count / FRAME[stratum]
            for week, counts in allocation.items()
            for stratum, count in counts.items()
        }
        data = build(allocation=allocation, probabilities=probabilities)
        result = run_monitoring(data, self.database)

        adoption = self.estimates(result, "adoption_rate")
        reference_estimate = result["reference"]["metrics"]["adoption_rate"]["pooled"]["estimate"]
        for estimate in adoption:
            self.assertEqual(estimate["status"], "ok")
            # Deliberately drawing more "modified" records makes the raw
            # proportion fall sharply.  The design-weighted estimate must stay
            # in the same region as the population value.
            self.assertLess(abs(estimate["estimate"] - reference_estimate), 0.20)
        # The raw count ratio for the same week is far below both, which is
        # exactly the mistake this module exists to avoid.
        week = "2026-02-05"
        rows = [row for row in data["samples"] if row["week"] == week]
        naive = sum(row["accepted"] for row in rows) / len(rows)
        self.assertLess(naive, reference_estimate - 0.30)

        self.assertFalse(
            any(alert["decision"] == "supported_shift" for alert in result["alerts"]),
            "a pure sampling change must never be reported as a quality regression",
        )
        design_reasons = {
            alert["reason"] for alert in result["alerts"] if alert["decision"] == "design_shift"
        }
        self.assertIn("sampling_structure_change", design_reasons)
        checks = {(check["week"], check["check"]) for check in result["design_checks"]}
        self.assertTrue(
            any(check in {"declared_sample_size", "stratum_allocation_shift"} for _, check in checks)
        )

    def test_naive_count_ratio_would_be_misleading(self):
        allocation = {
            week: ({"routine": 40, "modified": 40} if week in REFERENCE else {"routine": 10, "modified": 70})
            for week in WEEKS
        }
        probabilities = {
            (week, stratum): count / FRAME[stratum]
            for week, counts in allocation.items()
            for stratum, count in counts.items()
        }
        data = build(allocation=allocation, probabilities=probabilities)
        result = run_monitoring(data, self.database)
        week = "2026-01-22"
        rows = [row for row in data["samples"] if row["week"] == week]
        naive = sum(row["accepted"] for row in rows) / len(rows)
        weighted = next(
            estimate["estimate"]
            for estimate in self.estimates(result, "adoption_rate", {week})
        )
        self.assertGreater(abs(naive - weighted), 0.10)

    def test_coverage_is_not_used_as_the_adoption_denominator(self):
        # 8 routine records per week are sampled but never reviewed.  Counting
        # them as "not adopted" would drag adoption down; they belong in the
        # coverage metric instead.
        incomplete = {(week, "routine"): 8 for week in WEEKS}
        data = build(incomplete=incomplete)
        result = run_monitoring(data, self.database)
        adoption = self.estimates(result, "adoption_rate")
        coverage = self.estimates(result, "coverage_rate")
        for estimate in adoption:
            self.assertEqual(estimate["n"], 80)
            self.assertEqual(estimate["completed"], 72)
            self.assertGreaterEqual(estimate["estimate"], 0.60)
        for estimate in coverage:
            self.assertLess(estimate["estimate"], 1.0)
            self.assertGreater(estimate["estimate"], 0.0)
            self.assertEqual(estimate["n"], 80)
            self.assertEqual(estimate["completed"], 72)

    def test_sustained_behaviour_change_is_a_supported_shift(self):
        # Reference behaviour: routine accepts ~90%, modified ~20%.  From
        # 2026-01-15 routine acceptance drops to ~20% and stays there.  This is
        # genuine review behaviour change, sustained across consecutive weeks.
        data = build(rate_overrides={(week, "routine"): 0.2 for week in POST})
        result = run_monitoring(data, self.database)

        supported = [alert for alert in result["alerts"] if alert["decision"] == "supported_shift"]
        self.assertTrue(supported)
        alert = supported[0]
        self.assertIn("supported_behaviour_shift", alert["reason"])
        self.assertEqual(alert["category"], "quality_signal")
        self.assertEqual(len(alert["episode_weeks"]), 2)
        self.assertLess(alert["reference_comparison"]["difference"], 0)
        # The first week of the shift is not yet confirmed; it is evidence
        # insufficient rather than a regression finding.
        first_week = [
            item for item in result["alerts"] if item["week"] == "2026-01-15"
        ]
        self.assertTrue(first_week)
        self.assertTrue(all(item["decision"] == "insufficient_evidence" for item in first_week))

        # A sustained shift does not produce a fresh alert every week.  The
        # later weeks continue the open episode and point back to its opener.
        adoption = [
            item for item in result["estimates"] if item["metric"] == "adoption_rate"
        ]
        opener = next(item for item in adoption if item["week"] == "2026-01-22")
        later = [item for item in adoption if item["week"] in {"2026-01-29", "2026-02-05"}]
        self.assertTrue(all(item["continued"] for item in later))
        self.assertEqual(
            {item["episode_alert_id"] for item in later}, {opener["episode_alert_id"]}
        )
        self.assertIsNotNone(opener["episode_alert_id"])
        self.assertEqual(
            len([item for item in result["alerts"] if item["metric"] == "adoption_rate"]),
            2,
            "one unconfirmed-week alert and one supported-shift alert, not one per week",
        )

    def test_one_week_blip_is_not_confirmed(self):
        # Only a single week moves; the following week returns to baseline.
        # This must be reported as evidence insufficient, not as a regression.
        data = build(rate_overrides={("2026-01-15", "routine"): 0.2})
        result = run_monitoring(data, self.database)
        self.assertFalse(
            any(alert["decision"] == "supported_shift" for alert in result["alerts"])
        )
        unconfirmed = [
            alert for alert in result["alerts"] if alert["reason"] == "shift_not_confirmed"
        ]
        self.assertTrue(unconfirmed)
        self.assertTrue(all(alert["category"] == "evidence_signal" for alert in unconfirmed))

    def test_thin_week_is_insufficient_evidence_not_zero(self):
        allocation = {
            week: ({"routine": 40, "modified": 40} if week in REFERENCE else {"routine": 2, "modified": 2})
            for week in WEEKS
        }
        result = run_monitoring(build(allocation=allocation), self.database)
        thin = self.estimates(result, "adoption_rate", {"2026-01-15"})
        self.assertEqual(len(thin), 1)
        self.assertEqual(thin[0]["status"], "insufficient_evidence")
        self.assertIsNone(thin[0]["interval"])
        self.assertIsNotNone(thin[0]["estimate"])

    def test_unknown_sampling_stratum_is_unestimable_not_dropped(self):
        data = build()
        for row in data["samples"]:
            if row["week"] == "2026-01-22" and row["stratum"] == "modified":
                row["stratum"] = "edited-after-review"
        result = run_monitoring(data, self.database)
        affected = self.estimates(result, "adoption_rate", {"2026-01-22"})
        self.assertEqual(len(affected), 1)
        self.assertEqual(affected[0]["status"], "unestimable")
        self.assertIsNone(affected[0]["estimate"])
        self.assertEqual(affected[0]["reason"], "sample_stratum_not_in_frame")

    def test_revision_supersedes_instead_of_accumulating_evidence(self):
        data = build(rate_overrides={(week, "routine"): 0.2 for week in POST})
        first = run_monitoring(data, self.database)
        self.assertTrue(first["alerts"])

        # A late review corrects the 2026-01-22 conclusion: the routine
        # records of that week are now accepted.  Only that week changes.
        revised = copy.deepcopy(data)
        for row in revised["samples"]:
            if row["week"] == "2026-01-22" and row["stratum"] == "routine":
                row["revision"] = 2
                row["accepted"] = True
                row["modified"] = False
                row["known_at"] = "2026-01-29"
        revised["as_of"] = "2026-02-01"
        second = run_monitoring(revised, self.database)

        week_alerts = [
            alert for alert in second["alerts"] if alert["week"] == "2026-01-22"
        ]
        open_alerts = [alert for alert in week_alerts if alert["status"] == "open"]
        superseded = [alert for alert in week_alerts if alert["status"] == "superseded"]
        self.assertTrue(superseded, "the earlier conclusion must be kept but marked superseded")
        # The revision withdraws the week's supported-shift finding; it does not
        # leave a stale finding open, and it does not create a duplicate.
        self.assertFalse(
            any(
                alert["decision"] == "supported_shift"
                for alert in second["alerts"]
                if alert["week"] == "2026-01-22" and alert["status"] == "open"
            )
        )
        self.assertTrue(
            all(alert["week"] == "2026-01-22" for alert in superseded)
        )
        self.assertEqual(len(week_alerts), 2)  # one per metric, no accumulation
        # Replay of the revised input creates nothing new.
        third = run_monitoring(revised, self.database)
        self.assertEqual(
            sorted(alert["alert_id"] for alert in third["alerts"]),
            sorted(alert["alert_id"] for alert in second["alerts"]),
        )

    def test_revision_can_create_a_supported_shift_without_duplicating_it(self):
        # First look: no shift anywhere.  Later a corrected review moves the
        # preceding weeks; that revision is allowed to produce an alert once.
        data = build()
        first = run_monitoring(data, self.database)
        self.assertFalse(
            any(alert["decision"] == "supported_shift" for alert in first["alerts"])
        )
        revised = copy.deepcopy(data)
        for week in POST:
            for row in revised["samples"]:
                if row["week"] == week and row["stratum"] == "routine":
                    row["revision"] = 2
                    row["accepted"] = False
                    row["modified"] = True
                    row["known_at"] = "2026-02-06"
        revised["as_of"] = "2026-02-10"
        second = run_monitoring(revised, self.database)
        supported = [
            alert for alert in second["alerts"] if alert["decision"] == "supported_shift"
        ]
        self.assertTrue(supported)
        keys = [(alert["week"], alert["metric"]) for alert in supported]
        self.assertEqual(len(keys), len(set(keys)))
        third = run_monitoring(revised, self.database)
        self.assertEqual(
            sorted(alert["alert_id"] for alert in third["alerts"]),
            sorted(alert["alert_id"] for alert in second["alerts"]),
        )

    def test_repeated_execution_creates_no_new_alerts(self):
        data = build(rate_overrides={(week, "routine"): 0.2 for week in POST})
        runs = [run_monitoring(data, self.database) for _ in range(3)]
        identifier_sets = [sorted(a["alert_id"] for a in run["alerts"]) for run in runs]
        self.assertEqual(identifier_sets[0], identifier_sets[1])
        self.assertEqual(identifier_sets[1], identifier_sets[2])
        with sqlite3.connect(self.database) as connection:
            stored = connection.execute("SELECT COUNT(*) FROM monitoring_alerts").fetchone()[0]
        self.assertEqual(stored, len(identifier_sets[0]))

    def test_reference_change_starts_new_series_and_keeps_old_alerts(self):
        data = build(rate_overrides={(week, "routine"): 0.2 for week in POST})
        first = run_monitoring(data, self.database)
        old_alerts = {alert["alert_id"] for alert in first["alerts"]}
        self.assertTrue(old_alerts)

        changed = copy.deepcopy(data)
        changed["reference_start"] = "2026-01-08"
        changed["reference_end"] = "2026-01-21"
        second = run_monitoring(changed, self.database)
        self.assertNotEqual(first["series_id"], second["series_id"])
        store = MonitoringStore(self.database)
        preserved = {
            alert["alert_id"]
            for alert in store.load_alerts(first["series_id"])
        }
        self.assertTrue(old_alerts.issubset(preserved))
        self.assertEqual(len(second["series_history"]), 2)

    def test_alert_ids_are_stable_across_identical_runs(self):
        data = build()
        first = run_monitoring(data, self.database)
        second = run_monitoring(data, self.database)
        self.assertEqual(
            [alert["alert_key"] for alert in first["alerts"]],
            [alert["alert_key"] for alert in second["alerts"]],
        )

    def test_alert_reason_distinguishes_evidence_design_and_support(self):
        data = build(rate_overrides={(week, "routine"): 0.2 for week in POST})
        result = run_monitoring(data, self.database)
        reasons = {alert["reason"] for alert in result["alerts"]}
        self.assertTrue(reasons)
        self.assertTrue(all(reason in {
            "supported_behaviour_shift",
            "sampling_structure_change",
            "insufficient_review_evidence",
            "shift_not_confirmed",
        } for reason in reasons))

    def test_personnel_mix_change_is_not_called_model_regression(self):
        # Change *who* reviews in later weeks while keeping the sampling strata
        # and latent behaviour identical.  The design check should note the
        # personnel change; no quality regression should be raised.
        data = build()
        later_operators = ["op-20", "op-21", "op-22", "op-23"]
        for row in data["samples"]:
            if row["week"] in POST:
                index = int(row["sample_id"].rsplit("-", 1)[1])
                operator = later_operators[index % len(later_operators)]
                row["operator_id"] = operator
                row["site_id"] = "site-3" if index % 2 else "site-4"
        result = run_monitoring(data, self.database)
        checks = {(check["week"], check["check"]) for check in result["design_checks"]}
        self.assertTrue(any(name == "personnel_mix_shift" for _, name in checks))
        self.assertFalse(
            any(alert["decision"] == "supported_shift" for alert in result["alerts"])
        )

    def test_shipped_sampling_shift_fixture(self):
        # The committed example fixture changes only the sampling proportions
        # (40/40 -> 10/70) while latent behaviour is constant.  The weighted
        # estimate must stay near the reference; the raw ratio drops; the
        # finding is a single persistent design_shift, not a quality drop.
        import json

        from tests.analysis_inputs.monitoring_fixtures import FIXTURE_PATH, build as build_example

        path = Path(__file__).resolve().parents[2] / "fixtures/analysis/sampling_shift_only.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        # The committed example must match the generator, so the documented
        # "sampling changed, behaviour did not" property is reproducible.
        self.assertEqual(data, build_example())
        self.assertEqual(path, FIXTURE_PATH)
        result = run_monitoring(data, self.database)
        reference = result["reference"]["metrics"]["adoption_rate"]["pooled"]["estimate"]
        for estimate in result["estimates"]:
            if estimate["metric"] != "adoption_rate":
                continue
            self.assertEqual(estimate["status"], "ok")
            self.assertTrue(estimate["interval"])
            self.assertLess(estimate["interval"][0], estimate["interval"][1])

        # Raw proportions after the design change are materially lower than
        # the reference; that is exactly the misleading signal.
        post = [row for row in data["samples"] if row["week"] >= "2026-03-19"]
        naive = sum(row["accepted"] for row in post) / len(post)
        self.assertLess(naive, reference - 0.15)

        self.assertFalse(
            any(alert["decision"] == "supported_shift" for alert in result["alerts"])
        )
        design = [alert for alert in result["alerts"] if alert["decision"] == "design_shift"]
        self.assertTrue(design)
        self.assertEqual(len({alert["week"] for alert in design}), 1)
        self.assertTrue(all(alert["reason"] == "sampling_structure_change" for alert in design))

    def test_appending_a_week_does_not_change_earlier_verdicts(self):
        data = build(rate_overrides={(week, "routine"): 0.2 for week in POST})
        first = run_monitoring(data, self.database)
        first_alert_ids = sorted(
            (alert["week"], alert["metric"], alert["alert_id"], alert["decision"])
            for alert in first["alerts"]
        )

        # The same series continues one week later.  Alpha spending is
        # committed per week when it is first evaluated, so the earlier weeks
        # must keep exactly the verdicts and alert IDs they already had.
        extended = copy.deepcopy(data)
        source = [row for row in data["samples"] if row["week"] == "2026-02-05"]
        for row in source:
            clone = dict(row)
            clone["week"] = "2026-02-12"
            clone["sample_id"] = row["sample_id"].replace("2026-02-05", "2026-02-12")
            clone["known_at"] = "2026-02-12"
            extended["samples"].append(clone)
        extended["sampling_frame"].extend(
            [
                {"week": "2026-02-12", "stratum": "routine", "population": 800},
                {"week": "2026-02-12", "stratum": "modified", "population": 200},
            ]
        )
        second = run_monitoring(extended, self.database)
        second_alert_ids = sorted(
            (alert["week"], alert["metric"], alert["alert_id"], alert["decision"])
            for alert in second["alerts"]
            if alert["week"] <= "2026-02-05"
        )
        self.assertEqual(first_alert_ids, second_alert_ids)
        self.assertEqual(first["series_id"], second["series_id"])

    def test_two_versions_raise_the_statistical_bar(self):
        data = build(rate_overrides={(week, "routine"): 0.2 for week in POST})
        single = run_monitoring(data, str(Path(self.directory.name) / "single.sqlite3"))

        # Add a second assistant version in the same weeks, with the same rows.
        doubled = copy.deepcopy(data)
        extra = []
        for row in data["samples"]:
            if row["week"] in POST:
                clone = dict(row)
                clone["sample_id"] = row["sample_id"] + "-v3"
                clone["version"] = "assist-v3"
                extra.append(clone)
        doubled["samples"].extend(extra)
        two = run_monitoring(doubled, str(Path(self.directory.name) / "two.sqlite3"))
        self.assertEqual(two["alerts"], two["alerts"])
        checks = [check for check in two["design_checks"] if check["check"] == "multiple_versions_in_week"]
        self.assertTrue(checks)
        self.assertTrue(
            all(
                estimate["version"] in {"ref-a", "assist-v3"}
                for estimate in two["estimates"]
            )
        )
        self.assertEqual(len(two["estimates"]), 2 * len(single["estimates"]))


if __name__ == "__main__":
    unittest.main()
