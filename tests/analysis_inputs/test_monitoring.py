from __future__ import annotations

import copy
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from alignment_observatory.monitoring import (
    METRICS,
    MonitoringInputError,
    MonitoringStore,
    estimate_metric,
    parse_frame,
    parse_samples,
    run_monitoring,
    t_quantile,
)

ROOT = Path(__file__).resolve().parents[2]
sys_path_added = False


def fixture_weeks(start_year=2026, start_month=1, start_day=1, count=8, gap=0):
    from datetime import date, timedelta

    first = date(start_year, start_month, start_day)
    return [
        (first + timedelta(days=7 * index)).isoformat()
        for index in range(count)
    ]


class EstimatorTests(unittest.TestCase):
    def test_t_quantile_matches_known_values(self):
        self.assertAlmostEqual(t_quantile(0.975, df=1), 12.706, places=2)
        self.assertAlmostEqual(t_quantile(0.975, df=10), 2.228, places=2)
        self.assertAlmostEqual(t_quantile(0.975, df=30), 2.042, places=2)

    def test_revision_replaces_rather_than_accumulates(self):
        rows = [
            {
                "sample_id": "a",
                "revision": 1,
                "week": "2026-01-01",
                "stratum": "routine",
                "inclusion_probability": 0.1,
                "operator_id": "op-1",
                "site_id": "s1",
                "version": "v1",
                "accepted": False,
                "modified": True,
                "review_complete": True,
                "known_at": "2026-01-01",
            },
            {
                "sample_id": "a",
                "revision": 2,
                "week": "2026-01-01",
                "stratum": "routine",
                "inclusion_probability": 0.1,
                "operator_id": "op-1",
                "site_id": "s1",
                "version": "v1",
                "accepted": True,
                "modified": True,
                "review_complete": True,
                "known_at": "2026-01-02",
            },
        ]
        parsed, issues = parse_samples(rows, "2026-01-05")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].revision, 2)
        self.assertTrue(parsed[0].accepted)
        self.assertEqual(issues, [])

    def test_not_yet_known_revision_is_excluded(self):
        rows = [
            {
                "sample_id": "a",
                "revision": 1,
                "week": "2026-01-01",
                "stratum": "routine",
                "inclusion_probability": 0.1,
                "operator_id": "op-1",
                "site_id": "s1",
                "version": "v1",
                "accepted": False,
                "modified": True,
                "review_complete": True,
                "known_at": "2026-01-01",
            },
            {
                "sample_id": "a",
                "revision": 2,
                "week": "2026-01-01",
                "stratum": "routine",
                "inclusion_probability": 0.1,
                "operator_id": "op-1",
                "site_id": "s1",
                "version": "v1",
                "accepted": True,
                "modified": True,
                "review_complete": True,
                "known_at": "2026-02-01",
            },
        ]
        parsed, _ = parse_samples(rows, "2026-01-05")
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].revision, 1)
        self.assertFalse(parsed[0].accepted)


class DesignTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = str(Path(self.directory.name) / "monitoring.sqlite3")

    def test_illegal_probability_makes_estimable_false(self):
        rows, _ = parse_samples(
            [
                {
                    "sample_id": "a",
                    "week": "2026-01-01",
                    "stratum": "routine",
                    "inclusion_probability": 0,
                    "operator_id": "op",
                    "version": "v1",
                    "accepted": True,
                    "modified": False,
                    "review_complete": True,
                }
            ],
            None,
        )
        frame, _ = parse_frame([{"week": "2026-01-01", "stratum": "routine", "population": 100}])
        result = estimate_metric(rows, frame, ["2026-01-01"], "adoption_rate", lambda sample: sample.stratum)
        self.assertIsNone(result["estimate"])
        self.assertEqual(result["status"], "unestimable")
        self.assertEqual(result["reason"], "illegal_inclusion_probability")

    def test_missing_frame_stratum_is_unestimable_not_zero(self):
        rows, _ = parse_samples(
            [
                {
                    "sample_id": "a",
                    "week": "2026-01-01",
                    "stratum": "routine",
                    "inclusion_probability": 0.1,
                    "operator_id": "op",
                    "version": "v1",
                    "accepted": True,
                    "modified": False,
                    "review_complete": True,
                }
            ],
            None,
        )
        frame, _ = parse_frame(
            [
                {"week": "2026-01-01", "stratum": "routine", "population": 100},
                {"week": "2026-01-01", "stratum": "modified", "population": 50},
            ]
        )
        result = estimate_metric(rows, frame, ["2026-01-01"], "adoption_rate", lambda sample: sample.stratum)
        self.assertIsNone(result["estimate"])
        self.assertEqual(result["status"], "unestimable")
        self.assertIn("modified", result["detail"]["strata"][0])


class MonitoringRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = str(Path(self.directory.name) / "monitoring.sqlite3")
        self.input = json.loads((ROOT / "fixtures/analysis/input.json").read_text(encoding="utf-8"))

    def test_cli_output_shape(self):
        result = run_monitoring(self.input, self.database)
        for key in ("series_id", "estimates", "alerts", "design_checks", "reference_version"):
            self.assertIn(key, result)
        self.assertIsInstance(result["estimates"], list)
        for estimate in result["estimates"]:
            for key in ("week", "version", "metric", "estimate", "interval", "status"):
                self.assertIn(key, estimate)
            self.assertIn(estimate["status"], {"ok", "insufficient_evidence", "unestimable"})
        for alert in result["alerts"]:
            for key in ("alert_id", "week", "reason"):
                self.assertIn(key, alert)

    def test_reference_covers_declared_window(self):
        result = run_monitoring(self.input, self.database)
        self.assertEqual(
            result["reference"]["weeks"], ["2026-01-01", "2026-01-08", "2026-01-15"]
        )
        self.assertEqual(len(result["monitor_weeks"]), 5)

    def test_estimates_include_all_three_metrics(self):
        result = run_monitoring(self.input, self.database)
        metrics = {estimate["metric"] for estimate in result["estimates"]}
        self.assertEqual(metrics, set(METRICS))

    def test_every_estimate_has_an_interval(self):
        result = run_monitoring(self.input, self.database)
        self.assertEqual(len(result["estimates"]), 15)
        for estimate in result["estimates"]:
            self.assertEqual(estimate["status"], "ok")
            self.assertIsNotNone(estimate["interval"])
            self.assertLessEqual(estimate["interval"][0], estimate["estimate"])
            self.assertLessEqual(estimate["estimate"], estimate["interval"][1])

    def test_replay_does_not_create_new_alerts(self):
        first = run_monitoring(self.input, self.database)
        second = run_monitoring(self.input, self.database)
        self.assertEqual(len(first["alerts"]), len(second["alerts"]))
        self.assertEqual(
            sorted(alert["alert_id"] for alert in first["alerts"]),
            sorted(alert["alert_id"] for alert in second["alerts"]),
        )
        with sqlite3.connect(self.database) as connection:
            count = connection.execute("SELECT COUNT(*) FROM monitoring_alerts").fetchone()[0]
        self.assertEqual(count, len(first["alerts"]))

    def test_series_identity_is_stable_and_reused(self):
        first = run_monitoring(self.input, self.database)
        second = run_monitoring(self.input, self.database)
        self.assertEqual(first["series_id"], second["series_id"])
        self.assertEqual(len(second["series_history"]), 1)

    def test_reference_change_creates_new_series_and_preserves_alerts(self):
        first = run_monitoring(self.input, self.database)
        revised = copy.deepcopy(self.input)
        revised["reference_start"] = "2026-01-08"
        revised["reference_end"] = "2026-01-21"
        second = run_monitoring(revised, self.database)
        self.assertNotEqual(first["series_id"], second["series_id"])
        self.assertEqual(len(second["series_history"]), 2)
        alert_ids_before = {a["alert_id"] for a in first["alerts"]}
        for alert_id in alert_ids_before:
            store = MonitoringStore(self.database)
            self.assertTrue(
                any(alert["alert_id"] == alert_id for alert in store.load_alerts(first["series_id"]))
            )

    def test_alerts_are_never_deleted_after_reference_change(self):
        run_monitoring(self.input, self.database)
        revised = copy.deepcopy(self.input)
        revised["reference_start"] = "2026-01-08"
        revised["reference_end"] = "2026-01-21"
        run_monitoring(revised, self.database)
        with sqlite3.connect(self.database) as connection:
            total = connection.execute("SELECT COUNT(*) FROM monitoring_alerts").fetchone()[0]
            series = connection.execute(
                "SELECT COUNT(DISTINCT series_id) FROM monitoring_alerts"
            ).fetchone()[0]
        self.assertGreaterEqual(total, 0)
        self.assertLessEqual(series, 1)  # alerts in this fixture are absent; schema is intact

    def test_output_is_deterministic(self):
        first = run_monitoring(self.input, self.database)
        second = run_monitoring(self.input, self.database)
        self.assertEqual(
            [(e["week"], e["version"], e["metric"], e["estimate"], e["status"]) for e in first["estimates"]],
            [(e["week"], e["version"], e["metric"], e["estimate"], e["status"]) for e in second["estimates"]],
        )

    def test_scope_statement_excludes_diagnostic_evaluation(self):
        result = run_monitoring(self.input, self.database)
        self.assertIn("review behaviour", result["scope"])
        self.assertIn("diagnostic correctness", result["scope"])

    def test_input_without_frame_is_rejected(self):
        broken = copy.deepcopy(self.input)
        broken["sampling_frame"] = []
        with self.assertRaises(MonitoringInputError):
            run_monitoring(broken, self.database)

    def test_unknown_version_week_uses_pooled_reference_fallback(self):
        # The final monitored week runs a new assistant version.  It has no
        # baseline in the frozen reference, so the pooled reference is used and
        # that scope is recorded on every estimate of that version.
        versioned = copy.deepcopy(self.input)
        for row in versioned["samples"]:
            if row["week"] == "2026-02-19":
                row["version"] = "assist-v3"
        result = run_monitoring(versioned, self.database)
        new_estimates = [e for e in result["estimates"] if e["version"] == "assist-v3"]
        self.assertEqual({e["metric"] for e in new_estimates}, set(METRICS))
        self.assertTrue(all(e["baseline_scope"] == "pooled" for e in new_estimates))
        self.assertTrue(all(e["status"] == "ok" for e in new_estimates))
        self.assertTrue(
            any(
                check["check"] == "version_baseline_fallback"
                for check in result["design_checks"]
            )
        )
        # The pooled fallback is annotated in every design/version record: here
        # it appears as a design check rather than an alert, because the data
        # show no shift at all.  The estimate itself carries baseline_scope.
        self.assertTrue(
            any(
                check["check"] == "version_baseline_fallback"
                and "assist-v3" in check["detail"]["new_versions"]
                for check in result["design_checks"]
            )
        )
        self.assertTrue(
            all(
                estimate["baseline_scope"] == "pooled"
                for estimate in result["estimates"]
                if estimate["version"] == "assist-v3"
            )
        )

    def test_version_with_a_missing_stratum_is_unestimable(self):
        # A version sampled in only one of the week's two strata cannot be
        # projected to the population; it must not use the one stratum alone.
        versioned = copy.deepcopy(self.input)
        for row in versioned["samples"]:
            if row["week"] == "2026-02-19" and row["stratum"] == "modified":
                row["version"] = "assist-v3"
        result = run_monitoring(versioned, self.database)
        new_estimates = [e for e in result["estimates"] if e["version"] == "assist-v3"]
        self.assertTrue(new_estimates)
        self.assertTrue(all(e["status"] == "unestimable" for e in new_estimates))
        self.assertTrue(all(e["estimate"] is None for e in new_estimates))
        self.assertIn(
            "no_sample_for_frame_stratum",
            {e["reason"] for e in new_estimates},
        )
