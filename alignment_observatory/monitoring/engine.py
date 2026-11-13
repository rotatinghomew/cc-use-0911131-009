from __future__ import annotations

"""Weekly monitoring of review behaviour against a frozen reference period.

Design summary
--------------

*Estimates*
    Each weekly estimate is a design-based (Hajek ratio) estimate.  Selection
    probabilities come from ``inclusion_probability`` and population sizes from
    ``sampling_frame``.  Records collapse to their highest revision known at
    ``as_of``.  Standard errors are linearised, stratified by sampling stratum
    and clustered by operator, so repeated reviews by one operator do not
    masquerade as independent evidence.

*Frozen reference*
    The first run of a monitoring series computes and stores the reference
    estimate.  The series is identified by the reference dates, the reference
    version and a digest of the reference-period rows.  Re-running with the
    same data reuses the stored reference and does not move the goalposts.
    Changing the reference period, or its rows, starts a new series; old
    series and their alerts remain in the database.

*Alerting*
    A shift is only called supported when

      1. the weekly estimate differs from the frozen reference by at least a
         practical effect for that metric;
      2. the difference is statistically supported after a Bonferroni
         correction for the number of looks so far, the number of versions
         compared in the same week and the three metrics;
      3. the same direction is seen in this week and the immediately previous
         monitored week.

    A movement that coincides with a sampling-structure change (allocation,
    declared sample size, frame population, inclusion probability, or a pool
    fallback for a new version) is reported as ``design_shift`` and is never
    labelled a quality regression.  A personnel/site mix change is recorded as
    an attribution note on the finding; it describes who is reviewing.  Thin
    data yields an "insufficient evidence" alert only when something looked
    actionable; a quiet thin week is simply reported in ``estimates`` and
    ``design_checks``.

    A sustained finding is not re-alerted every week: later weeks of the same
    episode carry ``continued=true`` and point at the episode's opening
    ``episode_alert_id``.  The per-week alpha threshold is committed when a
    week is first evaluated and reused, so appending a week never changes an
    earlier week's verdict.

*Replay*
    Alerts carry a deterministic identity.  Re-running the same input against
    the same database inserts no new alert.  A revision of the same week's
    conclusion supersedes the earlier alert (which is kept, marked superseded)
    rather than accumulating duplicate evidence.
"""

import hashlib
from statistics import NormalDist
from typing import Any, Mapping, Sequence

from alignment_observatory.monitoring.estimators import (
    METRICS,
    MIN_COMPLETED,
    MonitoringInputError,
    build_reference,
    compare_to_reference,
    group_estimate,
    heterogeneity_checks,
    parse_frame,
    parse_samples,
    reference_comparison,
    week_design_checks,
)
from alignment_observatory.monitoring.store import MonitoringStore
from alignment_observatory.sampling.cohorts import stable_json, utc_now

ALPHA = 0.05
ALERT_VERSION = 1

# Design checks that make a quality comparison non-comparable when they fire.
RELEVANT_STRUCTURE_CHECKS = {
    "declared_sample_size",
    "stratum_allocation_shift",
    "personnel_mix_shift",
    "site_mix_shift",
    "frame_population_shift",
}

# A structure change that *directly invalidates the comparison*, as opposed to
# a personnel-mix change that may explain a real shift.  Only these suppress a
# supported quality signal and relabel it ``design_shift``.
BLOCKING_STRUCTURE_CHECKS = {
    "declared_sample_size",
    "stratum_allocation_shift",
    "frame_population_shift",
    # A change in inclusion probability changes the weights themselves; it is
    # a sampling-design change, never evidence about review behaviour.
    "inclusion_probability_shift",
}
EXPLANATORY_STRUCTURE_CHECKS = {
    # Personnel and site mix describe *who* is reviewing.  They can accompany a
    # genuine change, so they annotate a supported shift rather than block it.
    "personnel_mix_shift",
    "site_mix_shift",
}
BLOCKING_CHECKS = {
    "sampling_frame_present",
    "sample_present",
    "inclusion_probability",
    "sample_size_within_frame",
}

PRACTICAL_EFFECT = {
    "adoption_rate": 0.03,
    "modification_rate": 0.03,
    "coverage_rate": 0.02,
}


def _digest(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _iso_week(value: str) -> str:
    from datetime import date

    try:
        return date.fromisoformat(value[:10]).isoformat()
    except ValueError as error:
        raise MonitoringInputError(f"week is not ISO 8601: {value!r}") from error


def _design_signature(
    reference_start: str,
    reference_end: str,
    reference_version: str,
    reference_weeks: Sequence[str],
    reference_rows: Sequence[Any],
    frame: Mapping[tuple[str, str], int],
) -> str:
    """Frozen identity of a monitoring series.

    Any change to the reference dates, version, reference rows or
    reference-period frame creates a different signature and therefore a new
    series.  Nothing about the reference is silently mutated in place.
    """
    reference_payload = [
        {
            "sample_id": sample.sample_id,
            "revision": sample.revision,
            "week": sample.week,
            "stratum": sample.stratum,
            "inclusion_probability": sample.probability,
            "operator_id": sample.operator_id,
            "site_id": sample.site_id,
            "version": sample.version,
            "accepted": sample.accepted,
            "modified": sample.modified,
            "review_complete": sample.review_complete,
        }
        for sample in sorted(reference_rows, key=lambda item: (item.week, item.sample_id))
    ]
    reference_frame = [
        [week, stratum, population]
        for (week, stratum), population in sorted(frame.items())
        if week in set(reference_weeks)
    ]
    return _digest(
        {
            "reference_start": reference_start,
            "reference_end": reference_end,
            "reference_version": reference_version,
            "reference_weeks": list(reference_weeks),
            "reference_rows": reference_payload,
            "reference_frame": reference_frame,
            "alert_version": ALERT_VERSION,
            "practical_effect": PRACTICAL_EFFECT,
            "alpha": ALPHA,
        }
    )


def _series_id(signature: str) -> str:
    return "series-" + signature[:20]


def _thresholds() -> dict[str, Any]:
    return {
        "alpha": ALPHA,
        "metrics_tested": len(METRICS),
        "min_completed_reviews": MIN_COMPLETED,
        "consecutive_looks": 2,
        "alert_version": ALERT_VERSION,
        "practical_effect": dict(PRACTICAL_EFFECT),
        "version_baseline": "version-specific, pooled-reference fallback",
        "variance": "linearised, stratified by stratum, clustered by operator",
        "cross_version_comparison": "noted separately, not alerted",
    }


def _look_threshold(look: int, versions: int, metrics: int = len(METRICS)) -> float:
    """Two-sided Bonferroni threshold across looks, versions and metrics."""
    family = max(1, look) * max(1, versions) * max(1, metrics)
    probability = 1.0 - ALPHA / (2.0 * family)
    return NormalDist().inv_cdf(probability)


def _is_signal(comparison: Mapping[str, Any], threshold_z: float) -> bool:
    z_value = comparison.get("z")
    difference = comparison.get("difference")
    if z_value is None or difference is None:
        return False
    return abs(float(difference)) >= float(comparison.get("practical_threshold", 0.0)) and abs(
        float(z_value)
    ) >= threshold_z


def _direction(comparison: Mapping[str, Any]) -> int:
    difference = comparison.get("difference") or 0.0
    return 1 if difference > 0 else (-1 if difference < 0 else 0)


def _week_order(rows: Sequence[Any], reference_end: str) -> tuple[list[str], list[str]]:
    weeks = sorted({sample.week for sample in rows})
    reference_weeks = [week for week in weeks if week <= reference_end]
    monitor_weeks = [week for week in weeks if week > reference_end]
    return reference_weeks, monitor_weeks


def _existing_decision(
    alerts: Sequence[Mapping[str, Any]], week: str, version: str, metric: str
) -> tuple[str | None, str | None]:
    """The current alert for a week, ignoring retracted revisions."""
    for alert in alerts:
        if (
            alert.get("week") == week
            and alert.get("version") == version
            and alert.get("metric") == metric
            and alert.get("status") != "superseded"
        ):
            return alert.get("decision"), alert.get("alert_id")
    return None, None


def _alert_identity(
    series_id: str, week: str, version: str, metric: str, decision: str
) -> tuple[str, str]:
    key = _digest([series_id, week, version, metric, decision])
    return "alert-" + key[:16], key


def _evidence_weeks(previous: Mapping[str, Any], week: str, decision: str) -> list[str]:
    if decision in {"supported_shift", "design_shift"} and previous.get("week"):
        return [str(previous["week"]), week]
    return [week]


def _continues_previous(
    previous: Mapping[str, Any],
    decision: str,
    direction: int,
    structure_checks: Sequence[str],
    reason: str,
) -> bool:
    """Is this week a continuation of the previous week's open finding?

    The previous week's evaluation is recomputed deterministically from the
    input, so this is replay-safe and does not depend on what is stored.  A
    continuing episode is recorded in ``estimates`` but does not mint another
    alert; otherwise a sustained shift would produce a new alert every week.

    * ``supported_shift``: continues while the previous week was also a
      supported shift in the same direction.
    * ``design_shift``: continues while the previous week had the same
      decision and the structure problems still overlap.  A different design
      problem is a new finding.
    * ``insufficient_evidence``: continues while the previous week had the same
      reason, so a week of thin data does not nag every seven days.
    """
    if not previous.get("week"):
        return False
    if previous.get("decision") != decision:
        return False
    if decision == "design_shift":
        # A design change persists regardless of the sign of each week's
        # estimate.  It continues while the same structure problem remains.
        prior_checks = set(previous.get("structure_checks") or [])
        return bool(set(structure_checks) & prior_checks)
    if int(previous.get("direction") or 0) != direction:
        return False
    if decision == "insufficient_evidence":
        return previous.get("reason") == reason
    return True


def _episode_number(
    alerts: Sequence[Mapping[str, Any]], week: str, version: str, metric: str, decision: str
) -> int:
    prior = [
        alert
        for alert in alerts
        if alert.get("version") == version
        and alert.get("metric") == metric
        and alert.get("decision") == decision
        and alert.get("week", "") < week
    ]
    return 1 + len(prior)


def _alert_payload(
    *,
    series_id: str,
    week: str,
    version: str,
    metric: str,
    decision: str,
    episode: int,
    reason: str,
    explanation: str,
    estimate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    episode_weeks: Sequence[str],
    structure_checks: Sequence[str],
    baseline_scope: str,
    threshold_z: float,
    supersedes: str | None,
) -> dict[str, Any]:
    # Alert identity is stable for (series, week, version, metric, decision):
    # re-running or revising the same week never mints an additional alert.
    alert_id, key = _alert_identity(series_id, week, version, metric, decision)
    category = {
        "supported_shift": "quality_signal",
        "design_shift": "design_signal",
        "insufficient_evidence": "evidence_signal",
    }[decision]
    return {
        "alert_id": alert_id,
        "alert_key": key,
        "series_id": series_id,
        "week": week,
        "version": version,
        "metric": metric,
        "decision": decision,
        "direction": _direction(comparison),
        "category": category,
        "reason": reason,
        "explanation": explanation,
        "status": "open",
        "episode": episode,
        "episode_weeks": list(episode_weeks),
        "estimate": {
            "estimate": estimate.get("estimate"),
            "interval": estimate.get("interval"),
            "status": estimate.get("status"),
            "n": estimate.get("n"),
            "completed": estimate.get("completed"),
            "clusters": estimate.get("clusters"),
            "se": estimate.get("se"),
            "df": estimate.get("df"),
        },
        "reference_comparison": {
            "difference": comparison.get("difference"),
            "z": comparison.get("z"),
            "outcome": comparison.get("outcome"),
            "baseline_scope": baseline_scope,
            "baseline_estimate": comparison.get("baseline_estimate"),
            "baseline_interval": comparison.get("baseline_interval"),
            "threshold_z": round(threshold_z, 4),
        },
        "structure_checks": sorted(structure_checks),
        "supersedes": supersedes,
        "raised_at": utc_now(),
        "alert_version": ALERT_VERSION,
        "scope": "review behaviour on assistant text; not diagnostic correctness",
    }


def _decide(
    estimate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    previous: Mapping[str, Any],
    blocking: bool,
    structure_checks: Sequence[str],
    threshold_z: float,
    baseline_scope: str = "version",
) -> tuple[str, str, str]:
    """Return ``(decision, reason, explanation)`` for one metric.

    The distinctions are deliberate:

    * ``insufficient_evidence`` - the data cannot support a statement (thin
      reviews, unestimable design, unconfirmed one-week movement);
    * ``design_shift`` - the sampling structure changed, so the comparison is
      not a comparison of review behaviour;
    * ``supported_shift`` - a sustained, practically and statistically
      supported movement with no competing design explanation.

    Personnel/site mix changes are reported as ``explanation`` on a supported
    shift; they describe who is reviewing, not a sampler defect.
    """
    practical = float(comparison.get("practical_threshold", 0.03))
    difference = comparison.get("difference")
    material = difference is not None and abs(float(difference)) >= practical
    current_signal = _is_signal(comparison, threshold_z)
    confirmable = bool(previous.get("week")) and bool(previous.get("signal")) \
        and _direction(previous) == _direction(comparison)
    design_relation = sorted(
        check for check in structure_checks if check in BLOCKING_STRUCTURE_CHECKS
    )
    explanatory = sorted(
        check for check in structure_checks if check in EXPLANATORY_STRUCTURE_CHECKS
    )
    attribution = (
        " Related changes: " + ", ".join(explanatory) + "."
        if explanatory
        else ""
    )
    if baseline_scope == "pooled":
        attribution += (
            " This version has no version-specific baseline in the frozen reference; "
            "the comparison uses the pooled reference."
        )

    if blocking:
        return (
            "insufficient_evidence",
            "unestimable_design",
            "the sampling frame or inclusion probability is invalid for this week; no population "
            "estimate is formed and nothing is compared with the baseline",
        )
    if estimate.get("status") == "unestimable":
        return (
            "insufficient_evidence",
            estimate.get("reason") or "unestimable_metric",
            "no population estimate can be formed for this metric; this is not read as zero",
        )
    if comparison.get("outcome") == "reference_unavailable":
        return (
            "insufficient_evidence",
            "reference_not_available",
            "the version has no estimable frozen reference baseline; a movement cannot be attributed "
            "to review behaviour",
        )
    if estimate.get("status") == "insufficient_evidence":
        if material:
            return (
                "insufficient_evidence",
                "insufficient_review_evidence",
                "the observed movement is large enough to matter but too few completed reviews or "
                "operator clusters are available to support or reject it",
            )
        return ("no_alert", "no_actionable_change", "estimate is compatible with the frozen reference")
    if design_relation and material:
        return (
            "design_shift",
            "sampling_structure_change",
            "the estimate differs materially in a week whose sampling structure also changed ("
            + ", ".join(design_relation)
            + "); the movement is attributed to the design change and is not read as model regression",
        )
    if current_signal and confirmable:
        return (
            "supported_shift",
            "supported_behaviour_shift",
            "the change exceeds the practical effect, is statistically supported after "
            "repeated-look, version and metric controls, and reproduced in "
            f"{previous.get('week')} and the current week." + attribution,
        )
    if current_signal:
        return (
            "insufficient_evidence",
            "shift_not_confirmed",
            "the movement exceeded the practical effect and the corrected statistical threshold, "
            "but it has not yet reproduced in a second consecutive monitored week; this is a "
            "prompt to watch, not a regression finding",
        )
    return ("no_alert", "no_actionable_change", "estimate is compatible with the frozen reference")


def evaluate_monitoring(
    *,
    series_id: str,
    reference_version: str,
    frozen: Mapping[str, Any],
    rows: Sequence[Any],
    frame: Mapping[tuple[str, str], int],
    reference_rows: Sequence[Any],
    monitor_weeks: Sequence[str],
    stored_alerts: Sequence[Mapping[str, Any]],
    committed_thresholds: Mapping[tuple[str, str], float] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Evaluate every monitored week in order.

    Returns ``(estimates, design_checks, alerts, evaluations)``.  It is pure
    with respect to the database: the caller persists the results, which makes
    replay deterministic and lets the same code recompute a previous look.
    """
    estimates: list[dict[str, Any]] = []
    design_checks: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    local_stored = list(stored_alerts)
    committed_thresholds = dict(committed_thresholds or {})
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    baselines: dict[str, dict[str, tuple[dict[str, Any] | None, str]]] = {}
    episode_opener: dict[tuple[str, str], str] = {}

    for look, week in enumerate(monitor_weeks, start=1):
        week_rows = [sample for sample in rows if sample.week == week]
        week_versions = sorted({sample.version for sample in week_rows}) or [reference_version]
        checks, blocking_checks = week_design_checks(
            week, week_rows, frame, reference_rows, frame
        )
        if not set(week_versions) <= set(frozen["versions"]):
            checks.append(
                {
                    "week": week,
                    "check": "version_baseline_fallback",
                    "status": "noted",
                    "detail": {
                        "new_versions": sorted(set(week_versions) - set(frozen["versions"])),
                        "reference_versions": list(frozen["versions"]),
                        "note": "versions absent from the reference period fall back to the pooled "
                        "reference baseline; every affected estimate records baseline_scope=pooled "
                        "so a pooled comparison is never mistaken for a version-specific one",
                    },
                }
            )
        structure_checks = sorted(
            check["check"]
            for check in checks
            if check["check"] in RELEVANT_STRUCTURE_CHECKS
            and check["status"] in {"warn", "fail", "noted"}
        )
        blocking = bool(blocking_checks)

        for version in week_versions:
            version_rows = [sample for sample in week_rows if sample.version == version]
            version_estimates = {
                metric: group_estimate(version_rows, frame, [week], metric)
                for metric in METRICS
            }
            baseline_cache = baselines.setdefault(version, {})
            comparisons: dict[str, dict[str, Any]] = {}
            scopes: dict[str, str] = {}
            for metric in METRICS:
                if metric not in baseline_cache:
                    baseline_cache[metric] = reference_comparison(frozen, metric, version)
                baseline, scope = baseline_cache[metric]
                scopes[metric] = scope
                comparison = compare_to_reference(
                    {"metric": metric, **version_estimates[metric]}, baseline
                )
                comparison["baseline_estimate"] = (
                    None if baseline is None else baseline.get("estimate")
                )
                comparison["baseline_interval"] = (
                    None if baseline is None else baseline.get("interval")
                )
                comparisons[metric] = comparison

            # Alpha spending is committed when a week is first evaluated.  A
            # later run that appends weeks must not retroactively loosen or
            # tighten an earlier week's threshold, otherwise a historical
            # alert could silently change status as new data arrive.
            threshold_z = committed_thresholds.get(
                (week, version), _look_threshold(look, len(week_versions))
            )

            for metric in METRICS:
                estimate = version_estimates[metric]
                comparison = comparisons[metric]
                previous = latest.get((version, metric), {})
                decision, reason, explanation = _decide(
                    estimate,
                    comparison,
                    previous,
                    blocking,
                    structure_checks,
                    threshold_z,
                    scopes[metric],
                )
                signal = _is_signal(comparison, threshold_z)
                continuing = _continues_previous(
                    previous,
                    decision,
                    _direction(comparison),
                    structure_checks,
                    reason,
                )
                evaluation = {
                    "week": week,
                    "version": version,
                    "metric": metric,
                    "signal": "supported" if signal else "none",
                    "direction": _direction(comparison),
                    "decision": decision,
                    "reason": reason,
                    "explanation": explanation,
                    "estimate": estimate.get("estimate"),
                    "interval": estimate.get("interval"),
                    "status": estimate.get("status"),
                    "estimate_reason": estimate.get("reason"),
                    "n": estimate.get("n"),
                    "completed": estimate.get("completed"),
                    "clusters": estimate.get("clusters"),
                    "df": estimate.get("df"),
                    "se": estimate.get("se"),
                    "unweighted_estimate": estimate.get("unweighted_estimate"),
                    "difference": comparison.get("difference"),
                    "z": comparison.get("z"),
                    "comparison": comparison.get("outcome"),
                    "baseline_scope": scopes[metric],
                    "baseline_estimate": comparison["baseline_estimate"],
                    "baseline_interval": comparison["baseline_interval"],
                    "threshold_z": round(threshold_z, 4),
                    "structure_checks": structure_checks,
                    "blocking": blocking,
                    "continued": continuing,
                    "episode_alert_id": None,
                }
                if decision == "no_alert":
                    latest[(version, metric)] = evaluation
                else:
                    prior_decision, prior_id = _existing_decision(
                        local_stored, week, version, metric
                    )
                    if prior_decision == decision:
                        # Replay or identical re-evaluation: the alert stands.
                        pass
                    elif continuing:
                        # This week continues last week's open finding.
                        # Repeating it as a fresh alert would inflate the alert
                        # count and make a persistent condition look like new
                        # evidence.
                        evaluation["episode_alert_id"] = episode_opener.get(
                            (version, metric)
                        )
                    else:
                        payload = _alert_payload(
                            series_id=series_id,
                            week=week,
                            version=version,
                            metric=metric,
                            decision=decision,
                            episode=_episode_number(
                                local_stored, week, version, metric, decision
                            ),
                            reason=reason,
                            explanation=explanation,
                            estimate=estimate,
                            comparison=comparison,
                            episode_weeks=_evidence_weeks(previous, week, decision),
                            structure_checks=structure_checks,
                            baseline_scope=scopes[metric],
                            threshold_z=threshold_z,
                            supersedes=prior_id,
                        )
                        alerts.append(payload)
                        local_stored = local_stored + [payload]
                        evaluation["episode_alert_id"] = payload["alert_id"]
                        episode_opener[(version, metric)] = payload["alert_id"]
                    latest[(version, metric)] = evaluation

                evaluations.append(evaluation)
                estimates.append(
                    {
                        "series_id": series_id,
                        "week": week,
                        "version": version,
                        "metric": metric,
                        "estimate": evaluation["estimate"],
                        "interval": evaluation["interval"],
                        "status": evaluation["status"],
                        "reason": evaluation["estimate_reason"],
                        "n": evaluation["n"],
                        "completed": evaluation["completed"],
                        "clusters": evaluation["clusters"],
                        "df": evaluation["df"],
                        "se": evaluation["se"],
                        "unweighted_estimate": evaluation.get("unweighted_estimate"),
                        "reference_version": reference_version,
                        "baseline_scope": scopes[metric],
                        "baseline_estimate": evaluation["baseline_estimate"],
                        "baseline_interval": evaluation["baseline_interval"],
                        "difference": evaluation["difference"],
                        "z": evaluation["z"],
                        "threshold_z": evaluation["threshold_z"],
                        "comparison": evaluation["comparison"],
                        "evidence_weeks": _evidence_weeks(previous, week, decision),
                        "continued": evaluation["continued"],
                        "episode_alert_id": evaluation["episode_alert_id"],
                    }
                )

        if len(week_versions) > 1:
            design_checks.append(
                {
                    "week": week,
                    "check": "multiple_versions_in_week",
                    "status": "noted",
                    "detail": {
                        "versions": list(week_versions),
                        "note": "each version is alerted against its own frozen baseline; the "
                        "version count is included in the multiplicity correction",
                    },
                }
            )

        design_checks.extend(checks)

    # Latest-week site/operator differences: described, never alerted on their
    # own.  They help attribute a movement; they are not model regression.
    if monitor_weeks:
        last_week = monitor_weeks[-1]
        last_rows = [sample for sample in rows if sample.week == last_week]
        for version in sorted({sample.version for sample in last_rows}):
            version_rows = [sample for sample in last_rows if sample.version == version]
            site_records = {
                site: {
                    metric: group_estimate(
                        [sample for sample in version_rows if sample.site_id == site],
                        frame,
                        [last_week],
                        metric,
                    )
                    for metric in METRICS
                }
                for site in sorted({sample.site_id for sample in version_rows})
            }
            operator_records = {
                operator: {
                    metric: group_estimate(
                        [sample for sample in version_rows if sample.operator_id == operator],
                        frame,
                        [last_week],
                        metric,
                    )
                    for metric in METRICS
                }
                for operator in sorted({sample.operator_id for sample in version_rows})
            }
            design_checks.extend(
                heterogeneity_checks(last_week, version, site_records, operator_records)
            )

    return estimates, design_checks, alerts, evaluations


def run_monitoring(raw: Mapping[str, Any], database: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MonitoringInputError("input must be a JSON object")
    for field in ("reference_start", "reference_end", "reference_version"):
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise MonitoringInputError(f"{field} is required")
    reference_start = _iso_week(str(raw["reference_start"]))
    reference_end = _iso_week(str(raw["reference_end"]))
    reference_version = str(raw["reference_version"]).strip()
    if reference_start > reference_end:
        raise MonitoringInputError("reference_start is after reference_end")

    as_of = raw.get("as_of")
    if as_of is not None:
        if not isinstance(as_of, str):
            raise MonitoringInputError("as_of must be an ISO 8601 date when present")
        as_of = _iso_week(as_of)

    rows, parsing_issues = parse_samples(raw.get("samples"), as_of)
    frame, frame_issues = parse_frame(raw.get("sampling_frame"))
    if not rows:
        raise MonitoringInputError("input contains no usable sample rows")
    if not frame:
        raise MonitoringInputError(
            "input contains no sampling frame; population rates are unestimable"
        )
    for sample in rows:
        if _iso_week(sample.week) != sample.week:
            raise MonitoringInputError(f"week must be an ISO date: {sample.week!r}")

    reference_weeks, monitor_weeks = _week_order(rows, reference_end)
    if not reference_weeks:
        raise MonitoringInputError("no sample rows fall inside the reference period")

    reference_rows = [sample for sample in rows if sample.week in reference_weeks]
    reference = build_reference(reference_rows, frame, reference_weeks)
    if all(reference["metrics"][metric]["pooled"].get("estimate") is None for metric in METRICS):
        raise MonitoringInputError(
            "reference period cannot be estimated under the declared frame; no monitoring series is created"
        )

    signature = _design_signature(
        reference_start,
        reference_end,
        reference_version,
        reference_weeks,
        reference_rows,
        frame,
    )
    store = MonitoringStore(database)
    thresholds = _thresholds()
    series = store.find_series(signature)
    if series is None:
        series = store.create_series(
            _series_id(signature),
            signature,
            reference_version,
            reference_start,
            reference_end,
            reference,
            thresholds,
        )
    else:
        thresholds = store.load_reference(series["series_id"])["thresholds"]
    series_id = series["series_id"]
    frozen = store.load_reference(series_id)["reference"]
    stored_alerts = store.load_alerts(series_id)
    prior_evaluations = {
        (item["week"], item["version"], item["metric"]): item["payload"]
        for item in store.load_evaluations(series_id)
    }
    # The threshold used for each earlier week is fixed once and reused, so
    # appending a week never changes the verdict on an existing week.
    committed_thresholds: dict[tuple[str, str], float] = {}
    for (week, version, _metric), payload in prior_evaluations.items():
        value = payload.get("threshold_z")
        if value is not None:
            committed_thresholds.setdefault((week, version), float(value))

    estimates, design_checks, alerts, evaluations = evaluate_monitoring(
        series_id=series_id,
        reference_version=reference_version,
        frozen=frozen,
        rows=rows,
        frame=frame,
        reference_rows=reference_rows,
        monitor_weeks=monitor_weeks,
        stored_alerts=stored_alerts,
        committed_thresholds=committed_thresholds,
    )

    store.save_evaluations(series_id, evaluations)
    store.save_design_evaluations(
        series_id,
        [
            {"week": check["week"], "check": check["check"], "status": check["status"], "detail": check["detail"]}
            for check in design_checks
        ],
    )
    for alert in alerts:
        store.insert_alert(series_id, alert)

    # A revised conclusion for the same week replaces the earlier alert instead
    # of accumulating evidence.  The earlier alert is retained, marked
    # superseded, so the audit trail stays complete.
    current = {
        (item["week"], item["version"], item["metric"]): item["decision"]
        for item in evaluations
    }
    superseded: list[tuple[str, str]] = []
    for alert in stored_alerts:
        key = (alert.get("week"), alert.get("version"), alert.get("metric"))
        decision = current.get(key)
        if alert.get("status") == "superseded" or decision is None:
            continue
        if alert.get("decision") != decision:
            replacement = next(
                (
                    item["alert_id"]
                    for item in alerts
                    if item.get("week") == alert.get("week")
                    and item.get("version") == alert.get("version")
                    and item.get("metric") == alert.get("metric")
                ),
                "revision-withdrawn",
            )
            superseded.append((alert["alert_id"], replacement))
    store.apply_supersedes(series_id, superseded)

    return {
        "series_id": series_id,
        "reference_version": reference_version,
        "reference": {
            "start": reference_start,
            "end": reference_end,
            "weeks": reference_weeks,
            "versions": list(frozen["versions"]),
            "metrics": {
                metric: {
                    "pooled": frozen["metrics"][metric]["pooled"],
                    "by_version": frozen["metrics"][metric]["by_version"],
                }
                for metric in METRICS
            },
        },
        "as_of": as_of,
        "monitor_weeks": monitor_weeks,
        "estimates": estimates,
        "design_checks": design_checks,
        "alerts": store.load_alerts(series_id),
        "input_issues": sorted(parsing_issues + frame_issues, key=stable_json),
        "series_history": store.series_history(),
        "thresholds": thresholds,
        "scope": "review behaviour on assistant text only; diagnostic correctness is not evaluated",
    }


def run_monitoring_file(input_path: str, database: str) -> dict[str, Any]:
    import json
    from pathlib import Path

    raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    return run_monitoring(raw, database)
