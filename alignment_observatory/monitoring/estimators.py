from __future__ import annotations

"""Design-based population estimates for repeated review sampling.

The unit of analysis is one *sampled review record*.  Rows are selected with
unequal probabilities from strata, the same operator can review many records
and a record can be reviewed again.  Consequently:

* observations are deduplicated by ``sample_id`` keeping the highest
  ``revision`` (a revision replaces the earlier conclusion, it does not add
  evidence);
* a revision that was not known on the as-of date is not used;
* population rates are Horvitz--Thompson / Hajek ratio estimates with design
  weights ``1 / inclusion_probability``;
* variance is linearised and clustered on operator (within stratum), so a
  reviewer who appears many times is not treated as many independent reviewers;
* a stratum without a sampling-frame population, or with an illegal inclusion
  probability, makes the affected metric *unestimable*.  Nothing is silently
  dropped, zero-filled or replaced by an unweighted rate.
"""

import math
from dataclasses import asdict, dataclass
from datetime import date
from statistics import NormalDist
from typing import Any, Callable, Iterable, Mapping, Sequence

from alignment_observatory.sampling.cohorts import stable_json

METRICS = ("adoption_rate", "modification_rate", "coverage_rate")

# A point estimate is reported even below these limits, but its interval is
# withheld and its status is ``insufficient_evidence``.
MIN_COMPLETED = 10
MIN_CLUSTERS_PER_STRATUM = 2

# Alerting additionally requires a difference that is worth acting on.
MIN_EFFECT = {
    "adoption_rate": 0.03,
    "modification_rate": 0.03,
    "coverage_rate": 0.02,
}

_EPS = 1e-12


class MonitoringInputError(ValueError):
    """The input cannot be interpreted as a probability sample."""


# ---------------------------------------------------------------------------
# Student-t quantile (standard library only)
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta function."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: float) -> float:
    if df <= 0:
        raise ValueError("degrees of freedom must be positive")
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def t_quantile(probability: float, df: float) -> float:
    """Two-sided Student-t quantile via bisection on the CDF."""
    if not 0.0 < probability < 1.0:
        raise ValueError("probability outside (0, 1)")
    if df > 400:
        return NormalDist().inv_cdf(probability)
    low, high = 0.0, 40.0
    for _ in range(120):
        middle = (low + high) / 2.0
        if _t_cdf(middle, df) < probability:
            low = middle
        else:
            high = middle
    return (low + high) / 2.0


# ---------------------------------------------------------------------------
# Parsed records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Sample:
    sample_id: str
    revision: int
    week: str
    stratum: str
    probability: float
    operator_id: str
    site_id: str
    version: str
    accepted: bool
    modified: bool
    review_complete: bool
    known_at: str


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def _as_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def parse_samples(
    raw_samples: Any, as_of: str | None
) -> tuple[list[Sample], list[dict[str, Any]]]:
    """Parse rows, apply the as-of filter and collapse revisions."""
    issues: list[dict[str, Any]] = []
    if not isinstance(raw_samples, list):
        raise MonitoringInputError("samples must be a list")
    as_of_date = _as_date(as_of)

    illegal_probability: list[dict[str, Any]] = []
    by_id: dict[str, list[Sample]] = {}
    for position, item in enumerate(raw_samples):
        if not isinstance(item, Mapping):
            issues.append({"problem": "sample_not_object", "position": position})
            continue
        sample_id = _text(item.get("sample_id"))
        week = _text(item.get("week"))
        stratum = _text(item.get("stratum"))
        revision_raw = item.get("revision", 0)
        revision = revision_raw if isinstance(revision_raw, int) and not isinstance(revision_raw, bool) else None
        probability = item.get("inclusion_probability")
        accepted = _bool(item.get("accepted"))
        modified = _bool(item.get("modified"))
        complete = _bool(item.get("review_complete"))
        known_at = _text(item.get("known_at"))

        problems = []
        if sample_id is None:
            problems.append("missing_sample_id")
        if week is None:
            problems.append("missing_week")
        if stratum is None:
            problems.append("missing_stratum")
        if revision is None:
            problems.append("invalid_revision")
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            problems.append("invalid_inclusion_probability")
        if accepted is None:
            problems.append("invalid_accepted")
        if modified is None:
            problems.append("invalid_modified")
        if complete is None:
            problems.append("invalid_review_complete")

        if problems:
            issues.append(
                {
                    "problem": "invalid_sample",
                    "sample_id": sample_id,
                    "week": week,
                    "revision": revision,
                    "problems": sorted(problems),
                }
            )
            continue

        if not (0.0 < float(probability) <= 1.0):
            # Keep the row so the affected stratum can be reported as
            # unestimable; do not silently drop it and shrink the sample.
            illegal_probability.append(
                {"sample_id": sample_id, "week": week, "stratum": stratum, "inclusion_probability": probability}
            )

        sample = Sample(
            sample_id=sample_id,
            revision=revision,
            week=week,
            stratum=stratum,
            probability=float(probability),
            operator_id=_text(item.get("operator_id")) or "unknown-operator",
            site_id=_text(item.get("site_id")) or "unknown-site",
            version=_text(item.get("version")) or "unknown-version",
            accepted=bool(accepted),
            modified=bool(modified),
            review_complete=bool(complete),
            known_at=known_at or "",
        )

        known_date = _as_date(known_at)
        if known_at and known_date is None:
            issues.append(
                {
                    "problem": "invalid_known_at",
                    "sample_id": sample_id,
                    "week": week,
                    "revision": revision,
                }
            )
            continue
        if as_of_date is not None and known_date is not None and known_date > as_of_date:
            # Not known at the as-of date.  Future knowledge never enters a
            # monitoring run.
            continue

        by_id.setdefault(sample_id, []).append(sample)

    rows: list[Sample] = []
    for sample_id in sorted(by_id):
        candidates = by_id[sample_id]
        highest = max(item.revision for item in candidates)
        latest = [item for item in candidates if item.revision == highest]
        distinct = {stable_json(asdict(item)) for item in latest}
        if len(distinct) > 1:
            issues.append(
                {
                    "problem": "revision_conflict",
                    "sample_id": sample_id,
                    "week": latest[0].week,
                    "revision": highest,
                }
            )
        # Deterministic pick if two conflicting rows share the highest revision.
        rows.append(sorted(latest, key=lambda item: stable_json(asdict(item)))[0])
    for item in illegal_probability:
        issues.append({"problem": "illegal_inclusion_probability", **item})
    return rows, issues


def parse_frame(raw_frame: Any) -> tuple[dict[tuple[str, str], int], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    frame: dict[tuple[str, str], int] = {}
    if raw_frame is None:
        return frame, issues
    if not isinstance(raw_frame, list):
        raise MonitoringInputError("sampling_frame must be a list")
    for position, item in enumerate(raw_frame):
        if not isinstance(item, Mapping):
            issues.append({"problem": "frame_row_not_object", "position": position})
            continue
        week = _text(item.get("week"))
        stratum = _text(item.get("stratum"))
        population = item.get("population")
        if week is None or stratum is None:
            issues.append({"problem": "frame_row_incomplete", "position": position})
            continue
        if isinstance(population, bool) or not isinstance(population, int) or population < 0:
            issues.append(
                {
                    "problem": "invalid_frame_population",
                    "week": week,
                    "stratum": stratum,
                }
            )
            continue
        key = (week, stratum)
        if key in frame and frame[key] != population:
            issues.append(
                {
                    "problem": "conflicting_frame_population",
                    "week": week,
                    "stratum": stratum,
                }
            )
            continue
        frame[key] = population
    return frame, issues


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 8)


def _empty_metric(reason: str, detail: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "estimate": None,
        "unweighted_estimate": None,
        "interval": None,
        "status": "unestimable",
        "reason": reason,
        "n": 0,
        "completed": 0,
        "clusters": 0,
        "df": None,
        "se": None,
        "detail": dict(detail or {}),
    }


def _linearized_variance(
    rows: Sequence[Sample],
    y: Sequence[float],
    weights: Sequence[float],
    strata_keys: Sequence[str],
    cluster_keys: Sequence[str],
    point: float,
    denominator: float,
    known_denominator: bool,
) -> tuple[float | None, int, int]:
    """Clustered, stratified, with-replacement linearised variance.

    ``known_denominator`` is used for coverage, where the frame population is
    known.  Otherwise the ratio estimate's residual ``w * (y - point)`` is used.
    Returns ``(variance, df, cluster_count)``; variance is ``None`` when a
    stratum has fewer than two clusters.
    """
    strata: dict[str, dict[str, float]] = {}
    for sample, value, weight, stratum, cluster in zip(rows, y, weights, strata_keys, cluster_keys):
        if known_denominator:
            contribution = weight * value
        else:
            contribution = weight * (value - point)
        strata.setdefault(stratum, {})
        strata[stratum][cluster] = strata[stratum].get(cluster, 0.0) + contribution

    divisor = denominator * denominator
    variance = 0.0
    minimum_df = None
    total_clusters = 0
    for totals in strata.values():
        count = len(totals)
        total_clusters += count
        if count < MIN_CLUSTERS_PER_STRATUM:
            return None, 0, total_clusters
        values = list(totals.values())
        mean = sum(values) / count
        sum_squares = sum((value - mean) ** 2 for value in values)
        variance += count / (count - 1.0) * sum_squares
        minimum_df = count - 1 if minimum_df is None else min(minimum_df, count - 1)
    if divisor <= _EPS or total_clusters < 2:
        return None, 0, total_clusters
    variance /= divisor
    return max(variance, 0.0), int(minimum_df or 1), total_clusters


def estimate_metric(
    rows: Sequence[Sample],
    frame: Mapping[tuple[str, str], int],
    weeks: Sequence[str],
    metric: str,
    strata_key: Callable[[Sample], str],
) -> dict[str, Any]:
    """Estimate one metric for one group of rows (a week/version or reference)."""
    if not rows:
        return _empty_metric("no_sample_rows")

    # Every positive-population stratum in the frame must be represented; a
    # missing stratum cannot be treated as zero and cannot be dropped.
    required: dict[tuple[str, str], int] = {}
    for week in weeks:
        week_frame = {
            stratum: population
            for (frame_week, stratum), population in frame.items()
            if frame_week == week
        }
        if not week_frame:
            # Without a frame there is no population to estimate for this week.
            # Reporting the sample mean would silently switch to an unweighted
            # result, so the metric is unestimable instead.
            return _empty_metric("no_frame_for_week", {"week": week})
        for stratum, population in week_frame.items():
            if population > 0:
                required[(week, stratum)] = population
    present_weeks = {sample.week for sample in rows}
    for week in weeks:
        if week not in present_weeks:
            return _empty_metric("no_sample_for_frame_week", {"week": week})
    for sample in rows:
        if (sample.week, sample.stratum) not in frame:
            return _empty_metric(
                "sample_stratum_not_in_frame",
                {"week": sample.week, "stratum": sample.stratum},
            )
    sample_strata = {sample.week: set() for sample in rows}
    for sample in rows:
        sample_strata[sample.week].add(sample.stratum)
    missing = sorted(
        f"{week}/{stratum}"
        for (week, stratum) in required
        if stratum not in sample_strata.get(week, set())
    )
    if missing:
        return _empty_metric("no_sample_for_frame_stratum", {"strata": missing})
    illegal = sorted(
        {f"{sample.week}/{sample.stratum}" for sample in rows if not (0.0 < sample.probability <= 1.0)}
    )
    if illegal:
        return _empty_metric("illegal_inclusion_probability", {"strata": illegal})

    if metric == "coverage_rate":
        selected = list(rows)
        values = [1.0 if sample.review_complete else 0.0 for sample in selected]
        denominator = float(sum(required.values()))
        known = True
        completed = sum(1 for sample in selected if sample.review_complete)
    else:
        selected = [sample for sample in rows if sample.review_complete]
        completed = len(selected)
        if not selected:
            return _empty_metric("no_completed_reviews")
        field = "accepted" if metric == "adoption_rate" else "modified"
        values = [1.0 if getattr(sample, field) else 0.0 for sample in selected]
        denominator = sum(1.0 / sample.probability for sample in selected)
        known = False

    # A sample larger than the frame is impossible under the declared design.
    for sample in selected:
        population = frame[(sample.week, sample.stratum)]
        if len([other for other in selected if other.week == sample.week and other.stratum == sample.stratum]) > population:
            return _empty_metric(
                "sample_larger_than_frame",
                {"week": sample.week, "stratum": sample.stratum},
            )

    weights = [1.0 / sample.probability for sample in selected]
    numerator = sum(weight * value for weight, value in zip(weights, values))
    if denominator <= _EPS:
        return _empty_metric("zero_denominator")
    point = _clamp(numerator / denominator)
    strata_keys = [strata_key(sample) for sample in selected]
    cluster_keys = [f"{strata_key(sample)}|{sample.operator_id}" for sample in selected]
    variance, df, clusters = _linearized_variance(
        selected, values, weights, strata_keys, cluster_keys, point, denominator, known
    )

    # The unweighted rate is reported only so that the monitoring can explain
    # *why* a raw proportion moved when the design changed.  It is never used
    # as a population estimate and never replaces ``estimate``.
    unweighted = sum(values) / len(values) if values else None

    record: dict[str, Any] = {
        "estimate": _rounded(point),
        "unweighted_estimate": _rounded(unweighted),
        "interval": None,
        "status": "ok",
        "reason": None,
        # ``n`` is sampled records, ``completed`` is how many completed review.
        # They differ whenever unfinished reviews exist; an unfinished review
        # never counts as "not adopted".
        "n": len(rows),
        "completed": completed,
        "clusters": clusters,
        "df": df or None,
        "se": None,
        "detail": {},
    }

    if completed < MIN_COMPLETED:
        record["status"] = "insufficient_evidence"
        record["reason"] = "fewer_than_minimum_reviews"
        return record
    if variance is None or df <= 0:
        record["status"] = "insufficient_evidence"
        record["reason"] = "variance_not_estimable"
        return record

    if variance <= 0.0:
        # No observed between-cluster variation at all (for example every
        # completed review in the week agreed).  A zero-width interval would
        # claim unearned precision and could never be exceeded by a future
        # week, so fall back to the conservative "rule of three" margin based
        # on the effective evidence count.  The point estimate is unchanged.
        # For coverage the denominator is the frame, so the informative count
        # is the number of sampled records, not the number completed.
        effective = max(1, len(selected) if metric == "coverage_rate" else completed)
        margin = 3.0 / effective
        record["se"] = _rounded(margin / 2.0)
        record["interval"] = [
            _rounded(_clamp(point - margin)),
            _rounded(_clamp(point + margin)),
        ]
        record["detail"]["interval_method"] = "rule_of_three_degenerate_variance"
        record["detail"]["effective_n"] = effective
        return record

    standard_error = math.sqrt(variance)
    critical = t_quantile(0.975, df)
    record["se"] = _rounded(standard_error)
    record["interval"] = [
        _rounded(_clamp(point - critical * standard_error)),
        _rounded(_clamp(point + critical * standard_error)),
    ]
    record["detail"]["interval_method"] = "linearized_cluster_robust"
    return record


def group_estimate(
    rows: Sequence[Sample],
    frame: Mapping[tuple[str, str], int],
    weeks: Sequence[str],
    metric: str,
    *,
    pooled: bool = False,
) -> dict[str, Any]:
    """Estimate all requested metrics for one week/version group.

    ``pooled`` is used for the frozen reference period: the sampling strata
    become week x stratum because each week is an independent selection.
    """
    if pooled:
        strata_key = lambda sample: f"{sample.week}|{sample.stratum}"
    else:
        strata_key = lambda sample: sample.stratum
    return estimate_metric(rows, frame, weeks, metric, strata_key)


def build_reference(
    rows: Sequence[Sample],
    frame: Mapping[tuple[str, str], int],
    reference_weeks: Sequence[str],
) -> dict[str, Any]:
    """Build the reference-period estimate that will later be frozen."""
    reference_rows = [sample for sample in rows if sample.week in set(reference_weeks)]
    versions = sorted({sample.version for sample in reference_rows})
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        pooled = group_estimate(reference_rows, frame, reference_weeks, metric, pooled=True)
        by_version: dict[str, Any] = {}
        for version in versions:
            version_rows = [sample for sample in reference_rows if sample.version == version]
            # A version-specific baseline covers the reference weeks in which
            # that version actually appears, not the whole reference window.
            version_weeks = sorted({sample.week for sample in version_rows})
            by_version[version] = group_estimate(
                version_rows,
                frame,
                version_weeks,
                metric,
                pooled=True,
            )
        metrics[metric] = {"pooled": pooled, "by_version": by_version}
    return {
        "weeks": list(reference_weeks),
        "versions": versions,
        "metrics": metrics,
    }


def reference_comparison(
    frozen: Mapping[str, Any], metric: str, version: str
) -> tuple[dict[str, Any] | None, str]:
    """Pick the frozen baseline for a monitored version.

    A version-specific baseline is used when it was estimable in the reference
    period; otherwise the pooled reference baseline is used.  The pooled
    fallback is recorded explicitly so a version change is never mistaken for
    a quality shift.
    """
    record = frozen.get("metrics", {}).get(metric, {})
    candidate = record.get("by_version", {}).get(version)
    if candidate and candidate.get("status") == "ok" and candidate.get("interval"):
        return candidate, "version"
    pooled = record.get("pooled")
    if pooled and pooled.get("status") == "ok" and pooled.get("interval"):
        return pooled, "pooled"
    return None, "unavailable"


def compare_to_reference(
    estimate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Describe the difference; the alert decision belongs to the engine."""
    if baseline is None or estimate.get("estimate") is None:
        return {"outcome": "reference_unavailable", "z": None, "difference": None}
    difference = float(estimate["estimate"]) - float(baseline["estimate"])
    practical = MIN_EFFECT.get(str(estimate.get("metric")), 0.03)
    se_estimate = estimate.get("se")
    se_reference = baseline.get("se")
    if se_estimate is None or se_reference is None:
        return {
            "outcome": "evidence_insufficient",
            "z": None,
            "difference": _rounded(difference),
            "practical_threshold": practical,
        }
    standard_error = math.sqrt(float(se_estimate) ** 2 + float(se_reference) ** 2)
    if standard_error <= _EPS:
        z_value = 0.0 if abs(difference) <= _EPS else math.copysign(math.inf, difference)
    else:
        z_value = difference / standard_error
    return {
        "outcome": "above_practical_threshold"
        if abs(difference) >= practical
        else "below_practical_threshold",
        "z": None if math.isinf(z_value) else _rounded(z_value),
        "difference": _rounded(difference),
        "practical_threshold": practical,
    }


def heterogeneity_checks(
    week: str,
    version: str,
    site_estimates: Mapping[str, Mapping[str, Mapping[str, Any]]],
    operator_estimates: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Describe site/operator differences without turning them into alerts.

    Personnel mix and genuine differences between reviewers are a normal part
    of review behaviour.  They are reported so a reader can attribute a move,
    but only ``supported_shift`` in the engine is ever called a regression.
    """
    checks: list[dict[str, Any]] = []
    for metric in METRICS:
        site_values = {
            site: record[metric]
            for site, record in site_estimates.items()
            if record.get(metric, {}).get("estimate") is not None
            and record[metric].get("status") == "ok"
        }
        if len(site_values) >= 2:
            values = [record["estimate"] for record in site_values.values()]
            spread = max(values) - min(values)
            if spread >= 0.15:
                checks.append(
                    {
                        "week": week,
                        "check": "site_difference",
                        "status": "noted",
                        "detail": {
                            "metric": metric,
                            "version": version,
                            "site_estimates": {
                                site: record["estimate"]
                                for site, record in sorted(site_values.items())
                            },
                            "spread": round(spread, 4),
                        },
                    }
                )
        operator_values = {
            operator: record[metric]
            for operator, record in operator_estimates.items()
            if record.get(metric, {}).get("estimate") is not None
            and record[metric].get("status") == "ok"
            and record[metric].get("completed", 0) >= 5
        }
        if len(operator_values) >= 2:
            values = [record["estimate"] for record in operator_values.values()]
            spread = max(values) - min(values)
            if spread >= 0.30:
                checks.append(
                    {
                        "week": week,
                        "check": "operator_difference",
                        "status": "noted",
                        "detail": {
                            "metric": metric,
                            "version": version,
                            "spread": round(spread, 4),
                            "operators": len(operator_values),
                        },
                    }
                )
    return checks


def _shares(values: Iterable[str]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {}
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = float(len(values))
    return {key: count / total for key, count in sorted(counts.items())}


def _tv_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _sample_size_by_stratum(rows: Sequence[Sample]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in rows:
        counts[sample.stratum] = counts.get(sample.stratum, 0) + 1
    return counts


def week_design_checks(
    week: str,
    rows: Sequence[Sample],
    frame: Mapping[tuple[str, str], int],
    reference_rows: Sequence[Sample],
    reference_frame: Mapping[tuple[str, str], int],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return ``(checks, blocking_check_names)`` for one week.

    ``blocking`` means the population estimate is not formed at all; the other
    checks are diagnostics that explain a movement without invalidating it.
    """
    checks: list[dict[str, Any]] = []
    blocking: list[str] = []

    def add(check: str, status: str, detail: Mapping[str, Any]) -> None:
        checks.append({"week": week, "check": check, "status": status, "detail": dict(detail)})
        if status == "fail":
            blocking.append(check)

    week_frame = {
        stratum: population
        for (frame_week, stratum), population in frame.items()
        if frame_week == week
    }
    if not week_frame:
        add("sampling_frame_present", "fail", {"reason": "no_frame_for_week"})
        return checks, blocking
    if not rows:
        add("sampling_frame_present", "pass", {"strata": sorted(week_frame)})
        add("sample_present", "fail", {"reason": "no_sample_for_week"})
        return checks, blocking

    sample_strata = {sample.stratum for sample in rows}
    frame_strata = {stratum for stratum, population in week_frame.items() if population > 0}
    missing_frame = sorted(sample_strata - set(week_frame))
    missing_sample = sorted(frame_strata - sample_strata)
    add(
        "sampling_frame_present",
        "fail" if missing_frame or missing_sample else "pass",
        {
            "frame_strata": sorted(week_frame),
            "sample_strata": sorted(sample_strata),
            "sample_strata_not_in_frame": missing_frame,
            "frame_strata_without_sample": missing_sample,
        },
    )
    if missing_sample:
        blocking.append("sampling_frame_present")

    illegal = sorted(
        {sample.stratum for sample in rows if not (0.0 < sample.probability <= 1.0)}
    )
    add(
        "inclusion_probability",
        "fail" if illegal else "pass",
        {
            "range": [
                min((sample.probability for sample in rows), default=None),
                max((sample.probability for sample in rows), default=None),
            ],
            "illegal_strata": illegal,
        },
    )
    for stratum in sorted(week_frame):
        probabilities = sorted({round(sample.probability, 12) for sample in rows if sample.stratum == stratum})
        if len(probabilities) > 1:
            add(
                "inclusion_probability_consistency",
                "warn",
                {"stratum": stratum, "probabilities": probabilities},
            )

    # Declared design versus realised sample size, checked per version and
    # stratum.  Deliberately over-sampling a stratum shows up here as a change
    # of sampling structure, not as a change of review behaviour.  When two
    # versions are live in the same week, each draws from the same frame at its
    # own probability, so each is checked separately.
    observed: dict[tuple[str, str], int] = {}
    for sample in rows:
        key = (sample.version, sample.stratum)
        observed[key] = observed.get(key, 0) + 1
    versions = sorted({sample.version for sample in rows})
    for version in versions:
        for stratum, population in sorted(week_frame.items()):
            probabilities = [
                sample.probability
                for sample in rows
                if sample.stratum == stratum and sample.version == version
            ]
            if not probabilities or population <= 0:
                continue
            expected = population * (sum(probabilities) / len(probabilities))
            actual = observed.get((version, stratum), 0)
            if expected >= 5 and (actual < 0.5 * expected or actual > 2.0 * expected):
                add(
                    "declared_sample_size",
                    "warn",
                    {
                        "stratum": stratum,
                        "version": version,
                        "expected": round(expected, 3),
                        "observed": actual,
                        "ratio": round(actual / expected, 3),
                        "note": "the realised sample of this stratum differs from the declared "
                        "design; an unweighted count ratio would move with the allocation alone",
                    },
                )

    reference_alloc = _shares(sample.stratum for sample in reference_rows)
    current_alloc = _shares(sample.stratum for sample in rows)
    if reference_alloc:
        distance = _tv_distance(reference_alloc, current_alloc)
        if distance > 0.10:
            add(
                "stratum_allocation_shift",
                "warn",
                {
                    "total_variation_distance": round(distance, 4),
                    "reference_shares": {key: round(value, 4) for key, value in reference_alloc.items()},
                    "current_shares": {key: round(value, 4) for key, value in current_alloc.items()},
                },
            )

    completed = [sample for sample in rows if sample.review_complete]
    reference_completed = [sample for sample in reference_rows if sample.review_complete]
    if completed and reference_completed:
        operator_shift = _tv_distance(
            _shares(sample.operator_id for sample in reference_completed),
            _shares(sample.operator_id for sample in completed),
        )
        if operator_shift > 0.15:
            add(
                "personnel_mix_shift",
                "warn",
                {
                    "total_variation_distance": round(operator_shift, 4),
                    "reference_operators": sorted({sample.operator_id for sample in reference_completed}),
                    "current_operators": sorted({sample.operator_id for sample in completed}),
                },
            )
        site_shift = _tv_distance(
            _shares(sample.site_id for sample in reference_completed),
            _shares(sample.site_id for sample in completed),
        )
        if site_shift > 0.15:
            add(
                "site_mix_shift",
                "warn",
                {
                    "total_variation_distance": round(site_shift, 4),
                    "reference_sites": sorted({sample.site_id for sample in reference_completed}),
                    "current_sites": sorted({sample.site_id for sample in completed}),
                },
            )

    reference_population: dict[str, list[int]] = {}
    reference_weeks = {sample.week for sample in reference_rows}
    for (frame_week, stratum), population in reference_frame.items():
        if frame_week in reference_weeks and population > 0:
            reference_population.setdefault(stratum, []).append(population)
    for stratum, population in sorted(week_frame.items()):
        values = reference_population.get(stratum)
        if values:
            baseline = sorted(values)[len(values) // 2]
            if baseline > 0 and abs(population - baseline) / baseline > 0.20:
                add(
                    "frame_population_shift",
                    "warn",
                    {
                        "stratum": stratum,
                        "reference_population": baseline,
                        "current_population": population,
                    },
                )

    reference_probability: dict[str, list[float]] = {}
    for sample in reference_rows:
        reference_probability.setdefault(sample.stratum, []).append(sample.probability)
    current_probability: dict[str, list[float]] = {}
    for sample in rows:
        current_probability.setdefault(sample.stratum, []).append(sample.probability)
    for stratum, probabilities in sorted(current_probability.items()):
        values = reference_probability.get(stratum)
        if not values:
            continue
        baseline = sum(values) / len(values)
        current = sum(probabilities) / len(probabilities)
        if baseline > 0 and abs(current - baseline) / baseline > 0.20:
            add(
                "inclusion_probability_shift",
                "warn",
                {
                    "stratum": stratum,
                    "reference_probability": round(baseline, 6),
                    "current_probability": round(current, 6),
                },
            )

    for stratum in sample_strata:
        if stratum in week_frame and observed.get(stratum, 0) > week_frame[stratum]:
            add(
                "sample_size_within_frame",
                "fail",
                {"stratum": stratum, "sample": observed.get(stratum, 0), "population": week_frame[stratum]},
            )

    return checks, blocking
