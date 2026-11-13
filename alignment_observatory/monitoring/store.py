from __future__ import annotations

"""Append-only monitoring state.

The SQLite file is shared with the existing sampling/registry ledger.  Tables
created here are namespaced ``monitoring_*`` so the two workflows can share a
database without touching each other's rows.

Key invariants:

* a monitoring *series* is pinned to one frozen reference period; changing the
  reference creates a new series and never rewrites the old one;
* every alert carries a deterministic ``alert_key``; replaying the same input
  into the same database inserts nothing new;
* earlier alerts are never deleted, including when a reference changes.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from alignment_observatory.sampling.cohorts import stable_json

SCHEMA = """
CREATE TABLE IF NOT EXISTS monitoring_series(
    series_id TEXT PRIMARY KEY,
    design_signature TEXT NOT NULL UNIQUE,
    reference_version TEXT NOT NULL,
    reference_start TEXT NOT NULL,
    reference_end TEXT NOT NULL,
    reference_json TEXT NOT NULL,
    thresholds_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS monitoring_evaluations(
    series_id TEXT NOT NULL,
    week TEXT NOT NULL,
    version TEXT NOT NULL,
    metric TEXT NOT NULL,
    signal TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY(series_id, week, version, metric)
);
CREATE TABLE IF NOT EXISTS monitoring_design_evaluations(
    series_id TEXT NOT NULL,
    week TEXT NOT NULL,
    check_name TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    detail_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY(series_id, week, check_name, scope)
);
CREATE TABLE IF NOT EXISTS monitoring_alerts(
    alert_id TEXT PRIMARY KEY,
    series_id TEXT NOT NULL,
    alert_key TEXT NOT NULL,
    week TEXT NOT NULL,
    reason TEXT NOT NULL,
    category TEXT NOT NULL,
    version TEXT,
    metric TEXT,
    status TEXT NOT NULL,
    supersedes TEXT,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(series_id, alert_key)
);
CREATE TABLE IF NOT EXISTS monitoring_events(
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    series_id TEXT NOT NULL,
    at TEXT NOT NULL,
    action TEXT NOT NULL,
    body_json TEXT NOT NULL
);
"""


def _now() -> str:
    from alignment_observatory.sampling.cohorts import utc_now

    return utc_now()


class MonitoringStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._lock = threading.RLock()
        parent = Path(self.path).parent
        if parent and str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)
            self._migrate(connection)

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """Additive migrations only; existing monitoring rows are preserved."""
        columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(monitoring_design_evaluations)"
            ).fetchall()
        }
        if "scope" not in columns:
            # Rebuild so the primary key includes ``scope`` (one week may carry
            # several instances of the same check).  Existing rows get a scope
            # derived from their payload, so nothing is lost.
            connection.execute(
                "ALTER TABLE monitoring_design_evaluations RENAME TO monitoring_design_evaluations_old"
            )
            connection.execute(
                """CREATE TABLE monitoring_design_evaluations(
                       series_id TEXT NOT NULL, week TEXT NOT NULL, check_name TEXT NOT NULL,
                       scope TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
                       detail_json TEXT NOT NULL, recorded_at TEXT NOT NULL,
                       PRIMARY KEY(series_id, week, check_name, scope))"""
            )
            connection.execute(
                """INSERT OR REPLACE INTO monitoring_design_evaluations
                   (series_id,week,check_name,scope,status,detail_json,recorded_at)
                   SELECT series_id,week,check_name,detail_json,status,detail_json,recorded_at
                   FROM monitoring_design_evaluations_old"""
            )
            connection.execute("DROP TABLE monitoring_design_evaluations_old")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                yield connection
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    # -- series -----------------------------------------------------------

    def find_series(self, design_signature: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM monitoring_series WHERE design_signature=?",
                (design_signature,),
            ).fetchone()
        return None if row is None else dict(row)

    def create_series(
        self,
        series_id: str,
        design_signature: str,
        reference_version: str,
        reference_start: str,
        reference_end: str,
        reference: Mapping[str, Any],
        thresholds: Mapping[str, Any],
    ) -> dict[str, Any]:
        created = _now()
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO monitoring_series VALUES(?,?,?,?,?,?,?,?)",
                (
                    series_id,
                    design_signature,
                    reference_version,
                    reference_start,
                    reference_end,
                    stable_json(dict(reference)),
                    stable_json(dict(thresholds)),
                    created,
                ),
            )
            connection.execute(
                "INSERT INTO monitoring_events(series_id,at,action,body_json) VALUES(?,?,?,?)",
                (series_id, created, "series_created", stable_json({"reference_version": reference_version})),
            )
        return self.find_series(design_signature) or {}

    def load_reference(self, series_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT reference_json,reference_version,reference_start,reference_end,thresholds_json FROM monitoring_series WHERE series_id=?",
                (series_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown monitoring series {series_id}")
        return {
            "reference": json.loads(row["reference_json"]),
            "reference_version": row["reference_version"],
            "reference_start": row["reference_start"],
            "reference_end": row["reference_end"],
            "thresholds": json.loads(row["thresholds_json"]),
        }

    def series_history(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT s.series_id,s.reference_version,s.reference_start,s.reference_end,s.created_at,
                          (SELECT COUNT(*) FROM monitoring_alerts a WHERE a.series_id=s.series_id) AS alert_count
                   FROM monitoring_series s ORDER BY s.created_at,s.series_id"""
            ).fetchall()
        return [dict(row) for row in rows]

    # -- evaluations ------------------------------------------------------

    def save_evaluations(
        self, series_id: str, records: Iterable[Mapping[str, Any]]
    ) -> None:
        records = list(records)
        if not records:
            return
        stamped = _now()
        with self.transaction() as connection:
            for record in records:
                payload = stable_json(dict(record))
                existing = connection.execute(
                    "SELECT payload_json FROM monitoring_evaluations WHERE series_id=? AND week=? AND version=? AND metric=?",
                    (series_id, record["week"], record["version"], record["metric"]),
                ).fetchone()
                if existing is not None and existing["payload_json"] == payload:
                    continue
                connection.execute(
                    """INSERT INTO monitoring_evaluations(series_id,week,version,metric,signal,payload_json,recorded_at)
                       VALUES(?,?,?,?,?,?,?)
                       ON CONFLICT(series_id,week,version,metric) DO UPDATE SET
                         signal=excluded.signal,payload_json=excluded.payload_json,recorded_at=excluded.recorded_at""",
                    (
                        series_id,
                        record["week"],
                        record["version"],
                        record["metric"],
                        record["signal"],
                        payload,
                        stamped,
                    ),
                )

    def load_evaluations(self, series_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT week,version,metric,signal,payload_json FROM monitoring_evaluations WHERE series_id=? ORDER BY week,version,metric",
                (series_id,),
            ).fetchall()
        return [
            {
                "week": row["week"],
                "version": row["version"],
                "metric": row["metric"],
                "signal": row["signal"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def save_design_evaluations(
        self, series_id: str, records: Iterable[Mapping[str, Any]]
    ) -> None:
        records = list(records)
        if not records:
            return
        stamped = _now()
        with self.transaction() as connection:
            for record in records:
                detail_json = stable_json(dict(record["detail"]))
                # One week can carry several instances of the same check (one
                # per version, metric or stratum).  ``scope`` keeps them apart
                # without letting them overwrite each other on re-run.
                scope = record.get("scope") or detail_json
                connection.execute(
                    """INSERT INTO monitoring_design_evaluations(series_id,week,check_name,scope,status,detail_json,recorded_at)
                       VALUES(?,?,?,?,?,?,?)
                       ON CONFLICT(series_id,week,check_name,scope) DO UPDATE SET
                         status=excluded.status,detail_json=excluded.detail_json,recorded_at=excluded.recorded_at""",
                    (
                        series_id,
                        record["week"],
                        record["check"],
                        scope,
                        record["status"],
                        detail_json,
                        stamped,
                    ),
                )

    def load_design_evaluations(self, series_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT week,check_name,scope,status,detail_json FROM monitoring_design_evaluations WHERE series_id=? ORDER BY week,check_name,scope",
                (series_id,),
            ).fetchall()
        return [
            {
                "week": row["week"],
                "check": row["check_name"],
                "scope": row["scope"],
                "status": row["status"],
                "detail": json.loads(row["detail_json"]),
            }
            for row in rows
        ]

    # -- alerts -----------------------------------------------------------

    def alert_exists(self, series_id: str, alert_key: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM monitoring_alerts WHERE series_id=? AND alert_key=?",
                (series_id, alert_key),
            ).fetchone()
        return row is not None

    def insert_alert(self, series_id: str, payload: Mapping[str, Any]) -> bool:
        """Insert an alert, or restore it if an earlier revision retracted it.

        ``alert_key`` is deterministic per (series, week, version, metric,
        decision).  Replaying the same input, or revising a week back to a
        conclusion that previously existed, therefore never creates a second
        row for the same week and decision.
        """
        alert = dict(payload)
        created = _now()
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT alert_id,status FROM monitoring_alerts WHERE series_id=? AND alert_key=?",
                (series_id, alert["alert_key"]),
            ).fetchone()
            if existing is not None:
                if existing["status"] == "superseded":
                    restored = dict(alert)
                    restored["status"] = "open"
                    restored.pop("superseded_by", None)
                    restored.pop("superseded_at", None)
                    connection.execute(
                        "UPDATE monitoring_alerts SET status='open',payload_json=? WHERE alert_id=?",
                        (stable_json(restored), existing["alert_id"]),
                    )
                    connection.execute(
                        "INSERT INTO monitoring_events(series_id,at,action,body_json) VALUES(?,?,?,?)",
                        (
                            series_id,
                            created,
                            "alert_restored",
                            stable_json({"alert_id": existing["alert_id"]}),
                        ),
                    )
                return False
            connection.execute(
                """INSERT INTO monitoring_alerts
                   (alert_id,series_id,alert_key,week,reason,category,version,metric,status,supersedes,payload_json,created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    alert["alert_id"],
                    series_id,
                    alert["alert_key"],
                    alert["week"],
                    alert["reason"],
                    alert["category"],
                    alert.get("version"),
                    alert.get("metric"),
                    alert.get("status", "open"),
                    alert.get("supersedes"),
                    stable_json(alert),
                    created,
                ),
            )
            connection.execute(
                "INSERT INTO monitoring_events(series_id,at,action,body_json) VALUES(?,?,?,?)",
                (
                    series_id,
                    created,
                    "alert_raised",
                    stable_json({"alert_id": alert["alert_id"], "reason": alert["reason"]}),
                ),
            )
        return True

    def load_alerts(self, series_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json,created_at,status FROM monitoring_alerts WHERE series_id=? ORDER BY week,alert_id",
                (series_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = json.loads(row["payload_json"])
            item["created_at"] = row["created_at"]
            item["status"] = row["status"]
            result.append(item)
        return result

    def apply_supersedes(
        self, series_id: str, superseded: Iterable[tuple[str, str]]
    ) -> None:
        """Mark earlier alerts superseded without deleting them.

        ``superseded`` holds ``(alert_id, superseded_by)`` pairs.  A revision
        of the same week's conclusion therefore leaves an audit trail instead
        of rewriting history or raising a duplicate.
        """
        entries = list(superseded)
        if not entries:
            return
        stamped = _now()
        with self.transaction() as connection:
            for alert_id, replacement in entries:
                row = connection.execute(
                    "SELECT payload_json FROM monitoring_alerts WHERE alert_id=? AND series_id=?",
                    (alert_id, series_id),
                ).fetchone()
                if row is None:
                    continue
                payload = json.loads(row["payload_json"])
                if payload.get("status") == "superseded":
                    continue
                payload["status"] = "superseded"
                payload["superseded_by"] = replacement
                payload["superseded_at"] = stamped
                connection.execute(
                    "UPDATE monitoring_alerts SET status='superseded',payload_json=? WHERE alert_id=?",
                    (stable_json(payload), alert_id),
                )
                connection.execute(
                    "INSERT INTO monitoring_events(series_id,at,action,body_json) VALUES(?,?,?,?)",
                    (
                        series_id,
                        stamped,
                        "alert_superseded",
                        stable_json(
                            {"alert_id": alert_id, "superseded_by": replacement}
                        ),
                    ),
                )
