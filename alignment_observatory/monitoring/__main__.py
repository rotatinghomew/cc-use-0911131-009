from __future__ import annotations

"""``python3 -m alignment_observatory.monitoring`` entry point.

Returns JSON with ``series_id``, ``estimates``, ``alerts``, ``design_checks``
and ``reference_version``.  Each estimate carries ``week``, ``version``,
``metric``, ``estimate``, ``interval`` and ``status``; each alert carries
``alert_id``, ``week`` and ``reason``.

The existing blinded-sampling entry point
(``python3 -m alignment_observatory.lab.audit_cli``) is untouched.
"""

import argparse
import json
import sys
from pathlib import Path


def _failure(message: str, code: int = 2) -> int:
    print(json.dumps({"error": message}, ensure_ascii=False, sort_keys=True))
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m alignment_observatory.monitoring",
        description="Design-based weekly monitoring of review behaviour against a frozen reference.",
    )
    parser.add_argument("--input", required=True, help="analysis input JSON")
    parser.add_argument("--database", required=True, help="SQLite database for monitoring state")
    parser.add_argument("--output", required=True, help="output JSON path")
    args = parser.parse_args(argv)

    from alignment_observatory.monitoring.engine import run_monitoring

    try:
        raw = json.loads(Path(args.input).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return _failure(f"cannot read input: {error}")

    try:
        result = run_monitoring(raw, args.database)
    except ValueError as error:
        return _failure(str(error))

    output = Path(args.output)
    if output.parent and not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "series_id": result["series_id"],
                "reference_version": result["reference_version"],
                "estimates": len(result["estimates"]),
                "alerts": len(result["alerts"]),
                "design_checks": len(result["design_checks"]),
                "output": str(output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
