from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import MetricSummary, RequestResult


def write_request_results(
    output_dir: str | Path,
    results: list[RequestResult],
) -> tuple[Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "request_results.json"
    csv_path = directory / "request_results.csv"
    json_path.write_text(
        json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    rows = [result.to_dict() for result in results]
    fields = (
        list(rows[0])
        if rows
        else [*RequestResult.__dataclass_fields__, "tpot_ms"]
    )
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, json_path


def write_summary(output_dir: str | Path, summary: MetricSummary) -> Path:
    path = Path(output_dir) / "summary.json"
    path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_combined_summary(output_dir: str | Path, summaries: list[MetricSummary]) -> Path:
    path = Path(output_dir) / "summary.json"
    payload: dict[str, Any] = {
        "runs": [summary.to_dict() for summary in summaries],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
