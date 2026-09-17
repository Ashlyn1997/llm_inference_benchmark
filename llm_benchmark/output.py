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
    path = Path(output_dir) / "combined_summary.json"

    def comparison_row(summary: MetricSummary) -> dict[str, Any]:
        gpu = summary.metadata.get("gpu", {})
        return {
            "concurrency": summary.concurrency,
            "request_throughput_rps": summary.request_throughput_rps,
            "output_token_throughput_tps": summary.output_token_throughput_tps,
            "ttft_ms": summary.ttft_ms,
            "tpot_ms": summary.tpot_ms,
            "total_latency_ms": summary.total_latency_ms,
            "avg_gpu_utilization_pct": gpu.get("avg_utilization_pct"),
            "max_gpu_utilization_pct": gpu.get("max_utilization_pct"),
            "avg_memory_used_mib": gpu.get("avg_memory_used_mib"),
            "max_memory_used_mib": gpu.get("max_memory_used_mib"),
            "max_temperature_c": gpu.get("max_temperature_c"),
            "avg_power_w": gpu.get("avg_power_w"),
        }

    payload: dict[str, Any] = {
        "runs": [summary.to_dict() for summary in summaries],
        "comparison": [comparison_row(summary) for summary in summaries],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
