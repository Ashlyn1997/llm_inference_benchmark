#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PlotResultsError(ValueError):
    """Raised when benchmark result files cannot be plotted safely."""


@dataclass(frozen=True)
class ResultPoint:
    concurrency: int
    output_token_throughput_tps: float
    request_throughput_rps: float
    ttft_average_ms: float
    ttft_p95_ms: float
    tpot_average_ms: float
    total_latency_average_ms: float
    avg_gpu_utilization_pct: float | None
    avg_memory_used_mib: float | None


def _value_at(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise PlotResultsError(f"missing required field '{path}'")
        value = value[part]
    return value


def _required_number(data: dict[str, Any], path: str, source: Path) -> float:
    value = _value_at(data, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlotResultsError(
            f"Invalid {source}: field '{path}' must be a number, got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise PlotResultsError(f"Invalid {source}: field '{path}' must be finite")
    return number


def _optional_number(data: dict[str, Any], path: str, source: Path) -> float | None:
    try:
        value = _value_at(data, path)
    except PlotResultsError:
        return None
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlotResultsError(
            f"Invalid {source}: field '{path}' must be a number or null, got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise PlotResultsError(f"Invalid {source}: field '{path}' must be finite")
    return number


def _point_from_entry(entry: Any, source: Path, section: str) -> ResultPoint:
    if not isinstance(entry, dict):
        raise PlotResultsError(f"Invalid {source}: {section} entries must be objects")

    concurrency_value = _required_number(entry, "concurrency", source)
    if concurrency_value <= 0 or not concurrency_value.is_integer():
        raise PlotResultsError(
            f"Invalid {source}: concurrency must be a positive integer"
        )

    if section == "comparison":
        gpu_utilization_path = "avg_gpu_utilization_pct"
        memory_used_path = "avg_memory_used_mib"
    else:
        gpu_utilization_path = "metadata.gpu.avg_utilization_pct"
        memory_used_path = "metadata.gpu.avg_memory_used_mib"

    return ResultPoint(
        concurrency=int(concurrency_value),
        output_token_throughput_tps=_required_number(
            entry, "output_token_throughput_tps", source
        ),
        request_throughput_rps=_required_number(
            entry, "request_throughput_rps", source
        ),
        ttft_average_ms=_required_number(entry, "ttft_ms.average", source),
        ttft_p95_ms=_required_number(entry, "ttft_ms.p95", source),
        tpot_average_ms=_required_number(entry, "tpot_ms.average", source),
        total_latency_average_ms=_required_number(
            entry, "total_latency_ms.average", source
        ),
        avg_gpu_utilization_pct=_optional_number(
            entry, gpu_utilization_path, source
        ),
        avg_memory_used_mib=_optional_number(entry, memory_used_path, source),
    )


def load_results(result_directories: list[Path]) -> list[ResultPoint]:
    """Load and normalize one or more combined summary files."""
    points: list[ResultPoint] = []
    for result_directory in result_directories:
        summary_path = Path(result_directory) / "combined_summary.json"
        if not summary_path.is_file():
            raise PlotResultsError(
                f"Missing benchmark summary: {summary_path}"
            )
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise PlotResultsError(
                f"Invalid JSON in {summary_path}: line {exc.lineno}, column {exc.colno}"
            ) from exc
        except OSError as exc:
            raise PlotResultsError(f"Unable to read {summary_path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise PlotResultsError(f"Invalid {summary_path}: root must be an object")
        comparison = payload.get("comparison")
        runs = payload.get("runs")
        if comparison is not None and not isinstance(comparison, list):
            raise PlotResultsError(
                f"Invalid {summary_path}: 'comparison' must be a list"
            )
        if runs is not None and not isinstance(runs, list):
            raise PlotResultsError(f"Invalid {summary_path}: 'runs' must be a list")

        if comparison:
            entries = comparison
            section = "comparison"
        elif runs:
            entries = runs
            section = "runs"
        else:
            raise PlotResultsError(
                f"Invalid {summary_path}: no runs/comparison data"
            )
        points.extend(
            _point_from_entry(entry, summary_path, section) for entry in entries
        )

    if not points:
        raise PlotResultsError("No benchmark data was loaded")
    points.sort(key=lambda point: point.concurrency)
    duplicates = sorted(
        concurrency
        for concurrency in {point.concurrency for point in points}
        if sum(point.concurrency == concurrency for point in points) > 1
    )
    if duplicates:
        raise PlotResultsError(
            "Duplicate concurrency values across summaries: "
            + ", ".join(str(value) for value in duplicates)
        )
    return points


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlotResultsError(
            "matplotlib is required; run 'pip install -r requirements.txt'"
        ) from exc
    return plt


def create_plots(points: list[ResultPoint], output_directory: Path) -> list[Path]:
    """Render benchmark comparison plots and return their paths."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()
    concurrencies = [point.concurrency for point in points]
    created: list[Path] = []

    def save_single_line(
        filename: str,
        values: list[float],
        title: str,
        ylabel: str,
        label: str | None = None,
    ) -> None:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(concurrencies, values, marker="o", linewidth=2, label=label)
        axis.set_title(title)
        axis.set_xlabel("Concurrency")
        axis.set_ylabel(ylabel)
        axis.set_xticks(concurrencies)
        axis.grid(True, linestyle="--", alpha=0.4)
        if label:
            axis.legend()
        figure.tight_layout()
        path = output_directory / filename
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        created.append(path)

    save_single_line(
        "concurrency_vs_throughput.png",
        [point.output_token_throughput_tps for point in points],
        "Concurrency vs Output Throughput",
        "Output Throughput (tokens/s)",
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        concurrencies,
        [point.ttft_average_ms / 1000 for point in points],
        marker="o",
        linewidth=2,
        label="Average TTFT",
    )
    axis.plot(
        concurrencies,
        [point.total_latency_average_ms / 1000 for point in points],
        marker="o",
        linewidth=2,
        label="Average E2E Latency",
    )
    axis.set_title("Concurrency vs Latency")
    axis.set_xlabel("Concurrency")
    axis.set_ylabel("Latency (s)")
    axis.set_xticks(concurrencies)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    latency_path = output_directory / "concurrency_vs_latency.png"
    figure.savefig(latency_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    created.append(latency_path)

    save_single_line(
        "concurrency_vs_tpot.png",
        [point.tpot_average_ms for point in points],
        "Concurrency vs TPOT",
        "TPOT (ms/token)",
    )

    if any(point.avg_gpu_utilization_pct is not None for point in points):
        save_single_line(
            "concurrency_vs_gpu_utilization.png",
            [
                point.avg_gpu_utilization_pct
                if point.avg_gpu_utilization_pct is not None
                else math.nan
                for point in points
            ],
            "Concurrency vs GPU Utilization",
            "Average GPU Utilization (%)",
        )
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one or more llm_inference_benchmark combined summaries"
    )
    parser.add_argument(
        "result_directories",
        nargs="+",
        type=Path,
        help="directories containing combined_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plots"),
        help="directory for generated PNG files",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        points = load_results(args.result_directories)
        created = create_plots(points, args.output_dir)
    except PlotResultsError as exc:
        parser.error(str(exc))
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
