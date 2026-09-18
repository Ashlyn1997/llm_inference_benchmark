#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RESULT_DIRECTORY_PATTERN = re.compile(
    r"kv(?P<cache_size>\d+(?:\.\d+)?)g_c(?P<concurrency>\d+)",
    re.IGNORECASE,
)


class PlotKVCacheResultsError(ValueError):
    """Raised when KV cache benchmark data cannot be plotted safely."""


@dataclass(frozen=True)
class KVCachePoint:
    cache_size_gib: float
    concurrency: int
    output_token_throughput_tps: float
    ttft_average_ms: float
    total_latency_average_ms: float
    avg_memory_used_mib: float


def parse_result_directory_name(result_directory: Path) -> tuple[float, int]:
    """Parse kv{size}g_c{concurrency} from an exact result directory name."""
    name = Path(result_directory).name
    match = RESULT_DIRECTORY_PATTERN.fullmatch(name)
    if match is None:
        raise PlotKVCacheResultsError(
            f"Cannot parse KV cache size and concurrency from directory name "
            f"'{name}'; expected a name like 'kv4g_c32'"
        )
    cache_size_gib = float(match.group("cache_size"))
    concurrency = int(match.group("concurrency"))
    if cache_size_gib <= 0 or concurrency <= 0:
        raise PlotKVCacheResultsError(
            f"Invalid result directory name '{name}': values must be positive"
        )
    return cache_size_gib, concurrency


def _value_at(data: dict[str, Any], path: str) -> Any:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise PlotKVCacheResultsError(f"missing required field '{path}'")
        value = value[part]
    return value


def _required_number(data: dict[str, Any], path: str, source: Path) -> float:
    value = _value_at(data, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlotKVCacheResultsError(
            f"Invalid {source}: field '{path}' must be a number, got {value!r}"
        )
    number = float(value)
    if not math.isfinite(number):
        raise PlotKVCacheResultsError(
            f"Invalid {source}: field '{path}' must be finite"
        )
    return number


def _load_summary_entry(summary_path: Path, concurrency: int) -> tuple[dict[str, Any], str]:
    if not summary_path.is_file():
        raise PlotKVCacheResultsError(f"Missing benchmark summary: {summary_path}")
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlotKVCacheResultsError(
            f"Invalid JSON in {summary_path}: line {exc.lineno}, column {exc.colno}"
        ) from exc
    except OSError as exc:
        raise PlotKVCacheResultsError(f"Unable to read {summary_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PlotKVCacheResultsError(f"Invalid {summary_path}: root must be an object")

    comparison = payload.get("comparison")
    runs = payload.get("runs")
    if comparison is not None and not isinstance(comparison, list):
        raise PlotKVCacheResultsError(
            f"Invalid {summary_path}: 'comparison' must be a list"
        )
    if runs is not None and not isinstance(runs, list):
        raise PlotKVCacheResultsError(f"Invalid {summary_path}: 'runs' must be a list")
    if comparison:
        entries = comparison
        section = "comparison"
    elif runs:
        entries = runs
        section = "runs"
    else:
        raise PlotKVCacheResultsError(
            f"Invalid {summary_path}: no runs/comparison data"
        )

    matching = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("concurrency") == concurrency
    ]
    if len(matching) != 1:
        available = [
            entry.get("concurrency")
            for entry in entries
            if isinstance(entry, dict)
        ]
        raise PlotKVCacheResultsError(
            f"Concurrency mismatch for {summary_path}: directory expects "
            f"{concurrency}, summary contains {available}"
        )
    return matching[0], section


def load_kv_cache_results(result_directories: list[Path]) -> list[KVCachePoint]:
    """Load result directories and associate each summary with its KV cache size."""
    points: list[KVCachePoint] = []
    seen: set[tuple[float, int]] = set()
    for result_directory in result_directories:
        result_directory = Path(result_directory)
        cache_size_gib, concurrency = parse_result_directory_name(result_directory)
        summary_path = result_directory / "combined_summary.json"
        entry, section = _load_summary_entry(summary_path, concurrency)
        key = (cache_size_gib, concurrency)
        if key in seen:
            raise PlotKVCacheResultsError(
                f"Duplicate KV cache/concurrency result: {cache_size_gib:g} GiB, "
                f"concurrency {concurrency}"
            )
        seen.add(key)
        memory_path = (
            "avg_memory_used_mib"
            if section == "comparison"
            else "metadata.gpu.avg_memory_used_mib"
        )
        points.append(
            KVCachePoint(
                cache_size_gib=cache_size_gib,
                concurrency=concurrency,
                output_token_throughput_tps=_required_number(
                    entry, "output_token_throughput_tps", summary_path
                ),
                ttft_average_ms=_required_number(
                    entry, "ttft_ms.average", summary_path
                ),
                total_latency_average_ms=_required_number(
                    entry, "total_latency_ms.average", summary_path
                ),
                avg_memory_used_mib=_required_number(
                    entry, memory_path, summary_path
                ),
            )
        )
    points.sort(key=lambda point: (point.cache_size_gib, point.concurrency))
    return points


def _load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise PlotKVCacheResultsError(
            "matplotlib is required; run 'pip install -r requirements.txt'"
        ) from exc
    return plt


def create_kv_cache_plots(
    points: list[KVCachePoint], output_directory: Path
) -> list[Path]:
    """Create KV cache size comparison plots grouped by concurrency."""
    if not points:
        raise PlotKVCacheResultsError("No KV cache benchmark data was loaded")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    plt = _load_pyplot()
    created: list[Path] = []
    concurrencies = sorted({point.concurrency for point in points})
    cache_sizes = sorted({point.cache_size_gib for point in points})

    def save_grouped_plot(
        filename: str,
        title: str,
        ylabel: str,
        value_getter,
    ) -> None:
        figure, axis = plt.subplots(figsize=(8, 5))
        for concurrency in concurrencies:
            series = sorted(
                (point for point in points if point.concurrency == concurrency),
                key=lambda point: point.cache_size_gib,
            )
            axis.plot(
                [point.cache_size_gib for point in series],
                [value_getter(point) for point in series],
                marker="o",
                linewidth=2,
                label=f"Concurrency {concurrency}",
            )
        axis.set_title(title)
        axis.set_xlabel("KV Cache Size (GiB)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(cache_sizes)
        axis.grid(True, linestyle="--", alpha=0.4)
        axis.legend()
        figure.tight_layout()
        path = output_directory / filename
        figure.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        created.append(path)

    save_grouped_plot(
        "kv_cache_vs_throughput.png",
        "KV Cache Size vs Output Throughput",
        "Output Throughput (tokens/s)",
        lambda point: point.output_token_throughput_tps,
    )
    save_grouped_plot(
        "kv_cache_vs_ttft.png",
        "KV Cache Size vs TTFT",
        "Average TTFT (seconds)",
        lambda point: point.ttft_average_ms / 1000,
    )
    save_grouped_plot(
        "kv_cache_vs_e2e_latency.png",
        "KV Cache Size vs E2E Latency",
        "Average E2E Latency (seconds)",
        lambda point: point.total_latency_average_ms / 1000,
    )
    save_grouped_plot(
        "kv_cache_vs_gpu_memory.png",
        "KV Cache Size vs GPU Memory Usage",
        "Average GPU Memory Used (MiB)",
        lambda point: point.avg_memory_used_mib,
    )
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot KV cache pressure benchmark results"
    )
    parser.add_argument(
        "result_directories",
        nargs="+",
        type=Path,
        help="directories named like kv4g_c32 containing combined_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/kv_plots"),
        help="directory for generated PNG files",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        points = load_kv_cache_results(args.result_directories)
        created = create_kv_cache_plots(points, args.output_dir)
    except PlotKVCacheResultsError as exc:
        parser.error(str(exc))
    for path in created:
        print(path)


if __name__ == "__main__":
    main()
