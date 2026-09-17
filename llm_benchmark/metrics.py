from __future__ import annotations

import math
from collections.abc import Iterable

from .models import MetricSummary, RequestResult


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    """Return a linearly interpolated percentile, or None for no values."""
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile_value / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def summarize(
    results: list[RequestResult],
    *,
    concurrency: int,
    wall_time_s: float,
    metadata: dict | None = None,
) -> MetricSummary:
    successful = [result for result in results if result.success]
    ttft = [result.ttft_ms for result in successful if result.ttft_ms is not None]
    tpot = [result.tpot_ms for result in successful if result.tpot_ms is not None]
    latency = [
        result.total_latency_ms
        for result in successful
        if result.total_latency_ms is not None
    ]
    total_prompt_tokens = sum(result.prompt_tokens or 0 for result in successful)
    total_completion_tokens = sum(
        result.completion_tokens or 0 for result in successful
    )

    def stats(values: list[float]) -> dict[str, float | None]:
        return {
            "average": sum(values) / len(values) if values else None,
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
        }

    return MetricSummary(
        count=len(results),
        successful_requests=len(successful),
        failed_requests=len(results) - len(successful),
        concurrency=concurrency,
        wall_time_s=wall_time_s,
        request_throughput_rps=(len(successful) / wall_time_s if wall_time_s > 0 else 0),
        output_token_throughput_tps=(
            total_completion_tokens / wall_time_s if wall_time_s > 0 else 0
        ),
        ttft_ms=stats(ttft),
        tpot_ms=stats(tpot),
        total_latency_ms=stats(latency),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        metadata=metadata or {},
    )
