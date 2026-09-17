from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RequestResult:
    """All measurements captured for one benchmark request."""

    request_id: int
    concurrency: int
    success: bool
    status_code: int | None = None
    ttft_ms: float | None = None
    total_latency_ms: float | None = None
    streamed_chunks: int = 0  # Debug-only SSE frame count; not a token count.
    output_text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    error: str | None = None

    @property
    def tpot_ms(self) -> float | None:
        """Time per output token, excluding the first token covered by TTFT."""
        if (
            self.ttft_ms is None
            or self.total_latency_ms is None
            or self.completion_tokens is None
            or self.completion_tokens <= 1
        ):
            return None
        decode_time_ms = self.total_latency_ms - self.ttft_ms
        if decode_time_ms < 0:
            return None
        return decode_time_ms / (self.completion_tokens - 1)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["tpot_ms"] = self.tpot_ms
        return result


@dataclass
class MetricSummary:
    count: int
    successful_requests: int
    failed_requests: int
    concurrency: int
    wall_time_s: float
    request_throughput_rps: float
    output_token_throughput_tps: float
    ttft_ms: dict[str, float | None]
    tpot_ms: dict[str, float | None]
    total_latency_ms: dict[str, float | None]
    total_prompt_tokens: int
    total_completion_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
