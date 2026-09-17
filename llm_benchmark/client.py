from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .models import RequestResult


@dataclass(frozen=True)
class BenchmarkConfig:
    base_url: str
    model: str
    prompt: str
    max_tokens: int
    temperature: float
    requests: int
    concurrency: int
    timeout_s: float
    api_key: str | None = None
    system_prompt: str | None = None

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/v1/chat/completions"


def _extract_usage(payload: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = payload.get("usage") or {}
    return (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


def _content_from_chunk(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    content = delta.get("content", "")
    return content if isinstance(content, str) else ""


async def run_request(
    client: httpx.AsyncClient,
    config: BenchmarkConfig,
    request_id: int,
) -> RequestResult:
    started = time.perf_counter_ns()
    first_token_ns: int | None = None
    chunks = 0
    output_parts: list[str] = []
    prompt_tokens = completion_tokens = total_tokens = None
    headers = {"content-type": "application/json"}
    if config.api_key:
        headers["authorization"] = f"Bearer {config.api_key}"
    messages: list[dict[str, str]] = []
    if config.system_prompt:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": config.prompt})
    body = {
        "model": config.model,
        "messages": messages,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    try:
        async with client.stream(
            "POST", config.endpoint, headers=headers, json=body
        ) as response:
            status_code = response.status_code
            if status_code >= 400:
                error_body = (await response.aread()).decode("utf-8", errors="replace")
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
                return RequestResult(
                    request_id=request_id,
                    concurrency=config.concurrency,
                    success=False,
                    status_code=status_code,
                    total_latency_ms=elapsed_ms,
                    error=f"HTTP {status_code}: {error_body[:1000]}",
                )

            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    break
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    # Ignore provider-specific keepalives or malformed SSE frames.
                    continue
                chunks += 1
                content = _content_from_chunk(payload)
                if content:
                    output_parts.append(content)
                    if first_token_ns is None:
                        first_token_ns = time.perf_counter_ns()
                usage = payload.get("usage")
                if usage:
                    prompt_tokens, completion_tokens, total_tokens = _extract_usage(payload)

        finished_ns = time.perf_counter_ns()
        return RequestResult(
            request_id=request_id,
            concurrency=config.concurrency,
            success=True,
            status_code=status_code,
            ttft_ms=(
                (first_token_ns - started) / 1_000_000
                if first_token_ns is not None
                else None
            ),
            total_latency_ms=(finished_ns - started) / 1_000_000,
            streamed_chunks=chunks,
            output_text="".join(output_parts),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    except Exception as exc:  # capture per-request failures so a run can finish
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        return RequestResult(
            request_id=request_id,
            concurrency=config.concurrency,
            success=False,
            total_latency_ms=elapsed_ms,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_benchmark(config: BenchmarkConfig) -> tuple[list[RequestResult], float]:
    timeout = httpx.Timeout(config.timeout_s, connect=min(config.timeout_s, 10.0))
    limits = httpx.Limits(
        max_connections=max(config.concurrency, 1),
        max_keepalive_connections=max(config.concurrency, 1),
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        semaphore = asyncio.Semaphore(config.concurrency)

        async def bounded_request(request_id: int) -> RequestResult:
            async with semaphore:
                return await run_request(client, config, request_id)

        started = time.perf_counter()
        results = await asyncio.gather(
            *(bounded_request(request_id) for request_id in range(1, config.requests + 1))
        )
        return list(results), time.perf_counter() - started
