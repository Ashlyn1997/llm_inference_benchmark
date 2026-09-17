from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from .client import BenchmarkConfig, run_benchmark
from .metrics import summarize
from .output import write_combined_summary, write_request_results, write_summary


DEFAULT_CONCURRENCIES = [1, 4, 8, 16, 32]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _concurrencies(value: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("use comma-separated positive integers") from exc
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("concurrency values must be greater than zero")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible streaming vLLM service")
    parser.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--model", default=os.getenv("VLLM_MODEL", "") , required=not bool(os.getenv("VLLM_MODEL")))
    parser.add_argument("--api-key", default=os.getenv("VLLM_API_KEY"))
    parser.add_argument("--prompt", default="Explain what vLLM is in one short paragraph.")
    parser.add_argument("--system-prompt")
    parser.add_argument("--max-tokens", type=_positive_int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--requests", type=_positive_int, default=32)
    parser.add_argument(
        "--concurrency",
        type=_concurrencies,
        default=DEFAULT_CONCURRENCIES,
        help="comma-separated values, e.g. 1,4,8,16,32",
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="per-request timeout in seconds")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser


async def _run(args: argparse.Namespace) -> int:
    summaries = []
    for concurrency in args.concurrency:
        run_dir = args.output_dir / f"concurrency_{concurrency}"
        config = BenchmarkConfig(
            base_url=args.base_url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            requests=args.requests,
            concurrency=concurrency,
            timeout_s=args.timeout,
            api_key=args.api_key,
            system_prompt=args.system_prompt,
        )
        print(f"Running concurrency={concurrency}, requests={args.requests} ...", flush=True)
        results, wall_time_s = await run_benchmark(config)
        summary = summarize(
            results,
            concurrency=concurrency,
            wall_time_s=wall_time_s,
            metadata={
                "base_url": args.base_url,
                "model": args.model,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
        )
        write_request_results(run_dir, results)
        write_summary(run_dir, summary)
        summaries.append(summary)
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    write_combined_summary(args.output_dir, summaries)
    return 0 if all(summary.failed_requests == 0 for summary in summaries) else 1


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
