from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from .client import BenchmarkConfig, run_benchmark
from .gpu_monitor import GPUMonitor, summarize_gpu_metrics
from .metrics import summarize
from .output import write_combined_summary, write_request_results, write_summary


DEFAULT_CONCURRENCIES = [1, 4, 8, 16, 32]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class PromptSelection:
    prompts: tuple[str, ...]
    mode: str
    source: str


def _read_prompt_file(prompt_path: Path) -> str:
    try:
        return prompt_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ValueError(f"Prompt file does not exist: {prompt_path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read prompt file {prompt_path}: {exc}") from exc


def _read_prompt_pool(prompt_path: Path) -> tuple[str, ...]:
    prompts: list[str] = []
    try:
        with prompt_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL in prompts file {prompt_path} at line "
                        f"{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(item, dict) or not isinstance(item.get("prompt"), str):
                    raise ValueError(
                        f"Invalid JSONL in prompts file {prompt_path} at line "
                        f"{line_number}: expected an object with a string 'prompt' field"
                    )
                if not item["prompt"].strip():
                    raise ValueError(
                        f"Invalid JSONL in prompts file {prompt_path} at line "
                        f"{line_number}: prompt must not be empty"
                    )
                prompts.append(item["prompt"])
    except FileNotFoundError as exc:
        raise ValueError(f"Prompts file does not exist: {prompt_path}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read prompts file {prompt_path}: {exc}") from exc

    if not prompts:
        raise ValueError(f"Prompt pool is empty: {prompt_path}")
    return tuple(prompts)


def _resolve_prompts(args: argparse.Namespace) -> PromptSelection:
    """Load a prompt pool, one prompt file, or the inline prompt by precedence."""
    prompts_file = getattr(args, "prompts_file", None)
    if prompts_file is not None:
        prompt_path = prompts_file.expanduser()
        return PromptSelection(
            prompts=_read_prompt_pool(prompt_path),
            mode="pool",
            source=f"file:{prompt_path.resolve()}",
        )

    if args.prompt_file is None:
        return PromptSelection(
            prompts=(args.prompt,),
            mode="single",
            source="inline:--prompt",
        )

    prompt_path = args.prompt_file.expanduser()
    return PromptSelection(
        prompts=(_read_prompt_file(prompt_path),),
        mode="single",
        source=f"file:{prompt_path.resolve()}",
    )


def _resolve_prompt(args: argparse.Namespace) -> tuple[str, str]:
    """Backward-compatible helper for callers that require one resolved prompt."""
    selection = _resolve_prompts(args)
    return selection.prompts[0], selection.source


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
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="UTF-8 text file used as the prompt; overrides --prompt",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        help="JSONL prompt pool; overrides --prompt-file and --prompt",
    )
    parser.add_argument("--system-prompt")
    parser.add_argument("--max-tokens", type=_positive_int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--requests", type=_positive_int, default=32)
    parser.add_argument(
        "--warmup-requests",
        type=_nonnegative_int,
        default=5,
        help="requests per concurrency before measurement; use 0 to disable",
    )
    parser.add_argument("--gpu-index", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--gpu-monitor-interval-ms",
        type=_positive_int,
        default=200,
        help="nvidia-smi sampling interval in milliseconds",
    )
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
    prompt_selection = _resolve_prompts(args)
    prompt = prompt_selection.prompts[0]
    prompt_lengths = [len(item) for item in prompt_selection.prompts]
    summaries = []
    for concurrency in args.concurrency:
        run_dir = args.output_dir / f"concurrency_{concurrency}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = BenchmarkConfig(
            base_url=args.base_url,
            model=args.model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            requests=args.requests,
            concurrency=concurrency,
            timeout_s=args.timeout,
            api_key=args.api_key,
            system_prompt=args.system_prompt,
            prompts=prompt_selection.prompts,
        )
        if args.warmup_requests:
            print(
                f"Warming up concurrency={concurrency}, requests={args.warmup_requests} ...",
                flush=True,
            )
            warmup_results, _ = await run_benchmark(
                replace(config, requests=args.warmup_requests)
            )
            warmup_successes = sum(result.success for result in warmup_results)
            if warmup_successes != len(warmup_results):
                print(
                    f"Warning: warmup concurrency={concurrency} had "
                    f"{len(warmup_results) - warmup_successes}/{len(warmup_results)} "
                    "failed requests; continuing with measured benchmark.",
                    flush=True,
                )

        print(f"Running concurrency={concurrency}, requests={args.requests} ...", flush=True)
        gpu_metrics_path = run_dir / "gpu_metrics.csv"
        monitor = GPUMonitor(
            gpu_metrics_path,
            interval_ms=args.gpu_monitor_interval_ms,
        )
        benchmark_started_at = _utc_now()
        monitor.start()
        try:
            results, wall_time_s = await run_benchmark(config)
        finally:
            benchmark_finished_at = _utc_now()
            monitor.stop()

        summary = summarize(
            results,
            concurrency=concurrency,
            wall_time_s=wall_time_s,
            metadata={
                "base_url": args.base_url,
                "model": args.model,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "prompt_source": prompt_selection.source,
                "prompt_char_count": (
                    len(prompt) if prompt_selection.mode == "single" else None
                ),
                "prompt_mode": prompt_selection.mode,
                "prompt_count": len(prompt_selection.prompts),
                "min_prompt_char_count": min(prompt_lengths),
                "max_prompt_char_count": max(prompt_lengths),
                "avg_prompt_char_count": sum(prompt_lengths) / len(prompt_lengths),
                "warmup_requests": args.warmup_requests,
                "gpu_monitor_interval_ms": args.gpu_monitor_interval_ms,
                "benchmark_started_at": benchmark_started_at,
                "benchmark_finished_at": benchmark_finished_at,
                "gpu": summarize_gpu_metrics(gpu_metrics_path, args.gpu_index),
            },
        )
        write_request_results(run_dir, results)
        write_summary(run_dir, summary)
        summaries.append(summary)
        print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    write_combined_summary(args.output_dir, summaries)
    return 0 if all(summary.failed_requests == 0 for summary in summaries) else 1


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except ValueError as exc:
        parser.error(str(exc))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
