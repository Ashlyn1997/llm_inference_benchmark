# llm_inference_benchmark

用于远程 OpenAI-compatible vLLM 服务的异步并发性能 benchmark client。本项目只在本地发 HTTP 请求，不需要本地 GPU。

## 特性

- 使用 `httpx.AsyncClient` 并发调用 `/v1/chat/completions`
- 固定使用 `stream=true`，记录 TTFT、TPOT 和 E2E total latency
- 支持并发度 `1,4,8,16,32`，也可自定义
- 输出每次请求的 CSV、JSON，以及每个并发度和全部运行的汇总 JSON
- 使用 vLLM 返回的 usage 统计 output token throughput；`streamed_chunks` 仅用于调试，不代表 token 数

## 安装

```bash
cd llm_inference_benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 运行

```bash
python3 -m llm_benchmark.cli \
  --base-url http://your-vllm-host:8000 \
  --model your-model-name \
  --requests 32 \
  --concurrency 1,4,8,16,32 \
  --output-dir results/run-001
```

也可以使用批量脚本：

```bash
export VLLM_BASE_URL=http://your-vllm-host:8000
export VLLM_MODEL=your-model-name
export VLLM_API_KEY=optional-token
./scripts/run_benchmark.sh --requests 32
```

通过 `CONCURRENCIES=1,4,8` 和 `OUTPUT_DIR=results/experiment-a` 可覆盖脚本默认值。CLI 的 `--concurrency` 支持逗号分隔值，因此一次命令会按顺序执行多个并发度，每个并发度使用独立目录。

## 输出

```text
results/
├── summary.json                 # 所有并发度的汇总
└── concurrency_8/
    ├── request_results.csv      # 每次请求一行
    ├── request_results.json
    └── summary.json
```

TTFT 是从请求开始到第一个非空 generated content 的时间；如果没有收到非空 content，TTFT 为 `null`。TPOT 为 `(total_latency_ms - ttft_ms) / (completion_tokens - 1)`，仅在 vLLM usage 返回 completion token 数且其大于 1 时计算。request throughput 定义为成功请求数除以该并发度 run 的 wall-clock 时间；output token throughput 使用成功请求的 `completion_tokens / wall_time_s`。

请求失败会保留在 request 结果中；如果任一请求失败，CLI 以退出码 `1` 结束，但仍会写出完整结果文件。
