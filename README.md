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
├── concurrency_8/
│   ├── request_results.csv      # 每次请求一行
│   ├── request_results.json
│   ├── summary.json
│   └── gpu_metrics.csv          # nvidia-smi 每 200ms 采样
└── combined_summary.json        # 跨并发度对比
```

TTFT 是从请求开始到第一个非空 generated content 的时间；如果没有收到非空 content，TTFT 为 `null`。TPOT 为 `(total_latency_ms - ttft_ms) / (completion_tokens - 1)`，仅在 vLLM usage 返回 completion token 数且其大于 1 时计算。request throughput 定义为成功请求数除以该并发度 run 的 wall-clock 时间；output token throughput 使用成功请求的 `completion_tokens / wall_time_s`。

请求失败会保留在 request 结果中；如果任一请求失败，CLI 以退出码 `1` 结束，但仍会写出完整结果文件。

## 标准 Benchmark 流程

每个 concurrency 独立按以下顺序执行：

```text
warmup → GPU monitor start → benchmark → GPU monitor stop → summary
```

warmup 默认执行 5 个请求，不写入正式结果、不计入正式 wall time，也不写入 GPU CSV。GPU 监控默认以 200ms 频率调用 `nvidia-smi --loop-ms=200`；可通过 `--gpu-monitor-interval-ms` 调整。没有该命令或监控失败时会给出 warning，但正式 benchmark 继续执行。GPU summary 默认只统计 `--gpu-index 0` 的样本。

```bash
python -m llm_benchmark.cli \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --concurrency 1,4,8,16,32 \
  --requests 50 \
  --max-tokens 128 \
  --warmup-requests 5 \
  --gpu-monitor-interval-ms 200 \
  --gpu-index 0
```

使用 `--warmup-requests 0` 可关闭 warmup。

## Long Context / KV Cache Pressure Benchmark

项目内置 [long_2k_pool.jsonl](prompts/long_2k_pool.jsonl)，包含 64 条约 2k token 的独立英文技术 prompt，主题覆盖 CUDA、分布式推理、数据库、编译器、操作系统、网络、存储、视觉、推荐系统等。各 prompt 从开头就使用不同的场景、术语和段落顺序，前 300–500 tokens 不共享固定介绍模板，从而在不关闭 vLLM Prefix Cache 的前提下减少 prefix-cache reuse。

默认 workload 使用 concurrency `8,16,32,64`、每档 `64` 个请求、`max_tokens=256`，并特意设置 `warmup_requests=0`，避免 warmup 预先把正式 prompt pool 填入 Prefix Cache：

```bash
./scripts/run_long_context_benchmark.sh \
  --base-url http://127.0.0.1:8000
```

也可以直接调用 CLI：

```bash
python3 -m llm_benchmark.cli \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts-file prompts/long_2k_pool.jsonl \
  --concurrency 8,16,32,64 \
  --requests 64 \
  --max-tokens 256 \
  --warmup-requests 0 \
  --gpu-index 0 \
  --gpu-monitor-interval-ms 200 \
  --output-dir results/long_context
```

prompt 输入优先级为 `--prompts-file > --prompt-file > --prompt`。使用 prompt pool 时，第 `request_id` 个请求选择 `prompts[request_id % len(prompts)]`，因此 64 个测量请求会轮询整个 pool。Long-context KV Cache pressure experiment intentionally uses diverse prefixes and `warmup=0` to reduce prefix-cache reuse. 每个 run 的 summary metadata 会保存 `prompt_source`、`prompt_mode`、`prompt_count` 和 prompt 字符数范围；request-level 结果继续保留服务返回的实际 `prompt_tokens`。本项目不估算 prompt token 数，避免引入不可靠或额外的 tokenizer 依赖。

## Kaggle vLLM 实验续跑操作手册

用途：下次重新开启 Kaggle Session 后，继续 `llm_inference_benchmark` 的 KV Cache 压力实验。

当前已完成：`Qwen/Qwen2.5-0.5B-Instruct`，约 1.9k prompt tokens、256 output tokens；clean run 已完成 C8/C16/C32/C64；`gpu-memory-utilization=0.7` 配置下 KV Cache 约为 8.62 GiB。下一阶段从显式限制 KV Cache 为 4 GiB 开始。

### 1. 新 Session：拉取项目并安装

在 Kaggle Notebook 单元格中执行：

```python
%cd /kaggle/working

!rm -rf /kaggle/working/llm_inference_benchmark
!git clone https://github.com/Ashlyn1997/llm_inference_benchmark.git /kaggle/working/llm_inference_benchmark

%cd /kaggle/working/llm_inference_benchmark

!pip install -r requirements.txt
!pip install -e .
!pytest -q
```

如果不想重新 clone，而当前 Session 里仓库还在，可以改用 `git fetch` 和 `git reset`。新的 Kaggle Session 通常直接重新 clone 最省事。

### 2. 确认 GPU 和关键文件

```python
!nvidia-smi

!ls -lh prompts/long_2k_pool.jsonl
!wc -l prompts/long_2k_pool.jsonl
!python -m llm_benchmark.cli --help | grep prompts
```

预期：prompt pool 至少有 64 条，并且 CLI 帮助中能看到 `--prompts-file`。

### 3. 启动 4 GiB KV Cache 的 vLLM

重要：这一轮不要再传 `--gpu-memory-utilization`，只使用 `--kv-cache-memory`，避免两个控制变量混在一起。

```python
import subprocess

log_file = open("/kaggle/working/vllm.log", "w")

proc = subprocess.Popen(
    [
        "vllm",
        "serve",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--dtype", "half",
        "--max-model-len", "4096",
        "--kv-cache-memory", "4294967296",
    ],
    stdout=log_file,
    stderr=subprocess.STDOUT,
)

print("vLLM PID:", proc.pid)
```

### 4. 等待服务启动并确认 4 GiB 生效

```python
!tail -n 80 /kaggle/working/vllm.log

!curl http://127.0.0.1:8000/v1/models

!grep -Ei "KV cache|kv-cache-memory|GPU KV cache size|Maximum concurrency|Available KV cache memory" \
  /kaggle/working/vllm.log
```

只有确认服务成功启动、KV Cache 配置生效后，再开始 benchmark。

### 5. 4 GiB：先跑 clean C32

```python
!rm -rf results/kv4g_c32

!python -m llm_benchmark.cli \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts-file prompts/long_2k_pool.jsonl \
  --concurrency 32 \
  --requests 64 \
  --max-tokens 256 \
  --warmup-requests 0 \
  --gpu-index 0 \
  --gpu-monitor-interval-ms 200 \
  --output-dir results/kv4g_c32
```

### 6. C32 跑完后保存关键日志

```python
!cat results/kv4g_c32/combined_summary.json

!grep -Ei "Prefix cache hit rate|GPU KV cache usage|Running:|Waiting:" \
  /kaggle/working/vllm.log | tail -50

!grep -Ei "preempt|recompute|oom|out of memory|swap" \
  /kaggle/working/vllm.log | tail -30
```

如果最后一条命令没有任何输出，表示暂未看到 preemption、recompute、OOM 或 swap。

### 7. 重启 vLLM，再跑 clean C64

必须重启 vLLM，防止 C32 的 Prefix Cache 污染 C64：

```python
!pkill -f "vllm"
```

然后重新执行第 3 步的 Python 启动代码，仍使用 4 GiB KV Cache。确认 `/v1/models` 可以访问后，再运行：

```python
!rm -rf results/kv4g_c64

!python -m llm_benchmark.cli \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --prompts-file prompts/long_2k_pool.jsonl \
  --concurrency 64 \
  --requests 64 \
  --max-tokens 256 \
  --warmup-requests 0 \
  --gpu-index 0 \
  --gpu-monitor-interval-ms 200 \
  --output-dir results/kv4g_c64
```

### 8. C64 跑完后保存关键日志

```python
!cat results/kv4g_c64/combined_summary.json

!grep -Ei "Prefix cache hit rate|GPU KV cache usage|Running:|Waiting:" \
  /kaggle/working/vllm.log | tail -50

!grep -Ei "preempt|recompute|oom|out of memory|swap" \
  /kaggle/working/vllm.log | tail -30
```

### 9. 后续实验顺序

4 GiB 的 C32/C64 完成后，再按完全相同的方法逐步缩小 KV Cache。每一个 concurrency 都要重启 vLLM。

```text
4 GiB = 4294967296 bytes
2 GiB = 2147483648 bytes
1 GiB = 1073741824 bytes
```

建议顺序：

```text
4 GiB C32
→ 重启
→ 4 GiB C64
→ 重启并改为 2 GiB
→ 2 GiB C32
→ 重启
→ 2 GiB C64
→ 重启并改为 1 GiB
→ 1 GiB C32
→ 重启
→ 1 GiB C64
```

每次只修改 `--kv-cache-memory` 的字节数，其他模型、prompt pool、请求数量、output token 数和采样参数保持不变。

### 10. 每组实验重点观察

- TTFT：首 token 延迟，重点看 average、P95 和 P99。
- TPOT：生成阶段每个 output token 的平均延迟。
- Output token throughput：吞吐是否因 KV Cache 压力下降。
- GPU KV cache usage：是否逐渐接近容量上限。
- Waiting：是否开始出现等待请求。
- Preemption/recompute：KV Cache 不足时是否触发抢占或重计算。
- 成功请求数：是否出现失败或 OOM。

### 11. 当前基线

以下数据用于下一轮实验对照：

| Concurrency | Output tok/s | TTFT avg | TPOT avg | E2E avg |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 365 | 1.20 s | 17.3 ms | 5.60 s |
| 16 | 415 | 1.51 s | 32.7 ms | 9.86 s |
| 32 | 449 | 3.55 s | 57.2 ms | 18.14 s |
| 64 | 474 | 11.12 s | 89.1 ms | 33.85 s |

基线结论：C32 到 C64 的吞吐只继续小幅增加，但 TTFT 和 E2E 显著恶化。当前 `gpu-memory-utilization=0.7` 配置的 KV Cache 容量仍较充足，因此下一阶段通过显式缩小 KV Cache 制造容量压力。

### 12. 结束 Kaggle Session 前

```python
!pkill -f "vllm"
!nvidia-smi
```

确认 vLLM 进程已经结束后即可停止 Kaggle Session。实验结果如果需要长期保留，记得在关闭 Session 前下载，或提交到 Git/持久化位置；`/kaggle/working` 在新 Session 中通常不会保留。
