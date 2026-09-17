import asyncio
import sys
import tempfile
import types
import unittest
import warnings
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

try:
    import httpx  # noqa: F401
except ImportError:
    sys.modules["httpx"] = types.ModuleType("httpx")

from llm_benchmark.gpu_monitor import GPUMonitor, summarize_gpu_metrics
from llm_benchmark.models import RequestResult


class FakeProcess:
    def __init__(self):
        self.terminated = False
        self.killed = False
        self.wait_calls = []

    def poll(self):
        return None if not self.terminated and not self.killed else 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout):
        self.wait_calls.append(timeout)
        return 0


class GPUMonitorTest(unittest.TestCase):
    def test_start_stop_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "gpu_metrics.csv"
            process = FakeProcess()
            with patch(
                "llm_benchmark.gpu_monitor.subprocess.Popen", return_value=process
            ) as popen:
                monitor = GPUMonitor(output_path)
                monitor.start()
                monitor.stop()

            command = popen.call_args.args[0]
            self.assertIn("nvidia-smi", command)
            self.assertIn("--loop-ms=200", command)
            self.assertTrue(process.terminated)
            self.assertEqual(process.wait_calls, [5])
            self.assertTrue(output_path.exists())
            self.assertTrue(output_path.read_text(encoding="utf-8").startswith("timestamp,"))

    def test_custom_interval_is_passed_to_nvidia_smi(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "gpu_metrics.csv"
            process = FakeProcess()
            with patch(
                "llm_benchmark.gpu_monitor.subprocess.Popen", return_value=process
            ) as popen:
                monitor = GPUMonitor(output_path, interval_ms=500)
                monitor.start()
                monitor.stop()

            self.assertIn("--loop-ms=500", popen.call_args.args[0])

    def test_missing_nvidia_smi_warns_without_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "gpu_metrics.csv"
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with patch(
                    "llm_benchmark.gpu_monitor.subprocess.Popen",
                    side_effect=FileNotFoundError(2, "No such file or directory", "nvidia-smi"),
                ):
                    monitor = GPUMonitor(output_path)
                    monitor.start()
                    monitor.stop()

            self.assertIsNone(monitor._process)
            self.assertIsNotNone(monitor.warning)
            self.assertIn("GPU monitoring is unavailable", monitor.warning)
            self.assertTrue(caught)
            self.assertTrue(output_path.exists())

    def test_gpu_summary_filters_index_and_calculates_stats(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "gpu_metrics.csv"
            output_path.write_text(
                "timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit\n"
                "2026-01-01T00:00:00Z,0,10,100,1000,50,120,300\n"
                "2026-01-01T00:00:00Z,1,99,999,1000,99,299,300\n"
                "2026-01-01T00:00:01Z,0,30,300,1000,70,180,300\n",
                encoding="utf-8",
            )
            summary = summarize_gpu_metrics(output_path, gpu_index=0)

        self.assertEqual(summary["index"], 0)
        self.assertEqual(summary["avg_utilization_pct"], 20)
        self.assertEqual(summary["max_utilization_pct"], 30)
        self.assertEqual(summary["avg_memory_used_mib"], 200)
        self.assertEqual(summary["max_memory_used_mib"], 300)
        self.assertEqual(summary["avg_temperature_c"], 60)
        self.assertEqual(summary["max_temperature_c"], 70)
        self.assertEqual(summary["avg_power_w"], 150)
        self.assertEqual(summary["max_power_w"], 180)


class CLIGPUMonitorTest(unittest.TestCase):
    def test_gpu_monitor_interval_cli_argument(self):
        from llm_benchmark import cli

        default_args = cli.build_parser().parse_args(["--model", "test-model"])
        custom_args = cli.build_parser().parse_args(
            ["--model", "test-model", "--gpu-monitor-interval-ms", "500"]
        )

        self.assertEqual(default_args.gpu_monitor_interval_ms, 200)
        self.assertEqual(custom_args.gpu_monitor_interval_ms, 500)

    @staticmethod
    def _args(output_dir: Path) -> Namespace:
        return Namespace(
            base_url="http://vllm.example",
            model="test-model",
            api_key=None,
            prompt="hello",
            system_prompt=None,
            max_tokens=8,
            temperature=0,
            requests=1,
            concurrency=[1],
            timeout=10,
            output_dir=output_dir,
            warmup_requests=0,
            gpu_index=0,
            gpu_monitor_interval_ms=200,
        )

    def test_monitor_stops_when_benchmark_raises(self):
        from llm_benchmark import cli

        created_monitors = []

        class RecordingMonitor:
            def __init__(self, output_path, interval_ms=200):
                self.output_path = output_path
                self.interval_ms = interval_ms
                self.started = False
                self.stopped = False
                created_monitors.append(self)

            def start(self):
                self.started = True

            def stop(self):
                self.stopped = True

        async def failing_benchmark(config):
            raise RuntimeError("benchmark failed")

        with tempfile.TemporaryDirectory() as directory:
            with patch("llm_benchmark.cli.GPUMonitor", RecordingMonitor), patch(
                "llm_benchmark.cli.run_benchmark", new=failing_benchmark
            ):
                with self.assertRaisesRegex(RuntimeError, "benchmark failed"):
                    asyncio.run(cli._run(self._args(Path(directory))))

        self.assertEqual(len(created_monitors), 1)
        self.assertTrue(created_monitors[0].started)
        self.assertTrue(created_monitors[0].stopped)

    def test_missing_nvidia_smi_does_not_stop_benchmark(self):
        from llm_benchmark import cli

        async def successful_benchmark(config):
            return [RequestResult(1, config.concurrency, True)], 1.0

        with tempfile.TemporaryDirectory() as directory:
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with patch(
                    "llm_benchmark.gpu_monitor.subprocess.Popen",
                    side_effect=FileNotFoundError(2, "No such file or directory", "nvidia-smi"),
                ), patch("llm_benchmark.cli.run_benchmark", new=successful_benchmark):
                    exit_code = asyncio.run(cli._run(self._args(Path(directory))))

            self.assertEqual(exit_code, 0)
            self.assertTrue((Path(directory) / "concurrency_1" / "summary.json").exists())
            self.assertTrue((Path(directory) / "combined_summary.json").exists())
