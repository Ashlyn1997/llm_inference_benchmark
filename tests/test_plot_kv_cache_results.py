import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import plot_kv_cache_results


def summary_entry(concurrency):
    return {
        "concurrency": concurrency,
        "output_token_throughput_tps": concurrency * 10,
        "ttft_ms": {"average": concurrency * 100},
        "total_latency_ms": {"average": concurrency * 500},
        "avg_memory_used_mib": 1000 + concurrency,
    }


def write_summary(result_directory, concurrency):
    result_directory.mkdir(parents=True)
    (result_directory / "combined_summary.json").write_text(
        json.dumps({"comparison": [summary_entry(concurrency)], "runs": []}),
        encoding="utf-8",
    )


class FakeAxis:
    def __init__(self):
        self.plot_calls = []
        self.xticks = None

    def plot(self, *args, **kwargs):
        self.plot_calls.append((args, kwargs))

    def set_title(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def set_xticks(self, ticks):
        self.xticks = list(ticks)

    def grid(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None


class FakeFigure:
    def tight_layout(self):
        return None

    def savefig(self, path, **kwargs):
        Path(path).write_bytes(b"fake-png")


class FakePyplot:
    def __init__(self):
        self.axes = []

    def subplots(self, *args, **kwargs):
        axis = FakeAxis()
        self.axes.append(axis)
        return FakeFigure(), axis

    def close(self, figure):
        return None


class PlotKVCacheResultsTest(unittest.TestCase):
    def test_directory_name_parsing(self):
        self.assertEqual(
            plot_kv_cache_results.parse_result_directory_name(Path("kv4g_c32")),
            (4.0, 32),
        )
        self.assertEqual(
            plot_kv_cache_results.parse_result_directory_name(Path("kv2g_c64")),
            (2.0, 64),
        )
        with self.assertRaisesRegex(
            plot_kv_cache_results.PlotKVCacheResultsError,
            "Cannot parse KV cache size",
        ):
            plot_kv_cache_results.parse_result_directory_name(Path("result_c32"))

    def test_duplicate_concurrency_with_different_cache_size_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_1g = root / "kv1g_c32"
            result_2g = root / "kv2g_c32"
            write_summary(result_1g, 32)
            write_summary(result_2g, 32)

            points = plot_kv_cache_results.load_kv_cache_results(
                [result_2g, result_1g]
            )

        self.assertEqual(
            [(point.cache_size_gib, point.concurrency) for point in points],
            [(1.0, 32), (2.0, 32)],
        )

    def test_duplicate_cache_size_and_concurrency_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first" / "kv1g_c32"
            second = root / "second" / "kv1g_c32"
            write_summary(first, 32)
            write_summary(second, 32)

            with self.assertRaisesRegex(
                plot_kv_cache_results.PlotKVCacheResultsError,
                "Duplicate KV cache/concurrency result",
            ):
                plot_kv_cache_results.load_kv_cache_results([first, second])

    def test_results_are_sorted_by_cache_size_then_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [
                (root / "kv4g_c64", 64),
                (root / "kv1g_c64", 64),
                (root / "kv2g_c32", 32),
                (root / "kv1g_c32", 32),
            ]
            for path, concurrency in inputs:
                write_summary(path, concurrency)

            points = plot_kv_cache_results.load_kv_cache_results(
                [path for path, _ in inputs]
            )

        self.assertEqual(
            [(point.cache_size_gib, point.concurrency) for point in points],
            [(1.0, 32), (1.0, 64), (2.0, 32), (4.0, 64)],
        )

    def test_output_directory_is_created(self):
        points = [
            plot_kv_cache_results.KVCachePoint(
                cache_size_gib=cache_size,
                concurrency=concurrency,
                output_token_throughput_tps=400 + cache_size,
                ttft_average_ms=1000 + concurrency,
                total_latency_average_ms=5000 + concurrency,
                avg_memory_used_mib=12000 + cache_size,
            )
            for cache_size in (1.0, 2.0, 4.0)
            for concurrency in (32, 64)
        ]
        fake_pyplot = FakePyplot()
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "nested" / "kv_plots"
            with patch.object(
                plot_kv_cache_results,
                "_load_pyplot",
                return_value=fake_pyplot,
            ):
                created = plot_kv_cache_results.create_kv_cache_plots(
                    points, output_directory
                )

            self.assertTrue(output_directory.is_dir())
            self.assertEqual(
                {path.name for path in created},
                {
                    "kv_cache_vs_throughput.png",
                    "kv_cache_vs_ttft.png",
                    "kv_cache_vs_e2e_latency.png",
                    "kv_cache_vs_gpu_memory.png",
                },
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in created))

        self.assertTrue(
            all(axis.xticks == [1.0, 2.0, 4.0] for axis in fake_pyplot.axes)
        )
        self.assertTrue(
            all(
                call[1].get("marker") == "o"
                for axis in fake_pyplot.axes
                for call in axis.plot_calls
            )
        )
