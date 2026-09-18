import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_DIR / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import plot_results


class FakeAxis:
    def plot(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def set_xticks(self, *args, **kwargs):
        return None

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
    def subplots(self, *args, **kwargs):
        return FakeFigure(), FakeAxis()

    def close(self, figure):
        return None


def comparison_entry(concurrency):
    return {
        "concurrency": concurrency,
        "request_throughput_rps": concurrency / 2,
        "output_token_throughput_tps": concurrency * 10,
        "ttft_ms": {"average": concurrency * 100, "p95": concurrency * 120},
        "tpot_ms": {"average": concurrency / 2},
        "total_latency_ms": {"average": concurrency * 500},
        "avg_gpu_utilization_pct": 70 + concurrency / 10,
        "avg_memory_used_mib": 1000 + concurrency,
    }


def run_entry(concurrency):
    entry = comparison_entry(concurrency)
    gpu_utilization = entry.pop("avg_gpu_utilization_pct")
    memory_used = entry.pop("avg_memory_used_mib")
    entry["metadata"] = {
        "gpu": {
            "avg_utilization_pct": gpu_utilization,
            "avg_memory_used_mib": memory_used,
        }
    }
    return entry


class PlotResultsTest(unittest.TestCase):
    def test_multiple_summaries_are_loaded_and_sorted_by_concurrency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_32 = root / "result_32"
            result_8 = root / "result_8"
            result_32.mkdir()
            result_8.mkdir()
            (result_32 / "combined_summary.json").write_text(
                json.dumps({"comparison": [comparison_entry(32)], "runs": []}),
                encoding="utf-8",
            )
            (result_8 / "combined_summary.json").write_text(
                json.dumps({"comparison": [], "runs": [run_entry(8)]}),
                encoding="utf-8",
            )

            points = plot_results.load_results([result_32, result_8])

        self.assertEqual([point.concurrency for point in points], [8, 32])
        self.assertEqual(points[0].output_token_throughput_tps, 80)
        self.assertEqual(points[1].ttft_p95_ms, 3840)
        self.assertEqual(points[0].avg_memory_used_mib, 1008)

    def test_output_directory_is_created_and_pngs_are_written(self):
        points = [
            plot_results._point_from_entry(
                comparison_entry(concurrency),
                Path("test-summary.json"),
                "comparison",
            )
            for concurrency in (8, 16, 32, 64)
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "nested" / "plots"
            with patch.object(
                plot_results,
                "_load_pyplot",
                return_value=FakePyplot(),
            ):
                created = plot_results.create_plots(points, output_directory)

            self.assertTrue(output_directory.is_dir())
            self.assertEqual(
                {path.name for path in created},
                {
                    "concurrency_vs_throughput.png",
                    "concurrency_vs_latency.png",
                    "concurrency_vs_tpot.png",
                    "concurrency_vs_gpu_utilization.png",
                },
            )
            self.assertTrue(all(path.stat().st_size > 0 for path in created))

    def test_missing_summary_has_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_directory = Path(directory) / "missing-result"
            with self.assertRaisesRegex(
                plot_results.PlotResultsError,
                "Missing benchmark summary",
            ):
                plot_results.load_results([missing_directory])
