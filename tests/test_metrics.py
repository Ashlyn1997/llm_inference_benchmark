import unittest

from llm_benchmark.metrics import percentile, summarize
from llm_benchmark.models import RequestResult


class MetricsTest(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(percentile([1, 2, 3, 4], 50), 2.5)
        self.assertIsNone(percentile([], 50))

    def test_tpot_is_none_for_one_or_fewer_completion_tokens(self):
        for completion_tokens in (None, 0, 1):
            result = RequestResult(
                1,
                1,
                True,
                ttft_ms=10,
                total_latency_ms=100,
                completion_tokens=completion_tokens,
            )
            self.assertIsNone(result.tpot_ms)
        self.assertIsNone(
            RequestResult(
                1,
                1,
                True,
                ttft_ms=100,
                total_latency_ms=50,
                completion_tokens=5,
            ).tpot_ms
        )

    def test_tpot_calculation_and_request_serialization(self):
        result = RequestResult(
            1,
            1,
            True,
            ttft_ms=20,
            total_latency_ms=100,
            completion_tokens=5,
        )
        self.assertEqual(result.tpot_ms, 20)
        self.assertEqual(result.to_dict()["tpot_ms"], 20)

    def test_summary_uses_only_successful_requests_for_metrics(self):
        results = [
            RequestResult(
                1,
                2,
                True,
                ttft_ms=10,
                total_latency_ms=100,
                prompt_tokens=5,
                completion_tokens=10,
            ),
            RequestResult(
                2,
                2,
                True,
                ttft_ms=20,
                total_latency_ms=220,
                prompt_tokens=7,
                completion_tokens=11,
            ),
            RequestResult(
                3,
                2,
                False,
                ttft_ms=1,
                total_latency_ms=2,
                prompt_tokens=999,
                completion_tokens=999,
                error="failed",
            ),
        ]

        summary = summarize(results, concurrency=2, wall_time_s=2)

        self.assertEqual(summary.successful_requests, 2)
        self.assertEqual(summary.failed_requests, 1)
        self.assertEqual(summary.request_throughput_rps, 1)
        self.assertEqual(summary.total_prompt_tokens, 12)
        self.assertEqual(summary.total_completion_tokens, 21)
        self.assertEqual(summary.output_token_throughput_tps, 10.5)
        self.assertEqual(summary.ttft_ms["average"], 15)
        self.assertEqual(summary.tpot_ms["average"], 15)
