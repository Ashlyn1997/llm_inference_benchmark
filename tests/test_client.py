import asyncio
import sys
import types
import unittest
from unittest.mock import patch

try:
    import httpx  # noqa: F401
except ImportError:
    # run_request only needs a duck-typed async client in these unit tests.
    sys.modules["httpx"] = types.ModuleType("httpx")


class FakeResponse:
    status_code = 200

    def __init__(self, lines):
        self.lines = lines

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class FakeStream:
    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.request_body = None

    def stream(self, method, url, headers, json):
        self.request_body = json
        return FakeStream(self.response)


class ClientTTFTTest(unittest.TestCase):
    @staticmethod
    def _config():
        from llm_benchmark.client import BenchmarkConfig

        return BenchmarkConfig(
            base_url="http://vllm.example",
            model="test-model",
            prompt="hello",
            max_tokens=8,
            temperature=0,
            requests=1,
            concurrency=1,
            timeout_s=10,
        )

    def test_ttft_is_measured_from_first_nonempty_content(self):
        from llm_benchmark.client import run_request

        client = FakeClient(
            FakeResponse(
                [
                    'data: {"choices": [{"delta": {"role": "assistant"}}]}',
                    'data: {"choices": [{"delta": {"content": "hello"}}]}',
                    "data: [DONE]",
                ]
            )
        )
        with patch(
            "llm_benchmark.client.time.perf_counter_ns",
            side_effect=[0, 40_000_000, 70_000_000],
        ):
            result = asyncio.run(run_request(client, self._config(), request_id=1))

        self.assertEqual(result.ttft_ms, 40)
        self.assertEqual(result.total_latency_ms, 70)
        self.assertEqual(result.streamed_chunks, 2)
        self.assertEqual(client.request_body["stream_options"], {"include_usage": True})

    def test_ttft_is_none_when_stream_has_no_nonempty_content(self):
        from llm_benchmark.client import run_request

        client = FakeClient(
            FakeResponse(
                [
                    'data: {"choices": [{"delta": {"role": "assistant"}}]}',
                    "data: [DONE]",
                ]
            )
        )
        with patch(
            "llm_benchmark.client.time.perf_counter_ns",
            side_effect=[0, 20_000_000],
        ):
            result = asyncio.run(run_request(client, self._config(), request_id=1))

        self.assertIsNone(result.ttft_ms)
        self.assertEqual(result.total_latency_ms, 20)
