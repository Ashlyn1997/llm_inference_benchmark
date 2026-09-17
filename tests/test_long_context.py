import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

try:
    import httpx  # noqa: F401
except ImportError:
    sys.modules["httpx"] = types.ModuleType("httpx")


PROJECT_DIR = Path(__file__).resolve().parents[1]


class PromptFileTest(unittest.TestCase):
    def test_prompt_file_overrides_inline_prompt(self):
        from llm_benchmark import cli

        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompt.txt"
            prompt_path.write_text("Prompt content from file.", encoding="utf-8")
            args = cli.build_parser().parse_args(
                [
                    "--model",
                    "test-model",
                    "--prompt",
                    "Inline prompt should be ignored.",
                    "--prompt-file",
                    str(prompt_path),
                ]
            )
            prompt, source = cli._resolve_prompt(args)

        self.assertEqual(prompt, "Prompt content from file.")
        self.assertEqual(source, f"file:{prompt_path.resolve()}")

    def test_prompts_file_loads_jsonl_and_overrides_other_prompt_inputs(self):
        from llm_benchmark import cli

        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            prompt_file = directory_path / "single.txt"
            prompt_file.write_text("Single-file prompt.", encoding="utf-8")
            pool_file = directory_path / "pool.jsonl"
            pool_file.write_text(
                '{"prompt": "Pool prompt zero."}\n'
                '{"prompt": "Pool prompt one."}\n',
                encoding="utf-8",
            )
            args = cli.build_parser().parse_args(
                [
                    "--model",
                    "test-model",
                    "--prompt",
                    "Inline prompt should be ignored.",
                    "--prompt-file",
                    str(prompt_file),
                    "--prompts-file",
                    str(pool_file),
                ]
            )
            selection = cli._resolve_prompts(args)

        self.assertEqual(selection.mode, "pool")
        self.assertEqual(
            selection.prompts,
            ("Pool prompt zero.", "Pool prompt one."),
        )
        self.assertEqual(selection.source, f"file:{pool_file.resolve()}")

    def test_missing_prompt_file_has_clear_error(self):
        from llm_benchmark import cli

        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "does-not-exist.txt"
            args = cli.build_parser().parse_args(
                ["--model", "test-model", "--prompt-file", str(missing_path)]
            )

            with self.assertRaisesRegex(ValueError, "Prompt file does not exist"):
                cli._resolve_prompt(args)

    def test_invalid_jsonl_has_clear_error(self):
        from llm_benchmark import cli

        with tempfile.TemporaryDirectory() as directory:
            pool_file = Path(directory) / "invalid.jsonl"
            pool_file.write_text('{"prompt": "valid"}\nnot-json\n', encoding="utf-8")
            args = cli.build_parser().parse_args(
                ["--model", "test-model", "--prompts-file", str(pool_file)]
            )

            with self.assertRaisesRegex(ValueError, "Invalid JSONL.*line 2"):
                cli._resolve_prompts(args)

    def test_empty_prompt_pool_has_clear_error(self):
        from llm_benchmark import cli

        with tempfile.TemporaryDirectory() as directory:
            pool_file = Path(directory) / "empty.jsonl"
            pool_file.write_text("\n", encoding="utf-8")
            args = cli.build_parser().parse_args(
                ["--model", "test-model", "--prompts-file", str(pool_file)]
            )

            with self.assertRaisesRegex(ValueError, "Prompt pool is empty"):
                cli._resolve_prompts(args)

    def test_request_ids_round_robin_through_prompt_pool(self):
        from llm_benchmark.client import BenchmarkConfig

        config = BenchmarkConfig(
            base_url="http://vllm.example",
            model="test-model",
            prompt="fallback",
            prompts=("zero", "one", "two"),
            max_tokens=8,
            temperature=0,
            requests=3,
            concurrency=1,
            timeout_s=10,
        )

        self.assertEqual(config.prompt_for_request(1), "one")
        self.assertEqual(config.prompt_for_request(2), "two")
        self.assertEqual(config.prompt_for_request(3), "zero")
        self.assertEqual(config.prompt_for_request(4), "one")


class LongContextScriptTest(unittest.TestCase):
    def test_script_builds_expected_default_arguments(self):
        script_path = PROJECT_DIR / "scripts" / "run_long_context_benchmark.sh"
        with tempfile.TemporaryDirectory() as directory:
            temporary_dir = Path(directory)
            captured_args = temporary_dir / "args.txt"
            fake_python = temporary_dir / "fake_python.sh"
            fake_python.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE_FILE\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHON_BIN": str(fake_python),
                    "CAPTURE_FILE": str(captured_args),
                }
            )
            subprocess.run(
                ["bash", str(script_path)],
                cwd=PROJECT_DIR,
                env=environment,
                check=True,
            )
            arguments = captured_args.read_text(encoding="utf-8").splitlines()

        def value_after(flag):
            return arguments[arguments.index(flag) + 1]

        self.assertEqual(value_after("--model"), "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(value_after("--prompts-file"), "prompts/long_2k_pool.jsonl")
        self.assertEqual(value_after("--concurrency"), "8,16,32,64")
        self.assertEqual(value_after("--requests"), "64")
        self.assertEqual(value_after("--max-tokens"), "256")
        self.assertEqual(value_after("--warmup-requests"), "5")
        self.assertEqual(value_after("--gpu-index"), "0")
        self.assertEqual(value_after("--gpu-monitor-interval-ms"), "200")
        self.assertEqual(value_after("--output-dir"), "results/long_context")
