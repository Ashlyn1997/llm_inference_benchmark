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

    def test_missing_prompt_file_has_clear_error(self):
        from llm_benchmark import cli

        with tempfile.TemporaryDirectory() as directory:
            missing_path = Path(directory) / "does-not-exist.txt"
            args = cli.build_parser().parse_args(
                ["--model", "test-model", "--prompt-file", str(missing_path)]
            )

            with self.assertRaisesRegex(ValueError, "Prompt file does not exist"):
                cli._resolve_prompt(args)


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
        self.assertEqual(value_after("--prompt-file"), "prompts/long_2k.txt")
        self.assertEqual(value_after("--concurrency"), "8,16,32,64")
        self.assertEqual(value_after("--requests"), "64")
        self.assertEqual(value_after("--max-tokens"), "256")
        self.assertEqual(value_after("--warmup-requests"), "5")
        self.assertEqual(value_after("--gpu-index"), "0")
        self.assertEqual(value_after("--gpu-monitor-interval-ms"), "200")
        self.assertEqual(value_after("--output-dir"), "results/long_context")
