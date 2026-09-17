from __future__ import annotations

import csv
import subprocess
import warnings
from pathlib import Path
from typing import TextIO


GPU_METRIC_FIELDS = (
    "timestamp",
    "index",
    "utilization.gpu",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
    "power.limit",
)


class GPUMonitor:
    """Continuously record nvidia-smi query output without affecting a benchmark."""

    def __init__(self, output_path: Path, interval_ms: int = 200):
        if interval_ms <= 0:
            raise ValueError("interval_ms must be greater than zero")
        self.output_path = Path(output_path)
        self.interval_ms = interval_ms
        self.warning: str | None = None
        self._process: subprocess.Popen | None = None
        self._output_file: TextIO | None = None

    def start(self) -> None:
        """Start nvidia-smi sampling. Failures are warnings, never benchmark failures."""
        if self._process is not None:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_file = self.output_path.open("w", newline="", encoding="utf-8")
        csv.writer(self._output_file).writerow(GPU_METRIC_FIELDS)
        self._output_file.flush()
        command = [
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_METRIC_FIELDS)}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdout=self._output_file,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError as exc:
            self.warning = (
                "GPU monitoring is unavailable; continuing benchmark without GPU metrics: "
                f"{exc}"
            )
            warnings.warn(self.warning, RuntimeWarning, stacklevel=2)
            self._output_file.close()
            self._output_file = None

    def stop(self) -> None:
        """Terminate the monitor process and close its output file."""
        process = self._process
        self._process = None
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        except OSError as exc:
            warnings.warn(
                f"Failed to stop GPU monitor cleanly: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
        finally:
            if self._output_file is not None:
                self._output_file.close()
                self._output_file = None

    def __enter__(self) -> "GPUMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.stop()
        return False


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized or normalized.upper() in {"N/A", "[NOT SUPPORTED]"}:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def _stats(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return sum(values) / len(values), max(values)


def summarize_gpu_metrics(output_path: Path, gpu_index: int = 0) -> dict[str, int | float | None]:
    """Summarize samples for exactly one GPU index from a monitor CSV file."""
    values = {
        "utilization": [],
        "memory_used": [],
        "temperature": [],
        "power": [],
    }
    path = Path(output_path)
    if path.exists():
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("index", "").strip() != str(gpu_index):
                    continue
                for field, column in (
                    ("utilization", "utilization.gpu"),
                    ("memory_used", "memory.used"),
                    ("temperature", "temperature.gpu"),
                    ("power", "power.draw"),
                ):
                    value = _as_float(row.get(column))
                    if value is not None:
                        values[field].append(value)

    avg_utilization, max_utilization = _stats(values["utilization"])
    avg_memory_used, max_memory_used = _stats(values["memory_used"])
    avg_temperature, max_temperature = _stats(values["temperature"])
    avg_power, max_power = _stats(values["power"])
    return {
        "index": gpu_index,
        "avg_utilization_pct": avg_utilization,
        "max_utilization_pct": max_utilization,
        "avg_memory_used_mib": avg_memory_used,
        "max_memory_used_mib": max_memory_used,
        "avg_temperature_c": avg_temperature,
        "max_temperature_c": max_temperature,
        "avg_power_w": avg_power,
        "max_power_w": max_power,
    }
