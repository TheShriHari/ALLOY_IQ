from __future__ import annotations

import asyncio
import time
import tracemalloc
from typing import Any, Callable, Dict


class PerformanceProfiler:
    """Small callable profiler for API, websocket, checkpoint, and training paths."""

    def __init__(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    def profile_operation(self, operation_name: str, func: Callable[..., Any], *args, **kwargs) -> Dict[str, Any]:
        self._reset_peak()
        before_current, _ = tracemalloc.get_traced_memory()
        started = time.perf_counter()
        result = self._run_callable(func, *args, **kwargs)
        duration_ms = (time.perf_counter() - started) * 1000.0
        current, peak = tracemalloc.get_traced_memory()
        return {
            "operation": operation_name,
            "duration_ms": duration_ms,
            "current_memory_kb": current / 1024.0,
            "peak_memory_kb": peak / 1024.0,
            "memory_growth_kb": max(0.0, (current - before_current) / 1024.0),
            "result": result,
        }

    def measure_api_latency(self, request_callable: Callable[..., Any], iterations: int = 10, *args, **kwargs) -> Dict[str, Any]:
        durations = []
        last_result = None
        for _ in range(max(1, iterations)):
            started = time.perf_counter()
            last_result = self._run_callable(request_callable, *args, **kwargs)
            durations.append((time.perf_counter() - started) * 1000.0)
        return {
            "operation": "api_latency",
            "iterations": len(durations),
            "average_latency_ms": sum(durations) / len(durations),
            "max_latency_ms": max(durations),
            "min_latency_ms": min(durations),
            "last_result": last_result,
        }

    def measure_websocket_throughput(self, message_sender: Callable[..., Any], test_message: Any, iterations: int = 100) -> Dict[str, Any]:
        started = time.perf_counter()
        for _ in range(max(0, iterations)):
            self._run_callable(message_sender, test_message)
        total_duration = time.perf_counter() - started
        return {
            "operation": "websocket_throughput",
            "total_iterations": iterations,
            "total_duration_sec": total_duration,
            "throughput_messages_per_sec": iterations / total_duration if total_duration else 0.0,
            "average_latency_ms": (total_duration / iterations) * 1000.0 if iterations else 0.0,
        }

    def measure_checkpoint_overhead(
        self,
        serialization_func: Callable[[Any], Any],
        deserialization_func: Callable[[Any], Any],
        test_payload: Any,
    ) -> Dict[str, Any]:
        serialized = self.profile_operation("checkpoint_serialize", serialization_func, test_payload)
        restored = self.profile_operation("checkpoint_deserialize", deserialization_func, serialized["result"])
        payload = serialized["result"]
        return {
            "operation": "checkpoint_overhead",
            "payload_footprint_bytes": len(payload) if isinstance(payload, (bytes, bytearray)) else 0,
            "serialization_duration_ms": serialized["duration_ms"],
            "deserialization_duration_ms": restored["duration_ms"],
            "combined_overhead_ms": serialized["duration_ms"] + restored["duration_ms"],
            "peak_memory_overhead_kb": max(serialized["peak_memory_kb"], restored["peak_memory_kb"]),
        }

    def measure_queue_delay(self, enqueued_at: float, started_at: float | None = None) -> Dict[str, Any]:
        started = time.time() if started_at is None else started_at
        delay_ms = max(0.0, (started - enqueued_at) * 1000.0)
        return {
            "operation": "queue_delay",
            "delay_ms": delay_ms,
            "status": "degraded" if delay_ms > 5000.0 else "ok",
        }

    def measure_model_load_time(self, loader: Callable[..., Any], *args, **kwargs) -> Dict[str, Any]:
        report = self.profile_operation("model_load_time", loader, *args, **kwargs)
        report["status"] = "degraded" if report["duration_ms"] > 2000.0 else "ok"
        return report

    def profile_training_bottleneck(self, training_callable: Callable[..., Any], *args, **kwargs) -> Dict[str, Any]:
        report = self.profile_operation("training_bottleneck", training_callable, *args, **kwargs)
        report["bottleneck_hint"] = self._classify_latency(report["duration_ms"])
        return report

    def measure_memory_growth(self, operation: Callable[..., Any], iterations: int = 20, *args, **kwargs) -> Dict[str, Any]:
        self._reset_peak()
        start_current, _ = tracemalloc.get_traced_memory()
        for _ in range(max(1, iterations)):
            self._run_callable(operation, *args, **kwargs)
        end_current, peak = tracemalloc.get_traced_memory()
        return {
            "operation": "memory_growth",
            "iterations": iterations,
            "memory_growth_kb": max(0.0, (end_current - start_current) / 1024.0),
            "peak_memory_kb": peak / 1024.0,
        }

    @staticmethod
    def _run_callable(func: Callable[..., Any], *args, **kwargs) -> Any:
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            return asyncio.run(result)
        return result

    @staticmethod
    def _reset_peak() -> None:
        if hasattr(tracemalloc, "reset_peak"):
            tracemalloc.reset_peak()

    @staticmethod
    def _classify_latency(duration_ms: float) -> str:
        if duration_ms >= 10_000:
            return "long-running training path; profile data loading and estimator fit separately"
        if duration_ms >= 1_000:
            return "moderate training latency; inspect feature generation and model fit"
        return "no major training bottleneck detected in sampled callable"
