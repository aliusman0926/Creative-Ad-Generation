"""Prometheus metrics utilities for online and batch workflows."""

from __future__ import annotations

from typing import Iterable

from prometheus_client import Gauge


class BatchMetricsExporter:
    """Expose aggregate metrics for batch inference or evaluation runs.

    This helper keeps the Prometheus wiring small for offline or scheduled jobs
    that want to push summary metrics for a batch without re-implementing the
    instrumentation logic.
    """

    def __init__(self, registry=None) -> None:
        self.batch_latency_seconds = Gauge(
            "batch_latency_seconds",
            "Average latency for the batch",
            labelnames=["batch_name"],
            registry=registry,
        )
        self.batch_throughput = Gauge(
            "batch_throughput_records",
            "Records processed in the batch",
            labelnames=["batch_name"],
            registry=registry,
        )
        self.batch_quality_score = Gauge(
            "batch_quality_score",
            "Average content quality score for the batch",
            labelnames=["batch_name"],
            registry=registry,
        )
        self.batch_error_rate = Gauge(
            "batch_error_rate",
            "Error rate for the batch execution",
            labelnames=["batch_name"],
            registry=registry,
        )

    def export(
        self,
        batch_name: str,
        latencies: Iterable[float],
        qualities: Iterable[float],
        errors: int,
        total_records: int,
    ) -> None:
        """Publish batch metrics to the configured registry.

        Args:
            batch_name: Identifier for the batch run.
            latencies: Individual operation latencies.
            qualities: Per-record quality scores.
            errors: Count of failed records.
            total_records: Total records attempted in the batch.
        """

        total_records = max(total_records, 1)
        latency_values = list(latencies)
        quality_values = list(qualities)

        avg_latency = sum(latency_values) / len(latency_values) if latency_values else 0.0
        avg_quality = sum(quality_values) / len(quality_values) if quality_values else 0.0
        error_rate = errors / total_records

        self.batch_latency_seconds.labels(batch_name=batch_name).set(avg_latency)
        self.batch_throughput.labels(batch_name=batch_name).set(total_records)
        self.batch_quality_score.labels(batch_name=batch_name).set(avg_quality)
        self.batch_error_rate.labels(batch_name=batch_name).set(error_rate)

