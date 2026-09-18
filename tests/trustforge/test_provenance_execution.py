"""Tests for TrustForge execution identity construction."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from trustforge.provenance import (
    ExecutionIdentityError,
    build_execution_id,
    format_execution_timestamp,
)


class ExecutionIdentityTests(unittest.TestCase):
    """Validate portable execution identity behavior."""

    def test_utc_timestamp_format(self) -> None:
        started_at = datetime(
            2026,
            9,
            17,
            14,
            30,
            0,
            tzinfo=timezone.utc,
        )

        self.assertEqual(
            format_execution_timestamp(started_at),
            "20260917T143000Z",
        )

    def test_timestamp_is_converted_to_utc(self) -> None:
        eastern = timezone(
            -timedelta(hours=4)
        )

        started_at = datetime(
            2026,
            9,
            17,
            10,
            30,
            0,
            tzinfo=eastern,
        )

        self.assertEqual(
            format_execution_timestamp(started_at),
            "20260917T143000Z",
        )

    def test_naive_datetime_is_rejected(self) -> None:
        with self.assertRaises(ExecutionIdentityError):
            format_execution_timestamp(
                datetime(2026, 9, 17, 14, 30, 0)
            )

    def test_local_execution_id(self) -> None:
        started_at = datetime(
            2026,
            9,
            17,
            14,
            30,
            0,
            tzinfo=timezone.utc,
        )

        self.assertEqual(
            build_execution_id(
                started_at,
                "talon",
            ),
            "20260917T143000Z_talon_local",
        )

    def test_scheduler_execution_id(self) -> None:
        started_at = datetime(
            2026,
            9,
            17,
            14,
            30,
            0,
            tzinfo=timezone.utc,
        )

        self.assertEqual(
            build_execution_id(
                started_at,
                "talon",
                scheduler_job_id="123456",
            ),
            "20260917T143000Z_talon_job123456",
        )

    def test_scheduler_array_execution_id(self) -> None:
        started_at = datetime(
            2026,
            9,
            17,
            14,
            30,
            0,
            tzinfo=timezone.utc,
        )

        self.assertEqual(
            build_execution_id(
                started_at,
                "talon",
                scheduler_job_id="123456",
                scheduler_task_id="7",
            ),
            "20260917T143000Z_talon_job123456_task7",
        )

    def test_scheduler_task_requires_job(self) -> None:
        started_at = datetime(
            2026,
            9,
            17,
            14,
            30,
            0,
            tzinfo=timezone.utc,
        )

        with self.assertRaises(ExecutionIdentityError):
            build_execution_id(
                started_at,
                "talon",
                scheduler_task_id="7",
            )

    def test_invalid_machine_component_is_rejected(self) -> None:
        started_at = datetime(
            2026,
            9,
            17,
            14,
            30,
            0,
            tzinfo=timezone.utc,
        )

        with self.assertRaises(ExecutionIdentityError):
            build_execution_id(
                started_at,
                "../talon",
            )

    def test_positive_attempt_is_appended(self) -> None:
        started_at = datetime(
            2026,
            9,
            17,
            14,
            30,
            0,
            tzinfo=timezone.utc,
        )

        self.assertEqual(
            build_execution_id(
                started_at,
                "talon",
                scheduler_job_id="123456",
                scheduler_task_id="7",
                attempt=2,
            ),
            "20260917T143000Z_talon_job123456_task7_attempt2",
        )

    def test_non_positive_attempt_is_rejected(self) -> None:
        started_at = datetime(
            2026,
            9,
            17,
            14,
            30,
            0,
            tzinfo=timezone.utc,
        )

        with self.assertRaises(ExecutionIdentityError):
            build_execution_id(
                started_at,
                "talon",
                attempt=0,
            )

    def test_boolean_attempt_is_rejected(self) -> None:
        started_at = datetime(
            2026,
            9,
            17,
            14,
            30,
            0,
            tzinfo=timezone.utc,
        )

        with self.assertRaises(ExecutionIdentityError):
            build_execution_id(
                started_at,
                "talon",
                attempt=True,
            )


if __name__ == "__main__":
    unittest.main()
