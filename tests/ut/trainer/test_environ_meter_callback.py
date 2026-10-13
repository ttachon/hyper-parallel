# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Unit tests for Trainer environment and throughput metrics."""

import math
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

import hyper_parallel.trainer.callbacks.environ_meter_callback as environ_meter_module
from hyper_parallel.trainer.callbacks.environ_meter_callback import (
    ALLOCATOR_COUNTERS,
    ALLOCATOR_STATS_ENV,
    EnvironMeterCallback,
)
from hyper_parallel.trainer.state import TrainerState
from tests.common.mark_utils import arg_mark

GIB = 1024 ** 3


def _counter_row(**gib: float) -> list[int]:
    """Return one rank's allocator counters, in ALLOCATOR_COUNTERS order.

    Args:
        **gib: Counter values by metric name, GiB for names ending in ``_gb``.

    Returns:
        The counters as memory_stats() reports them: bytes, or plain counts.
    """
    row = []
    for _, name in ALLOCATOR_COUNTERS:
        value = gib.get(name, 1.0)
        row.append(int(value * GIB) if name.endswith("_gb") else int(value))
    return row


class _NoItemScalar:
    """Represent a device scalar whose host conversion is forbidden."""

    def detach(self) -> "_NoItemScalar":
        """Return a detached scalar view."""
        return self

    def clone(self) -> "_NoItemScalar":
        """Return an independently owned scalar."""
        return _NoItemScalar()

    def item(self) -> float:
        """Fail if metric collection synchronizes the scalar to the host."""
        raise AssertionError("micro-step metric collection must not call item()")


class TestEnvironMeterCallback(unittest.TestCase):
    """Verify token throughput uses streamed, prepared text batches."""

    @staticmethod
    def _build_trainer(mesh: object = None) -> SimpleNamespace:
        """Build the callback's minimal Trainer dependency surface."""
        return SimpleNamespace(
            mesh=mesh,
            lr_scheduler=None,
            optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_micro_step_publishes_nonzero_token_throughput(
            self,
            mock_time: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Publish the accumulated micro-step token count as throughput.

        Feature: Trainer throughput metrics.
        Description: Stream a prepared text micro-batch into the environment meter.
        Expectation: Step tokens and tokens per second are positive and scalar state is released.
        """
        del mock_time, mock_world_size, mock_device_type
        trainer = self._build_trainer()
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {
                "input_ids": torch.ones(2, 4),
                "token_count": torch.tensor(6),
            },
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        metrics = trainer.step_env_metrics
        self.assertEqual(
            metrics["data/step_tokens"],
            6.0,
            f"expected six accumulated tokens, got={metrics['data/step_tokens']}",
        )
        self.assertEqual(
            metrics["performance/tokens_per_second"],
            3.0,
            f"expected three tokens per second, got={metrics['performance/tokens_per_second']}",
        )
        step_tokens = vars(callback)["_local_step_tokens"]
        self.assertIsNone(
            step_tokens,
            f"step token scalar must be released after publishing, got={step_tokens!r}",
        )

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_micro_step_avoids_host_conversion_and_step_begin_resets(self) -> None:
        """Keep device token scalars asynchronous and clear them at the next step.

        Feature: Environment meter memory lifecycle.
        Description: Accumulate a detached scalar that rejects host conversion and begin a new step.
        Expectation: Accumulation avoids item calls and the next step clears prior scalar state.
        """
        callback = EnvironMeterCallback(self._build_trainer())
        state = TrainerState(global_step=0, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {"token_count": _NoItemScalar()},
        )

        step_tokens = vars(callback)["_local_step_tokens"]
        self.assertIsInstance(
            step_tokens,
            _NoItemScalar,
            f"expected detached scalar state, got={step_tokens!r}",
        )
        callback.on_step_begin(state)
        step_tokens = vars(callback)["_local_step_tokens"]
        self.assertIsNone(
            step_tokens,
            f"new step must release prior scalar state, got={step_tokens!r}",
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="cpu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=4)
    @patch.object(environ_meter_module, "all_reduce")
    @patch.object(environ_meter_module.time, "perf_counter", side_effect=(10.0, 12.0))
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_context_parallel_reduction_does_not_duplicate_samples(
            self,
            mock_time: Mock,
            mock_all_reduce: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Sum CP-sharded tokens while counting each logical sample once.

        Feature: Context-parallel throughput accounting.
        Description: Reduce local token and sample counts over a mocked DP+CP group.
        Expectation: Tokens sum across ranks while CP-replicated samples are counted once.
        """
        del mock_time, mock_world_size, mock_device_type
        mock_all_reduce.side_effect = lambda value, op, group: float(value) * 4 if op == "sum" else float(value)
        mesh = SimpleNamespace(dp_cp_mesh=None, cp_size=2)
        trainer = self._build_trainer(mesh=mesh)
        callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)

        callback.on_step_begin(state)
        callback.on_micro_step_begin(
            state,
            {
                "input_ids": torch.ones(2, 4),
                "token_count": torch.tensor(6),
            },
        )
        callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)

        metrics = trainer.step_env_metrics
        self.assertEqual(
            metrics["data/step_tokens"],
            24.0,
            f"expected tokens summed across DP+CP, got={metrics['data/step_tokens']}",
        )
        self.assertEqual(
            metrics["data/step_samples"],
            4.0,
            f"expected CP replicas removed from samples, got={metrics['data/step_samples']}",
        )

    def _run_one_step(self, stats: dict[str, int], switch: str) -> tuple[dict[str, float], Mock]:
        """Run one optimizer step on an accelerator whose allocator reports *stats*.

        Args:
            stats: What the device's memory_stats() returns.
            switch: Value of the allocator statistics switch while the callback is built.

        Returns:
            The published environment metrics and the device's memory_stats mock.
        """
        memory_stats = Mock(return_value=stats)
        device = SimpleNamespace(
            max_memory_allocated=Mock(return_value=2 * GIB),
            max_memory_reserved=Mock(return_value=3 * GIB),
            memory_stats=memory_stats,
        )
        with patch.dict(os.environ, {ALLOCATOR_STATS_ENV: switch}):
            trainer = self._build_trainer()
            callback = EnvironMeterCallback(trainer)
        state = TrainerState(global_step=1, epoch=0)
        with patch.object(environ_meter_module, "get_torch_device", return_value=device):
            callback.on_step_begin(state)
            callback.on_step_end(state, loss=1.0, loss_dict=None, grad_norm=0.5)
        return trainer.step_env_metrics, memory_stats

    @patch.object(environ_meter_module, "get_device_type", return_value="npu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_allocator_counters_stay_off_without_the_switch(
            self,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Publish only the two peaks unless the allocator statistics are asked for.

        Feature: Allocator statistics in the environment meter.
        Description: Run one step on an accelerator with the switch at its default.
        Expectation: No allocator metric is published and memory_stats() is never read.
        """
        del mock_world_size, mock_device_type
        metrics, memory_stats = self._run_one_step({"reserved_bytes.all.peak": 3 * GIB}, switch="0")

        allocator = sorted(name for name in metrics if name.startswith("memory/allocator_"))
        self.assertEqual(allocator, [], f"expected no allocator metric with the switch off, got={allocator}")
        self.assertEqual(
            memory_stats.call_count,
            0,
            f"expected memory_stats() unread with the switch off, got calls={memory_stats.call_count}",
        )
        self.assertEqual(
            metrics["memory/device_max_reserved_gb"],
            3.0,
            f"expected the reserved peak still published, got={metrics['memory/device_max_reserved_gb']}",
        )

    @patch.object(environ_meter_module, "get_device_type", return_value="npu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=1)
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_allocator_counters_publish_in_gib_and_mark_missing_ones(
            self,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Convert byte counters to GiB, keep counts as counts, and flag what is absent.

        Feature: Allocator statistics in the environment meter.
        Description: Run one single-process step with the switch on and one counter missing.
        Expectation: Bytes become GiB, counts are unchanged, and the missing counter reads NaN.
        """
        del mock_world_size, mock_device_type
        stats = {
            "reserved_bytes.all.peak": 40 * GIB,
            "allocated_bytes.all.peak": 26 * GIB,
            "inactive_split_bytes.all.peak": 23 * GIB,
            "num_alloc_retries": 3,
        }
        metrics, memory_stats = self._run_one_step(stats, switch="1")

        self.assertEqual(memory_stats.call_count, 1, f"expected one read a step, got={memory_stats.call_count}")
        expected = {
            "memory/allocator_rank": 0.0,
            "memory/allocator_reserved_peak_gb": 40.0,
            "memory/allocator_allocated_peak_gb": 26.0,
            "memory/allocator_inactive_split_peak_gb": 23.0,
            "memory/allocator_alloc_retries": 3.0,
        }
        got = {name: metrics.get(name) for name in expected}
        self.assertEqual(got, expected, f"expected={expected}, got={got}")
        requested = metrics["memory/allocator_requested_peak_gb"]
        self.assertTrue(math.isnan(requested), f"expected a missing counter to read NaN, got={requested}")

    @patch.object(environ_meter_module, "get_device_type", return_value="npu")
    @patch.object(environ_meter_module, "get_world_size_safe", return_value=4)
    @patch.object(environ_meter_module, "get_global_rank_safe", return_value=1)
    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="onecard",
        essential_mark="essential",
    )
    def test_allocator_counters_come_from_the_rank_that_reserves_most(
            self,
            mock_global_rank: Mock,
            mock_world_size: Mock,
            mock_device_type: Mock,
    ) -> None:
        """Report every counter from one rank, the one with the largest reserved peak.

        Feature: Allocator statistics in the environment meter.
        Description: Run rank 1 of four, where ranks 2 and 3 tie on the reserved peak and
            rank 1 holds the largest inactive split.
        Expectation: Rank 1 sends its counters in its own slot only, and the lowest of the
            tied ranks is reported, with its own inactive split.
        """
        del mock_global_rank, mock_world_size, mock_device_type
        rows = [
            _counter_row(reserved_peak_gb=30.0, inactive_split_peak_gb=5.0),
            _counter_row(reserved_peak_gb=31.0, inactive_split_peak_gb=9.0),
            _counter_row(reserved_peak_gb=34.0, inactive_split_peak_gb=7.0, alloc_retries=2),
            _counter_row(reserved_peak_gb=34.0, inactive_split_peak_gb=8.0),
        ]
        sent = []

        def sum_four_ranks(value: object, op: str, group: object) -> object:
            """Fill every rank's slot as the sum would; pass every scalar metric through."""
            if not isinstance(value, list):
                return float(value)
            sent.append((list(value), op, group))
            return [float(counter) for row in rows for counter in row]

        own_stats = {key: counter for (key, _), counter in zip(ALLOCATOR_COUNTERS, rows[1])}
        with patch.object(environ_meter_module, "all_reduce", side_effect=sum_four_ranks):
            metrics, _ = self._run_one_step(own_stats, switch="1")

        width = len(ALLOCATOR_COUNTERS)
        expected_sent = [([0] * width + rows[1] + [0] * (2 * width), "sum", None)]
        self.assertEqual(sent, expected_sent, f"expected rank 1's slot only, sent={sent}")
        expected = {
            "memory/allocator_rank": 2.0,
            "memory/allocator_reserved_peak_gb": 34.0,
            "memory/allocator_inactive_split_peak_gb": 7.0,
            "memory/allocator_alloc_retries": 2.0,
        }
        got = {name: metrics.get(name) for name in expected}
        self.assertEqual(got, expected, f"expected={expected}, got={got}")


if __name__ == "__main__":
    unittest.main()
