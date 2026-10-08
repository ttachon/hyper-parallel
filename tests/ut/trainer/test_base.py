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
"""Unit tests for the trainer's model build arguments."""

import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from hyper_parallel.trainer.config.target import Target
from tests.common.mark_utils import arg_mark

_BATCHING_MODULE = ModuleType("hyper_parallel.data.batching")
_BATCHING_MODULE.build_dataloader = MagicMock()

# This module tests the model build, not dataloader construction. Stub that
# boundary so collecting the UT does not require the optional torchdata.
with patch.dict(sys.modules, {"hyper_parallel.data.batching": _BATCHING_MODULE}):
    base_module = importlib.import_module("hyper_parallel.trainer.base")


def _declares_plan(activation_checkpoint_layers=None) -> None:  # pylint: disable=unused-argument
    """Model target that declares the per-layer activation checkpoint plan."""


def _takes_any(**kwargs) -> None:  # pylint: disable=unused-argument
    """Model target that accepts any keyword argument."""


def _declares_nothing(activation_checkpoint=None) -> None:  # pylint: disable=unused-argument
    """Model target written before the per-layer plan existed."""


class TestActivationCheckpointLayerKwargs(unittest.TestCase):
    """Tests for carrying ``activation_checkpoint.layers`` to the model target."""

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_plan_reaches_targets_that_can_take_it(self) -> None:
        """
        Feature: BaseTrainer per-layer activation checkpoint plan.
        Description: Build the model kwargs with and without a plan, for targets that declare it or take any kwarg.
        Expectation: The plan is passed only when set, and to both kinds of target.
        """
        plan = {"6-7": "off"}
        for target in (_declares_plan, _takes_any):
            with self.subTest(target=target.__name__):
                model_target = SimpleNamespace(callable=target)
                kwargs = base_module._activation_checkpoint_layer_kwargs(model_target, plan)
                self.assertEqual(kwargs, {"activation_checkpoint_layers": plan}, f"kwargs={kwargs}")
                kwargs = base_module._activation_checkpoint_layer_kwargs(model_target, None)
                self.assertEqual(kwargs, {}, f"kwargs={kwargs}")

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_plan_is_refused_by_a_target_that_would_drop_it(self) -> None:
        """
        Feature: BaseTrainer per-layer activation checkpoint plan.
        Description: Set a plan for a target that neither declares it nor takes any kwarg.
        Expectation: The build is refused instead of training without the plan.
        """
        model_target = SimpleNamespace(callable=_declares_nothing)
        kwargs = base_module._activation_checkpoint_layer_kwargs(model_target, None)
        self.assertEqual(kwargs, {}, f"kwargs={kwargs}")
        with self.assertRaisesRegex(ValueError, "takes no activation_checkpoint_layers argument"):
            base_module._activation_checkpoint_layer_kwargs(model_target, {"0": "off"})

    @arg_mark(
        plat_marks=["cpu_linux", "cpu_macos"],
        level_mark="level0",
        card_mark="allcards",
        essential_mark="essential",
    )
    def test_build_model_passes_the_plan_to_the_model_target(self) -> None:
        """
        Feature: BaseTrainer per-layer activation checkpoint plan.
        Description: Build the model from a target that records its keyword arguments.
        Expectation: The target receives the configured mode and plan.
        """
        received = {}

        def _build(**kwargs: Any) -> MagicMock:
            """Record the build arguments and return a stand-in model."""
            received.update(kwargs)
            return MagicMock()

        trainer = MagicMock()
        trainer.global_rank = 1
        trainer.config = SimpleNamespace(
            peft=None,
            model=Target(_build, target_path=f"{__name__}._build"),
            activation_checkpoint=SimpleNamespace(mode="full", swap_inputs=False, layers={"6-7": "off"}),
            activation_swap="none",
            compile=None,
            model_init_dtype=None,
            accelerator=SimpleNamespace(loss_parallel=False),
        )

        base_module.BaseTrainer._build_model(trainer)

        passed = {name: received.get(name) for name in ("activation_checkpoint", "activation_checkpoint_layers")}
        expected = {"activation_checkpoint": "full", "activation_checkpoint_layers": {"6-7": "off"}}
        self.assertEqual(passed, expected, f"passed={passed}")


if __name__ == "__main__":
    unittest.main()
