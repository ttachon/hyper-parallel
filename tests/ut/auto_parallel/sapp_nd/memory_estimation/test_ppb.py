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
"""Tests for the pipeline balancer's layer description: the memory of its recompute options.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/memory_estimation/test_ppb.py -v
"""
import unittest
from types import SimpleNamespace
from typing import Any, Optional, Tuple

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._ppb import _PPB
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType

MEGABYTE = 1024 * 1024
_OPS = ("attBMM", "headCast", "dropout", "softmax", "normOp", "gather", "ffAct")
_KEEP_ALL = dict.fromkeys(_OPS, 1)


class _Memory:
    """Per micro-batch in flight, a layer keeps 8 MB of activation and a 2 MB gathered buffer.

    Each op a selective layer recomputes frees 1 MB of activation; recomputed
    gathers free the gathered buffer. A fully recomputed layer keeps 1 MB.
    The gathered buffers share memory with a *dp* MB parameter buffer, as
    the memory model's ``max(dp, tp)`` has them.
    """

    def __init__(self, ctx: Context, ccfg: SimpleNamespace, dp: int = 0) -> None:
        """Answer for the layer *ctx* is on, with *ccfg*'s switches."""
        self.ctx, self.ccfg, self.dp = ctx, ccfg, dp * MEGABYTE

    def __call__(self, ppb: bool = False, default_micro_factor: Optional[int] = None) -> Tuple[int, int]:
        """``(activation, communication buffers)`` of the current layer, *default_micro_factor* in flight."""
        micro = 1 if ppb else default_micro_factor
        if self.ctx.current_node == LayerType.FULL_REC_LAYER:
            return MEGABYTE * micro, self.dp
        switches = vars(self.ccfg.rec_op)
        selective = self.ctx.current_node == LayerType.SEL_REC_LAYER
        freed = sum(1 for op in _OPS if op != "gather" and selective and not switches[op])
        gathered = 0 if selective and not switches["gather"] else 2 * MEGABYTE * micro
        return (8 - freed) * MEGABYTE * micro, max(self.dp, gathered)


def _describe(switches: dict, dp: int = 0, **strategy: Any) -> Tuple[dict, _Memory]:
    """The description ``lay_ppb`` builds for a body layer, and the memory it read."""
    ctx = Context()
    ctx.head_node, ctx.tail_node = "head", "tail"
    ctx.current_node = LayerType.NOT_REC_LAYER
    ccfg = SimpleNamespace(model_name="unit", rec_op=SimpleNamespace(**switches), **strategy)
    memory = _Memory(ctx, ccfg, dp)
    ppb = _PPB(SimpleNamespace(ppb_combined=[]), memory)
    return ppb.lay_ppb(ccfg, ctx, 4 * MEGABYTE), memory


class TestMemorySplit(unittest.TestCase):
    """The balancer charges an option's memory once per micro-batch in flight."""

    def test_buffers_that_grow_with_the_micro_batches_count_per_micro_batch(self):
        """
        Feature: _PPB.lay_ppb.
        Description: A body whose gathered buffer grows with the micro-batches
            in flight.
        Expectation: The options that keep it are charged it with their
            activations; the layer's constant memory does not include it.
        """
        desc, _ = _describe(dict(_KEEP_ALL, softmax=0))
        self.assertEqual(desc["memory_parameter"], 4)
        self.assertEqual(
            (desc["memory_activation"], desc["memory_select_rec"], desc["memory_recompute"]), (8 + 2, 7 + 2, 1)
        )

    def test_buffers_that_do_not_grow_are_a_constant_of_the_layer(self):
        """
        Feature: _PPB.lay_ppb.
        Description: A 5 MB parameter buffer, larger than the gathered buffers
            of every micro-batch a stage keeps in flight.
        Expectation: The 5 MB go to the layer's constant memory, once.
        """
        desc, _ = _describe(dict(_KEEP_ALL, softmax=0), dp=5)
        self.assertEqual(desc["memory_parameter"], 4 + 5)
        self.assertEqual((desc["memory_activation"], desc["memory_select_rec"], desc["memory_recompute"]), (8, 7, 1))

    def test_a_stage_keeping_the_most_micro_batches_is_charged_what_the_memory_model_keeps(self):
        """
        Feature: _PPB.lay_ppb.
        Description: A 1F1B stage of PP 4 with 8 micro-batches keeps 4 in
            flight, each with its gathered buffer.
        Expectation: Beside the layer's parameters, what the balancer charges
            the stage, the constant once and an option's memory per
            micro-batch, is what the memory model keeps with 4 in flight,
            under every option.
        """
        desc, memory = _describe(dict(_KEEP_ALL, softmax=0), p=4, m=8)
        once = desc["memory_parameter"] - 4
        for key, node in (
            ("memory_activation", LayerType.NOT_REC_LAYER),
            ("memory_select_rec", LayerType.SEL_REC_LAYER),
            ("memory_recompute", LayerType.FULL_REC_LAYER),
        ):
            memory.ctx.current_node = node
            activation, buffers = memory(default_micro_factor=4)
            with self.subTest(option=key):
                self.assertEqual(once + 4 * desc[key], (activation + buffers) / MEGABYTE)


if __name__ == "__main__":
    unittest.main()
