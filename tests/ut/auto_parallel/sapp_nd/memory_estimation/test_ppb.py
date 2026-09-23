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
"""Tests for the times in the pipeline balancer's layer description.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/memory_estimation/test_ppb.py -v
"""
import unittest
from types import SimpleNamespace
from typing import Any, Tuple

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._ppb import _PPB
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute

MEGABYTE = 1024 * 1024
_KEEP_ALL = dict.fromkeys(("attBMM", "headCast", "dropout", "softmax", "normOp", "gather", "ffAct"), 1)

# (forward, backward) per layer type, as a pricer would give them.
_TIMES = {
    LayerType.EMBEDDING_LAYER: (1.0, 2.0),
    LayerType.OUTPUT_LAYER: (3.0, 6.0),
    LayerType.NOT_REC_LAYER: (10.0, 20.0),
    LayerType.SEL_REC_LAYER: (10.0, 21.0),
    LayerType.FULL_REC_LAYER: (10.0, 30.0),
}


class _Pricer:
    """Records every call and answers from ``_TIMES``."""

    def __init__(self) -> None:
        """No call yet."""
        self.calls = []

    def __call__(self, cfg: Any, hook: Any, layer_type: LayerType) -> Tuple[float, float]:
        """Record the call, answer from ``_TIMES``."""
        self.calls.append((cfg, hook, layer_type))
        return _TIMES[layer_type]


def _describe(node, switches, pricer=None, hook=None):
    """The description ``lay_ppb`` builds for one *node*."""
    ppb = _PPB(SimpleNamespace(ppb_combined=[]), lambda ppb=False: (MEGABYTE, 0))
    ppb.layer_times = pricer
    ctx = Context()
    ctx.head_node, ctx.tail_node = "head", "tail"
    ctx.current_node = node
    ccfg = SimpleNamespace(model_name="unit", rec_op=SimpleNamespace(**switches))
    return ppb.lay_ppb(ccfg, ctx, 4 * MEGABYTE, hook), ccfg


class TestTimedDescription(unittest.TestCase):
    """With a pricer, every option carries a time from the performance model."""

    def test_body_options_carry_their_backward_time(self):
        """Selective recompute of softmax is offered, with its own time."""
        hook = object()
        pricer = _Pricer()
        desc, ccfg = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0), pricer, hook)
        times = {k: desc[k] for k in ("time", "forward_time", "backward_time", "select_rec_time", "recompute_time")}
        self.assertEqual(times, {"time": 10.0, "forward_time": 10.0, "backward_time": 20.0,
                                 "select_rec_time": 21.0, "recompute_time": 30.0})
        self.assertIn("memory_select_rec", desc)
        self.assertTrue(all(cfg is ccfg and got is hook for cfg, got, _ in pricer.calls), pricer.calls)

    def test_selective_that_keeps_everything_is_not_offered(self):
        """It is the plain layer again: offering it would hand the balancer a tie."""
        desc, _ = _describe(LayerType.NOT_REC_LAYER, _KEEP_ALL, _Pricer())
        self.assertNotIn("select_rec_time", desc)
        self.assertNotIn("memory_select_rec", desc)
        self.assertEqual((desc["backward_time"], desc["recompute_time"]), (20.0, 30.0))

    def test_head_and_tail_are_priced_as_embedding_and_output(self):
        """No hook reaches them: they belong to no layer group."""
        pricer = _Pricer()
        head, _ = _describe("head", _KEEP_ALL, pricer, hook=object())
        tail, _ = _describe("tail", _KEEP_ALL, pricer, hook=object())
        self.assertEqual((head["forward_time"], head["backward_time"]), (1.0, 2.0))
        self.assertEqual((tail["forward_time"], tail["backward_time"]), (3.0, 6.0))
        self.assertEqual([(hook, kind) for _, hook, kind in pricer.calls],
                         [(None, LayerType.EMBEDDING_LAYER), (None, LayerType.OUTPUT_LAYER)])

    def test_without_a_pricer_the_description_is_unchanged(self):
        """The placeholder time stays, and so does every memory option."""
        desc, _ = _describe(LayerType.NOT_REC_LAYER, _KEEP_ALL)
        self.assertEqual(desc["time"], 1)
        self.assertIn("memory_select_rec", desc)
        self.assertNotIn("forward_time", desc)


class TestBalancerReadsTheKeys(unittest.TestCase):
    """The names ND writes are the names the balancer reads."""

    def test_time_and_memory_keys_match(self):
        """Every option's time and memory key, with no stray whitespace."""
        desc, _ = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0), _Pricer())
        pairs = {Recompute.TYPE.NONE, Recompute.TYPE.SLCT, Recompute.TYPE.FULL}
        for rec in pairs:
            self.assertIn(Recompute.JSON_TIME_NAME[rec], desc, rec)
            self.assertIn(Recompute.JSON_MEMORY_NAME[rec], desc, rec)
        for name in list(Recompute.JSON_TIME_NAME.values()) + list(Recompute.JSON_MEMORY_NAME.values()):
            self.assertEqual(name, name.strip(), f"key {name!r} carries whitespace")


if __name__ == "__main__":
    unittest.main()
