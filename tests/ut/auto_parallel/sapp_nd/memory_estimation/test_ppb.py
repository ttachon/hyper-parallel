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
"""Tests for the pipeline balancer's layer description: its recompute options,
their memory and their times.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/memory_estimation/test_ppb.py -v
"""
import itertools
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Optional, Tuple

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._ppb import _MEMORY_KEY, _OPTIONS, _PPB, _TIME_KEY
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.recompute.front import price_option
from hyper_parallel.auto_parallel.sapp_ppb.utils import recompute as Recompute
from hyper_parallel.auto_parallel.sapp_ppb.utils.layer import generate_layers_list

MEGABYTE = 1024 * 1024
_OPS = ("attBMM", "headCast", "dropout", "softmax", "normOp", "gather", "ffAct")
_KEEP_ALL = dict.fromkeys(_OPS, 1)

# (forward, backward) per layer type, as a pricer would give them.
_TIMES = {
    LayerType.EMBEDDING_LAYER: (1.0, 2.0),
    LayerType.OUTPUT_LAYER: (3.0, 6.0),
    LayerType.NOT_REC_LAYER: (10.0, 20.0),
    LayerType.SEL_REC_LAYER: (10.0, 20.0),
    LayerType.FULL_REC_LAYER: (10.0, 30.0),
}


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


class _Pricer:
    """Records every call; recomputing an op adds 1 to a selective layer's backward, the gathers 3."""

    def __init__(self) -> None:
        """No call yet."""
        self.calls = []

    def __call__(self, cfg: Any, kind: Any, layer_type: LayerType,
                 switches: Optional[dict] = None) -> Tuple[float, float]:
        """Record the call, answer from ``_TIMES``."""
        self.calls.append((cfg, kind, layer_type, switches))
        forward, backward = _TIMES[layer_type]
        if layer_type == LayerType.SEL_REC_LAYER:
            backward += sum(3 if op == "gather" else 1 for op, keep in switches.items() if not keep)
        return forward, backward


def _describe(node: Any, switches: dict, pricer: Optional[_Pricer] = None, kind: Any = None,
              dp: int = 0, model: str = "unit") -> Tuple[dict, SimpleNamespace]:
    """The description ``lay_ppb`` builds for one *node*, and the config it read."""
    ctx = Context()
    ctx.head_node, ctx.tail_node = "head", "tail"
    ctx.current_node = node
    ccfg = SimpleNamespace(model_name=model, rec_op=SimpleNamespace(**switches))
    ppb = _PPB(SimpleNamespace(ppb_combined=[]), _Memory(ctx, ccfg, dp))
    ppb.layer_times = pricer
    return ppb.lay_ppb(ccfg, ctx, 4 * MEGABYTE, kind), ccfg


def _options(desc: dict) -> dict:
    """``{option: (memory, backward time)}`` of the options *desc* offers."""
    return {name: (desc[_MEMORY_KEY[name]], desc.get(_TIME_KEY[name]))
            for name in _OPTIONS if _MEMORY_KEY[name] in desc}


def _measure(dp: int, counts: Tuple[int, ...]) -> Tuple[Any, _Memory, SimpleNamespace]:
    """The profile ``lay_ppb`` measures for a body when its stages keep *counts* micro-batches in flight.

    Returns:
        ``(profile, memory, config)``: the memory answers for the config's
        switches and the layer its context is on.
    """
    ctx = Context()
    ctx.head_node, ctx.tail_node = "head", "tail"
    ctx.current_node = LayerType.NOT_REC_LAYER
    ccfg = SimpleNamespace(model_name="unit", rec_op=SimpleNamespace(**_KEEP_ALL))
    memory = _Memory(ctx, ccfg, dp)
    ppb = _PPB(SimpleNamespace(ppb_combined=[]), memory)
    ppb.layer_times, ppb.profiles = _Pricer(), {}
    ppb.profile_in_flight, ppb.profile_counts = max(counts), counts
    ppb.lay_ppb(ccfg, ctx, 4 * MEGABYTE)
    return ppb.profiles["unit", None], memory, ccfg


def _withdraw(descriptions: list) -> list:
    """*descriptions*, once the options no body is better off with are withdrawn."""
    _PPB(SimpleNamespace(ppb_combined=[]), None).ppb_withdraw_dominated(descriptions)
    return descriptions


class TestTimedDescription(unittest.TestCase):
    """With a pricer, a body offers every option, each with its memory and time."""

    def test_every_option_carries_its_memory_and_backward_time(self):
        """
        Feature: _PPB.lay_ppb.
        Description: A body whose config recomputes its softmax, with a pricer.
        Expectation: SLCT recomputes the softmax, COMM the gathers and BOTH
            both, each priced with its own switches on the layer's config
            and kind; the forward time replaces the placeholder.
        """
        kind, pricer = object(), _Pricer()
        desc, ccfg = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0), pricer, kind)
        self.assertEqual(_options(desc), {"NONE": (10, 20.0), "SLCT": (9, 21.0), "COMM": (8, 23.0),
                                          "BOTH": (7, 24.0), "FULL": (1, 30.0)})
        self.assertEqual((desc["memory_parameter"], desc["forward_time"], desc["time"]), (4, 10.0, 10.0))
        selective = {tuple(sorted(op for op, keep in switches.items() if not keep))
                     for _, _, layer_type, switches in pricer.calls if layer_type == LayerType.SEL_REC_LAYER}
        self.assertEqual(selective, {("softmax",), ("gather",), ("gather", "softmax")})
        self.assertTrue(all(cfg is ccfg and got is kind for cfg, got, _, _ in pricer.calls), pricer.calls)

    def test_the_config_switches_come_back_unchanged(self):
        """
        Feature: _PPB.lay_ppb.
        Description: Describe a body, which sets the switches of COMM and BOTH
            on the config to size them.
        Expectation: The config's own switches are back afterwards.
        """
        desc, ccfg = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0), _Pricer())
        self.assertIn("memory_both_comm_select", desc)
        self.assertEqual(vars(ccfg.rec_op), dict(_KEEP_ALL, softmax=0))

    def test_head_and_tail_are_priced_as_embedding_and_output(self):
        """
        Feature: _PPB.lay_ppb.
        Description: Describe the head and the tail with a pricer and a kind.
        Expectation: No kind reaches them: they are no layer of the stack.
        """
        pricer = _Pricer()
        head, _ = _describe("head", _KEEP_ALL, pricer, kind=object())
        tail, _ = _describe("tail", _KEEP_ALL, pricer, kind=object())
        self.assertEqual((head["forward_time"], head["backward_time"]), (1.0, 2.0))
        self.assertEqual((tail["forward_time"], tail["backward_time"]), (3.0, 6.0))
        self.assertEqual([(kind, layer_type) for _, kind, layer_type, _ in pricer.calls],
                         [(None, LayerType.EMBEDDING_LAYER), (None, LayerType.OUTPUT_LAYER)])

    def test_without_a_pricer_only_the_configured_options_are_described(self):
        """
        Feature: _PPB.lay_ppb.
        Description: Describe a body with no pricer.
        Expectation: The placeholder time stays, and the body offers the plain
            layer, the configured selective recompute and full recompute only.
        """
        desc, _ = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0))
        self.assertEqual(_options(desc), {"NONE": (10, None), "SLCT": (9, None), "FULL": (1, None)})
        self.assertEqual(desc["time"], 1)
        self.assertNotIn("forward_time", desc)


class TestMemorySplit(unittest.TestCase):
    """The balancer charges an option's memory once per micro-batch in flight."""

    def test_buffers_that_grow_with_the_micro_batches_count_per_micro_batch(self):
        """
        Feature: _PPB.lay_ppb.
        Description: A body whose gathered buffer grows with the micro-batches
            in flight, and which only the gather recomputing options free.
        Expectation: The options that keep it are charged it with their
            activations; the layer's constant memory does not include it.
        """
        desc, _ = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0), _Pricer())
        self.assertEqual(desc["memory_parameter"], 4)
        self.assertEqual((desc["memory_activation"], desc["memory_select_comm"]), (8 + 2, 8))

    def test_buffers_that_do_not_grow_are_a_constant_of_the_layer(self):
        """
        Feature: _PPB.lay_ppb.
        Description: A 5 MB parameter buffer, larger than the gathered buffers
            of every micro-batch a stage keeps in flight.
        Expectation: The 5 MB go to the layer's constant memory, and
            recomputing the gathers saves nothing.
        """
        desc, _ = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0), _Pricer(), dp=5)
        self.assertEqual(desc["memory_parameter"], 4 + 5)
        self.assertEqual((desc["memory_activation"], desc["memory_select_comm"]), (8, 8))


class TestProfileAtEveryCount(unittest.TestCase):
    """An option's memory is exact at every count of micro-batches in flight a stage keeps."""

    def test_the_split_states_what_it_charges_beyond_the_buffers_in_between(self):
        """
        Feature: _PPB profiles.
        Description: Stages keep 1 to 4 micro-batches in flight, and a 5 MB
            parameter buffer hides the gathered buffers of the first two.
        Expectation: At 2 and 3, the split charges the plain layer 1 MB beyond
            its buffers; recomputing the gathers, or the whole layer, leaves
            buffers that do not grow, which the split charges exactly.
        """
        profile, _, _ = _measure(5, (1, 2, 3, 4))
        self.assertEqual(profile.counts, (2, 3))
        self.assertEqual(profile.plain.excess, (MEGABYTE, MEGABYTE))
        self.assertEqual(profile.alone["softmax"].excess, (MEGABYTE, MEGABYTE))
        self.assertEqual(profile.alone["gather"].excess, (0, 0))
        self.assertEqual(profile.full.excess, (0, 0))

    def test_every_option_keeps_what_the_memory_model_keeps_at_every_count(self):
        """
        Feature: _PPB profiles, price_option and LayerOption.memory.
        Description: The 128 settings of the switches and full recompute,
            with stages keeping 1 to 4 micro-batches in flight and a 5 MB
            parameter buffer hiding the gathered buffers of the first two.
        Expectation: At every count, each option keeps what the memory model
            keeps for a layer running it.
        """
        profile, memory, ccfg = _measure(5, (1, 2, 3, 4))
        settings = [frozenset(names) for size in range(len(_OPS) + 1) for names in itertools.combinations(_OPS, size)]
        for recompute in settings + [None]:
            option = price_option(profile, recompute)
            if recompute is None:
                memory.ctx.current_node = LayerType.FULL_REC_LAYER
            else:
                memory.ctx.current_node = LayerType.SEL_REC_LAYER if recompute else LayerType.NOT_REC_LAYER
            ccfg.rec_op = SimpleNamespace(**{op: int(op not in (recompute or ())) for op in _OPS})
            for count in (1, 2, 3, 4):
                with self.subTest(recompute=recompute, count=count):
                    self.assertEqual(option.memory(count), sum(memory(default_micro_factor=count)))


class TestWithdrawal(unittest.TestCase):
    """An option no body is better off with is not offered."""

    def test_options_that_tie_with_an_earlier_one_are_withdrawn(self):
        """
        Feature: _PPB.ppb_withdraw_dominated.
        Description: A config whose selective recompute keeps every op: SLCT
            is the plain layer again, and BOTH is COMM.
        Expectation: SLCT and BOTH are withdrawn; COMM stays.
        """
        desc, _ = _describe(LayerType.NOT_REC_LAYER, _KEEP_ALL, _Pricer())
        self.assertEqual(set(_options(_withdraw([desc])[0])), {"NONE", "COMM", "FULL"})

    def test_an_option_that_saves_no_memory_is_withdrawn(self):
        """
        Feature: _PPB.ppb_withdraw_dominated.
        Description: A parameter buffer every option keeps hides the
            gathered buffer, so recomputing the gathers saves nothing.
        Expectation: COMM and BOTH are withdrawn: the plain layer and SLCT
            need as much memory, in less time.
        """
        desc, _ = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0), _Pricer(), dp=5)
        self.assertEqual(set(_options(_withdraw([desc])[0])), {"NONE", "SLCT", "FULL"})

    def test_every_body_keeps_an_option_one_body_is_better_off_with(self):
        """
        Feature: _PPB.ppb_withdraw_dominated.
        Description: Two bodies: recomputing the gathers saves nothing in the
            first and saves memory in the second.
        Expectation: Both bodies offer every option, as the balancer reads
            the options of the first body for all of them.
        """
        first, _ = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0), _Pricer(), dp=5, model="a")
        second, _ = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0), _Pricer(), model="b")
        for desc in _withdraw([first, second]):
            self.assertEqual(set(_options(desc)), set(_OPTIONS), desc["model_name"])

    def test_descriptions_without_times_are_left_alone(self):
        """
        Feature: _PPB.ppb_withdraw_dominated.
        Description: A description built without a pricer.
        Expectation: It keeps every option it has.
        """
        desc, _ = _describe(LayerType.NOT_REC_LAYER, _KEEP_ALL)
        self.assertEqual(set(_options(_withdraw([dict(desc)])[0])), set(_options(desc)))


class TestTimeUnit(unittest.TestCase):
    """The balancer is given times relative to one another."""

    def test_times_are_in_units_of_the_first_body_forward_time(self):
        """
        Feature: _PPB.ppb_scale_times.
        Description: Scale the times of a timed head, body and tail.
        Expectation: The body's forward time is 1, every other time keeps its
            ratio to it, and no memory changes.
        """
        descriptions = [_describe(node, dict(_KEEP_ALL, softmax=0), _Pricer())[0]
                        for node in ("head", LayerType.NOT_REC_LAYER, "tail")]
        memory = [{k: v for k, v in desc.items() if k.startswith("memory")} for desc in descriptions]
        _PPB.ppb_scale_times(descriptions)
        head, body, tail = descriptions
        self.assertEqual((body["forward_time"], body["time"], body["backward_time"], body["recompute_time"]),
                         (1.0, 1.0, 2.0, 3.0))
        self.assertEqual((head["forward_time"], tail["backward_time"]), (0.1, 0.6))
        self.assertEqual([{k: v for k, v in desc.items() if k.startswith("memory")} for desc in descriptions], memory)

    def test_descriptions_without_times_keep_their_placeholder(self):
        """
        Feature: _PPB.ppb_scale_times.
        Description: Scale a description built without a pricer.
        Expectation: Its placeholder time stays 1.
        """
        desc, _ = _describe(LayerType.NOT_REC_LAYER, _KEEP_ALL)
        _PPB.ppb_scale_times([desc])
        self.assertEqual(desc["time"], 1)


class TestCombinedBodies(unittest.TestCase):
    """Combined bodies add up."""

    def test_combined_bodies_add_up_their_memory_and_times(self):
        """
        Feature: _PPB.ppb_combine_bodies.
        Description: Combine the timed bodies of two models.
        Expectation: Every memory and time adds up, and the forward time
            replaces the placeholder.
        """
        first, _ = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0), _Pricer(), model="a")
        second, _ = _describe(LayerType.NOT_REC_LAYER, dict(_KEEP_ALL, softmax=0), _Pricer(), dp=5, model="b")
        descriptions = [dict(first, name="BODY_0", nb_layer=1), dict(second, name="BODY_1", nb_layer=1)]
        _PPB(SimpleNamespace(ppb_combined=[[("a", "body"), ("b", "body")]]), None).ppb_combine_bodies(descriptions)
        self.assertEqual(len(descriptions), 1)
        combined = descriptions[0]
        for key in ("memory_parameter", "forward_time") + tuple(_MEMORY_KEY.values()) + tuple(_TIME_KEY.values()):
            self.assertEqual(combined[key], first[key] + second[key], key)
        self.assertEqual(combined["time"], combined["forward_time"])


class TestBalancerReadsTheKeys(unittest.TestCase):
    """The names ND writes are the names the balancer reads."""

    def test_every_option_is_written_where_the_balancer_reads_it(self):
        """
        Feature: _PPB option keys.
        Description: Compare ND's options and keys with the balancer's.
        Expectation: Same options, in the same order, under the same keys.
        """
        self.assertEqual(_OPTIONS, tuple(rec.name for rec in Recompute.TYPE))
        for rec in Recompute.TYPE:
            self.assertEqual(_MEMORY_KEY[rec.name], Recompute.JSON_MEMORY_NAME[rec], rec)
            self.assertEqual(_TIME_KEY[rec.name], Recompute.JSON_TIME_NAME[rec], rec)

    def test_the_balancer_reads_every_option_back(self):
        """
        Feature: _PPB description.
        Description: Write a timed description and read it with the
            balancer's own reader.
        Expectation: The balancer considers the five options of the body,
            with the memory and backward times ND wrote.
        """
        descriptions = []
        for node in ("head", LayerType.NOT_REC_LAYER, "tail"):
            desc, _ = _describe(node, dict(_KEEP_ALL, softmax=0), _Pricer())
            _PPB.add_to_ppb_list(descriptions, desc)
        _withdraw(descriptions)
        body = next(desc for desc in descriptions if desc["type"] == "BODY")
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "unit.json"), "w", encoding="utf-8") as handle:
                json.dump({"layers_description": descriptions}, handle)
            layers = generate_layers_list(folder, "unit")
        read = next(layer for layer in layers if layer.name_ == body["name"])
        self.assertEqual(read.forward_time_, body["forward_time"])
        for rec in Recompute.TYPE:
            self.assertTrue(read.recompute_considered_[rec], rec)
            self.assertEqual(read.memory_activation_rec_[rec], body[_MEMORY_KEY[rec.name]], rec)
            self.assertEqual(read.backward_time_rec_[rec], body[_TIME_KEY[rec.name]], rec)


if __name__ == "__main__":
    unittest.main()
