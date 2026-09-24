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
"""Tests for choosing each layer's option under a memory budget.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/recompute/test_knapsack.py -v
"""
import itertools
import math
import os
import random
import unittest
from typing import Dict, Hashable, Optional, Sequence, Tuple

# The package has an import cycle that only the memory estimator's import order
# settles; the performance modules cannot be the first a process loads.
import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2  # pylint: disable=unused-import
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.recompute.front import LayerOption, layer_fronts
from hyper_parallel.auto_parallel.sapp_nd.recompute.knapsack import (
    MEGABYTE,
    Layers,
    Stage,
    choose,
    pp_lite,
)

DEEPSEEK_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nd", "deepseek.yaml"
)


def _option(per_micro_batch: float, once: float, backward: float) -> LayerOption:
    """An option keeping *per_micro_batch* and *once* MB, with a forward time of 10."""
    return LayerOption(recompute=frozenset(), memory_per_micro_batch=per_micro_batch * MEGABYTE,
                       memory_once=once * MEGABYTE, forward_time=10.0, backward_time=backward)


# A layer kind: plain, two selective options and full recompute.
_FRONT = (_option(100, 10, 20), _option(70, 10, 20.5), _option(40, 20, 23), _option(5, 20, 30))


def _kept(option: LayerOption, in_flight: int) -> float:
    """The bytes a layer keeps under *option*."""
    return in_flight * option.memory_per_micro_batch + option.memory_once


def _brute_force(groups: Sequence[Layers], fronts: Dict[Hashable, Sequence[LayerOption]], budget: float,
                 bucket: float) -> Tuple[Optional[float], Optional[float]]:
    """The least time of any choice that fits, and of any whose bucketed memory fits, trying every one."""
    layers = [(group, fronts[group.kind]) for group in groups for _ in range(group.count)]
    exact, bucketed = None, None
    for picks in itertools.product(*(range(len(options)) for _, options in layers)):
        chosen = [(options[pick], group.in_flight) for (group, options), pick in zip(layers, picks)]
        time = sum(option.forward_time + option.backward_time for option, _ in chosen)
        if sum(_kept(option, flight) for option, flight in chosen) <= budget:
            exact = time if exact is None else min(exact, time)
        if sum(math.ceil(_kept(option, flight) / bucket) for option, flight in chosen) * bucket <= budget:
            bucketed = time if bucketed is None else min(bucketed, time)
    return exact, bucketed


class TestChoose(unittest.TestCase):
    """The fastest options that fit a stage's budget."""

    def test_the_choice_is_as_fast_as_any_that_fits(self):
        """
        Feature: choose.
        Description: Random stages of two kinds of layers, some kept longer
            in flight, under random budgets.
        Expectation: At least as fast as the best choice whose memory fits
            in whole buckets, no faster than the best that fits, and never
            keeping more than the budget.
        """
        rng = random.Random(7)
        for _ in range(40):
            fronts = {kind: [_option(rng.uniform(0, 100), rng.uniform(0, 20), rng.uniform(20, 40))
                             for _ in range(rng.randint(1, 4))] for kind in ("a", "b")}
            groups = [Layers("a", rng.randint(1, 3), rng.randint(1, 3)), Layers("b", rng.randint(0, 2))]
            budget = rng.uniform(0, 600) * MEGABYTE
            bucket = rng.choice([MEGABYTE, 7 * MEGABYTE])
            exact, bucketed = _brute_force(groups, fronts, budget, bucket)
            choice = choose(groups, fronts, budget, bucket)
            if exact is None:
                self.assertIsNone(choice)
                continue
            if bucketed is not None:
                self.assertIsNotNone(choice)
                self.assertLessEqual(choice.time, bucketed + 1e-9)
            if choice is None:
                continue
            self.assertGreaterEqual(choice.time, exact - 1e-9)
            self.assertLessEqual(choice.memory, budget)
            for group in groups:
                self.assertEqual(sum(item.count for item in choice.assignments if item.layers is group),
                                 group.count)

    def test_a_budget_the_lightest_options_exceed_has_no_choice(self):
        """
        Feature: choose.
        Description: Four layers fully recomputed keep 100 MB; the budget is 99 MB.
        Expectation: No choice.
        """
        self.assertIsNone(choose([Layers("a", 4)], {"a": _FRONT}, 99 * MEGABYTE))

    def test_a_roomy_budget_keeps_every_layer_plain(self):
        """
        Feature: choose.
        Description: A budget every plain layer fits in.
        Expectation: Every layer runs plain, the fastest option.
        """
        choice = choose([Layers("a", 4)], {"a": _FRONT}, 10 ** 12)
        self.assertEqual([(item.option, item.count) for item in choice.assignments], [(_FRONT[0], 4)])

    def test_more_micro_batches_in_flight_need_more_recompute(self):
        """
        Feature: choose.
        Description: The same layers and budget, with one micro-batch in
            flight and with four.
        Expectation: Four in flight recompute more, so run slower, and still fit.
        """
        budget = 800 * MEGABYTE
        one = choose([Layers("a", 4, 1)], {"a": _FRONT}, budget)
        four = choose([Layers("a", 4, 4)], {"a": _FRONT}, budget)
        self.assertLess(one.time, four.time)
        self.assertLessEqual(four.memory, budget)

    def test_a_budget_the_fastest_options_fit_needs_no_search(self):
        """
        Feature: choose.
        Description: A budget far beyond what any bucket count could index.
        Expectation: Every layer runs its fastest option, found without
            searching over the budget.
        """
        choice = choose([Layers("a", 4, 3)], {"a": _FRONT}, 1e30)
        self.assertEqual([(item.option, item.count) for item in choice.assignments], [(_FRONT[0], 4)])
        self.assertEqual(choice.memory, 4 * _kept(_FRONT[0], 3))

    def test_of_the_fastest_choices_the_lightest_is_given(self):
        """
        Feature: choose.
        Description: Two options that cost the same time, one keeping less.
        Expectation: The lighter one.
        """
        same = (_option(100, 0, 20), _option(50, 0, 20))
        choice = choose([Layers("a", 2)], {"a": same}, 10 ** 12)
        self.assertEqual(choice.memory, 100 * MEGABYTE)

    def test_groups_of_the_same_kind_are_kept_apart(self):
        """
        Feature: choose.
        Description: Two equal groups of layers of one kind, as two chunks of
            a stage might be.
        Expectation: Each group's layers are all assigned, in its own group.
        """
        first, second = Layers("a", 2), Layers("a", 2)
        choice = choose([first, second], {"a": _FRONT}, 300 * MEGABYTE)
        for group in (first, second):
            self.assertEqual(sum(item.count for item in choice.assignments if item.layers is group), 2)

    def test_the_slower_options_come_first(self):
        """
        Feature: choose.
        Description: A budget that needs a mix of options.
        Expectation: A group's assignments list the slower options first,
            as the first layers of a stage are the recomputed ones.
        """
        choice = choose([Layers("a", 4)], {"a": _FRONT}, 200 * MEGABYTE)
        times = [item.option.backward_time for item in choice.assignments]
        self.assertEqual(times, [30.0, 23.0, 20.5])
        self.assertEqual(times, sorted(times, reverse=True))


class TestPpLite(unittest.TestCase):
    """Each stage chosen within its own budget; the pipeline runs at its slowest."""

    def test_each_stage_is_chosen_within_its_own_budget(self):
        """
        Feature: pp_lite.
        Description: Two stages of four layers: the first keeps two
            micro-batches in flight and also runs the embedding.
        Expectation: Each stage's choice is the one choose makes for it, and
            the pipeline's time is the slower stage's.
        """
        fronts = {"a": _FRONT}
        stages = [Stage((Layers("a", 4, 2),), 500 * MEGABYTE, fixed_time=5.0),
                  Stage((Layers("a", 4, 1),), 500 * MEGABYTE)]
        pipeline = pp_lite(stages, fronts)
        for stage, choice, time in zip(stages, pipeline.stages, pipeline.times):
            self.assertEqual(choice, choose(stage.groups, fronts, stage.budget))
            self.assertEqual(time, choice.time + stage.fixed_time)
        self.assertEqual(pipeline.slowest, max(pipeline.times))
        self.assertGreater(pipeline.times[0], pipeline.times[1])

    def test_a_stage_that_cannot_fit_leaves_no_choice(self):
        """
        Feature: pp_lite.
        Description: A second stage whose budget the lightest options exceed.
        Expectation: No choice.
        """
        stages = [Stage((Layers("a", 4),), 10 ** 12), Stage((Layers("a", 4),), 99 * MEGABYTE)]
        self.assertIsNone(pp_lite(stages, {"a": _FRONT}))


class TestOnAFront(unittest.TestCase):
    """DeepSeek's fronts: a stage of dense and MoE layers under a tightening budget."""

    def test_a_tighter_budget_never_runs_faster(self):
        """
        Feature: choose.
        Description: Three dense and five MoE layers, two micro-batches in
            flight, from all plain down to all fully recomputed.
        Expectation: Every choice fits, and the time never falls as the
            budget tightens.
        """
        fronts = {(front.model_name, front.kind): front.options
                  for front in layer_fronts(EvaluatorV2(DEEPSEEK_YAML, framework="mindformers", log_level=0),
                                            Hard.Device_A2)}
        dense, moe = list(fronts)
        groups = [Layers(dense, 3, 2), Layers(moe, 5, 2)]
        plain = sum(group.count * _kept(fronts[group.kind][0], 2) for group in groups)
        full = sum(group.count * min(_kept(option, 2) for option in fronts[group.kind]) for group in groups)
        times = []
        for share in (1.0, 0.9, 0.7, 0.5, 0.3, 0.0):
            budget = full + share * (plain - full) + MEGABYTE * 8
            choice = choose(groups, fronts, budget)
            self.assertLessEqual(choice.memory, budget)
            times.append(choice.time)
        self.assertEqual(times, sorted(times))
        self.assertLess(times[0], times[-1])


if __name__ == "__main__":
    unittest.main()
