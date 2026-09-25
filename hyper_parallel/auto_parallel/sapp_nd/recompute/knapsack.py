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
"""Choosing each layer's option under a memory budget.

Every layer runs one option of its kind's front. At a stage that keeps k
micro-batches of a layer in flight, an option keeps k times its memory per
micro-batch plus its memory held once, less what those charge beyond the
layer at k (:meth:`~.front.LayerOption.memory`), and costs its forward and
backward time. Choosing one option per layer so that a stage's layers are as
fast as possible within the memory left to them is a multiple-choice
knapsack. :func:`choose` solves it exactly over memory counted in buckets,
each option's memory rounded up to a whole bucket, so a choice never keeps
more than the budget.

Without pipeline parallelism the model is one stage, and :func:`choose` is
the whole answer. With it, :func:`pp_lite` solves every stage of a split and
times the pipeline by its slowest stage: a cheap score to rank candidates by
before the pipeline balancer places their layers unevenly.
"""
import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hyper_parallel.auto_parallel.sapp_nd.recompute.front import LayerOption

MEGABYTE = 2 ** 20


@dataclass(frozen=True)
class Layers:
    """Layers of one kind that a stage keeps as many micro-batches of in flight.

    Attributes:
        kind: The key of the kind's front.
        count: How many layers.
        in_flight: How many micro-batches of each the stage keeps at once.
    """

    kind: Hashable
    count: int
    in_flight: int = 1


@dataclass(frozen=True)
class Assignment:
    """How many of a group's layers run one option."""

    layers: Layers
    option: LayerOption
    count: int


@dataclass(frozen=True)
class StageChoice:
    """The options a stage's layers run, and what they cost together.

    Attributes:
        assignments: Per group of layers, how many run each option, the
            slower options first.
        memory: The bytes the layers keep, each option's memory counted
            exactly rather than in buckets.
        time: The layers' forward and backward time.
    """

    assignments: Tuple[Assignment, ...]
    memory: float
    time: float


@dataclass(frozen=True)
class Stage:
    """A pipeline stage: its layers, the memory left to them, and its time outside them.

    Attributes:
        groups: The stage's layers.
        budget: The bytes the layers may keep: the device's memory less
            everything on the stage that the choice does not change.
        fixed_time: The stage's time outside its layers, the embedding's or
            the output layer's.
    """

    groups: Tuple[Layers, ...]
    budget: float
    fixed_time: float = 0.0


@dataclass(frozen=True)
class PipelineChoice:
    """The choice of every stage, and each stage's time."""

    stages: Tuple[StageChoice, ...]
    times: Tuple[float, ...]

    @property
    def slowest(self) -> float:
        """The slowest stage's time, which the pipeline runs at."""
        return max(self.times)


def _kept(option: LayerOption, in_flight: int) -> float:
    """The bytes a layer running *option* keeps with *in_flight* micro-batches in flight."""
    return option.memory(in_flight)


def _time(option: LayerOption) -> float:
    """A layer's forward and backward time under *option*."""
    return option.forward_time + option.backward_time


def _useful(options: Sequence[LayerOption], in_flight: int) -> List[LayerOption]:
    """The options no other beats on memory and time at *in_flight* micro-batches in flight."""
    kept, fastest = [], math.inf
    for option in sorted(options, key=lambda option: (_kept(option, in_flight), _time(option))):
        if _time(option) < fastest:
            kept.append(option)
            fastest = _time(option)
    return kept


def choose(
    groups: Sequence[Layers],
    fronts: Mapping[Hashable, Sequence[LayerOption]],
    budget: float,
    bucket: float = MEGABYTE,
) -> Optional[StageChoice]:
    """The fastest options for a stage's layers that keep at most *budget* bytes.

    Of the fastest choices, the one that keeps the least memory.

    Args:
        groups: The stage's layers, by kind and micro-batches in flight.
        fronts: Each kind's options.
        budget: The bytes the layers may keep.
        bucket: The memory granularity, in bytes. Each option's memory is
            rounded up to a whole bucket, so a coarser bucket solves faster
            and may give up a little of the budget.

    Returns:
        The choice, or ``None`` when not even the lightest options fit.
    """
    if budget < 0:
        return None
    useful = [_useful(fronts[group.kind], group.in_flight) for group in groups]
    if sum(group.count * _kept(options[-1], group.in_flight) for group, options in zip(groups, useful)) <= budget:
        # Every layer's fastest option fits: nothing to choose.
        return _stage_choice(
            [Assignment(group, options[-1], group.count) for group, options in zip(groups, useful) if group.count]
        )
    size = int(budget // bucket)
    # fastest[w]: the least time of the layers seen so far keeping exactly w buckets.
    fastest = np.full(size + 1, np.inf)
    fastest[0] = 0.0
    steps = []
    for position, (group, options) in enumerate(zip(groups, useful)):
        weights = [math.ceil(_kept(option, group.in_flight) / bucket) for option in options]
        for _ in range(group.count):
            after = np.full(size + 1, np.inf)
            picked = np.full(size + 1, -1, dtype=np.int16)
            for index, (option, weight) in enumerate(zip(options, weights)):
                if weight > size:
                    continue
                candidate = fastest[:size + 1 - weight] + _time(option)
                better = candidate < after[weight:]
                after[weight:][better] = candidate[better]
                picked[weight:][better] = index
            fastest = after
            steps.append((position, weights, picked))
    if not np.isfinite(fastest).any():
        return None
    used = int(np.argmin(fastest))
    counts = Counter()
    for position, weights, picked in reversed(steps):
        index = int(picked[used])
        counts[position, index] += 1
        used -= weights[index]
    assignments = []
    for position, (group, options) in enumerate(zip(groups, useful)):
        chosen = [(options[index], count) for (found, index), count in counts.items() if found == position]
        for option, count in sorted(chosen, key=lambda pair: -_time(pair[0])):
            assignments.append(Assignment(group, option, count))
    return _stage_choice(assignments)


def _stage_choice(assignments: Sequence[Assignment]) -> StageChoice:
    """A stage's assignments, with the memory they keep and the time they take."""
    return StageChoice(
        assignments=tuple(assignments),
        memory=sum(item.count * _kept(item.option, item.layers.in_flight) for item in assignments),
        time=sum(item.count * _time(item.option) for item in assignments),
    )


def pp_lite(
    stages: Sequence[Stage], fronts: Mapping[Hashable, Sequence[LayerOption]], bucket: float = MEGABYTE
) -> Optional[PipelineChoice]:
    """The fastest choice for every stage, each within its own budget.

    Stages share no memory, so choosing each stage's fastest options also
    makes the slowest stage as fast as it can be.

    Args:
        stages: The pipeline's stages.
        fronts: Each kind's options.
        bucket: The memory granularity, as for :func:`choose`.

    Returns:
        The choice, or ``None`` when some stage cannot fit.
    """
    choices = []
    for stage in stages:
        choice = choose(stage.groups, fronts, stage.budget, bucket)
        if choice is None:
            return None
        choices.append(choice)
    return PipelineChoice(
        stages=tuple(choices),
        times=tuple(choice.time + stage.fixed_time for choice, stage in zip(choices, stages)),
    )
