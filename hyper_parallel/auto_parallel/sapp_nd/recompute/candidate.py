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
"""Choosing every layer's recompute option for one candidate of the search.

The search keeps the candidates that fit with every layer fully recomputed,
the lightest policy. :func:`choose_recompute` gives a candidate's layers the
fastest options that fit instead. Each pipeline stage's layers share what the
stage has left under the config's own recompute, plus what those layers keep
under it: the memory model's peak for the stage is the sum of what the choice
does not change and of what each layer keeps at the micro-batches the stage
holds of it in flight. :func:`~.knapsack.pp_lite` then picks the options,
stage by stage.

At the end of warm-up, the memory model also charges each stage one layer's
working set at one micro-batch: the plain layer's if the stage's last layer is
fully recomputed, the layer's own otherwise. The budget keeps room for the
heaviest of these, whatever option the last layer runs.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Hashable, List, Optional, Sequence, Tuple

from hyper_parallel.auto_parallel._op_profiles import LayerKind
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import layer_kinds
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig
from hyper_parallel.auto_parallel.sapp_nd.recompute.front import LayerOption, layer_fronts
from hyper_parallel.auto_parallel.sapp_nd.recompute.knapsack import MEGABYTE, Layers, PipelineChoice, Stage, pp_lite
from hyper_parallel.auto_parallel.sapp_nd.recompute.profile import SWITCHES

# The pipeline schedules whose micro-batches in flight and end-of-warm-up
# working set the budget follows.
SCHEDULES = ("1f1b",)
_ENDS = (LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER)


@dataclass(frozen=True)
class LayerRange:
    """Consecutive layers of one kind that run one option.

    Attributes:
        first: The first layer's index, counting the model's layers in order
            without the embedding and output layers.
        count: How many layers.
        kind: Their kind; ``None`` for a model priced on its config as it
            stands.
        option: The option they run.
    """

    first: int
    count: int
    kind: Optional[LayerKind]
    option: LayerOption


@dataclass(frozen=True)
class RecomputeChoice:
    """Every layer's option for one candidate, and what the options change.

    Attributes:
        ranges: The layers' options, in model order.
        stage_memory: Each stage's peak memory under the options, in MB.
        stage_savings: The time each stage's layers save against the
            recompute the config gives them, in the performance estimate's
            units.
    """

    ranges: Tuple[LayerRange, ...]
    stage_memory: Tuple[float, ...]
    stage_savings: Tuple[float, ...]

    @property
    def memory(self) -> float:
        """The peak memory in MB, the heaviest stage's."""
        return max(self.stage_memory)


@dataclass(frozen=True)
class _Layer:
    """A body layer: where it sits, its front's key and how the config runs it."""

    index: int
    key: Hashable
    in_flight: int
    own: LayerOption


def _kept(option: LayerOption, in_flight: int) -> float:
    """The bytes a layer running *option* keeps with *in_flight* micro-batches in flight."""
    return in_flight * option.memory_per_micro_batch + option.memory_once


def _time(option: LayerOption) -> float:
    """A layer's forward and backward time under *option*."""
    return option.forward_time + option.backward_time


def micro_batches_in_flight(evaluator: EvaluatorV2) -> List[List[int]]:
    """How many micro-batches each stage keeps in flight of each chunk's layers.

    Counted as the memory model counts them, with the schedule's own formula
    and at least one.

    Args:
        evaluator: The memory evaluator, set to the candidate's strategy.

    Returns:
        ``[stage][chunk]`` counts.
    """
    ccfg = evaluator.ccfg
    count = evaluator.ctx.pp_micro_eval[ccfg.pp_sched]
    less_memory = evaluator.ctx.vpp_less_mem

    def _at(stage: int, chunk: int) -> int:
        """The count of one stage's chunk, on a context of its own."""
        where = SimpleNamespace(current_stage_id=stage, current_chunk_id=chunk, vpp_less_mem=less_memory)
        return max(1, count(ccfg, where))

    return [[_at(stage, chunk) for chunk in range(ccfg.vp)] for stage in range(ccfg.p)]


def _own(node: LayerType, options: Sequence[LayerOption]) -> Optional[LayerOption]:
    """The option a layer the partition gives *node* runs, if its kind's front has it."""
    for option in options:
        if node == LayerType.FULL_REC_LAYER and option.recompute is None:
            return option
        if node == LayerType.NOT_REC_LAYER and option.recompute == frozenset():
            return option
        if node == LayerType.SEL_REC_LAYER and "SLCT" in option.names:
            return option
    return None


def _overhead_position(chunk: Sequence[LayerType], has_mtp: bool) -> Optional[int]:
    """The position in a stage's last chunk of the layer whose working set warm-up ends on, if a body layer."""
    if not chunk:
        return None
    if chunk[-1] not in _ENDS:
        return len(chunk) - 1
    if chunk[-1] == LayerType.OUTPUT_LAYER and has_mtp and len(chunk) > 1 and chunk[-2] not in _ENDS:
        return len(chunk) - 2
    return None


def _body_layers(
    evaluator: EvaluatorV2, fronts: Dict[Hashable, Tuple[LayerOption, ...]], counts: List[List[int]]
) -> Optional[Tuple[List[List[_Layer]], List[List[_Layer]]]]:
    """Each stage's body layers, and the layers each stage's warm-up ends on.

    The memory model visits chunk after chunk, and within a chunk stage after
    stage, which is the model's order of layers; it gives each body layer the
    next kind in that order.

    Returns:
        ``(layers, ends)`` per stage, or ``None`` when some layer runs an
        option its kind's front does not have.
    """
    ccfg = evaluator.ccfg
    partition = ccfg.generate_partitions_vpp()
    kinds = layer_kinds(ccfg)
    layers = [[] for _ in range(ccfg.p)]
    at = {}
    index = 0
    for chunk in range(ccfg.vp):
        for stage in range(ccfg.p):
            for position, node in enumerate(partition[stage][chunk]):
                if node in _ENDS:
                    continue
                key = (ccfg.model_name, kinds[index])
                own = _own(node, fronts[key])
                if own is None:
                    return None
                layer = _Layer(index, key, counts[stage][chunk], own)
                layers[stage].append(layer)
                at[stage, chunk, position] = layer
                index += 1
    ends = [[] for _ in range(ccfg.p)]
    if not ccfg.freeze:
        last = ccfg.vp - 1
        for stage in range(ccfg.p):
            position = _overhead_position(partition[stage][last], ccfg.n_mtp > 0)
            if position is not None:
                ends[stage].append(at[stage, last, position])
    return layers, ends


def _reserve(options: Sequence[LayerOption]) -> float:
    """The largest working set a layer of a kind can end warm-up on, whatever option it runs.

    A layer ends it on its own working set at one micro-batch, and a fully
    recomputed one on the plain layer's.
    """
    return max(_kept(option, 1) for option in options if option.recompute is not None)


def _plain(options: Sequence[LayerOption]) -> LayerOption:
    """A kind's plain option."""
    return next(option for option in options if option.recompute == frozenset())


def _stage(
    layers: Sequence[_Layer],
    ends: Sequence[_Layer],
    fronts: Dict[Hashable, Tuple[LayerOption, ...]],
    peak: float,
    capacity: float,
) -> Tuple[Stage, float]:
    """A stage for the knapsack, and the bytes it keeps outside the choice.

    Args:
        layers: The stage's body layers.
        ends: The layers its warm-up ends on.
        fronts: Each kind's options.
        peak: The memory model's peak for the stage, in MB.
        capacity: The device's memory, in MB.

    Returns:
        ``(stage, fixed)``: *fixed* is what the stage keeps whatever its
        layers run, the room for the end of warm-up included.
    """
    kept = sum(_kept(layer.own, layer.in_flight) for layer in layers)
    working = 0.0
    reserved = 0.0
    for layer in ends:
        options = fronts[layer.key]
        # A fully recomputed layer ends warm-up on the plain layer's working set.
        run = layer.own if layer.own.recompute is not None else _plain(options)
        working += _kept(run, 1)
        reserved += _reserve(options)
    fixed = peak * MEGABYTE - kept - working + reserved
    groups = {}
    for layer in layers:
        groups[layer.key, layer.in_flight] = groups.get((layer.key, layer.in_flight), 0) + 1
    stage = Stage(
        groups=tuple(Layers(key, count, flight) for (key, flight), count in groups.items()),
        budget=capacity * MEGABYTE - fixed,
    )
    return stage, fixed


def _assign(layers: Sequence[_Layer], choice: PipelineChoice) -> Dict[int, LayerOption]:
    """Each layer's option: a group's layers take its options in model order, the slower first."""
    by_group: Dict[Tuple[Hashable, int], List[int]] = {}
    for layer in layers:
        by_group.setdefault((layer.key, layer.in_flight), []).append(layer.index)
    chosen = {}
    for stage in choice.stages:
        for assignment in stage.assignments:
            waiting = by_group[assignment.layers.kind, assignment.layers.in_flight]
            for _ in range(assignment.count):
                chosen[waiting.pop(0)] = assignment.option
    return chosen


def _ranges(layers: Sequence[_Layer], chosen: Dict[int, LayerOption]) -> Tuple[LayerRange, ...]:
    """The options as ranges of consecutive layers of one kind, in model order."""
    ranges = []
    for layer in sorted(layers, key=lambda layer: layer.index):
        option = chosen[layer.index]
        kind = layer.key[1]
        last = ranges[-1] if ranges else None
        if last is not None and last.option == option and last.kind == kind and last.first + last.count == layer.index:
            ranges[-1] = LayerRange(last.first, last.count + 1, kind, option)
        else:
            ranges.append(LayerRange(layer.index, 1, kind, option))
    return tuple(ranges)


def choose_recompute(
    evaluator: EvaluatorV2,
    device_type: Any,
    ccfg: Optional[CustomConfig] = None,
    bucket: float = MEGABYTE,
) -> Optional[RecomputeChoice]:
    """The fastest recompute option of every layer that fits, at the evaluator's current strategy.

    Args:
        evaluator: The memory evaluator, set to the candidate's strategy.
        device_type: The device the times are priced on.
        ccfg: Estimator options; the search's defaults when omitted.
        bucket: The knapsack's memory granularity, in bytes.

    Returns:
        The choice, or ``None`` when there is none to make: a multimodal
        model, a schedule other than those in :data:`SCHEDULES`, a layer
        that runs an option its kind's front does not have, or a stage that
        does not fit even fully recomputed.
    """
    config = evaluator.ccfg
    if config.multimodal or config.pp_sched not in SCHEDULES:
        return None
    counts = micro_batches_in_flight(evaluator)
    most = max(max(row) for row in counts)
    fronts = {
        (front.model_name, front.kind): front.options
        for front in layer_fronts(evaluator, device_type, ccfg, most_in_flight=most)
    }
    found = _body_layers(evaluator, fronts, counts)
    if found is None:
        return None
    layers, ends = found
    capacity = config.device_capacity.to_mb().size
    stages, fixed = [], []
    for stage_layers, stage_ends, insight in zip(layers, ends, evaluator.estimate_peak_insight()):
        stage, kept = _stage(stage_layers, stage_ends, fronts, insight["Static"] + insight["Dynamic"], capacity)
        stages.append(stage)
        fixed.append(kept)
    choice = pp_lite(stages, fronts, bucket)
    if choice is None:
        return None
    every = [layer for stage_layers in layers for layer in stage_layers]
    chosen = _assign(every, choice)
    return RecomputeChoice(
        ranges=_ranges(every, chosen),
        stage_memory=tuple((kept + made.memory) / MEGABYTE for kept, made in zip(fixed, choice.stages)),
        stage_savings=tuple(
            sum(_time(layer.own) - _time(chosen[layer.index]) for layer in stage_layers) for stage_layers in layers
        ),
    )


def option_label(option: LayerOption) -> str:
    """How an option reads in the search's output."""
    if option.recompute is None:
        return "full recompute"
    if not option.recompute:
        return "no recompute"
    return "recompute " + "+".join(name for name in SWITCHES if name in option.recompute)


def describe(choice: RecomputeChoice) -> str:
    """The choice as one line per range of layers."""
    lines = []
    for item in choice.ranges:
        last = item.first + item.count - 1
        layers = f"layer {item.first}" if item.count == 1 else f"layers {item.first}-{last}"
        kind = f" ({item.kind.name})" if item.kind is not None else ""
        lines.append(f"{layers}{kind}: {option_label(item.option)}")
    return "\n".join(lines)
