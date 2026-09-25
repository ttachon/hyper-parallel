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
fully recomputed, the layer's own otherwise. So that layer's options each keep
the working set they end warm-up on too, and the knapsack weighs it with them.

A runtime that runs every layer one way, such as HyperParallel's trainer with
its ``activation_checkpoint.mode``, gets the fastest of its modes that fits
instead, from the same budgets: see :data:`MODES`.
"""
import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Dict, FrozenSet, Hashable, List, Mapping, Optional, Sequence, Tuple

from hyper_parallel.auto_parallel._op_profiles import LayerKind
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import layer_kinds
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig
from hyper_parallel.auto_parallel.sapp_nd.recompute.front import (
    LayerOption,
    build_front,
    configured_switches,
    layer_profiles,
    price_option,
)
from hyper_parallel.auto_parallel.sapp_nd.recompute.knapsack import MEGABYTE, Layers, PipelineChoice, Stage, pp_lite
from hyper_parallel.auto_parallel.sapp_nd.recompute.profile import SWITCHES

# The pipeline schedules whose micro-batches in flight and end-of-warm-up
# working set the budget follows.
SCHEDULES = ("1f1b",)
# The ways a runtime can run every layer: nothing recomputed, the switches the
# config sets recomputed, or everything recomputed.
MODES = ("off", "selective", "full")
_ENDS = (LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER)
# Marks the front of a layer that ends warm-up, whose options keep its working set.
_ENDS_WARM_UP = "ends warm-up"


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
        mode: The mode every layer runs, one of :data:`MODES`, when one mode
            was chosen for all of them; ``None`` for a choice per layer.
    """

    ranges: Tuple[LayerRange, ...]
    stage_memory: Tuple[float, ...]
    stage_savings: Tuple[float, ...]
    mode: Optional[str] = None

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


def mode_recompute(mode: str, configured: Mapping[str, Any]) -> Optional[FrozenSet[str]]:
    """The switches a mode recomputes; ``None`` for full recompute.

    Args:
        mode: One of :data:`MODES`.
        configured: The switches the config sets, 1 to keep an op and 0 to
            recompute it, which say what ``selective`` recomputes.

    Returns:
        The switches.

    Raises:
        ValueError: For a mode not in :data:`MODES`.
    """
    if mode == "off":
        return frozenset()
    if mode == "full":
        return None
    if mode == "selective":
        return frozenset(name for name in SWITCHES if not int(bool(configured.get(name, 1))))
    raise ValueError(f"unknown recompute mode {mode!r}; expected one of {', '.join(MODES)}")


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


def _plain(options: Sequence[LayerOption]) -> LayerOption:
    """A kind's plain option."""
    return next(option for option in options if option.recompute == frozenset())


def _working(option: LayerOption, options: Sequence[LayerOption]) -> float:
    """The working set a layer running *option* ends warm-up on.

    Its own at one micro-batch, and for a fully recomputed layer the plain
    layer's, *options* being its kind's.
    """
    return _kept(option if option.recompute is not None else _plain(options), 1)


def _charge_working_sets(
    layers: List[List[_Layer]],
    ends: Sequence[Sequence[_Layer]],
    fronts: Dict[Hashable, Tuple[LayerOption, ...]],
    by_mode: Optional[Dict[Hashable, Dict[str, LayerOption]]],
) -> Tuple[Any, ...]:
    """Give each layer that ends warm-up a front whose options keep its working set too.

    Its kind's options, each with the working set it ends warm-up on added to
    what it keeps once, so that the knapsack weighs the working set with the
    option instead of keeping room for the heaviest.

    Returns:
        ``(layers, fronts, by_mode, own)``: the layers, those that end warm-up
        on their new front; the fronts and the options by mode with the new
        fronts added; and each added option's option of its kind.
    """
    fronts = dict(fronts)
    by_mode = None if by_mode is None else dict(by_mode)
    own = {}
    ending = {layer.index for stage_ends in ends for layer in stage_ends}
    for stage_layers in layers:
        for position, layer in enumerate(stage_layers):
            if layer.index not in ending:
                continue
            key = layer.key + (_ENDS_WARM_UP,)
            options = fronts[layer.key]
            charged = {
                option: replace(option, memory_once=option.memory_once + _working(option, options))
                for option in options
            }
            fronts[key] = tuple(charged.values())
            if by_mode is not None:
                by_mode[key] = {mode: charged[option] for mode, option in by_mode[layer.key].items()}
            own.update({new: old for old, new in charged.items()})
            stage_layers[position] = replace(layer, key=key, own=charged[layer.own])
    return layers, fronts, by_mode, own


def _stage(layers: Sequence[_Layer], peak: float, capacity: float) -> Tuple[Stage, float]:
    """A stage for the knapsack, and the bytes it keeps outside the choice.

    Args:
        layers: The stage's body layers, the one that ends warm-up charged
            its working set (:func:`_charge_working_sets`).
        peak: The memory model's peak for the stage, in MB.
        capacity: The device's memory, in MB.

    Returns:
        ``(stage, fixed)``: *fixed* is what the stage keeps whatever its
        layers run.
    """
    fixed = peak * MEGABYTE - sum(_kept(layer.own, layer.in_flight) for layer in layers)
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


def _result(
    layers: Sequence[Sequence[_Layer]],
    fixed: Sequence[float],
    chosen: Dict[int, LayerOption],
    mode: Optional[str],
    own: Mapping[LayerOption, LayerOption],
) -> RecomputeChoice:
    """The choice of *chosen* options, with each stage's memory and time saved.

    *own* maps an option charged a working set to its option of its kind,
    which the ranges report.
    """
    every = [layer for stage_layers in layers for layer in stage_layers]
    return RecomputeChoice(
        ranges=_ranges(every, {index: own.get(option, option) for index, option in chosen.items()}),
        stage_memory=tuple(
            (kept + sum(_kept(chosen[layer.index], layer.in_flight) for layer in stage_layers)) / MEGABYTE
            for kept, stage_layers in zip(fixed, layers)
        ),
        stage_savings=tuple(
            sum(_time(layer.own) - _time(chosen[layer.index]) for layer in stage_layers) for stage_layers in layers
        ),
        mode=mode,
    )


def _one_mode(
    layers: Sequence[Sequence[_Layer]],
    stages: Sequence[Stage],
    by_mode: Dict[Hashable, Dict[str, LayerOption]],
    modes: Sequence[str],
) -> Optional[Tuple[str, Dict[int, LayerOption]]]:
    """The fastest of *modes* that every stage fits when every layer runs it, and each layer's option."""
    fastest, best = math.inf, None
    for mode in modes:
        chosen = {layer.index: by_mode[layer.key][mode] for stage_layers in layers for layer in stage_layers}
        fits = all(
            sum(_kept(chosen[layer.index], layer.in_flight) for layer in stage_layers) <= stage.budget
            for stage_layers, stage in zip(layers, stages)
        )
        time = sum(_time(option) for option in chosen.values())
        if fits and time < fastest:
            fastest, best = time, (mode, chosen)
    return best


def _offered(
    profiles: Dict[Hashable, Any], configured: Mapping[str, Any], modes: Optional[Sequence[str]]
) -> Tuple[Dict[Hashable, Tuple[LayerOption, ...]], Optional[Dict[Hashable, Dict[str, LayerOption]]]]:
    """Each kind's options: its front, or with *modes*, the option of each mode the profiles can price.

    Returns:
        ``(options, by_mode)``: each kind's options, and with *modes* each
        kind's option by mode.
    """
    if modes is None:
        return {key: build_front(profile, configured) for key, profile in profiles.items()}, None
    offered = MODES if "selective" in modes else tuple(mode for mode in MODES if mode != "selective")
    by_mode = {
        key: {mode: price_option(profile, mode_recompute(mode, configured), configured) for mode in offered}
        for key, profile in profiles.items()
    }
    return {key: tuple(options.values()) for key, options in by_mode.items()}, by_mode


def _stages(evaluator: EvaluatorV2, layers: Sequence[Sequence[_Layer]]) -> Tuple[List[Stage], List[float]]:
    """Every stage for the knapsack, and the bytes each keeps outside the choice."""
    capacity = evaluator.ccfg.device_capacity.to_mb().size
    stages, fixed = [], []
    for stage_layers, insight in zip(layers, evaluator.estimate_peak_insight()):
        stage, kept = _stage(stage_layers, insight["Static"] + insight["Dynamic"], capacity)
        stages.append(stage)
        fixed.append(kept)
    return stages, fixed


def choose_recompute(
    evaluator: EvaluatorV2,
    device_type: Any,
    ccfg: Optional[CustomConfig] = None,
    bucket: float = MEGABYTE,
    modes: Optional[Sequence[str]] = None,
) -> Optional[RecomputeChoice]:
    """The fastest recompute option of every layer that fits, at the evaluator's current strategy.

    Args:
        evaluator: The memory evaluator, set to the candidate's strategy.
        device_type: The device the times are priced on.
        ccfg: Estimator options; the search's defaults when omitted.
        bucket: The knapsack's memory granularity, in bytes.
        modes: For a runtime that runs every layer one way, the modes it
            runs, of :data:`MODES`: the fastest that fits is chosen for every
            layer. Without them, each layer gets its own option from its
            kind's front.

    Returns:
        The choice, or ``None`` when there is none to make: a multimodal
        model, a schedule other than those in :data:`SCHEDULES`, a layer
        that runs an option its kind cannot offer, or a stage that does not
        fit even with the lightest option or mode.
    """
    config = evaluator.ccfg
    if config.multimodal or config.pp_sched not in SCHEDULES:
        return None
    counts = micro_batches_in_flight(evaluator)
    profiles = layer_profiles(evaluator, device_type, ccfg, most_in_flight=max(max(row) for row in counts),
                              each_switch=modes is None or "selective" in modes)
    fronts, by_mode = _offered(profiles, configured_switches(evaluator), modes)
    found = _body_layers(evaluator, fronts, counts)
    if found is None:
        return None
    layers, fronts, by_mode, own = _charge_working_sets(*found, fronts, by_mode)
    stages, fixed = _stages(evaluator, layers)
    if by_mode is not None:
        one = _one_mode(layers, stages, by_mode, modes)
        return None if one is None else _result(layers, fixed, one[1], one[0], own)
    choice = pp_lite(stages, fronts, bucket)
    if choice is None:
        return None
    return _result(layers, fixed, _assign([layer for stage in layers for layer in stage], choice), None, own)


def option_label(option: LayerOption) -> str:
    """How an option reads in the search's output."""
    if option.recompute is None:
        return "full recompute"
    if not option.recompute:
        return "no recompute"
    return "recompute " + "+".join(name for name in SWITCHES if name in option.recompute)


def option_record(option: LayerOption) -> Any:
    """An option as plain data: ``"none"``, ``"full"``, or the switches it recomputes, in switch order."""
    if option.recompute is None:
        return "full"
    if not option.recompute:
        return "none"
    return [name for name in SWITCHES if name in option.recompute]


def to_records(choice: RecomputeChoice) -> List[Dict[str, Any]]:
    """The choice's ranges as plain data, for a result file.

    Returns:
        One ``{"first", "count", "kind", "recompute"}`` per range, in model
        order; ``kind`` is the kind's name, or ``None``.
    """
    return [
        {
            "first": item.first,
            "count": item.count,
            "kind": item.kind.name if item.kind is not None else None,
            "recompute": option_record(item.option),
        }
        for item in choice.ranges
    ]


def describe(choice: RecomputeChoice) -> str:
    """The choice as one line per range of layers, after its mode when one was chosen for every layer."""
    lines = [] if choice.mode is None else [f"every layer: {choice.mode}"]
    for item in choice.ranges:
        last = item.first + item.count - 1
        layers = f"layer {item.first}" if item.count == 1 else f"layers {item.first}-{last}"
        kind = f" ({item.kind.name})" if item.kind is not None else ""
        lines.append(f"{layers}{kind}: {option_label(item.option)}")
    return "\n".join(lines)
