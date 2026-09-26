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
fully recomputed, the layer's own otherwise, what it keeps then and, under FSDP
that reshards, the gathered parameters its backward holds besides. So that
layer's options each keep the working set they end warm-up on too, and the
knapsack weighs it with them.

A runtime that runs every layer one way, such as HyperParallel's trainer with
its ``activation_checkpoint.mode``, gets the fastest of its modes that fits
instead, from the same budgets: see :data:`MODES`.

With a host link, a choice per layer may also offload: a stage's first layers
run plain and move what they keep per micro-batch to the host after their
forward, and back before their backward, so that they keep only what they
hold once. A micro-batch's copies to the host keep pace with its own forward:
each offloaded layer's, with those of the offloaded layers after it, fit in
the forward left after it, so that a stage's last layers cannot offload when
their window is too short. The copies back run in the longer backward, one
layer ahead, on the same stream, which then has room for them; the stage
keeps one offloaded layer's worth in transit. Offload is priced at one chunk
per stage.

Where a stage's FSDP holds each layer's gradient output until the backward
ends, and accumulates gradients over several micro-batches, its peak can also
come as a later micro-batch's backward ends: the memory model then counts
what the micro-batches still in flight keep, beside every layer's gradient
output and the working set of the first layer's backward, the last the stage
runs. The stage's layers then fit at both points. Each pair of options of the
first layer and of the layer that ends warm-up leaves the others one budget,
since they keep a share of what they keep as warm-up ends: exact at one chunk
per stage, and with two, the largest share of the chunks'.
"""
import itertools
import math
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Dict, FrozenSet, Hashable, List, Mapping, Optional, Sequence, Tuple

from hyper_parallel.auto_parallel._op_profiles import LayerKind
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware import HostLink
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
from hyper_parallel.auto_parallel.sapp_nd.recompute.knapsack import (
    MEGABYTE,
    Layers,
    Stage,
    StageChoice,
    choose,
    least_times,
    suffix_times,
)
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
# A time shorter than another by this share of it or less is the same time: the
# same options' times summed in another order can differ in their last bits.
_ROUNDING = 1e-9


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


@dataclass(frozen=True)
class _Peaks:
    """What a stage keeps whatever its layers run, at each point its peak can come, in bytes.

    Attributes:
        warm_up: As warm-up ends.
        backward: As a later micro-batch's backward ends, for a stage whose
            FSDP holds its gradient outputs until then; ``None`` otherwise.
    """

    warm_up: float
    backward: Optional[float] = None


def _kept(option: LayerOption, in_flight: int) -> float:
    """The bytes a layer running *option* keeps with *in_flight* micro-batches in flight."""
    return option.memory(in_flight)


def _share(in_flight: int) -> float:
    """The share of what a layer keeps that stays as one of its *in_flight* micro-batches' backward ends."""
    return (in_flight - 1) / in_flight


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


def _working(option: LayerOption, options: Sequence[LayerOption], first: bool = False) -> float:
    """The working set a layer running *option* ends warm-up on, or with *first*, the backward on a stage's first layer.

    What it keeps at one micro-batch, the plain layer for full recompute,
    *options* being its kind's, and what its backward holds beyond.
    """
    run = option if option.recompute is not None else _plain(options)
    return _kept(run, 1) + (option.first_working_extra if first else option.working_extra)


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


def _kind_key(key: Hashable) -> Hashable:
    """The key of a layer's kind's own front, for a layer on a front charged its working set."""
    return key[:-1] if isinstance(key, tuple) and key and key[-1] == _ENDS_WARM_UP else key


def _first_working(
    layer: _Layer,
    option: LayerOption,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> float:
    """The working set of a stage's first layer's backward, the last the stage runs, under *option*.

    As for the layer that ends warm-up (:func:`_working`); for an offloaded
    layer, the plain layer's, whose activations come back for it.
    """
    options = fronts[_kind_key(layer.key)]
    option = own.get(option, option)
    return _working(_plain(options) if option.link_bandwidth else option, options, first=True)


def _backward_kept(
    stage_layers: Sequence[_Layer],
    chosen: Mapping[int, LayerOption],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> float:
    """What a stage's layers keep as a later micro-batch's backward ends, running *chosen*.

    Each keeps what it keeps of the micro-batches still in flight, a share of
    what it keeps as warm-up ends, and the first layer the working set of its
    backward, the last the stage runs.
    """
    if not stage_layers:
        return 0.0
    kept = sum(
        _kept(own.get(chosen[layer.index], chosen[layer.index]), layer.in_flight) * _share(layer.in_flight)
        for layer in stage_layers
    )
    first = stage_layers[0]
    return kept + _first_working(first, chosen[first.index], fronts, own)


def _stage_memory(
    stage_layers: Sequence[_Layer],
    peaks: _Peaks,
    chosen: Mapping[int, LayerOption],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> float:
    """A stage's peak in bytes, its layers running *chosen*: the higher of the points it can come at."""
    memory = peaks.warm_up + sum(_kept(chosen[layer.index], layer.in_flight) for layer in stage_layers)
    if peaks.backward is None:
        return memory
    return max(memory, peaks.backward + _backward_kept(stage_layers, chosen, fronts, own))


def _groups(layers: Sequence[_Layer]) -> Tuple[Layers, ...]:
    """Layers grouped by kind and micro-batches in flight, for the knapsack."""
    groups: Dict[Tuple[Hashable, int], int] = {}
    for layer in layers:
        groups[layer.key, layer.in_flight] = groups.get((layer.key, layer.in_flight), 0) + 1
    return tuple(Layers(key, count, flight) for (key, flight), count in groups.items())


def _stage(layers: Sequence[_Layer], peak: float, capacity: float) -> Tuple[Stage, float]:
    """A stage for the knapsack, and the bytes it keeps outside the choice as warm-up ends.

    Args:
        layers: The stage's body layers, the one that ends warm-up charged
            its working set (:func:`_charge_working_sets`).
        peak: The memory model's stage memory as warm-up ends, in MB.
        capacity: The device's memory, in MB.

    Returns:
        ``(stage, fixed)``: *fixed* is what the stage keeps whatever its
        layers run.
    """
    fixed = peak * MEGABYTE - sum(_kept(layer.own, layer.in_flight) for layer in layers)
    return Stage(groups=_groups(layers), budget=capacity * MEGABYTE - fixed), fixed


def _assign(layers: Sequence[_Layer], choices: Sequence[StageChoice]) -> Dict[int, LayerOption]:
    """Each layer's option: a group's layers take its options in model order, the slower first."""
    by_group: Dict[Tuple[Hashable, int], List[int]] = {}
    for layer in layers:
        by_group.setdefault((layer.key, layer.in_flight), []).append(layer.index)
    chosen = {}
    for choice in choices:
        for assignment in choice.assignments:
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
    peaks: Sequence[_Peaks],
    chosen: Dict[int, LayerOption],
    mode: Optional[str],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
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
            _stage_memory(stage_layers, stage_peaks, chosen, fronts, own) / MEGABYTE
            for stage_layers, stage_peaks in zip(layers, peaks)
        ),
        stage_savings=tuple(
            sum(_time(layer.own) - _time(chosen[layer.index]) for layer in stage_layers) for stage_layers in layers
        ),
        mode=mode,
    )


def _one_mode(
    layers: Sequence[Sequence[_Layer]],
    peaks: Sequence[_Peaks],
    by_mode: Dict[Hashable, Dict[str, LayerOption]],
    modes: Sequence[str],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    capacity: float,
) -> Optional[Tuple[str, Dict[int, LayerOption]]]:
    """The fastest of *modes* that every stage fits when every layer runs it, and each layer's option."""
    fastest, best = math.inf, None
    for mode in modes:
        chosen = {layer.index: by_mode[layer.key][mode] for stage_layers in layers for layer in stage_layers}
        fits = all(
            _stage_memory(stage_layers, stage_peaks, chosen, fronts, own) <= capacity
            for stage_layers, stage_peaks in zip(layers, peaks)
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


@dataclass(frozen=True)
class _Link:
    """A host link in the performance estimate's time units.

    Attributes:
        per_byte: The time one byte takes over the link, one way.
        overlap: The share of the compute time the copies may take.
    """

    per_byte: float
    overlap: float


def _link(link: HostLink, evaluator: EvaluatorV2) -> _Link:
    """*link* in the estimate's units: FLOPs of forward and backward, times the precision's bytes, per device."""
    units_per_second = link.flops_per_second() * evaluator.ccfg.bytes_p
    return _Link(link.seconds_per_byte() * units_per_second, link.overlap)


def _offloaded(option: LayerOption) -> LayerOption:
    """*option* with what it keeps per micro-batch moved to the host, so that it keeps only what it holds once."""
    return replace(
        option, memory_per_micro_batch=0.0, excess=(), link_bandwidth=option.memory_per_micro_batch, names=()
    )


def _offload_limit(stage_layers: Sequence[_Layer], plain: Sequence[LayerOption], link: _Link) -> int:
    """How many of a stage's first layers can offload running plain, the copies keeping pace with the forward.

    Each offloaded layer's copy to the host, with those of the offloaded
    layers after it, fits in the forward time left after it, of the link's
    share of it. The layer that ends warm-up, and every layer after it, keep
    their activations.
    """
    ending = next((position for position, layer in enumerate(stage_layers) if _ENDS_WARM_UP in layer.key),
                  len(stage_layers))
    count, slack, left = 0, math.inf, sum(option.forward_time for option in plain)
    for position in range(ending):
        copy = plain[position].memory_per_micro_batch * link.per_byte
        left -= plain[position].forward_time
        # Every earlier offloaded layer's copies now queue this one too.
        slack = min(slack, link.overlap * left) - copy
        if slack < 0:
            break
        count = position + 1
    return count


def _backward_can_bind(
    stage_layers: Sequence[_Layer],
    room: float,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> bool:
    """Whether some choice of a stage's layers keeps more than *room* bytes as a later micro-batch's backward ends."""
    first = stage_layers[0]
    most = max(
        _kept(own.get(option, option), first.in_flight) * _share(first.in_flight)
        + _first_working(first, option, fronts, own)
        for option in fronts[first.key]
    )
    most += sum(
        max(_kept(own.get(option, option), layer.in_flight) for option in fronts[layer.key]) * _share(layer.in_flight)
        for layer in stage_layers[1:]
    )
    return most > room


def _ends(stage_layers: Sequence[_Layer]) -> List[_Layer]:
    """A stage's first layer and the layer that ends its warm-up, once each."""
    first = stage_layers[0]
    ending = next((layer for layer in stage_layers if _ENDS_WARM_UP in layer.key), first)
    return [first] if ending.index == first.index else [first, ending]


def _middle_budget(
    ends: Sequence[_Layer],
    chosen: Mapping[int, LayerOption],
    budgets: Tuple[float, float],
    share: float,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> float:
    """What a stage's other layers may keep as warm-up ends, its ends running *chosen*; negative if none fits.

    *budgets* are the bytes all its layers may keep as warm-up ends and as a
    later micro-batch's backward ends, and *share* the share the other layers
    keep of the one at the other.
    """
    left = budgets[0] - sum(_kept(chosen[end.index], end.in_flight) for end in ends)
    left_back = budgets[1] - _backward_kept(ends, chosen, fronts, own)
    if left < 0 or left_back < 0:
        return -1.0
    return min(left, left_back / share) if share else left


def _two_peaks(
    stage_layers: Sequence[_Layer],
    budget: float,
    room: float,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    bucket: float,
) -> Optional[Dict[int, LayerOption]]:
    """The fastest options of a stage's layers within *budget* bytes as warm-up ends and *room* as a backward ends.

    What the layers keep as a later micro-batch's backward ends is a share of
    what they keep as warm-up ends, but for the first layer, which adds its
    working set, and the layer that ends warm-up, which leaves its own out.
    So each pair of options of those two leaves the other layers one budget:
    exact when they keep as many micro-batches in flight, at one chunk per
    stage, and with the largest share of theirs otherwise, which fits.
    """
    ends = _ends(stage_layers)
    middle = [layer for layer in stage_layers if layer not in ends]
    share = max((_share(layer.in_flight) for layer in middle), default=0.0)
    groups = _groups(middle)
    within = least_times(groups, fronts, budget, bucket)
    fastest, best, best_budget = math.inf, None, 0.0
    for picks in itertools.product(*(fronts[end.key] for end in ends)):
        chosen = {end.index: option for end, option in zip(ends, picks)}
        limit = _middle_budget(ends, chosen, (budget, room), share, fronts, own)
        time = within(limit) + sum(_time(option) for option in picks)
        if time < fastest:
            fastest, best, best_budget = time, chosen, limit
    if best is None:
        return None
    choice = choose(groups, fronts, best_budget, bucket)
    if choice is None:
        return None
    best.update(_assign(middle, [choice]))
    return best


def _choose_stage(
    stage_layers: Sequence[_Layer],
    stage: Stage,
    peaks: _Peaks,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    bucket: float,
    capacity: float,
) -> Optional[Dict[int, LayerOption]]:
    """The fastest options of a stage's layers that keep it within *capacity* bytes wherever its peak comes."""
    if peaks.backward is not None and stage_layers:
        room = capacity - peaks.backward
        if _backward_can_bind(stage_layers, room, fronts, own):
            return _two_peaks(stage_layers, stage.budget, room, fronts, own, bucket)
    choice = choose(stage.groups, fronts, stage.budget, bucket)
    return None if choice is None else _assign(stage_layers, [choice])


def _lightest_working(stage_layers: Sequence[_Layer], fronts: Mapping[Hashable, Tuple[LayerOption, ...]]) -> float:
    """The least working set the layer that ends a stage's warm-up can end it on; 0 without one."""
    ending = next((layer for layer in stage_layers if _ENDS_WARM_UP in layer.key), None)
    if ending is None:
        return 0.0
    options = fronts[_kind_key(ending.key)]
    return min(_working(option, options) for option in options)


def _offload_budgets(
    stage_layers: Sequence[_Layer],
    stage: Stage,
    peaks: _Peaks,
    plain: Sequence[LayerOption],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    capacity: float,
) -> List[float]:
    """For every count of a stage's first layers offloaded, what the other layers may keep as warm-up ends.

    Less what the offloaded layers keep once and in transit. Where the
    stage's peak can also come as a later micro-batch's backward ends, the
    other layers keep room for it as though the layer that ends warm-up ran
    its lightest option there, which fits.
    """
    room = math.inf if peaks.backward is None else capacity - peaks.backward
    share = _share(stage_layers[0].in_flight)
    lightest = _lightest_working(stage_layers, fronts)
    first_working = _first_working(stage_layers[0], _offloaded(plain[0]), fronts, own)
    budgets, held, largest = [stage.budget], 0.0, 0.0
    for position, layer in enumerate(stage_layers):
        held += _kept(_offloaded(plain[position]), layer.in_flight)
        largest = max(largest, plain[position].memory_per_micro_batch)
        left_back = room - held * share - first_working - largest
        rest_back = left_back / share + lightest if share else (math.inf if left_back >= 0 else -1.0)
        budgets.append(min(stage.budget - held - largest, rest_back))
    return budgets


def _offload_stage(
    stage_layers: Sequence[_Layer],
    stage: Stage,
    peaks: _Peaks,
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    link: _Link,
    bucket: float,
    capacity: float,
) -> Optional[Tuple[Dict[int, LayerOption], float]]:
    """The fastest choice of a stage's layers when its first layers may offload.

    Every count of first layers the link allows is weighed against keeping
    them, the rest of the layers choosing their options in what the
    offloaded ones leave (:func:`_offload_budgets`).

    Returns:
        ``(chosen, transit)``: each layer's option by index, and the memory
        the offloaded layers keep in transit; ``None`` when the stage cannot
        fit.
    """
    stay = _choose_stage(stage_layers, stage, peaks, fronts, own, bucket, capacity)
    kept = None if stay is None else (stay, 0.0)
    plain = [_plain(fronts[layer.key]) for layer in stage_layers]
    most = _offload_limit(stage_layers, plain, link)
    if not most:
        return kept
    budgets = _offload_budgets(stage_layers, stage, peaks, plain, fronts, own, capacity)
    rest = suffix_times([Layers(layer.key, 1, layer.in_flight) for layer in stage_layers], fronts, budgets, bucket)
    offloaded = list(itertools.accumulate((_time(option) for option in plain), initial=0.0))
    # Offload only where it saves time, more than rounding.
    count, fastest = 0, math.inf if stay is None else sum(_time(option) for option in stay.values()) * (1 - _ROUNDING)
    for first in range(1, most + 1):
        if offloaded[first] + rest[first] < fastest:
            count, fastest = first, offloaded[first] + rest[first]
    if not count:
        return kept
    choice = choose(_groups(stage_layers[count:]), fronts, budgets[count], bucket)
    if choice is None:
        return None
    chosen = {layer.index: _offloaded(plain[position]) for position, layer in enumerate(stage_layers[:count])}
    chosen.update(_assign(stage_layers[count:], [choice]))
    return chosen, max(option.memory_per_micro_batch for option in plain[:count])


def _stages(
    evaluator: EvaluatorV2,
    layers: Sequence[Sequence[_Layer]],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
) -> Tuple[List[Stage], List[_Peaks]]:
    """Every stage for the knapsack, and what each keeps whatever its layers run, at each point its peak can come."""
    capacity = evaluator.ccfg.device_capacity.to_mb().size
    insights = evaluator.estimate_peak_insight()
    stages, peaks = [], []
    for stage_layers, insight, (warm_up, backward) in zip(layers, insights, evaluator.peak_points):
        stage, fixed = _stage(stage_layers, insight["Static"] + warm_up, capacity)
        stages.append(stage)
        second = None
        if backward:
            configured = {layer.index: layer.own for layer in stage_layers}
            second = (insight["Static"] + backward) * MEGABYTE - _backward_kept(stage_layers, configured, fronts, own)
        peaks.append(_Peaks(fixed, second))
    return stages, peaks


def choose_recompute(
    evaluator: EvaluatorV2,
    device_type: Any,
    ccfg: Optional[CustomConfig] = None,
    bucket: float = MEGABYTE,
    modes: Optional[Sequence[str]] = None,
    link: Optional[HostLink] = None,
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
        link: For a choice per layer at one chunk per stage, the host link
            each stage's first layers may offload over; ``None`` keeps every
            layer's activations on the device.

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
                              each_switch=modes is None or "selective" in modes,
                              in_flight=[count for row in counts for count in row])
    fronts, by_mode = _offered(profiles, configured_switches(evaluator), modes)
    found = _body_layers(evaluator, fronts, counts)
    if found is None:
        return None
    layers, fronts, by_mode, own = _charge_working_sets(*found, fronts, by_mode)
    stages, peaks = _stages(evaluator, layers, fronts, own)
    capacity = config.device_capacity.to_mb().size * MEGABYTE
    if by_mode is not None:
        one = _one_mode(layers, peaks, by_mode, modes, fronts, own, capacity)
        return None if one is None else _result(layers, peaks, one[1], one[0], fronts, own)
    if link is not None and config.vp == 1:
        return _offload_result(layers, stages, peaks, fronts, own, _link(link, evaluator), bucket, capacity)
    chosen: Dict[int, LayerOption] = {}
    for stage_layers, stage, stage_peaks in zip(layers, stages, peaks):
        stage_chosen = _choose_stage(stage_layers, stage, stage_peaks, fronts, own, bucket, capacity)
        if stage_chosen is None:
            return None
        chosen.update(stage_chosen)
    return _result(layers, peaks, chosen, None, fronts, own)


def _offload_result(
    layers: Sequence[Sequence[_Layer]],
    stages: Sequence[Stage],
    peaks: Sequence[_Peaks],
    fronts: Mapping[Hashable, Tuple[LayerOption, ...]],
    own: Mapping[LayerOption, LayerOption],
    link: _Link,
    bucket: float,
    capacity: float,
) -> Optional[RecomputeChoice]:
    """The choice per layer, each stage's first layers offloading where that is faster."""
    chosen: Dict[int, LayerOption] = {}
    held = []
    for stage_layers, stage, stage_peaks in zip(layers, stages, peaks):
        found = _offload_stage(stage_layers, stage, stage_peaks, fronts, own, link, bucket, capacity)
        if found is None:
            return None
        chosen.update(found[0])
        transit = found[1]
        held.append(_Peaks(stage_peaks.warm_up + transit,
                           None if stage_peaks.backward is None else stage_peaks.backward + transit))
    return _result(layers, held, chosen, None, fronts, own)


def option_label(option: LayerOption) -> str:
    """How an option reads in the search's output."""
    if option.recompute is None:
        label = "full recompute"
    elif not option.recompute:
        label = "no recompute"
    else:
        label = "recompute " + "+".join(name for name in SWITCHES if name in option.recompute)
    return label + ", offloaded to the host" if option.link_bandwidth else label


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
        order, with ``"offload": True`` for a range that offloads; ``kind``
        is the kind's name, or ``None``.
    """
    records = []
    for item in choice.ranges:
        record = {
            "first": item.first,
            "count": item.count,
            "kind": item.kind.name if item.kind is not None else None,
            "recompute": option_record(item.option),
        }
        if item.option.link_bandwidth:
            record["offload"] = True
        records.append(record)
    return records


def describe(choice: RecomputeChoice) -> str:
    """The choice as one line per range of layers, after its mode when one was chosen for every layer."""
    lines = [] if choice.mode is None else [f"every layer: {choice.mode}"]
    for item in choice.ranges:
        last = item.first + item.count - 1
        layers = f"layer {item.first}" if item.count == 1 else f"layers {item.first}-{last}"
        kind = f" ({item.kind.name})" if item.kind is not None else ""
        lines.append(f"{layers}{kind}: {option_label(item.option)}")
    return "\n".join(lines)
