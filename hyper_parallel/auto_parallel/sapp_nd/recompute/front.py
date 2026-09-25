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
"""The recompute front of a layer kind: the options no other option beats.

An option runs a layer one way: plain, recomputing some of the seven ops the
recompute switches name, or fully recomputed. Of the 128 settings of the
switches and full recompute, the front keeps those that no other option beats
on memory per micro-batch, memory held once, memory at each count of
micro-batches in flight a stage keeps, and backward time together. The plain
layer and full recompute are always on it.

:func:`layer_fronts` is the interim form of the search's entry point for
layer options (shared decision S3): it measures each layer kind on the memory
backbone's config at the evaluator's current strategy, and phase 5 of the
search re-implements it on the model IR.
"""
import itertools
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterator, List, Mapping, Optional, Sequence, Tuple

from hyper_parallel.auto_parallel._op_profiles import LayerKind
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import LayerTimes
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig
from hyper_parallel.auto_parallel.sapp_nd.recompute.profile import SWITCHES, Cost, SwitchProfile


@dataclass(frozen=True)
class LayerOption:
    """One way of running a layer of a kind, and what it costs.

    Attributes:
        recompute: The switches set to recompute their op; ``None`` for full
            recompute.
        memory_per_micro_batch: Bytes kept for each micro-batch in flight.
        memory_once: Bytes kept once, however many micro-batches are in flight.
        forward_time: The forward time.
        backward_time: The backward time, recompute included.
        link_bandwidth: For an option that offloads, the bytes each
            micro-batch moves to the host after its forward and back before
            its backward; 0 for one that keeps them on the device.
        names: The names the pipeline balancer and the parsers know the
            option by: NONE, SLCT, COMM, BOTH or FULL.
        excess: ``(count, bytes)`` for each count of micro-batches in flight
            at which the two memories charge more than the layer keeps, by
            that many bytes; see :attr:`Cost.excess`.
    """

    recompute: Optional[FrozenSet[str]]
    memory_per_micro_batch: float
    memory_once: float
    forward_time: float
    backward_time: float
    link_bandwidth: float = 0.0
    names: Tuple[str, ...] = ()
    excess: Tuple[Tuple[int, float], ...] = ()

    def memory(self, in_flight: int) -> float:
        """The bytes a layer running the option keeps with *in_flight* micro-batches in flight."""
        kept = in_flight * self.memory_per_micro_batch + self.memory_once
        return kept - next((excess for count, excess in self.excess if count == in_flight), 0.0)

    @property
    def switches(self) -> Optional[Dict[str, int]]:
        """The switch setting the option stands for, 1 to keep an op and 0 to recompute it; ``None`` if full."""
        if self.recompute is None:
            return None
        return {name: int(name not in self.recompute) for name in SWITCHES}


@dataclass(frozen=True)
class KindFront:
    """The front of one layer kind of one model.

    Attributes:
        model_name: The model the kind belongs to, a sub-model of a
            multimodal one.
        kind: The layer kind; ``None`` for a model whose layers are all priced
            on its config as it stands.
        options: The kind's options, fastest first.
    """

    model_name: str
    kind: Optional[LayerKind]
    options: Tuple[LayerOption, ...]


def _candidates(profile: SwitchProfile) -> Iterator[Tuple[Optional[FrozenSet[str]], Cost]]:
    """Every option and its cost, those that recompute fewer ops first and full recompute last."""
    for size in range(len(SWITCHES) + 1):
        for names in itertools.combinations(SWITCHES, size):
            yield frozenset(names), profile.selective(names)
    yield None, profile.full


def _names(recompute: Optional[FrozenSet[str]], configured: Mapping[str, Any]) -> Tuple[str, ...]:
    """The names an option is known by; the configured switches name SLCT, COMM and BOTH."""
    if recompute is None:
        return ("FULL",)
    if not recompute:
        return ("NONE",)
    selective = frozenset(name for name in SWITCHES if not int(bool(configured.get(name, 1))))
    names = []
    if selective and recompute == selective:
        names.append("SLCT")
    if recompute == {"gather"}:
        names.append("COMM")
    if selective and recompute == selective | {"gather"}:
        names.append("BOTH")
    return tuple(names)


def price_option(
    profile: SwitchProfile, recompute: Optional[FrozenSet[str]], configured: Optional[Mapping[str, Any]] = None
) -> LayerOption:
    """The option recomputing *recompute*, priced from the kind's *profile*.

    Args:
        profile: The kind's costs, each switch measured alone.
        recompute: The switches to recompute; ``None`` for full recompute.
        configured: The switches the config sets, which name the option.

    Returns:
        The option, whether or not it is on the kind's front.
    """
    cost = profile.full if recompute is None else profile.selective(recompute)
    return LayerOption(
        recompute=None if recompute is None else frozenset(recompute),
        memory_per_micro_batch=cost.memory_per_micro_batch,
        memory_once=cost.memory_once,
        forward_time=profile.forward_time,
        backward_time=cost.backward_time,
        names=_names(recompute, configured or {}),
        excess=tuple((count, excess) for count, excess in zip(profile.counts, cost.excess) if excess),
    )


def _compared(cost: Cost, counts: Sequence[int]) -> Tuple[float, ...]:
    """What an option is compared on: its two memories and backward time, and its memory at each of *counts*."""
    at = (
        count * cost.memory_per_micro_batch + cost.memory_once - excess
        for count, excess in zip(counts, cost.excess)
    )
    return cost.values() + tuple(at)


def build_front(
    profile: SwitchProfile, configured: Optional[Mapping[str, Any]] = None
) -> Tuple[LayerOption, ...]:
    """The options of a layer kind that no other option beats.

    An option beats another when it needs no more memory per micro-batch, no
    more memory once and no more backward time, and no more memory at any of
    the profile's counts of micro-batches in flight. Of options that cost the
    same, the one that recomputes fewer ops stays. The plain layer and full
    recompute are always kept.

    Args:
        profile: The kind's costs, each switch measured alone.
        configured: The switches the config sets, which name the options the
            pipeline balancer knows as SLCT, COMM and BOTH.

    Returns:
        The options, fastest first.
    """
    candidates: List[Tuple[Optional[FrozenSet[str]], Cost]] = list(_candidates(profile))
    compared = [_compared(cost, profile.counts) for _, cost in candidates]
    kept = []
    for index, (recompute, _) in enumerate(candidates):
        mine = compared[index]
        beaten = any(
            all(theirs <= own for theirs, own in zip(other, mine))
            and (other != mine or other_index < index)
            for other_index, other in enumerate(compared)
            if other_index != index
        )
        if not beaten or recompute is None or not recompute:
            kept.append(price_option(profile, recompute, configured))
    return tuple(sorted(kept, key=lambda option: (option.backward_time, -option.memory_per_micro_batch)))


def configured_switches(evaluator: EvaluatorV2) -> Mapping[str, Any]:
    """The recompute switches the evaluator's config sets."""
    rec_op = getattr(evaluator.ccfg, "rec_op", None)
    return vars(rec_op) if rec_op is not None else {}


def layer_profiles(
    evaluator: EvaluatorV2,
    device_type: Any,
    ccfg: Optional[CustomConfig] = None,
    most_in_flight: Optional[int] = None,
    each_switch: bool = True,
    in_flight: Sequence[int] = (),
) -> Dict[Tuple[str, Optional[LayerKind]], SwitchProfile]:
    """What each switch of every layer kind saves and costs, at the evaluator's current strategy.

    Args:
        evaluator: The memory evaluator, set to the strategy to price.
        device_type: The device the times are priced on.
        ccfg: Estimator options; the search's defaults when omitted.
        most_in_flight: The most micro-batches any stage keeps in flight, up
            to which an option's memory is exact; see
            :meth:`EvaluatorV2.estimate_switch_profiles`.
        each_switch: Whether to measure each switch alone; without, a profile
            prices only the plain and the fully recomputed layer.
        in_flight: The counts of micro-batches in flight the stages keep, at
            each of which an option's memory is exact too.

    Returns:
        ``{(model name, layer kind): SwitchProfile}``, in model order.
    """
    return evaluator.estimate_switch_profiles(
        LayerTimes(device_type, ccfg), most_in_flight=most_in_flight, each_switch=each_switch, in_flight=in_flight
    )


def layer_fronts(
    evaluator: EvaluatorV2,
    device_type: Any,
    ccfg: Optional[CustomConfig] = None,
    most_in_flight: Optional[int] = None,
) -> Tuple[KindFront, ...]:
    """The recompute front of every layer kind of the evaluator's model, at its current strategy.

    Measures each kind's switches with the memory backbone and the
    performance estimate the search scores with, and leaves the evaluator's
    config as it found it.

    Args:
        evaluator: The memory evaluator, set to the strategy to price.
        device_type: The device the times are priced on.
        ccfg: Estimator options; the search's defaults when omitted.
        most_in_flight: The most micro-batches any stage keeps in flight, up
            to which an option's memory is exact; see
            :meth:`EvaluatorV2.estimate_switch_profiles`.

    Returns:
        One front per model and layer kind, in model order.
    """
    profiles = layer_profiles(evaluator, device_type, ccfg, most_in_flight)
    configured = configured_switches(evaluator)
    return tuple(
        KindFront(model_name, kind, build_front(profile, configured))
        for (model_name, kind), profile in profiles.items()
    )
