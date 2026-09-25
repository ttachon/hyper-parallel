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
"""What each recompute switch of a layer kind saves and costs, measured alone.

A selective layer's memory and backward time add up over its switches. Each
switch gates its own activation term in the memory model and its own op in
the recompute estimate, and only ``gather`` acts on the communication
buffers, both on the memory kept and on the volume sent again. So a layer that
recomputes a set of ops costs the plain layer plus what each of those ops
costs alone, and a :class:`SwitchProfile`, nine measurements, prices all 128
settings of the seven switches.
"""
from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

# The recompute switches; 1 keeps an op's activation, 0 recomputes the op.
SWITCHES = ("attBMM", "headCast", "dropout", "softmax", "normOp", "gather", "ffAct")


@dataclass(frozen=True)
class Cost:
    """What running a layer one way costs.

    Attributes:
        memory_per_micro_batch: Bytes kept for each micro-batch in flight: the
            activations, and the communication buffers that grow with the
            micro-batches in flight.
        memory_once: Bytes of communication buffers kept once, however many
            micro-batches are in flight.
        backward_time: The backward time, recompute included.
        excess: At each of its profile's :attr:`~SwitchProfile.counts`, the
            bytes the two memories charge beyond what the layer keeps with
            that many micro-batches in flight. The split is exact at one
            micro-batch and at the most any stage keeps, and a buffer that
            does not grow can hide one that does in between.
    """

    memory_per_micro_batch: float
    memory_once: float
    backward_time: float
    excess: Tuple[float, ...] = ()

    def __add__(self, other: "Cost") -> "Cost":
        """The sum, cost by cost."""
        return Cost(
            *(mine + theirs for mine, theirs in zip(self.values(), other.values())),
            excess=tuple(mine + theirs for mine, theirs in zip(self.excess, other.excess)),
        )

    def __sub__(self, other: "Cost") -> "Cost":
        """The difference, cost by cost."""
        return Cost(
            *(mine - theirs for mine, theirs in zip(self.values(), other.values())),
            excess=tuple(mine - theirs for mine, theirs in zip(self.excess, other.excess)),
        )

    def values(self) -> Tuple[float, float, float]:
        """``(memory per micro-batch, memory once, backward time)``."""
        return self.memory_per_micro_batch, self.memory_once, self.backward_time


@dataclass(frozen=True)
class SwitchProfile:
    """A layer kind's costs: plain, with each op alone recomputed, and fully recomputed.

    Attributes:
        forward_time: The layer's forward time, whatever it recomputes.
        plain: The layer keeping every op.
        alone: Per switch, the layer recomputing that op alone.
        full: The layer fully recomputed.
        counts: The counts of micro-batches in flight, between one and the
            most any stage keeps, at which each cost states its excess.
        working: What the working set of the layer's backward, which the
            memory model charges the layer that ends warm-up, holds beyond
            what the layer keeps at one micro-batch: ``(gathers kept,
            gathers recomputed)``. Only ``gather`` acts on it: FSDP that
            reshards holds two layers' gathered parameters in a backward and
            none between the layer's passes, in the buffers the gathers take.
        first_working: The same for the backward a stage runs last, its
            first layer's, as a micro-batch's backward ends: one layer's
            gathered parameters, with none left to prefetch.
    """

    forward_time: float
    plain: Cost
    alone: Mapping[str, Cost]
    full: Cost
    counts: Tuple[int, ...] = ()
    working: Tuple[float, float] = (0.0, 0.0)
    first_working: Tuple[float, float] = (0.0, 0.0)

    def selective(self, recompute: Iterable[str]) -> Cost:
        """The cost of recomputing the ops *recompute* names: the plain layer's, plus what each costs alone."""
        chosen = set(recompute)
        cost = self.plain
        for name in SWITCHES:
            if name in chosen:
                cost = cost + (self.alone[name] - self.plain)
        return cost
