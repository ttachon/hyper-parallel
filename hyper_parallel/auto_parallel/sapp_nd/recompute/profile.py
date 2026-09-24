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
    """

    memory_per_micro_batch: float
    memory_once: float
    backward_time: float

    def __add__(self, other: "Cost") -> "Cost":
        """The sum, cost by cost."""
        return Cost(*(mine + theirs for mine, theirs in zip(self.values(), other.values())))

    def __sub__(self, other: "Cost") -> "Cost":
        """The difference, cost by cost."""
        return Cost(*(mine - theirs for mine, theirs in zip(self.values(), other.values())))

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
    """

    forward_time: float
    plain: Cost
    alone: Mapping[str, Cost]
    full: Cost

    def selective(self, recompute: Iterable[str]) -> Cost:
        """The cost of recomputing the ops *recompute* names: the plain layer's, plus what each costs alone."""
        chosen = set(recompute)
        cost = self.plain
        for name in SWITCHES:
            if name in chosen:
                cost = cost + (self.alone[name] - self.plain)
        return cost
