# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""hardware abstraction"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger


@dataclass(frozen=True)
class HostLink:
    """How fast a device moves activations to its host's memory and back, as offload prices it.

    The figures each device states below are assumptions, not measurements:
    measure the link on the target with hyper_offload's
    ``profile_transfer_bandwidth``, which reports GiB/s, and the throughput
    from a profiled training step, or take the vendor's, and state them
    here or with ``run_nd --host_link_gibps`` and ``--sustained_tflops``.

    Attributes:
        gib_per_s: The sustained copy bandwidth between the device and
            pinned host memory, in GiB/s, with one copy stream carrying both
            directions, as hyper_offload and the trainer's input swap run it.
        sustained_tflops: The device's sustained dense throughput at the
            training precision, in TFLOP/s, which turns a copy's seconds into
            the performance estimate's units.
        overlap: The share of the compute time the copy stream may take
            without slowing the compute.
    """

    gib_per_s: float
    sustained_tflops: float
    overlap: float = 0.8

    def __post_init__(self) -> None:
        """Refuse a figure that cannot price a copy."""
        if self.gib_per_s <= 0 or self.sustained_tflops <= 0 or not 0 < self.overlap <= 1:
            raise ValueError(f"a host link needs positive figures and an overlap in (0, 1], not {self}")

    def seconds_per_byte(self) -> float:
        """The seconds one byte takes over the link, one way."""
        return 1.0 / (self.gib_per_s * 2 ** 30)

    def flops_per_second(self) -> float:
        """The device's sustained FLOP/s."""
        return self.sustained_tflops * 10 ** 12


class Type:
    """Machine type"""

    name: str
    levels: int  # levels in hierarchy
    level_bound_number: list[int]  # devices per level
    level_bandwidth: list[int]  # bandwidth (GB/s) per level
    host_link: Optional[HostLink]  # the copy path to the host, for offload

    def __init__(self, name, bounds, bandwidths, host_link=None):
        self.name = name
        self.level_bound_number = bounds
        self.level_bandwidth = bandwidths
        if len(bounds) != len(bandwidths):
            raise ValueError("bounds and bandwidths must have the same length")
        self.levels = len(bounds)
        self.host_link = host_link

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def devices_below_level(self, level):
        """Number of devices below the given hierarchy level"""
        devices = 1
        for lvl in range(min(level, self.levels)):
            devices *= self.level_bound_number[lvl]
        return devices

    def intra_node_num(self):
        """Number of devices in a node"""
        return self.devices_below_level(1)

    def levels_used(self, device_number):
        """Number of hierarchy level used"""
        devices = 1
        for lvl in range(self.levels):
            if self.level_bound_number[lvl]:
                devices *= self.level_bound_number[lvl]
                if device_number <= devices:
                    return lvl
            else:
                return lvl
        return self.levels

    def level_assign(self, dp=1, tp=1, cp=1, pp=1, ep=1):
        """device assignment of the different parallel dimensions"""
        # EP borrows from DP, not counted in total devices; kept for
        # topology tracking only — callers should NOT pass ep > 1
        # unless they also account for EP-in-DP convention.
        device_number = dp * tp * cp * pp * ep
        logger.debug("DP = %d, TP = %d, EP = %d, CP = %d, PP = %d", dp, tp, ep, cp, pp)
        # Order matters: each dim takes the devices the previous ones left.
        degrees = {Dim.TP: tp, Dim.EP: ep, Dim.CP: cp, Dim.DP: dp, Dim.PP: pp}
        assignment = {dim: [] for dim in degrees}
        for level in range(self.levels):
            bound = self.level_bound_number[level]
            if bound:
                remaining_devices = max(min(device_number, bound), 1)
                device_number = device_number // bound
            else:
                remaining_devices = max(device_number, 1)
            for dim, degree in degrees.items():
                dim_level = min(degree, remaining_devices)
                assignment[dim].append(dim_level)
                degrees[dim] = degree // dim_level
                remaining_devices = remaining_devices // dim_level

        return assignment


# The host links are placeholders until measured (HostLink): 16 GiB/s is
# hyper_offload's own default before it profiles the link, and the
# throughputs are about half of each device's dense BF16 or FP16 peak.
# Device_A2 = Machine(devices_per_node=8, inter_node_bw=10, intra_node_bw=50)
Device_A2 = Type(
    name="A2", bounds=[8, None], bandwidths=[50, 10], host_link=HostLink(gib_per_s=16.0, sustained_tflops=140.0)
)
Device_A3 = Type(
    name="A3",
    bounds=[16, 24, None],
    bandwidths=[200, 25, 10],
    host_link=HostLink(gib_per_s=16.0, sustained_tflops=160.0),
)
device_map = {
    "A2": Device_A2,
    "A3": Device_A3,
    "V100": Type(
        name="V100", bounds=[8, None], bandwidths=[50, 10], host_link=HostLink(gib_per_s=12.0, sustained_tflops=60.0)
    ),
}


class Machine:
    """Hardware description"""

    number: int
    device: Type

    def __init__(self, number, device):
        self.number = number
        if isinstance(device, int):
            if device == 2:
                self.device = Device_A2
            elif device == 3:
                self.device = Device_A3
            else:
                raise ValueError(f"Ascend A{device} unknown")
        elif isinstance(device, str):
            if device not in device_map:
                raise ValueError(
                    f"Device {device} is not supported. "
                    f"Supported devices: {list(device_map.keys())}"
                )
            self.device = device_map[device]
        else:
            self.device = device

    def update_num_if_none(self, num):
        """Assign number of device if not already precised"""
        if self.number is None:
            self.number = num

    def pipeline_bound(self):
        """Return pipeline bound from hardware topology because as pipeline may currently not cross hierarchy levels"""
        max_bound = 1
        devices = self.number
        while devices > 1:
            max_bound = max(
                max_bound,
                devices
                // self.device.devices_below_level(
                    self.device.levels_used(devices)
                ),
            )
            devices = devices // 2
        # devices = self.devices_below_level(self.levels_used(device_number))
        # return device_number // devices
        return max_bound


def prime_factors(n):
    """Decompose n into a product of prime factors"""
    divisor = 2
    factors = []
    while n > 1:
        while n % divisor != 0:
            divisor += 1
        factors.append(divisor)
        n = n // divisor
    return factors


def all_factors_combinations(factors):
    """Computes all divisors from a prime factor list"""
    def rec_factors(n, factors):
        combinations = {n}
        for u in set(factors):
            remaining = factors.copy()
            remaining.remove(u)
            combinations = combinations.union(rec_factors(n * u, remaining))
        return combinations
    return rec_factors(1, factors)


def all_divisors(n, reverse=False, min_bound=1, max_bound=float("inf")):
    """Computes all divisors of an integer n"""
    divisors = sorted(
        all_factors_combinations(prime_factors(n)), reverse=reverse
    )
    div_in_bound = []
    for d in divisors:
        if min_bound <= d <= max_bound:
            div_in_bound.append(d)

    return div_in_bound


def from_prime_factors(factors):
    """Compute a number from its prime factor decomposition"""
    number = 1
    for f in factors:
        number *= f
    return number


def split_node(n, device):
    """Split decompositions into intra & inter devices"""
    devices_per_node = device.intra_node_num()
    nodes = prime_factors(max(1, n // devices_per_node))
    intra = prime_factors(min(n, devices_per_node))
    return [intra, nodes]


def unique_factors(factors):
    """Remove duplicates. Factors are sorted"""
    offset = 0
    for i, f in enumerate(factors[:-1]):
        j = i - offset
        if factors[j + 1] == f:
            factors.pop(j)
            offset += 1
    return factors


def highest_power_of_2_divisor(divisor_of):
    """Compute the highest number that is both a divisor of 'divisor_of' and a power of 2"""
    divisor = 1
    factors = prime_factors(divisor_of)
    for f in factors:
        if f == 2:
            divisor *= f
    return divisor


def get_cp_topology(tp_degree: int, cp_degree: int, device_per_node: int) -> tuple:
    """Determine CP topology and effective bandwidth.

    Args:
        tp_degree: Tensor parallelism degree.
        cp_degree: Context parallelism degree.
        device_per_node: Number of devices per node.

    Returns:
        Tuple of (topology_type, effective_bandwidth, is_intra_node).
        - topology_type: "intra-node" or "cross-node"
        - effective_bandwidth: Bandwidth in GB/s
        - is_intra_node: True if CP stays within node
    """
    total_devices_needed = tp_degree * cp_degree

    if total_devices_needed <= device_per_node:
        topology_type = "intra-node"
        is_intra_node = True
        effective_bandwidth = 300.0
    else:
        topology_type = "cross-node"
        is_intra_node = False
        effective_bandwidth = 25.0

    return topology_type, effective_bandwidth, is_intra_node


def get_cp_bandwidth(topology_type: str, device_type: str = "A2") -> float:
    """Get effective bandwidth for CP communication based on topology.

    Args:
        topology_type: "intra-node" or "cross-node"
        device_type: Device type string (e.g., "A2", "A3")

    Returns:
        Bandwidth in GB/s
    """
    device = device_map.get(device_type, Device_A2)

    if topology_type == "intra-node":
        return device.level_bandwidth[0] if device.level_bandwidth else 300.0
    return device.level_bandwidth[1] if len(device.level_bandwidth) > 1 else 25.0


def recommend_cp_max_by_attention(attention_type: str) -> int:
    """Recommend maximum CP degree based on attention type.

    Args:
        attention_type: "mla", "gqa", or "mha"

    Returns:
        Recommended maximum CP degree
    """
    attention_type_upper = attention_type.upper()
    if attention_type_upper == "MLA":
        return 16
    if attention_type_upper == "GQA":
        return 8
    return 4
