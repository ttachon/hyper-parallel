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
"""expert_parallel: the Qwen3.5-MoE EP compute archetype factory.

Qwen3.5-MoE composes the routed branch with a sigmoid-gated shared expert, so
it shares the ``qwen2moe_shared_expert_gate`` archetype rather than the
Qwen3-MoE one, whose combine returns the routed branch alone. It differs from
the generic ``recipes.qwen2moe_ep_compute_fn`` only in exposing
``use_grouped_gemm``, which selects the grouped-GEMM local expert path instead
of the per-expert Python loop.
"""

import logging
from typing import Any, Callable

from hyper_parallel.distributed.expert_parallel.recipes import (
    build_ep_compute,
)
from hyper_parallel.distributed.expert_parallel.routing import (
    MOE_ROUTER_ADAPTERS,
)
from hyper_parallel.distributed.recipe_spec import (
    local_compute,
)

logger = logging.getLogger(__name__)


@local_compute
def qwen3_5_moe_ep_compute_fn(
        *,
        module: Any,
        mesh: Any,
        tp_mesh: Any,
        cp_mesh: Any,
        ep_mesh: Any,
        use_grouped_gemm: bool = False,
) -> Callable:
    """Archetype ``qwen2moe_shared_expert_gate`` with a selectable local path.

    Output is ``routed + sigmoid(shared_expert_gate(x)) * shared_expert(x)``.
    The shared branch must stay in this combine: with ``region_dispatch:
    false`` this factory owns the whole MoE block output, so a routed-only
    combine drops the shared expert silently rather than failing.

    ``use_grouped_gemm`` selects ``npu_grouped_swiglu`` over the per-expert
    loop. The loop's cost is not its GEMMs, whose FLOPs are EP-invariant, but
    its indexing: it selects ``gate_up_proj[i]`` and ``down_proj[i]`` from the
    stacked parameters once per LOCAL expert, and each select's backward
    materialises a zero buffer the size of the whole stack before accumulating
    into it. Zeroing and accumulation are therefore quadratic in the local
    expert count, which grows as ep_size falls. The grouped path passes the
    stacks whole and never indexes them, so those buffers do not appear at all.

    Expected module interface: ``gate``, ``experts``, ``shared_expert``,
    ``shared_expert_gate``.
    """
    del mesh, tp_mesh, cp_mesh

    def combine(module: Any, hidden_states: Any, routed: Any) -> Any:
        """Merge the routed branch with the gated shared-expert branch."""
        shared = module.shared_expert(hidden_states)            # nested boundary
        gate = module.shared_expert_gate(hidden_states).sigmoid()
        return routed + gate * shared

    compute_fn = build_ep_compute(
        module,
        ep_mesh,
        router_fn=MOE_ROUTER_ADAPTERS["qwen2moe"],
        archetype_key="qwen2moe_shared_expert_gate",
        expected_attrs=["gate", "experts", "shared_expert", "shared_expert_gate"],
        combine=combine,
        use_grouped_gemm=use_grouped_gemm,
    )
    # Stated once per boundary at apply time: a config that never reached this
    # factory is otherwise indistinguishable, in a profile, from one that did.
    logger.info(
        "Qwen3.5-MoE EP compute: use_grouped_gemm=%s, local_experts=%s",
        use_grouped_gemm,
        getattr(module.experts, "local_expert_count", "unknown"),
    )
    return compute_fn


__all__ = [
    "qwen3_5_moe_ep_compute_fn",
]
