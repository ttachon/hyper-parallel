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
"""Architecture identity and sharding rules for Qwen3.5-MoE.

Qwen3.5-MoE interleaves Gated DeltaNet (linear attention) layers with full
attention layers and pairs 256 routed experts with a sigmoid-gated shared
expert. The GDN block is the same one the dense Qwen3.5 family ships, so the
CP wrappers are reused from ``models/qwen3_5``; only the MoE-specific
``shared_expert_gate`` rule is added here.
"""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_context_parallel():
    """Return the shared Qwen3.5 GDN CP wrapper module through a lazy provider."""
    from hyper_parallel.models.qwen3_5.adapter.distributed import (  # pylint: disable=C0415
        context_parallel,
    )
    return context_parallel


def _load_expert_parallel():
    """Return the family's EP compute factory through a lazy provider."""
    from hyper_parallel.models.qwen3_5_moe.adapter.distributed import (  # pylint: disable=C0415
        expert_parallel,
    )
    return expert_parallel


def _load_sharding_rules():
    """Return GDN and MoE parameter roles not covered by generic naming rules.

    The GDN rules mirror the dense Qwen3.5 family. ``shared_expert_gate`` is a
    scalar gate Linear that the default rules deliberately leave unmatched
    (``shared_expert`` is matched as a whole segment), which would otherwise
    leave it uncovered and fail the planner's coverage check.
    """
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )
    return [
        ("in_proj_qkv", ParamRole.FUSED_QKV),
        (["in_proj_z", "in_proj_b", "in_proj_a", "conv1d"], ParamRole.COLWISE),
        (["A_log", "dt_bias"], ParamRole.COLWISE),
        ("out_proj", ParamRole.ROWWISE),
        ("shared_expert_gate", ParamRole.REPLICATED),
    ]


QWEN3_5_MOE_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3_5MoeForConditionalGeneration",
    model_type="qwen3_5_moe",
    context_parallel=_load_context_parallel,
    expert_parallel=_load_expert_parallel,
    sharding_rules=_load_sharding_rules,
)

QWEN3_5_MOE_TEXT_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="Qwen3_5MoeForCausalLM",
    model_type="qwen3_5_moe_text",
    context_parallel=_load_context_parallel,
    expert_parallel=_load_expert_parallel,
    sharding_rules=_load_sharding_rules,
)

register_model_adapter(QWEN3_5_MOE_ADAPTER_SPEC)
register_model_adapter(QWEN3_5_MOE_TEXT_ADAPTER_SPEC)
