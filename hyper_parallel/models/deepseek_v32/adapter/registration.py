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
"""Architecture and lazy provider registration for DeepSeek-V3.2."""

from hyper_parallel.models.adapter_spec import ModelAdapterSpec
from hyper_parallel.models.registry import register_model_adapter


def _load_replacements():
    """Return the family's replacement factories without importing NPU ops."""
    from hyper_parallel.models.deepseek_v32.adapter import replacements  # pylint: disable=C0415

    return replacements


def _load_context_parallel():
    """Return the DeepSeek-V3.2 DSA CP wrapper provider lazily."""
    from hyper_parallel.models.deepseek_v32.adapter.distributed import (  # pylint: disable=C0415
        context_parallel,
    )

    return context_parallel


def _load_sharding_rules():
    """Return the DSA/MLA tensor-parallel naming-rule overrides.

    The fused latent projection and DSA indexer stay replicated.  Main MLA
    up-projections are sharded by attention head; the generic rule keeps the
    output projection rowwise and reduces its output across TP ranks.
    """
    from hyper_parallel.distributed.tensor_parallel.param_role import (  # pylint: disable=C0415
        ParamRole,
    )

    return [
        (["linear_qkv"], ParamRole.REPLICATED),
        (["q_b_proj", "kv_b_proj"], ParamRole.COLWISE),
        (
            [
                "indexer.wq_b",
                "indexer.wk",
                "indexer.weights_proj",
            ],
            ParamRole.REPLICATED,
        ),
    ]


DEEPSEEK_V32_ADAPTER_SPEC = ModelAdapterSpec(
    architecture="DeepseekV32ForCausalLM",
    model_type="deepseek_v32",
    replacements=_load_replacements,
    context_parallel=_load_context_parallel,
    sharding_rules=_load_sharding_rules,
)
register_model_adapter(DEEPSEEK_V32_ADAPTER_SPEC)
