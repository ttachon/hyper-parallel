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
"""Build a layer-cropped Qwen3.5-MoE model from configuration only."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PreTrainedModel

from hyper_parallel.models._transformers import HyperAutoModelForCausalLM
from hyper_parallel.distributed.mesh import DistributedSetup
from hyper_parallel.models.build_options import CompileConfig

_QWEN3_5_MOE_MODEL_TYPES = ("qwen3_5_moe", "qwen3_5_moe_text")


def build_cropped_qwen3_5_moe(
        config_path: str,
        num_hidden_layers: int = 8,
        num_experts: int | None = None,
        local_files_only: bool = True,
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        experts_implementation: str | None = "eager",
        validate_placement: bool = False,
        distributed_setup: DistributedSetup | None = None,
        peft_config: Any | None = None,
        compile_config: CompileConfig | dict[str, Any] | None = None,
        activation_checkpoint: str | None = None,
        activation_checkpoint_layer_ranges: list[dict[str, Any]] | None = None,
        activation_swap: str = "none",
) -> PreTrainedModel:
    """Create a Qwen3.5-MoE model with fewer decoder layers and random weights.

    The function calls ``from_config`` instead of ``from_pretrained``: it reads
    only the Hugging Face configuration from ``config_path`` and loads no
    checkpoint tensor, so every parameter is randomly initialized. Only the
    text tower is built; the vision tower and the MTP layer are not part of the
    causal-LM class and are absent from the cropped model.

    Qwen3.5-MoE alternates three ``linear_attention`` layers with one
    ``full_attention`` layer. Keep ``num_hidden_layers`` a multiple of four so
    the cropped model preserves that ratio.

    ``num_experts`` shrinks the routed-expert count for a cheaper smoke test.
    It must stay a multiple of ``accelerator.ep_size``. Leave it unset to keep
    the configured width, which is what a representative run wants.

    Args:
        config_path: Local Hugging Face Qwen3.5-MoE model directory.
        num_hidden_layers: Decoder layers retained in the cropped model.
        num_experts: Routed experts retained, or None to keep the configured count.
        local_files_only: Disable implicit Hub downloads when true.
        torch_dtype: Model parameter dtype accepted by HyperAutoModel.
        attn_implementation: Hugging Face attention implementation name.
        experts_implementation: Hugging Face experts implementation name, or
            None to keep the configured default. Transformers defaults to
            ``grouped_mm``, which needs ``torch._grouped_mm``; ``eager`` is the
            portable per-expert path.
        validate_placement: Enable HyperParallel placement validation.
        distributed_setup: Trainer-provided distributed topology.
        peft_config: Optional Trainer-provided PEFT configuration.
        compile_config: Optional Trainer-provided compile configuration.
        activation_checkpoint: Activation checkpoint mode.
        activation_checkpoint_layer_ranges: The layer ranges that run another
            recompute mode than ``activation_checkpoint``, as the trainer's
            ``activation_checkpoint.layer_ranges`` states them, or None.
        activation_swap: Activation swap mode.

    Returns:
        A parallelized, randomly initialized cropped Qwen3.5-MoE model.
    """
    if num_hidden_layers <= 0:
        raise ValueError("num_hidden_layers must be positive")

    config = AutoConfig.from_pretrained(
        config_path,
        local_files_only=local_files_only,
        trust_remote_code=False,
    )
    model_type = getattr(config, "model_type", None)
    if model_type not in _QWEN3_5_MOE_MODEL_TYPES:
        raise ValueError(
            "config_path must contain a Qwen3.5-MoE configuration; "
            f"got model_type={model_type!r}"
        )

    # Qwen3.5-MoE carries decoder depth in text_config, and layer_types must be
    # truncated with it or the linear/full attention pattern desynchronises.
    text_config = getattr(config, "text_config", config)
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types is not None:
        if num_hidden_layers > len(layer_types):
            raise ValueError(
                f"num_hidden_layers={num_hidden_layers} exceeds the configured "
                f"depth of {len(layer_types)}"
            )
        text_config.layer_types = list(layer_types[:num_hidden_layers])
    text_config.num_hidden_layers = num_hidden_layers
    if num_experts is not None:
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        text_config.num_experts = num_experts
        if getattr(text_config, "num_experts_per_tok", 0) > num_experts:
            text_config.num_experts_per_tok = num_experts
    if experts_implementation is not None:
        text_config.experts_implementation = experts_implementation
    config.use_cache = False

    return HyperAutoModelForCausalLM.from_config(
        config,
        distributed_setup=distributed_setup,
        peft_config=peft_config,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        validate_placement=validate_placement,
        compile_config=compile_config,
        activation_checkpoint=activation_checkpoint,
        activation_checkpoint_layer_ranges=activation_checkpoint_layer_ranges,
        activation_swap=activation_swap,
    )
