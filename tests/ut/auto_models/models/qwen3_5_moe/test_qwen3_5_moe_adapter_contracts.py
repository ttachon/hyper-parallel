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
"""Essential contracts for the Qwen3.5-MoE adapter identity and recipe."""
# pylint: disable=wrong-import-position

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import nn

from hyper_parallel.distributed.tensor_parallel.param_role import (
    ParameterClassifier,
    ParamRole,
)
from hyper_parallel.models.qwen3_5.adapter.distributed import (
    context_parallel as qwen3_5_context_parallel,
)
from hyper_parallel.models.registry import get_model_adapter

RECIPE = (
    Path(__file__).resolve().parents[5]
    / "hyper_parallel" / "models" / "qwen3_5_moe" / "recipes" / "train.yaml"
)


class _TinyGatedDeltaNet(nn.Module):
    """Parameter-only Qwen3.5 GDN boundary carrying the real parameter names."""

    def __init__(self) -> None:
        """Create the projection, Conv, state, norm and output parameters."""
        super().__init__()
        self.in_proj_qkv = nn.Linear(8, 24, bias=False)
        self.in_proj_z = nn.Linear(8, 8, bias=False)
        self.in_proj_b = nn.Linear(8, 2, bias=False)
        self.in_proj_a = nn.Linear(8, 2, bias=False)
        self.conv1d = nn.Conv1d(24, 24, 3, groups=24, bias=False)
        self.A_log = nn.Parameter(torch.zeros(2))  # pylint: disable=invalid-name
        self.dt_bias = nn.Parameter(torch.zeros(2))
        self.norm = nn.LayerNorm(8)
        self.out_proj = nn.Linear(8, 8, bias=False)


class _TinyExperts(nn.Module):
    """Stacked routed-expert parameters in the Hugging Face fused layout."""

    def __init__(self) -> None:
        """Create the fused gate/up and down expert stacks."""
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16))
        self.down_proj = nn.Parameter(torch.zeros(4, 8, 8))


class _TinySharedExpert(nn.Module):
    """Dense shared-expert body run beside the routed branch."""

    def __init__(self) -> None:
        """Create the three shared-expert projections."""
        super().__init__()
        self.gate_proj = nn.Linear(8, 8, bias=False)
        self.up_proj = nn.Linear(8, 8, bias=False)
        self.down_proj = nn.Linear(8, 8, bias=False)


class _TinySparseMoeBlock(nn.Module):
    """Router, routed experts, shared expert and its scalar sigmoid gate."""

    def __init__(self) -> None:
        """Create the MoE block boundary parameters."""
        super().__init__()
        self.gate = nn.Linear(8, 4, bias=False)
        self.experts = _TinyExperts()
        self.shared_expert = _TinySharedExpert()
        self.shared_expert_gate = nn.Linear(8, 1, bias=False)


class _TinyDecoderLayer(nn.Module):
    """One linear-attention decoder layer with its MoE block."""

    def __init__(self) -> None:
        """Create the layer's two boundaries."""
        super().__init__()
        self.linear_attn = _TinyGatedDeltaNet()
        self.mlp = _TinySparseMoeBlock()


class _TinyInner(nn.Module):
    """``model`` submodule holding the decoder stack."""

    def __init__(self) -> None:
        """Create the one-layer stack."""
        super().__init__()
        self.layers = nn.ModuleList([_TinyDecoderLayer()])


class _TinyQwen35MoeModel(nn.Module):
    """Small official-identity model with one GDN layer and one MoE block.

    The decoder stack is nested under ``model.layers.N`` because several
    default naming rules (notably the MoE router's ``.mlp.gate.``) are dotted
    path patterns that a flattened fixture would not match.
    """

    def __init__(self) -> None:
        """Expose the architecture and the stable layer FQNs."""
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"]
        )
        self.model = _TinyInner()


class TestQwen35MoeAdapterContracts(unittest.TestCase):
    """Pin the Qwen3.5-MoE identity, sharding coverage and shipped recipe."""

    def test_both_identities_resolve_and_share_the_qwen3_5_cp_wrappers(self):
        """Text and multimodal identities discover one adapter and one CP provider."""
        # Resolve the nested text identity first to exercise cold discovery
        # through the family-directory alias.
        text_spec = get_model_adapter("qwen3_5_moe_text")
        top_spec = get_model_adapter("Qwen3_5MoeForConditionalGeneration")
        self.assertIsNotNone(text_spec)
        self.assertIsNotNone(top_spec)
        self.assertEqual(text_spec.model_type, "qwen3_5_moe_text")
        self.assertEqual(top_spec.model_type, "qwen3_5_moe")
        self.assertIs(text_spec.context_parallel(), qwen3_5_context_parallel)
        self.assertIs(top_spec.context_parallel(), qwen3_5_context_parallel)
        self.assertIsNotNone(get_model_adapter("Qwen3_5MoeForCausalLM"))

    def test_dense_qwen3_5_identities_are_unaffected(self):
        """Adding the MoE family must not capture the dense family's lookups."""
        self.assertEqual(get_model_adapter("qwen3_5").model_type, "qwen3_5")
        self.assertEqual(
            get_model_adapter("qwen3_5_text").model_type, "qwen3_5_text"
        )
        self.assertEqual(
            get_model_adapter("Qwen3_5ForCausalLM").model_type, "qwen3_5"
        )

    def test_sharding_rules_cover_every_gdn_and_moe_parameter(self):
        """No parameter falls through to SKIP, which fails the planner's coverage
        check, and none reaches the SPECIAL Shard(0) handler."""
        model = _TinyQwen35MoeModel()
        arch = "qwen3_5_moe_text"
        rules = list(get_model_adapter(arch).sharding_rules())
        roles = ParameterClassifier(arch_overrides={arch: rules}).classify(
            model, arch
        )
        uncovered = sorted(
            name for name, role in roles.items()
            if role in (ParamRole.SKIP, ParamRole.SPECIAL)
        )
        self.assertEqual(uncovered, [])
        layer = "model.layers.0"
        expected = {
            f"{layer}.mlp.shared_expert_gate.weight": ParamRole.REPLICATED,
            f"{layer}.mlp.gate.weight": ParamRole.MOE_GATE,
            f"{layer}.mlp.experts.gate_up_proj": ParamRole.MOE_EXPERT,
            f"{layer}.mlp.shared_expert.gate_proj.weight": ParamRole.SHARED_EXPERT,
            f"{layer}.linear_attn.A_log": ParamRole.COLWISE,
            f"{layer}.linear_attn.dt_bias": ParamRole.COLWISE,
            f"{layer}.linear_attn.conv1d.weight": ParamRole.COLWISE,
            f"{layer}.linear_attn.in_proj_qkv.weight": ParamRole.FUSED_QKV,
            f"{layer}.linear_attn.out_proj.weight": ParamRole.ROWWISE,
        }
        for name, role in expected.items():
            self.assertEqual(roles[name], role, name)

    def test_recipe_uses_the_shared_expert_aware_ep_archetype(self):
        """The EP override must keep the gated shared-expert branch.

        ``qwen3moe_ep_compute_fn`` returns the routed branch only: it passes the
        archetype interface check and silently drops ``shared_expert``.
        """
        entries = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))["plan_overrides"]
        ep_entries = [e for e in entries if e.get("when") == "ep"]
        self.assertEqual(len(ep_entries), 1)
        self.assertEqual(ep_entries[0]["match"], "*.mlp")
        self.assertEqual(
            ep_entries[0]["local_compute_fn"]["_target_"],
            "hyper_parallel.distributed.expert_parallel.recipes"
            ".qwen2moe_ep_compute_fn",
        )

    def test_recipe_pins_the_topologies_master_can_actually_run(self):
        """TP breaks the GDN grouped conv and master's Trainer has no pipeline
        schedule, so the shipped recipe must not ship either enabled."""
        config = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
        self.assertEqual(config["accelerator"]["tp_size"], 1)
        self.assertEqual(config["accelerator"]["pp_size"], 1)
        self.assertEqual(
            config["fsdp_config"]["edp_shard_size"]
            * config["accelerator"]["ep_size"],
            config["fsdp_config"]["dp_shard_size"],
        )


if __name__ == "__main__":
    unittest.main()
