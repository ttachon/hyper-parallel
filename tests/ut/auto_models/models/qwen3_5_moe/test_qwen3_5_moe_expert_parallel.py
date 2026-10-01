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
"""Contracts for the Qwen3.5-MoE expert-parallel compute factory."""
# pylint: disable=wrong-import-position

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
from torch import nn

from hyper_parallel.distributed.expert_parallel import experts as ep_experts
from hyper_parallel.distributed.recipe_spec import LOCAL_COMPUTE
from hyper_parallel.models.qwen3_5_moe.adapter.distributed import expert_parallel
from hyper_parallel.models.registry import get_model_adapter

RECIPES = [
    Path(__file__).resolve().parents[5] / "hyper_parallel" / "models"
    / "qwen3_5_moe" / "recipes" / "train.yaml",
    Path(__file__).resolve().parents[5] / "examples" / "training_demo"
    / "train_qwen3_5_moe.yaml",
]


class _TinyExperts(nn.Module):
    """Stacked routed experts in the fused Hugging Face layout."""

    def __init__(self, num_experts: int = 4, hidden: int = 8, inter: int = 8) -> None:
        """Create the fused gate/up and down expert stacks."""
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.zeros(num_experts, 2 * inter, hidden))
        self.down_proj = nn.Parameter(torch.zeros(num_experts, hidden, inter))


class _TinySharedExpert(nn.Module):
    """Dense shared-expert body returning a recognisable constant."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return a constant so the combine's shared term is identifiable."""
        return torch.full_like(hidden_states, 3.0)


class _TinyGate(nn.Module):
    """Scalar sigmoid gate returning zeros, so sigmoid(0) is exactly 0.5."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return a zero logit per token."""
        return torch.zeros(
            *hidden_states.shape[:-1], 1, dtype=hidden_states.dtype
        )


class _TinyMoeBlock(nn.Module):
    """Minimal module exposing the qwen2moe archetype interface."""

    def __init__(self) -> None:
        """Create the router, routed experts, shared expert and its gate."""
        super().__init__()
        self.gate = nn.Linear(8, 4, bias=False)
        self.experts = _TinyExperts()
        self.shared_expert = _TinySharedExpert()
        self.shared_expert_gate = _TinyGate()


class _FakeEpMesh:
    """Mesh stub exposing only what build_ep_compute reads."""

    class _Axis:
        """One named mesh axis."""

        def __init__(self, size: int) -> None:
            """Store the axis extent."""
            self._size = size

        def size(self) -> int:
            """Return the axis extent."""
            return self._size

    def __init__(self, ep_size: int = 2) -> None:
        """Store the EP extent."""
        self._ep_size = ep_size

    def __getitem__(self, name: str) -> "_FakeEpMesh._Axis":
        """Return the named axis."""
        del name
        return _FakeEpMesh._Axis(self._ep_size)

    def get_group(self, name: str) -> str:
        """Return a placeholder process group handle."""
        del name
        return "ep-group"


class TestQwen35MoeExpertParallel(unittest.TestCase):
    """Pin the archetype composition and the grouped-GEMM switch."""

    def test_factory_is_a_local_compute_injection_point(self):
        """The factory must be usable as a plan_overrides local_compute_fn."""
        meta = expert_parallel.qwen3_5_moe_ep_compute_fn._injection_meta
        self.assertEqual(meta.kind, LOCAL_COMPUTE)

    def test_adapter_exposes_the_expert_parallel_provider(self):
        """Both Qwen3.5-MoE identities resolve to this provider module."""
        for key in ("qwen3_5_moe", "qwen3_5_moe_text"):
            spec = get_model_adapter(key)
            self.assertIsNotNone(spec.expert_parallel, key)
            self.assertIs(spec.expert_parallel(), expert_parallel, key)

    def test_combine_keeps_the_gated_shared_expert(self):
        """Output is routed + sigmoid(gate(x)) * shared(x).

        A routed-only combine (the Qwen3-MoE archetype) satisfies every other
        check in this file and silently drops the shared expert, so this is the
        regression that matters.
        """
        module = _TinyMoeBlock()
        hidden = torch.ones(5, 8)
        routed = torch.full_like(hidden, 7.0)
        with patch.object(
            expert_parallel, "build_ep_compute", wraps=expert_parallel.build_ep_compute
        ) as spy, patch.object(
            ep_experts, "_install_bound_forward"
        ):
            expert_parallel.qwen3_5_moe_ep_compute_fn(
                module=module,
                mesh=None,
                tp_mesh=None,
                cp_mesh=None,
                ep_mesh=_FakeEpMesh(ep_size=2),
            )
        combine = spy.call_args.kwargs["combine"]
        out = combine(module, hidden, routed)
        # sigmoid(0) = 0.5, shared = 3.0, so 7.0 + 0.5 * 3.0 = 8.5
        self.assertTrue(torch.allclose(out, torch.full_like(hidden, 8.5)))
        self.assertFalse(torch.allclose(out, routed), "shared expert was dropped")

    def test_use_grouped_gemm_reaches_the_expert_module(self):
        """The flag must select npu_grouped_swiglu over the per-expert loop."""
        for requested in (True, False):
            module = _TinyMoeBlock()
            with patch.object(ep_experts, "_install_bound_forward"):
                expert_parallel.qwen3_5_moe_ep_compute_fn(
                    module=module,
                    mesh=None,
                    tp_mesh=None,
                    cp_mesh=None,
                    ep_mesh=_FakeEpMesh(ep_size=2),
                    use_grouped_gemm=requested,
                )
            self.assertEqual(module.experts.ep_use_grouped_gemm, requested)
            self.assertEqual(module.experts.local_expert_count, 2)

    def test_recipes_request_the_grouped_path(self):
        """Both shipped configs must select this factory with grouped GEMM."""
        for recipe in RECIPES:
            entries = yaml.safe_load(recipe.read_text(encoding="utf-8"))["plan_overrides"]
            ep_entries = [e for e in entries if e.get("when") == "ep"]
            self.assertEqual(len(ep_entries), 1, recipe.name)
            target = ep_entries[0]["local_compute_fn"]
            self.assertEqual(
                target["_target_"],
                "hyper_parallel.models.qwen3_5_moe.adapter.distributed"
                ".expert_parallel.qwen3_5_moe_ep_compute_fn",
                recipe.name,
            )
            self.assertTrue(target["use_grouped_gemm"], recipe.name)


if __name__ == "__main__":
    unittest.main()
