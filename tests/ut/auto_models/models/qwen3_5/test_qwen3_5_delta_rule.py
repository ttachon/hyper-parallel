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
"""Tests for switching Qwen3.5's linear-attention layers to HyperParallel's gated delta rule."""
import importlib.util
import unittest
from typing import Any
from unittest.mock import patch

import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from hyper_parallel.components.modules.gated_delta_net import torch_chunk_gated_delta_rule
from hyper_parallel.models.qwen3_5.adapter import delta_rule
from hyper_parallel.models.qwen3_5.adapter.delta_rule import chunk_gated_delta_rule_solve, install_delta_rule

_RULE_NAME = "torch_chunk_gated_delta_rule"
_RULE_ATTRIBUTE = "chunk_gated_delta_rule"
_HYPER_RULES = {torch_chunk_gated_delta_rule.__name__, chunk_gated_delta_rule_solve.__name__}
# Seventy tokens: a full chunk of 64 and a padded one.
_TOKENS = 70


def _model():
    """A two-layer Qwen3.5-MoE causal LM, one layer of each attention kind, in float32 on the CPU."""
    torch.manual_seed(11)
    config = Qwen3_5MoeTextConfig.from_dict({
        "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 16, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16, "linear_num_key_heads": 2, "linear_key_head_dim": 8,
        "linear_num_value_heads": 2, "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4,
        "vocab_size": 64, "max_position_embeddings": 128, "layer_types": ["linear_attention", "full_attention"],
    })
    config.experts_implementation = "eager"
    return modeling_qwen3_5_moe.Qwen3_5MoeForCausalLM(config).float()


def _step(model):
    """The logits and every parameter's gradient of one forward and backward."""
    model.zero_grad(set_to_none=True)
    ids = torch.arange(_TOKENS).remainder(64).unsqueeze(0)
    logits = model(input_ids=ids, use_cache=False).logits
    (logits * torch.linspace(-1, 1, logits.numel()).reshape(logits.shape)).sum().backward()
    return [logits.detach()] + [param.grad.clone() for param in model.parameters() if param.grad is not None]


def _linear_layer(model):
    """The model's linear-attention module."""
    return model.model.layers[0].linear_attn


def _rule_called(module):
    """The rule *module* calls: the one it holds up to Transformers 5.14, its modeling module's name after."""
    if hasattr(module, _RULE_ATTRIBUTE):
        return getattr(module, _RULE_ATTRIBUTE)
    return getattr(modeling_qwen3_5_moe, _RULE_NAME)


class TestQwen35DeltaRule(unittest.TestCase):
    """The switch keeps every number and leaves what it cannot reproduce alone."""

    def setUp(self) -> None:
        """Put Transformers' rule back after each test, which a switch from 5.15 on replaces by name."""
        fallback = getattr(modeling_qwen3_5_moe, _RULE_NAME)
        self.addCleanup(setattr, modeling_qwen3_5_moe, _RULE_NAME, fallback)

    def test_solve_rule_agrees_with_the_oracle(self):
        """The triangular-solve formulation computes the oracle's map, to rounding."""
        generator = torch.Generator().manual_seed(2)
        inputs = [torch.randn(1, 19, 2, 4, generator=generator), torch.randn(1, 19, 2, 4, generator=generator),
                  torch.randn(1, 19, 2, 3, generator=generator), -torch.rand(1, 19, 2, generator=generator),
                  torch.rand(1, 19, 2, generator=generator)]
        results = []
        for rule in (torch_chunk_gated_delta_rule, chunk_gated_delta_rule_solve):
            leaves = [tensor.clone().requires_grad_(True) for tensor in inputs]
            output, state = rule(*leaves, chunk_size=8, output_final_state=True, use_qk_l2norm_in_kernel=True)
            (output.sum() + state.sum()).backward()
            results.append([output.detach(), state.detach()] + [leaf.grad for leaf in leaves])
        for oracle, solved in zip(*results):
            torch.testing.assert_close(solved, oracle, rtol=1e-5, atol=1e-5)

    def test_install_keeps_every_number(self):
        """The linear layer runs a HyperParallel rule and the logits and gradients stay bit for bit."""
        model = _model()
        fallback = _rule_called(_linear_layer(model))
        expected = _step(model)

        switched = install_delta_rule(model)

        rule = _rule_called(_linear_layer(model))
        self.assertEqual(switched, 1, f"expected the one linear-attention layer switched, got {switched}")
        self.assertIsNot(rule, fallback, f"the layer still calls Transformers' {fallback!r}")
        self.assertIn(rule.__name__, _HYPER_RULES, f"the layer calls {rule.__name__}, not a HyperParallel rule")
        for index, (got, want) in enumerate(zip(_step(model), expected)):
            self.assertTrue(torch.equal(got, want),
                            f"tensor {index} moved: largest difference {(got - want).abs().max().item()}")

    def test_a_rule_no_candidate_reproduces_is_left_running(self):
        """An installed fallback that no HyperParallel rule reproduces is kept, and the run is told why."""
        fallback = getattr(modeling_qwen3_5_moe, _RULE_NAME)

        def _doubled(*args: Any, **kwargs: Any) -> tuple:
            """The installed rule with its output doubled, which no HyperParallel rule reproduces."""
            output, state = fallback(*args, **kwargs)
            return output * 2, state

        setattr(modeling_qwen3_5_moe, _RULE_NAME, _doubled)
        model = _model()
        with self.assertLogs(delta_rule.logger, level="WARNING") as logs:
            switched = install_delta_rule(model)

        self.assertEqual(switched, 0, f"expected no layer switched, got {switched}")
        self.assertIs(_rule_called(_linear_layer(model)), _doubled,
                      f"the layer now calls {_rule_called(_linear_layer(model))!r} instead of the installed rule")
        self.assertIn("no HyperParallel gated delta rule reproduces", "\n".join(logs.output),
                      f"unexpected warnings: {logs.output}")

    def test_a_layer_running_a_kernel_is_left_alone(self):
        """A layer that runs the fla kernels rather than the fallback keeps them."""
        model = _model()
        layer = _linear_layer(model)

        def _kernel(*args: Any, **kwargs: Any) -> tuple:
            """Stands in for an fla kernel; never called."""
            return args, kwargs

        if hasattr(layer, _RULE_ATTRIBUTE):
            setattr(layer, _RULE_ATTRIBUTE, _kernel)
            switched = install_delta_rule(model)
            kept = getattr(layer, _RULE_ATTRIBUTE)
            expected = _kernel
        else:
            find_spec = importlib.util.find_spec
            with patch.object(importlib.util, "find_spec",
                              side_effect=lambda name, *rest: object() if name == "fla" else find_spec(name, *rest)):
                expected = getattr(modeling_qwen3_5_moe, _RULE_NAME)
                switched = install_delta_rule(model)
            kept = getattr(modeling_qwen3_5_moe, _RULE_NAME)
        self.assertEqual(switched, 0, f"expected no layer switched, got {switched}")
        self.assertIs(kept, expected, f"the layer's kernel {expected!r} was replaced by {kept!r}")


if __name__ == "__main__":
    unittest.main()
