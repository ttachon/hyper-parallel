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
"""Tests for the layer census, the activations it states in the model spec, and their pricing."""
import functools
import os
import sys
import tempfile
import unittest
from typing import Optional
from unittest.mock import patch

import torch
import yaml
from torch._subclasses.fake_tensor import FakeTensorMode

from hyper_parallel.auto_parallel._hf_model_spec import resolve_hf_model_spec
from hyper_parallel.auto_parallel._layer_census import (  # pylint: disable=protected-access
    census_activations,
    census_final_norm,
    census_flops,
    census_layer,
    census_output_activations,
    census_parameters,
    census_recomputed,
    census_saved_ops,
    replacement_specs,
    _measure,
    _RecomputedMatmuls,
    _selective_contexts,
    tp_config,
)
from hyper_parallel.auto_parallel._model_spec import KindActivations, ModelSpec, ModelSpecError
from hyper_parallel.auto_parallel._npu_contracts import npu_contracts
from hyper_parallel.models.replacement import module_replacement
from hyper_parallel.core.activation_memory import api as activation_memory
from hyper_parallel.core.activation_memory import sac
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import bind_layer_stack

_HF_CONFIG = "hyper_parallel.auto_parallel._hf_model_spec._get_hf_config"
_CAUSAL_LM = "hyper_parallel.models._transformers.HyperAutoModelForCausalLM.from_pretrained"
_IMAGE_TEXT = "hyper_parallel.models._transformers.HyperAutoModelForImageTextToText.from_pretrained"


def _qwen35_text():
    """A two-layer Qwen3.5-MoE text config, one layer of each attention kind."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # pylint: disable=C0415
        Qwen3_5MoeTextConfig,
    )
    return Qwen3_5MoeTextConfig.from_dict({
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32, "linear_num_key_heads": 2, "linear_key_head_dim": 16,
        "linear_num_value_heads": 4, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
        "vocab_size": 128, "max_position_embeddings": 256, "layer_types": ["linear_attention", "full_attention"],
    })


_STACK = [{"kind": "linear_attention", "count": 1}, {"kind": "full_attention", "count": 1}]


class _Layer(torch.nn.Module):
    """A layer of an attention projection and a feed-forward of two, 64 wide within 256."""

    def __init__(self):
        """Its three projections."""
        super().__init__()
        self.self_attn = torch.nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(64, 256, bias=False, dtype=torch.bfloat16), torch.nn.ReLU(),
                                       torch.nn.Linear(256, 64, bias=False, dtype=torch.bfloat16))

    def forward(self, hidden):
        """The attention's projection, then the feed-forward."""
        return self.mlp(self.self_attn(hidden))


def _qwen35(text):
    """The vision-language config around *text*, its vision tower Transformers' default."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # pylint: disable=C0415
        Qwen3_5MoeConfig,
    )
    return Qwen3_5MoeConfig.from_dict({"text_config": text.to_dict()})


def _train_yaml(folder: str, target: str = _CAUSAL_LM, mode: str = "off", pp: int = 1,
                plan_overrides: Optional[list] = None, **context) -> str:
    """A train yaml of the hybrid model on two ranks, at DP shard 2 over PP 1 or on two stages."""
    config = {
        "model": {"_target_": target, "pretrained_model_name_or_path": "local/qwen3_5_moe",
                  "torch_dtype": "bfloat16"},
        "training": {"global_batch_size": 4, "micro_batch_size": 1},
        "accelerator": {"tp_size": 1, "pp_size": pp, "ep_size": 1, "cp_size": 1},
        "fsdp_config": {"dp_shard_size": 2 // pp},
        "activation_checkpoint": {"mode": mode},
        "dataset": {"data_transform": {"max_seq_len": 4096}},
        "context": dict(context, device_num=2),
    }
    if plan_overrides:
        config["plan_overrides"] = plan_overrides
    path = os.path.join(folder, "train.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return path


def _evaluator(model_config, **train) -> EvaluatorV2:
    """The evaluator of :func:`_train_yaml`'s run of *model_config*."""
    with patch(_HF_CONFIG, return_value=model_config):
        with tempfile.TemporaryDirectory() as folder:
            return EvaluatorV2(_train_yaml(folder, **train), framework="hyper_v2", log_level=0)


def _spec(**activations) -> dict:
    """A two-kind hybrid spec stating *activations*."""
    return {
        "name": "qwen3_5_moe", "arch": "qwen3_5", "hidden_size": 64, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 128,
        "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "layers": _STACK, "activations": activations,
    }


class TestLayerCensus(unittest.TestCase):
    """The census counts what a fake layer keeps and holds."""

    def test_a_tensor_parallel_rank_holds_a_share(self):
        """
        Feature: tp_config and census_layer.
        Description: Each kind's layer, and the same layer as one of two
            tensor-parallel ranks holds it.
        Expectation: The rank's layer has half the heads and widths, the same
            head width, and keeps and holds less, but more than half.
        """
        config = _qwen35_text()
        rank = tp_config(config, 2)
        self.assertEqual((rank.num_attention_heads, rank.num_key_value_heads, rank.head_dim), (2, 1, 16))
        self.assertEqual((rank.moe_intermediate_size, rank.linear_num_value_heads), (16, 2))
        for index in (0, 1):
            whole, share = census_layer(config, index, 64), census_layer(rank, index, 64)
            for full, half in zip(whole, share):
                self.assertGreater(full, half)
                self.assertGreater(2 * half, full)

    def test_a_gradient_the_size_of_a_parameter_is_an_activations(self):
        """
        Feature: _measure, which tells gradients apart.
        Description: A projection of a 64-wide input of 64 tokens, whose
            input's gradient has as many elements as its weight.
        Expectation: The forward keeps the input; when the backward peaks,
            the input, the output's gradient and the input's are the
            activations held, and the weight's gradient is not.
        """
        with FakeTensorMode():
            weight = torch.nn.Parameter(torch.empty(64, 64, dtype=torch.bfloat16))
            hidden = torch.empty(64, 64, dtype=torch.bfloat16, requires_grad=True)
            size = hidden.untyped_storage().nbytes()
            saved, working = _measure([weight], (hidden,), lambda: hidden @ weight.t(),
                                      lambda out: out.backward(torch.ones_like(out)))
        self.assertEqual((saved, working), (size, 3 * size))

    def test_selective_checkpointing_keeps_every_other_matmul(self):
        """
        Feature: _measure of a forward under HyperParallel's selective
            activation checkpointing.
        Description: Two projections of a 64-wide input of 64 tokens with
            an activation between them, checkpointed as the trainer
            checkpoints a layer.
        Expectation: The forward keeps its input and the first projection's
            output, which the policy saves; it recomputes the activation
            and the second projection.
        """
        with FakeTensorMode():
            first, second = (torch.nn.Parameter(torch.empty(64, 64, dtype=torch.bfloat16)) for _ in range(2))
            hidden = torch.empty(64, 64, dtype=torch.bfloat16, requires_grad=True)
            size = hidden.untyped_storage().nbytes()
            kept, _ = _measure(
                [first, second], (hidden,),
                lambda: activation_memory.checkpoint(lambda states: torch.relu(states @ first.t()) @ second.t(),
                                                     hidden, swap_inputs=False, context_fn=_selective_contexts),
                lambda out: out.backward(torch.ones_like(out)), checkpointed=True)
        self.assertEqual(kept, 2 * size)

    def test_the_policy_runs_every_other_matmul_again(self):
        """
        Feature: the census's ledger of HyperParallel's selective policy.
        Description: A layer of an attention projection and a feed-forward
            of two, of 64 tokens, run under selective checkpointing.
        Expectation: The policy saves the first and the third projections'
            outputs and runs the second again: none of the attention's
            FLOPs, and half the feed-forward's.
        """
        with FakeTensorMode():
            layer = _Layer()
            hidden = torch.empty(64, 64, dtype=torch.bfloat16, requires_grad=True)
            ledger = _RecomputedMatmuls(layer)
            activation_memory.checkpoint(layer, hidden, swap_inputs=False,
                                         context_fn=functools.partial(_selective_contexts, ledger))
        self.assertEqual((ledger.share(attention=True), ledger.share(attention=False)), (0.0, 0.5))

    def test_a_selective_layer_keeps_what_the_policy_saves(self):
        """
        Feature: census_layer under HyperParallel's selective checkpointing.
        Description: Each kind's layer checkpointed selectively; the
            full-attention one also with SAC ignoring the ops that build
            empty tensors, as the trainer's selective setup has it.
        Expectation: A layer keeps less than without recompute and more
            than its input; the full-attention one keeps the attention's
            output and two statistics, 24 KiB at 64 tokens, whatever SAC
            ignores.
        """
        config = _qwen35_text()
        size = 64 * 64 * 2
        for index in (0, 1):
            (saved, _), (kept, _) = (census_layer(config, index, 64, selective=each) for each in (False, True))
            self.assertLess(kept, saved)
            self.assertGreater(kept, size)
        self.assertGreaterEqual(kept, size + 24 * 1024)
        ignored = set(sac.SAC_IGNORED_OPS) | {torch.ops.aten.empty.memory_format, torch.ops.aten.empty_like.default}
        with patch.object(sac, "SAC_IGNORED_OPS", ignored):
            self.assertEqual(census_layer(config, 1, 64, selective=True)[0], kept)

    def test_each_kind_gets_its_record(self):
        """
        Feature: census_activations.
        Description: The census of a stack of one linear-attention layer and
            one full-attention layer.
        Expectation: A record per kind, at the census's length, splitting
            each layer's bytes between what TP splits and what it does not,
            with and without selective checkpointing.
        """
        got = census_activations(_qwen35_text(), _STACK, 64)
        self.assertEqual(sorted(got), ["full_attention", "linear_attention"])
        for kind, index in (("linear_attention", 0), ("full_attention", 1)):
            record = got[kind]
            saved, working = census_layer(_qwen35_text(), index, 64)
            kept, _ = census_layer(_qwen35_text(), index, 64, selective=True)
            self.assertEqual(record.seq_length, 64)
            self.assertAlmostEqual((record.saved + record.saved_tp) * 64, saved)
            self.assertAlmostEqual((record.working + record.working_tp) * 64, working)
            self.assertAlmostEqual((record.selective + record.selective_tp) * 64, kept)
            self.assertEqual((record.selective_attention_mm, record.selective_ffn_mm),
                             census_recomputed(_qwen35_text(), index, 64))
            self.assertGreater(record.saved_tp, 0)
            self.assertGreater(record.selective_tp, 0)
            for share in (record.selective_attention_mm, record.selective_ffn_mm):
                self.assertTrue(0 < share < 1, share)
            self.assertAlmostEqual(sum(record.ops.values()), record.saved)
            self.assertAlmostEqual(sum(record.ops_tp.values()), record.saved_tp)
        self.assertGreater(got["linear_attention"].ops_tp["linrec"], 0)
        self.assertGreater(got["full_attention"].ops_tp["attBMM"], 0)

    def test_what_a_layer_saves_for_each_op(self):
        """
        Feature: census_saved_ops, the op records the census fills.
        Description: A Llama layer of width 64, 4 query heads and 2 key
            heads 16 wide, a gated feed-forward 128 wide, in bf16.
        Expectation: Per token: the projections' input, the rotary tables
            and the output projection's input for the attention's
            projections; the kernel's queries, keys, values and output for
            its batched matmuls, and its fp32 statistics for the softmax;
            each RMSNorm's fp32 input, its scale and its bf16 output; the
            feed-forward's input, its gate and up outputs as its product
            takes them and that product for its projections, and the gate's
            output for its activation function.  They sum to what the layer
            keeps.
        """
        from transformers import LlamaConfig  # pylint: disable=C0415
        config = LlamaConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             intermediate_size=128, vocab_size=128)
        ops = census_saved_ops(config, 0, 64)
        self.assertEqual(ops, {
            "attMM": 2 * (64 + 2 * 16 + 64), "attBMM": 2 * (4 * 16 + 2 * 2 * 16 + 4 * 16), "softmax": 4 * 2 * 4 * 8,
            "normOp": 2 * (4 * 64 + 4 + 2 * 64), "ffMM": 2 * (64 + 3 * 128), "ffAct": 2 * 128})
        self.assertEqual(sum(ops.values()) * 64, census_layer(config, 0, 64)[0])

    def test_a_layers_parameters_by_part(self):
        """
        Feature: census_parameters and census_final_norm.
        Description: Each kind's layer of the model of width 64, with 4
            routed experts and a shared expert, each 32 wide.
        Expectation: The router holds a weight per expert; the routed
            experts three projections each; the shared expert its three and
            its gate; the norms two RMSNorms of the width, and in the
            full-attention layer the per-head query and key norms beside
            them.  The final norm is one RMSNorm.
        """
        config = _qwen35_text()
        for index, norms in ((0, 2 * 64), (1, 2 * 64 + 2 * 16)):
            parts = census_parameters(config, index)
            self.assertEqual((parts["router"], parts["routed"], parts["shared"], parts["norm"]),
                             (64 * 4, 4 * 3 * 64 * 32, 3 * 64 * 32 + 64, norms))
            self.assertGreater(parts["attention"], 0)
        self.assertEqual(census_final_norm(config), 64)

    def test_a_layers_forward_flops_by_part(self):
        """
        Feature: census_flops.
        Description: Each kind's layer of the model of width 64 on 32
            tokens: 4 query heads of 16 behind an output gate and 2 key
            heads; 4 routed experts 32 wide, 2 chosen, a shared expert 32
            wide and its gate.
        Expectation: The full-attention layer's projections; its scores and
            values at every pair of tokens; each token's two experts' three
            projections, the shared expert's three and its gate, and the
            router's.  The linear-attention layer's recurrence is counted
            apart from its projections, and it has no scores.
        """
        config, seq = _qwen35_text(), 32
        full = census_flops(config, 1, seq)
        self.assertEqual(full["attention"], 2 * seq * 64 * (2 * 4 * 16 + 2 * 2 * 16 + 4 * 16))
        self.assertEqual(full["scores"], 2 * 4 * seq * seq * (16 + 16))
        self.assertEqual((full["routed"], full["shared"], full["router"]),
                         (2 * seq * 2 * 3 * 64 * 32, 2 * seq * (3 * 64 * 32 + 64), 2 * seq * 64 * 4))
        linear = census_flops(config, 0, seq)
        self.assertGreater(linear["linrec"], 0)
        self.assertNotIn("scores", linear)

    def test_mla_values_keep_their_width(self):
        """
        Feature: census_flops and census_layer on MLA.
        Description: A DeepSeek-V3 layer of width 64 on 32 tokens: 4 heads
            whose queries and keys are 16 wide, 12 and a rotary 4, and whose
            values are 8.
        Expectation: The scores run at the queries' and keys' width and the
            values at their own, unpadded, as under HyperParallel's sdpa; the
            layer runs its backward.
        """
        from transformers import DeepseekV3Config  # pylint: disable=C0415
        config = DeepseekV3Config(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=32,
            kv_lora_rank=16, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=8, intermediate_size=32,
            first_k_dense_replace=1, vocab_size=128, n_group=1, topk_group=1)
        self.assertEqual(census_flops(config, 0, 32)["scores"], 2 * 4 * 32 * 32 * (16 + 8))
        self.assertGreater(census_layer(config, 0, 32)[0], 0)

    def test_the_output_layer_keeps_its_fp32_log_probabilities(self):
        """
        Feature: census_output_activations.
        Description: The output layer of the model with a vocabulary of
            4096, which its logits outweigh.
        Expectation: Per token and vocabulary entry, the layer keeps the
            loss's fp32 log-probabilities and its backward holds three such
            tensors: the part a vocabulary-parallel loss splits.
        """
        config = _qwen35_text()
        config.vocab_size = 4096
        record = census_output_activations(config, 48)
        self.assertEqual((record.saved_tp, record.working_tp, record.seq_length), (4 * 4096, 12 * 4096, 48))
        self.assertGreater(record.saved, 0)


class TestKindActivations(unittest.TestCase):
    """The model spec states a census's records."""

    def test_round_trip_through_yaml(self):
        """
        Feature: ModelSpec.activations.
        Description: A spec stating each kind's record, dumped to YAML and
            read back.
        Expectation: The same spec.
        """
        record = {"saved": 2036.25, "saved_tp": 3812.5, "working": 2162.375, "working_tp": 3812.5,
                  "seq_length": 64}
        selective = dict(record, selective=512.0, selective_tp=1024.5, selective_attention_mm=0.25,
                         selective_ffn_mm=0.625, ops={"attMM": 1000.25, "other": 1036.0},
                         ops_tp={"attBMM": 3812.5})
        spec = ModelSpec.from_dict(dict(_spec(linear_attention=selective, full_attention=record),
                                         output_activations=record))
        self.assertEqual(spec.activations["linear_attention"].to_dict(), selective)
        self.assertEqual(spec.activations["full_attention"].to_dict(), record)
        self.assertIsInstance(spec.activations["linear_attention"], KindActivations)
        self.assertIsInstance(spec.output_activations, KindActivations)
        self.assertEqual(ModelSpec.from_dict(yaml.safe_load(yaml.safe_dump(spec.to_dict()))), spec)

    def test_a_record_is_checked(self):
        """
        Feature: KindActivations.from_dict and ModelSpec.validate.
        Description: A negative size, a missing field, what a layer keeps
            for each op without the part TP splits, an op the op vector
            lacks, a negative op's size, and a kind no layer of the stack
            is.
        Expectation: Each raises, naming what is wrong.
        """
        record = {"saved": 1.0, "saved_tp": 1.0, "working": 1.0, "working_tp": 1.0, "seq_length": 64}
        with self.assertRaisesRegex(ModelSpecError, "negative"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, saved=-1.0)))
        with self.assertRaisesRegex(ModelSpecError, "lacks"):
            ModelSpec.from_dict(_spec(linear_attention={"saved": 1.0}))
        with self.assertRaisesRegex(ModelSpecError, "without the other"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, selective=1.0)))
        with self.assertRaisesRegex(ModelSpecError, "exceed 1"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, selective_attention_mm=0.5,
                                                            selective_ffn_mm=1.5)))
        with self.assertRaisesRegex(ModelSpecError, "without the other"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, ops={"attMM": 1.0})))
        with self.assertRaisesRegex(ModelSpecError, "must map ops"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, ops={"matmul": 1.0}, ops_tp={})))
        with self.assertRaisesRegex(ModelSpecError, "negative"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, ops={"attMM": -1.0}, ops_tp={})))
        with self.assertRaisesRegex(ModelSpecError, "decoder"):
            ModelSpec.from_dict(_spec(decoder=record))
        with self.assertRaisesRegex(ModelSpecError, "output_activations must map"):
            ModelSpec.from_dict(dict(_spec(), output_activations=[1.0]))


# The recipe whose plan_overrides install HyperParallel's fused modules on
# the Qwen3-MoE Transformers builds.
_RECIPE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 3,
    "hyper_parallel", "models", "qwen3_moe", "recipes", "train.yaml",
)


def _qwen3_moe():
    """A two-layer Qwen3-MoE config of width 64, 4 experts 32 wide, as the recipe's rules match it."""
    from transformers import Qwen3MoeConfig  # pylint: disable=C0415
    return Qwen3MoeConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          head_dim=16, moe_intermediate_size=32, num_experts=4, num_experts_per_tok=2,
                          intermediate_size=128, vocab_size=128)


def _recipe_specs():
    """The module replacements the shipped Qwen3-MoE recipe installs."""
    with open(_RECIPE, encoding="utf-8") as handle:
        return replacement_specs(yaml.safe_load(handle).get("plan_overrides"))


@module_replacement
def _needs_a_library(*, module, module_fqn, context):
    """A factory of a module this host cannot build."""
    del module, module_fqn, context
    raise ImportError("No module named 'a_native_extension'")


class TestModuleReplacements(unittest.TestCase):
    """The census runs the modules a train.yaml's plan_overrides install."""

    def test_the_entries_that_replace_a_module(self):
        """
        Feature: replacement_specs.
        Description: The shipped Qwen3-MoE recipe, whose plan_overrides state
            three replacements and two entries that shard alone, under
            ``when: cp`` and ``when: ep``.
        Expectation: One rule per replacement, in the order stated, each with
            its patterns, its factory and the source type it replaces; the
            sharding entries are left out.
        """
        specs = _recipe_specs()
        self.assertEqual([spec.factory.__name__ for spec in specs],
                         ["replace_qwen3_moe_rms_norm", "replace_qwen3_moe_flash_attention",
                          "replace_qwen3_moe_grouped_experts"])
        self.assertEqual(specs[0].match, ("*.input_layernorm", "*.post_attention_layernorm", "model.norm"))
        self.assertEqual([spec.module_type.__name__ for spec in specs],
                         ["Qwen3MoeRMSNorm", "Qwen3MoeAttention", "Qwen3MoeExperts"])

    def test_an_entry_is_checked(self):
        """
        Feature: replacement_specs' checks.
        Description: An entry that replaces a module without stating its
            type, one naming a module that cannot be imported, and one
            naming no factory of that name.
        Expectation: Each raises, naming the entry and what is wrong.
        """
        entry = {"match": "*.self_attn", "module_type": "torch.nn.Linear",
                 "replace_module": {"_target_": "hyper_parallel.models.replacement.module_replacement"}}
        cases = (
            ({**entry, "module_type": None}, "without module_type"),
            ({**entry, "module_type": "no.such.module.Type"}, "cannot import"),
            ({**entry, "replace_module": {"_target_": "torch.nn.no_such_factory"}}, "cannot import"),
        )
        for raw, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                replacement_specs([raw])

    def test_a_fused_layer_keeps_less_than_an_eager_one(self):
        """
        Feature: census_layer with replacements.
        Description: A Qwen3-MoE layer as Transformers builds it, and with
            the recipe's fused RMSNorm, grouped-query attention and grouped
            experts installed, at 256 tokens, plain and under HyperParallel's
            selective policy.
        Expectation: The fused layer keeps less between its passes and less
            under the policy, and its backward holds less; it holds the
            parameters its source held, so the parts count the same.
        """
        config, specs = _qwen3_moe(), _recipe_specs()
        eager = census_layer(config, 0, 256)
        fused = census_layer(config, 0, 256, replacements=specs)
        self.assertLess(fused[0], eager[0])
        self.assertLess(fused[1], eager[1])
        self.assertLess(census_layer(config, 0, 256, selective=True, replacements=specs)[0],
                        census_layer(config, 0, 256, selective=True)[0])
        self.assertEqual(census_parameters(config, 0, replacements=specs), census_parameters(config, 0))

    def test_what_the_fused_kernels_save_is_named_by_op(self):
        """
        Feature: census_saved_ops with replacements.
        Description: The same layer's bytes a token by op, eager and fused.
        Expectation: The fused layer's norms keep less, since its kernel
            keeps one reciprocal root mean square a row where the eager norm
            keeps its upcast input; its attention kernel's saves count toward
            the batched matmuls and its statistics toward the softmax, and
            they sum to what the layer keeps.
        """
        config, specs = _qwen3_moe(), _recipe_specs()
        eager = census_saved_ops(config, 0, 256)
        fused = census_saved_ops(config, 0, 256, replacements=specs)
        self.assertLess(fused["normOp"], eager["normOp"])
        self.assertGreater(fused["softmax"], 0)
        self.assertGreater(fused["attBMM"], 0)
        self.assertAlmostEqual(sum(fused.values()) * 256, census_layer(config, 0, 256, replacements=specs)[0])

    def test_a_replacement_this_host_cannot_build_is_reported(self):
        """
        Feature: census_layer with a replacement whose factory needs a
            library this host lacks, as DeepSeek-V3.2's attention needs a
            native extension.
        Expectation: Every module stays as Transformers built it, the census
            measures it, and a warning names the factories and the import
            that failed.
        """
        from hyper_parallel.models.replacement import ModuleReplacementSpec  # pylint: disable=C0415
        from transformers.models.qwen3_moe import modeling_qwen3_moe  # pylint: disable=C0415
        spec = ModuleReplacementSpec(match=("*.input_layernorm",), factory=_needs_a_library,
                                     module_type=modeling_qwen3_moe.Qwen3MoeRMSNorm)
        config = _qwen3_moe()
        with self.assertLogs("hyper_parallel.auto_parallel._layer_census", "WARNING") as logs:
            self.assertEqual(census_layer(config, 0, 256, replacements=(spec,)), census_layer(config, 0, 256))
        self.assertIn("a_native_extension", logs.output[0])
        self.assertIn("_needs_a_library", logs.output[0])

    def test_the_contracts_stand_in_for_the_kernels(self):
        """
        Feature: npu_contracts.
        Description: The stand-in entered and left, on a host with no
            ``torch_npu``; then a kernel it has no contract for.
        Expectation: While it stands in, ``torch_npu`` is the contracts and
            its fused RMSNorm kernel returns the normalized rows and one
            statistic a row; after, nothing is bound; a kernel with no
            contract raises, naming it.
        """
        with npu_contracts() as stand_in:
            self.assertIs(sys.modules["torch_npu"], stand_in)
            rows = torch.randn(2, 8, dtype=torch.bfloat16)
            out, rstd = stand_in.npu_rms_norm(rows, torch.ones(8, dtype=torch.bfloat16), 1e-6)
            self.assertEqual((tuple(out.shape), tuple(rstd.shape), rstd.dtype), ((2, 8), (2, 1), torch.float32))
            with self.assertRaisesRegex(NotImplementedError, "npu_no_such_kernel"):
                stand_in.npu_no_such_kernel(rows)
        self.assertNotIn("torch_npu", sys.modules)

    def test_a_runs_replacements_reach_its_census(self):
        """
        Feature: context.census with plan_overrides, through the parser.
        Description: A train.yaml of the Qwen3-MoE asking for a census, with
            the recipe's plan_overrides and without them.
        Expectation: ND prices the run with what the census measured of the
            modules the run builds, which keep less than Transformers' own.
        """
        with open(_RECIPE, encoding="utf-8") as handle:
            entries = yaml.safe_load(handle)["plan_overrides"]
        peaks = {}
        for name, plan in (("eager", None), ("fused", entries)):
            with patch(_HF_CONFIG, return_value=_qwen3_moe()), tempfile.TemporaryDirectory() as folder:
                path = _train_yaml(folder, census=True, plan_overrides=plan)
                ccfg = EvaluatorV2(path, framework="hyper_v2", log_level=0).ccfg
                peaks[name] = EvaluatorV2(path, framework="hyper_v2", log_level=0).estimate_peak()
                self.assertIsNotNone(ccfg.census)
        self.assertLess(peaks["fused"], peaks["eager"])


class TestCensusPricing(unittest.TestCase):
    """``context.census`` has the memory model price a layer with its kind's census."""

    def test_the_resolver_states_each_kinds_census(self):
        """
        Feature: resolve_hf_model_spec's census_seq_len.
        Description: The hybrid model resolved twice with a census at 48
            tokens, and a spec of config_overrides alone.
        Expectation: The spec states each kind's record at that length, the
            census runs once; with no checkpoint config there is none.
        """
        model = {"pretrained_model_name_or_path": "local/qwen3_5_moe"}
        with patch(_HF_CONFIG, return_value=_qwen35_text()), patch(
                "hyper_parallel.auto_parallel._hf_model_spec.census_activations", wraps=census_activations) as run:
            spec = resolve_hf_model_spec(model, census_seq_len=48)
            self.assertEqual(resolve_hf_model_spec(model, census_seq_len=48), spec)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(sorted(spec["activations"]), ["full_attention", "linear_attention"])
        self.assertEqual(spec["activations"]["linear_attention"]["seq_length"], 48)
        self.assertEqual(spec["output_activations"]["seq_length"], 48)
        overrides = {"name": "llama", "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                     "intermediate_size": 128, "vocab_size": 128}
        with self.assertLogs("hyper_parallel.auto_parallel._hf_model_spec", "WARNING"):
            plain = resolve_hf_model_spec({"config_overrides": overrides}, census_seq_len=48)
        self.assertNotIn("activations", plain)
        self.assertNotIn("output_activations", plain)

    def test_each_kind_binds_its_record(self):
        """
        Feature: the Hyper parser's context.census.
        Description: The hybrid model parsed with a census and without, and
            its vision-language checkpoint trained with its tower.
        Expectation: With it, the config holds each kind's record and the
            output layer's at the dataset's length, and each kind binds its
            own; the tower holds none; without it, there is none.
        """
        ccfg = _evaluator(_qwen35_text(), census=True).ccfg
        self.assertEqual({record.seq_length for record in ccfg.census.values()}, {4096})
        self.assertEqual(ccfg.output_census.seq_length, 4096)
        self.assertIsNone(ccfg.kind_activations)
        bind_layer_stack(ccfg)
        for kind, fields in ccfg.layer_binding.items():
            self.assertIs(fields["kind_activations"], ccfg.census[kind])
        plain = _evaluator(_qwen35_text()).ccfg
        self.assertIsNone(plain.census)
        self.assertIsNone(plain.output_census)
        towers = _evaluator(_qwen35(_qwen35_text()), target=_IMAGE_TEXT, census=True).ccfg.mm_ccfgs
        self.assertEqual(towers["text"].census, ccfg.census)
        self.assertEqual(towers["text"].output_census, ccfg.output_census)
        for name in ("census", "kind_activations", "output_census"):
            self.assertIsNone(getattr(towers["vision"], name))

    def test_a_layer_is_priced_with_its_kinds_census(self):
        """
        Feature: the census on the memory path.
        Description: The hybrid model without recompute on one stage, and
            fully recomputed on two, each with a census.
        Expectation: Each layer keeps its kind's census bytes for the
            micro-batch's 4096 tokens, and the output layer its own; the
            output layer's backward holds what its census states beyond
            that; the first stage's last layer, which recomputes, holds its
            kind's working set in its backward.
        """
        evaluator = _evaluator(_qwen35_text(), census=True)
        census, output = evaluator.ccfg.census, evaluator.ccfg.output_census
        log = evaluator.estimate_peak_insight()[0]["Node Log"]
        for index, kind in enumerate(("linear_attention", "full_attention")):
            kept = 4096 * (census[kind].saved + census[kind].saved_tp) / 2 ** 20
            self.assertAlmostEqual(log[(0, 0, index, "N")]["_activ"], kept, delta=1)
        kept = output.saved + output.saved_tp
        self.assertAlmostEqual(log[(0, 0, "", "O")]["_activ"], 4096 * kept / 2 ** 20, delta=1)
        held = output.working + output.working_tp - kept
        self.assertAlmostEqual(log[(0, 0, "G_", "O")]["_activ"], 4096 * held / 2 ** 20, delta=1)
        log = _evaluator(_qwen35_text(), mode="full", pp=2, census=True).estimate_peak_insight()[0]["Node Log"]
        working = [value["_activ"] for key, value in log.items() if str(key[2]).startswith("rec_")]
        linear = census["linear_attention"]
        self.assertAlmostEqual(working[0], 4096 * (linear.working + linear.working_tp) / 2 ** 20, delta=1)

    def test_a_selective_layer_is_priced_with_its_kinds_census(self):
        """
        Feature: the census on the memory path, under HyperParallel's
            selective activation checkpointing.
        Description: The hybrid model checkpointed selectively on one stage,
            with a census.
        Expectation: Each layer keeps what its kind keeps under the
            trainer's selective policy for the micro-batch's 4096 tokens.
        """
        evaluator = _evaluator(_qwen35_text(), mode="selective", census=True)
        census = evaluator.ccfg.census
        log = evaluator.estimate_peak_insight()[0]["Node Log"]
        for index, kind in enumerate(("linear_attention", "full_attention")):
            kept = 4096 * (census[kind].selective + census[kind].selective_tp) / 2 ** 20
            self.assertAlmostEqual(log[(0, 0, index, "S")]["_activ"], kept, delta=1)


if __name__ == "__main__":
    unittest.main()
