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
"""Tests for the layer stack: how a config's layers get their kinds, as data."""
import unittest

from hyper_parallel.auto_parallel._hf_model_spec import resolve_hf_model_spec
from hyper_parallel.auto_parallel._layer_stack import (
    LinearAttentionDims,
    derive_layers,
    resolve_layers,
    spec_layer_stack,
)
from hyper_parallel.auto_parallel._model_spec import LayerGroup, ModelSpec, ModelSpecError, OpCounts
from hyper_parallel.auto_parallel._op_profiles import LayerKind, OpProfile, load_op_profile

_COUNTS = {
    "attMM": 4, "attBMM": 2, "ffMM": 3, "softmax": 1, "dropout": 0,
    "normOp": 2, "gather": 4, "headCast": 1, "ffAct": 1, "linrec": 0,
}
_LINEAR = {
    "linear_num_key_heads": 16,
    "linear_key_head_dim": 128,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
}


def _hybrid_profile() -> OpProfile:
    """A two-kind hybrid profile shaped like Qwen3.5's, full attention by default."""
    full = LayerKind("full_attention", OpCounts.from_dict(_COUNTS))
    linear = LayerKind(
        "linear_attention",
        OpCounts.from_dict(dict(_COUNTS, attBMM=0, softmax=0, headCast=0, linrec=1)),
        attention="linear",
    )
    return OpProfile("hybrid", {kind.name: kind for kind in (full, linear)}, "full_attention")


def _groups(*groups) -> tuple:
    """Groups from ``(kind, count)`` or ``(kind, count, mtp)`` tuples."""
    return tuple(LayerGroup(*group) for group in groups)


class TestDerivedFromLayerTypes(unittest.TestCase):
    """A config that names each layer's kind gets exactly that stack."""

    def test_interleaved_kinds_group_by_run(self):
        """
        Feature: layer_types.
        Description: Three linear layers, one full, two linear, one full.
        Expectation: One group per run, in model order.
        """
        types = ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 2 + ["full_attention"]
        layers = derive_layers(_hybrid_profile(), 7, layer_types=types)
        self.assertEqual(layers, _groups(
            ("linear_attention", 3), ("full_attention", 1), ("linear_attention", 2), ("full_attention", 1),
        ))

    def test_mtp_repeats_the_last_body_kind(self):
        """
        Feature: MTP layers.
        Description: A hybrid body ending in full attention, with one MTP layer.
        Expectation: The MTP layer is a full-attention layer, marked as MTP.
        """
        types = ["linear_attention", "full_attention"]
        layers = derive_layers(_hybrid_profile(), 2, mtp_depth=1, layer_types=types)
        self.assertEqual(layers[-1], LayerGroup("full_attention", 1, mtp=True))

    def test_a_longer_list_is_cut_to_the_body(self):
        """
        Feature: layer_types.
        Description: A cropped model keeps the checkpoint's full layer_types.
        Expectation: Only the first num_hidden_layers entries count.
        """
        types = ["linear_attention", "full_attention"] * 4
        layers = derive_layers(_hybrid_profile(), 3, layer_types=types)
        self.assertEqual(sum(group.count for group in layers), 3)
        self.assertEqual(layers[-1], LayerGroup("linear_attention", 1))

    def test_a_shorter_list_is_refused(self):
        """
        Feature: layer_types.
        Description: layer_types names fewer layers than the body has.
        Expectation: Refused, naming both counts.
        """
        with self.assertRaises(ModelSpecError) as ctx:
            derive_layers(_hybrid_profile(), 4, layer_types=["full_attention"] * 3)
        self.assertIn("3", str(ctx.exception))
        self.assertIn("4", str(ctx.exception))

    def test_kinds_the_profile_lacks_keep_the_default_and_say_so(self):
        """
        Feature: layer_types of another family.
        Description: Sliding-window attention, which no profile declares.
        Expectation: Every layer takes the default kind, with a warning.
        """
        types = ["sliding_attention", "full_attention"] * 2
        with self.assertLogs("hyper_parallel.auto_parallel._layer_stack", "WARNING"):
            layers = derive_layers(load_op_profile("qwen"), 4, layer_types=types)
        self.assertEqual(layers, _groups(("decoder", 4)))


class TestDerivedFromTheProfile(unittest.TestCase):
    """Without layer_types, the profile's kinds imply the stack."""

    def test_dense_prefix_then_moe(self):
        """
        Feature: first_k_dense_replace.
        Description: DeepSeek-V3: 61 layers, the first 3 dense, one MTP layer.
        Expectation: 3 dense, 58 MoE, then one MoE MTP layer.
        """
        layers = derive_layers(load_op_profile("deepseek"), 61, mtp_depth=1, first_k_dense=3)
        self.assertEqual(layers, _groups(("dense", 3), ("moe", 58), ("moe", 1, True)))

    def test_no_dense_prefix_is_all_moe(self):
        """
        Feature: first_k_dense_replace.
        Description: No dense prefix declared.
        Expectation: One MoE group; an empty dense group is left out.
        """
        self.assertEqual(derive_layers(load_op_profile("deepseek"), 8), _groups(("moe", 8)))

    def test_all_dense_puts_mtp_on_a_dense_layer(self):
        """
        Feature: MTP layers.
        Description: Every body layer dense, as the old hook's n_moe == 0 case.
        Expectation: The MTP layer is dense too.
        """
        layers = derive_layers(load_op_profile("deepseek"), 4, mtp_depth=1, first_k_dense=4)
        self.assertEqual(layers, _groups(("dense", 4), ("dense", 1, True)))

    def test_encoder_then_decoder_halves(self):
        """
        Feature: encoder-decoder stack.
        Description: t5 with 24 layers.
        Expectation: 12 encoder layers, then 12 decoder layers.
        """
        self.assertEqual(derive_layers(load_op_profile("t5"), 24), _groups(("encoder", 12), ("decoder", 12)))

    def test_an_odd_encoder_decoder_depth_is_refused(self):
        """
        Feature: encoder-decoder stack.
        Description: 23 layers cannot split into two equal halves.
        Expectation: Refused at parse time rather than failing in the estimator.
        """
        with self.assertRaises(ModelSpecError):
            derive_layers(load_op_profile("t5"), 23)

    def test_one_kind_runs_throughout(self):
        """
        Feature: single-kind families.
        Description: A qwen decoder with 32 layers and one MTP layer.
        Expectation: One body group and one MTP group of the same kind.
        """
        layers = derive_layers(load_op_profile("qwen"), 32, mtp_depth=1)
        self.assertEqual(layers, _groups(("decoder", 32), ("decoder", 1, True)))


class TestResolve(unittest.TestCase):
    """A stack is checked against its profile before anything prices it."""

    def test_an_unknown_kind_is_refused_by_name(self):
        """
        Feature: resolve_layers.
        Description: A group names a kind the family does not have.
        Expectation: Refused, naming the kind and the ones the profile has.
        """
        with self.assertRaises(ModelSpecError) as ctx:
            resolve_layers("deepseek", _groups(("decoder", 4)))
        self.assertIn("decoder", str(ctx.exception))
        self.assertIn("moe", str(ctx.exception))

    def test_kinds_carry_their_flavours(self):
        """
        Feature: resolve_layers.
        Description: DeepSeek's two kinds differ in their feed-forward only.
        Expectation: Dense and MoE flavours, identical op counts.
        """
        stack = resolve_layers("deepseek", _groups(("dense", 3), ("moe", 5)))
        dense, moe = stack.distinct_kinds()
        self.assertEqual((dense.ffn, moe.ffn), ("dense", "moe"))
        self.assertEqual(dense.ops, moe.ops)
        self.assertEqual(len(stack.kinds()), 8)

    def test_declared_ops_replace_the_counts(self):
        """
        Feature: declared ops.
        Description: A spec declares its own counts for the qwen decoder.
        Expectation: The resolved kind prices those counts.
        """
        ops = {"decoder": OpCounts.from_dict(dict(_COUNTS, gather=6))}
        stack = resolve_layers("qwen", _groups(("decoder", 2)), ops=ops)
        self.assertEqual(stack.kinds()[0].ops.gather, 6)

    def test_a_round_trip_through_layers(self):
        """
        Feature: serialised form.
        Description: A resolved stack goes back to the groups it came from.
        Expectation: The same groups.
        """
        layers = _groups(("dense", 1), ("moe", 2), ("moe", 1, True))
        self.assertEqual(resolve_layers("deepseek", layers).to_layers(), layers)


class TestLinearDimensions(unittest.TestCase):
    """A linear-attention layer is priced on dimensions the model must declare."""

    def test_all_five_or_none(self):
        """
        Feature: LinearAttentionDims.
        Description: Only some of the five linear fields are set.
        Expectation: Refused, naming the missing ones.
        """
        self.assertIsNone(LinearAttentionDims.from_fields({}))
        partial = dict(_LINEAR)
        del partial["linear_conv_kernel_dim"]
        with self.assertRaises(ModelSpecError) as ctx:
            LinearAttentionDims.from_fields(partial)
        self.assertIn("linear_conv_kernel_dim", str(ctx.exception))

    def test_a_non_positive_dimension_is_refused(self):
        """
        Feature: LinearAttentionDims.
        Description: A head count of zero.
        Expectation: Refused by name.
        """
        with self.assertRaises(ModelSpecError) as ctx:
            LinearAttentionDims.from_fields(dict(_LINEAR, linear_num_value_heads=0))
        self.assertIn("linear_num_value_heads", str(ctx.exception))


class TestSpecStack(unittest.TestCase):
    """A spec's stack, stated or implied, is checked when the spec is resolved."""

    @staticmethod
    def _spec(**fields) -> dict:
        spec = {
            "name": "unit", "hidden_size": 1024, "num_hidden_layers": 4,
            "num_attention_heads": 8, "vocab_size": 32000,
        }
        spec.update(fields)
        return spec

    def test_an_implied_stack(self):
        """
        Feature: spec_layer_stack.
        Description: A DeepSeek-shaped spec that states no layers.
        Expectation: The stack its first_k_dense_replace and mtp_depth imply.
        """
        spec = ModelSpec.from_dict(self._spec(
            arch="deepseek", num_experts=8, moe_intermediate_size=256,
            first_k_dense_replace=1, mtp_depth=1,
        ))
        self.assertEqual(spec_layer_stack(spec).to_layers(), _groups(("dense", 1), ("moe", 3), ("moe", 1, True)))

    def test_a_stated_stack_is_checked_at_parse_time(self):
        """
        Feature: resolve_hf_model_spec.
        Description: config_overrides states a stack with a kind the family lacks.
        Expectation: Refused while the spec is resolved, not in the estimator.
        """
        overrides = self._spec(arch="qwen", layers=[{"kind": "moe", "count": 4}])
        with self.assertRaises(ModelSpecError):
            resolve_hf_model_spec({"name": "unit", "config_overrides": overrides})

    def test_a_stated_stack_round_trips(self):
        """
        Feature: resolve_hf_model_spec.
        Description: config_overrides states a valid stack.
        Expectation: The resolved spec carries the same groups.
        """
        layers = [{"kind": "decoder", "count": 4}, {"kind": "decoder", "count": 1, "mtp": True}]
        overrides = self._spec(arch="qwen", mtp_depth=1, layers=layers)
        spec = resolve_hf_model_spec({"name": "unit", "config_overrides": overrides})
        self.assertEqual(spec["layers"], layers)


if __name__ == "__main__":
    unittest.main()
