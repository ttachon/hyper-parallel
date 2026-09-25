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
"""Tests for the layer census and the activations it states in the model spec."""
import unittest

import yaml

from hyper_parallel.auto_parallel._layer_census import census_activations, census_layer, tp_config
from hyper_parallel.auto_parallel._model_spec import KindActivations, ModelSpec, ModelSpecError


def _qwen35_text():
    """A two-layer Qwen3.5-MoE text config, one layer of each attention kind."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # pylint: disable=C0415
        Qwen3_5MoeTextConfig,
    )
    return Qwen3_5MoeTextConfig(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
        linear_num_key_heads=2, linear_key_head_dim=16, linear_num_value_heads=4, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, vocab_size=128, max_position_embeddings=256,
        layer_types=["linear_attention", "full_attention"],
    )


_STACK = [{"kind": "linear_attention", "count": 1}, {"kind": "full_attention", "count": 1}]


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

    def test_each_kind_gets_its_record(self):
        """
        Feature: census_activations.
        Description: The census of a stack of one linear-attention layer and
            one full-attention layer.
        Expectation: A record per kind, at the census's length, splitting
            each layer's bytes between what TP splits and what it does not.
        """
        got = census_activations(_qwen35_text(), _STACK, 64)
        self.assertEqual(sorted(got), ["full_attention", "linear_attention"])
        for kind, index in (("linear_attention", 0), ("full_attention", 1)):
            record = got[kind]
            saved, working = census_layer(_qwen35_text(), index, 64)
            self.assertEqual(record.seq_length, 64)
            self.assertAlmostEqual((record.saved + record.saved_tp) * 64, saved)
            self.assertAlmostEqual((record.working + record.working_tp) * 64, working)
            self.assertGreater(record.saved_tp, 0)


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
        spec = ModelSpec.from_dict(_spec(linear_attention=record, full_attention=record))
        self.assertIsInstance(spec.activations["linear_attention"], KindActivations)
        self.assertEqual(ModelSpec.from_dict(yaml.safe_load(yaml.safe_dump(spec.to_dict()))), spec)

    def test_a_record_is_checked(self):
        """
        Feature: KindActivations.from_dict and ModelSpec.validate.
        Description: A negative size, a missing field, and a kind no layer of
            the stack is.
        Expectation: Each raises, naming what is wrong.
        """
        record = {"saved": 1.0, "saved_tp": 1.0, "working": 1.0, "working_tp": 1.0, "seq_length": 64}
        with self.assertRaisesRegex(ModelSpecError, "negative"):
            ModelSpec.from_dict(_spec(linear_attention=dict(record, saved=-1.0)))
        with self.assertRaisesRegex(ModelSpecError, "lacks"):
            ModelSpec.from_dict(_spec(linear_attention={"saved": 1.0}))
        with self.assertRaisesRegex(ModelSpecError, "decoder"):
            ModelSpec.from_dict(_spec(decoder=record))


if __name__ == "__main__":
    unittest.main()
