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
"""Tests for verify mode: the parameters ND prices beside the census's, and run_nd -V."""
import os
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from hyper_parallel.auto_parallel._layer_census import census_final_norm, census_parameters
from hyper_parallel.auto_parallel.sapp_nd.nd.verify import report, verify_parameters

_HF_CONFIG = "hyper_parallel.auto_parallel._hf_model_spec._get_hf_config"
_RUN_ND = "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd"


def _qwen35_text(tie: bool = False):
    """A two-layer Qwen3.5-MoE text config of width 64, one layer of each attention kind."""
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (  # pylint: disable=C0415
        Qwen3_5MoeTextConfig,
    )
    return Qwen3_5MoeTextConfig.from_dict({
        "hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32, "linear_num_key_heads": 2, "linear_key_head_dim": 16,
        "linear_num_value_heads": 4, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
        "vocab_size": 128, "max_position_embeddings": 256, "layer_types": ["linear_attention", "full_attention"],
        "tie_word_embeddings": tie,
    })


def _deepseek_v3(q_lora_rank=32):
    """A two-layer DeepSeek-V3 config of width 64, one dense layer and one of 4 experts, its MLA heads 12, 4 and 8 wide."""
    from transformers import DeepseekV3Config  # pylint: disable=C0415
    return DeepseekV3Config(
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        q_lora_rank=q_lora_rank, kv_lora_rank=16, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=8,
        n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=16, n_shared_experts=1,
        first_k_dense_replace=1, intermediate_size=32, vocab_size=128, n_group=1, topk_group=1)


def _train_yaml(folder: str) -> str:
    """A train yaml of the model on two ranks at DP shard 2."""
    config = {
        "model": {"_target_": "hyper_parallel.models._transformers.HyperAutoModelForCausalLM.from_pretrained",
                  "pretrained_model_name_or_path": "local/qwen3_5_moe", "torch_dtype": "bfloat16"},
        "training": {"global_batch_size": 4, "micro_batch_size": 1},
        "accelerator": {"tp_size": 1, "pp_size": 1, "ep_size": 1, "cp_size": 1},
        "fsdp_config": {"dp_shard_size": 2},
        "activation_checkpoint": {"mode": "off"},
        "dataset": {"data_transform": {"max_seq_len": 4096}},
        "context": {"device_num": 2},
    }
    path = os.path.join(folder, "train.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return path


def _verify(config):
    """verify_parameters of the train yaml of *config*."""
    with patch(_HF_CONFIG, return_value=config), tempfile.TemporaryDirectory() as folder:
        return verify_parameters(_train_yaml(folder))


class TestVerifyParameters(unittest.TestCase):
    """Verify mode sets what ND prices of each part beside the census's."""

    def test_each_part_beside_its_census(self):
        """
        Feature: verify_parameters.
        Description: The hybrid model, one layer of each kind, untied.
        Expectation: A row per part of each kind, whose census is the
            layer's; the embedding's and the output layer's tables, the
            output layer's final norm with it; and the model's whole, the
            sum of the rest.
        """
        config = _qwen35_text()
        rows = _verify(config)
        by = {(row.where, row.part): row for row in rows}
        for index, kind in enumerate(("linear_attention", "full_attention")):
            for part, count in census_parameters(config, index).items():
                self.assertEqual(by[(f"{kind} x1", part)].census, count)
            self.assertGreater(by[(f"{kind} x1", "attention")].nd, 0)
        table = 128 * 64
        self.assertEqual((by[("embedding", "table")].nd, by[("embedding", "table")].census), (table, table))
        self.assertEqual(by[("output", "table, norm")].census, table + census_final_norm(config))
        total = by[("model", "total")]
        self.assertEqual(total.census, sum(row.count * row.census for row in rows[:-1]))
        self.assertEqual(total.nd, sum(row.count * row.nd for row in rows[:-1]))
        self.assertEqual(len(report(rows)), len(rows) + 1)

    def test_a_tied_table_is_counted_at_the_output(self):
        """
        Feature: verify_parameters of a tied model.
        Description: The hybrid model with its embedding and output tied,
            on one stage.
        Expectation: ND prices the one table at the output, and so does the
            census: the embedding holds none.
        """
        by = {(row.where, row.part): row for row in _verify(_qwen35_text(tie=True))}
        self.assertEqual((by[("embedding", "table")].nd, by[("embedding", "table")].census), (0, 0))
        self.assertGreater(by[("output", "table, norm")].census, 128 * 64)

    def test_mla_attention_matches_its_census(self):
        """
        Feature: verify_parameters of an MLA model.
        Description: DeepSeek-V3's attention at width 64, with a query
            latent and without, its non-rotary key heads wider than its
            value heads.
        Expectation: ND prices exactly the attention Transformers builds,
            in the dense layer and in the expert one.
        """
        for q_lora_rank in (32, None):
            rows = {(row.where, row.part): row for row in _verify(_deepseek_v3(q_lora_rank))}
            for kind in ("dense x1", "moe x1"):
                self.assertEqual(rows[(kind, "attention")].nd, rows[(kind, "attention")].census, kind)

    def test_a_train_yaml_without_a_checkpoint_is_refused(self):
        """
        Feature: verify_parameters' input.
        Description: A yaml that names no Transformers checkpoint.
        Expectation: A ValueError naming the file.
        """
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "mf.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump({"model": {"model_config": {"hidden_size": 64}}}, handle)
            with self.assertRaisesRegex(ValueError, "mf.yaml"):
                verify_parameters(path)


class TestRunNdVerify(unittest.TestCase):
    """run_nd -V prints the verify report and exits."""

    def test_the_flag_prints_the_report(self):
        """
        Feature: run_nd -V.
        Description: The hybrid model's train yaml under -f hyper_v2, and
            under the default framework.
        Expectation: With hyper_v2, the report's header and a row per part,
            and exit 0; otherwise the parser refuses the flag.
        """
        with tempfile.TemporaryDirectory() as folder, patch(_HF_CONFIG, return_value=_qwen35_text()):
            path = _train_yaml(folder)
            argv = ["run_nd.py", "-f", "hyper_v2", "-y", path, "-V", "-o", folder]
            with patch.object(sys, "argv", argv), self.assertLogs("ND", level="CRITICAL") as logs, \
                    self.assertRaises(SystemExit) as done:
                runpy.run_module(_RUN_ND, run_name="__main__")
            self.assertEqual(done.exception.code, 0)
            self.assertIn("census", logs.output[0])
            self.assertTrue(any("linear_attention x1" in line and "router" in line for line in logs.output))
            with patch.object(sys, "argv", ["run_nd.py", "-y", path, "-V"]), self.assertRaises(SystemExit) as done:
                runpy.run_module(_RUN_ND, run_name="__main__")
            self.assertEqual(done.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
