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
"""Tests for the demo's cropped Qwen3.5-MoE builder and the gated delta rule it selects."""
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from examples.training_demo import cropped_qwen3_5_moe
from hyper_parallel.trainer.config.resolver import ConfigResolutionError, resolve_component
from hyper_parallel.trainer.config.target import Target

_DEMO_YAML = (Path(__file__).resolve().parents[4]
              / "examples" / "training_demo" / "train_qwen3_5_moe.yaml")


def _text_config():
    """A two-layer Qwen3.5-MoE text config, as a released config directory would hold it."""
    return Qwen3_5MoeTextConfig.from_dict({
        "hidden_size": 32, "num_hidden_layers": 4, "num_attention_heads": 2, "num_key_value_heads": 1,
        "head_dim": 16, "num_experts": 4, "num_experts_per_tok": 2, "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 16, "vocab_size": 64,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    })


class TestCroppedQwen35MoeDeltaRule(unittest.TestCase):
    """The demo states which gated delta rule its layers run, and the builder honours it."""

    def _model_section(self, **changes):
        """The demo's model section, with *changes* applied, resolved as the trainer resolves it."""
        raw = yaml.safe_load(_DEMO_YAML.read_text(encoding="utf-8"))["model"]
        raw.update(changes)
        return resolve_component(raw, annotation=Target, path="model")

    def test_the_demo_runs_hyperparallels_rule(self):
        """The demo yaml resolves, and it asks for HyperParallel's rule."""
        target = self._model_section()
        self.assertEqual(target.delta_rule, "hyperparallel",
                         f"the demo states delta_rule={target.delta_rule!r}")

    def test_a_timing_can_ask_for_transformers_rule(self):
        """The run before the change is one key away, and a misspelt rule is refused when the config resolves."""
        self.assertEqual(self._model_section(delta_rule="transformers").delta_rule, "transformers",
                         "delta_rule=transformers did not resolve as stated")
        with self.assertRaises(ConfigResolutionError):
            self._model_section(delta_rule="hyper_parallel")

    def test_the_builder_installs_the_rule_it_is_asked_for(self):
        """hyperparallel switches the built model's layers; transformers leaves them as built."""
        built = object()
        for rule, calls in (("hyperparallel", 1), ("transformers", 0)):
            with patch.object(cropped_qwen3_5_moe.AutoConfig, "from_pretrained", return_value=_text_config()), \
                    patch.object(cropped_qwen3_5_moe.HyperAutoModelForCausalLM, "from_config",
                                 return_value=built), \
                    patch.object(cropped_qwen3_5_moe, "install_delta_rule", return_value=3) as install:
                model = cropped_qwen3_5_moe.build_cropped_qwen3_5_moe("local/qwen3_5_moe", num_hidden_layers=4,
                                                                      delta_rule=rule)
            self.assertIs(model, built, f"delta_rule={rule} returned {model!r}, not the built model")
            self.assertEqual(install.call_count, calls,
                             f"delta_rule={rule}: install called {install.call_count} times, expected {calls}")
            if calls:
                install.assert_called_once_with(built)

    def test_an_unknown_rule_is_refused_before_any_build(self):
        """A rule the builder does not know raises before the configuration is read."""
        with patch.object(cropped_qwen3_5_moe.AutoConfig, "from_pretrained") as read:
            with self.assertRaisesRegex(ValueError, "delta_rule must be one of"):
                cropped_qwen3_5_moe.build_cropped_qwen3_5_moe("local/qwen3_5_moe", delta_rule="triton")
        self.assertEqual(read.call_count, 0, f"the configuration was read {read.call_count} times")


if __name__ == "__main__":
    unittest.main()
