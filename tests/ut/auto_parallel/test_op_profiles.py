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
"""Tests for the op profiles and how a spec settles the one it is priced with."""
import unittest

from hyper_parallel.auto_parallel._hf_model_spec import resolve_hf_model_spec
from hyper_parallel.auto_parallel._model_spec import ModelSpecError, OpCounts
from hyper_parallel.auto_parallel._op_profiles import (
    DEFAULT_ARCH,
    infer_arch,
    known_archs,
    load_op_profile,
    resolve_ops,
)


def _counts(**overrides) -> dict:
    """The default decoder's op counts, with *overrides* applied."""
    counts = {
        "attMM": 4, "attBMM": 2, "ffMM": 3, "ffBMM": 0, "softmax": 1,
        "dropout": 0, "normOp": 2, "gather": 4, "headCast": 1, "ffAct": 1,
    }
    counts.update(overrides)
    return counts


def _overrides(name: str = "unit", **extra) -> dict:
    """A ``model`` section the offline config_overrides path can resolve."""
    overrides = {
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "vocab_size": 32000,
    }
    overrides.update(extra)
    return {"name": name, "config_overrides": overrides}


class TestProfiles(unittest.TestCase):
    """Every family's op counts are a file, and every file is a valid vector."""

    def test_the_families(self):
        """The seven families the arch hooks knew, the default and the tower."""
        self.assertEqual(set(known_archs()), {
            "default", "llama2", "mixtral", "t5", "pangualpha",
            "deepseek", "qwen", "cm", "vision",
        })

    def test_every_profile_loads(self):
        """Each profile declares at least one kind, each a full OpCounts."""
        for arch in known_archs():
            with self.subTest(arch=arch):
                profile = load_op_profile(arch)
                self.assertEqual(profile.arch, arch)
                self.assertTrue(profile.kinds)
                for counts in profile.kinds.values():
                    self.assertIsInstance(counts, OpCounts)

    def test_unknown_arch_raises_naming_the_profiles(self):
        """A misspelt arch is refused, and the message lists the real ones."""
        with self.assertRaises(ModelSpecError) as ctx:
            load_op_profile("qwne")
        self.assertIn("qwne", str(ctx.exception))
        self.assertIn("qwen", str(ctx.exception))

    def test_unknown_kind_raises_naming_the_kinds(self):
        """Asking a profile for a kind it lacks names the kinds it has."""
        self.assertEqual(load_op_profile("t5").counts("decoder").attMM, 8)
        with self.assertRaises(ModelSpecError) as ctx:
            load_op_profile("t5").counts("block")
        self.assertIn("encoder", str(ctx.exception))


class TestInferArch(unittest.TestCase):
    """A free-text model name maps to the family the cost model always chose."""

    def test_names_in_use(self):
        """The model names the repo's configs and tests carry."""
        cases = {
            "deepseekV3": "deepseek",
            "deepseek_v3": "deepseek",
            "qwen2_72b": "qwen",
            "Qwen3": "qwen",
            "qwen3_moe": "qwen",
            "qwen3_vl_moe": "qwen",
            "mixtral-8x7b": "mixtral",
            "llama2_7b": "llama2",
            "pangualpha_2_6b": "pangualpha",
            "t5_small": "t5",
            "cm_llama_moe": "cm",
        }
        for name, arch in cases.items():
            with self.subTest(name=name):
                self.assertEqual(infer_arch(name), arch)

    def test_first_family_in_order_wins(self):
        """The order is load-bearing: deepseek is tried before qwen."""
        self.assertEqual(infer_arch("deepseek_qwen_distill"), "deepseek")
        self.assertEqual(infer_arch("llama2_t5_hybrid"), "llama2")

    def test_no_match_is_the_default_and_says_so(self):
        """A name nothing claims is priced as the default, with a warning."""
        with self.assertLogs("hyper_parallel.auto_parallel._op_profiles", "WARNING"):
            self.assertEqual(infer_arch("llama"), DEFAULT_ARCH)


class TestResolveOps(unittest.TestCase):
    """Declared counts replace a profile's, but only kind for kind."""

    def test_profile_counts_when_nothing_is_declared(self):
        """Without declared ops the profile's kinds are returned."""
        self.assertEqual(set(resolve_ops("t5")), {"encoder", "decoder"})

    def test_declared_counts_win(self):
        """Counts the spec declares are the ones it is priced with."""
        declared = {"decoder": OpCounts.from_dict(_counts(softmax=0))}
        self.assertEqual(resolve_ops("qwen", declared)["decoder"].softmax, 0)

    def test_declared_kinds_must_match_the_profile(self):
        """Counts for a kind the family never assigns would be dropped."""
        declared = {"block": OpCounts.from_dict(_counts())}
        with self.assertRaises(ModelSpecError) as ctx:
            resolve_ops("qwen", declared)
        self.assertIn("block", str(ctx.exception))
        self.assertIn("decoder", str(ctx.exception))


class TestResolvedSpecStatesItsArch(unittest.TestCase):
    """The producer writes the profile it settled on into the spec."""

    def test_inferred_arch_is_written_out(self):
        """A spec that declares no arch gets the one its name implies."""
        spec = resolve_hf_model_spec(_overrides(name="qwen3_moe"))
        self.assertEqual(spec["arch"], "qwen")
        self.assertNotIn("ops", spec)

    def test_declared_arch_wins_over_the_name(self):
        """An explicit arch is kept even when the name implies another."""
        spec = resolve_hf_model_spec(_overrides(name="qwen3_moe", arch="default"))
        self.assertEqual(spec["arch"], "default")

    def test_unknown_declared_arch_raises(self):
        """A misspelt arch fails at parse time instead of pricing as default."""
        with self.assertRaises(ModelSpecError):
            resolve_hf_model_spec(_overrides(arch="qwne"))

    def test_declared_ops_round_trip(self):
        """Counts declared in config_overrides come back in the spec."""
        ops = {"decoder": _counts(softmax=0)}
        spec = resolve_hf_model_spec(_overrides(arch="default", ops=ops))
        self.assertEqual(spec["ops"], ops)

    def test_declared_ops_for_the_wrong_kinds_raise(self):
        """Counts that no layer of the family would use are refused."""
        with self.assertRaises(ModelSpecError):
            resolve_hf_model_spec(_overrides(arch="t5", ops={"decoder": _counts()}))


if __name__ == "__main__":
    unittest.main()
