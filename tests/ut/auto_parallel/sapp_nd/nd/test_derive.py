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
"""Tests for the derived fields of a cost-model config.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_derive.py -v
"""
import unittest
from types import SimpleNamespace
from typing import Any

from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import (
    HYPER_SELECTIVE_REC_OP,
    derive,
    derive_embedding_sharding,
    derive_flash_attention_factor,
    derive_recompute_switches,
)


def _config(**fields: Any) -> SimpleNamespace:
    """A dense config holding the facts derive reads, *fields* overriding them."""
    facts = {
        "d": 4, "t": 2, "p": 1, "cp": 1, "ep": 1, "etp": 0, "n_exp": 1, "hff_exp": 64,
        "has_op": True, "has_grad_shard": False, "os_max_shard": 8,
        "vocab_emb_dp": False, "emb_dp_sharded": True, "recompute_slice_activation": False,
        "sel_rec": False, "sel_comm_rec": False, "sel_rec_rule": "hyperparallel",
        "has_fa": True, "sp": 1, "s": 4096, "a": 32,
    }
    facts.update(fields)
    return SimpleNamespace(**facts)


class TestDerive(unittest.TestCase):
    """derive computes each field from the facts a parser states."""

    def test_embedding_sharding(self):
        """
        Feature: derive_embedding_sharding.
        Description: The embedding facts the four parsers state, at d=4 and t=2.
        Expectation: MindFormers and Hyper split the table over d, and over t
            too unless the vocabulary embedding is data parallel without
            pipelining; MindSpeed splits it over t * d and TOML over t alone.
        """
        cases = [
            ({"vocab_emb_dp": True, "p": 1}, 4),
            ({"vocab_emb_dp": True, "p": 2}, 8),
            ({"vocab_emb_dp": False}, 8),
            ({"vocab_emb_dp": False, "emb_dp_sharded": False}, 2),
        ]
        for fields, expected in cases:
            ccfg = _config(**fields)
            derive_embedding_sharding(ccfg)
            self.assertEqual(ccfg.shard_embed, expected, f"{fields}: shard_embed={ccfg.shard_embed}")

    def test_mindformers_recompute_switches(self):
        """
        Feature: derive_recompute_switches.
        Description: MindFormers' selective recompute under flash attention and
            sequence parallelism, then its communication recompute alone.
        Expectation: The head cast, norm and activation are recomputed and the
            attention kernels kept; the all-gather is recomputed only by
            select_comm_recompute.
        """
        ccfg = _config(sel_rec_rule="mindformers", sel_rec=True, sp=2)
        derive_recompute_switches(ccfg)
        expected = {"attBMM": 1, "headCast": 0, "dropout": 1, "softmax": 1, "normOp": 0, "gather": 1, "ffAct": 0}
        self.assertEqual(vars(ccfg.rec_op), expected, f"rec_op={vars(ccfg.rec_op)}")

        ccfg = _config(sel_rec_rule="mindformers", sel_comm_rec=[1, 0], sp=2)
        derive_recompute_switches(ccfg)
        expected = {**dict.fromkeys(HYPER_SELECTIVE_REC_OP, 1), "gather": 0}
        self.assertEqual(vars(ccfg.rec_op), expected, f"rec_op={vars(ccfg.rec_op)}")

    def test_unknown_recompute_rule_is_refused(self):
        """
        Feature: derive_recompute_switches.
        Description: A config naming no known selective recompute rule.
        Expectation: ValueError rather than a guess.
        """
        with self.assertRaises(ValueError):
            derive_recompute_switches(_config(sel_rec_rule="megatron"))

    def test_flash_attention_factor(self):
        """
        Feature: derive_flash_attention_factor.
        Description: With and without flash attention, and a model with no heads.
        Expectation: s / a under flash attention, s otherwise.
        """
        for fields, expected in (({}, 128.0), ({"has_fa": False}, 4096), ({"a": 0}, 4096)):
            ccfg = _config(**fields)
            derive_flash_attention_factor(ccfg)
            self.assertEqual(ccfg.s_fa, expected, f"{fields}: s_fa={ccfg.s_fa}")

    def test_derive_sets_every_derived_field(self):
        """
        Feature: derive.
        Description: One call on a dense config that slices the recompute
            input and recomputes selectively.
        Expectation: The expert degrees, sharding factors, communication
            flags, recompute switches and flash attention factor are all set.
        """
        ccfg = _config(recompute_slice_activation=True, sel_rec=True)
        derive(ccfg)
        got = (ccfg.t_exp, ccfg.d_exp, ccfg.comm_d_non_exp, ccfg.s_fa)
        self.assertEqual(got, (2, 4, 2, 128.0), f"t_exp, d_exp, comm_d_non_exp, s_fa={got}")
        got = (ccfg.shard_embed, ccfg.shard_recompute_input, ccfg.shard_output_activ)
        self.assertEqual(got, (8, 2, 1), f"shard_embed, shard_recompute_input, shard_output_activ={got}")
        self.assertEqual(vars(ccfg.rec_op), HYPER_SELECTIVE_REC_OP, f"rec_op={vars(ccfg.rec_op)}")


if __name__ == "__main__":
    unittest.main()
