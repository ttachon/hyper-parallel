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
"""Tests for the performance estimate: its entry point, and the per-op compute
loads it prices.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/perf_estimation/test_estimate.py -v
"""
import os
import unittest
from types import SimpleNamespace
from typing import Any, Dict

# The package has an import cycle that only the memory estimator's import order
# settles; perf_estimation.estimate cannot be the first module a process loads.
import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2  # pylint: disable=unused-import
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import (
    MOE_DISPATCH,
    _flavour_tables,
    estimate_performance,
    op_table,
)

DEEPSEEK_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nd", "deepseek.yaml"
)


def _plain_values(ccfg: CostModelConfig) -> Dict[str, Any]:
    """The config's plain fields, the ones layer hooks overwrite."""
    return {name: value for name, value in vars(ccfg).items()
            if isinstance(value, (bool, int, float, str))}


def _cfg(s: int) -> SimpleNamespace:
    """A dense, non-MLA config at sequence length *s*."""
    return SimpleNamespace(a=8, n_kv=8, dh=0, b=1, s=s, h=512, hff=1024, t=2, sp=2, cp=1, dc_kv=0, bytes_p=2)


class TestEstimatePerformance(unittest.TestCase):
    """The estimate leaves the caller's config as it found it."""

    def test_estimating_twice_gives_the_same_score(self):
        """
        Feature: estimate_performance.
        Description: DeepSeek's dense and MoE layers are two groups, and the
            estimate applies their hooks to its config in place.  Estimate
            the same config twice.
        Expectation: The caller's config comes back unchanged, and the
            second score equals the first.
        """
        ccfg = CostModelConfig(DEEPSEEK_YAML)
        before = _plain_values(ccfg)
        first = estimate_performance(ccfg, device_type=Hard.Device_A2)
        self.assertEqual(_plain_values(ccfg), before)
        self.assertEqual(estimate_performance(ccfg, device_type=Hard.Device_A2), first)


class TestOpTable(unittest.TestCase):
    """Each op's load follows the tensor it runs over."""

    def test_token_wise_ops_scale_with_the_sequence(self):
        """The feed-forward activation runs once per token, like the projections around it."""
        short, long = op_table(_cfg(128)), op_table(_cfg(256))
        for op in ("n_attMM", "n_ffMM", "n_gather", "n_normOp", "n_ffAct"):
            self.assertEqual(long[op], 2 * short[op], f"{op}: s=128 gives {short[op]}, s=256 gives {long[op]}")

    def test_a_qk_norm_runs_over_every_head(self):
        """
        Feature: the load of a QK-norm.
        Description: The same model with a QK-norm and without, 8 query and 8
            key heads 64 wide, TP 2.
        Expectation: Only the first has the entry: 30 per element over every
            head of a token, each TP rank its half, in the parameters' bytes.
        """
        normed = SimpleNamespace(**vars(_cfg(128)), n_qknorm=1)
        self.assertEqual((op_table(normed)["n_qknorm"], "n_qknorm" in op_table(_cfg(128))),
                         (30 * 128 * (8 + 8) * 64 * 2 / 2, False))


    def test_scores_run_at_the_heads_widths(self):
        """
        Feature: the load of the attention scores.
        Description: 8 heads on width 512, 64 wide as h / a, then 128 wide;
            then MLA heads whose keys are 64 wide and a rotary 32, values 64.
        Expectation: Every head's queries against every key, then the
            weights against the values: 3 b s^2 a (d_qk + d_v), each TP
            rank its half, in the parameters' bytes.
        """
        s = 128
        for fields, d_qk, d_v in (({}, 64, 64), ({"dh": 128}, 128, 128),
                                  ({"dh": 64, "qk_nope_head_dim": 64, "dhr": 32}, 96, 64)):
            cfg = SimpleNamespace(**{**vars(_cfg(s)), **fields})
            self.assertEqual(op_table(cfg)["n_attBMM"], 3 * s * s * 8 * (d_qk + d_v) * 2 / 2, fields)

    def test_mla_projections_as_the_model_holds_them(self):
        """
        Feature: the load of MLA's projections.
        Description: DeepSeek-V3's attention: width 7168, 128 heads,
            latents of 1536 and 512, heads 128 and a rotary 64 wide; then the
            same without a query latent.
        Expectation: Six multiply-adds a token per weight, forward and
            backward, over the four attention matmuls: the 187105280
            weights of the model's projections; without a query latent, one
            projection to every head in place of the latent's two.
        """
        s = 128
        mla = SimpleNamespace(**{**vars(_cfg(s)), "h": 7168, "a": 128, "n_kv": 128, "dh": 128, "dhr": 64,
                                 "dc_q": 1536, "dc_kv": 512, "t": 1, "n_attMM": 4})
        self.assertEqual(op_table(mla)["n_attMM"], 6 * s * 187105280 / 4 * 2)
        direct = 187105280 - 1536 * (7168 + 128 * 192) + 7168 * 128 * 192
        mla.dc_q = 0
        self.assertEqual(op_table(mla)["n_attMM"], 6 * s * direct / 4 * 2)


    def test_a_moe_layer_prices_its_whole_feed_forward(self):
        """
        Feature: _flavour_tables, a MoE layer's feed-forward.
        Description: A layer of width 512, its dense layers 1024 wide, with
            8 experts 64 wide, 2 chosen, a shared expert and its gate, three
            feed-forward matmuls.
        Expectation: The feed-forward entry prices each token's two experts
            and the shared expert, each 64 wide, and, over the three
            matmuls, the router's 8 weights and the gate's one; the
            activation function's entry the three experts' activations, each
            64 wide, where the dense layers' is 1024 wide.
        """
        cfg = SimpleNamespace(**{**vars(_cfg(128)), "hff_exp": 64, "n_exp": 8, "n_chosen_exp": 2, "cap_fact": 1,
                                 "n_shared_exp": 1, "shared_expert_gate": True, "n_ffMM": 3})
        base, experts = _flavour_tables(cfg)
        self.assertEqual(experts["n_ffMM"], base["n_ffMM"] / 1024 * (3 * 64 + (8 + 1) / 3))
        self.assertEqual((experts["n_ffAct"], base["n_ffAct"]), (21 * 128 * 3 * 64 * 2 / 2, 21 * 128 * 1024 * 2 / 2))

    def test_a_moe_layer_pays_for_its_dispatch(self):
        """
        Feature: _flavour_tables, the dispatch entry.
        Description: The same layer of width 512 with 8 experts, 2 of them
            chosen a token, at expert parallel 1 and at 4.
        Expectation: Only the expert table prices a dispatch, a dense layer
            running none; it prices each token's two experts over the ranks
            the experts sit on, so it grows with the degree itself rather
            than with the degree less one, as the measured compute does.
        """
        plain = SimpleNamespace(**{**vars(_cfg(128)), "hff_exp": 64, "n_exp": 8, "n_chosen_exp": 2,
                                   "cap_fact": 1, "n_shared_exp": 1, "n_ffMM": 3, "ep": 1})
        spread = SimpleNamespace(**{**vars(plain), "ep": 4})
        base, one = _flavour_tables(plain)
        _, four = _flavour_tables(spread)
        self.assertNotIn("n_dispatch", base)
        self.assertEqual(one["n_dispatch"], MOE_DISPATCH * 128 * 512 * 2 * 1 * 2 / 2)
        self.assertEqual(four["n_dispatch"], 4 * one["n_dispatch"])
        stated = SimpleNamespace(**{**vars(plain), "moe_dispatch": 2 * MOE_DISPATCH})
        self.assertEqual(_flavour_tables(stated)[1]["n_dispatch"], 2 * one["n_dispatch"])

    def test_the_delta_rule_runs_in_chunks(self):
        """
        Feature: the load of the gated delta rule.
        Description: A linear-attention group of 32 value heads, keys and
            values 128 wide, on 128 tokens.
        Expectation: Per token and value head, forward and backward, three
            times what a chunk of 64 tokens runs over each of its tokens:
            its keys against its keys and its queries, its solved weights
            against its values and its decayed keys, its scores against its
            new values, 2 x 64 x (3 x 128 + 2 x 128), and the state read
            twice and written once, 6 x 128 x 128.
        """
        cfg = SimpleNamespace(**{**vars(_cfg(128)), "lin_n_v": 32, "lin_d_k": 128, "lin_d_v": 128})
        self.assertEqual(op_table(cfg)["n_linrec"],
                         3 * 128 * 32 * (2 * 64 * (3 * 128 + 2 * 128) + 6 * 128 * 128) * 2 / 2)

if __name__ == "__main__":
    unittest.main()
