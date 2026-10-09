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
import copy
import math
import os
import tempfile
import unittest
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

import yaml

# The package has an import cycle that only the memory estimator's import order
# settles; perf_estimation.estimate cannot be the first module a process loads.
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.debug as Debug
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
import hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate as Estimate
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import (
    check_and_apply_custom_hook,
    layer_groups,
    layer_kinds,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common import cost_model_preprocess as PreProcess
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import estimate_comm
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import (
    LayerTimes,
    MOE_DISPATCH,
    _flavour_tables,
    estimate_comp,
    estimate_layer_times,
    estimate_performance,
    estimate_stage,
    op_table,
)
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.getters import get_model_order
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig

DEEPSEEK_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nd", "deepseek.yaml"
)
_HF_CONFIG = "hyper_parallel.auto_parallel._hf_model_spec._get_hf_config"


def _vision_language() -> SimpleNamespace:
    """A small Qwen3-VL-MoE Transformers config: a MoE language model and a vision tower."""
    return SimpleNamespace(
        model_type="qwen3_vl_moe",
        text_config=SimpleNamespace(
            hidden_size=2048, num_hidden_layers=8, num_attention_heads=16, num_key_value_heads=8,
            intermediate_size=5632, vocab_size=32000, max_position_embeddings=8192, head_dim=128,
            num_experts=16, num_experts_per_tok=4, moe_intermediate_size=768,
        ),
        vision_config=SimpleNamespace(
            hidden_size=1152, depth=6, num_heads=16, intermediate_size=4304, out_hidden_size=3584,
            patch_size=16, spatial_merge_size=2, num_position_embeddings=2304,
        ),
    )


def _vision_language_yaml() -> Dict[str, Any]:
    """A trainer yaml for that model on 8 devices, TP 2, without pipeline parallelism."""
    return {
        "model": {"pretrained_model_name_or_path": "local/qwen3_vl_moe", "torch_dtype": "bfloat16"},
        "training": {"global_batch_size": 16, "micro_batch_size": 1, "max_grad_norm": 1.0},
        "accelerator": {"tp_size": 2, "pp_size": 1, "cp_size": 1, "ep_size": 1},
        "fsdp_config": {"dp_shard_size": 4},
        "activation_checkpoint": {"mode": "full"},
        "dataset": {"data_transform": {"max_seq_len": 4096}},
        "context": {"max_device_memory": "64GB", "device_num": 8},
    }


def _plain_values(ccfg: CostModelConfig) -> Dict[str, Any]:
    """The config's plain fields, the ones layer kinds overwrite."""
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

    @staticmethod
    def _debugged(ccfg: CostModelConfig, **kwargs: Any) -> tuple:
        """The score and its parts."""
        debugger = Debug.Debug(Dim.Dimensions([(Dim.DP, ccfg.d)], all_dims=[Dim.DP]), info_type=Debug.PerfParts,
                               enable=True)
        score = estimate_performance(ccfg, debugger=debugger, device_type=Hard.Device_A2, **kwargs)
        return score, debugger.info

    def test_stage_savings_come_off_the_recompute(self):
        """
        Feature: estimate_performance stage_savings.
        Description: DeepSeek-V3 over 16 stages and 32 micro-batches,
            saving nothing, then the same time on every stage.
        Expectation: Saving nothing changes nothing. Saving the same time
            everywhere lowers the score, and the recompute part of the
            slowest stage by that time for each micro-batch.
        """
        ccfg = CostModelConfig(DEEPSEEK_YAML)
        score, parts = self._debugged(ccfg)
        self.assertEqual(self._debugged(ccfg, stage_savings=[0.0] * ccfg.p)[0], score)
        saved = parts[Debug.PerfParts.RECOMPUTE] / ccfg.m / 4
        lower, lower_parts = self._debugged(ccfg, stage_savings=[saved] * ccfg.p)
        self.assertLess(lower, score)
        self.assertAlmostEqual(lower_parts[Debug.PerfParts.RECOMPUTE] / (parts[Debug.PerfParts.RECOMPUTE]
                                                                           - saved * ccfg.m), 1.0, places=12)

    def test_stage_savings_need_one_per_stage(self):
        """
        Feature: estimate_performance stage_savings.
        Description: Savings for fewer stages than the pipeline has.
        Expectation: Refused.
        """
        ccfg = CostModelConfig(DEEPSEEK_YAML)
        with self.assertRaises(ValueError):
            estimate_performance(ccfg, device_type=Hard.Device_A2, stage_savings=[0.0] * (ccfg.p - 1))
    def test_a_vision_language_model_records_every_towers_communication(self):
        """
        Feature: estimate_performance on a multimodal model, its recorded parts.
        Description: A Qwen3-VL-MoE at TP 2 and no pipeline parallelism, so
            its vision tower and its language model share the one stage and
            both communicate there, priced with a debugger on A2 and A3.
        Expectation: The recorded parts add up to the stage's time, so a run
            without pipeline parallelism shows no bubble and the check of
            the straggler's parts stays silent; kept as the last tower's
            alone, the parts missed the vision tower's collectives and the
            bubble took them.
        """
        with tempfile.TemporaryDirectory() as folder, patch(_HF_CONFIG, return_value=_vision_language()):
            path = os.path.join(folder, "train.yaml")
            with open(path, "w", encoding="utf-8") as stream:
                yaml.safe_dump(_vision_language_yaml(), stream)
            ccfg = EvaluatorV2(path, framework="hyper_v2", log_level=0).ccfg
            self.assertTrue(ccfg.multimodal)
            for device in (Hard.Device_A2, Hard.Device_A3):
                debugger = Debug.Debug(Dim.Dimensions([(Dim.DP, ccfg.d)], all_dims=[Dim.DP]), Debug.PerfParts)
                with patch.object(Estimate, "nd_logger") as nd_logger:
                    total = estimate_performance(deepcopy(ccfg), debugger=debugger, device_type=device)
                self.assertEqual(nd_logger.error.call_count, 0)
                self.assertGreater(debugger.info[Debug.PerfParts.MP_COMM], 0)
                self.assertTrue(math.isclose(debugger.info[Debug.PerfParts.BUBBLE], 0.0, abs_tol=1e-12 * total))

    def test_a_vision_language_models_towers_count_their_cast_and_activation(self):
        """
        Feature: estimate_performance on a multimodal model, its op counts.
        Description: The same Qwen3-VL-MoE, estimated on A3 while the config
            fields no parser set are recorded as they are read.
        Expectation: No layer of either tower reads its score cast or its
            feed-forward activation as unset: each is counted once a layer,
            as on a model with one tower, where every layer of both towers
            used to read them as 0 and priced the two ops at nothing.
        """
        reads = PreProcess.defaultdict(PreProcess.Counter)
        with tempfile.TemporaryDirectory() as folder, patch(_HF_CONFIG, return_value=_vision_language()):
            path = os.path.join(folder, "train.yaml")
            with open(path, "w", encoding="utf-8") as stream:
                yaml.safe_dump(_vision_language_yaml(), stream)
            ccfg = EvaluatorV2(path, framework="hyper_v2", log_level=0).ccfg
            with patch.object(PreProcess, "UNSET_READS", reads):
                estimate_performance(deepcopy(ccfg), device_type=Hard.Device_A3)
        self.assertEqual({"n_headCast", "n_ffAct"} & set(reads), set())


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

    def test_a_stated_dispatch_cost_of_zero_prices_no_dispatch(self):
        """
        Feature: _flavour_tables, a stated dispatch cost.
        Description: The MoE layer of width 512 at expert parallel 4, with no
            dispatch cost, with None, which is how the parser hands over a
            run that states none, and with a stated 0.
        Expectation: No cost and None keep the measured default; 0 prices no
            dispatch at all, for a model whose compute does not grow with
            the degree, where it used to fall back to the default.
        """
        plain = SimpleNamespace(**{**vars(_cfg(128)), "hff_exp": 64, "n_exp": 8, "n_chosen_exp": 2,
                                   "cap_fact": 1, "n_shared_exp": 1, "n_ffMM": 3, "ep": 4})
        default = _flavour_tables(plain)[1]["n_dispatch"]
        self.assertEqual(default, MOE_DISPATCH * 128 * 512 * 2 * 4 * 2 / 2)
        for stated, want in ((None, default), (0, 0)):
            cfg = SimpleNamespace(**{**vars(plain), "moe_dispatch": stated})
            self.assertEqual(_flavour_tables(cfg)[1]["n_dispatch"], want)

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


_SWITCHES = ("attBMM", "headCast", "dropout", "softmax", "normOp", "gather", "ffAct")


class TestLayerTimes(unittest.TestCase):
    """One layer priced alone, as the pipeline balancer is given it."""

    @classmethod
    def setUpClass(cls) -> None:
        """DeepSeek: a dense prefix, then MoE layers, on sixteen stages."""
        cls.ccfg = CostModelConfig(DEEPSEEK_YAML)
        check_and_apply_custom_hook(cls.ccfg)
        cls.stages = cls.ccfg.generate_partitions_vpp()
        cls.kinds = dict(zip(get_model_order(cls.ccfg, cls.stages), layer_kinds(cls.ccfg)))
        # The kinds of the stack's groups: dense, MoE, and the MTP layer's MoE.
        cls.groups = [kind for kind, _ in layer_groups(cls.ccfg)]

    def _times(self, layer_type: LayerType, **switches: int):
        """``(forward, backward)`` of a MoE layer, with these recompute switches."""
        cfg = copy.deepcopy(self.ccfg)
        cfg.rec_op = Config(dict(dict.fromkeys(_SWITCHES, 1), **switches))
        return estimate_layer_times(cfg, self.groups[1], layer_type, Hard.Device_A2)

    def test_a_stage_of_one_group_is_the_sum_of_its_layers(self):
        """
        Feature: estimate_layer_times.
        Description: Price every layer alone, and every stage as the search does.
        Expectation: A stage whose layers all belong to one group costs the
            sum of its layers.
        """
        custom = CustomConfig()
        comp = estimate_comp(copy.deepcopy(self.ccfg), custom, self.stages)
        recomp = estimate_comp(copy.deepcopy(self.ccfg), custom, self.stages, with_recomp=True)
        walked = copy.deepcopy(self.ccfg)
        comm = estimate_comm(walked, custom, self.stages, Hard.Device_A2)
        recomm = estimate_comm(copy.deepcopy(self.ccfg), custom, self.stages, Hard.Device_A2, with_recomp=True)
        search = estimate_stage(walked, custom, comp, comm, recomp, recomm)
        times = LayerTimes(Hard.Device_A2)
        checked = 0
        for s, stage in enumerate(self.stages):
            positions = [(s, c, i) for c, chunk in enumerate(stage) for i, _ in enumerate(chunk)]
            if len({self.kinds.get(position) for position in positions} - {None}) != 1 or any(
                    position not in self.kinds for position in positions):
                continue
            total = sum(sum(times(self.ccfg, self.kinds[p], stage[p[1]][p[2]])) for p in positions)
            self.assertAlmostEqual(total / search[s], 1.0, places=12, msg=f"stage {s}: {total} vs {search[s]}")
            checked += 1
        self.assertGreater(checked, 0)

    def test_full_recompute_costs_backward_time_only(self):
        """
        Feature: estimate_layer_times.
        Description: The same layer, without and with full recompute.
        Expectation: The forward time is the same; the backward time grows.
        """
        plain, full = self._times(LayerType.NOT_REC_LAYER), self._times(LayerType.FULL_REC_LAYER)
        self.assertEqual(full[0], plain[0])
        self.assertGreater(full[1], plain[1])

    def test_selective_recompute_costs_what_its_switches_recompute(self):
        """
        Feature: estimate_layer_times.
        Description: A selective layer that keeps every op, then one that
            recomputes its softmax.
        Expectation: Keeping every op costs what the plain layer costs;
            recomputing the softmax costs more.
        """
        plain = self._times(LayerType.NOT_REC_LAYER)
        self.assertEqual(self._times(LayerType.SEL_REC_LAYER), plain)
        self.assertGreater(self._times(LayerType.SEL_REC_LAYER, softmax=0)[1], plain[1])

    def test_switches_price_a_selective_layer_as_a_config_that_sets_them(self):
        """
        Feature: estimate_layer_times switches.
        Description: Price a selective MoE layer of a config that keeps every
            op, with switches that recompute its softmax.
        Expectation: It costs what the layer of a config that recomputes its
            softmax costs, and the config keeps its own switches.
        """
        cfg = copy.deepcopy(self.ccfg)
        cfg.rec_op = Config(dict.fromkeys(_SWITCHES, 1))
        switched = estimate_layer_times(cfg, self.groups[1], LayerType.SEL_REC_LAYER, Hard.Device_A2,
                                        switches=dict(dict.fromkeys(_SWITCHES, 1), softmax=0))
        self.assertEqual(switched, self._times(LayerType.SEL_REC_LAYER, softmax=0))
        self.assertEqual(vars(cfg.rec_op), dict.fromkeys(_SWITCHES, 1))

    def test_pricing_leaves_the_config_alone(self):
        """
        Feature: estimate_layer_times.
        Description: Price a MoE layer on the config.
        Expectation: The caller's config comes back unchanged.
        """
        cfg = copy.deepcopy(self.ccfg)
        before = _plain_values(cfg)
        estimate_layer_times(cfg, self.groups[1], LayerType.FULL_REC_LAYER, Hard.Device_A2)
        self.assertEqual(_plain_values(cfg), before)

    def test_a_config_is_copied_once_and_each_layer_priced_once(self):
        """
        Feature: LayerTimes.
        Description: Price a layer, change the config the way a kind would,
            then price the same layer and another type.
        Expectation: The change is not seen, and the repeated layer is not
            priced again.
        """
        cfg = copy.deepcopy(self.ccfg)
        times = LayerTimes(Hard.Device_A2)
        first = times(cfg, self.groups[1], LayerType.NOT_REC_LAYER)
        cfg.s *= 2  # a field no layer kind sets
        with patch.object(Estimate, "estimate_layer_times", wraps=Estimate.estimate_layer_times) as spy:
            self.assertEqual(times(cfg, self.groups[1], LayerType.NOT_REC_LAYER), first)
            full = times(cfg, self.groups[1], LayerType.FULL_REC_LAYER)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(full, LayerTimes(Hard.Device_A2)(self.ccfg, self.groups[1], LayerType.FULL_REC_LAYER))

    def test_each_set_of_switches_is_priced_once(self):
        """
        Feature: LayerTimes switches.
        Description: Price a selective layer with the same switches twice,
            listed in another order the second time, then with other ones.
        Expectation: Two estimates; the switches decide the time.
        """
        times = LayerTimes(Hard.Device_A2)
        softmax = dict(dict.fromkeys(_SWITCHES, 1), softmax=0)
        with patch.object(Estimate, "estimate_layer_times", wraps=Estimate.estimate_layer_times) as spy:
            first = times(self.ccfg, self.groups[1], LayerType.SEL_REC_LAYER, softmax)
            again = times(self.ccfg, self.groups[1], LayerType.SEL_REC_LAYER, dict(reversed(list(softmax.items()))))
            gather = times(self.ccfg, self.groups[1], LayerType.SEL_REC_LAYER,
                           dict(dict.fromkeys(_SWITCHES, 1), gather=0))
        self.assertEqual(spy.call_count, 2)
        self.assertEqual(again, first)
        self.assertNotEqual(gather, first)

    def test_a_kinds_options_share_their_plain_times(self):
        """
        Feature: LayerTimes plain times.
        Description: Price a MoE layer plain, fully recomputed and with two
            sets of switches, then the embedding.
        Expectation: The MoE options share one plain pricing, the embedding
            has its own, and each option's time is what pricing it alone
            gives.
        """
        times = LayerTimes(Hard.Device_A2)
        options = ((LayerType.NOT_REC_LAYER, None), (LayerType.FULL_REC_LAYER, None),
                   (LayerType.SEL_REC_LAYER, dict(dict.fromkeys(_SWITCHES, 1), softmax=0)),
                   (LayerType.SEL_REC_LAYER, dict(dict.fromkeys(_SWITCHES, 1), gather=0, ffAct=0)))
        with patch.object(Estimate, "plain_layer_times", wraps=Estimate.plain_layer_times) as spy:
            priced = [times(self.ccfg, self.groups[1], layer_type, switches) for layer_type, switches in options]
            times(self.ccfg, None, LayerType.EMBEDDING_LAYER)
        self.assertEqual(spy.call_count, 2)
        for (layer_type, switches), both in zip(options, priced):
            alone = estimate_layer_times(copy.deepcopy(self.ccfg), self.groups[1], layer_type, Hard.Device_A2,
                                         switches=switches)
            self.assertEqual(both, alone)


if __name__ == "__main__":
    unittest.main()
