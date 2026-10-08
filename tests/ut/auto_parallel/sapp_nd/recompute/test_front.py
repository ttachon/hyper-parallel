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
"""Tests for the recompute fronts: every way of running a layer kind, and the
ones no other way beats.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/recompute/test_front.py -v
"""
import copy
import dataclasses
import itertools
import os
import unittest
from types import SimpleNamespace
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple
from unittest.mock import patch

# The package has an import cycle that only the memory estimator's import order
# settles; the performance modules cannot be the first a process loads.
import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2  # pylint: disable=unused-import
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel import _hf_model_spec
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import estimate_layer_times
from hyper_parallel.auto_parallel.sapp_nd.recompute.front import (
    LayerOption,
    build_front,
    layer_fronts,
    layer_profiles,
    price_option,
)
from hyper_parallel.auto_parallel.sapp_nd.recompute.profile import SWITCHES, Cost, SwitchProfile

DEEPSEEK_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nd", "deepseek.yaml"
)

# A layer that keeps 100 per micro-batch and 10 once, and runs backward in 50.
# Its dropout is free and saves nothing; its gathers trade memory per
# micro-batch for memory held once.
_PLAIN = Cost(100.0, 10.0, 50.0)
_ALONE = {
    "attBMM": Cost(99.0, 10.0, 55.0),
    "headCast": Cost(92.0, 10.0, 50.1),
    "dropout": Cost(100.0, 10.0, 50.0),
    "softmax": Cost(84.0, 10.0, 50.2),
    "normOp": Cost(80.0, 10.0, 50.05),
    "gather": Cost(90.0, 20.0, 53.0),
    "ffAct": Cost(70.0, 10.0, 50.01),
}
_PROFILE = SwitchProfile(forward_time=25.0, plain=_PLAIN, alone=_ALONE, full=Cost(2.0, 20.0, 80.0))


def _every_option(profile: SwitchProfile) -> Iterator[Tuple[Optional[frozenset], Cost]]:
    """Every setting of the switches that act on the kind with its cost, then full recompute."""
    switches = profile.acting()
    for size in range(len(switches) + 1):
        for names in itertools.combinations(switches, size):
            yield frozenset(names), profile.selective(names)
    yield None, profile.full


def _cost(option: LayerOption) -> Tuple[float, float, float]:
    """The option's memory per micro-batch, memory once and backward time."""
    return option.memory_per_micro_batch, option.memory_once, option.backward_time


def _no_worse(these: Sequence[float], those: Sequence[float]) -> bool:
    """Whether *these* costs are all at most *those*."""
    return all(mine <= theirs for mine, theirs in zip(these, those))


class _FrontChecks(unittest.TestCase):
    """Checks every front must pass."""

    def assert_front(self, options: Sequence[LayerOption]) -> None:
        """The plain layer first, full recompute on it, and no option beating another."""
        self.assertEqual(options[0].recompute, frozenset())
        self.assertIn(None, [option.recompute for option in options])
        for mine, theirs in itertools.permutations(options, 2):
            self.assertFalse(
                _no_worse(_cost(mine), _cost(theirs)) and _cost(mine) != _cost(theirs),
                f"{mine.recompute} beats {theirs.recompute}",
            )
        times = [option.backward_time for option in options]
        self.assertEqual(times, sorted(times))


class TestSwitchProfile(unittest.TestCase):
    """A selective layer costs the plain layer plus what each of its ops costs alone."""

    def test_an_option_adds_what_each_of_its_ops_costs(self):
        """
        Feature: SwitchProfile.selective.
        Description: Recompute the softmax and the gathers together.
        Expectation: The plain layer's cost, plus each op's own difference
            from it.
        """
        cost = _PROFILE.selective(["gather", "softmax"])
        self.assertEqual(cost.values(), (74.0, 20.0, 53.2))

    def test_recomputing_nothing_is_the_plain_layer(self):
        """
        Feature: SwitchProfile.selective.
        Description: Recompute no op.
        Expectation: The plain layer's cost.
        """
        self.assertEqual(_PROFILE.selective([]), _PLAIN)

    def test_a_setting_adds_up_from_the_selective_base_where_it_differs(self):
        """
        Feature: SwitchProfile.selective, for a kind a census prices.
        Description: A profile whose layer selective with every op kept
            keeps less than its plain layer, as where a census prices the
            plain layer and the formulas a selective one, and whose gathers
            recomputed change the working sets; recompute the softmax and
            the gathers together, and no op.
        Expectation: The base's cost plus each op's own difference from it,
            working sets included; the plain layer's for no op.
        """
        base = Cost(80.0, 10.0, 50.0, working=5.0, first_working=3.0)
        alone = dict(_ALONE, gather=Cost(70.0, 20.0, 53.0, working=2.0, first_working=1.0),
                     softmax=Cost(64.0, 10.0, 50.2, working=5.0, first_working=3.0))
        profile = SwitchProfile(25.0, _PLAIN, alone, Cost(2.0, 20.0, 80.0), selective_base=base)
        cost = profile.selective(["gather", "softmax"])
        self.assertEqual((cost.values(), cost.working, cost.first_working), ((54.0, 20.0, 53.2), 2.0, 1.0))
        self.assertEqual(profile.selective([]), _PLAIN)

    def test_a_setting_measured_whole_costs_what_was_measured(self):
        """
        Feature: SwitchProfile.selective, for a setting measured whole.
        Description: A profile that measured recomputing the softmax and the
            activation together, as a census prices HyperParallel's
            selective policy.
        Expectation: That setting costs what was measured; any other adds
            up over its ops.
        """
        policy = Cost(30.0, 10.0, 60.0, working=7.0)
        profile = SwitchProfile(25.0, _PLAIN, _ALONE, Cost(2.0, 20.0, 80.0),
                                whole={frozenset({"softmax", "ffAct"}): policy})
        self.assertIs(profile.selective(["ffAct", "softmax"]), policy)
        self.assertEqual(profile.selective(["gather", "softmax"]).values(), (74.0, 20.0, 53.2))

    def test_a_census_working_set_is_clamped_once_it_adds_up(self):
        """
        Feature: SwitchProfile.selective, for a kind a census's records per
            op price.
        Description: The layer selective with every op kept keeps 40 of
            what its census states and its backward holds 30, so that its
            working set as warm-up ends is clamped; the base and each op
            alone state that working set before the clamp, 100 less twice
            what they keep, beside what they keep. Recompute the softmax,
            the softmax and the activation, and the norms too.
        Expectation: Each setting's working set is what the memory model
            clamps: before the clamp, plus what the layer keeps beyond what
            its backward holds, where it still keeps more.
        """
        def cost(kept: float) -> Cost:
            """A setting that keeps *kept* of what the census states, its working set before the clamp."""
            return Cost(kept, 10.0, 50.0, working=30.0 + 100.0 - 2 * kept, census_kept=kept)

        alone = dict(_ALONE, softmax=cost(36.0), ffAct=cost(32.0), normOp=cost(25.0))
        profile = SwitchProfile(25.0, _PLAIN, alone, Cost(2.0, 20.0, 80.0), selective_base=cost(40.0),
                                census_held=30.0)
        self.assertEqual(profile.selective(["softmax"]).working, 58.0 + 6.0)
        self.assertEqual(profile.selective(["softmax", "ffAct"]).working, 74.0)
        self.assertEqual(profile.selective(["softmax", "ffAct", "normOp"]).working, 104.0)
        self.assertEqual(profile.selective([]), _PLAIN)


class TestBuildFront(_FrontChecks):
    """The front keeps the options no other option beats."""

    def test_the_front_is_a_front(self):
        """
        Feature: build_front.
        Description: Build the front of a profile.
        Expectation: The plain layer first, full recompute on it, fastest
            first, and no option on it beating another.
        """
        self.assert_front(build_front(_PROFILE))

    def test_every_option_left_out_is_beaten_by_one_kept(self):
        """
        Feature: build_front.
        Description: Compare each of the 129 options with the front.
        Expectation: An option off the front costs at least as much as one
            on it, in memory per micro-batch, memory once and time alike.
        """
        front = build_front(_PROFILE)
        kept = {option.recompute for option in front}
        for recompute, cost in _every_option(_PROFILE):
            if recompute not in kept:
                self.assertTrue(any(_no_worse(_cost(option), cost.values()) for option in front), recompute)

    def test_of_options_that_cost_the_same_the_one_recomputing_less_stays(self):
        """
        Feature: build_front.
        Description: Recomputing the dropout saves nothing and costs nothing.
        Expectation: No option on the front recomputes it.
        """
        self.assertFalse(any(option.recompute and "dropout" in option.recompute
                             for option in build_front(_PROFILE)))

    def test_plain_and_full_are_always_offered(self):
        """
        Feature: build_front.
        Description: A profile whose full recompute costs more than the plain
            layer in everything.
        Expectation: Full recompute is still offered, and so is the plain layer.
        """
        worse = SwitchProfile(25.0, _PLAIN, _ALONE, Cost(200.0, 50.0, 100.0))
        recomputes = [option.recompute for option in build_front(worse)]
        self.assertIn(None, recomputes)
        self.assertIn(frozenset(), recomputes)

    def test_the_options_the_balancer_knows_are_named(self):
        """
        Feature: build_front.
        Description: A config whose selective recompute recomputes the
            activation, on a layer whose gathers are worth recomputing.
        Expectation: The options are named as the pipeline balancer knows
            them: SLCT, COMM, BOTH, and the plain and full ones.
        """
        noop = Cost(100.0, 10.0, 51.0)
        profile = SwitchProfile(25.0, _PLAIN, dict(dict.fromkeys(SWITCHES, noop), ffAct=_ALONE["ffAct"],
                                                   gather=Cost(40.0, 10.0, 51.0)), Cost(2.0, 10.0, 80.0))
        configured = dict(dict.fromkeys(SWITCHES, 1), ffAct=0)
        names = {option.recompute: option.names for option in build_front(profile, configured)}
        self.assertEqual(names, {
            frozenset(): ("NONE",),
            frozenset({"ffAct"}): ("SLCT",),
            frozenset({"gather"}): ("COMM",),
            frozenset({"ffAct", "gather"}): ("BOTH",),
            None: ("FULL",),
        })

    def test_an_option_lighter_at_a_count_in_between_stays(self):
        """
        Feature: build_front.
        Description: Two options that keep as much per micro-batch and once,
            the faster one charged 20 beyond what it keeps at a count of
            micro-batches in flight between one and the most, which a stage
            keeps.
        Expectation: Both stay, each keeping what it keeps at that count;
            without the count, only the faster one stays.
        """
        plain = Cost(100.0, 10.0, 50.0, (0.0,))
        alone = dict(dict.fromkeys(SWITCHES, plain), ffAct=Cost(70.0, 10.0, 50.01, (0.0,)),
                     normOp=Cost(70.0, 10.0, 50.05, (20.0,)))
        profile = SwitchProfile(25.0, plain, alone, Cost(2.0, 20.0, 80.0, (0.0,)), counts=(2,))
        options = {option.recompute: option for option in build_front(profile)}
        self.assertEqual((options[frozenset({"ffAct"})].memory(2), options[frozenset({"normOp"})].memory(2)),
                         (150.0, 130.0))
        without = {option.recompute for option in build_front(dataclasses.replace(profile, counts=()))}
        self.assertIn(frozenset({"ffAct"}), without)
        self.assertNotIn(frozenset({"normOp"}), without)

    def test_an_option_stands_for_a_switch_setting(self):
        """
        Feature: LayerOption.switches.
        Description: A selective option and the full one.
        Expectation: The selective one sets its ops' switches to 0 and the
            others to 1; full recompute has no switch setting.
        """
        options = {option.recompute: option for option in build_front(_PROFILE)}
        self.assertEqual(options[frozenset({"ffAct"})].switches, dict(dict.fromkeys(SWITCHES, 1), ffAct=0))
        self.assertIsNone(options[None].switches)


def _plain_values(ccfg: Any) -> Dict[str, Any]:
    """The config's plain fields."""
    return {name: value for name, value in vars(ccfg).items() if isinstance(value, (bool, int, float, str))}


class TestLayerFronts(_FrontChecks):
    """DeepSeek, a dense prefix then MoE layers: a front per kind, measured on the model."""

    @classmethod
    def setUpClass(cls) -> None:
        """The fixture's evaluator and its fronts."""
        cls.evaluator = EvaluatorV2(DEEPSEEK_YAML, framework="mindformers", log_level=0)
        cls.fronts = layer_fronts(cls.evaluator, Hard.Device_A2)

    def test_each_layer_kind_gets_its_front(self):
        """
        Feature: layer_fronts.
        Description: The fronts of a model with a dense and a MoE kind.
        Expectation: One front per kind, in model order, each a front.
        """
        self.assertEqual([front.kind.name for front in self.fronts], ["dense", "moe"])
        for front in self.fronts:
            self.assert_front(front.options)

    def test_combined_options_match_a_full_evaluation(self):
        """
        Feature: layer_fronts.
        Description: Evaluate a few MoE options that recompute several ops in
            full: their backward time alone, and their memory in the pipeline
            balancer's description of a config that sets their switches.
        Expectation: The same backward time, and the same memory to the MB
            the description rounds to.
        """
        front = self.fronts[1]
        several = [option for option in front.options if option.recompute and len(option.recompute) > 1]
        self.assertGreater(len(several), 2)
        for option in several[::max(1, len(several) // 3)]:
            backward = estimate_layer_times(copy.deepcopy(self.evaluator.ccfg), front.kind, LayerType.SEL_REC_LAYER,
                                            Hard.Device_A2, switches=option.switches)[1]
            self.assertAlmostEqual(backward / option.backward_time, 1.0, places=12, msg=option.recompute)
            evaluator = EvaluatorV2(DEEPSEEK_YAML, framework="mindformers", log_level=0)
            evaluator.ccfg.rec_op = Config(option.switches)
            description = evaluator.estimate_layer_memory()["layers_description"]
            moe = [desc for desc in description if desc["type"] == "BODY"][1]
            self.assertLessEqual(abs(moe["memory_select_rec"] - EvalUtils.mb(option.memory_per_micro_batch)), 1,
                                 option.recompute)

    def test_an_mla_kind_offers_its_up_projections(self):
        """
        Feature: layer_fronts on an MLA model.
        Description: DeepSeek's kinds build their query, key and value heads
            from latents; price recomputing the up-projections alone, and
            evaluate it in full, its backward time alone and its memory in
            the pipeline balancer's description of a config that sets it.
        Expectation: Each kind weighs attUp: recomputing it keeps less per
            micro-batch than the plain layer and takes longer in backward,
            by what a full evaluation of the setting charges; and each front
            has an option recomputing it.
        """
        profiles = layer_profiles(self.evaluator, Hard.Device_A2)
        for index, ((_, kind), profile) in enumerate(profiles.items()):
            self.assertIn("attUp", profile.acting(), kind)
            option = price_option(profile, frozenset({"attUp"}))
            self.assertLess(option.memory_per_micro_batch, profile.plain.memory_per_micro_batch, kind)
            self.assertGreater(option.backward_time, profile.plain.backward_time, kind)
            backward = estimate_layer_times(copy.deepcopy(self.evaluator.ccfg), kind, LayerType.SEL_REC_LAYER,
                                            Hard.Device_A2, switches=option.switches)[1]
            self.assertAlmostEqual(backward / option.backward_time, 1.0, places=12, msg=kind)
            evaluator = EvaluatorV2(DEEPSEEK_YAML, framework="mindformers", log_level=0)
            evaluator.ccfg.rec_op = Config(option.switches)
            description = evaluator.estimate_layer_memory()["layers_description"]
            body = [desc for desc in description if desc["type"] == "BODY"][index]
            self.assertLessEqual(abs(body["memory_select_rec"] - EvalUtils.mb(option.memory_per_micro_batch)), 1,
                                 kind)
            self.assertTrue(any(item.recompute and "attUp" in item.recompute for item in self.fronts[index].options))

    def test_asking_twice_gives_the_same_fronts_and_leaves_the_config_alone(self):
        """
        Feature: layer_fronts.
        Description: Ask for the fronts again.
        Expectation: The same fronts, and the evaluator's config as it was.
        """
        before = _plain_values(self.evaluator.ccfg)
        switches = dict(vars(self.evaluator.ccfg.rec_op))
        self.assertEqual(layer_fronts(self.evaluator, Hard.Device_A2), self.fronts)
        self.assertEqual(_plain_values(self.evaluator.ccfg), before)
        self.assertEqual(vars(self.evaluator.ccfg.rec_op), switches)


# A dense model, from its config overrides alone.
_DENSE = {
    "model": {"name": "nonexistent", "config_overrides": {
        "hidden_size": 2048, "num_hidden_layers": 4, "num_attention_heads": 16, "num_key_value_heads": 8,
        "intermediate_size": 5632, "vocab_size": 32000, "max_position_embeddings": 4096}},
    "data": {"max_seq_len": 4096},
    "train": {"accelerator": {"dp_shard": 1, "dp_replicate": 1, "tp_degree": 4, "pipeline_parallel_degree": 2},
              "micro_batch_size": 1, "micro_batch_num": 4,
              "gradient_checkpointing": {"activation_checkpoint": "full"}, "optimizer": {"max_grad_norm": 1.0}},
    "context": {"max_device_memory": "64GB"},
}

# A Qwen3.5-style hybrid: linear-attention layers between full-attention ones.
_HYBRID = SimpleNamespace(
    model_type="qwen3_5_moe", hidden_size=1024, num_hidden_layers=7, num_attention_heads=8,
    num_key_value_heads=2, head_dim=128, vocab_size=32000, max_position_embeddings=8192,
    num_experts=16, num_experts_per_tok=4, moe_intermediate_size=256,
    shared_expert_intermediate_size=256, mtp_num_hidden_layers=1, attn_output_gate=True,
    layer_types=["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 2 + ["full_attention"],
    linear_num_key_heads=8, linear_key_head_dim=64, linear_num_value_heads=16,
    linear_value_head_dim=64, linear_conv_kernel_dim=4)
_HYBRID_TRAINING = {
    "model": {"pretrained_model_name_or_path": "local/qwen3_5", "torch_dtype": "bfloat16"},
    "training": {"global_batch_size": 16, "micro_batch_size": 1, "max_grad_norm": 1.0},
    "accelerator": {"tp_size": 2, "pp_size": 2},
    "activation_checkpoint": {"mode": "full"},
    "dataset": {"data_transform": {"max_seq_len": 4096}},
}


class TestOtherModels(_FrontChecks):
    """A dense and a hybrid model get fronts without hand input."""

    def test_a_dense_model_gets_one_front(self):
        """
        Feature: layer_fronts.
        Description: A dense model with one kind of layer, at TP 4.
        Expectation: One front, and recomputing its gathers is on it: at
            DP 1 they are the only buffers it keeps.
        """
        evaluator = EvaluatorV2(copy.deepcopy(_DENSE), framework="hyper_v2", log_level=0)
        fronts = layer_fronts(evaluator, Hard.Device_A2)
        self.assertEqual([front.kind for front in fronts], [None])
        self.assert_front(fronts[0].options)
        self.assertTrue(any(option.recompute and "gather" in option.recompute for option in fronts[0].options))

    def test_a_model_without_latents_weighs_the_seven_switches(self):
        """
        Feature: layer_profiles and build_front.
        Description: A dense model, which compresses nothing into latents.
        Expectation: attUp keeps nothing there: its profile weighs the other
            seven switches alone, and no option recomputes it.
        """
        evaluator = EvaluatorV2(copy.deepcopy(_DENSE), framework="hyper_v2", log_level=0)
        profiles = layer_profiles(evaluator, Hard.Device_A2)
        for profile in profiles.values():
            self.assertEqual(profile.acting(), tuple(name for name in SWITCHES if name != "attUp"))
            self.assertEqual(profile.alone["attUp"], profile.plain)
        fronts = layer_fronts(evaluator, Hard.Device_A2)
        self.assertFalse(any(option.recompute and "attUp" in option.recompute for option in fronts[0].options))

    def test_a_hybrid_model_gets_a_front_per_attention_flavour(self):
        """
        Feature: layer_fronts.
        Description: The hybrid's linear- and full-attention layers.
        Expectation: A front for each; the linear layers, which hold no
            attention score, have no option recomputing one.
        """
        with patch.object(_hf_model_spec, "_get_hf_config", return_value=_HYBRID):
            evaluator = EvaluatorV2(copy.deepcopy(_HYBRID_TRAINING), framework="hyper_v2", log_level=0)
            fronts = layer_fronts(evaluator, Hard.Device_A2)
        self.assertEqual([front.kind.name for front in fronts], ["linear_attention", "full_attention"])
        for front in fronts:
            self.assert_front(front.options)
        score_ops = {"attBMM", "softmax", "headCast"}
        self.assertFalse(any(option.recompute and option.recompute & score_ops for option in fronts[0].options))
        self.assertTrue(any(option.recompute and option.recompute & score_ops for option in fronts[1].options))


if __name__ == "__main__":
    unittest.main()
