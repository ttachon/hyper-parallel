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
"""Tests for choosing every layer's recompute option for a candidate of the search.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/recompute/test_candidate.py -v
"""
import copy
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence
from unittest.mock import patch

import yaml

# The package has an import cycle that only the memory estimator's import order
# settles; the performance modules cannot be the first a process loads.
import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2  # pylint: disable=unused-import
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel import _hf_model_spec
from hyper_parallel.auto_parallel._exec_spec import ExecSpec, RecomputeRange
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.nd import parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.nd.common.apply_exec import apply_exec
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import layer_kinds
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level
from hyper_parallel.auto_parallel.sapp_nd.recompute import candidate as Candidate
from hyper_parallel.auto_parallel.sapp_nd.recompute.candidate import (
    MODES,
    LayerRange,
    RecomputeChoice,
    choose_recompute,
    describe,
    micro_batches_in_flight,
    mode_recompute,
    option_label,
    to_records,
)
from hyper_parallel.auto_parallel.sapp_nd.recompute.front import LayerOption

DEEPSEEK_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nd", "deepseek.yaml"
)

# A dense model, from its config overrides alone, at TP 4 and PP 2.
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


# A vision-language model: a small tower in front of a four-layer MoE text model.
_VL = SimpleNamespace(
    model_type="qwen3_vl_moe",
    text_config=SimpleNamespace(
        hidden_size=2048, num_hidden_layers=4, num_attention_heads=16, num_key_value_heads=8,
        intermediate_size=5632, vocab_size=32000, max_position_embeddings=8192, head_dim=128,
        num_experts=16, num_experts_per_tok=4, moe_intermediate_size=768),
    vision_config=SimpleNamespace(
        hidden_size=1152, depth=2, num_heads=16, intermediate_size=4304, out_hidden_size=2048,
        patch_size=16, spatial_merge_size=2, num_position_embeddings=2304),
)
_VL_TRAINING = {
    "model": {"pretrained_model_name_or_path": "local/vl", "torch_dtype": "bfloat16"},
    "training": {"global_batch_size": 16, "micro_batch_size": 1, "max_grad_norm": 1.0},
    "accelerator": {"tp_size": 1, "pp_size": 2, "cp_size": 1, "ep_size": 1},
    "fsdp_config": {"dp_shard_size": 4},
    "activation_checkpoint": {"mode": "full"},
    "dataset": {"data_transform": {"max_seq_len": 4096}},
    "context": {"max_device_memory": "64GB", "device_num": 8},
}


def _small_deepseek(folder: str, interleave: int, **parallel: int) -> str:
    """A seven-layer DeepSeek, fully recomputed, with *interleave* chunks per stage.

    At DP 4, TP 2, PP 2, EP 2 and 4 micro-batches, but for what *parallel*
    states in MindFormers' ``parallel_config``.
    """
    with open(DEEPSEEK_YAML, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model = config["model"]["model_config"]
    model.update(num_layers=7, offset=0, pp_interleave_num=interleave)
    config["parallel_config"].update(data_parallel=4, model_parallel=2, pipeline_stage=2, expert_parallel=2,
                                     micro_batch_num=4)
    config["parallel_config"].update(parallel)
    config["moe_config"]["expert_num"] = 16
    config["recompute_config"]["recompute"] = True
    name = "_".join([f"deepseek_{interleave}"] + [f"{key}{value}" for key, value in sorted(parallel.items())])
    path = os.path.join(folder, f"{name}.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return path


def _stage_peaks(evaluator: EvaluatorV2, full_rec: bool) -> List[float]:
    """The memory model's peak of each stage, every layer fully recomputed or plain."""
    config = copy.deepcopy(evaluator.ccfg)
    config.full_rec, config.sel_rec = full_rec, False
    own = evaluator.ccfg
    evaluator.set_config(config)
    try:
        return [insight["Static"] + insight["Dynamic"] for insight in evaluator.estimate_peak_insight()]
    finally:
        evaluator.set_config(own)


def _stage_peaks_of(evaluator: EvaluatorV2, choice: RecomputeChoice) -> List[float]:
    """The memory model's peak of each stage with *choice* stated as recompute ranges."""
    ranges = []
    for item in choice.ranges:
        if item.option.recompute is None:
            ranges.append(RecomputeRange(first=item.first, count=item.count, option="full"))
        elif not item.option.recompute:
            ranges.append(RecomputeRange(first=item.first, count=item.count, option="none"))
        else:
            ranges.append(RecomputeRange(first=item.first, count=item.count, option="selective",
                                         ops=dict.fromkeys(item.option.recompute, "recompute")))
    config = copy.deepcopy(evaluator.ccfg)
    apply_exec(config, ExecSpec(recompute=tuple(ranges)))
    own = evaluator.ccfg
    evaluator.set_config(config)
    try:
        return [insight["Static"] + insight["Dynamic"] for insight in evaluator.estimate_peak_insight()]
    finally:
        evaluator.set_config(own)


def _with_capacity(evaluator: EvaluatorV2, megabytes: float) -> EvaluatorV2:
    """*evaluator*, its device holding *megabytes*."""
    evaluator.ccfg.device_capacity.set(Memory.from_mb(megabytes))
    return evaluator


def _plain_values(config: Any) -> Dict[str, Any]:
    """The config's plain attributes."""
    return {name: value for name, value in vars(config).items() if isinstance(value, (bool, int, float, str))}


def _is_plain(option: LayerOption) -> bool:
    """Whether the option recomputes nothing."""
    return option.recompute == frozenset()


def _option(recompute: Any) -> LayerOption:
    """An option recomputing *recompute*, costs aside."""
    return LayerOption(recompute=recompute, memory_per_micro_batch=0.0, memory_once=0.0, forward_time=1.0,
                       backward_time=2.0)


class TestOnDeepSeek(unittest.TestCase):
    """DeepSeek-V3 at its own strategy: 61 layers over 16 stages, 32 micro-batches, 58 GB."""

    @classmethod
    def setUpClass(cls) -> None:
        """The fixture's evaluator and its choice."""
        cls.evaluator = EvaluatorV2(DEEPSEEK_YAML, framework="mindformers", log_level=0)
        cls.choice = choose_recompute(cls.evaluator, Hard.Device_A2)

    def test_every_layer_gets_one_option_in_model_order(self):
        """
        Feature: choose_recompute.
        Description: The ranges of the choice.
        Expectation: They cover the 61 layers once, in order, each range
            one kind as the model gives it.
        """
        kinds = layer_kinds(self.evaluator.ccfg)
        start = 0
        for item in self.choice.ranges:
            self.assertEqual(item.first, start)
            self.assertEqual(set(kinds[item.first:item.first + item.count]), {item.kind})
            start += item.count
        self.assertEqual(start, len(kinds))

    def test_the_choice_fits_and_saves_time(self):
        """
        Feature: choose_recompute.
        Description: The memory and savings of the choice, every layer fully
            recomputed as the config says.
        Expectation: Every stage fits the device, none is slower, and some
            are faster.
        """
        capacity = self.evaluator.ccfg.device_capacity.to_mb().size
        self.assertEqual(len(self.choice.stage_memory), 16)
        self.assertLessEqual(self.choice.memory, capacity)
        self.assertTrue(all(saved >= 0 for saved in self.choice.stage_savings))
        self.assertGreater(sum(self.choice.stage_savings), 0)
        self.assertTrue(any(not _is_plain(item.option) for item in self.choice.ranges))

    def test_the_micro_batches_in_flight_are_the_memory_models(self):
        """
        Feature: micro_batches_in_flight.
        Description: 1F1B over 16 stages with 32 micro-batches.
        Expectation: Stage s keeps 16 - s in flight.
        """
        self.assertEqual(micro_batches_in_flight(self.evaluator), [[16 - stage] for stage in range(16)])

    def test_asking_twice_gives_the_same_choice_and_leaves_the_config_alone(self):
        """
        Feature: choose_recompute.
        Description: Choose again.
        Expectation: The same choice, and the evaluator's config and peak as
            they were.
        """
        before = _plain_values(self.evaluator.ccfg)
        peak = self.evaluator.estimate_peak()
        self.assertEqual(choose_recompute(self.evaluator, Hard.Device_A2), self.choice)
        self.assertEqual(_plain_values(self.evaluator.ccfg), before)
        self.assertEqual(self.evaluator.estimate_peak(), peak)


class TestStageMemory(unittest.TestCase):
    """The memory the choice says each stage keeps is what the memory model says."""

    @classmethod
    def setUpClass(cls) -> None:
        """Seven DeepSeek layers, without and with two chunks per stage."""
        cls.folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.paths = {interleave: _small_deepseek(cls.folder.name, interleave) for interleave in (1, 2)}

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the configs."""
        cls.folder.cleanup()

    def _evaluator(self, interleave: int) -> EvaluatorV2:
        """A fresh evaluator of the small DeepSeek."""
        return EvaluatorV2(self.paths[interleave], framework="mindformers", log_level=0)

    def test_a_roomy_device_runs_every_layer_plain(self):
        """
        Feature: choose_recompute.
        Description: A 1 TB device, one and two chunks per stage.
        Expectation: Every layer runs plain, and each stage keeps what the
            memory model says it keeps with every layer plain, to its MB.
        """
        for interleave in (1, 2):
            evaluator = _with_capacity(self._evaluator(interleave), 1024 * 1024)
            choice = choose_recompute(evaluator, Hard.Device_A2)
            self.assertTrue(all(_is_plain(item.option) for item in choice.ranges), interleave)
            for mine, model in zip(choice.stage_memory, _stage_peaks(evaluator, full_rec=False)):
                self.assertLessEqual(abs(mine - model), 1.0, interleave)

    def test_a_device_the_fully_recomputed_stages_fill_keeps_them_so(self):
        """
        Feature: choose_recompute.
        Description: A device a little larger than the heaviest stage fully
            recomputed, one and two chunks per stage.
        Expectation: The heaviest stage stays fully recomputed, and every
            stage fits.
        """
        for interleave in (1, 2):
            evaluator = self._evaluator(interleave)
            full = _stage_peaks(evaluator, full_rec=True)
            evaluator = _with_capacity(evaluator, max(full) + 16)
            choice = choose_recompute(evaluator, Hard.Device_A2)
            self.assertLessEqual(choice.memory, max(full) + 16)
            heaviest = full.index(max(full))
            self.assertEqual(choice.stage_savings[heaviest], 0.0, interleave)

    def test_a_smaller_device_never_runs_faster(self):
        """
        Feature: choose_recompute.
        Description: Devices from all plain down to all fully recomputed.
        Expectation: Every choice fits, and the time saved never grows as the
            device shrinks.
        """
        evaluator = self._evaluator(1)
        plain, full = max(_stage_peaks(evaluator, full_rec=False)), max(_stage_peaks(evaluator, full_rec=True))
        saved = []
        for share in (1.0, 0.75, 0.5, 0.25, 0.0):
            capacity = full + 16 + share * (plain - full)
            choice = choose_recompute(_with_capacity(evaluator, capacity), Hard.Device_A2)
            self.assertLessEqual(choice.memory, capacity)
            saved.append(sum(choice.stage_savings))
        self.assertEqual(saved, sorted(saved, reverse=True))
        self.assertGreater(saved[0], saved[-1])

    def test_a_mix_keeps_what_the_config_priced_whole_keeps(self):
        """
        Feature: choose_recompute.
        Description: Devices between all plain and all fully recomputed, one
            and two chunks per stage. Each choice that holds one selective
            setting at most is stated as recompute ranges and the whole
            config priced with them.
        Expectation: Each stage keeps what the choice says it keeps, to its
            MB, a choice whose last layer ends warm-up on a selective option
            included.
        """
        selective_ends = 0
        for interleave in (1, 2):
            evaluator = self._evaluator(interleave)
            plain, full = max(_stage_peaks(evaluator, full_rec=False)), max(_stage_peaks(evaluator, full_rec=True))
            for share in (0.7, 0.4, 0.35, 0.2, 0.175):
                choice = choose_recompute(_with_capacity(evaluator, full + 16 + share * (plain - full)),
                                          Hard.Device_A2)
                settings = {item.option.recompute for item in choice.ranges if item.option.recompute}
                if len(settings) > 1:
                    continue
                selective_ends += bool(choice.ranges[-1].option.recompute)
                for mine, model in zip(choice.stage_memory, _stage_peaks_of(evaluator, choice)):
                    self.assertLessEqual(abs(mine - model), 1.0, (interleave, share, describe(choice)))
        self.assertGreater(selective_ends, 0)

    def test_a_stage_between_the_first_and_the_last_keeps_what_the_config_priced_whole_keeps(self):
        """
        Feature: choose_recompute.
        Description: The small DeepSeek at DP 2, TP 4, PP 4 with 8
            micro-batches, whose stages keep 4, 3, 2 and 1 in flight, and
            devices between all plain and all fully recomputed. On the second
            stage a DP buffer hides the dense layer's gathers of the first
            micro-batches. Each choice that holds one selective setting at
            most is stated as recompute ranges and the whole config priced
            with them.
        Expectation: Each stage keeps what the choice says it keeps, to its
            MB.
        """
        path = _small_deepseek(self.folder.name, 1, data_parallel=2, model_parallel=4, pipeline_stage=4,
                               micro_batch_num=8)
        evaluator = EvaluatorV2(path, framework="mindformers", log_level=0)
        self.assertEqual(micro_batches_in_flight(evaluator), [[4], [3], [2], [1]])
        plain, full = max(_stage_peaks(evaluator, full_rec=False)), max(_stage_peaks(evaluator, full_rec=True))
        for share in (0.1, 0.5, 0.9):
            choice = choose_recompute(_with_capacity(evaluator, full + 16 + share * (plain - full)), Hard.Device_A2)
            self.assertLessEqual(len({item.option.recompute for item in choice.ranges if item.option.recompute}), 1)
            for mine, model in zip(choice.stage_memory, _stage_peaks_of(evaluator, choice)):
                self.assertLessEqual(abs(mine - model), 1.0, (share, describe(choice)))


# A link so fast that a copy takes no time, and one too slow for any layer's.
_FAST_LINK = Hard.HostLink(gib_per_s=10.0 ** 6, sustained_tflops=140.0)
_SLOW_LINK = Hard.HostLink(gib_per_s=10.0 ** -3, sustained_tflops=140.0)


def _offloaded_indices(choice: RecomputeChoice) -> List[int]:
    """The layers the choice offloads, in model order."""
    return [index for item in choice.ranges if item.option.link_bandwidth
            for index in range(item.first, item.first + item.count)]


class TestOffloadWindow(unittest.TestCase):
    """A stage's first layers offload while their copies keep pace with the forward."""

    @staticmethod
    def _limit(copies: Sequence[float], ending: Optional[int] = None) -> int:
        """How many of four layers, each with a forward of 1, may offload with *copies* to the host."""
        stage_layers = []
        for position in range(len(copies)):
            key = ("unit", None) + (("ends warm-up",) if position == ending else ())
            stage_layers.append(Candidate._Layer(position, key, 1, _option(frozenset())))  # pylint: disable=protected-access
        plain = [LayerOption(recompute=frozenset(), memory_per_micro_batch=copy, memory_once=0.0, forward_time=1.0,
                             backward_time=2.0) for copy in copies]
        link = Candidate._Link(per_byte=1.0, overlap=1.0)  # pylint: disable=protected-access
        return Candidate._offload_limit(stage_layers, plain, link)  # pylint: disable=protected-access

    def test_a_layer_cannot_offload_more_than_the_forward_left_after_it(self):
        """
        Feature: the offload window.
        Description: Four layers with a forward of 1 each, whose copies take
            half a forward, one forward, one and a half forwards, or four.
        Expectation: Each offloaded layer's copy, with those after it, fits
            in the forward left after it: three layers at half and at one
            forward, two at one and a half, none at four; the last layer,
            with no forward left after it, never.
        """
        self.assertEqual(self._limit([0.5] * 4), 3)
        self.assertEqual(self._limit([1.0] * 4), 3)
        self.assertEqual(self._limit([1.5] * 4), 2)
        self.assertEqual(self._limit([4.0] * 4), 0)

    def test_the_layer_that_ends_warm_up_keeps_its_activations(self):
        """
        Feature: the offload window.
        Description: Copies that take no time, the third layer ending warm-up.
        Expectation: Only the two layers before it may offload.
        """
        self.assertEqual(self._limit([0.0] * 4, ending=2), 2)


class TestOffload(unittest.TestCase):
    """With a host link, a choice per layer may offload each stage's first layers."""

    @classmethod
    def setUpClass(cls) -> None:
        """The small DeepSeek at PP 1 and at PP 2, and at two chunks per stage."""
        cls.folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.paths = {
            "pp1": _small_deepseek(cls.folder.name, 1, pipeline_stage=1, data_parallel=8),
            "pp2": _small_deepseek(cls.folder.name, 1),
            "vpp": _small_deepseek(cls.folder.name, 2),
        }

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the configs."""
        cls.folder.cleanup()

    def _evaluator(self, name: str, share: float) -> EvaluatorV2:
        """The small DeepSeek *name*, its device *share* of the way from all fully recomputed to all plain."""
        evaluator = EvaluatorV2(self.paths[name], framework="mindformers", log_level=0)
        plain, full = max(_stage_peaks(evaluator, full_rec=False)), max(_stage_peaks(evaluator, full_rec=True))
        return _with_capacity(evaluator, full + 16 + share * (plain - full))

    def test_early_layers_offload_and_the_last_cannot(self):
        """
        Feature: choose_recompute with a host link.
        Description: PP 1, a device a fifth of the way from all fully
            recomputed to all plain, A2's placeholder link and one whose
            copies take no time.
        Expectation: Each offloads a run of first layers, which run plain,
            the faster link more of them, and never the last layer.
        """
        evaluator = self._evaluator("pp1", 0.2)
        last = len(layer_kinds(evaluator.ccfg)) - 1
        offloaded = {}
        for name, link in (("A2", Hard.Device_A2.host_link), ("fast", _FAST_LINK)):
            choice = choose_recompute(evaluator, Hard.Device_A2, link=link)
            offloaded[name] = _offloaded_indices(choice)
            self.assertEqual(offloaded[name], list(range(len(offloaded[name]))), name)
            self.assertNotIn(last, offloaded[name], name)
            self.assertTrue(all(item.option.recompute == frozenset() for item in choice.ranges
                                if item.option.link_bandwidth), name)
        self.assertGreater(len(offloaded["A2"]), 0)
        self.assertGreater(len(offloaded["fast"]), len(offloaded["A2"]))

    def test_an_offloading_choice_fits_and_is_no_slower(self):
        """
        Feature: choose_recompute with a host link.
        Description: PP 1 and PP 2, devices a fifth and three fifths of the
            way from all fully recomputed to all plain, with a link whose
            copies take no time.
        Expectation: Every stage fits the device, and the layers save at
            least as much time as they do without the link, more in some.
        """
        gained = 0
        for name in ("pp1", "pp2"):
            for share in (0.2, 0.6):
                evaluator = self._evaluator(name, share)
                capacity = evaluator.ccfg.device_capacity.to_mb().size
                without = choose_recompute(evaluator, Hard.Device_A2)
                choice = choose_recompute(evaluator, Hard.Device_A2, link=_FAST_LINK)
                self.assertLessEqual(choice.memory, capacity, (name, share))
                self.assertGreaterEqual(sum(choice.stage_savings), sum(without.stage_savings), (name, share))
                gained += sum(choice.stage_savings) > sum(without.stage_savings)
        self.assertGreater(gained, 0)

    def test_a_link_too_slow_for_any_layer_changes_nothing(self):
        """
        Feature: choose_recompute with a host link.
        Description: PP 1, a link too slow for any layer's copy.
        Expectation: The choice made without a link.
        """
        evaluator = self._evaluator("pp1", 0.2)
        self.assertEqual(choose_recompute(evaluator, Hard.Device_A2, link=_SLOW_LINK),
                         choose_recompute(evaluator, Hard.Device_A2))

    def test_one_mode_and_two_chunks_per_stage_do_not_offload(self):
        """
        Feature: choose_recompute with a host link.
        Description: One mode for every layer, and two chunks per stage.
        Expectation: Each the choice made without a link: a runtime with one
            mode runs no offload, and offload is priced at one chunk per
            stage.
        """
        evaluator = self._evaluator("pp1", 0.2)
        self.assertEqual(choose_recompute(evaluator, Hard.Device_A2, modes=MODES, link=_FAST_LINK),
                         choose_recompute(evaluator, Hard.Device_A2, modes=MODES))
        evaluator = self._evaluator("vpp", 0.2)
        self.assertEqual(choose_recompute(evaluator, Hard.Device_A2, link=_FAST_LINK),
                         choose_recompute(evaluator, Hard.Device_A2))

    def test_an_offloading_choice_reads_so(self):
        """
        Feature: describe and to_records.
        Description: A choice that offloads its first layers.
        Expectation: Their line says so, and their record states offload.
        """
        choice = choose_recompute(self._evaluator("pp1", 0.2), Hard.Device_A2, link=_FAST_LINK)
        first = choice.ranges[0]
        self.assertTrue(first.option.link_bandwidth)
        self.assertTrue(describe(choice).splitlines()[0].endswith("no recompute, offloaded to the host"))
        self.assertEqual(to_records(choice)[0]["offload"], True)
        self.assertTrue(all("offload" not in record for record, item in zip(to_records(choice), choice.ranges)
                            if not item.option.link_bandwidth))


class TestOneMode(unittest.TestCase):
    """One mode for every layer, the way a runtime with one activation checkpoint mode runs them."""

    @classmethod
    def setUpClass(cls) -> None:
        """Seven DeepSeek layers over two stages."""
        cls.folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        cls.path = _small_deepseek(cls.folder.name, 1)

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the config."""
        cls.folder.cleanup()

    def _evaluator(self) -> EvaluatorV2:
        """A fresh evaluator of the small DeepSeek."""
        return EvaluatorV2(self.path, framework="mindformers", log_level=0)

    def test_a_roomy_device_runs_every_layer_off(self):
        """
        Feature: choose_recompute modes.
        Description: A 1 TB device, choosing between off and full.
        Expectation: Off, every layer plain, and each stage keeps what the
            memory model says it keeps with every layer plain, to its MB.
        """
        evaluator = _with_capacity(self._evaluator(), 1024 * 1024)
        choice = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"))
        self.assertEqual(choice.mode, "off")
        self.assertTrue(all(_is_plain(item.option) for item in choice.ranges))
        for mine, model in zip(choice.stage_memory, _stage_peaks(evaluator, full_rec=False)):
            self.assertLessEqual(abs(mine - model), 1.0)

    def test_a_device_only_full_recompute_fits_keeps_it(self):
        """
        Feature: choose_recompute modes.
        Description: A device a little larger than the heaviest stage fully
            recomputed, smaller than it plain.
        Expectation: Full, no stage saving any time, at the memory model's
            peak.
        """
        evaluator = self._evaluator()
        full = _stage_peaks(evaluator, full_rec=True)
        self.assertLess(max(full) + 16, max(_stage_peaks(evaluator, full_rec=False)))
        choice = choose_recompute(_with_capacity(evaluator, max(full) + 16), Hard.Device_A2, modes=MODES)
        self.assertEqual(choice.mode, "full")
        self.assertEqual(choice.stage_savings, tuple(0.0 for _ in full))
        self.assertLessEqual(abs(choice.memory - max(full)), 1.0)

    def test_a_choice_per_layer_is_as_fast_as_one_mode(self):
        """
        Feature: choose_recompute modes.
        Description: The heaviest stage, plain, a little too big for the
            device: one mode for every layer, and a choice per layer.
        Expectation: One mode must recompute every layer fully; the choice
            per layer saves more time, still within the device.
        """
        evaluator = self._evaluator()
        capacity = max(_stage_peaks(evaluator, full_rec=False)) - 64
        _with_capacity(evaluator, capacity)
        one = choose_recompute(evaluator, Hard.Device_A2, modes=("off", "full"))
        each = choose_recompute(evaluator, Hard.Device_A2)
        self.assertEqual(one.mode, "full")
        self.assertIsNone(each.mode)
        self.assertGreater(sum(each.stage_savings), sum(one.stage_savings))
        self.assertLessEqual(each.memory, capacity)

    def test_the_modes_recompute_what_their_names_say(self):
        """
        Feature: mode_recompute.
        Description: Each mode, with the switches a config sets.
        Expectation: Off recomputes nothing, full everything, selective the
            switches set to 0; another name is refused.
        """
        configured = {"attBMM": 1, "normOp": 0, "ffAct": 0}
        self.assertEqual(mode_recompute("off", configured), frozenset())
        self.assertIsNone(mode_recompute("full", configured))
        self.assertEqual(mode_recompute("selective", configured), frozenset({"normOp", "ffAct"}))
        with self.assertRaises(ValueError):
            mode_recompute("sometimes", configured)


class TestNoChoice(unittest.TestCase):
    """What the choice does not cover."""

    def test_a_multimodal_model_has_no_choice(self):
        """
        Feature: choose_recompute.
        Description: A multimodal model.
        Expectation: No choice.
        """
        evaluator = SimpleNamespace(ccfg=SimpleNamespace(multimodal=True, pp_sched="1f1b"))
        self.assertIsNone(choose_recompute(evaluator, Hard.Device_A2))

    def test_another_schedule_has_no_choice(self):
        """
        Feature: choose_recompute.
        Description: A V-shaped schedule, whose end of warm-up the budget
            does not follow.
        Expectation: No choice.
        """
        evaluator = SimpleNamespace(ccfg=SimpleNamespace(multimodal=False, pp_sched="zero_bubble_v"))
        self.assertIsNone(choose_recompute(evaluator, Hard.Device_A2))

    def test_a_stage_that_does_not_fit_fully_recomputed_has_no_choice(self):
        """
        Feature: choose_recompute.
        Description: A device smaller than DeepSeek-V3's heaviest stage
            fully recomputed.
        Expectation: No choice.
        """
        evaluator = EvaluatorV2(DEEPSEEK_YAML, framework="mindformers", log_level=0)
        heaviest = max(insight["Static"] + insight["Dynamic"] for insight in evaluator.estimate_peak_insight())
        self.assertIsNone(choose_recompute(_with_capacity(evaluator, heaviest - 1024), Hard.Device_A2))


class TestDescribe(unittest.TestCase):
    """How a choice reads."""

    def test_each_range_reads_as_one_line(self):
        """
        Feature: describe.
        Description: Three ranges of a model without kinds: plain, some ops
            recomputed, and fully recomputed.
        Expectation: One line per range, its ops in the switches' order.
        """
        choice = RecomputeChoice(
            ranges=(LayerRange(0, 2, None, _option(frozenset())),
                    LayerRange(2, 1, None, _option(frozenset({"ffAct", "attBMM"}))),
                    LayerRange(3, 4, None, _option(None))),
            stage_memory=(1.0, 2.0),
            stage_savings=(0.0, 0.0),
        )
        self.assertEqual(describe(choice).splitlines(),
                         ["layers 0-1: no recompute", "layer 2: recompute attBMM+ffAct", "layers 3-6: full recompute"])
        self.assertEqual(option_label(_option(None)), "full recompute")
        self.assertEqual(choice.memory, 2.0)
        self.assertEqual(to_records(choice), [
            {"first": 0, "count": 2, "kind": None, "recompute": "none"},
            {"first": 2, "count": 1, "kind": None, "recompute": ["attBMM", "ffAct"]},
            {"first": 3, "count": 4, "kind": None, "recompute": "full"},
        ])

    def test_one_mode_reads_first(self):
        """
        Feature: describe.
        Description: A choice of one mode for every layer.
        Expectation: Its mode on the first line.
        """
        choice = RecomputeChoice(ranges=(LayerRange(0, 4, None, _option(frozenset())),), stage_memory=(1.0,),
                                 stage_savings=(0.0,), mode="off")
        self.assertEqual(describe(choice).splitlines(), ["every layer: off", "layers 0-3: no recompute"])


class TestAutoRecomputeSearch(unittest.TestCase):
    """The search scores each candidate with its layers' options."""

    @staticmethod
    def _scored(auto: bool):
        """The search over TP and PP of the dense model on eight devices, and its runner."""
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()
        runner = Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"),
                                 dimensions=[Dim.TP, Dim.PP], auto_recompute=auto).instance
        results, _ = runner.device_loops(({}, 0), None)
        space = [(config, peak) for config, peak in results.items() if runner.mem_eval.mem_fit(peak)]
        scored, _ = runner.order_search_space(space, None, None)
        return scored, runner

    def test_every_candidate_is_as_fast_or_faster_and_still_fits(self):
        """
        Feature: ParallelizeLayer auto_recompute.
        Description: Order the same space fully recomputed and with auto
            recompute.
        Expectation: Every candidate gets its layers' options, none scores
            worse or stops fitting the device, and some score better.
        """
        set_verbose_level(1)
        full, _ = self._scored(False)
        auto, runner = self._scored(True)
        self.assertEqual(len(auto), len(full))
        own = {str(config): score for config, _, score, _ in full}
        for config, memory, score, _ in auto:
            self.assertIn(config, runner.recompute_choices)
            self.assertLessEqual(score, own[str(config)])
            self.assertTrue(runner.mem_eval.mem_fit(memory))
        self.assertLess(auto[0][2], full[0][2])

    def test_offload_needs_a_choice_per_layer_and_takes_the_device_link(self):
        """
        Feature: ParallelizeLayer auto_offload.
        Description: Ask for offload without auto recompute, then with it,
            then with a link of one's own.
        Expectation: Refused without it; with it, the device's link, or the
            one given.
        """
        with self.assertRaises(ValueError):
            Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                            auto_offload=True)
        runner = Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                                 auto_recompute=True, auto_offload=True).instance
        self.assertEqual(runner.offload_link, Hard.Device_A2.host_link)
        runner = Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                                 auto_recompute=True, auto_offload=True, host_link=_FAST_LINK).instance
        self.assertEqual(runner.offload_link, _FAST_LINK)

    def test_the_recompute_cannot_also_come_from_the_config(self):
        """
        Feature: ParallelizeLayer auto_recompute.
        Description: Ask for auto recompute and for the config's recompute,
            then for a mode no runtime has.
        Expectation: Both refused.
        """
        with self.assertRaises(ValueError):
            Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                            auto_recompute=True, mppb=True)
        with self.assertRaises(ValueError):
            Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"), dimensions=[Dim.TP],
                            auto_recompute=True, recompute_modes=("off", "sometimes"))

    def test_one_mode_per_candidate_and_the_per_layer_choice_of_one(self):
        """
        Feature: ParallelizeLayer recompute_modes and recompute_per_layer.
        Description: Order the space choosing off or full for every layer,
            then ask the best configuration for a choice per layer.
        Expectation: Every candidate gets one of the two modes; the choice
            per layer scores no worse than the best candidate's mode.
        """
        set_verbose_level(1)
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()
        runner = Par.Parallelize("hyper_v2", copy.deepcopy(_DENSE), Hard.Machine(8, "A2"),
                                 dimensions=[Dim.TP, Dim.PP], auto_recompute=True,
                                 recompute_modes=("off", "full")).instance
        results, _ = runner.device_loops(({}, 0), None)
        space = [(config, peak) for config, peak in results.items() if runner.mem_eval.mem_fit(peak)]
        scored, _ = runner.order_search_space(space, None, None)
        self.assertEqual({choice.mode for choice in runner.recompute_choices.values()} - {"off", "full"}, set())
        self.assertEqual(len(runner.recompute_choices), len(scored))
        per_layer, score = runner.recompute_per_layer(scored[0][0])
        self.assertIsNone(per_layer.mode)
        self.assertLessEqual(score, scored[0][2] * (1 + 1e-12))

    def test_a_multimodal_search_chooses_for_the_model_it_prices(self):
        """
        Feature: ParallelizeMultiModal auto_recompute.
        Description: A vision-language model, whose search prices its text
            model (IR finding F2).
        Expectation: Every candidate gets options for the text model's
            layers.
        """
        set_verbose_level(1)
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "vl.yaml")
            with open(path, "w", encoding="utf-8") as handle:
                yaml.safe_dump(_VL_TRAINING, handle)
            with patch.object(_hf_model_spec, "_get_hf_config", return_value=_VL), \
                    patch.dict(os.environ, {"MPLCONFIGDIR": folder}):
                runner = Par.Parallelize("hyper_v2", path, Hard.Machine(8, "A2"), global_batch_size=16,
                                         dimensions=[Dim.DP], auto_recompute=True).instance
                self.assertIsInstance(runner, Par.ParallelizeMultiModal)
                results, _ = runner.device_loops(({}, 0), None)
                space = [(config, peak) for config, peak in results.items() if runner.mem_eval.mem_fit(peak)]
                scored, _ = runner.order_search_space(space, None, None)
        self.assertTrue(scored)
        text_layers = len(layer_kinds(runner.config.ccfg))
        for config, _, _, _ in scored:
            ranges = runner.recompute_choices[config].ranges
            self.assertEqual(sum(item.count for item in ranges), text_layers)


if __name__ == "__main__":
    unittest.main()
