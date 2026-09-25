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
"""Tests for how the performance path assigns layers to their groups.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/perf_estimation/test_getters.py -v
"""
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import patch

import yaml

from hyper_parallel.auto_parallel._layer_census import KindActivations
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.comm import EvalLayerComm
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalAttn
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import check_and_apply_custom_hook
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.framework_parsers._cost_model_parser import (
    HYPER_SELECTIVE_REC_OP,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import estimate_comm
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import estimate_comp
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.getters import (
    get_layer_configs_by_position,
    get_model_order,
    get_recomp_factor,
    get_table_quantity,
    selective_shares,
)
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig

EMB, OUT, LAY = LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER, LayerType.NOT_REC_LAYER
FULL, LINEAR = "full_attention", "linear_attention"


def _marking(kind: str):
    """A layer hook that only records which group it belongs to."""

    def hook(cfg: Any) -> None:
        """Mark *cfg* with the group's kind."""
        cfg.kind = kind

    return hook


def _config(groups, p: int = 1, vp: int = 1, sched: str = "1f1b") -> SimpleNamespace:
    """A bare config with *groups* as its layer_custom_config."""
    return SimpleNamespace(p=p, vp=vp, pp_sched=sched, n_lay=0, kind=None,
                           layer_custom_config=[(count, _marking(kind)) for count, kind in groups])


def _kinds_in_model_order(cfg, stages):
    """The group each regular layer is priced with, in model order."""
    by_position = get_layer_configs_by_position(cfg, stages)
    return [by_position[position].kind for position in get_model_order(cfg, stages)]


def _hybrid_config(folder: str, layer_types: List[str], pp: int = 1, vp: int = 1,
                   sched: str = "1f1b") -> CostModelConfig:
    """A Qwen3.5-style stack of *layer_types*, parsed by the hyper_v2 front end."""
    hf_config = SimpleNamespace(
        model_type="qwen3_5_moe", hidden_size=1024, num_hidden_layers=len(layer_types),
        num_attention_heads=8, num_key_value_heads=2, head_dim=128, vocab_size=32000,
        max_position_embeddings=8192, num_experts=16, num_experts_per_tok=4,
        moe_intermediate_size=256, shared_expert_intermediate_size=256,
        attn_output_gate=True, layer_types=list(layer_types),
        linear_num_key_heads=8, linear_key_head_dim=64, linear_num_value_heads=16,
        linear_value_head_dim=64, linear_conv_kernel_dim=4,
    )
    train = {
        "model": {"pretrained_model_name_or_path": "local/unit", "torch_dtype": "bfloat16"},
        "training": {"global_batch_size": 8, "micro_batch_size": 1},
        "accelerator": {"tp_size": 1, "pp_size": pp, "pp_interleave_num": vp,
                        "pipeline_scheduler": sched},
        "fsdp_config": {"dp_shard_size": 2},
        "dataset": {"data_transform": {"max_seq_len": 2048}},
        "context": {"max_device_memory": "64GB", "device_num": 2 * pp},
    }
    path = os.path.join(folder, f"hybrid_{len(os.listdir(folder))}.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(train, handle)
    with patch("hyper_parallel.auto_parallel._hf_model_spec._get_hf_config", return_value=hf_config):
        ccfg = CostModelConfig(path, framework="hyper_v2")
    check_and_apply_custom_hook(ccfg)
    return ccfg


class TestModelOrder(unittest.TestCase):
    """Model order is chunk by chunk across the stages, not stage by stage."""

    def test_without_interleaving_it_is_stage_order(self):
        """
        Feature: model order.
        Description: One chunk per stage.
        Expectation: The stages in turn, skipping embedding and output.
        """
        stages = [[[EMB, LAY, LAY]], [[LAY, LAY, OUT]]]
        self.assertEqual(get_model_order(_config([], p=2), stages),
                         [(0, 0, 1), (0, 0, 2), (1, 0, 0), (1, 0, 1)])

    def test_interleaved_chunks_run_across_the_stages(self):
        """
        Feature: model order.
        Description: Two chunks per stage: the first chunk of every stage
            comes before the second chunk of any.
        Expectation: s0c0, s1c0, s0c1, s1c1.
        """
        stages = [[[EMB, LAY], [LAY]], [[LAY], [LAY, OUT]]]
        self.assertEqual(get_model_order(_config([], p=2, vp=2), stages),
                         [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0)])

    def test_a_v_schedule_climbs_back_up(self):
        """
        Feature: model order.
        Description: A V schedule sends the second chunk back up the stages,
            the order the memory backbone assumes for it.
        Expectation: s0c0, s1c0, then s1c1 before s0c1.
        """
        stages = [[[EMB, LAY], [LAY]], [[LAY], [LAY, OUT]]]
        order = get_model_order(_config([], p=2, vp=2, sched="zero_bubble_v"), stages)
        self.assertEqual(order, [(0, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0)])


class TestGroupBoundaries(unittest.TestCase):
    """Each group covers exactly its own count of layers."""

    def test_each_group_covers_its_count(self):
        """
        Feature: layer groups on the performance path.
        Description: Three groups of 1, 2 and 1 layers on one stage.  The
            boundary used to move one layer early, giving the first group
            one layer less and the last one more.
        Expectation: A, B, B, C.
        """
        stages = [[[EMB, LAY, LAY, LAY, LAY, OUT]]]
        cfg = _config([(1, "A"), (2, "B"), (1, "C")])
        self.assertEqual(_kinds_in_model_order(cfg, stages), ["A", "B", "B", "C"])

    def test_groups_follow_model_order_across_chunks(self):
        """
        Feature: layer groups under interleaving.
        Description: Two groups of two layers on two stages of two chunks.
        Expectation: The first group fills the first chunk of both stages.
        """
        stages = [[[EMB, LAY], [LAY]], [[LAY], [LAY, OUT]]]
        cfg = _config([(2, "A"), (2, "B")], p=2, vp=2)
        by_position = get_layer_configs_by_position(cfg, stages)
        self.assertEqual(by_position[(1, 0, 0)].kind, "A")
        self.assertEqual(by_position[(0, 1, 0)].kind, "B")

    def test_layers_past_the_groups_keep_the_last(self):
        """
        Feature: layer groups on the performance path.
        Description: The groups declare fewer layers than the stages hold.
        Expectation: The extra layers keep the last group.
        """
        stages = [[[EMB, LAY, LAY, LAY, OUT]]]
        cfg = _config([(1, "A"), (1, "B")])
        self.assertEqual(_kinds_in_model_order(cfg, stages), ["A", "B", "B"])

    def test_a_two_group_stack_costs_the_mean_of_its_groups(self):
        """
        Feature: layer groups on the compute path.
        Description: Two full then two linear layers on one stage, against
            four full and four linear layers.  Embedding and output cost the
            same in all three, so the mixed stack costs exactly the mean of
            the other two when each group covers its own two layers.
        Expectation: The mean; the early boundary priced one full layer
            and three linear ones.
        """
        stages = [[[EMB, LAY, LAY, LAY, LAY, OUT]]]
        with tempfile.TemporaryDirectory() as folder:
            mixed, full, linear = (
                estimate_comp(_hybrid_config(folder, kinds), CustomConfig(), stages)[0]
                for kinds in ([FULL, FULL, LINEAR, LINEAR], [FULL] * 4, [LINEAR] * 4)
            )
        self.assertNotAlmostEqual(full / linear, 1.0, places=3)
        self.assertAlmostEqual(mixed / ((full + linear) / 2), 1.0, places=9)


class TestPerformanceAgreesWithMemory(unittest.TestCase):
    """Both estimators give every layer the same kind."""

    def setUp(self) -> None:
        """A scratch folder for the stacks' config files."""
        folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(folder.cleanup)
        self.folder = folder.name

    def _stack(self, sched: str) -> CostModelConfig:
        """Eight layers on two stages of two chunks, alternating kinds by pairs."""
        return _hybrid_config(self.folder, [FULL, FULL, LINEAR, LINEAR] * 2, pp=2, vp=2, sched=sched)

    def _check(self, sched: str) -> None:
        """Assert the performance and memory paths give every layer the same kind."""
        ccfg = self._stack(sched)
        stages = ccfg.generate_partitions_vpp()
        perf = {pos: cfg.n_softmax for pos, cfg in get_layer_configs_by_position(ccfg, stages).items()}

        memory = {}

        def spy(cfg: CostModelConfig, ctx: Any) -> float:
            """The real score formula, recording the counts at each layer's first visit."""
            stage_id, chunk_id, lay_id = ctx.current_stage_id, ctx.current_chunk_id, ctx.current_lay_id
            # The backward-overhead pass revisits positions, output included.
            if isinstance(lay_id, int) and stages[stage_id][chunk_id][lay_id] == LAY:
                memory.setdefault((stage_id, chunk_id, lay_id), cfg.n_softmax)
            return EvalAttn.attn_score_activations(cfg, ctx)

        evaluator = EvaluatorV2(None, ccfg=ccfg)
        evaluator.set_attn_eval_fun(score=spy)
        evaluator.estimate_peak(stages=stages)
        self.assertEqual(memory, perf)
        self.assertEqual(sorted(perf.values()), [0, 0, 0, 0, 1, 1, 1, 1])

    def test_interleaved(self):
        """
        Feature: one layer order for every estimator.
        Description: Interleaved 1F1B, two chunks per stage.
        Expectation: The performance path gives each position the kind the
            memory backbone gives it.
        """
        self._check("1f1b")

    def test_comm_path_follows_model_order(self):
        """
        Feature: one layer order for every estimator.
        Description: The communication estimate applies each layer's group
            hook as it walks the stages one at a time.  Record the kind each
            layer's DP term sees, in that walk: under FSDP, the parts its
            traffic reads.
        Expectation: Stage 0 holds model layers 0-1 and 4-5, both full
            pairs; stage 1 holds the two linear pairs.
        """
        ccfg = self._stack("1f1b")
        stages = ccfg.generate_partitions_vpp()
        seen = []
        real = EvalLayerComm.fsdp_layer_parts

        def spy(cfg: CostModelConfig, ctx: Any) -> tuple:
            """The real parts, recording the kind of the layer they price."""
            seen.append(cfg.n_softmax)
            return real(cfg, ctx)

        with patch.object(EvalLayerComm, "fsdp_layer_parts", side_effect=spy):
            estimate_comm(ccfg, CustomConfig(), stages, Hard.Device_A2)
        self.assertEqual(seen, [1, 1, 1, 1, 0, 0, 0, 0])

    def test_v_schedule(self):
        """
        Feature: one layer order for every estimator.
        Description: A V schedule, second chunk back up the stages.
        Expectation: The performance path gives each position the kind the
            memory backbone gives it.
        """
        self._check("zero_bubble_v")


_SWITCHES = {"softmax": 0, "normOp": 1}


def _factor(rec_op: Any, op_name: str, layer: LayerType = LayerType.SEL_REC_LAYER) -> int:
    """The recompute factor of *op_name* in a layer with these switches."""
    return get_recomp_factor(SimpleNamespace(rec_op=rec_op), layer, op_name)


class TestFsdpTraffic(unittest.TestCase):
    """The bytes FSDP's collectives move, as the time model's DP term."""

    def test_a_ring_moves_n_minus_one_over_n(self):
        """
        Feature: EvalLayerComm.fsdp_traffic.
        Description: A part of 100 parameters over 1 rank, sharded over 4 and
            held by 16, bf16, gathered twice a micro-batch, its copies
            all-reducing every other; and one no rank shards, held by 4.
        Expectation: Each collective over n shards moves (n - 1) / n of the
            part: two gathers and a reduce-scatter of 3 / 4, and an
            all-reduce of the shard over its 4 copies, 2 * 3 / 4, halved;
            the unsharded part only all-reduces.
        """
        ccfg = SimpleNamespace(bytes_p=2, bytes_grad=2)
        self.assertAlmostEqual(EvalLayerComm.fsdp_traffic(ccfg, ((100, 1, 4, 16),), 2, 0.5),
                               0.75 * 100 * (2 * 2 + 2) + 2 * 0.75 * 25 * 2 * 0.5)
        self.assertAlmostEqual(EvalLayerComm.fsdp_traffic(ccfg, ((100, 1, 1, 4),), 2, 1),
                               2 * 0.75 * 100 * 2)

    def test_the_walk_counts_each_micro_batch_s_collectives(self):
        """
        Feature: the DP term of the communication walk under FSDP.
        Description: A HyperParallel stack at DP shard 2, four micro-batches,
            with its FSDP resharding and keeping the parameters gathered.
        Expectation: Resharded, each layer is gathered twice a micro-batch
            and the root once; kept, each once a step.  No copy, so no
            all-reduce, but the group shards: once a step.
        """
        self.assertEqual(self._rounds(True), [(1, 1, 0.25), (3, 2, 0.25)])
        self.assertEqual(self._rounds(False), [(1, 0.25, 0.25), (3, 0.25, 0.25)])

    def _rounds(self, reshards: bool) -> list:
        """The parts, gathers and all-reduces the walk passes FSDP's traffic, with *reshards*."""
        ccfg = _hybrid_config(self.folder, [FULL, LINEAR])
        ccfg.reshards = reshards
        calls = []
        real = EvalLayerComm.fsdp_traffic

        def spy(cfg: CostModelConfig, parts: tuple, gathers: float, reduces: float) -> float:
            """The real traffic, recording its rounds."""
            calls.append((len(parts), gathers, reduces))
            return real(cfg, parts, gathers, reduces)

        with patch.object(EvalLayerComm, "fsdp_traffic", side_effect=spy):
            estimate_comm(ccfg, CustomConfig(), ccfg.generate_partitions_vpp(), Hard.Device_A2)
        return sorted(set(calls))

    def setUp(self):
        self.folder = tempfile.mkdtemp()


class TestSelectiveRecompute(unittest.TestCase):
    """A switch at 0 recomputes its op, as the memory model reads it."""

    def test_a_switch_at_zero_is_recomputed(self):
        """softmax is switched to 0 and normOp to 1."""
        rec_op = Config(_SWITCHES)
        self.assertEqual(_factor(rec_op, "softmax"), 1)
        self.assertEqual(_factor(rec_op, "normOp"), 0)

    def test_an_op_without_a_switch_is_kept(self):
        """A Config answers 0 for an attribute it lacks; that must not read as recompute."""
        for rec_op in (Config(_SWITCHES), SimpleNamespace(**_SWITCHES), None):
            self.assertEqual(_factor(rec_op, "attMM"), 0, f"rec_op={rec_op}")

    def test_full_and_plain_layers_ignore_the_switches(self):
        """Full recompute runs every op again; a plain layer runs none."""
        rec_op = Config(_SWITCHES)
        for op in ("softmax", "normOp", "attMM"):
            self.assertEqual(_factor(rec_op, op, LayerType.FULL_REC_LAYER), 1, op)
            self.assertEqual(_factor(rec_op, op, LayerType.NOT_REC_LAYER), 0, op)

    def test_time_and_memory_agree_on_every_switch(self):
        """In a selective layer an op is kept in memory or recomputed in time, never both or neither."""
        for switch in (0, 1):
            kept = EvalUtils.rec_coeff(True, switch)
            recomputed = _factor(Config({"softmax": switch}), "softmax")
            self.assertEqual(kept + recomputed, 1, f"switch={switch}: kept={kept}, recomputed={recomputed}")

    def test_a_qk_norm_is_recomputed_with_the_norms(self):
        """A QK-norm has no switch of its own: the normOp switch recomputes it, as it drops its inputs."""
        for switch in (0, 1):
            self.assertEqual(_factor(Config({"normOp": switch}), "qknorm"), 1 - switch, f"normOp={switch}")

    def test_only_recomputed_ops_are_charged_twice(self):
        """The recompute pass adds the load of softmax alone."""
        lccfg = SimpleNamespace(n_softmax=1, n_normOp=1, n_attMM=1, rec_op=Config(_SWITCHES))
        table = {"n_softmax": 10.0, "n_normOp": 100.0, "n_attMM": 1000.0}
        once = get_table_quantity(lccfg, table, LayerType.SEL_REC_LAYER, False)
        again = get_table_quantity(lccfg, table, LayerType.SEL_REC_LAYER, True)
        self.assertEqual(again - once, 10.0, f"without recompute {once}, with recompute {again}")

    def test_a_census_prices_the_matmuls_hyperparallels_policy_runs_again(self):
        """
        A selective layer of HyperParallel's policy, whose kind's census
        states the shares of its attention's and feed-forward's matmul FLOPs
        the policy runs again, prices that share of each again, beside the
        ops its switches recompute; a full layer, and a selective one of
        other switches, have no shares.
        """
        census = KindActivations(1.0, 1.0, 1.0, 1.0, 4096, selective_attention_mm=0.25, selective_ffn_mm=0.5)
        lccfg = SimpleNamespace(n_softmax=1, n_attMM=1, n_ffMM=1, kind_activations=census,
                                rec_op=Config(dict(HYPER_SELECTIVE_REC_OP)))
        table = {"n_softmax": 10.0, "n_attMM": 1000.0, "n_ffMM": 100.0}
        shares = selective_shares(lccfg, LayerType.SEL_REC_LAYER)
        self.assertEqual(shares, {"attMM": 0.25, "ffMM": 0.5})
        once = get_table_quantity(lccfg, table, LayerType.SEL_REC_LAYER, False, shares=shares)
        again = get_table_quantity(lccfg, table, LayerType.SEL_REC_LAYER, True, shares=shares)
        self.assertEqual(again - once, 10.0 + 250.0 + 50.0)
        self.assertEqual(selective_shares(lccfg, LayerType.FULL_REC_LAYER), {})
        lccfg.rec_op = Config(dict(HYPER_SELECTIVE_REC_OP, ffAct=1))
        self.assertEqual(selective_shares(lccfg, LayerType.SEL_REC_LAYER), {})


if __name__ == "__main__":
    unittest.main()
