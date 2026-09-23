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
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel._model_spec import OpCounts
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.comm import EvalLayerComm
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalAttn
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import (
    check_and_apply_custom_hook,
    layer_hook,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import estimate_comm
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.getters import (
    get_layer_configs_by_position,
    get_model_order,
)
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig

EMB, OUT, LAY = LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER, LayerType.NOT_REC_LAYER


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


class TestPerformanceAgreesWithMemory(unittest.TestCase):
    """Both estimators give every layer the same kind."""

    @staticmethod
    def _stack(sched: str) -> CostModelConfig:
        """Eight layers on two stages of two chunks, alternating kinds by pairs."""
        overrides = {
            "hidden_size": 1024, "num_hidden_layers": 8, "num_attention_heads": 8,
            "num_key_value_heads": 8, "intermediate_size": 2816, "vocab_size": 32000,
            "max_position_embeddings": 2048, "arch": "default",
        }
        ccfg = CostModelConfig({
            "model": {"name": "unit", "config_overrides": overrides},
            "training": {"global_batch_size": 8, "micro_batch_size": 1},
            "accelerator": {"tp_size": 1, "pp_size": 2, "pp_interleave_num": 2,
                            "pipeline_scheduler": sched},
            "fsdp_config": {"dp_shard_size": 2},
            "dataset": {"data_transform": {"max_seq_len": 2048}},
            "context": {"max_device_memory": "64GB", "device_num": 4},
        }, framework="hyper_v2")
        counts = {"attMM": 4, "attBMM": 2, "ffMM": 3, "softmax": 1,
                  "dropout": 0, "normOp": 2, "gather": 4, "headCast": 1, "ffAct": 1}
        ccfg.op_counts = {
            "decoder": OpCounts.from_dict(counts),
            "linear_attention": OpCounts.from_dict(dict(counts, attBMM=0, softmax=0, headCast=0)),
        }
        full, linear = layer_hook("default", "decoder"), layer_hook("default", "linear_attention")
        ccfg.layer_custom_config = [(2, full), (2, linear), (2, full), (2, linear)]
        return ccfg

    def _check(self, sched: str) -> None:
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
            layer's DP term sees, in that walk.
        Expectation: Stage 0 holds model layers 0-1 and 4-5, both full
            pairs; stage 1 holds the two linear pairs.
        """
        ccfg = self._stack("1f1b")
        check_and_apply_custom_hook(ccfg)
        stages = ccfg.generate_partitions_vpp()
        seen = []
        real = EvalLayerComm.dp_comm_layer

        def spy(cfg: CostModelConfig, ctx: Any) -> float:
            """The real DP term, recording the kind of the layer it prices."""
            seen.append(cfg.n_softmax)
            return real(cfg, ctx)

        with patch.object(EvalLayerComm, "dp_comm_layer", side_effect=spy):
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


if __name__ == "__main__":
    unittest.main()
