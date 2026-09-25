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
"""Tests for the communication a recomputed layer transfers again.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/perf_estimation/test_comm_time.py -v
"""
import copy
import os
import unittest

# The package has an import cycle that only the memory estimator's import order
# settles; the performance modules cannot be the first a process loads.
import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2  # pylint: disable=unused-import
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.comm import EvalLayerComm
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import check_and_apply_custom_hook
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import (
    cp_comm_layer_detailed,
    _recomputed_comm,
    estimate_comm,
    prepare_context,
)
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import estimate_performance
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig, RecType

DEEPSEEK_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nd", "deepseek.yaml"
)
_SWITCHES = ("attBMM", "headCast", "dropout", "softmax", "normOp", "gather", "ffAct")


class TestRecomputedComm(unittest.TestCase):
    """DeepSeek at TP 4 and EP 8: a recompute transfers again what its memory drops."""

    @classmethod
    def setUpClass(cls) -> None:
        """The fixture's config, its layer hooks applied."""
        cls.ccfg = CostModelConfig(DEEPSEEK_YAML)
        check_and_apply_custom_hook(cls.ccfg)
        # In a list, not a class attribute: a function read through ``self``
        # would come back as a method bound to the test case.
        cls.groups = [hook for _, hook in cls.ccfg.layer_custom_config]

    def _moe_layer(self, **switches: int) -> CostModelConfig:
        """A MoE layer's config, every op kept but for *switches*."""
        cfg = copy.deepcopy(self.ccfg)
        self.groups[1](cfg)
        cfg.rec_op = Config(dict(dict.fromkeys(_SWITCHES, 1), **switches))
        return cfg

    @staticmethod
    def _plain(cfg: CostModelConfig) -> tuple:
        """TP, EP and CP volume of the layer run once."""
        ctx = prepare_context()
        ctx.current_node = LayerType.NOT_REC_LAYER
        return (EvalLayerComm.tp_comm_layer(cfg, ctx, 1), EvalLayerComm.ep_comm_layer(cfg, ctx, 1), 0)

    def test_a_selective_layer_that_keeps_its_gathers_sends_nothing_again(self):
        """
        Feature: _recomputed_comm.
        Description: A selective layer whose gather switch keeps the gathered tensors.
        Expectation: Nothing is transferred again.
        """
        again = _recomputed_comm(self._moe_layer(softmax=0), prepare_context(), LayerType.SEL_REC_LAYER)
        self.assertEqual(again, (0, 0, 0))

    def test_recomputed_gathers_resend_the_tensor_parallel_volume(self):
        """
        Feature: _recomputed_comm.
        Description: A selective layer that recomputes its gathers.
        Expectation: Its whole TP volume is sent again, the buffer its memory
            no longer keeps; the expert all-to-all is not.
        """
        cfg = self._moe_layer(gather=0)
        tp, ep, cp = _recomputed_comm(cfg, prepare_context(), LayerType.SEL_REC_LAYER)
        self.assertGreater(tp, 0)
        self.assertEqual((tp, ep, cp), (self._plain(cfg)[0], 0, 0))

    def test_full_recompute_resends_every_activation_collective(self):
        """
        Feature: _recomputed_comm.
        Description: A fully recomputed MoE layer.
        Expectation: Its TP and EP volume are sent again; CP is off in this
            config, so none of it.
        """
        cfg = self._moe_layer()
        again = _recomputed_comm(cfg, prepare_context(), LayerType.FULL_REC_LAYER)
        self.assertGreater(again[1], 0)
        self.assertEqual(again, self._plain(cfg))

    def test_a_recompute_resends_the_forward_half_of_cp(self):
        """
        Feature: _recomputed_comm, under context parallelism.
        Description: The MoE layer at CP 4, fully recomputed, and selective
            with its gathers recomputed and kept.
        Expectation: A recompute runs the forward's K and V exchange again,
            half the layer's CP traffic; a selective layer only where its
            gather switch recomputes.
        """
        cfg = self._moe_layer()
        cfg.cp, cfg.comm_cp = 4, 1
        forward = cp_comm_layer_detailed(cfg, prepare_context()).comm_volume / 2
        self.assertGreater(forward, 0)
        self.assertEqual(_recomputed_comm(cfg, prepare_context(), LayerType.FULL_REC_LAYER)[2], forward)
        cfg.rec_op = Config(dict(dict.fromkeys(_SWITCHES, 1), gather=0))
        self.assertEqual(_recomputed_comm(cfg, prepare_context(), LayerType.SEL_REC_LAYER)[2], forward)
        cfg.rec_op = Config(dict.fromkeys(_SWITCHES, 1))
        self.assertEqual(_recomputed_comm(cfg, prepare_context(), LayerType.SEL_REC_LAYER)[2], 0)

    def test_only_recomputed_layers_add_communication(self):
        """
        Feature: estimate_comm with_recomp.
        Description: The fixture's stages, fully recomputed, then the same
            stages with every layer plain.
        Expectation: Plain stages send nothing again; every recomputed stage
            sends more.
        """
        custom = CustomConfig()
        stages = self.ccfg.generate_partitions_vpp()
        plain = [[[LayerType.NOT_REC_LAYER if layer == LayerType.FULL_REC_LAYER else layer for layer in chunk]
                  for chunk in stage] for stage in stages]

        def _comm(layout: list, with_recomp: bool) -> list:
            return estimate_comm(copy.deepcopy(self.ccfg), custom, layout, Hard.Device_A2, with_recomp=with_recomp)

        self.assertEqual(_comm(plain, True), _comm(plain, False))
        once, again = _comm(stages, False), _comm(stages, True)
        self.assertTrue(all(more > less for more, less in zip(again, once)), f"once {once}, again {again}")


class TestRecomputePricing(unittest.TestCase):
    """The search prices a recompute with the communication it runs again."""

    def test_the_default_prices_recomputed_communication(self):
        """
        Feature: CustomConfig.
        Description: DeepSeek, fully recomputed at TP 4 and EP 8, priced with
            the default options and with compute only.
        Expectation: The default includes the communication, and costs more.
        """
        self.assertEqual(CustomConfig().retype, RecType.WITH)
        ccfg = CostModelConfig(DEEPSEEK_YAML)
        default = estimate_performance(ccfg, device_type=Hard.Device_A2)
        compute_only = estimate_performance(
            ccfg, ccfg=CustomConfig(retype=RecType.COMPUTE_ONLY), device_type=Hard.Device_A2)
        self.assertGreater(default, compute_only)


if __name__ == "__main__":
    unittest.main()
