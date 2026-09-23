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
import os
import unittest
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

# The package has an import cycle that only the memory estimator's import order
# settles; perf_estimation.estimate cannot be the first module a process loads.
import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2  # pylint: disable=unused-import
import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
import hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate as Estimate
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import check_and_apply_custom_hook
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import estimate_comm
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import (
    LayerTimes,
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


_SWITCHES = ("attBMM", "headCast", "dropout", "softmax", "normOp", "gather", "ffAct")


class TestLayerTimes(unittest.TestCase):
    """One layer priced alone, as the pipeline balancer is given it."""

    @classmethod
    def setUpClass(cls) -> None:
        """DeepSeek: a dense prefix, then MoE layers, on sixteen stages."""
        cls.ccfg = CostModelConfig(DEEPSEEK_YAML)
        check_and_apply_custom_hook(cls.ccfg)
        cls.stages = cls.ccfg.generate_partitions_vpp()
        flat = sum(([hook] * count for count, hook in cls.ccfg.layer_custom_config), [])
        cls.hooks = dict(zip(get_model_order(cls.ccfg, cls.stages), flat))
        # In a list, not a class attribute: a function read through ``self``
        # would come back as a method bound to the test case.
        cls.groups = [hook for _, hook in cls.ccfg.layer_custom_config]

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
        search = estimate_stage(walked, custom, comp, comm, recomp, [0] * self.ccfg.p)
        times = LayerTimes(Hard.Device_A2)
        checked = 0
        for s, stage in enumerate(self.stages):
            positions = [(s, c, i) for c, chunk in enumerate(stage) for i, _ in enumerate(chunk)]
            if len({self.hooks.get(position) for position in positions} - {None}) != 1 or any(
                    position not in self.hooks for position in positions):
                continue
            total = sum(sum(times(self.ccfg, self.hooks[p], stage[p[1]][p[2]])) for p in positions)
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
        Description: Price a layer, change the config the way a hook would,
            then price the same layer and another type.
        Expectation: The change is not seen, and the repeated layer is not
            priced again.
        """
        cfg = copy.deepcopy(self.ccfg)
        times = LayerTimes(Hard.Device_A2)
        first = times(cfg, self.groups[1], LayerType.NOT_REC_LAYER)
        cfg.s *= 2  # a field no layer hook sets
        with patch.object(Estimate, "estimate_layer_times", wraps=Estimate.estimate_layer_times) as spy:
            self.assertEqual(times(cfg, self.groups[1], LayerType.NOT_REC_LAYER), first)
            full = times(cfg, self.groups[1], LayerType.FULL_REC_LAYER)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(full, LayerTimes(Hard.Device_A2)(self.ccfg, self.groups[1], LayerType.FULL_REC_LAYER))


if __name__ == "__main__":
    unittest.main()
