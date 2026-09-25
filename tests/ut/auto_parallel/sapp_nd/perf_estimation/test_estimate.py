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
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import estimate_performance, op_table

DEEPSEEK_YAML = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nd", "deepseek.yaml"
)


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


if __name__ == "__main__":
    unittest.main()
