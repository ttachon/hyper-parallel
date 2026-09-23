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
"""Tests for the per-op compute loads the performance path prices.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/perf_estimation/test_estimate.py -v
"""
import unittest
from types import SimpleNamespace

# The package has an import cycle that only the memory estimator's import order
# settles; perf_estimation.estimate cannot be the first module a process loads.
import hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2  # pylint: disable=unused-import
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import op_table


def _cfg(s: int) -> SimpleNamespace:
    """A dense, non-MLA config at sequence length *s*."""
    return SimpleNamespace(a=8, n_kv=8, b=1, s=s, h=512, hff=1024, t=2, sp=2, cp=1, dc_kv=0, bytes_p=2)


class TestOpTable(unittest.TestCase):
    """Each op's load follows the tensor it runs over."""

    def test_token_wise_ops_scale_with_the_sequence(self):
        """The feed-forward activation runs once per token, like the projections around it."""
        short, long = op_table(_cfg(128)), op_table(_cfg(256))
        for op in ("n_attMM", "n_ffMM", "n_gather", "n_normOp", "n_ffAct"):
            self.assertEqual(long[op], 2 * short[op], f"{op}: s=128 gives {short[op]}, s=256 gives {long[op]}")


if __name__ == "__main__":
    unittest.main()
