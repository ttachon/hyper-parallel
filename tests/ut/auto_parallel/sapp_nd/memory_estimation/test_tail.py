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
"""Unit tests for tail.py: what the output layer and the MTP layers keep."""
import unittest
from unittest.mock import MagicMock

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalMTP, EvalTailSingle


def _ccfg(cp: int = 1, n_mtp: int = 0) -> MagicMock:
    """An output layer of 4096 tokens, width 1024 and a 32000-entry vocabulary, at *cp*."""
    ccfg = MagicMock()
    ccfg.output_census = None
    ccfg.s, ccfg.b, ccfg.h, ccfg.v = 4096, 1, 1024, 32000
    ccfg.bytes_norm, ccfg.bytes_compute = 4, 2
    ccfg.shard_output_activ, ccfg.cp, ccfg.n_mtp = 1, cp, n_mtp
    return ccfg


def _ctx() -> MagicMock:
    """A context at micro factor 2, whose embedding keeps nothing."""
    ctx = MagicMock()
    ctx.micro_factor = 2
    ctx.eval.dyn.activation = lambda ccfg, ctx: 0
    return ctx


class TestOutputActivations(unittest.TestCase):
    """The output layer's formulas."""

    def test_a_cp_rank_keeps_its_own_chunks_logits(self):
        """
        Feature: EvalTailSingle.activ_out_single and EvalMTP.activ_mtp.
        Description: The output layer, and a model with one MTP layer,
            without context parallelism and at CP 4.
        Expectation: A rank keeps the final norm's input and the logits of
            its own chunk of the sequence, a quarter at CP 4; so does the
            MTP layer.
        """
        whole = EvalTailSingle.activ_out_single(_ccfg(), _ctx())
        self.assertEqual(whole, 2 * 4096 * (4 * 1024 + 2 * 32000))
        self.assertEqual(EvalTailSingle.activ_out_single(_ccfg(cp=4), _ctx()), whole / 4)
        mtp = EvalMTP.activ_mtp(_ccfg(n_mtp=1), _ctx())
        self.assertEqual(mtp, 2 * 2 * 4096 * 3 * 1024 + whole)
        self.assertEqual(EvalMTP.activ_mtp(_ccfg(cp=4, n_mtp=1), _ctx()), mtp / 4)


if __name__ == "__main__":
    unittest.main()
