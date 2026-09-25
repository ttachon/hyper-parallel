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
"""Tests for the embedding layer's memory.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/memory_estimation/test_head.py -v
"""
import unittest
from types import SimpleNamespace

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.head import EvalHead
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config


def _embedding(tied: bool, p: int) -> SimpleNamespace:
    """An embedding of 1000 x 64 parameters, sharded over 4 ranks."""
    return SimpleNamespace(h=64, v=1000, shard_embed=4, cp=1, bytes_p=2, bytes_os=4, bytes_grad=2,
                           tie_emb_out=tied, p=p)


_CTX = SimpleNamespace(eval=SimpleNamespace(num_p=EvalHead.num_params_embed), swap_os=False)


class TestTiedEmbedding(unittest.TestCase):
    """A tied table is one only where the embedding and the output layer share a stage."""

    def test_a_tied_table_is_held_once_on_one_stage_and_twice_across_stages(self):
        """
        Feature: EvalHead's static memory of a tied embedding.
        Description: The same embedding tied and untied, at PP 1 and PP 2.
        Expectation: Tied at PP 1, the embedding holds no parameters,
            gradients or optimizer states, since the output layer holds the
            table; at PP 2 the first stage holds its own copy, as untied.
        """
        for tied, p in ((True, 1), (True, 2), (False, 1), (False, 2)):
            ccfg = _embedding(tied, p)
            got = (EvalHead.stat_embed_p(ccfg, _CTX), EvalHead.stat_embed_grad(ccfg, _CTX),
                   EvalHead.stat_embed_os(ccfg, _CTX))
            want = (0, 0, 0) if tied and p == 1 else (32000, 32000, 128000)
            self.assertEqual(got, want, f"tied={tied}, p={p}")



class TestEmbeddingGather(unittest.TestCase):
    """The embedding is never recomputed, so no switch drops what it keeps."""

    def test_the_gather_switch_leaves_the_embedding_its_tp_buffer(self):
        """
        Feature: EvalHead's TP buffer.
        Description: A config at TP 4 whose selective layers recompute their
            gathers (gather 0), and one whose layers keep them.
        Expectation: The embedding keeps the same buffer, s * h * b * 3 / 4.
        """
        buffers = [
            EvalHead.tp_comm_embed(SimpleNamespace(comm_t=1, s=8, h=64, b=2, t=4, cp=1,
                                                   rec_op=Config({"gather": gather})), None)
            for gather in (0, 1)
        ]
        self.assertEqual(buffers, [8 * 64 * 2 * 3 / 4] * 2)


if __name__ == "__main__":
    unittest.main()
