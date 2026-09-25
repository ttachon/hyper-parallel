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
"""Tests for the attention and norm formulas of the memory model.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/memory_estimation/test_layer_block.py -v
"""
import unittest
from types import SimpleNamespace

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalAttn, EvalNorm
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType


def _score(softmax: int, layer: LayerType) -> float:
    """Score-tensor activations of one layer with the given softmax switch."""
    ccfg = SimpleNamespace(
        s_fa=4, b=1, a=2, s=4, n_softmax=1, t=1, cp=1,
        bytes_softmax=4, bytes_dropout=1, bytes_compute=2,
        rec_op=Config({"softmax": softmax, "dropout": 1, "headCast": 1}),
    )
    return EvalAttn.attn_score_activations(ccfg, SimpleNamespace(current_node=layer, micro_factor=1))


class TestAttentionScore(unittest.TestCase):
    """The softmax switch acts only in a selective layer, like every other switch."""

    def test_a_plain_layer_keeps_its_softmax(self):
        """A layer that recomputes nothing stores the softmax whatever the switch says."""
        on, off = _score(1, LayerType.NOT_REC_LAYER), _score(0, LayerType.NOT_REC_LAYER)
        self.assertEqual(off, on, f"switch at 0 gives {off}, at 1 gives {on}")

    def test_a_selective_layer_drops_the_softmax_it_recomputes(self):
        """The saving is the softmax output: s_fa * b * a * s * bytes_softmax."""
        kept, dropped = _score(1, LayerType.SEL_REC_LAYER), _score(0, LayerType.SEL_REC_LAYER)
        self.assertEqual(kept - dropped, 4 * 1 * 2 * 4 * 4, f"kept {kept}, dropped {dropped}")



def _norms(n_qk_norm: int, norm_switch: int = 1) -> SimpleNamespace:
    """A layer with two norms, 4 query and 2 key heads 32 wide, and *n_qk_norm* QK-norms."""
    return SimpleNamespace(
        h=64, a=4, n_kv=2, dh=32, n_normOp=2, n_qknorm=n_qk_norm, s=8, b=1, bytes_norm=4, t=1, cp=1,
        rec_op=Config({"normOp": norm_switch}),
    )


class TestQkNorm(unittest.TestCase):
    """A QK-norm is priced as a norm over every head's queries and keys."""

    def test_a_qk_norm_holds_its_weights_and_its_inputs(self):
        """
        Feature: the memory of a QK-norm.
        Description: The same layer with a QK-norm and without.
        Expectation: The QK-norm adds a query and a key weight of one head's
            width, 2 x 32 parameters, and keeps every head's queries and
            keys, 6 heads of 32 per token, at the norms' width in bytes.
        """
        ctx = SimpleNamespace(current_node=LayerType.NOT_REC_LAYER, micro_factor=1)
        params = [EvalNorm.num_params_norm(_norms(n), ctx) for n in (1, 0)]
        activations = [EvalNorm.norm_activations(_norms(n), ctx) for n in (1, 0)]
        self.assertEqual((params[0] - params[1], activations[0] - activations[1]), (2 * 32, 8 * 1 * 4 * 6 * 32))

    def test_a_selective_layer_recomputes_it_with_the_norms(self):
        """
        Feature: the norm switch.
        Description: A selective layer whose normOp switch is 0.
        Expectation: It keeps neither its norms' inputs nor its QK-norm's.
        """
        ctx = SimpleNamespace(current_node=LayerType.SEL_REC_LAYER, micro_factor=1)
        self.assertEqual(EvalNorm.norm_activations(_norms(1, norm_switch=0), ctx), 0)


if __name__ == "__main__":
    unittest.main()
