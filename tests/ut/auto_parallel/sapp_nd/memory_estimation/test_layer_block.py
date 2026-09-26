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
"""Tests for the attention, feed-forward and norm formulas of the memory model.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/memory_estimation/test_layer_block.py -v
"""
import unittest
from types import SimpleNamespace

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalAttn, EvalFFn, EvalNorm
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTail
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



def _qkv(cp: int, cp_algo: str = "colossalai_cp", n_linrec: int = 0) -> float:
    """The q, k and v activations of a layer of 4 query and 2 key heads 32 wide, width 128, at *cp*."""
    ccfg = SimpleNamespace(
        s=64, b=1, h=128, a=4, n_kv=2, dh=32, dc_kv=0, n_attMM=4, n_attParamCast=0, n_attBMM=2,
        bytes_compute=2, t=1, cp=cp, cp_algo=cp_algo, n_linrec=n_linrec, rec_op=Config({"attBMM": 1}),
    )
    return EvalAttn.attn_qkv_activations(ccfg, SimpleNamespace(current_node=LayerType.NOT_REC_LAYER,
                                                              micro_factor=1))


class TestKeysAndValuesUnderCp(unittest.TestCase):
    """What a CP rank keeps of a layer's keys and values."""

    def test_gathered_keys_and_values_stay_whole(self):
        """
        Feature: EvalAttn.attn_qkv_activations and kv_shards.
        Description: A layer without CP, and at CP 4 under colossalai,
            hybrid and Ulysses CP, and as a linear-attention layer.
        Expectation: Colossal-AI and hybrid CP keep the whole sequence's
            keys and values, 2 heads of 32 each, and a quarter of the rest;
            Ulysses CP and a linear-attention layer keep a quarter of all.
        """
        whole = _qkv(1)
        kv = 64 * 2 * 2 * 2 * 32
        for algo in ("colossalai_cp", "hybrid_cp"):
            self.assertEqual(_qkv(4, algo), (whole - kv) / 4 + kv)
        self.assertEqual(_qkv(4, "ulysses_cp"), whole / 4)
        self.assertEqual(_qkv(4, n_linrec=1), whole / 4)


def _mla(**fields) -> SimpleNamespace:
    """DeepSeek-V3's attention: width 7168, 128 heads, latents of 1536 and 512, heads 128 and 64 wide."""
    return SimpleNamespace(**{"h": 7168, "a": 128, "n_kv": 128, "dh": 128, "dhr": 64, "dc_q": 1536, "dc_kv": 512,
                              "n_attMM": 4, "qk_nope_head_dim": None, **fields})


class TestMlaParameters(unittest.TestCase):
    """What an MLA layer's attention holds."""

    def test_the_shared_latent_is_counted_once(self):
        """
        Feature: EvalAttn.num_params_mla.
        Description: DeepSeek-V3's attention; the same without a query
            latent; and with non-rotary key heads 192 wide.
        Expectation: The query latent's two projections and norm, the one
            down-projection the keys and values share with the rotary key,
            their latent's norm, their up-projection and the output
            projection: 187107328, the count of the model Transformers
            builds.  Without a query latent, one projection to every head;
            a wider key widens the two up-projections.
        """
        self.assertEqual(EvalAttn.num_params_mla(_mla(), None), 187107328)
        without = 187107328 - (7168 * 1536 + 1536 + 1536 * 128 * 192) + 7168 * 128 * 192
        self.assertEqual(EvalAttn.num_params_mla(_mla(dc_q=0), None), without)
        wider = 187107328 + 64 * 128 * (1536 + 512)
        self.assertEqual(EvalAttn.num_params_mla(_mla(qk_nope_head_dim=192), None), wider)


def _vectors(**stated) -> SimpleNamespace:
    """A layer of width 64, 4 query and 2 key heads 16 wide, gated experts 32 wide, a vocabulary of 100."""
    return SimpleNamespace(**{"h": 64, "a": 4, "n_kv": 2, "dh": 16, "dc_kv": 0, "n_attMM": 4, "attn_output_gate": False,
                              "attn_extra_p": 0, "n_ffMM": 3, "hff": 128, "hff_exp": 32, "etp": 1, "n_exp": 4,
                              "n_shared_exp": 1, "n_normOp": 2, "n_qknorm": 0, "v": 100, "qkv_bias": None,
                              "o_bias": None, "mlp_bias": None, "norm_bias": None, "layer_norms": None,
                              "shared_expert_gate": None, **stated})


class TestVectors(unittest.TestCase):
    """The biases and norm weights a layer holds, as its model states them."""

    def test_unstated_the_formulas_keep_their_convention(self):
        """
        Feature: the parameter formulas of a model that states no bias or norm.
        Description: The layer with every fact unstated.
        Expectation: A bias of the hidden width on each projection, one of
            the expert's width on each of its projections, two vectors a
            norm op, and an output bias per vocabulary entry.
        """
        ccfg = _vectors()
        weights = 64 * 64 * 2 + 64 * 32 * 2
        self.assertEqual(EvalAttn.num_params_attn(ccfg, None), weights + 4 * 64)
        self.assertEqual(EvalFFn.num_params_shared_expert(ccfg, None), 3 * (32 * 64 + 32))
        self.assertEqual(EvalNorm.num_params_norm(ccfg, None), 2 * 2 * 64)
        self.assertEqual(EvalTail.num_params_output(ccfg, None), 64 * 100 + 100)

    def test_stated_each_vector_is_the_models(self):
        """
        Feature: the parameter formulas of a model that states its biases and norms.
        Description: The layer biasing its query, key and value projections
            and its feed-forward, as Qwen2 and a biased Llama do; then with
            no bias, two RMSNorms and a gated shared expert; then with two
            LayerNorms.
        Expectation: Each bias as wide as its projection's output, none
            where stated none; a weight per RMSNorm, a weight and a bias per
            LayerNorm; the shared expert's gate a weight of the width; the
            final norm at the output, and no bias there.
        """
        weights = 64 * 64 * 2 + 64 * 32 * 2
        biased = _vectors(qkv_bias=True, o_bias=False, mlp_bias=True)
        self.assertEqual(EvalAttn.num_params_attn(biased, None), weights + 64 + 2 * 32)
        self.assertEqual(EvalFFn.num_params_routed_expert(biased, None), 4 * (3 * 32 * 64 + 2 * 32 + 64))
        plain = _vectors(qkv_bias=False, o_bias=False, mlp_bias=False, norm_bias=False, layer_norms=2,
                         shared_expert_gate=True)
        self.assertEqual(EvalAttn.num_params_attn(plain, None), weights)
        self.assertEqual(EvalFFn.num_params_shared_expert(plain, None), 3 * 32 * 64 + 64)
        self.assertEqual(EvalNorm.num_params_norm(plain, None), 2 * 64)
        self.assertEqual(EvalTail.num_params_output(plain, None), 64 * 100 + 64)
        layer_norm = _vectors(norm_bias=True, layer_norms=2)
        self.assertEqual(EvalNorm.num_params_norm(layer_norm, None), 4 * 64)
        self.assertEqual(EvalTail.num_params_output(layer_norm, None), 64 * 100 + 2 * 64)


def _experts(hff: int) -> SimpleNamespace:
    """A MoE layer of width 64 over 16 tokens: 4 experts 32 wide, 2 chosen, one shared, the dense width *hff*."""
    return SimpleNamespace(s=16, b=1, h=64, hff=hff, hff_exp=32, n_ffMM=3, n_ffParamCast=0, bytes_compute=2, t=1,
                           cp=1, n_exp=4, n_chosen_exp=2, n_shared_exp=1, gmm=True, cap_fact=1, mlp_bias=None,
                           shared_expert_gate=None, rec_op=Config({"ffAct": 1}))


class TestExpertWidth(unittest.TestCase):
    """A MoE layer's experts are as wide as the model states them, whatever its dense layers' width."""

    def test_every_expert_is_hff_exp_wide(self):
        """
        Feature: EvalFFn's routed and shared expert activations and shared
            expert parameters.
        Description: The MoE layer of a model whose dense layers are 128
            wide, four times its experts.
        Expectation: Each of the 32 token-expert pairs and each token of
            the shared expert keeps its input, two projections into 32 and
            the activation's output, in bf16; the shared expert holds three
            projections of 32 and their biases.
        """
        ccfg = _experts(128)
        ctx = SimpleNamespace(current_node=LayerType.NOT_REC_LAYER, micro_factor=1, dropless_tok_factor=1)
        per_token = 2 * (64 + 3 * 32)
        self.assertEqual(EvalFFn.routed_exp_activations(ccfg, ctx), 16 * 2 * per_token)
        self.assertEqual(EvalFFn.shared_exp_activations(ccfg, ctx), 16 * per_token)
        self.assertEqual(EvalFFn.num_params_shared_expert(ccfg, None), 3 * (32 * 64 + 32))


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
