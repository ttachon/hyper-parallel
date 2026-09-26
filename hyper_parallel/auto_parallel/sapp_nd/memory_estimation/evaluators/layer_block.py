# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""Layer's blocks submodule"""
from __future__ import annotations
from typing import TYPE_CHECKING
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cp_types import CPAlgo, _resolve_cp_algo
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation._context import Context

mb = EvalUtils.mb


def _bias(ccfg: CostModelConfig, stated: str, width: float) -> float:
    """A projection's bias: *width* where the model states one, none where it states none.

    Where the model states neither, as a parser that reads no bias does,
    each projection is counted a bias of the hidden width, the formulas'
    convention.
    """
    has_bias = getattr(ccfg, stated, None)
    if has_bias is None:
        return ccfg.h
    return width if has_bias else 0


def _mlp_biases(ccfg: CostModelConfig, width: float) -> float:
    """The biases of one feed-forward of *width*, or one expert's, as :func:`_bias` counts a projection's.

    Stated, each projection into the width has one of it and the last,
    back to the hidden width, one of that; unstated, each projection one of
    the width.
    """
    has_bias = getattr(ccfg, "mlp_bias", None)
    if has_bias is None:
        return ccfg.n_ffMM * width
    return (ccfg.n_ffMM - 1) * width + ccfg.h if has_bias else 0


class EvalAttn:
    """Attention formulas class"""

    @staticmethod
    def num_params_mla(ccfg: CostModelConfig, _) -> float:
        """Parameters count for Multi-Head Latent Attention.

        The queries: a down-projection to their latent, its norm and an
        up-projection to every head's non-rotary and rotary part, or one
        projection where the model has no query latent.  The keys and
        values: one down-projection to their shared latent beside the
        rotary key, its norm, and an up-projection to every head's
        non-rotary key and value.  The output projection from the values.
        Each head's non-rotary key is ``qk_nope_head_dim`` wide, ``dh``,
        its value head's width, unless the model states it.
        """
        d_qk = getattr(ccfg, "qk_nope_head_dim", None) or ccfg.dh
        heads_q = ccfg.a * (d_qk + ccfg.dhr)
        if ccfg.dc_q:
            query = ccfg.h * ccfg.dc_q + ccfg.dc_q + ccfg.dc_q * heads_q
        else:
            query = ccfg.h * heads_q
        key_value = (ccfg.h * (ccfg.dc_kv + ccfg.dhr) + ccfg.dc_kv
                     + ccfg.dc_kv * ccfg.n_kv * (d_qk + ccfg.dh))
        output = ccfg.a * ccfg.dh * ccfg.h
        return 0.25 * ccfg.n_attMM * (query + key_value + output)

    @staticmethod
    def num_params_attn(ccfg: CostModelConfig, ctx: Context) -> float:
        """Parameters count for Multi-Head/Grouped-Q./Multi-Q. Attention"""
        if ccfg.dc_kv == 0:
            # Q,O and K,V have distinct shapes. Q and O are h x (a*dh),
            # which equals h x h only when head_dim = h/a; a fused output
            # gate (Qwen3.5) doubles the Q projection.
            d_h = ccfg.dh or (ccfg.h / ccfg.a if ccfg.a else 0)
            d_q = ccfg.a * d_h
            q_fact = 2 if ccfg.attn_output_gate else 1
            return 0.25 * ccfg.n_attMM * (
                q_fact * ccfg.h * d_q + _bias(ccfg, "qkv_bias", q_fact * d_q)
            ) + 0.25 * ccfg.n_attMM * (
                ccfg.h * d_q + _bias(ccfg, "o_bias", ccfg.h)
            ) + 0.5 * ccfg.n_attMM * (
                ccfg.h * ccfg.n_kv * d_h + _bias(ccfg, "qkv_bias", ccfg.n_kv * d_h)
            ) + ccfg.attn_extra_p
        return EvalAttn.num_params_mla(ccfg, ctx)

    @staticmethod
    def kv_shards(ccfg: CostModelConfig) -> float:
        """Over how many CP ranks a layer's keys and values are split.

        Colossal-AI and hybrid CP all-gather the keys and values, and the
        attention kernel keeps them whole for the backward: none.  Ulysses
        CP gives each rank its heads' share, and a linear-attention layer
        passes its state rather than gathering them: all of CP.
        """
        if _resolve_cp_algo(ccfg) == CPAlgo.ULYSSES_CP or getattr(ccfg, "n_linrec", 0):
            return ccfg.cp
        return 1

    @staticmethod
    def gathered_kv_bytes(ccfg: CostModelConfig) -> float:
        """The bytes per token of a rank's share of the sequence it keeps of the other ranks' keys and values.

        Where CP gathers the keys and values (:meth:`kv_shards`), the
        attention keeps the whole sequence's, the rank's own share and
        ``cp - 1`` others, each split over TP.
        """
        if ccfg.cp <= 1 or EvalAttn.kv_shards(ccfg) > 1:
            return 0
        return (ccfg.cp - 1) * 2 * ccfg.n_kv * ccfg.dh * ccfg.bytes_compute / max(1, ccfg.t)

    @staticmethod
    def attn_qkv_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """QKV linear Activations"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        att_qkv_size = 0
        if ccfg.dc_kv == 0:
            n_op = ccfg.n_attMM + ccfg.n_attParamCast
            att_qkv_size = (
                ccfg.s
                * ccfg.b
                * ccfg.bytes_compute
                * (
                    0.25 * n_op * ccfg.h
                    + 0.5 * n_op * ccfg.dh * ccfg.n_kv * ccfg.cp / EvalAttn.kv_shards(ccfg)
                    + EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.attBMM)
                    * ccfg.n_attBMM
                    * ccfg.dh
                )
            )
        else:
            q_size = (
                0.25
                * (ccfg.n_attMM + ccfg.n_attParamCast)
                * (ccfg.dc_q + 2 * ccfg.a * (ccfg.dh + ccfg.dhr))
            )
            k_size = (
                0.25
                * (ccfg.n_attMM + ccfg.n_attParamCast)
                * (ccfg.dhr + ccfg.n_kv * (2 * ccfg.dh + ccfg.dhr))
            )
            v_size = (
                0.25
                * (ccfg.n_attMM + ccfg.n_attParamCast)
                * (ccfg.n_kv * ccfg.dh + ccfg.dc_kv)
            )
            att_qkv_size = (
                ccfg.s
                * ccfg.b
                * ccfg.bytes_compute
                * (
                    q_size
                    + (k_size + v_size) * ccfg.cp / EvalAttn.kv_shards(ccfg)
                    + EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.attBMM)
                    * ccfg.n_attBMM
                    * ccfg.dh
                )
            )
        micro_factor = ctx.micro_factor
        return micro_factor * att_qkv_size / (ccfg.t * ccfg.cp)

    @staticmethod
    def attn_score_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Score/Softmax Activations"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        att_score = (
            ccfg.s_fa
            * ccfg.b
            * ccfg.a
            * ccfg.s
            * (
                ccfg.n_softmax
                * (
                    EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.softmax)
                    * ccfg.bytes_softmax
                    + EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.dropout)
                    * ccfg.bytes_dropout
                    + EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.headCast)
                    * ccfg.bytes_compute
                )
            )
        )
        micro_factor = ctx.micro_factor
        # cp_sq_div=1: Ring CP shards Q along seq and all-gathers KV, so the S²
        # score tensor has only one S dim divided → B·H·S²/cp (slope -1).
        # Only an unimplemented blockwise-ring path (shard both Q and KV along
        # seq → (s/cp)² scores) would need cp_sq_div=cp; do not re-add the
        # conditional for the current Ring/Ulysses algorithms.
        cp_sq_div = 1
        return micro_factor * att_score / (ccfg.t * ccfg.cp * cp_sq_div)

    @staticmethod
    def attn_proj_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Output projection Activations"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        att_proj = (
            ccfg.s
            * ccfg.b
            * ccfg.h
            * ccfg.bytes_compute
            * (
                0.25 * (ccfg.n_attMM + ccfg.n_attParamCast)
                + EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.dropout)
                * ccfg.n_dropout
                * ccfg.bytes_dropout
            )
        )
        micro_factor = ctx.micro_factor
        return micro_factor * att_proj / max(ccfg.sp, ccfg.cp)


class EvalFFn:
    """Feed-forward formulas class"""

    @staticmethod
    def num_params_ffn(ccfg: CostModelConfig, _) -> float:
        """Parameters count"""
        return (ccfg.n_exp + ccfg.n_shared_exp) * (
            ccfg.n_ffMM * ccfg.hff * ccfg.h + _mlp_biases(ccfg, ccfg.hff))

    @staticmethod
    def num_params_routed_expert(ccfg: CostModelConfig, _) -> float:
        """Routed expert parameters count (with ETP correction)"""
        hff_sliced = ccfg.hff_exp / max(ccfg.etp, 1)
        return ccfg.n_exp * (ccfg.n_ffMM * hff_sliced * ccfg.h + _mlp_biases(ccfg, hff_sliced))

    @staticmethod
    def num_params_router(ccfg: CostModelConfig, _) -> float:
        """The router's parameters: a weight per routed expert, the hidden width wide, where the layer routes."""
        return ccfg.h * ccfg.n_exp if ccfg.n_exp > 1 else 0

    @staticmethod
    def num_params_shared_expert(ccfg: CostModelConfig, _) -> float:
        """Shared expert parameters count, and the weight that gates its output where the model has one"""
        gate = ccfg.h if ccfg.n_shared_exp and getattr(ccfg, "shared_expert_gate", None) else 0
        return ccfg.n_shared_exp * (ccfg.n_ffMM * ccfg.hff * ccfg.h + _mlp_biases(ccfg, ccfg.hff)) + gate

    @staticmethod
    def ffn_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """ "Activations count"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        tok_size = ccfg.s * ccfg.b
        n_mm = ccfg.n_ffMM
        if n_mm % 2 == 0:
            matmul = 0.5 * ccfg.h + 0.5 * ccfg.hff
        else:
            matmul = 1 / 3 * ccfg.h + 2 / 3 * ccfg.hff
        matmul *= ccfg.bytes_compute * n_mm
        activ_fun = ccfg.bytes_compute * ccfg.hff
        activ_fun *= EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.ffAct)
        pcast = ccfg.bytes_compute * ccfg.hff * ccfg.n_ffParamCast
        activ_size = matmul + pcast + activ_fun
        micro_factor = ctx.micro_factor
        return micro_factor * tok_size * activ_size / (ccfg.t * ccfg.cp)

    @staticmethod
    def ffn_router_and_concat_activations(
        ccfg: CostModelConfig, ctx: Context
    ) -> float:
        """MoE router and output activations"""
        # Router activations (logits, probs, mask)
        r = ccfg.s * ccfg.b * ccfg.bytes_compute
        r *= 2 * ccfg.n_exp + ccfg.n_chosen_exp
        # Concat all exp output
        c = ccfg.s * ccfg.b * ccfg.bytes_compute * ccfg.h
        micro_factor = ctx.micro_factor
        return micro_factor * (r + c) / (ccfg.t * ccfg.cp)

    @staticmethod
    def shared_exp_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Shared expert activations"""
        return ccfg.n_shared_exp * EvalFFn.ffn_activations(ccfg, ctx)

    @staticmethod
    def routed_exp_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """MoE topK activations"""
        tok_size = ccfg.s * ccfg.b
        activ_size = EvalFFn.ffn_activations(ccfg, ctx) / tok_size
        avg_num_toks = tok_size * ccfg.n_chosen_exp / ccfg.n_exp
        if not ccfg.gmm:  # Capacity mode
            expert_capacity = avg_num_toks * ccfg.cap_fact * ccfg.n_exp
            routed_activ = activ_size * expert_capacity
        else:  # Dropless mode
            load = avg_num_toks * ccfg.n_exp * ctx.dropless_tok_factor
            routed_activ = load * activ_size
        return routed_activ

    @staticmethod
    def ffn_moe_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Sum of Activations"""
        return (
            EvalFFn.routed_exp_activations(ccfg, ctx)
            + EvalFFn.shared_exp_activations(ccfg, ctx)
            + EvalFFn.ffn_router_and_concat_activations(ccfg, ctx)
        )


class EvalNorm:
    """Normalization formulas class"""

    @staticmethod
    def head_dim(ccfg: CostModelConfig) -> float:
        """The width a QK-norm normalizes over: one attention head's"""
        return ccfg.dh or (ccfg.h / ccfg.a if ccfg.a else 0)

    @staticmethod
    def num_params_norm(ccfg: CostModelConfig, _) -> float:
        """Parameters count: the layer's norms, and a QK-norm's query and key weights.

        A model that states its norms holds a weight in each, and a bias
        beside it in a LayerNorm; one that does not is counted two vectors
        per norm op.
        """
        layer_norms = getattr(ccfg, "layer_norms", None)
        if layer_norms is None:
            vectors = ccfg.n_normOp * 2
        else:
            vectors = layer_norms * (2 if getattr(ccfg, "norm_bias", None) else 1)
        return vectors * ccfg.h + getattr(ccfg, "n_qknorm", 0) * 2 * EvalNorm.head_dim(ccfg)

    @staticmethod
    def norm_activations(ccfg: CostModelConfig, ctx: Context) -> float:
        """Activations: the norms' inputs, a QK-norm's every head's queries and keys"""
        rec_layer = ctx.current_node == LayerType.SEL_REC_LAYER
        qk_width = getattr(ccfg, "n_qknorm", 0) * (ccfg.a + ccfg.n_kv) * EvalNorm.head_dim(ccfg)
        norm = (
            ccfg.s
            * ccfg.b
            * ccfg.bytes_norm
            * (ccfg.h * ccfg.n_normOp + qk_width)
            * EvalUtils.rec_coeff(rec_layer, ccfg.rec_op.normOp)
        )
        micro_factor = ctx.micro_factor
        return micro_factor * norm / (ccfg.t * ccfg.cp)
