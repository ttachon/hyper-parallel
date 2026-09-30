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
"""cost model parser module"""
from __future__ import annotations
from typing import TYPE_CHECKING, Any

import math
from abc import ABC
from abc import abstractmethod

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import _CostModVar

# The recompute switches of HyperParallel's selective activation checkpointing
# (hyper_parallel/distributed/activation_checkpoint.py). It keeps the outputs of
# matmul and attention kernels and of reduce-scatter, all-to-all and all-reduce,
# and recomputes everything else, all-gathers included. A switch at 1 keeps the
# op's activation and 0 recomputes it. The policy also recomputes every other
# projection matmul, which has no switch, so that part is not priced.
HYPER_SELECTIVE_REC_OP = {
    "attBMM": 1,
    "headCast": 0,
    "dropout": 0,
    "softmax": 0,
    "normOp": 0,
    "gather": 0,
    "ffAct": 0,
}


def runs_hyper_selective(ccfg: Any) -> bool:
    """Whether *ccfg*'s selective layers run HyperParallel's policy: their switches are its switches."""
    switches = vars(ccfg.rec_op) if getattr(ccfg, "rec_op", None) is not None else {}
    return all(switches.get(name) == state for name, state in HYPER_SELECTIVE_REC_OP.items())


class _CostModelParser(ABC):
    """abstract parser class"""

    def __init__(self, ccfg: _CostModVar):
        self.ccfg = ccfg
        self.config = ccfg.config

    @abstractmethod
    def parse(self):
        """Parse the cost model configuration and populate the cost model variables.

        Subclasses must implement this method to read framework-specific
        configuration values into the shared _CostModVar instance.
        """

    @staticmethod
    def hyper_rec_op(selective: bool | list) -> dict[str, int]:
        """Recompute switches for a HyperParallel activation checkpoint mode.

        Args:
            selective: The parsed ``sel_rec``: truthy when the run uses
                selective activation checkpointing.

        Returns:
            The switches for ``ccfg.rec_op``: :data:`HYPER_SELECTIVE_REC_OP`
            for a selective run, and every op kept otherwise.
        """
        if selective:
            return dict(HYPER_SELECTIVE_REC_OP)
        return dict.fromkeys(HYPER_SELECTIVE_REC_OP, 1)

    @staticmethod
    def state_qk_norm(ccfg, qk_norm):
        """State whether the model normalizes each head's queries and keys, and the QK-norm each layer runs.

        A layer group whose attention has none, a linear one, states 0 instead.
        """
        ccfg.qk_norm = bool(qk_norm)
        ccfg.n_qknorm = 1 if ccfg.qk_norm else 0

    @staticmethod
    def optimizer_ranks(ccfg):
        """How many data-parallel ranks the optimizer shards a parameter over.

        ``os_max_shard`` counts them, as MindSpore's ``optimizer_weight_shard_size``
        and HyperParallel's ``dp_shard`` do, on top of TP's sharding.  A count
        that does not divide DP shards over all of it, as MindSpore does.
        """
        ranks = int(ccfg.os_max_shard or 0)
        return ranks if ranks >= 1 and ccfg.d % ranks == 0 else ccfg.d

    @staticmethod
    def routed_expert_shard(ccfg, ranks):
        """How many ranks a routed expert's parameters and optimizer states are sharded over.

        A run that states its expert shard, as HyperParallel's ``edp_shard_size``,
        shards an expert under expert parallelism over as many ranks of its expert
        data-parallel group: the stage's ranks over EP, whatever their DP, CP or
        TP, the group of ranks that hold the same experts.  Without expert
        parallelism its FSDP shards the experts with the other parameters, over
        the optimizer's *ranks*.  Stated by no one, the optimizer shards them over
        the whole group.
        """
        stated = getattr(ccfg, "expert_shard", None)
        if not stated:
            return (ccfg.d_exp if ccfg.has_op else 1) * ccfg.cp * ccfg.t_exp
        if ccfg.ep > 1:
            return math.gcd(int(stated), max(1, ccfg.d * ccfg.cp * ccfg.t // ccfg.ep))
        return ranks * ccfg.cp * ccfg.t_exp

    def config_optimizer_shard(self, ccfg):
        """OP related variables; a routed expert is sharded as :meth:`routed_expert_shard` says."""
        # With optimizer sharding, a parameter is sharded over TP and then
        # over the optimizer's data-parallel ranks.
        ranks = _CostModelParser.optimizer_ranks(ccfg) if ccfg.has_op else 1
        # Non expert params
        ccfg.shard_p_os_non_exp_partial = ranks * ccfg.t * ccfg.cp
        ccfg.shard_p_os_non_exp = (
            (ccfg.d if ccfg.has_op else 1) * ccfg.cp * ccfg.t
        )

        # Expert params
        ccfg.shard_p_os_exp_partial = math.gcd(ccfg.n_exp, ranks * ccfg.t_exp)
        ccfg.shard_p_os_exp = _CostModelParser.routed_expert_shard(ccfg, ranks)

        # Gradients: as the parameters are when FSDP holds them so, over the
        # whole optimizer shard under gradient sharding, over TP alone
        # otherwise.
        if getattr(ccfg, "grads_as_params", False):
            grads = (ccfg.shard_p_os_non_exp_partial, ccfg.shard_p_os_exp, ccfg.shard_p_os_exp_partial)
        elif ccfg.has_grad_shard:
            grads = (ccfg.shard_p_os_non_exp, ccfg.shard_p_os_exp, ccfg.shard_p_os_exp_partial)
        else:
            grads = (ccfg.t, ccfg.t_exp, ccfg.t_exp)
        ccfg.shard_grad_non_exp, ccfg.shard_grad_exp, ccfg.shard_grad_exp_partial = grads

    # def config_op_level(self, ccfg, strategy):
    #     def full_partial():
    #         return Config({"full":0, "partial":0})
    #     def exp_or_not():
    #         return Config({
    #             "non_exp":full_partial(),
    #             "exp":full_partial()
    #         })
    #     ccfg.op = Config({
    #         "p":exp_or_not(),
    #         "os":exp_or_not(),
    #         "grad"exp_or_not()
    #     })
    #     shard_strat = {
    #         "grad":0, #zero 1
    #         "os+grad":0, #zero 2
    #         "p+os+grad":0, # zero 3
    #         "p+os":0 # zero2 mindspore
    #     }
    #     shard_strat[strategy]

    def init_hff(self):
        """MindFormers format for FFn hidden size"""
        # Assuming following 3 variables are already parsed
        hidden_size = self.ccfg.h
        ffn_dim_multiplier = self.ccfg.fdm
        multiple_of = self.ccfg.multiple_of
        hff = 4 * hidden_size
        if ffn_dim_multiplier:
            hff = int((ffn_dim_multiplier + 0.01) * hff)
        hff = int(2 * hff / 3)
        hff = multiple_of * ((hff + multiple_of - 1) // multiple_of)
        return hff

    def config_comm_flag(self, ccfg):
        """comm flag variables"""
        ccfg.comm_d_non_exp = (
            0
            if ((ccfg.d == 1) or not ccfg.has_op)
            else (2 if not ccfg.has_grad_shard else 3)
        )  # data parallel comm factor
        ccfg.comm_d_exp = (
            0
            if ((ccfg.d_exp == 1) or not ccfg.has_op)
            else (2 if not ccfg.has_grad_shard else 3)
        )  # data parallel comm factor
        ccfg.comm_t = float(ccfg.t > 1)  # tensor parallel comm factor
        ccfg.comm_ep = float(
            ccfg.ep > 1 or ccfg.n_exp > 1
        )  # expert parallel comm factor
        ccfg.comm_cp = float(ccfg.cp > 1)  # context parallel comm factor
        ccfg.comm_dp_overlap = 0.9  # transitional overlap, see _cost_model_variables.py
        ccfg.comm_tp_overlap = 0.5  # transitional overlap, see _cost_model_variables.py

    def config_dp_tp_exp(self, ccfg):
        """MoE strategy variables"""
        if ccfg.etp > 1:
            ccfg.t_exp = ccfg.etp
            # d * t = inner dp * outer dp * etp
            # inner dp = EP, outer dp = the rest
            ccfg.d_exp = ccfg.d * ccfg.t * ccfg.cp // ccfg.t_exp // ccfg.ep
        else:
            ccfg.t_exp = ccfg.t
            if ccfg.d >= ccfg.ep:
                ccfg.d_exp = ccfg.d // ccfg.ep
            else:
                ccfg.d_exp = ccfg.d * ccfg.t // ccfg.ep
            if ccfg.t_exp * ccfg.ep > ccfg.d * ccfg.t:
                ccfg.t_exp = 1

        exp_group1_invalid = ccfg.d_exp < 1 or ccfg.t_exp < 1
        exp_group2_invalid = ccfg.hff_exp < 1 or ccfg.n_exp < 1
        if exp_group1_invalid or exp_group2_invalid:
            raise TypeError(
                f"MoE parsing error: d_exp({ccfg.d_exp})/t_exp({ccfg.t_exp})/"
                f"hff_exp({ccfg.hff_exp})/n_exp({ccfg.n_exp})/"
                f"DP = {ccfg.d}, TP = {ccfg.t}, EP = {ccfg.ep}/"
            )
