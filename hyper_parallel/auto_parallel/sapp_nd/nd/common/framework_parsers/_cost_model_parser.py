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

import logging
import math
from abc import ABC
from abc import abstractmethod

if TYPE_CHECKING:
    from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import _CostModVar

logger = logging.getLogger(__name__)

# The optimizer a config that states none is priced as, with a warning: the
# shipped recipes state AdamW and Muon both, so none is assumed silently.
DEFAULT_OPTIMIZER = "adamw"

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
    def state_optimizer(ccfg: Any, stated: Any) -> None:
        """Name the optimizer a config states, and the states it keeps a parameter, which its name decides.

        Muon keeps one momentum a matrix and AdamW two moments a parameter.
        A config that states no optimizer is priced as AdamW, and the parser
        says so: the HyperParallel parser named one silently and charged its
        two states, which a Muon run described without its optimizer does
        not keep, and the MindFormers parser failed on the missing section
        (M3). ``ccfg.optimizer`` is what the state count is read from (M5).
        """
        if not stated:
            logger.warning("the config states no optimizer: it is priced as %s, two states a parameter",
                           DEFAULT_OPTIMIZER)
            stated = DEFAULT_OPTIMIZER
        ccfg.optimizer = str(stated)
        ccfg.optimizer_states = 1 if "muon" in ccfg.optimizer.lower() else 2

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
    def expert_dp_group(ccfg):
        """How many ranks hold the same experts.

        The runtime spreads the experts over a whole pipeline stage: its expert
        mesh is ``(edp_replicate, edp_shard, ep)`` over the ``dp * cp * tp``
        ranks of the device mesh, so ``dp * cp * tp / ep`` ranks hold the same
        experts, and an expert tensor shard divides that group again
        (``hyper_parallel/distributed/mesh.py``,
        ``MeshContext._build_expert_parallel_mesh``).

        ``d_exp`` cannot stand in for this group.  It is floored by EP against
        DP alone, so it loses the remainder when EP does not divide DP and
        falls to zero once EP passes DP, which is what context parallelism
        does: DP counts the stage's ranks apart from CP and TP, so a strategy
        keeps its group while its DP shrinks with CP.  Returning zero made a
        runnable strategy look invalid, and the consumers that rebuild the
        group as ``d_exp * cp * t_exp`` cannot express a group narrower than
        CP.
        """
        expert_tp = int(ccfg.etp) if ccfg.etp > 1 else 1
        return int(ccfg.d * ccfg.t * ccfg.cp) // int(expert_tp * ccfg.ep)

    @staticmethod
    def expert_dp_ranks(ccfg: Any) -> int:
        """How many ranks hold the same slice of a routed expert, so reduce its gradient together.

        The stage's ranks over EP and over the expert's tensor shard
        ``t_exp``, whose ranks hold different slices: context parallelism's
        ranks hold the same weights and reduce their gradients with the
        data-parallel ones, where ``d_exp`` counts the latter alone outside an
        expert tensor shard, and :meth:`expert_dp_group` counts TP's ranks
        whatever slice they hold.
        """
        return max(1, int(ccfg.d * ccfg.t * ccfg.cp) // (max(1, int(ccfg.t_exp)) * max(1, int(ccfg.ep))))

    @staticmethod
    def routed_expert_shard(ccfg, ranks):
        """How many ranks a routed expert's parameters and optimizer states are sharded over.

        A run that states its expert shard, as HyperParallel's ``edp_shard_size``,
        shards an expert under expert parallelism over as many ranks of its expert
        data-parallel group: the stage's ranks over EP, whatever their DP, CP or
        TP, the group of ranks that hold the same experts.  Without expert
        parallelism its FSDP shards the experts with the other parameters, over
        the optimizer's *ranks*.  Stated by no one, the optimizer shards them over
        the whole group.  A run that shards each strategy's experts over its
        whole group (``expert_shard_group``) does so under expert parallelism
        whatever shard it states, since the group changes with the strategy.
        """
        stated = getattr(ccfg, "expert_shard", None)
        # The group as the runtime forms it, once a strategy has been laid over
        # the config; a dense model and the parser's own doubles state none, and
        # keep the older reconstruction from d_exp, which agrees with it
        # wherever EP divides DP.
        whole = max(1, int(ccfg.d * ccfg.cp * ccfg.t // max(1, int(ccfg.ep))))
        group = max(1, int(ccfg.edp_group)) if ccfg.edp_group else whole
        if getattr(ccfg, "expert_shard_group", False) and ccfg.ep > 1:
            return group
        if not stated:
            if not ccfg.has_op:
                return ccfg.cp * ccfg.t_exp
            return group if ccfg.edp_group else ccfg.d_exp * ccfg.cp * ccfg.t_exp
        if ccfg.ep > 1:
            return math.gcd(int(stated), group)
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
        # Whether another rank holds the same slice of a routed expert, as a
        # shard or a copy, so that its gradient is reduced: d_exp left CP's
        # ranks out and read 1 where they reduce it (X3).
        ccfg.comm_d_exp = (
            0
            if ((_CostModelParser.expert_dp_ranks(ccfg) == 1) or not ccfg.has_op)
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

        # The group the runtime forms, which is what every consumer wants; a
        # strategy is impossible only when the stage holds fewer ranks than it
        # spreads experts over.  d_exp stays the older form, and is clamped
        # because a group of one still leaves it at zero under CP.
        ccfg.edp_group = _CostModelParser.expert_dp_group(ccfg)
        ccfg.d_exp = max(1, ccfg.d_exp)
        # A MoE layer dispatches its tokens once and takes them back once.
        ccfg.n_dispatch = 1 if ccfg.n_exp > 1 else 0

        exp_group1_invalid = ccfg.edp_group < 1 or ccfg.t_exp < 1
        exp_group2_invalid = ccfg.hff_exp < 1 or ccfg.n_exp < 1
        if exp_group1_invalid or exp_group2_invalid:
            raise TypeError(
                f"MoE parsing error: edp_group({ccfg.edp_group})/"
                f"t_exp({ccfg.t_exp})/"
                f"hff_exp({ccfg.hff_exp})/n_exp({ccfg.n_exp})/"
                f"DP = {ccfg.d}, TP = {ccfg.t}, EP = {ccfg.ep}, CP = {ccfg.cp}/"
            )
