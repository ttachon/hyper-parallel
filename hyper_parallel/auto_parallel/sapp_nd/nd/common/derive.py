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
"""The derived fields of a cost-model config, computed in one place.

A parser states primary facts: the model's dimensions, the parallel strategy
and the fixed facts of the run.  :func:`derive` computes what follows from
them, once when a config is parsed and again whenever its strategy changes,
where each parser used to compute it.
"""
import logging
import math
from typing import Any

logger = logging.getLogger(__name__)


def derive_expert_degrees(ccfg: Any, strict: bool = True) -> None:
    """Set the degrees an expert layer runs with, ``t_exp`` and ``d_exp``.

    Args:
        ccfg: The config, with its degrees and expert layout set.
        strict: Refuse degrees that leave an expert group, the expert width
            or the expert count below 1.  When false, clamp them to 1
            instead, with a warning, so that a search can still reach the
            strategy and reject it by memory.

    Raises:
        TypeError: When *strict* and the degrees cannot hold the experts.
    """
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
    if not (exp_group1_invalid or exp_group2_invalid):
        return
    if strict:
        raise TypeError(
            f"MoE parsing error: d_exp({ccfg.d_exp})/t_exp({ccfg.t_exp})/"
            f"hff_exp({ccfg.hff_exp})/n_exp({ccfg.n_exp})/"
            f"DP = {ccfg.d}, TP = {ccfg.t}, EP = {ccfg.ep}/"
        )
    logger.warning(
        "Expert degrees invalid for d=%d t=%d ep=%d etp=%d n_exp=%d; clamping to minimum values.",
        ccfg.d, ccfg.t, ccfg.ep, ccfg.etp, ccfg.n_exp,
    )
    ccfg.d_exp = max(1, ccfg.d_exp)
    ccfg.t_exp = max(1, ccfg.t_exp)
    ccfg.hff_exp = max(1, ccfg.hff_exp)
    ccfg.n_exp = max(1, ccfg.n_exp)


def derive_optimizer_sharding(ccfg: Any) -> None:
    """Set how parameters, optimizer states and gradients are sharded."""
    # Non expert params
    ccfg.shard_p_os_non_exp_partial = (
        ccfg.os_max_shard if ccfg.has_op else ccfg.t
    ) * ccfg.cp
    ccfg.shard_p_os_non_exp = (
        (ccfg.d if ccfg.has_op else 1) * ccfg.cp * ccfg.t
    )
    ccfg.shard_grad_non_exp = (
        ccfg.shard_p_os_non_exp if ccfg.has_grad_shard else ccfg.t
    )

    # Expert params
    ccfg.shard_p_os_exp_partial = math.gcd(
        ccfg.n_exp,
        (ccfg.os_max_shard if ccfg.has_op else 1) * ccfg.t_exp,
    )
    ccfg.shard_p_os_exp = (
        (ccfg.d_exp if ccfg.has_op else 1) * ccfg.cp * ccfg.t_exp
    )
    ccfg.shard_grad_exp = (
        ccfg.shard_p_os_exp
        if ccfg.has_grad_shard
        else ccfg.t_exp
    )
    ccfg.shard_grad_exp_partial = (
        ccfg.shard_p_os_exp_partial
        if ccfg.has_grad_shard
        else ccfg.t_exp
    )


def derive_comm_flags(ccfg: Any) -> None:
    """Set the communication factors and the transitional overlaps."""
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


def derive(ccfg: Any, strict: bool = True) -> None:
    """Compute the config's derived fields from its primary ones.

    Args:
        ccfg: The config, with its model, strategy and run facts set.
        strict: As in :func:`derive_expert_degrees`.

    Raises:
        TypeError: When *strict* and the degrees cannot hold the experts.
    """
    derive_expert_degrees(ccfg, strict)
    derive_optimizer_sharding(ccfg)
    derive_comm_flags(ccfg)
