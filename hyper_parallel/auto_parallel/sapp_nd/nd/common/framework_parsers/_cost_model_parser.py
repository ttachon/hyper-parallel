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
from typing import TYPE_CHECKING

from abc import ABC
from abc import abstractmethod

from hyper_parallel.auto_parallel._layer_stack import derive_layers, resolve_layers
from hyper_parallel.auto_parallel._op_profiles import infer_arch, load_op_profile, resolve_ops
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger

if TYPE_CHECKING:
    from typing import Mapping, Optional
    from hyper_parallel.auto_parallel._layer_stack import LayerStack
    from hyper_parallel.auto_parallel._model_spec import OpCounts
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

    def config_op_counts(
        self,
        ccfg: _CostModVar,
        arch: Optional[str] = None,
        ops: Optional[Mapping[str, OpCounts]] = None,
    ) -> None:
        """Settle the op profile the arch hooks price the model with.

        A parser that reads a declared arch passes it, with any op counts the
        spec declares; one that has only a model name leaves the family to
        :func:`infer_arch`.
        """
        ccfg.arch = arch or infer_arch(ccfg.model_name)
        ccfg.op_counts = resolve_ops(ccfg.arch, ops)

    @staticmethod
    def config_layer_stack(ccfg: _CostModVar, stack: Optional[LayerStack] = None) -> None:
        """Settle the layer stack the estimators price.

        Args:
            ccfg: The config, with its arch, op counts and layer counts parsed.
            stack: The stack the model states.  By default, the one its layer
                counts imply: dense layers first, two halves, or one kind.
        """
        if stack is None:
            layers = derive_layers(
                load_op_profile(ccfg.arch), int(ccfg.n_lay), int(ccfg.n_mtp or 0),
                first_k_dense=int(ccfg.k_1st_dense or 0),
            )
            stack = resolve_layers(ccfg.arch, layers, ccfg.op_counts)
        ccfg.layer_stack = stack
        if len(stack.groups) > 1:
            logger.info(
                "%s layer stack: %s", ccfg.model_name,
                ", ".join(f"{group.count}x{group.kind.name}" for group in stack.groups),
            )

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
