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
