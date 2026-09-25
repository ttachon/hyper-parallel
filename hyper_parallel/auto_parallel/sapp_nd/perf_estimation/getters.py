# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Getters"""

from copy import deepcopy
from typing import Any, Dict, List, Tuple
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import perf_logger as logger


# Configs are updated depending on the type of the transformer layer
def get_layer_custom_configs(cfg):
    """Stores each configuration along with how many layers are affected by it
    in ascending order of execution in a forward pass
    """

    if cfg.layer_custom_config is None or any(
        func is None for (_, func) in cfg.layer_custom_config
    ):
        return [(cfg, cfg.n_lay)]

    lccfgs = []
    for nb_layers, func in cfg.layer_custom_config:
        lccfg = deepcopy(cfg)
        func(lccfg)
        lccfgs.append((lccfg, nb_layers))

    return lccfgs


def get_model_order(cfg: Any, stages: List) -> List[Tuple[int, int, int]]:
    """Positions ``(stage, chunk, index)`` of the regular layers, in model order.

    Model order runs chunk by chunk across the stages, and back up them in
    the second chunk of a V schedule; walking the stages one at a time is
    only the same order without interleaving.  This is the order the memory
    backbone gives layers their groups in.
    """
    positions = [
        (stage_id, chunk_id, lay_id)
        for chunk_id in range(cfg.vp)
        for stage_id in range(cfg.p)
        for lay_id, layer in enumerate(stages[stage_id][chunk_id])
        if layer not in (LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER)
    ]
    if cfg.pp_sched == "zero_bubble_v":
        # First-chunk entries less the embedding, as the memory backbone counts.
        first = sum(len(stage[0]) for stage in stages) - 1
        positions = positions[:first] + positions[first:][::-1]
    return positions


def get_layer_configs_by_position(cfg: Any, stages: List) -> Dict[Tuple[int, int, int], Any]:
    """Map each regular layer's position to the config it is priced with.

    Each group of ``layer_custom_config`` covers exactly its own count of
    layers in model order; layers past the declared counts keep the last
    group, as a configuration with no custom groups keeps ``cfg``.
    """
    lccfgs = get_layer_custom_configs(cfg)
    per_layer = [lccfg for lccfg, count in lccfgs for _ in range(count)]
    last = lccfgs[-1][0]
    return {
        position: per_layer[idx] if idx < len(per_layer) else last
        for idx, position in enumerate(get_model_order(cfg, stages))
    }


# The switch an op answers to where it has none of its own: a QK-norm is a norm.
_SWITCH_OF = {"qknorm": "normOp"}


def get_recomp_factor(lccfg, layer, op_name):
    """Whether a layer of this type runs the op again in its backward pass.

    A selective layer recomputes exactly the ops whose switch in
    ``lccfg.rec_op`` is 0. A switch at 1 keeps the op's activation, which is
    how the memory model's ``EvalUtils.rec_coeff`` reads it, and an op with no
    switch is kept. The switches are read from ``vars`` because a ``Config``
    answers 0 for any attribute it lacks.
    """
    if layer == LayerType.FULL_REC_LAYER:
        return 1
    if layer == LayerType.NOT_REC_LAYER:
        return 0
    if layer == LayerType.SEL_REC_LAYER:
        switches = vars(lccfg.rec_op) if lccfg.rec_op is not None else {}
        return int(not switches.get(_SWITCH_OF.get(op_name, op_name), 1))
    logger.warning("Unrecognized recompute type %s", layer)
    return 0


def get_table_quantity(lccfg, table, layer, with_recomp):
    """op compute load from given table"""
    qt_layer = 0
    for op, quantity in table.items():
        op_name = op[2:]

        qt_layer += (
            (1 + with_recomp * get_recomp_factor(lccfg, layer, op_name))
            * getattr(lccfg, op)
            * quantity
        )
    return qt_layer
