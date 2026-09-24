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
"""How an execution spec reaches a cost-model config.

:func:`apply_exec` is the one way a strategy reaches a config: it writes the
fields an :class:`~hyper_parallel.auto_parallel._exec_spec.ExecSpec` states
and derives the rest.  :func:`exec_of` reads a config's ExecSpec back, for a
parser that fills the config itself, and :func:`strategy_exec` is the
ExecSpec a search's keyword strategy states.  A config a search owns refuses
any other write of a degree, but :func:`apply_layer_strategy`, which gives
the layer about to be priced the degrees its kind runs with.
"""
from typing import Any, Mapping

from hyper_parallel.auto_parallel._exec_spec import ExecSpec
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import derive

# Each ExecSpec field, and the config field that holds it.
CONFIG_FIELDS = {
    "dp": "d",
    "tp": "t",
    "pp": "p",
    "vpp": "vp",
    "cp": "cp",
    "ep": "ep",
    "etp": "etp",
    "sequence_parallel": "sequence_parallel",
    "micro_batch_size": "b",
    "micro_batch_num": "m",
    "global_batch_size": "gbs",
    "optimizer_parallel": "has_op",
    "optimizer_shard": "os_max_shard",
    "grad_shard": "has_grad_shard",
    "pp_schedule": "pp_sched",
    "offset": "offset",
    "seq_split": "n_s_split",
    "pp_partition": "pp_partition",
    "mtp_in_offset": "is_mtp_in_offset",
    "emb_out_in_offset": "emb_out_in_offset",
    "full_recompute": "full_rec",
    "selective_recompute": "sel_rec",
    "selective_comm_recompute": "sel_comm_rec",
    "selective_rule": "sel_rec_rule",
    "recompute_slice_activation": "recompute_slice_activation",
    "param_bytes": "bytes_p",
    "compute_bytes": "bytes_compute",
    "softmax_bytes": "bytes_softmax",
    "grad_bytes": "bytes_grad",
    "optimizer_state_bytes": "bytes_os",
    "norm_bytes": "bytes_norm",
    "flash_attention": "has_fa",
    "grad_clip": "has_clip",
    "grouped_gemm": "gmm",
    "vocab_emb_dp": "vocab_emb_dp",
    "emb_dp_sharded": "emb_dp_sharded",
    "tie_embeddings": "tie_emb_out",
    "cp_algo": "cp_algo",
    "capacity_factor": "cap_fact",
    "shard_mtp_param": "is_shard_mtp_param",
    "frozen": "freeze",
    "optimizer": "optimizer",
    "seq_length": "s",
    "device_memory": "device_capacity",
}

# The keyword strategy of ``CostModelConfig.set_strategy``: each key that
# states an integer, and the ExecSpec field it states.
STRATEGY_KEYS = (
    ("dp", "dp"),
    ("mp", "tp"),
    ("ep", "ep"),
    ("etp", "etp"),
    ("cp", "cp"),
    ("vpp", "vpp"),
    ("pp", "pp"),
    ("mb", "micro_batch_num"),
    ("mbs", "micro_batch_size"),
)


def exec_of(ccfg: Any) -> ExecSpec:
    """Read back the ExecSpec a config holds, every field it states."""
    return ExecSpec(**{name: getattr(ccfg, field) for name, field in CONFIG_FIELDS.items()})


def apply_exec(ccfg: Any, exec_spec: ExecSpec, strict: bool = True) -> None:
    """Write the fields *exec_spec* states on *ccfg*, then derive the rest.

    A field the spec does not state keeps the value the config holds.

    Args:
        ccfg: The config to change.
        exec_spec: What to change it to.
        strict: As in :func:`~hyper_parallel.auto_parallel.sapp_nd.nd.common.derive.derive`.
    """
    for name, field in CONFIG_FIELDS.items():
        value = getattr(exec_spec, name)
        if value is None:
            continue
        if name == "device_memory" and isinstance(value, str):
            value = _memory(value)
        # The one sanctioned write of a strategy field, past the guard.
        object.__setattr__(ccfg, field, value)
    derive(ccfg, strict)


def apply_layer_strategy(ccfg: Any, degrees: Mapping[str, Any]) -> None:
    """Give the layer about to be priced the degrees its kind runs with.

    A dense kind in a MoE model runs with no expert parallelism.  Only the
    degrees are written: the fields derived from the model's degrees stay as
    the estimators read them.

    Args:
        ccfg: The config the layer is priced on.
        degrees: The strategy fields the kind states.
    """
    for name, value in degrees.items():
        object.__setattr__(ccfg, name, value)


def strategy_exec(ccfg: Any, strategy: Mapping[str, Any]) -> ExecSpec:
    """Return the ExecSpec a keyword strategy states, over the config it changes.

    A degree or a micro-batching is stated when it is given as an integer,
    and ``op`` states the optimizer sharding.  The search runs sequence
    parallelism at every TP degree, and the global batch follows the
    micro-batching the strategy leaves.  The recompute layers and the offset
    are stated when given.

    Args:
        ccfg: The config the strategy changes, which holds what it leaves.
        strategy: ``set_strategy``'s keywords.
    """
    stated = {
        name: strategy[key] for key, name in STRATEGY_KEYS if isinstance(strategy.get(key), int)
    }
    op = strategy.get("op")
    if isinstance(op, int):
        stated["optimizer_shard"] = op
        # op <= 1 means no optimizer sharding
        stated["optimizer_parallel"] = op > 1
    stated["sequence_parallel"] = True
    stated["global_batch_size"] = (
        stated.get("micro_batch_size", ccfg.b) * stated.get("dp", ccfg.d)
        * stated.get("micro_batch_num", ccfg.m)
    )
    if strategy.get("full_rec") is not None:
        stated["full_recompute"] = strategy["full_rec"]
    if strategy.get("sel_rec") is not None:
        stated["selective_recompute"] = strategy["sel_rec"]
    if isinstance(strategy.get("offset"), (int, list)):
        stated["offset"] = strategy["offset"]
    return ExecSpec(**stated)


def _memory(text: str) -> Any:
    """The cost model's memory size for a string such as ``"64GB"``."""
    # The memory estimation package imports the config module, which imports
    # this one: importing it here keeps this module importable first.
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory  # pylint: disable=import-outside-toplevel
    return Memory.from_string(text)
