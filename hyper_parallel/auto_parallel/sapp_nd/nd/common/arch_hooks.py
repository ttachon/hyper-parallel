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
"""Custom variables per model (expert knowledge)

A family's op counts are data: its profile in ``auto_parallel/op_profiles``,
or the counts its model spec declares.  What the hooks below still set is
what a profile cannot express yet: byte widths, activation sharding, and the
dense/MoE layer stack.  A hook is chosen by ``ccfg.arch``, which the parser
settles, and never by matching the model name.
"""
import math
from typing import Any, Callable, Optional
from hyper_parallel.auto_parallel._model_spec import ModelSpecError, OpCounts
from hyper_parallel.auto_parallel._op_profiles import load_op_profile
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.logger import logger


class CWrap:
    """Temporary evaluator-like instance"""

    def __init__(self, e) -> None:
        self.ccfg = e

    def set_ccfg(self, hook):
        """Apply a hook to the wrapped config object."""
        return hook(self.ccfg)

    def get_model_name(self):
        """Return model name from the wrapped config."""
        return self.ccfg.model_name

    def reset(self, e):
        """Replace the wrapped config object."""
        self.ccfg = e

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            return lambda *args, **kwargs: None
        return self.__dict__[attr]

    def set_strategy(self, **kwargs):
        """Forward strategy updates to the wrapped config."""
        self.ccfg.set_strategy(**kwargs)

    def get_strategy(self):
        """Return strategy from the wrapped config."""
        return self.ccfg.get_strategy()


def layer_op_counts(ccfg: Any, arch: str, kind: str) -> OpCounts:
    """Return the op counts of one layer kind.

    The counts the parser recorded from the model spec when there are any,
    else the family's own profile, which is what a config assembled without
    a parser gets.

    Raises:
        ModelSpecError: If the recorded counts have no such kind.
    """
    table = getattr(ccfg, "op_counts", None)
    if not table:
        return load_op_profile(arch).counts(kind)
    if kind not in table:
        raise ModelSpecError(
            f"{ccfg.model_name}: no op counts for layer kind {kind!r}, "
            f"the spec declares {sorted(table)}"
        )
    return table[kind]


def apply_op_counts(ccfg: Any, counts: OpCounts) -> None:
    """Set one layer kind's op counts on *ccfg*, each as ``n_<op>``."""
    for name, count in counts.to_dict().items():
        setattr(ccfg, "n_" + name, count)
    # Parameters are cast when the optimizer does not shard them.
    ccfg.n_attParamCast = ccfg.n_attMM if not ccfg.has_op else 0
    ccfg.n_ffParamCast = ccfg.n_ffMM if not ccfg.has_op else 0


def layer_hook(
    arch: str, kind: str, extra: Optional[Callable[[Any], None]] = None
) -> Callable[[Any], None]:
    """Return the hook that makes a group of layers one kind.

    Same contract as every layer hook: the memory backbone calls it with an
    evaluator, the performance path with a bare config.

    Args:
        arch: The family whose profile holds the counts when the config
            carries none of its own.
        kind: The layer kind to apply.
        extra: Sets what the kind carries beyond its op counts.
    """

    def apply(c: Any) -> None:
        """Give config *c* the kind's counts, then its extra fields."""
        apply_op_counts(c, layer_op_counts(c, arch, kind))
        if extra is not None:
            extra(c)

    def hook(e: Any) -> None:
        """Apply the kind to an evaluator, or to a bare config."""
        if isinstance(e, CostModelConfig):
            e = CWrap(e)
        e.set_ccfg(apply)

    hook.__name__ = f"hook_{kind}"
    return hook


def _set_bytes(ccfg: Any, grad: int = 4, dropout: int = 0) -> None:
    """Byte widths a family sets alongside its op counts."""
    ccfg.bytes_grad = grad if ccfg.p > 1 else 0  # gradients
    ccfg.bytes_os = 4  # optimizer states
    ccfg.bytes_dropout = dropout  # dropout mask
    ccfg.bytes_norm = 4  # normalization input


def _decoder(ccfg: Any, arch: str) -> None:
    """Op counts of the family's default kind, and the byte widths of a decoder."""
    apply_op_counts(ccfg, layer_op_counts(ccfg, arch, load_op_profile(arch).default))
    _set_bytes(ccfg)


def custom_default_transformer(ccfg):
    """base"""
    _decoder(ccfg, "default")


def custom_llama2(ccfg):
    """llama2"""
    _decoder(ccfg, "llama2")
    ccfg.bytes_grad = 2  # gradients


def custom_mixtral(ccfg):
    """mixtral"""
    apply_op_counts(ccfg, layer_op_counts(ccfg, "mixtral", "decoder"))
    _set_bytes(ccfg, grad=2)
    ccfg.hff = ccfg.hff_exp


def custom_t5(ccfg):
    """t5"""

    def t5_bytes(c: Any) -> None:
        """Both stacks store a one-byte dropout mask."""
        _set_bytes(c, dropout=1)

    # Encoder + Decoder
    ccfg.layer_custom_config = [
        (ccfg.n_lay // 2, layer_hook("t5", "encoder", t5_bytes)),
        (ccfg.n_lay // 2, layer_hook("t5", "decoder", t5_bytes)),
    ]


def custom_pangualpha(ccfg):
    """pangualpha"""
    apply_op_counts(ccfg, layer_op_counts(ccfg, "pangualpha", "decoder"))
    _set_bytes(ccfg, dropout=1)


def custom_deepseek3(ccfg, arch="deepseek"):
    """deepseekv3"""
    saved = Config({})
    if ccfg.config_format == "yaml":
        saved.hff = int(ccfg.hff)
    elif ccfg.config_format == "json":
        saved.hff = ccfg.ffn_hidden_size
    else:
        saved.hff = ccfg.specs.inter_dim
        if not saved.hff:
            saved.hff = ccfg.specs.hidden_dim
        if not saved.hff:
            saved.hff = ccfg.h
    saved.n_chosen_exp = ccfg.n_chosen_exp
    saved.n_exp = ccfg.n_exp
    saved.n_shared_exp = ccfg.n_shared_exp
    saved.ep = ccfg.ep
    _decoder(ccfg, arch)
    ccfg.dh = 128

    def dense(c):
        c.hff = saved.hff
        c.n_chosen_exp = 1
        c.n_exp = 1
        c.n_shared_exp = 0

    def moe(c):
        c.hff = c.hff_exp
        c.n_chosen_exp = saved.n_chosen_exp
        c.n_exp = saved.n_exp
        c.n_shared_exp = saved.n_shared_exp

    def hook_dense(e):
        if isinstance(e, CostModelConfig):
            e = CWrap(e)
        # e.ccfg.ep = 1
        e.set_ccfg(dense)
        e.ccfg.ep = 1
        # e.set_strategy(ep=1)

    def hook_moe(e):
        if isinstance(e, CostModelConfig):
            e = CWrap(e)
        # e.ccfg.ep = saved.ep
        e.set_ccfg(moe)
        e.ccfg.ep = saved.ep
        # e.set_strategy(ep=saved.ep)

    n_moe = ccfg.n_lay - ccfg.k_1st_dense
    ccfg.layer_custom_config = [
        (ccfg.k_1st_dense, hook_dense),
        (n_moe, hook_moe),
        (ccfg.n_mtp, hook_moe if n_moe > 0 else hook_dense),
    ]


def custom_qwen(ccfg):
    """qwen2"""
    _decoder(ccfg, "qwen")
    # if "72b" in ccfg.model_name :
    #     ccfg.s = ccfg.s * 3/4
    ccfg.shard_recompute_input = ccfg.t
    ccfg.shard_output_activ = ccfg.t
    # ccfg.bytes_grad = 4


def custom_cm(ccfg):
    """llama moe"""
    shard_p_os_exp = ccfg.shard_p_os_exp_partial
    shard_p_os_non_exp_partial = math.gcd(ccfg.n_exp, ccfg.shard_p_os_non_exp)
    shard_embed = ccfg.t
    custom_deepseek3(ccfg, arch="cm")

    def custom_shard(c):
        c.shard_p_os_exp = shard_p_os_exp
        c.shard_p_os_non_exp_partial = shard_p_os_non_exp_partial
        c.shard_embed = shard_embed

    for idx, f in enumerate(ccfg.layer_custom_config):

        def wrap_hook(e, f=f):
            if isinstance(e, CostModelConfig):
                e = CWrap(e)
            f[1](e)
            e.set_ccfg(custom_shard)

        ccfg.layer_custom_config[idx] = (f[0], wrap_hook)

    def num_params_norm_cm(c, _):
        return c.n_normOp * 2 * c.h + 0.5 * c.n_attMM * c.dh

    ccfg.overwrite_eval_functions["num_params_norm"] = num_params_norm_cm


def custom_vision_tower(ccfg: Any) -> None:
    """Vision tower of a multimodal model.

    Always priced with the vision profile.  A tower's config carries the arch
    of its language model, whose hook the evaluator applies to it first.
    """
    apply_op_counts(ccfg, load_op_profile("vision").counts("encoder"))
    _set_bytes(ccfg)


ARCH_HOOKS = {
    "default": custom_default_transformer,
    "llama2": custom_llama2,
    "mixtral": custom_mixtral,
    "t5": custom_t5,
    "pangualpha": custom_pangualpha,
    "deepseek": custom_deepseek3,
    "qwen": custom_qwen,
    "cm": custom_cm,
    "vision": custom_vision_tower,
}


def check_and_apply_custom_hook(e: Any) -> None:
    """Apply the hook of the family the config declares in ``ccfg.arch``."""
    if isinstance(e, CostModelConfig):
        e = CWrap(e)
    arch = getattr(e.ccfg, "arch", None)
    hook = ARCH_HOOKS.get(arch)
    if hook is None:
        logger.warning(
            "Hook not defined for: %s (arch %r). Default one is chosen",
            e.get_model_name(), arch,
        )
        hook = custom_default_transformer
    e.set_ccfg(hook)
