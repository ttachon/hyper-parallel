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
what a profile cannot express yet: byte widths and activation sharding.  A
hook is chosen by ``ccfg.arch``, which the parser settles, and never by
matching the model name.

The layer stack is data too (``ccfg.layer_stack``), and needs no hook of its
own.  The estimators read each layer's kind from :func:`layer_groups`, and
:func:`apply_layer_kind` gives a layer its kind from the fields
:func:`bind_layer_stack` recorded when the family hook ran.  A family whose
layers take some fields per layer rather than on the model, such as t5's
byte widths, leaves them in ``ccfg.layer_fields`` for every kind to assign.
"""
import math
from typing import Any, Dict, List, Optional, Tuple
from hyper_parallel.auto_parallel._layer_stack import LayerStack, LinearAttentionDims
from hyper_parallel.auto_parallel._model_spec import ModelSpecError, OpCounts
from hyper_parallel.auto_parallel._op_profiles import VISION_ARCH, LayerKind, load_op_profile
from hyper_parallel.auto_parallel.sapp_nd.nd.common.apply_exec import apply_layer_strategy
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


# The fields an attention flavour assigns.  Every kind of a stack whose kinds
# differ in attention writes all of them, so applying kinds in place, one
# layer after another and in any order, leaves each layer the same config.
_ATTENTION_FIELDS = (
    "attn_kind", "a", "dh", "n_kv", "attn_output_gate", "attn_extra_p",
    "lin_n_k", "lin_d_k", "lin_n_v", "lin_d_v", "lin_conv",
)

# The fields a feed-forward flavour assigns, likewise.
_FFN_FIELDS = ("hff", "n_chosen_exp", "n_exp", "n_shared_exp", "ep")

# Strategy a kind still sets.  The guard refuses a strategy write through
# set_ccfg, so the applier gives them through apply_layer_strategy.
_STRATEGY_FIELDS = ("ep",)


def _linear_attention(snapshot: Any, linear: LinearAttentionDims) -> Dict[str, Any]:
    """The attention fields of a gated-DeltaNet layer.

    The flavour maps onto the q/k/v/o formula: the value heads carry the
    q-side width, the key heads the kv-side, and the output gate is a second
    q-wide tensor.  What the formula does not describe is stated apart: the
    short convolution over the projected stream and the two per-head gates
    as extra parameters, the recurrent state update as the ``linrec`` op.
    """
    n_k, d_k = linear.num_key_heads, linear.key_head_dim
    n_v, d_v = linear.num_value_heads, linear.value_head_dim
    qkv_width = 2 * n_k * d_k + n_v * d_v
    return {
        "attn_kind": "linear",
        "a": n_v,
        "dh": d_v,
        "n_kv": n_k * d_k / d_v,
        "attn_output_gate": True,
        "attn_extra_p": linear.conv_kernel_dim * qkv_width + 2 * snapshot.h * n_v,
        "lin_n_k": n_k,
        "lin_d_k": d_k,
        "lin_n_v": n_v,
        "lin_d_v": d_v,
        "lin_conv": linear.conv_kernel_dim,
    }


def _feed_forward(snapshot: Any, flavour: Optional[str]) -> Dict[str, Any]:
    """The feed-forward fields of a layer of *flavour*, from the bound config.

    A MoE layer runs the routed experts at their width.  A dense layer runs
    one expert at the model's feed-forward width, the parser's ``hff``, with
    no shared expert and no expert parallelism.  A kind without a flavour
    keeps the model's own.
    """
    if flavour == "dense":
        return {"hff": snapshot.hff, "n_chosen_exp": 1, "n_exp": 1, "n_shared_exp": 0, "ep": 1}
    fields = {name: getattr(snapshot, name) for name in _FFN_FIELDS}
    if flavour == "moe":
        fields["hff"] = snapshot.hff_exp
    return fields


def _kind_fields(snapshot: Any, stack: LayerStack, kind: LayerKind) -> Dict[str, Any]:
    """The fields *kind* assigns beyond its op counts, from the bound config.

    For every flavour some kind of the stack states, the fields it assigns,
    with the kind's values or the model's; then the fields the family gives
    every layer.
    """
    kinds = stack.distinct_kinds()
    fields: Dict[str, Any] = {}
    if any(other.attention != "full" for other in kinds):
        if kind.attention == "linear":
            fields.update(_linear_attention(snapshot, stack.linear))
        else:
            fields.update({name: getattr(snapshot, name) for name in _ATTENTION_FIELDS})
    if any(other.ffn is not None for other in kinds):
        fields.update(_feed_forward(snapshot, kind.ffn))
    fields.update(getattr(snapshot, "layer_fields", None) or {})
    return fields


def bind_layer_stack(ccfg: Any) -> None:
    """Record, per kind of the config's stack, the fields it assigns.

    Called where the family hook runs, per search candidate and per
    estimate, so the values a kind restores are the model's own as that
    hook left them.  A config without a stack gets no binding.
    """
    stack = getattr(ccfg, "layer_stack", None)
    if stack is None:
        ccfg.layer_binding = None
        return
    ccfg.layer_binding = {
        kind.name: _kind_fields(ccfg, stack, kind) for kind in stack.distinct_kinds()
    }


def apply_layer_kind(e: Any, kind: LayerKind) -> None:
    """Make the layer about to be priced one of *kind*.

    The memory backbone calls it with an evaluator, the performance path
    with a bare config.  A config without a stack gives the kind its op
    counts only.
    """
    if isinstance(e, CostModelConfig):
        e = CWrap(e)
    if getattr(e.ccfg, "layer_binding", None) is None:
        bind_layer_stack(e.ccfg)
    binding = e.ccfg.layer_binding
    fields = binding[kind.name] if binding is not None else {}

    def assign(c: Any) -> None:
        """Give config *c* the kind's counts, then its model fields."""
        apply_op_counts(c, kind.ops)
        for name, value in fields.items():
            if name not in _STRATEGY_FIELDS:
                setattr(c, name, value)

    e.set_ccfg(assign)
    apply_layer_strategy(e.ccfg, {name: fields[name] for name in _STRATEGY_FIELDS if name in fields})


def _needs_kinds(stack: LayerStack) -> bool:
    """Whether a stack's layers differ from the config the family hook leaves."""
    kinds = stack.distinct_kinds()
    return len(kinds) > 1 or any(kind.attention != "full" or kind.ffn is not None for kind in kinds)


def layer_groups(ccfg: Any) -> List[Tuple[Optional[LayerKind], int]]:
    """Return the groups of layers the estimators price alike, in model order.

    Returns:
        ``(kind, count)`` per group of the config's stack.  A config without
        a stack, or whose stack's one kind is the config the family hook
        leaves, is one group of all its layers, MTP included, with no kind:
        they are priced on the config as it stands.
    """
    stack = getattr(ccfg, "layer_stack", None)
    if stack is None or not _needs_kinds(stack):
        return [(None, int(ccfg.n_lay + ccfg.n_mtp))]
    return [(group.kind, group.count) for group in stack.groups]


def layer_kinds(ccfg: Any) -> List[Optional[LayerKind]]:
    """Return the kind of every layer in model order, as :func:`layer_groups` groups them."""
    return [kind for kind, count in layer_groups(ccfg) for _ in range(count)]


def _byte_widths(ccfg: Any, grad: int = 4, dropout: int = 0) -> Dict[str, int]:
    """Byte widths a family sets alongside its op counts."""
    return {
        "bytes_grad": grad if ccfg.p > 1 else 0,  # gradients
        "bytes_os": 4,  # optimizer states
        "bytes_dropout": dropout,  # dropout mask
        "bytes_norm": 4,  # normalization input
    }


def _set_bytes(ccfg: Any, grad: int = 4, dropout: int = 0) -> None:
    """Set a family's byte widths on the model."""
    for name, value in _byte_widths(ccfg, grad, dropout).items():
        setattr(ccfg, name, value)


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
    """t5: the encoder and decoder are kinds of its layer stack.

    Both store a one-byte dropout mask, and t5 sets its byte widths per
    layer, not on the model.
    """
    ccfg.layer_fields = _byte_widths(ccfg, dropout=1)


def custom_pangualpha(ccfg):
    """pangualpha"""
    apply_op_counts(ccfg, layer_op_counts(ccfg, "pangualpha", "decoder"))
    _set_bytes(ccfg, dropout=1)


def custom_deepseek3(ccfg, arch="deepseek"):
    """DeepSeek-V3: its dense and MoE layers are kinds of its layer stack."""
    _decoder(ccfg, arch)
    ccfg.dh = 128


def custom_qwen(ccfg, arch="qwen"):
    """qwen2"""
    _decoder(ccfg, arch)
    # if "72b" in ccfg.model_name :
    #     ccfg.s = ccfg.s * 3/4
    ccfg.shard_recompute_input = ccfg.t
    ccfg.shard_output_activ = ccfg.t
    # ccfg.bytes_grad = 4


def custom_qwen3_5(ccfg: Any) -> None:
    """Qwen3.5: Qwen's hook, on the full-attention kind of its hybrid profile.

    The linear-attention layers get their kind from the layer stack.
    """
    custom_qwen(ccfg, arch="qwen3_5")


def custom_cm(ccfg):
    """llama moe: DeepSeek's stack, each layer sharding its states as below."""
    layer_fields = {
        "shard_p_os_exp": ccfg.shard_p_os_exp_partial,
        "shard_p_os_non_exp_partial": math.gcd(ccfg.n_exp, ccfg.shard_p_os_non_exp),
        "shard_embed": ccfg.t,
    }
    custom_deepseek3(ccfg, arch="cm")
    ccfg.layer_fields = layer_fields

    def num_params_norm_cm(c, _):
        return c.n_normOp * 2 * c.h + 0.5 * c.n_attMM * c.dh

    ccfg.overwrite_eval_functions["num_params_norm"] = num_params_norm_cm


def custom_vision_tower(ccfg: Any) -> None:
    """Vision tower of a multimodal model.

    Always priced with the vision profile.  A tower inherits the hook of its
    language model's family, named in ``ccfg.inherited_arch``, which runs
    first and gives it that family's activation sharding.
    """
    inherited = getattr(ccfg, "inherited_arch", None)
    if inherited not in (None, VISION_ARCH):
        ARCH_HOOKS[inherited](ccfg)
    apply_op_counts(ccfg, load_op_profile(VISION_ARCH).counts("encoder"))
    _set_bytes(ccfg)


ARCH_HOOKS = {
    "default": custom_default_transformer,
    "llama2": custom_llama2,
    "mixtral": custom_mixtral,
    "t5": custom_t5,
    "pangualpha": custom_pangualpha,
    "deepseek": custom_deepseek3,
    "qwen": custom_qwen,
    "qwen3_5": custom_qwen3_5,
    "cm": custom_cm,
    VISION_ARCH: custom_vision_tower,
}


def check_and_apply_custom_hook(e: Any) -> None:
    """Apply the hook of the family the config declares in ``ccfg.arch``.

    Then bind the config's layer stack, so each kind restores the values
    that hook left.
    """
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
    bind_layer_stack(e.ccfg)
