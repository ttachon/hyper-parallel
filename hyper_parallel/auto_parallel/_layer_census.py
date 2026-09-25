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
"""Census of what a layer of each kind keeps for its backward, on a fake layer.

The census builds one decoder layer of a Transformers model under
``FakeTensorMode``, so it allocates no memory, runs its forward and its
backward at micro-batch 1, and counts bytes: those the autograd graph saves
between the two passes, and the most the backward holds live at once, less
the parameters and their gradients, which the memory model counts on its
own.  It runs the kernels the runtime runs: flash attention, which saves its
inputs, its output and two fp32 softmax statistics per head and token, and
HyperParallel's chunked gated-delta-rule kernel, which saves the queries,
keys, values, the cumulated gates, beta and one chunk matrix.  A layer built
with half the heads and half the feed-forward widths, as a tensor-parallel
rank of 2 holds it, tells the bytes tensor parallelism splits from those it
does not.

The output layer's census runs the final norm, the output projection and
Transformers' causal-LM loss, which casts the logits to fp32, as
HyperParallel's trainer runs them by default, dropping the model's logits
before the backward as the trainer does; half the vocabulary tells the
bytes a vocabulary-parallel loss splits.  A model spec states a census as
:class:`KindActivations` records.
"""
from __future__ import annotations

import copy
import importlib
import inspect
import weakref
from dataclasses import dataclass, fields
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch  # pylint: disable=forbidden-backend-import
from torch._subclasses.fake_tensor import FakeTensorMode  # pylint: disable=forbidden-backend-import
from torch.utils._python_dispatch import TorchDispatchMode  # pylint: disable=forbidden-backend-import
from torch.utils._pytree import tree_flatten  # pylint: disable=forbidden-backend-import



@dataclass(frozen=True)
class KindActivations:
    """What one layer of a kind keeps for its backward, and the most its backward holds, per token.

    Bytes per token at micro-batch 1, as the census measures them on one
    layer at ``seq_length`` tokens under the runtime's kernels: the part no
    tensor-parallel rank splits, and the part it splits, of which a layer at
    TP 2 holds half.  The backward's working set leaves out the parameters
    and their gradients, which the memory model counts on its own.
    """

    saved: float
    saved_tp: float
    working: float
    working_tp: float
    seq_length: int

    def to_dict(self) -> Dict[str, Any]:
        """Return the record as a plain mapping."""
        return {record_field.name: getattr(self, record_field.name) for record_field in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str = "activations") -> "KindActivations":
        """Build a record, refusing a key it does not know, a missing one or a negative size."""
        if not isinstance(data, Mapping):
            raise ValueError(f"{where} must map the record's fields to their values, got {data!r}")
        names = [record_field.name for record_field in fields(cls)]
        unknown = sorted(set(data) - set(names))
        missing = [name for name in names if name not in data]
        if unknown or missing:
            raise ValueError(f"{where} has unknown keys {unknown} and lacks {missing}")
        sizes = {name: float(data[name]) for name in names if name != "seq_length"}
        if any(size < 0 for size in sizes.values()):
            raise ValueError(f"{where}: bytes per token cannot be negative, got {sizes}")
        seq_length = int(data["seq_length"])
        if seq_length <= 0:
            raise ValueError(f"{where}.seq_length must be positive, got {seq_length}")
        return cls(seq_length=seq_length, **sizes)


def activations_from_dict(data: Any) -> Dict[str, KindActivations]:
    """Parse a spec's ``activations``, a mapping of layer kind to its :class:`KindActivations`."""
    if not isinstance(data, Mapping):
        raise ValueError(f"activations must map layer kinds to their records, got {data!r}")
    return {str(kind): KindActivations.from_dict(record, f"activations.{kind}") for kind, record in data.items()}


# The Transformers attention implementation the census registers its flash
# attention under.
_FLASH = "nd_census_flash"

# The config fields a tensor-parallel rank holds a share of.
_TP_FIELDS = (
    "num_attention_heads", "num_key_value_heads", "intermediate_size", "moe_intermediate_size",
    "shared_expert_intermediate_size", "linear_num_key_heads", "linear_num_value_heads",
)

# The softmax statistics flash attention keeps per head and token.
_FLASH_STATS = 8


class _FlashAttention(torch.autograd.Function):
    """Flash attention's saved set, shapes only: its inputs, its output and two fp32 statistics."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor,  # pylint: disable=arguments-differ
                value: torch.Tensor) -> torch.Tensor:
        """Save the inputs, an output of the queries' shape and the statistics; return that output."""
        out = torch.empty_like(query)
        batch, heads, seq, _ = query.shape
        stats = [torch.empty(batch, heads, seq, _FLASH_STATS, dtype=torch.float32, device=query.device)
                 for _ in range(2)]
        ctx.save_for_backward(query, key, value, out, *stats)
        return out

    @staticmethod
    def backward(ctx: Any,  # pylint: disable=arguments-differ
                 grad: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return gradients of the inputs' shapes."""
        del grad
        query, key, value = ctx.saved_tensors[:3]
        return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


def _flash_attention(module, query, key, value, attention_mask, **kwargs):  # pylint: disable=unused-argument
    """A Transformers attention function running :class:`_FlashAttention`, grouped K and V unexpanded."""
    return _FlashAttention.apply(query, key, value).transpose(1, 2).contiguous(), None


class _GatedDeltaRule(torch.autograd.Function):
    """HyperParallel's chunked gated-delta-rule kernel's saved set, shapes only."""

    @staticmethod
    def forward(ctx: Any, query: torch.Tensor, key: torch.Tensor,  # pylint: disable=arguments-differ
                value: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor, chunk: int) -> torch.Tensor:
        """Save the inputs, the cumulated gates, beta and a chunk matrix; return an output of the values' shape."""
        batch, seq, heads, _ = key.shape
        gates = torch.empty(batch, seq, heads, dtype=torch.float32, device=query.device)
        chunks = torch.empty(batch, seq, heads, chunk, dtype=key.dtype, device=query.device)
        ctx.save_for_backward(query, key, value, gates, beta, chunks)
        del gate
        return torch.empty_like(value)

    @staticmethod
    def backward(ctx: Any,  # pylint: disable=arguments-differ
                 grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """Return gradients of the inputs' shapes, and none of the chunk size."""
        del grad
        query, key, value, gates, beta, _ = ctx.saved_tensors
        return (torch.empty_like(query), torch.empty_like(key), torch.empty_like(value),
                torch.empty_like(gates), torch.empty_like(beta), None)


def _gated_delta_rule(modeling: Any) -> Callable[..., Tuple[torch.Tensor, None]]:
    """The drop-in for a Transformers ``chunk_gated_delta_rule`` running :class:`_GatedDeltaRule`."""
    l2norm = getattr(modeling, "l2norm", None)

    def run(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
            chunk_size: int = 64, use_qk_l2norm_in_kernel: bool = False,
            **_kwargs: Any) -> Tuple[torch.Tensor, None]:
        """Run the kernel's contract as Transformers calls its kernel, with no final state."""
        if use_qk_l2norm_in_kernel and l2norm is not None:
            query, key = l2norm(query, dim=-1, eps=1e-6), l2norm(key, dim=-1, eps=1e-6)
        return _GatedDeltaRule.apply(query, key, value, g, beta, chunk_size), None

    return run


class _LiveBytes(TorchDispatchMode):
    """The storages the ops it sees allocate, and the bytes they hold while they live."""

    def __init__(self) -> None:
        """Start with no storage counted."""
        super().__init__()
        self.sizes: Dict[Tuple[int, int], int] = {}
        self.generation: Dict[int, int] = {}
        self.events: list = []

    def key(self, tensor: torch.Tensor) -> Tuple[int, int]:
        """The storage behind *tensor*, in the lifetime it has now."""
        address = tensor.untyped_storage()._cdata  # pylint: disable=protected-access
        return address, self.generation.get(address, 0)

    def track(self, tensor: torch.Tensor) -> None:
        """Count *tensor*'s storage from now until it is freed."""
        storage = tensor.untyped_storage()
        key = self.key(tensor)
        if key in self.sizes:
            return
        self.sizes[key] = storage.nbytes()
        self.events.append((key, self.sizes[key]))
        weakref.finalize(storage, self._free, key)

    def _free(self, key: Tuple[int, int]) -> None:
        size = self.sizes.pop(key, None)
        self.generation[key[0]] = key[1] + 1
        if size:
            self.events.append((key, -size))

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        out = func(*args, **(kwargs or {}))
        for tensor in tree_flatten(out)[0]:
            if isinstance(tensor, torch.Tensor):
                self.track(tensor)
        return out

    def peak(self, start: int, excluded: Iterable[Tuple[int, int]], apart: Iterable[Tuple[int, int]] = ()) -> int:
        """The bytes live when the most are, from event *start* on, leaving out the *excluded* storages.

        The storages *apart* count toward finding that moment, not toward
        the bytes returned.
        """
        excluded, apart = set(excluded), set(apart)
        live = held = 0
        for key, delta in self.events[:start]:
            if key not in excluded:
                live += delta
                held += delta if key in apart else 0
        top, bytes_at_top = live, live - held
        for key, delta in self.events[start:]:
            if key in excluded:
                continue
            live += delta
            held += delta if key in apart else 0
            if live > top:
                top, bytes_at_top = live, live - held
        return bytes_at_top


def _measure(params: Sequence[torch.Tensor], inputs: Iterable[torch.Tensor],
             forward: Callable[[], torch.Tensor], backward: Callable[[torch.Tensor], None]) -> Tuple[int, int]:
    """Bytes a forward saves for its backward, and the activations the backward holds when it holds the most.

    Args:
        params: The parameters of what runs.
        inputs: The tensors the forward takes, which count from its start.
        forward: Runs the forward and returns its output.
        backward: Runs the backward from that output.

    Returns:
        The bytes the forward saves, its inputs included, and the bytes of
        activations live when the backward peaks, its output left out.  The
        parameters, which the forward's views of them bring into the
        tracker, are the memory model's to count, and so are their
        gradients, which hooks on the parameters tell apart from the
        activations' however they are shaped: the gradients the backward
        hands each parameter and the one it keeps.  They settle when the
        backward peaks, and the bytes returned are the activations held then.
    """
    stored = {p.untyped_storage()._cdata for p in params}  # pylint: disable=protected-access
    saved: Dict[int, int] = {}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        """Count a tensor autograd saves, unless it is a parameter's."""
        address = tensor.untyped_storage()._cdata  # pylint: disable=protected-access
        if address not in stored:
            saved[address] = tensor.untyped_storage().nbytes()
        return tensor

    live = _LiveBytes()
    grads: List[Tuple[int, int]] = []
    handles = [p.register_hook(lambda grad: grads.append(live.key(grad))) for p in params]
    try:
        with live, torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            for tensor in inputs:
                live.track(tensor)
            out = forward()
            start = len(live.events)
            backward(out)
    finally:
        for handle in handles:
            handle.remove()
    grads += [live.key(p.grad) for p in params if p.grad is not None]
    return sum(saved.values()), live.peak(start, [live.key(out), *(live.key(p) for p in params)], grads)


def _modeling(config: Any) -> Any:
    """The Transformers modeling module of *config*'s model."""
    model_type = str(config.model_type)
    names = [model_type]
    if model_type.endswith("_text"):
        names.append(model_type[:-len("_text")])
    for name in names:
        try:
            return importlib.import_module(f"transformers.models.{name}.modeling_{name}")
        except ImportError:
            continue
    raise ValueError(f"no Transformers modeling module for model type {model_type!r}")


def _classes(modeling: Any) -> Tuple[type, type]:
    """The decoder layer and the language model's rotary embedding of *modeling*."""
    classes = {name: cls for name, cls in vars(modeling).items() if isinstance(cls, type)}
    layers = [cls for name, cls in classes.items() if name.endswith("DecoderLayer")]
    rotary = [cls for name, cls in classes.items()
              if name.endswith("RotaryEmbedding") and "Vision" not in name]
    rotary.sort(key=lambda cls: "Text" not in cls.__name__)
    if not layers or not rotary:
        raise ValueError(f"{modeling.__name__} states no decoder layer or rotary embedding")
    return layers[0], rotary[0]


def tp_config(config: Any, tp: int) -> Any:
    """*config* as one of *tp* tensor-parallel ranks holds its layers: heads and widths divided."""
    config = copy.deepcopy(config)
    config.head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    for name in _TP_FIELDS:
        value = getattr(config, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            setattr(config, name, max(1, value // tp))
    return config


def _run(layer: Any, hidden: torch.Tensor, rotary: Any, seq_length: int) -> torch.Tensor:
    """One forward of *layer*, as its model calls it."""
    position_ids = torch.arange(seq_length).unsqueeze(0)
    with torch.no_grad():
        position_embeddings = rotary(hidden, position_ids)
    accepted = inspect.signature(layer.forward).parameters
    kwargs = {name: value for name, value in (
        ("position_embeddings", position_embeddings), ("position_ids", position_ids),
        ("attention_mask", None), ("use_cache", False)) if name in accepted}
    out = layer(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def census_layer(config: Any, layer_index: int, seq_length: int) -> Tuple[int, int]:
    """Bytes layer *layer_index* of *config* keeps for its backward, and the most its backward holds.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the micro-batch of one sequence it runs.

    Returns:
        The bytes the forward saves, its input included, and the bytes of
        activations the backward holds when it holds the most, its own
        gradients included, less the parameters and those gradients.
    """
    modeling = _modeling(config)
    layer_cls, rotary_cls = _classes(modeling)
    config = copy.deepcopy(config)
    config._attn_implementation = _FLASH  # pylint: disable=protected-access
    config._experts_implementation = "grouped_mm"  # pylint: disable=protected-access
    modeling.ALL_ATTENTION_FUNCTIONS.register(_FLASH, _flash_attention)
    with FakeTensorMode(allow_non_fake_inputs=True):
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            layer = layer_cls(config, layer_index)
            rotary = rotary_cls(config=config)
        finally:
            torch.set_default_dtype(default)
        for module in layer.modules():
            if hasattr(module, "chunk_gated_delta_rule"):
                module.chunk_gated_delta_rule = _gated_delta_rule(modeling)
        layer.train()
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16)
        return _measure(list(layer.parameters()), (hidden, grad),
                        lambda: _run(layer, hidden, rotary, seq_length), lambda out: out.backward(grad))


def census_output(config: Any, seq_length: int) -> Tuple[int, int]:
    """Bytes the output layer of *config* keeps for its backward, and the most its backward holds.

    Args:
        config: The language model's Transformers config.
        seq_length: Tokens of the micro-batch of one sequence it runs.

    Returns:
        As :func:`census_layer`: the final norm's input included, the
        output table and its gradient left out.
    """
    modeling = _modeling(config)
    layer_cls, _ = _classes(modeling)
    loss_function = importlib.import_module("transformers.loss.loss_utils").ForCausalLMLoss
    with FakeTensorMode(allow_non_fake_inputs=True):
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            # The final norm is of the class of a layer's input norm.
            norm = layer_cls(config, 0).input_layernorm
            head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        finally:
            torch.set_default_dtype(default)
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(0, config.vocab_size, (1, seq_length))
        # The loss alone holds the logits: the trainer drops the model's
        # output before the backward.
        return _measure([*norm.parameters(), *head.parameters()], (hidden,),
                        lambda: loss_function(head(norm(hidden)), labels, config.vocab_size),
                        lambda loss: loss.backward())


def census_output_activations(config: Any, seq_length: int = 4096) -> KindActivations:
    """What the output layer of *config* keeps and holds, per token, with its vocabulary and half of it.

    Args:
        config: The language model's Transformers config.
        seq_length: The tokens the census runs the layer at.

    Returns:
        The layer's :class:`KindActivations`, whose TP part is the part a
        vocabulary-parallel loss over two ranks halves.
    """
    half = copy.deepcopy(config)
    half.vocab_size = max(1, config.vocab_size // 2)
    (saved_1, working_1), (saved_2, working_2) = (census_output(each, seq_length) for each in (config, half))
    return KindActivations(
        saved=max(0.0, 2 * saved_2 - saved_1) / seq_length,
        saved_tp=max(0.0, 2 * (saved_1 - saved_2)) / seq_length,
        working=max(0.0, 2 * working_2 - working_1) / seq_length,
        working_tp=max(0.0, 2 * (working_1 - working_2)) / seq_length,
        seq_length=int(seq_length),
    )


def census_activations(config: Any, layers: Iterable[Mapping[str, Any]],
                       seq_length: int = 4096) -> Dict[str, KindActivations]:
    """What a layer of each kind of *layers* keeps and holds, per token, at TP 1 and TP 2.

    Args:
        config: The language model's Transformers config.
        layers: The model spec's layer stack, groups in model order; MTP
            groups are left out.
        seq_length: The tokens the census runs a layer at.

    Returns:
        Each kind's :class:`KindActivations`: of the bytes a layer at TP 2
        holds, the part TP splits is half the part at TP 1.
    """
    firsts: Dict[str, int] = {}
    index = 0
    for group in layers:
        if group.get("mtp"):
            continue
        firsts.setdefault(str(group["kind"]), index)
        index += int(group["count"])
    out: Dict[str, KindActivations] = {}
    for kind, layer_index in firsts.items():
        (saved_1, working_1), (saved_2, working_2) = (
            census_layer(tp_config(config, tp), layer_index, seq_length) for tp in (1, 2))
        out[kind] = KindActivations(
            saved=max(0.0, 2 * saved_2 - saved_1) / seq_length,
            saved_tp=max(0.0, 2 * (saved_1 - saved_2)) / seq_length,
            working=max(0.0, 2 * working_2 - working_1) / seq_length,
            working_tp=max(0.0, 2 * (working_1 - working_2)) / seq_length,
            seq_length=int(seq_length),
        )
    return out
