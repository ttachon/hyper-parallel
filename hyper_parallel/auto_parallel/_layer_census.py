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
does not.  Run again under HyperParallel's selective activation checkpointing,
its own checkpoint and policy, a layer keeps its input and the outputs the
policy saves: every other matmul's, the attention kernel's and the
convolution's; the matmuls it does not save its backward runs again, a share
of the attention's projections and of the rest of the layer.

The output layer's census runs the final norm, the output projection and
Transformers' causal-LM loss, which casts the logits to fp32, as
HyperParallel's trainer runs them by default, dropping the model's logits
before the backward as the trainer does; half the vocabulary tells the
bytes a vocabulary-parallel loss splits.

A layer's parameters are counted by part, as ND prices its parts: the
attention's, the norms', the dense feed-forward's, the routed experts', the
shared expert's and the router's (:func:`census_parameters`), for verify
mode to set beside what ND prices.  So are its forward FLOPs
(:func:`census_flops`): each matmul's toward the part of the layer that
runs it, flash attention's toward the scores, at every pair of tokens, as
the runtime's kernel computes them under an explicit causal mask, and the
gated delta rule's through Transformers' own chunked implementation, whose
matmuls the runtime's kernel runs alike.
"""
from __future__ import annotations

import contextlib
import copy
import functools
import importlib
import inspect
import weakref
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch  # pylint: disable=forbidden-backend-import
from torch._subclasses.fake_tensor import FakeTensorMode  # pylint: disable=forbidden-backend-import
from torch.utils._python_dispatch import TorchDispatchMode  # pylint: disable=forbidden-backend-import
from torch.utils._pytree import tree_flatten  # pylint: disable=forbidden-backend-import

from hyper_parallel.auto_parallel._model_spec import KindActivations
from hyper_parallel.core.activation_memory import api as activation_memory
from hyper_parallel.core.activation_memory.policy import CheckpointPolicy

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

# The per-head query and key norms an attention holds, which ND prices with
# the layer's norms.
_QK_NORMS = ("q_norm", "k_norm")


def _flash_outputs(query: torch.Tensor, key: torch.Tensor,
                   value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flash attention's outputs, shapes only: one of the queries' shape and two fp32 statistics."""
    del key, value
    batch, heads, seq, _ = query.shape
    stats = [torch.empty(batch, heads, seq, _FLASH_STATS, dtype=torch.float32, device=query.device)
             for _ in range(2)]
    return torch.empty_like(query), stats[0], stats[1]


# Flash attention as one op, as the runtime's kernel is: HyperParallel's
# selective policy saves what the attention kernels it lists return, and
# saves this one's too.
_FLASH_KERNEL = torch.library.custom_op(
    "nd_census::flash_attention", mutates_args=(),
    schema="(Tensor query, Tensor key, Tensor value) -> (Tensor, Tensor, Tensor)")(_flash_outputs)
_FLASH_KERNEL.register_fake(_flash_outputs)


def _flash_saves(ctx: Any, inputs: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
    """Keep the inputs, the output and the statistics, which no gradient reaches."""
    ctx.mark_non_differentiable(*output[1:])
    ctx.save_for_backward(*inputs, *output)


def _flash_backward(ctx: Any, *grads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients of the inputs' shapes."""
    del grads
    query, key, value = ctx.saved_tensors[:3]
    return torch.empty_like(query), torch.empty_like(key), torch.empty_like(value)


_FLASH_KERNEL.register_autograd(_flash_backward, setup_context=_flash_saves)


def _flash_attention(module, query, key, value, attention_mask, **kwargs):  # pylint: disable=unused-argument
    """A Transformers attention function running the census's flash attention, grouped K and V unexpanded."""
    return _FLASH_KERNEL(query, key, value)[0].transpose(1, 2).contiguous(), None


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
             forward: Callable[[], torch.Tensor], backward: Callable[[torch.Tensor], None],
             grad_inputs: Iterable[torch.Tensor] = (), checkpointed: bool = False,
             shared: Iterable[torch.Tensor] = ()) -> Tuple[int, int]:
    """Bytes a forward saves for its backward, and the activations the backward holds when it holds the most.

    Args:
        params: The parameters of what runs.
        inputs: The tensors the forward takes, which count from its start.
        forward: Runs the forward and returns its output.
        backward: Runs the backward from that output.
        grad_inputs: The tensors the backward takes, which count from its
            start.
        checkpointed: Whether the forward runs under activation
            checkpointing, whose own hooks hold what it keeps.
        shared: Tensors the model shares between its layers, which a
            checkpointed forward's count leaves out.

    Returns:
        The bytes the forward saves, its inputs included, and the bytes of
        activations live when the backward peaks, its output left out.  A
        checkpointed forward keeps what is live once it has run, but for
        the *shared* tensors: its inputs and the outputs its policy saves.
        The parameters, which the forward's views of them bring into the
        tracker, are the memory model's to count, and so are their
        gradients, which hooks on the parameters tell apart from the
        activations' however they are shaped: the gradients the backward
        hands each parameter and the one it keeps.  They settle when the
        backward peaks, and the bytes returned are the activations held
        then.
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
            left_out = {live.key(out), *(live.key(p) for p in params)}
            not_kept = left_out | {live.key(tensor) for tensor in shared}
            kept = sum(size for key, size in live.sizes.items() if key not in not_kept)
            for tensor in grad_inputs:
                live.track(tensor)
            backward(out)
    finally:
        for handle in handles:
            handle.remove()
    grads += [live.key(p.grad) for p in params if p.grad is not None]
    return kept if checkpointed else sum(saved.values()), live.peak(start, left_out, grads)


def _matmul_flops(func: Any, args: Sequence[Any]) -> int:
    """The FLOPs of *func* on *args* where it is a matmul, and 0 otherwise."""
    if func not in (torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default,
                    torch.ops.aten._grouped_mm.default):  # pylint: disable=protected-access
        return 0
    first, second = args[1:3] if func is torch.ops.aten.addmm.default else args[:2]
    return 2 * first.numel() * second.shape[-1]


class _RecomputedMatmuls:
    """The matmul FLOPs a layer's selective forward saves and leaves to run again, its attention's and the rest's."""

    def __init__(self, layer: Any) -> None:
        """Tell the attention's parameters, those of a child named for it, from the rest of *layer*'s."""
        self.attention = {param.untyped_storage()._cdata  # pylint: disable=protected-access
                          for name, param in layer.named_parameters()
                          if "attn" in name.split(".")[0] or "attention" in name.split(".")[0]}
        self.flops: Dict[Tuple[bool, bool], int] = {}

    def note(self, func: Any, args: Sequence[Any], policy: CheckpointPolicy) -> None:
        """Count the FLOPs of a matmul the policy decided on, by part and by whether it runs again."""
        flops = _matmul_flops(func, args)
        if flops:
            attention = any(isinstance(arg, torch.Tensor)
                            and arg.untyped_storage()._cdata in self.attention  # pylint: disable=protected-access
                            for arg in args)
            key = (attention, policy == CheckpointPolicy.MUST_RECOMPUTE)
            self.flops[key] = self.flops.get(key, 0) + flops

    def share(self, attention: bool) -> float:
        """The share of the attention's matmul FLOPs, or of the rest's, that runs again."""
        again = self.flops.get((attention, True), 0)
        total = again + self.flops.get((attention, False), 0)
        return again / total if total else 0.0


def _selective_contexts(ledger: Optional[_RecomputedMatmuls] = None) -> Tuple[Any, Any]:
    """HyperParallel's selective checkpointing contexts, its policy saving the census's flash attention.

    Its forward decisions on matmuls go to *ledger*, if given.
    """
    # HyperParallel's distributed package, which holds the policy, takes as
    # long to import as the rest of the cost model: it loads only when a
    # census runs.
    checkpointing = importlib.import_module("hyper_parallel.distributed.activation_checkpoint")
    policy = checkpointing._make_selective_checkpoint_policy_fn()  # pylint: disable=protected-access

    def census_policy(ctx: Any, func: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
        """The trainer's policy, which saves the census's flash attention as it saves the runtime's."""
        if func is torch.ops.nd_census.flash_attention.default:
            decision = CheckpointPolicy.MUST_SAVE
        else:
            decision = policy(ctx, func, *args, **kwargs)
        if ledger is not None and not ctx.is_recompute:
            ledger.note(func, args, decision)
        return decision

    return activation_memory.create_selective_checkpoint_contexts(census_policy)


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


def _positions(rotary: Any, hidden: torch.Tensor, seq_length: int) -> Tuple[torch.Tensor, Any]:
    """The position ids of a sequence of *seq_length* tokens, and the embeddings *rotary* gives them."""
    position_ids = torch.arange(seq_length).unsqueeze(0)
    with torch.no_grad():
        return position_ids, rotary(hidden, position_ids)


def _run(layer: Any, hidden: torch.Tensor, positions: Tuple[torch.Tensor, Any],
         call: Optional[Callable[..., Any]] = None) -> torch.Tensor:
    """One forward of *layer*, as its model calls it at *positions*, through *call* if given."""
    position_ids, position_embeddings = positions
    accepted = inspect.signature(layer.forward).parameters
    kwargs = {name: value for name, value in (
        ("position_embeddings", position_embeddings), ("position_ids", position_ids),
        ("attention_mask", None), ("use_cache", False)) if name in accepted}
    out = (call or layer)(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@contextlib.contextmanager
def _fake_layer(config: Any, layer_index: int, contracts: bool = True) -> Iterator[Tuple[Any, Any]]:
    """Layer *layer_index* of *config* and its model's rotary embedding, on fake tensors and the runtime's kernels.

    The gated delta rule runs HyperParallel's kernel's contract, or, where
    *contracts* is false, Transformers' own chunked implementation.
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
                module.chunk_gated_delta_rule = (_gated_delta_rule(modeling) if contracts
                                                 else modeling.torch_chunk_gated_delta_rule)
        layer.train()
        yield layer, rotary


class _PartFlops(TorchDispatchMode):
    """A layer's forward FLOPs by part: its matmuls' toward the part running them, flash attention's the scores'."""

    def __init__(self, layer: Any) -> None:
        """Follow which of *layer*'s modules runs, and which tensors are its parameters."""
        super().__init__()
        self.flops: Dict[str, int] = {}
        self.params = {param.untyped_storage()._cdata  # pylint: disable=protected-access
                       for param in layer.parameters()}
        self.running: List[str] = []
        for name, module in layer.named_modules():
            if name:
                module.register_forward_pre_hook(functools.partial(self._enter, name))
                module.register_forward_hook(self._leave)

    def _enter(self, name: str, *_: Any) -> None:
        """Note that the module at *name* runs."""
        self.running.append(name)

    def _leave(self, *_: Any) -> None:
        """Note that the innermost module running has returned."""
        self.running.pop()

    def _part(self, args: Sequence[Any]) -> str:
        """The part a matmul on *args* runs for: its module's, or an attention's recurrence where no weight takes part."""
        part = _parameter_part(f"{self.running[-1]}.weight") if self.running else "ffn"
        weighted = any(isinstance(arg, torch.Tensor)
                       and arg.untyped_storage()._cdata in self.params  # pylint: disable=protected-access
                       for arg in args)
        return "linrec" if part == "attention" and not weighted else part

    def _add(self, part: str, flops: int) -> None:
        """Count *flops* toward *part*."""
        self.flops[part] = self.flops.get(part, 0) + flops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        out = func(*args, **(kwargs or {}))
        if func is torch.ops.nd_census.flash_attention.default:
            query, key, value = args[:3]
            batch, heads, seq, width = query.shape
            self._add("scores", 2 * batch * heads * seq * key.shape[2] * (width + value.shape[-1]))
        elif flops := _matmul_flops(func, args):
            self._add(self._part(args), flops)
        return out


def census_flops(config: Any, layer_index: int, seq_length: int) -> Dict[str, int]:
    """The forward FLOPs of layer *layer_index* of *config* on one sequence, by part.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the sequence it runs.

    Returns:
        The FLOPs of each part that runs a matmul: the attention's
        projections (``attention``), its scores and values at every pair of
        tokens (``scores``), a linear attention's recurrence (``linrec``),
        and the parts of the feed-forward :func:`_parameter_part` names.
    """
    with _fake_layer(config, layer_index, contracts=False) as (layer, rotary):
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16)
        positions = _positions(rotary, hidden, seq_length)
        counter = _PartFlops(layer)
        with counter:
            _run(layer, hidden, positions)
    return counter.flops


def census_layer(config: Any, layer_index: int, seq_length: int, selective: bool = False) -> Tuple[int, int]:
    """Bytes layer *layer_index* of *config* keeps for its backward, and the most its backward holds.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the micro-batch of one sequence it runs.
        selective: Whether the layer runs under HyperParallel's selective
            activation checkpointing, as its trainer wraps a layer.

    Returns:
        The bytes the forward keeps for the backward, its input included,
        and the bytes of activations the backward holds when it holds the
        most, its own gradients included, less the parameters and those
        gradients.
    """
    with _fake_layer(config, layer_index) as (layer, rotary):
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16)
        if not selective:
            return _measure(list(layer.parameters()), (hidden,),
                            lambda: _run(layer, hidden, _positions(rotary, hidden, seq_length)),
                            lambda out: out.backward(grad), grad_inputs=(grad,))
        # The model embeds the positions once for all its layers, outside
        # their checkpoints, and the trainer checkpoints each layer's call.
        positions = _positions(rotary, hidden, seq_length)
        call = functools.partial(activation_memory.checkpoint, layer, swap_inputs=False,
                                 context_fn=_selective_contexts)
        return _measure(list(layer.parameters()), (hidden,), lambda: _run(layer, hidden, positions, call),
                        lambda out: out.backward(grad), grad_inputs=(grad,), checkpointed=True,
                        shared=tree_flatten(positions)[0])


def census_recomputed(config: Any, layer_index: int, seq_length: int) -> Tuple[float, float]:
    """The shares of layer *layer_index*'s matmul FLOPs its backward runs again under selective checkpointing.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the micro-batch of one sequence it runs.

    Returns:
        The share of its attention's projections' FLOPs, and of the rest of
        the layer's, that HyperParallel's selective policy recomputes: it
        saves every other matmul's output, in the order the layer runs
        them, and a biased projection's always.
    """
    with _fake_layer(config, layer_index) as (layer, rotary):
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
        ledger = _RecomputedMatmuls(layer)
        call = functools.partial(activation_memory.checkpoint, layer, swap_inputs=False,
                                 context_fn=functools.partial(_selective_contexts, ledger))
        _run(layer, hidden, _positions(rotary, hidden, seq_length), call)
    return ledger.share(attention=True), ledger.share(attention=False)


def _parameter_part(name: str) -> str:
    """The part of a decoder layer the parameter at path *name* belongs to, as ND prices the parts.

    The layer's norms and its attention's per-head query and key norms are
    its norms; the rest of a child named for attention is its attention.
    Of the feed-forward, a shared expert's parameters, the routed experts'
    and the router's are parts of their own, and the rest is the dense
    feed-forward's.
    """
    segments = name.split(".")
    module = segments[-2] if len(segments) > 1 else ""
    if "norm" in segments[0] or module in _QK_NORMS:
        return "norm"
    if "attn" in segments[0] or "attention" in segments[0]:
        return "attention"
    if any("shared" in segment for segment in segments):
        return "shared"
    if "experts" in segments:
        return "routed"
    if module in ("gate", "router"):
        return "router"
    return "ffn"


def census_parameters(config: Any, layer_index: int) -> Dict[str, int]:
    """The parameters of layer *layer_index* of *config*, by part.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.

    Returns:
        The parameter count of each part the layer has
        (:func:`_parameter_part`).
    """
    parts: Dict[str, int] = {}
    with _fake_layer(config, layer_index) as (layer, _):
        for name, param in layer.named_parameters():
            part = _parameter_part(name)
            parts[part] = parts.get(part, 0) + param.numel()
    return parts


def census_final_norm(config: Any) -> int:
    """The parameters of *config*'s final norm, of the class of a layer's input norm."""
    with _fake_layer(config, 0) as (layer, _):
        return sum(param.numel() for param in layer.input_layernorm.parameters())


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
        holds, the part TP splits is half the part at TP 1; what it keeps
        under HyperParallel's selective activation checkpointing too, and
        the shares of its matmul FLOPs that recomputes.
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
        (kept_1, _), (kept_2, _) = (
            census_layer(tp_config(config, tp), layer_index, seq_length, selective=True) for tp in (1, 2))
        attention_mm, ffn_mm = census_recomputed(config, layer_index, seq_length)
        out[kind] = KindActivations(
            saved=max(0.0, 2 * saved_2 - saved_1) / seq_length,
            saved_tp=max(0.0, 2 * (saved_1 - saved_2)) / seq_length,
            working=max(0.0, 2 * working_2 - working_1) / seq_length,
            working_tp=max(0.0, 2 * (working_1 - working_2)) / seq_length,
            seq_length=int(seq_length),
            selective=max(0.0, 2 * kept_2 - kept_1) / seq_length,
            selective_tp=max(0.0, 2 * (kept_1 - kept_2)) / seq_length,
            selective_attention_mm=attention_mm,
            selective_ffn_mm=ffn_mm,
        )
    return out
