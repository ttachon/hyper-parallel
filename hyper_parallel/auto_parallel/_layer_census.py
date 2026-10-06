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

What a layer saves is also told apart by the op of ND's op vector whose
backward takes it (:class:`_SavedOps`), the records of shared decision S1 as
the census fills them: the delta rule's, the attention kernel's (its
inputs and output for the batched matmuls, its fp32 statistics for the
softmax), a dropout's mask, what a norm saves, what the feed-forward's
activation function saves, and the rest by the part of the layer saving it,
the attention's projections or the feed-forward's.

The output layer's census runs the final norm, the output projection and
Transformers' causal-LM loss, which casts the logits to fp32, as
HyperParallel's trainer runs them by default, dropping the model's logits
before the backward as the trainer does; half the vocabulary tells the
bytes a vocabulary-parallel loss splits.  A model spec states a census as
:class:`KindActivations` records.

Where a train.yaml replaces Transformers' modules with HyperParallel's
fused ones (``plan_overrides`` with ``replace_module``: its RMSNorm, its
grouped-query attention, its grouped experts), the census runs those
instead, so what it measures is what the trainer saves
(:func:`replacement_specs`).  Their kernels are ``torch_npu``'s, which a
host cannot call, so they run under the kernels' shape contracts
(:mod:`hyper_parallel.auto_parallel._npu_contracts`).  A factory that needs
a library the host lacks leaves the census with a layer the run does not
build, which it refuses (:exc:`CensusUnavailable`): Transformers' own
modules keep other bytes, and under the selective policy a DeepSeek-V3.2
layer's eager indexer keeps 2.3 times what the layer keeps plain (F53).  The FLOP census
(:func:`census_flops`) and the shares of matmul FLOPs a selective layer
recomputes (:func:`census_recomputed`) stay on Transformers' own modules:
both count the arithmetic the time model prices, which fusing a module
does not change, and a fused kernel runs matmuls no dispatch sees.

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
import logging
import weakref
from dataclasses import dataclass, fields
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

import torch  # pylint: disable=forbidden-backend-import
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode  # pylint: disable=forbidden-backend-import
from torch.utils._python_dispatch import TorchDispatchMode  # pylint: disable=forbidden-backend-import
from torch.utils._pytree import tree_flatten  # pylint: disable=forbidden-backend-import

from hyper_parallel.auto_parallel._npu_contracts import npu_contracts
from hyper_parallel.auto_parallel._op_records import OPS
from hyper_parallel.core.activation_memory import api as activation_memory
from hyper_parallel.core.activation_memory.policy import CheckpointPolicy

logger = logging.getLogger(__name__)


class CensusUnavailable(RuntimeError):
    """A census cannot run here: this host cannot build the modules the run installs."""


class LayerTraffic(NamedTuple):
    """What one forward of a layer moves, by part: its bytes, the parameters among them, and its ops.

    Bytes of the whole layer on a micro-batch of one sequence, as
    :func:`census_flops` counts its FLOPs, so that the two quantities can be
    set beside each other part by part.  ``parameter`` is the share of
    ``moved`` that is the layer's own weights, which a strategy shards where
    it leaves the activations alone, and ``launches`` is how many ops the
    forward dispatches, views excluded.
    """

    moved: Dict[str, float]
    parameter: Dict[str, float]
    launches: Dict[str, int]


# The fields a census record states in pairs: what a layer keeps under
# HyperParallel's selective activation checkpointing, the shares of its
# matmul FLOPs that recomputes, and what it keeps for each op.
_PAIRED = (("selective", "selective_tp"), ("selective_attention_mm", "selective_ffn_mm"), ("ops", "ops_tp"))

# The census record's fields that map ops to bytes per token.
_BY_OP = ("ops", "ops_tp")


@dataclass(frozen=True)
class KindActivations:
    """What one layer of a kind keeps for its backward, and the most its backward holds, per token.

    Bytes per token at micro-batch 1, as the census measures them on one
    layer at ``seq_length`` tokens under the runtime's kernels: the part no
    tensor-parallel rank splits, and the part it splits, of which a layer at
    TP 2 holds half.  The backward's working set leaves out the parameters
    and their gradients, which the memory model counts on its own.
    ``selective`` and ``selective_tp`` are what the layer keeps under
    HyperParallel's selective activation checkpointing, whose backward
    recomputes the rest and holds the same working set, and
    ``selective_attention_mm`` and ``selective_ffn_mm`` the shares of the
    FLOPs of its attention's projections and of the rest of its matmuls
    that recomputes.  ``ops`` and ``ops_tp`` are what it keeps for each op
    of :data:`~hyper_parallel.auto_parallel._op_records.OPS`, the two parts
    of ``saved`` and ``saved_tp``, and ``other`` for its own code: the op
    records of shared decision S1, as the census fills them.  A record
    states each pair whole or not at all.
    """

    saved: float
    saved_tp: float
    working: float
    working_tp: float
    seq_length: int
    selective: Optional[float] = None
    selective_tp: Optional[float] = None
    selective_attention_mm: Optional[float] = None
    selective_ffn_mm: Optional[float] = None
    ops: Optional[Mapping[str, float]] = None
    ops_tp: Optional[Mapping[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the record as a plain mapping, the selective part and the ops only when stated."""
        out = {record_field.name: getattr(self, record_field.name) for record_field in fields(self)
               if getattr(self, record_field.name) is not None}
        for name in _BY_OP:
            if name in out:
                out[name] = dict(out[name])
        return out

    @staticmethod
    def _check_keys(data: Mapping[str, Any], names: Sequence[str], where: str) -> None:
        """Raise unless *data* states the fields of *names*, each of :data:`_PAIRED` whole or not at all."""
        unknown = sorted(set(data) - set(names))
        missing = [name for name in names if name not in data and all(name not in pair for pair in _PAIRED)]
        if unknown or missing:
            raise ValueError(f"{where} has unknown keys {unknown} and lacks {missing}")
        for pair in _PAIRED:
            if len({data.get(name) is None for name in pair}) > 1:
                raise ValueError(f"{where} states one of {list(pair)} without the other")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str = "activations") -> "KindActivations":
        """Build a record, refusing a key it does not know, a missing one, a negative size or a share above 1."""
        if not isinstance(data, Mapping):
            raise ValueError(f"{where} must map the record's fields to their values, got {data!r}")
        names = [record_field.name for record_field in fields(cls)]
        cls._check_keys(data, names, where)
        sizes = {name: float(data[name]) for name in names
                 if name != "seq_length" and name not in _BY_OP and data.get(name) is not None}
        if any(size < 0 for size in sizes.values()):
            raise ValueError(f"{where}: bytes per token cannot be negative, got {sizes}")
        if any(sizes.get(name, 0) > 1 for name in _PAIRED[1]):
            raise ValueError(f"{where}: a share of FLOPs cannot exceed 1, got {sizes}")
        seq_length = int(data["seq_length"])
        if seq_length <= 0:
            raise ValueError(f"{where}.seq_length must be positive, got {seq_length}")
        by_op = {name: _op_bytes(data[name], f"{where}.{name}") for name in _BY_OP if data.get(name) is not None}
        return cls(seq_length=seq_length, **sizes, **by_op)


def _op_bytes(data: Any, where: str) -> Dict[str, float]:
    """Parse ``{op: bytes per token}``, the ops of :data:`OPS` and ``other``, refusing a negative size."""
    known = [*OPS, "other"]
    if not isinstance(data, Mapping) or any(op not in known for op in data):
        raise ValueError(f"{where} must map ops of {known} to bytes per token, got {data!r}")
    sizes = {str(op): float(size) for op, size in data.items()}
    if any(size < 0 for size in sizes.values()):
        raise ValueError(f"{where}: bytes per token cannot be negative, got {sizes}")
    return sizes


def activations_from_dict(data: Any) -> Dict[str, KindActivations]:
    """Parse a spec's ``activations``, a mapping of layer kind to its :class:`KindActivations`."""
    if not isinstance(data, Mapping):
        raise ValueError(f"activations must map layer kinds to their records, got {data!r}")
    return {str(kind): KindActivations.from_dict(record, f"activations.{kind}") for kind, record in data.items()}


# The Transformers attention implementation the census registers its flash
# attention under: a name Transformers does not take for a flash
# attention's, as it does not take HyperParallel's default, sdpa, so a model
# runs sdpa's path, and DeepSeek-V3's values are not padded to its queries'
# width.
_FLASH = "nd_census_attention"

# The config fields a tensor-parallel rank holds a share of.
_TP_FIELDS = (
    "num_attention_heads", "num_key_value_heads", "intermediate_size", "moe_intermediate_size",
    "shared_expert_intermediate_size", "linear_num_key_heads", "linear_num_value_heads",
)

# The softmax statistics flash attention keeps per head and token.
_FLASH_STATS = 8

# The gated delta rules a run can take, named as the runtime names them
# (``components/modules/gated_delta_net.py``): Transformers' own chunked
# implementation, and HyperParallel's kernel, whose saved set the census holds
# as a contract.  The runtime defaults to eager and so does the census.
_EAGER_GDN = "eager"
_HYPER_GDN = "triton"
_GDN_BACKENDS = (_EAGER_GDN, _HYPER_GDN)

# The census's attention kernels, which HyperParallel's selective policy
# saves as it saves the runtime's: its own flash attention, and the contract
# of the kernel its fused attention calls.
_ATTENTION_KERNELS = ("nd_census::flash_attention", "nd_census_npu::fusion_attention")

# The per-head query and key norms an attention holds, which ND prices with
# the layer's norms.
_QK_NORMS = ("q_norm", "k_norm")


def _flash_outputs(query: torch.Tensor, key: torch.Tensor,
                   value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flash attention's outputs, shapes only: each head's, as wide as its values, and two fp32 statistics."""
    del key
    batch, heads, seq, _ = query.shape
    stats = [torch.empty(batch, heads, seq, _FLASH_STATS, dtype=torch.float32, device=query.device)
             for _ in range(2)]
    return query.new_empty(batch, heads, seq, value.shape[-1]), stats[0], stats[1]


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


# Reading a scalar off a fake tensor raises, since it holds no data. Transformers'
# grouped_mm fallback reads its group offsets so in its backward, and it takes that
# fallback on CPU with torch 2.8 or older, the 910C's 2.7.1 among them.
_SCALAR_READ = torch.ops.aten._local_scalar_dense.default  # pylint: disable=protected-access


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
        if func is _SCALAR_READ and isinstance(args[0], FakeTensor):
            return 0  # shapes alone price the memory, whatever the scalar
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
             shared: Iterable[torch.Tensor] = (), ops: Optional["_SavedOps"] = None) -> Tuple[int, int]:
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
        ops: Where given, tells each storage the forward saves apart by
            the op that first saves it, counting its bytes into
            ``ops.saved``, which then sums to the bytes returned.

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
    # Counted by the storage in the lifetime it has now, as the tracker counts
    # them: a fake tensor's storage is cheap and short lived, so one the
    # forward has freed can hand its address to a later save, and counting by
    # address alone lost whichever of the two it saw first.  Which pair
    # collided depended on what had run before, so a layer did not measure the
    # same twice.
    saved: Dict[Tuple[int, int], int] = {}
    saved_op: Dict[Tuple[int, int], str] = {}

    def pack(tensor: torch.Tensor) -> torch.Tensor:
        """Count a tensor autograd saves, unless it is a parameter's, toward the op that first saves it."""
        # Named before the storage is read, which dispatches ops of its own.
        op = ops.op(tensor) if ops is not None else None
        address = tensor.untyped_storage()._cdata  # pylint: disable=protected-access
        if address not in stored:
            key = live.key(tensor)
            if op is not None:
                saved_op.setdefault(key, op)
            saved[key] = tensor.untyped_storage().nbytes()
        return tensor

    live = _LiveBytes()
    grads: List[Tuple[int, int]] = []
    handles = [p.register_hook(lambda grad: grads.append(live.key(grad))) for p in params]
    try:
        with live, ops or contextlib.nullcontext(), \
                torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
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
    # A storage's bytes count toward the op that first saved it, read from the
    # same record as the total, so the two agree whatever a fake tensor does
    # with a storage address it has freed.
    for address, size in saved.items():
        op = saved_op.get(address)
        if op is not None:
            ops.saved[op] = ops.saved.get(op, 0) + size
    grads += [live.key(p.grad) for p in params if p.grad is not None]
    return kept if checkpointed else sum(saved.values()), live.peak(start, left_out, grads)


# The ops a matmul runs as.
_MATMULS = ("mm", "addmm", "bmm", "_grouped_mm", "baddbmm")


class _SavedOps(TorchDispatchMode):
    """What each tensor a layer's forward saves is saved for: the op of ND's op vector whose backward takes it.

    The delta rule's saves are its own (``linrec``); the attention kernel's
    are its batched matmuls' (``attBMM``), its fp32 statistics the
    softmax's; a dropout's mask is its own; what a norm saves, the norm's
    (``normOp``), and what the feed-forward's activation function saves,
    the function's (``ffAct``).  The rest counts toward the part of the
    layer saving it: the attention's projections (``attMM``), the
    feed-forward's (``ffMM``), or ``other`` for the layer's own code.
    """

    def __init__(self, layer: Any) -> None:
        """Follow which of *layer*'s modules runs, and when its delta rule does."""
        super().__init__()
        self.saved: Dict[str, int] = {}
        self.running: List[str] = []
        self.func: Any = None
        self.delta_rule = False
        self.modules = dict(layer.named_modules())
        for name, module in self.modules.items():
            if name:
                module.register_forward_pre_hook(functools.partial(self._enter, name))
                module.register_forward_hook(self._leave)
            if hasattr(module, "chunk_gated_delta_rule"):
                module.chunk_gated_delta_rule = self._rule(module.chunk_gated_delta_rule)

    def _rule(self, rule: Callable[..., Any]) -> Callable[..., Any]:
        """*rule*, noting that it runs."""
        def run(*args: Any, **kwargs: Any) -> Any:
            self.delta_rule = True
            try:
                return rule(*args, **kwargs)
            finally:
                self.delta_rule = False
        return run

    def _enter(self, name: str, *_: Any) -> None:
        """Note that the module at *name* runs."""
        self.running.append(name)

    def _leave(self, *_: Any) -> None:
        """Note that the innermost module running has returned."""
        self.running.pop()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        # A tensor's metadata is read through prims, between an op and its saves.
        if func.namespace != "prim":
            self.func = func
        return func(*args, **(kwargs or {}))

    def op(self, tensor: torch.Tensor) -> str:
        """The op that saves *tensor*, which the forward saves now."""
        if self.delta_rule:
            return "linrec"
        if self.func in _attention_ops():
            return "softmax" if tensor.dtype == torch.float32 else "attBMM"
        if self.func is torch.ops.aten.native_dropout.default:
            return "dropout"
        if self.func is torch.ops.nd_census_npu.rms_norm.default:
            return "normOp"
        if self.func is torch.ops.nd_census_npu.swiglu.default:
            return "ffAct"
        name = self.running[-1] if self.running else ""
        module = self.modules.get(name)
        if name and "norm" in type(module).__name__.lower():
            return "normOp"
        part = name.split(".", maxsplit=1)[0]
        if not part:
            return "other"
        attention = "attn" in part or "attention" in part
        if not attention and type(module).__module__ in ("transformers.activations", "torch.nn.modules.activation"):
            return "ffAct"
        return "attMM" if attention else "ffMM"


def _matmul_flops(func: Any, args: Sequence[Any]) -> int:
    """The FLOPs of *func* on *args* where it is a matmul, and 0 otherwise."""
    matmuls = (torch.ops.aten.mm.default, torch.ops.aten.addmm.default, torch.ops.aten.bmm.default)
    grouped = getattr(torch.ops.aten, "_grouped_mm", None)  # first shipped in torch 2.8
    if func not in (matmuls if grouped is None else (*matmuls, grouped.default)):
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
    kernels = _attention_ops()

    def census_policy(ctx: Any, func: Any, *args: Any, **kwargs: Any) -> CheckpointPolicy:
        """The trainer's policy, which saves the census's attention kernels as it saves the runtime's."""
        if func in kernels:
            decision = CheckpointPolicy.MUST_SAVE
        else:
            decision = policy(ctx, func, *args, **kwargs)
        if ledger is not None and not ctx.is_recompute:
            ledger.note(func, args, decision)
        return decision

    return activation_memory.create_selective_checkpoint_contexts(census_policy)


@functools.lru_cache(maxsize=None)
def _attention_ops() -> FrozenSet[Any]:
    """The census's attention kernels, as ops."""
    ops = []
    for name in _ATTENTION_KERNELS:
        namespace, kernel = name.split("::")
        ops.append(getattr(getattr(torch.ops, namespace), kernel).default)
    return frozenset(ops)


def _import(path: str, where: str) -> Any:
    """The object a dotted *path* names.

    Raises:
        ValueError: If the path names no module or no attribute of one.
    """
    module, _, name = str(path).rpartition(".")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError, ValueError) as exc:
        raise ValueError(f"{where}: cannot import {path!r}: {exc}") from exc


def replacement_specs(plan_overrides: Any) -> Tuple[Any, ...]:
    """The module replacements a train.yaml's ``plan_overrides`` install, as replacement rules.

    Args:
        plan_overrides: The ``plan_overrides`` list, as plain mappings.

    Returns:
        One :class:`~hyper_parallel.models.replacement.ModuleReplacementSpec`
        per entry that states ``replace_module``, in the order stated; the
        entries that state a sharding action alone, or a replacement
        conditioned on a run the census does not price (``when``), are left
        out.

    Raises:
        ValueError: If an entry states ``replace_module`` without
            ``module_type``, or names a module or factory that cannot be
            imported.
    """
    from hyper_parallel.models.replacement import ModuleReplacementSpec  # pylint: disable=C0415

    specs = []
    for index, entry in enumerate(plan_overrides or ()):
        where = f"plan_overrides[{index}]"
        if not isinstance(entry, Mapping) or entry.get("replace_module") is None or entry.get("when") is not None:
            continue
        if entry.get("module_type") is None:
            raise ValueError(f"{where} states replace_module without module_type")
        match = entry.get("match")
        patterns = (match,) if isinstance(match, str) else tuple(match or ())
        factory = entry["replace_module"]
        specs.append(ModuleReplacementSpec(
            match=patterns,
            factory=_import(factory.get("_target_") if isinstance(factory, Mapping) else factory, where),
            module_type=_import(entry["module_type"], where),
            exact_type=bool(entry.get("exact_type", False)),
        ))
    return tuple(specs)


class _Holder(torch.nn.Module):
    """A module tree holding one part where a model holds it, so a rule's patterns match its path."""

    def __init__(self, **parts: Any) -> None:
        """Hold each part of *parts* under ``model``, a layer in a list as a model's stack holds it."""
        super().__init__()
        self.model = torch.nn.Module()
        for name, part in parts.items():
            setattr(self.model, name, torch.nn.ModuleList([part]) if name == "layers" else part)


def _matching(spec: Any, names: Sequence[str]) -> Any:
    """*spec* with the patterns that match a module of *names*, or ``None`` where none does."""
    import fnmatch  # pylint: disable=C0415

    import dataclasses  # pylint: disable=C0415

    kept = tuple(pattern for pattern in spec.match
                 if any(fnmatch.fnmatchcase(name, pattern) for name in names))
    return None if not kept else dataclasses.replace(spec, match=kept)


def _replaced(holder: _Holder, specs: Sequence[Any]) -> None:
    """Install on *holder* the replacements of *specs* whose patterns match one of its modules.

    A rule's other patterns name modules no fake part holds, such as a
    model's final norm beside a layer, and are left out rather than refused.

    Raises:
        CensusUnavailable: If a factory needs a library this host lacks, such
            as an attention built on a native extension.  Transformers' own
            modules are not what the run trains, and measuring them prices a
            layer nobody runs: a DeepSeek-V3.2 layer whose indexer runs eager
            keeps, under the selective policy, 1192192 bytes a token against
            the 516368 it keeps plain, its scores an fp32 tensor of one
            sequence by another that the policy holds and no backward takes
            (F53).
    """
    from hyper_parallel.models.replacement import (  # pylint: disable=C0415
        apply_module_replacements,
        compile_module_replacements,
    )

    names = [name for name, _ in holder.named_modules()]
    kept = [found for found in (_matching(spec, names) for spec in specs) if found is not None]
    if not kept:
        return
    try:
        apply_module_replacements(holder, compile_module_replacements(holder, kept),
                                  weights_mapping=[], context={}, capture_checkpoint_metadata=False)
    except (ImportError, OSError) as exc:
        raise CensusUnavailable(
            f"this host cannot build the modules the run installs "
            f"({[spec.factory.__name__ for spec in kept]}: {exc}); "
            f"run the cost model where they build, or drop context.census and let the formulas price them"
        ) from exc


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
def _fake_layer(config: Any, layer_index: int, gdn_backend: str = _EAGER_GDN,
                replacements: Sequence[Any] = ()) -> Iterator[Tuple[Any, Any]]:
    """Layer *layer_index* of *config* and its model's rotary embedding, on fake tensors and the runtime's kernels.

    *gdn_backend* names the gated delta rule the layer runs, as the runtime
    names it: ``"eager"`` for Transformers' own chunked implementation,
    ``"triton"`` for HyperParallel's kernel's contract, which saves the rule's
    inputs, the cumulated gates, beta and one chunk matrix and nothing more.
    Eager is the default because it is the rule a run reaches: the runtime's
    dispatcher defaults to it (``components/modules/gated_delta_net.py``,
    ``chunk_gated_delta_rule``), the only caller that can select the kernel is
    the context-parallel wrapper, which defaults to eager as well, and no
    recipe states a backend.  The two are not interchangeable: a layer keeping
    the kernel's saved set keeps less than half as much.

    Where *replacements* are given, the modules they name are HyperParallel's
    fused ones, running under the kernels' contracts for as long as the layer
    does.

    Raises:
        ValueError: If *gdn_backend* names no backend the runtime has.
    """
    if gdn_backend not in _GDN_BACKENDS:
        raise ValueError(
            f"unsupported GDN backend {gdn_backend!r}; "
            f"expected one of {sorted(_GDN_BACKENDS)}."
        )
    modeling = _modeling(config)
    layer_cls, rotary_cls = _classes(modeling)
    config = copy.deepcopy(config)
    config._attn_implementation = _FLASH  # pylint: disable=protected-access
    config._experts_implementation = "grouped_mm"  # pylint: disable=protected-access
    modeling.ALL_ATTENTION_FUNCTIONS.register(_FLASH, _flash_attention)
    with npu_contracts() if replacements else contextlib.nullcontext(), \
            FakeTensorMode(allow_non_fake_inputs=True):
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            layer = layer_cls(config, layer_index)
            rotary = rotary_cls(config=config)
        finally:
            torch.set_default_dtype(default)
        for module in layer.modules():
            if hasattr(module, "chunk_gated_delta_rule"):
                module.chunk_gated_delta_rule = (
                    _gated_delta_rule(modeling) if gdn_backend == _HYPER_GDN
                    else modeling.torch_chunk_gated_delta_rule)
        if replacements:
            holder = _Holder(layers=layer)
            _replaced(holder, replacements)
            layer = holder.model.layers[0]
        layer.train()
        yield layer, rotary


class _ByPart(TorchDispatchMode):
    """A walk over a layer's forward that attributes each op to the part of the layer running it."""

    def __init__(self, layer: Any, recurrent: bool = True) -> None:
        """Follow which of *layer*'s modules runs, and which tensors are its parameters.

        *recurrent* is whether an attention of this layer has a recurrence to
        name: where it has none, an op of the attention that no weight takes
        part in is the attention's own, its mask or its rotary.  A walk that
        counts only matmuls leaves it at the default, because a weightless
        matmul under an attention is a recurrence or a score, and a score is
        counted on its own.
        """
        super().__init__()
        self.recurrent = recurrent
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
        """The part an op on *args* runs for: its module's, or an attention's recurrence where no weight takes part."""
        part = _parameter_part(f"{self.running[-1]}.weight") if self.running else "ffn"
        weighted = any(isinstance(arg, torch.Tensor)
                       and arg.untyped_storage()._cdata in self.params  # pylint: disable=protected-access
                       for arg in args)
        if part == "attention" and not weighted and self.recurrent:
            return "linrec"
        return part


class _PartFlops(_ByPart):
    """A layer's forward FLOPs by part: its matmuls' toward the part running them, flash attention's the scores'."""

    def __init__(self, layer: Any) -> None:
        """Count no FLOPs yet, and follow *layer*'s modules."""
        super().__init__(layer)
        self.flops: Dict[str, int] = {}

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


class _PartTraffic(_ByPart):
    """The bytes a layer's forward moves by part, the parameters among them, and the ops it dispatches.

    An op reads the tensors it is given and writes the tensors it returns.
    A view writes nothing and reads nothing: it renames bytes its producer
    already wrote, and no kernel runs for it, so it counts for neither.  A
    tensor given to an op twice is read twice, as the count is of the op's
    arguments rather than of the storages behind them.

    The parameters are counted again on their own, because a strategy
    divides their traffic where it leaves the activations' alone: the degree
    sharding a parameter divides the bytes read of it, and the routed
    experts' part is the one the expert-parallel degree divides.
    """

    def __init__(self, layer: Any, recurrent: bool = True) -> None:
        """Move nothing yet, and follow *layer*'s modules."""
        super().__init__(layer, recurrent=recurrent)
        self.moved: Dict[str, float] = {}
        self.parameter: Dict[str, float] = {}
        self.launches: Dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if func.namespace == "prim" or getattr(func, "is_view", False):
            return out
        part = ("scores" if func is torch.ops.nd_census.flash_attention.default
                else self._part(args))
        moved = parameter = 0
        for tensor in tree_flatten((args, kwargs))[0]:
            if not isinstance(tensor, torch.Tensor):
                continue
            size = tensor.numel() * tensor.element_size()
            moved += size
            if tensor.untyped_storage()._cdata in self.params:  # pylint: disable=protected-access
                parameter += size
        for tensor in tree_flatten(out)[0]:
            if isinstance(tensor, torch.Tensor):
                moved += tensor.numel() * tensor.element_size()
        self.moved[part] = self.moved.get(part, 0.0) + moved
        self.parameter[part] = self.parameter.get(part, 0.0) + parameter
        self.launches[part] = self.launches.get(part, 0) + 1
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
    with _fake_layer(config, layer_index) as (layer, rotary):
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16)
        positions = _positions(rotary, hidden, seq_length)
        counter = _PartFlops(layer)
        with counter:
            _run(layer, hidden, positions)
    return counter.flops


def census_traffic(config: Any, layer_index: int, seq_length: int,
                   gdn_backend: str = _EAGER_GDN) -> LayerTraffic:
    """What one forward of layer *layer_index* of *config* moves, by the part of the layer moving it.

    The quantity the time model counts is matmul FLOPs, and the cluster's
    kernel tables put those at 6.3 to 7.1% of a profiled step of this model.
    This counts the other quantity a step could be priced by, on the same
    forward and with no device, so that a measured round can set each
    beside the time it took.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the micro-batch of one sequence it runs.
        gdn_backend: The gated delta rule the run takes (:func:`_fake_layer`).
            It is the largest single choice here: Transformers' own chunked
            rule runs a matmul a chunk and the kernel's contract is one op.

    Returns:
        A :class:`LayerTraffic` of the whole layer at tensor and context
        parallelism 1, by the parts :func:`census_flops` prices
        (:class:`_PartTraffic`).
    """
    with _fake_layer(config, layer_index, gdn_backend=gdn_backend) as (layer, rotary):
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16)
        positions = _positions(rotary, hidden, seq_length)
        # The rule's own module is what a layer has a recurrence by, as
        # _fake_layer finds it to settle which rule the layer runs.
        recurrent = any(hasattr(module, "chunk_gated_delta_rule") for module in layer.modules())
        counter = _PartTraffic(layer, recurrent=recurrent)
        with counter:
            _run(layer, hidden, positions)
    return LayerTraffic(counter.moved, counter.parameter, counter.launches)


def census_layer(config: Any, layer_index: int, seq_length: int, selective: bool = False,
                 ops: Optional[Dict[str, int]] = None,
                 replacements: Sequence[Any] = (),
                 gdn_backend: str = _EAGER_GDN) -> Tuple[int, int]:
    """Bytes layer *layer_index* of *config* keeps for its backward, and the most its backward holds.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the micro-batch of one sequence it runs.
        selective: Whether the layer runs under HyperParallel's selective
            activation checkpointing, as its trainer wraps a layer.
        ops: Where given, and the layer runs without checkpointing, takes
            the bytes it saves by the op saving them (:class:`_SavedOps`).
        replacements: The module replacements a run installs
            (:func:`replacement_specs`).
        gdn_backend: The gated delta rule the run takes (:func:`_fake_layer`).

    Returns:
        The bytes the forward keeps for the backward, its input included,
        and the bytes of activations the backward holds when it holds the
        most, its own gradients included, less the parameters and those
        gradients.
    """
    with _fake_layer(config, layer_index, gdn_backend=gdn_backend,
                     replacements=replacements) as (layer, rotary):
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16)
        if replacements:
            # A fused kernel keeps a constant per device, such as the mask
            # the attention runs its causal sparse mode with, which the
            # runtime allocates once for every layer and micro-batch: one
            # forward allocates them before the layer's own bytes are
            # counted, so they count toward no layer.
            with torch.no_grad():
                _run(layer, hidden, _positions(rotary, hidden, seq_length))
        if not selective:
            saved_ops = None if ops is None else _SavedOps(layer)
            measured = _measure(list(layer.parameters()), (hidden,),
                                lambda: _run(layer, hidden, _positions(rotary, hidden, seq_length)),
                                lambda out: out.backward(grad), grad_inputs=(grad,), ops=saved_ops)
            if saved_ops is not None:
                ops.update(saved_ops.saved)
            return measured
        # The model embeds the positions once for all its layers, outside
        # their checkpoints, and the trainer checkpoints each layer's call.
        positions = _positions(rotary, hidden, seq_length)
        call = functools.partial(activation_memory.checkpoint, layer, swap_inputs=False,
                                 context_fn=_selective_contexts)
        return _measure(list(layer.parameters()), (hidden,), lambda: _run(layer, hidden, positions, call),
                        lambda out: out.backward(grad), grad_inputs=(grad,), checkpointed=True,
                        shared=tree_flatten(positions)[0])


def census_saved_ops(config: Any, layer_index: int, seq_length: int,
                     replacements: Sequence[Any] = ()) -> Dict[str, float]:
    """What layer *layer_index* of *config* keeps for its backward for each op, per token, the whole layer.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        seq_length: Tokens of the micro-batch of one sequence it runs.
        replacements: The module replacements a run installs
            (:func:`replacement_specs`).

    Returns:
        ``{op: bytes per token}``, by the op saving them
        (:class:`_SavedOps`), which sum to what the layer keeps.
    """
    ops: Dict[str, int] = {}
    census_layer(config, layer_index, seq_length, ops=ops, replacements=replacements)
    return {op: size / seq_length for op, size in ops.items()}


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


def census_parameters(config: Any, layer_index: int, replacements: Sequence[Any] = ()) -> Dict[str, int]:
    """The parameters of layer *layer_index* of *config*, by part.

    Args:
        config: The language model's Transformers config.
        layer_index: The layer to build, which settles its kind.
        replacements: The module replacements a run installs
            (:func:`replacement_specs`).  A replacement holds the
            parameters its source held, fused or renamed, so it counts the
            same.

    Returns:
        The parameter count of each part the layer has
        (:func:`_parameter_part`).
    """
    parts: Dict[str, int] = {}
    with _fake_layer(config, layer_index, replacements=replacements) as (layer, _):
        for name, param in layer.named_parameters():
            part = _parameter_part(name)
            parts[part] = parts.get(part, 0) + param.numel()
    return parts


def census_final_norm(config: Any, replacements: Sequence[Any] = ()) -> int:
    """The parameters of *config*'s final norm, of the class of a layer's input norm."""
    with _fake_layer(config, 0, replacements=replacements) as (layer, _):
        return sum(param.numel() for param in layer.input_layernorm.parameters())


def census_output(config: Any, seq_length: int, replacements: Sequence[Any] = ()) -> Tuple[int, int]:
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
    with npu_contracts() if replacements else contextlib.nullcontext(), \
            FakeTensorMode(allow_non_fake_inputs=True):
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            # The final norm is of the class of a layer's input norm.
            norm = layer_cls(config, 0).input_layernorm
            head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        finally:
            torch.set_default_dtype(default)
        if replacements:
            holder = _Holder(norm=norm)
            _replaced(holder, replacements)
            norm = holder.model.norm
        hidden = torch.randn(1, seq_length, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
        labels = torch.randint(0, config.vocab_size, (1, seq_length))
        # The loss alone holds the logits: the trainer drops the model's
        # output before the backward.
        return _measure([*norm.parameters(), *head.parameters()], (hidden,),
                        lambda: loss_function(head(norm(hidden)), labels, config.vocab_size),
                        lambda loss: loss.backward())


def census_output_activations(config: Any, seq_length: int = 4096,
                             replacements: Sequence[Any] = ()) -> KindActivations:
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
    (saved_1, working_1), (saved_2, working_2) = (
        census_output(each, seq_length, replacements) for each in (config, half))
    return KindActivations(
        saved=max(0.0, 2 * saved_2 - saved_1) / seq_length,
        saved_tp=max(0.0, 2 * (saved_1 - saved_2)) / seq_length,
        working=max(0.0, 2 * working_2 - working_1) / seq_length,
        working_tp=max(0.0, 2 * (working_1 - working_2)) / seq_length,
        seq_length=int(seq_length),
    )


def census_activations(config: Any, layers: Iterable[Mapping[str, Any]], seq_length: int = 4096,
                       replacements: Sequence[Any] = (),
                       gdn_backend: str = _EAGER_GDN) -> Dict[str, KindActivations]:
    """What a layer of each kind of *layers* keeps and holds, per token, at TP 1 and TP 2.

    Args:
        config: The language model's Transformers config.
        layers: The model spec's layer stack, groups in model order; MTP
            groups are left out.
        seq_length: The tokens the census runs a layer at.
        replacements: The module replacements a run installs
            (:func:`replacement_specs`).
        gdn_backend: The gated delta rule the run takes (:func:`_fake_layer`).

    Returns:
        Each kind's :class:`KindActivations`: of the bytes a layer at TP 2
        holds, the part TP splits is half the part at TP 1; what it keeps
        under HyperParallel's selective activation checkpointing too, the
        shares of its matmul FLOPs that recomputes, and what it keeps for
        each op, its records per op.
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
        ops_1: Dict[str, int] = {}
        ops_2: Dict[str, int] = {}
        (saved_1, working_1), (saved_2, working_2) = (
            census_layer(tp_config(config, tp), layer_index, seq_length, ops=ops,
                         replacements=replacements, gdn_backend=gdn_backend)
            for tp, ops in ((1, ops_1), (2, ops_2)))
        (kept_1, _), (kept_2, _) = (
            census_layer(tp_config(config, tp), layer_index, seq_length, selective=True,
                         replacements=replacements, gdn_backend=gdn_backend)
            for tp in (1, 2))
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
            ops={op: max(0.0, 2 * ops_2.get(op, 0) - ops_1[op]) / seq_length for op in sorted(ops_1)},
            ops_tp={op: max(0.0, 2 * (ops_1[op] - ops_2.get(op, 0))) / seq_length for op in sorted(ops_1)},
        )
    return out
