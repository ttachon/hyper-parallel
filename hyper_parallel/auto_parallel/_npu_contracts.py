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
"""Shape contracts of the NPU kernels HyperParallel's fused modules call, for the census.

A train.yaml's ``plan_overrides`` may replace Transformers' modules with
HyperParallel's fused ones (``replace_module``): its RMSNorm, its grouped-query
attention, its grouped experts.  Their kernels are ``torch_npu``'s, which the
census, running fake tensors on a host, cannot call.  :func:`npu_contracts`
stands in for ``torch_npu`` with a contract for each kernel those modules
call: an op returning outputs of the kernel's shapes, which saves for its
backward what the kernel's backward takes, as ``torch_npu``'s derivatives
state it.

- ``npu_rms_norm`` saves its input, its weight and the fp32 reciprocal of
  each row's root mean square;
- ``npu_swiglu`` saves its input, the gate and up projections side by side;
- ``npu_rotary_mul`` saves its input and the two rotary tables;
- ``npu_fusion_attention`` saves its queries, keys and values, its output
  and its two fp32 softmax statistics per head and token; the mask it also
  takes is one every layer shares, which no layer's census counts;
- ``npu_moe_token_permute`` saves its tokens, their experts and the order it
  sorts them in, and ``npu_moe_token_unpermute`` the expert outputs, that
  order and the routing weights;
- ``npu_grouped_matmul`` runs inside HyperParallel's own autograd functions,
  which save what their backward takes, so its contract only shapes its
  output.

Any other kernel raises when called, naming it.
"""
from __future__ import annotations

import contextlib
import importlib
import importlib.machinery
import sys
import types
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import torch  # pylint: disable=forbidden-backend-import

# The softmax statistics the fused attention keeps per head and token.
_STATS = 8

# What ``sys.modules`` held for ``torch_npu`` when nothing did.
_ABSENT = object()


def _rms_norm_outputs(x: torch.Tensor, gamma: torch.Tensor, epsilon: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """The normalized rows, and each row's fp32 reciprocal root mean square."""
    del gamma, epsilon
    return torch.empty_like(x), x.new_empty((*x.shape[:-1], 1), dtype=torch.float32)


_RMS_NORM = torch.library.custom_op(
    "nd_census_npu::rms_norm", mutates_args=(),
    schema="(Tensor x, Tensor gamma, float epsilon) -> (Tensor, Tensor)")(_rms_norm_outputs)
_RMS_NORM.register_fake(_rms_norm_outputs)


def _rms_norm_saves(ctx: Any, inputs: Tuple[Any, ...], output: Tuple[torch.Tensor, ...]) -> None:
    """Keep the input, the weight and the reciprocal root mean squares."""
    ctx.mark_non_differentiable(output[1])
    ctx.save_for_backward(inputs[0], inputs[1], output[1])


def _rms_norm_backward(ctx: Any, *grads: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
    """Gradients of the input's and the weight's shapes."""
    del grads
    x, gamma, _ = ctx.saved_tensors
    return torch.empty_like(x), torch.empty_like(gamma), None


_RMS_NORM.register_autograd(_rms_norm_backward, setup_context=_rms_norm_saves)


def _swiglu_output(x: torch.Tensor, dim: int) -> torch.Tensor:
    """The gated half of *x*'s width along *dim*."""
    shape = list(x.shape)
    shape[dim] //= 2
    return x.new_empty(shape)


_SWIGLU = torch.library.custom_op(
    "nd_census_npu::swiglu", mutates_args=(), schema="(Tensor x, int dim) -> Tensor")(_swiglu_output)
_SWIGLU.register_fake(_swiglu_output)
_SWIGLU.register_autograd(
    lambda ctx, grad: (torch.empty_like(ctx.saved_tensors[0]), None),
    setup_context=lambda ctx, inputs, output: ctx.save_for_backward(inputs[0]))


def _rotary_output(x: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """The rotated input, of its shape."""
    del r1, r2
    return torch.empty_like(x)


_ROTARY = torch.library.custom_op(
    "nd_census_npu::rotary_mul", mutates_args=(), schema="(Tensor x, Tensor r1, Tensor r2) -> Tensor")(_rotary_output)
_ROTARY.register_fake(_rotary_output)
_ROTARY.register_autograd(
    lambda ctx, grad: tuple(torch.empty_like(tensor) for tensor in ctx.saved_tensors),
    setup_context=lambda ctx, inputs, output: ctx.save_for_backward(*inputs))


def _attention_outputs(query: torch.Tensor, key: torch.Tensor,
                       value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The fused attention's output, as wide as the values, and its two fp32 statistics, BNSD."""
    del key
    batch, heads, seq, _ = query.shape
    stats = [query.new_empty((batch, heads, seq, _STATS), dtype=torch.float32) for _ in range(2)]
    return query.new_empty((batch, heads, seq, value.shape[-1])), stats[0], stats[1]


_ATTENTION = torch.library.custom_op(
    "nd_census_npu::fusion_attention", mutates_args=(),
    schema="(Tensor query, Tensor key, Tensor value) -> (Tensor, Tensor, Tensor)")(_attention_outputs)
_ATTENTION.register_fake(_attention_outputs)


def _attention_saves(ctx: Any, inputs: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
    """Keep the queries, keys and values, the output and the statistics."""
    ctx.mark_non_differentiable(*output[1:])
    ctx.save_for_backward(*inputs, *output)


_ATTENTION.register_autograd(
    lambda ctx, *grads: tuple(torch.empty_like(tensor) for tensor in ctx.saved_tensors[:3]),
    setup_context=_attention_saves)


def _permute_outputs(tokens: torch.Tensor, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Each token once per expert it routes to, expert by expert, and the order they were sorted in."""
    routes = indices.numel()
    return tokens.new_empty((routes, tokens.shape[-1])), indices.new_empty((routes,), dtype=torch.int32)


_PERMUTE = torch.library.custom_op(
    "nd_census_npu::moe_token_permute", mutates_args=(),
    schema="(Tensor tokens, Tensor indices) -> (Tensor, Tensor)")(_permute_outputs)
_PERMUTE.register_fake(_permute_outputs)


def _permute_saves(ctx: Any, inputs: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
    """Keep the tokens, their experts and the sorting order."""
    ctx.mark_non_differentiable(output[1])
    ctx.save_for_backward(inputs[0], inputs[1], output[1])


_PERMUTE.register_autograd(
    lambda ctx, *grads: (torch.empty_like(ctx.saved_tensors[0]), None), setup_context=_permute_saves)


def _unpermute_output(permuted: torch.Tensor, sorted_indices: torch.Tensor,
                      probs: Optional[torch.Tensor]) -> torch.Tensor:
    """Each token's routes combined back into one row per token."""
    tokens = probs.shape[0] if probs is not None else sorted_indices.numel()
    return permuted.new_empty((tokens, permuted.shape[-1]))


_UNPERMUTE = torch.library.custom_op(
    "nd_census_npu::moe_token_unpermute", mutates_args=(),
    schema="(Tensor permuted, Tensor sorted_indices, Tensor? probs) -> Tensor")(_unpermute_output)
_UNPERMUTE.register_fake(_unpermute_output)


def _unpermute_saves(ctx: Any, inputs: Tuple[Any, ...], output: torch.Tensor) -> None:
    """Keep the expert outputs, the sorting order and the routing weights."""
    del output
    ctx.has_probs = inputs[2] is not None
    ctx.save_for_backward(*(tensor for tensor in inputs if tensor is not None))


def _unpermute_backward(ctx: Any, grad: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
    """Gradients of the expert outputs' and the routing weights' shapes."""
    del grad
    saved = ctx.saved_tensors
    return torch.empty_like(saved[0]), None, torch.empty_like(saved[2]) if ctx.has_probs else None


_UNPERMUTE.register_autograd(_unpermute_backward, setup_context=_unpermute_saves)


def npu_rms_norm(x: torch.Tensor, gamma: torch.Tensor, epsilon: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """``torch_npu.npu_rms_norm``'s contract."""
    return _RMS_NORM(x, gamma, float(epsilon))


def npu_swiglu(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """``torch_npu.npu_swiglu``'s contract."""
    return _SWIGLU(x, int(dim) % x.dim())


def npu_rotary_mul(x: torch.Tensor, r1: torch.Tensor, r2: torch.Tensor, rotary_mode: str = "half") -> torch.Tensor:
    """``torch_npu.npu_rotary_mul``'s contract."""
    del rotary_mode
    return _ROTARY(x, r1, r2)


def npu_fusion_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, head_num: int,
                         input_layout: str, *args: Any, **kwargs: Any) -> Tuple[Any, ...]:
    """``torch_npu.npu_fusion_attention``'s contract, for the BNSD and BSND layouts.

    Returns the kernel's seven outputs: the attention's, the two statistics,
    an empty softmax output, and the dropout's seed, offset and count.

    Raises:
        NotImplementedError: For another layout.
    """
    del head_num, args, kwargs
    if input_layout not in ("BNSD", "BSND"):
        raise NotImplementedError(f"the census has no contract for npu_fusion_attention's {input_layout} layout")
    bnsd = input_layout == "BNSD"
    tensors = [tensor if bnsd else tensor.transpose(1, 2) for tensor in (query, key, value)]
    output, softmax_max, softmax_sum = _ATTENTION(*tensors)
    return (output if bnsd else output.transpose(1, 2), softmax_max, softmax_sum,
            query.new_empty((0,)), 0, 0, 0)


def npu_moe_token_permute(tokens: torch.Tensor, indices: torch.Tensor, num_out_tokens: Optional[int] = None,
                          padded_mode: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """``torch_npu.npu_moe_token_permute``'s contract, every route kept."""
    del num_out_tokens, padded_mode
    return _PERMUTE(tokens, indices)


def npu_moe_token_unpermute(permuted_tokens: torch.Tensor, sorted_indices: torch.Tensor,
                            probs: Optional[torch.Tensor] = None, padded_mode: bool = False,
                            restore_shape: Optional[Sequence[int]] = None) -> torch.Tensor:
    """``torch_npu.npu_moe_token_unpermute``'s contract."""
    del padded_mode, restore_shape
    return _UNPERMUTE(permuted_tokens, sorted_indices, probs)


def npu_grouped_matmul(x: Sequence[torch.Tensor], weight: Sequence[torch.Tensor], *args: Any,
                       group_list: Any = None, group_type: int = 0, **kwargs: Any) -> List[torch.Tensor]:
    """``torch_npu.npu_grouped_matmul``'s output shape, one tensor of a single split.

    Grouped along the rows (``group_type`` 0), each group's rows times its
    weight; along the reduced dimension (2, a weight's gradient), one
    product per group.
    """
    del args, kwargs
    rows, weights = x[0], weight[0]
    if group_type == 2:
        groups = group_list.shape[0] if isinstance(group_list, torch.Tensor) else len(group_list)
        return [rows.new_empty((groups, rows.shape[0], weights.shape[-1]))]
    return [rows.new_empty((rows.shape[0], weights.shape[-1]))]


def _missing(name: str) -> Any:
    """What the stand-in answers for a kernel it has no contract for: one that raises when called."""
    if name.startswith("_"):
        raise AttributeError(name)

    def kernel(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise NotImplementedError(f"the census has no contract for torch_npu.{name}")

    return kernel


def _holders(bound: Any) -> List[types.ModuleType]:
    """HyperParallel's modules that hold *bound* as their ``torch_npu``."""
    return [module for name, module in list(sys.modules.items())
            if name.startswith("hyper_parallel.") and bound is not _ABSENT
            and getattr(module, "torch_npu", None) is bound]


def _stand_in() -> types.ModuleType:
    """A ``torch_npu`` module of the contracts."""
    module = types.ModuleType("torch_npu")
    module.__spec__ = importlib.machinery.ModuleSpec("torch_npu", None)
    for kernel in (npu_rms_norm, npu_swiglu, npu_rotary_mul, npu_fusion_attention, npu_moe_token_permute,
                   npu_moe_token_unpermute, npu_grouped_matmul):
        setattr(module, kernel.__name__, kernel)
    module.__getattr__ = _missing
    return module


TORCH_NPU = _stand_in()


@contextlib.contextmanager
def npu_contracts() -> Iterator[types.ModuleType]:
    """``torch_npu`` as the contracts, for what runs inside.

    A module that imports ``torch_npu`` inside, as HyperParallel's fused
    modules do, keeps the contracts; one that had imported the real
    ``torch_npu`` before calls the contracts until the context ends.
    HyperParallel's model package, whose build options probe the device when
    ``torch_npu`` imports, is imported before.
    """
    importlib.import_module("hyper_parallel.models")
    previous = sys.modules.get("torch_npu", _ABSENT)
    sys.modules["torch_npu"] = TORCH_NPU
    for module in _holders(previous):
        module.torch_npu = TORCH_NPU
    try:
        yield TORCH_NPU
    finally:
        if previous is _ABSENT:
            sys.modules.pop("torch_npu", None)
        else:
            sys.modules["torch_npu"] = previous
            # Including a module that imported the stand-in while it stood in.
            for module in _holders(TORCH_NPU):
                module.torch_npu = previous
