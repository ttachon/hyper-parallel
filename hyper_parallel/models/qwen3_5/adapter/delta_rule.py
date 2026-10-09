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
"""Run Qwen3.5's chunked gated delta rule with a backward that stays linear in the sequence.

Without context parallelism a Qwen3.5 linear-attention layer runs the rule
Transformers ships, and where the ``fla`` kernels are absent that is its torch
fallback. The fallback walks the chunks reading ``x[:, :, i]`` and writing
``out[:, :, i]``, so autograd answers every chunk with a gradient as large as
the whole tensor and adds them up: the backward grows with the square of the
sequence. :func:`install_delta_rule` points the layers at a HyperParallel rule
that takes each tensor apart once and stacks the outputs once, after checking
on a probe that it reproduces the installed fallback exactly, outputs, state
and gradients alike. Two formulations exist, and the probe picks the one the
installed Transformers runs: HyperParallel's oracle, which Transformers 5.16
and earlier match, and :func:`chunk_gated_delta_rule_solve`, which follows the
triangular solve of later versions.
"""

import importlib.util
import inspect
import logging
import sys
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch  # pylint: disable=forbidden-backend-import
from torch import nn  # pylint: disable=forbidden-backend-import
from torch.nn import functional as F  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.gated_delta_net import torch_chunk_gated_delta_rule

__all__ = ["chunk_gated_delta_rule_solve", "install_delta_rule"]

logger = logging.getLogger(__name__)

# The name a Transformers modeling module gives its torch fallback, which a
# layer of 5.15 and later calls by name.
_RULE_NAME = "torch_chunk_gated_delta_rule"
# The attribute a layer of 5.14 and earlier calls its rule through.
_RULE_ATTRIBUTE = "chunk_gated_delta_rule"
# The package whose kernels Transformers prefers to its fallback.
_FAST_PATH_PACKAGE = "fla"
# Where the layers this switches are defined; HyperParallel's own layers call
# its rule already.
_TRANSFORMERS_PACKAGE = "transformers."
# The probe: ten tokens in chunks of four, so the last chunk is padded.
_PROBE_TOKENS = 10
_PROBE_CHUNK = 4
_PROBE_HEADS = 2
_PROBE_K_DIM = 4
_PROBE_V_DIM = 3


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize along ``dim``, as Transformers' Qwen3.5 modeling does."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def chunk_gated_delta_rule_solve(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The chunked gated delta rule by triangular solve, its chunks taken apart once.

    The formulation of Transformers 5.17's torch fallback: the within-chunk
    system is solved with ``torch.linalg.solve_triangular`` and the decays are
    applied before the scan, so a chunk's step is two products and a sum.

    Args:
        query: ``(batch, seq, k_heads, k_dim)``.
        key: ``(batch, seq, k_heads, k_dim)``.
        value: ``(batch, seq, v_heads, v_dim)``.
        g: The log decay, ``(batch, seq, v_heads)``, entries at most 0.
        beta: ``(batch, seq, v_heads)``.
        chunk_size: Tokens a chunk holds.
        initial_state: The recurrent state to start from,
            ``(batch, v_heads, k_dim, v_dim)``, or None for zeros.
        output_final_state: Whether to return the last recurrent state.
        use_qk_l2norm_in_kernel: Whether to L2-normalize queries and keys.

    Returns:
        The output, ``(batch, seq, v_heads, v_dim)`` in the query's dtype, and
        the last recurrent state when ``output_final_state`` asks for it, else
        None.
    """
    initial_dtype = query.dtype
    batch_size, sequence_length, _, k_head_dim = key.shape
    num_v_heads, v_head_dim = value.shape[-2:]
    query, key, value, beta, decay = [
        x.transpose(1, 2).to(torch.float32, memory_format=torch.contiguous_format)
        for x in (query, key, value, beta, g)
    ]
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query = query * query.shape[-1] ** -0.5

    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad_size)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad_size)) for x in (beta, decay))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)
    ]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

    strictly_upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum_decay = decay.cumsum(dim=3)
    pairwise_decay = (cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)).masked_fill(strictly_upper, float("-inf"))
    pairwise_decay = pairwise_decay.exp()
    ut_system = (k_beta @ key.transpose(-1, -2)) * pairwise_decay
    intra_chunk_attn = (query @ key.transpose(-1, -2)) * pairwise_decay
    decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)
    new_values = torch.linalg.solve_triangular(  # pylint: disable=not-callable
        ut_system, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(  # pylint: disable=not-callable
        ut_system, decayed_k_beta, upper=False, unitriangular=True)
    if initial_state is None:
        state = torch.zeros((batch_size, num_v_heads, k_head_dim, v_head_dim),
                            dtype=new_values.dtype, device=new_values.device)
    else:
        state = initial_state.to(new_values)
    query = query * cum_decay.exp().unsqueeze(-1)
    key = key * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
    chunk_decay = cum_decay[..., -1].exp()[..., None, None]

    # Taken apart once and stacked once, for the reason the module states.
    chunks = zip(new_values.unbind(2), k_cumdecay.unbind(2), query.unbind(2), intra_chunk_attn.unbind(2),
                 key.unbind(2), chunk_decay.unbind(2))
    outputs = []
    for new_values_i, k_cumdecay_i, query_i, intra_i, key_i, chunk_decay_i in chunks:
        v_new = new_values_i - k_cumdecay_i @ state
        outputs.append(query_i @ state + intra_i @ v_new)
        state = state * chunk_decay_i + key_i.transpose(-1, -2) @ v_new
    core_attn_out = torch.stack(outputs, dim=2)

    core_attn_out = core_attn_out.reshape(batch_size, num_v_heads, -1, v_head_dim)[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).to(initial_dtype, memory_format=torch.contiguous_format)
    return core_attn_out, (state if output_final_state else None)


# The HyperParallel rules a Transformers fallback may be replaced by, in the
# order they are tried against it.
_CANDIDATES = (torch_chunk_gated_delta_rule, chunk_gated_delta_rule_solve)


def _probe_inputs(dtype: torch.dtype) -> list[torch.Tensor]:
    """Query, key, value, log decay, beta and a start state, as a layer hands them in *dtype*."""
    generator = torch.Generator().manual_seed(0)
    k_shape = (1, _PROBE_TOKENS, _PROBE_HEADS, _PROBE_K_DIM)
    query = torch.randn(k_shape, generator=generator).to(dtype)
    key = torch.randn(k_shape, generator=generator).to(dtype)
    value = torch.randn((1, _PROBE_TOKENS, _PROBE_HEADS, _PROBE_V_DIM), generator=generator).to(dtype)
    # A layer computes its log decay in float32 whatever the model's dtype.
    decay = -torch.rand((1, _PROBE_TOKENS, _PROBE_HEADS), generator=generator)
    beta = torch.rand((1, _PROBE_TOKENS, _PROBE_HEADS), generator=generator).to(dtype)
    state = torch.randn((1, _PROBE_HEADS, _PROBE_K_DIM, _PROBE_V_DIM), generator=generator)
    return [query, key, value, decay, beta, state]


def _probe(rule: Callable[..., Any], inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """The output, the last state and every input's gradient *rule* gives on *inputs*."""
    leaves = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    output, state = rule(*leaves[:5], chunk_size=_PROBE_CHUNK, initial_state=leaves[5],
                         output_final_state=True, use_qk_l2norm_in_kernel=True)
    generator = torch.Generator().manual_seed(1)
    output_weight = torch.randn(output.shape, generator=generator)
    state_weight = torch.randn(state.shape, generator=generator)
    ((output.float() * output_weight).sum() + (state * state_weight).sum()).backward()
    return [output.detach(), state.detach()] + [leaf.grad for leaf in leaves]


def _reproduces(candidate: Callable[..., Any], reference: Callable[..., Any]) -> bool:
    """Whether *candidate* gives exactly what *reference* gives, in float32 and in bfloat16."""
    for dtype in (torch.float32, torch.bfloat16):
        inputs = _probe_inputs(dtype)
        try:
            expected = _probe(reference, inputs)
        except (TypeError, RuntimeError) as error:
            logger.warning("the installed gated delta rule failed the probe: %s", error)
            return False
        actual = _probe(candidate, inputs)
        if not all(torch.equal(got, want) for got, want in zip(actual, expected)):
            return False
    return True


def _transformers_call(rule: Callable[..., Any]) -> Callable[..., Any]:
    """*rule* called as Transformers calls its fallback, whose further keywords, ``cu_seqlens`` among them, both ignore.

    The call takes *rule*'s name and documentation but not ``__wrapped__``:
    whatever unwraps a layer's rule, as ND's census does, must still reach a
    callable that takes Transformers' keywords.
    """

    def _call(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, g: torch.Tensor, beta: torch.Tensor,
              chunk_size: int = 64, initial_state: Optional[torch.Tensor] = None, output_final_state: bool = False,
              use_qk_l2norm_in_kernel: bool = False,
              **_unused: Any) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        return rule(query, key, value, g, beta, chunk_size=chunk_size, initial_state=initial_state,
                    output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

    _call.__name__ = rule.__name__
    _call.__qualname__ = rule.__qualname__
    _call.__doc__ = rule.__doc__
    return _call


def _calls_rule_by_name(module: nn.Module) -> bool:
    """Whether *module*'s forward calls its modeling module's fallback by name, as from Transformers 5.15."""
    forward = getattr(type(module), "forward", None)
    code = getattr(inspect.unwrap(forward), "__code__", None) if forward is not None else None
    return code is not None and _RULE_NAME in code.co_names


def _runs_transformers_rule(module: nn.Module) -> bool:
    """Whether *module* is a Transformers layer running a chunked gated delta rule."""
    if not type(module).__module__.startswith(_TRANSFORMERS_PACKAGE):
        return False
    return hasattr(module, _RULE_ATTRIBUTE) or _calls_rule_by_name(module)


def _replacement(modeling: Any) -> Optional[Callable[..., Any]]:
    """The HyperParallel rule that reproduces *modeling*'s fallback, called as Transformers calls it, or None."""
    fallback = getattr(modeling, _RULE_NAME, None)
    if fallback is None:
        logger.warning("%s has no %s; its layers keep their gated delta rule", modeling.__name__, _RULE_NAME)
        return None
    reference = inspect.unwrap(fallback)
    for candidate in _CANDIDATES:
        if _reproduces(candidate, reference):
            return _transformers_call(candidate)
    logger.warning("no HyperParallel gated delta rule reproduces %s.%s; its layers keep running it",
                   modeling.__name__, _RULE_NAME)
    return None


def _switch(module: nn.Module, modeling: Any, rule: Callable[..., Any]) -> bool:
    """Point *module* at *rule* where it runs *modeling*'s fallback; whether it now does."""
    if hasattr(module, _RULE_ATTRIBUTE):
        # Up to Transformers 5.14 a layer holds its rule, the fallback or a
        # kernel; a kernel is left alone.
        if getattr(module, _RULE_ATTRIBUTE) is not getattr(modeling, _RULE_NAME, None):
            return False
        setattr(module, _RULE_ATTRIBUTE, rule)
        return True
    if importlib.util.find_spec(_FAST_PATH_PACKAGE) is not None:
        # From 5.15 the name prefers the kernels wherever they import.
        return False
    # The name is the modeling module's, so every layer it builds runs the rule.
    setattr(modeling, _RULE_NAME, rule)
    return True


def install_delta_rule(model: nn.Module) -> int:
    """Have *model*'s gated delta rule layers run a HyperParallel rule whose backward stays linear.

    Each layer that runs its Transformers modeling module's torch fallback is
    pointed at the HyperParallel rule that reproduces that fallback exactly on
    a probe, in float32 and bfloat16, outputs, last state and gradients; a
    layer running a kernel, or a fallback no rule reproduces, is left as it
    is. From Transformers 5.15 a layer calls the fallback by its module-level
    name, so the switch holds for every layer that modeling module builds in
    this process.

    Args:
        model: A model holding Transformers gated delta rule layers, such as a
            Qwen3.5 or Qwen3.5-MoE one, built or parallelized.

    Returns:
        The number of layers that now run a HyperParallel rule.
    """
    rules: dict[str, Optional[Callable[..., Any]]] = {}
    switched = 0
    for module in model.modules():
        if not _runs_transformers_rule(module):
            continue
        modeling = sys.modules.get(type(module).__module__)
        if modeling is None:
            continue
        if modeling.__name__ not in rules:
            rules[modeling.__name__] = _replacement(modeling)
        rule = rules[modeling.__name__]
        if rule is not None and _switch(module, modeling, rule):
            switched += 1
    if switched:
        logger.info("%d gated delta rule layers run %s", switched,
                    ", ".join(sorted({rule.__name__ for rule in rules.values() if rule is not None})))
    return switched
