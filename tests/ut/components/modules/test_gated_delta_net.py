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
"""Tests for the chunked gated delta rule's eager oracle."""
import pytest
import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode  # pylint: disable=forbidden-backend-import

from hyper_parallel.components.modules.gated_delta_net import torch_chunk_gated_delta_rule


def _l2norm(x, dim=-1, eps=1e-6):
    """The oracle's L2 normalization."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _indexed_chunk_rule(query, key, value, g, beta, chunk_size=64, initial_state=None,
                        output_final_state=False, use_qk_l2norm_in_kernel=False):
    """The oracle as it read before its scan took the chunks apart once: chunk i read and written by index."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32)
                                  for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                         for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    state = (torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
             if initial_state is None else initial_state.to(value))
    core_attn_out = torch.zeros_like(value)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_new = v_i - k_cumdecay[:, :, i] @ state
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        state = (state * g[:, :, i, -1, None, None].exp()
                 + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1,
                                          core_attn_out.shape[-1])[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, (state if output_final_state else None)


def _inputs(dtype, tokens):
    """Query, key, value, log decay, beta and a start state, as a layer hands them in *dtype*."""
    generator = torch.Generator().manual_seed(5)
    query = torch.randn(2, tokens, 2, 4, generator=generator).to(dtype)
    key = torch.randn(2, tokens, 2, 4, generator=generator).to(dtype)
    value = torch.randn(2, tokens, 2, 3, generator=generator).to(dtype)
    decay = -torch.rand(2, tokens, 2, generator=generator)
    beta = torch.rand(2, tokens, 2, generator=generator).to(dtype)
    state = torch.randn(2, 2, 4, 3, generator=generator)
    return [query, key, value, decay, beta, state]


def _run(rule, inputs, chunk_size):
    """The output, the last state and every input's gradient *rule* gives."""
    leaves = [tensor.clone().requires_grad_(True) for tensor in inputs]
    output, state = rule(*leaves[:5], chunk_size=chunk_size, initial_state=leaves[5],
                         output_final_state=True, use_qk_l2norm_in_kernel=True)
    generator = torch.Generator().manual_seed(6)
    weight = torch.randn(output.shape, generator=generator)
    ((output.float() * weight).sum() + (state * state.detach()).sum()).backward()
    return [output.detach(), state.detach()] + [leaf.grad for leaf in leaves]


class _ElementsWritten(TorchDispatchMode):
    """Sums the elements the ops of one name write while it is active."""

    def __init__(self, name: str) -> None:
        """Count nothing yet, for the ops named *name*."""
        super().__init__()
        self.name = name
        self.elements = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):  # pylint: disable=unused-argument
        out = func(*args, **(kwargs or {}))
        if func.overloadpacket.__name__ == self.name:
            self.elements += out.numel()
        return out


def _chunk_read_gradients(rule, chunks, chunk_size=4):
    """The elements the backward of *rule* writes answering reads by index, over *chunks* chunks.

    A read ``x[..., i, ...]`` is answered by ``select_backward``, which writes
    a gradient as large as ``x``: of a chunk where ``x`` is one, of the whole
    sequence where ``x`` holds every chunk.
    """
    leaves = [tensor.requires_grad_(True) for tensor in _inputs(torch.float32, chunks * chunk_size)]
    output, state = rule(*leaves[:5], chunk_size=chunk_size, initial_state=leaves[5], output_final_state=True)
    loss = output.sum() + state.sum()
    counter = _ElementsWritten("select_backward")
    with counter:
        loss.backward()
    return counter.elements


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_oracle_matches_the_indexed_chunk_loop_exactly(dtype):
    """Taking the chunks apart once changes no output, state or gradient, a padded last chunk included."""
    inputs = _inputs(dtype, tokens=10)
    expected = _run(_indexed_chunk_rule, inputs, chunk_size=4)
    actual = _run(torch_chunk_gated_delta_rule, inputs, chunk_size=4)
    names = ["output", "state", "d_query", "d_key", "d_value", "d_g", "d_beta", "d_state"]
    for name, got, want in zip(names, actual, expected):
        assert torch.equal(got, want), \
            (f"{name} differs from the indexed loop in {dtype}: "
             f"largest difference {(got.float() - want.float()).abs().max().item()}")


def test_oracle_backward_grows_linearly_with_the_chunks():
    """What the backward writes for reads by index doubles with the chunks; the indexed loop's quadruples."""
    oracle = [_chunk_read_gradients(torch_chunk_gated_delta_rule, chunks) for chunks in (4, 8)]
    indexed = [_chunk_read_gradients(_indexed_chunk_rule, chunks) for chunks in (4, 8)]
    assert oracle[1] == 2 * oracle[0],         (f"the oracle's backward should grow linearly: "
         f"4 chunks {oracle[0]} elements, 8 chunks {oracle[1]}")
    assert indexed[1] > 3 * indexed[0],         (f"the indexed loop's backward should grow quadratically: "
         f"4 chunks {indexed[0]} elements, 8 chunks {indexed[1]}")
