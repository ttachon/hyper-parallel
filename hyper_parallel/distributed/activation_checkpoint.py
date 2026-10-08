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
"""Activation checkpointing helpers for distributed model components."""

import logging
import re
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch import nn

from hyper_parallel.core.activation_memory.api import (
    CheckpointPolicy,
    checkpoint_wrapper,
    create_selective_checkpoint_contexts,
    ignore_sac_ops as _ignore_sac_ops,
)
from hyper_parallel.core.activation_memory.swap import (
    SwapManager,
    _teardown_wired_swap_layers,
)


logger = logging.getLogger(__name__)


def _resolve_torch_op(dotted_path: str):
    """Resolve a torch operator to its default overload when available."""
    if dotted_path.startswith("ops."):
        dotted_path = dotted_path[4:]
    if dotted_path.count(".") == 1:
        dotted_path = f"{dotted_path}.default"
    obj = torch.ops
    try:
        for part in dotted_path.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    return obj


def _resolve_op_attr(root: object, dotted_path: str):
    """Resolve an optional operator outside the regular ``torch.ops`` tree."""
    obj = root
    try:
        for part in dotted_path.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    return obj


def _existing_ops(*ops):
    """Return the available operators, omitting optional torch operators."""
    return frozenset(op for op in ops if op is not None)


# Matmul operators alternate between saving and recomputing their outputs. The
# counter is scoped to one checkpoint region by ``make_selective_checkpoint_context_fn``.
_SELECTIVE_AC_MATMUL_OPS = _existing_ops(
    _resolve_torch_op("aten.matmul"),
    _resolve_torch_op("aten.mm"),
    _resolve_torch_op("aten.linear"),
    _resolve_torch_op("aten._grouped_mm"),
    _resolve_torch_op("aten._scaled_grouped_mm"),
)

# Some model implementations mutate these operator outputs in-place. Keeping
# the output by reference would either trip SAC's version check or replay the
# mutation on an already-mutated tensor during recomputation.
_SELECTIVE_AC_FORCE_RECOMPUTE_OPS = _existing_ops(
    _resolve_torch_op("aten.topk"),
)


def _default_compute_intensive_ops() -> tuple:
    """Get PyTorch's compute-intensive operator list when the private API exists."""
    try:
        # This private PyTorch API is unavailable on some supported versions.
        from torch._functorch.partitioners import get_default_op_list  # pylint: disable=import-outside-toplevel

        return tuple(op.default for op in get_default_op_list().compute_intensive_ops)
    except (ImportError, AttributeError, RuntimeError):
        return ()


def _ffpa_forward_ops() -> tuple:
    """Resolve optional FFPA forward operators after their extension registers."""
    try:
        # The optional extension registers its operators only when imported.
        import ffpa_attn.cute  # pylint: disable=C0415, W0611
    except (ImportError, OSError, RuntimeError):
        return ()
    return (
        _resolve_op_attr(torch.ops, "ffpa_attn._fwd_cute.default"),
        _resolve_op_attr(torch.ops, "ffpa_attn._varlen_fwd_cute.default"),
    )


_SELECTIVE_AC_COMPUTE_OP_NAMES = (
    "aten.mm",
    "aten.addmm",
    "aten.bmm",
    "aten.linear",
    "aten._scaled_mm",
    "aten._scaled_dot_product_cudnn_attention",
    "aten._scaled_dot_product_efficient_attention",
    "aten._scaled_dot_product_flash_attention",
    "aten._scaled_dot_product_flash_attention_for_cpu",
    "aten._scaled_dot_product_fused_attention_overrideable",
    "aten.scaled_dot_product_attention",
    # Every fused attention of this repository calls
    # ``torch_npu.npu_fusion_attention``, the operator the shard registry
    # names as well (``core/shard/ops/yaml/torch_flash_attention_score.yaml``).
    # A torch_npu whose entry point dispatches to a later interface registers
    # that operator instead, so both are named and whichever the install has
    # resolves; the other is dropped by ``_existing_ops``.
    "npu.npu_fusion_attention",
    "npu.npu_fusion_attention_v3",
    "aten._flex_attention",
    "aten.topk",
    "aten.max",
)

_SELECTIVE_AC_COMM_OP_NAMES = (
    "aten.all_to_all_single",
    "aten.reduce_scatter_tensor",
    "_c10d_functional.all_to_all_single",
    "_c10d_functional.reduce_scatter_tensor",
    "c10d.allreduce_",
)


def _build_selective_ac_must_save_ops():
    """Build the expensive/communication operator set for selective AC."""
    save_ops = set(_default_compute_intensive_ops())
    compute_ops = _existing_ops(
        *(_resolve_torch_op(name) for name in _SELECTIVE_AC_COMPUTE_OP_NAMES),
        _resolve_op_attr(torch, "_higher_order_ops.flex_attention"),
        _resolve_op_attr(torch, "_higher_order_ops.inductor_compiled_code"),
        _resolve_op_attr(torch.ops, "torch_attn._varlen_attn.default"),
        *_ffpa_forward_ops(),
    )
    comm_ops = _existing_ops(
        *(_resolve_torch_op(name) for name in _SELECTIVE_AC_COMM_OP_NAMES),
        _resolve_op_attr(torch.ops, "deepep.dispatch.default"),
        _resolve_op_attr(torch.ops, "deepep.combine.default"),
        _resolve_op_attr(torch.ops, "hybridep.dispatch.default"),
        _resolve_op_attr(torch.ops, "hybridep.combine.default"),
    )
    save_ops.update(compute_ops)
    save_ops.update(comm_ops)
    save_ops.difference_update(_SELECTIVE_AC_FORCE_RECOMPUTE_OPS)
    return frozenset(save_ops)


_SELECTIVE_AC_MUST_SAVE_OPS = _build_selective_ac_must_save_ops()


@dataclass(frozen=True)
class _TransformerBlockInfo:
    """A transformer block discovered below an HF checkpointing owner.

    ``parent`` is the registered repeated-block container (for example, a
    ``ModuleList``), while ``child_name`` is the actual registered key. Keeping
    both lets callers replace a block without assuming that the key is a
    contiguous integer or that the container is a specific PyTorch class.
    """

    fqn: str
    module: nn.Module
    parent: nn.Module
    child_name: str
    container_fqn: str


@dataclass(frozen=True)
class _LayerContainerInfo:
    """One repeated-block container and the blocks selected from it."""

    container: nn.Module
    path: str
    blocks: tuple[_TransformerBlockInfo, ...]


@dataclass(frozen=True)
class _LayerRecomputeRange:
    """Mode overrides for consecutive transformer layers."""

    first: int
    count: Optional[int]
    mode: str


# The modes a block can run in a per-layer plan.
_LAYER_MODES = ("off", "full", "selective")
# A per-layer plan's key: one block index, or an inclusive range of them.
_LAYER_RANGE_PATTERN = re.compile(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?")


@dataclass(frozen=True)
class _LayerModeRange:
    """Consecutive blocks that one entry of a per-layer plan runs in one mode."""

    key: str
    first: int
    last: int
    mode: str


def _get_checkpoint_wrapped_module(module: nn.Module) -> Optional[nn.Module]:
    """Return the module directly held by a supported checkpoint wrapper."""
    for attr_name in ("_wrapped_module", "_checkpoint_wrapped_module"):
        wrapped_module = getattr(module, attr_name, None)
        if isinstance(wrapped_module, nn.Module):
            return wrapped_module
    return None


def _find_transformer_block_modules(
    model: nn.Module,
) -> tuple[list[_TransformerBlockInfo], set[int]]:
    """Find transformer blocks below modules with HF checkpointing support.

    HuggingFace model components expose ``gradient_checkpointing`` on modules
    that own a repeated block container.  The marker is the structural
    contract; no model class name or conventional ``layers`` path is needed.
    Each direct child of a marked repeated container is returned with its
    registered name so callers can safely replace it in-place.

    Args:
        model: Model whose transformer blocks should be located.

    Returns:
        A list of block metadata and the IDs of blocks already selected. The
        ID set is useful to callers that need to add other wrap targets while
        avoiding duplicate modules.
    """
    block_infos = []
    discovered_block_ids = set()
    for owner_fqn, owner in model.named_modules():
        # Match FSDP2's behavior for nested checkpointing owners: once an
        # owner has itself been selected as a block, do not scan inside it.
        if id(owner) in discovered_block_ids:
            continue
        if not hasattr(owner, "gradient_checkpointing"):
            continue

        for container_name, container in owner.named_children():
            children = list(container.named_children())
            if not children:
                continue
            container_fqn = (
                f"{owner_fqn}.{container_name}" if owner_fqn else container_name
            )
            for block_name, block in children:
                if id(block) in discovered_block_ids:
                    continue
                discovered_block_ids.add(id(block))
                wrapped_module = _get_checkpoint_wrapped_module(block)
                if wrapped_module is not None:
                    # A checkpoint wrapper may proxy attributes from its inner
                    # block, including ``gradient_checkpointing``. Treat the
                    # wrapper and inner module as one logical transformer block
                    # during the rest of the module-tree traversal.
                    discovered_block_ids.add(id(wrapped_module))
                block_fqn = f"{container_fqn}.{block_name}"
                block_infos.append(
                    _TransformerBlockInfo(
                        fqn=block_fqn,
                        module=block,
                        parent=container,
                        child_name=block_name,
                        container_fqn=container_fqn,
                    )
                )
    return block_infos, discovered_block_ids


def ignore_sac_ops(ops: list[object | None]) -> None:
    """Exclude available runtime operators from selective-AC replay accounting.

    Args:
        ops: Backend operators to ignore. ``None`` entries represent optional
            operators that are unavailable in the installed PyTorch version.
    """
    _ignore_sac_ops(ops)


def ensure_profiler_ops_sac_ignored() -> None:
    """Keep profiler record-function operators out of selective-AC replay.

    FSDP hooks run under ``record_function`` and may execute a different number
    of profiler range operators in the original forward and recomputation. The
    range operators carry no activations, so excluding them only removes them
    from replay accounting while preserving their execution.
    """
    profiler_ops = getattr(torch.ops, "profiler", None)
    if profiler_ops is None:
        return

    ops_to_ignore = []
    for packet_name in (
        "_record_function_enter",
        "_record_function_enter_new",
        "_record_function_exit",
    ):
        packet = getattr(profiler_ops, packet_name, None)
        if packet is None:
            continue
        for overload_name in packet.overloads():
            ops_to_ignore.append(getattr(packet, overload_name))
    ignore_sac_ops(ops_to_ignore)


_FSDP_SAC_IGNORED_OP_NAMES = (
    "fsdp.all_gather_copy_in",
    "fsdp.split_with_sizes_copy",
    "fsdp.chunk_cat",
    "fsdp.copy_",
    "c10d._allgather_base_",
    "aten.empty.memory_format",
    "aten.empty_like",
    "aten.view",
)

_FSDP_SAC_IGNORED_OPS = [_resolve_torch_op(name) for name in _FSDP_SAC_IGNORED_OP_NAMES]


def ensure_fsdp_ops_sac_ignored() -> None:
    """Keep FSDP parameter-lifecycle operators out of selective-AC replay.

    Forward prefetch may unshard parameters before a checkpoint region, while
    recomputation may need to unshard them inside that region. These allocation,
    copy and collective operators manage parameters rather than model
    activations, and therefore must not be matched against the forward replay.
    """
    ignore_sac_ops(_FSDP_SAC_IGNORED_OPS)


def compile_selective_checkpoint_policy(
    ctx: Any,
    func: Any,
    *args: Any,
    **kwargs: Any,
) -> CheckpointPolicy:
    """Choose a stateless selective-checkpoint policy for compile mode.

    Args:
        ctx: Checkpoint context supplied by the native selective-checkpoint API.
        func: The operator being traced.
        *args: Operator arguments (unused).
        **kwargs: Operator keyword arguments (unused).

    Returns:
        The checkpoint policy for ``func``.
    """
    del ctx, args, kwargs
    if func in _SELECTIVE_AC_FORCE_RECOMPUTE_OPS:
        return CheckpointPolicy.MUST_RECOMPUTE
    if func in _SELECTIVE_AC_MATMUL_OPS:
        return CheckpointPolicy.MUST_SAVE
    if func in _SELECTIVE_AC_MUST_SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE


def _make_selective_checkpoint_policy_fn() -> Callable:
    """Create an isolated eager selective activation checkpointing policy."""
    matmul_counts = {False: 0, True: 0}

    def selective_checkpointing_policy(
        ctx: Any,
        func: Any,
        *args: Any,
        **kwargs: Any,
    ) -> CheckpointPolicy:
        """Decide whether ``func``'s output is saved or recomputed.

        Follows the selective-activation-checkpointing policy contract: matmuls
        alternate between save and recompute, expensive/communication ops are
        always saved, and everything else is recomputed.

        Args:
            ctx: Checkpoint context carrying the ``is_recompute`` phase flag.
            func: The operator being traced.
            *args: Operator arguments (unused).
            **kwargs: Operator keyword arguments (unused).

        Returns:
            The checkpoint policy for ``func``.
        """
        del args, kwargs
        if func in _SELECTIVE_AC_FORCE_RECOMPUTE_OPS:
            return CheckpointPolicy.MUST_RECOMPUTE
        if func in _SELECTIVE_AC_MATMUL_OPS:
            matmul_counts[ctx.is_recompute] += 1
            if matmul_counts[ctx.is_recompute] % 2:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.MUST_RECOMPUTE
        if func in _SELECTIVE_AC_MUST_SAVE_OPS:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.MUST_RECOMPUTE

    return selective_checkpointing_policy


def make_selective_checkpoint_context_fn() -> Callable[[], tuple[object, object]]:
    """Create a per-checkpoint-region selective activation policy context.

    Expensive operations are saved, ordinary operations are recomputed, and
    matmul operations alternate between the two decisions. A new counter is
    created every time the returned factory is invoked, matching the
    ``context_fn`` contract of non-reentrant checkpointing.

    Returns:
        A no-argument factory that creates the forward and recompute contexts.
    """
    ensure_profiler_ops_sac_ignored()
    ensure_fsdp_ops_sac_ignored()

    def selective_checkpoint_context_fn() -> tuple[object, object]:
        """Create a fresh pair of forward/recompute selective-AC contexts.

        Returns:
            The ``(forward_context, recompute_context)`` pair expected by the
            non-reentrant checkpointing ``context_fn`` contract.
        """
        return create_selective_checkpoint_contexts(
            _make_selective_checkpoint_policy_fn()
        )

    return selective_checkpoint_context_fn


def _find_transformer_layer_container_infos(
    model: nn.Module,
) -> list[_LayerContainerInfo]:
    """Find repeated block containers without model-specific path rules.

    Containers are derived from :func:`_find_transformer_block_modules`, so a
    model can freely name its towers and layer attributes.  A container is
    included once even when blocks are shared or the module tree exposes an
    alias to it.

    Args:
        model: Model whose transformer layer containers should be located.

    Returns:
        A list of discovered containers, ordered by the model's registration
        order. An empty list means no module advertises HF checkpointing
        support for a repeated block container.
    """
    block_infos, _ = _find_transformer_block_modules(model)
    containers = []
    blocks_by_container = {}
    seen_container_ids = set()
    for block_info in block_infos:
        container_id = id(block_info.parent)
        blocks_by_container.setdefault(container_id, []).append(block_info)

    for block_info in block_infos:
        container_id = id(block_info.parent)
        if container_id in seen_container_ids:
            continue
        seen_container_ids.add(container_id)
        containers.append(
            _LayerContainerInfo(
                container=block_info.parent,
                path=block_info.container_fqn,
                blocks=tuple(blocks_by_container[container_id]),
            )
        )
    return containers


def _flatten_layer_container_infos(
    containers: list[_LayerContainerInfo],
) -> list[nn.Module]:
    """Expand discovered containers into individual transformer blocks."""
    return [
        block_info.module
        for container in containers
        for block_info in container.blocks
    ]


def _should_use_hf_native_gradient_checkpointing(
    model: nn.Module,
    layers: list[nn.Module],
    *,
    enable_compile: bool = False,
) -> bool:
    """Return whether full checkpointing can use HuggingFace's native API."""
    if enable_compile:
        return False

    if not layers or any(
        not any(parameter.requires_grad for parameter in layer.parameters())
        for layer in layers
    ):
        return False

    try:
        # Guarded import: transformers is optional and older versions may not
        # expose GradientCheckpointingLayer.
        from transformers.modeling_layers import GradientCheckpointingLayer  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False

    return (
        all(isinstance(layer, GradientCheckpointingLayer) for layer in layers)
        and getattr(model, "supports_gradient_checkpointing", False)
        and hasattr(model, "gradient_checkpointing_enable")
    )


def _wrap_layer_containers(
    containers: list[_LayerContainerInfo],
    wrapper: Callable,
    *,
    context_fn: Optional[Callable[[], tuple[object, object]]] = None,
) -> int:
    """Wrap every layer in the discovered containers and return the count."""
    wrapped_count = 0
    for container_info in containers:
        for block_info in container_info.blocks:
            checkpoint_kwargs = {} if context_fn is None else {"context_fn": context_fn}
            current_block = getattr(block_info.parent, block_info.child_name, None)
            if not isinstance(current_block, nn.Module) or _is_checkpoint_wrapped(current_block):
                continue
            setattr(
                block_info.parent,
                block_info.child_name,
                wrapper(current_block, **checkpoint_kwargs),
            )
            wrapped_count += 1
    return wrapped_count


def _is_checkpoint_wrapped(module: nn.Module) -> bool:
    """Return whether a module is already wrapped by a supported checkpoint wrapper."""
    return hasattr(module, "_wrapped_module") or hasattr(
        module,
        "_checkpoint_wrapped_module",
    )


def _warn_if_nothing_wrapped(
    wrapped_count: int,
    activation_checkpoint: str,
    containers: list[_LayerContainerInfo],
) -> None:
    """Warn when a checkpointing request wrapped no module at all.

    Every wrapping strategy either probes for known submodule names or skips
    modules that are already wrapped, so a zero count silently leaves the model
    running fully eager and memory-unbounded.
    """
    if wrapped_count != 0 or not containers:
        return
    layer_count = sum(len(container.blocks) for container in containers)
    logger.warning(
        "%s activation checkpointing wrapped no module on %d layer(s) in %s; the "
        "model is running without activation checkpointing. Expected submodule "
        "names may not match this architecture, or the layers are already wrapped.",
        activation_checkpoint.capitalize(),
        layer_count,
        ", ".join(container.path for container in containers),
    )


def _report_wrapped(
    wrapped_count: int,
    activation_checkpoint: str,
    containers: list[_LayerContainerInfo],
) -> None:
    """Warn when a mode wrapped no module, then log what it wrapped."""
    _warn_if_nothing_wrapped(wrapped_count, activation_checkpoint, containers)
    paths = ", ".join(container.path for container in containers)
    if activation_checkpoint == "selective":
        logger.info(
            "Selective activation checkpointing applied to %d layer(s) in: %s",
            wrapped_count,
            paths,
        )
        return
    logger.info(
        "%s activation checkpointing wrapped %d submodule(s) in: %s",
        activation_checkpoint.capitalize(),
        wrapped_count,
        paths,
    )


def _find_checkpoint_wrappers(module: nn.Module, prefix: str = "") -> dict[str, nn.Module]:
    """Find outermost checkpoint wrappers by their relative module paths."""
    if _is_checkpoint_wrapped(module):
        return {prefix: module}

    wrappers = {}
    for child_name, child in module.named_children():
        child_path = f"{prefix}.{child_name}" if prefix else child_name
        wrappers.update(_find_checkpoint_wrappers(child, child_path))
    return wrappers


def _register_forward_prefetch_layers(containers: list[_LayerContainerInfo]) -> None:
    """Register swap prefetch chains within each repeated-block container.

    Each wrapped block is chained to the next wrapped one, across any block a
    per-layer plan leaves off: a block with no next keeps what it saved on the
    device, and the block after the gap is fetched back while the one left off
    runs its backward.
    """
    swap_manager = SwapManager()
    for container_info in containers:
        wrapper_chains = {}
        for block_info in container_info.blocks:
            current_block = getattr(block_info.parent, block_info.child_name, None)
            if not isinstance(current_block, nn.Module):
                continue
            for relative_path, wrapper in _find_checkpoint_wrappers(current_block).items():
                wrapper_chains.setdefault(relative_path, []).append(wrapper)

        wired_modules: list[nn.Module] = []
        for wrappers in wrapper_chains.values():
            for current_wrapper, next_wrapper in zip(wrappers, wrappers[1:]):
                swap_manager.set_forward_prefetch_layer(current_wrapper, next_wrapper)
                wired_modules.extend((current_wrapper, next_wrapper))
        wired_modules = list(dict.fromkeys(wired_modules))  # dedupe, keep order

        # swap layers registered above live in the process-wide SwapManager
        # singleton and are never torn down otherwise, so a long-lived process
        # that rebuilds or discards models leaks SwapGroup entries and hook
        # handles.  Tear them down when the container is collected.  The
        # container must not be a finalizer argument (that would keep it alive);
        # only the wired child modules are captured, and the callback receives
        # them directly because the container is unreachable at that point.
        if wired_modules:
            weakref.finalize(
                container_info.container,
                _teardown_wired_swap_layers,
                wired_modules,
            )


def _wrap_first_existing_attr(
    module: nn.Module,
    attr_names: tuple[str, ...],
    wrapper: Callable,
    *,
    skip: bool = False,
) -> int:
    """Checkpoint-wrap the first registered child matching ``attr_names``."""
    if skip:
        return 0

    for attr in attr_names:
        child = getattr(module, attr, None)
        if not isinstance(child, nn.Module):
            continue
        child_name = next(
            (
                name
                for name, registered_child in module.named_children()
                if registered_child is child
            ),
            None,
        )
        if child_name is None:
            continue
        if _is_checkpoint_wrapped(child):
            return 0
        setattr(module, child_name, wrapper(child))
        return 1
    return 0


def _eager_checkpoint_kwargs(swap_inputs: bool) -> dict[str, Any]:
    """Build the eager-mode checkpoint keyword arguments.

    Compile mode drops ``swap_inputs`` entirely, but eager checkpointing accepts
    it in both states: ``False`` keeps ordinary recomputation, while ``True``
    offloads the checkpoint inputs to host memory.
    """
    return {"swap_inputs": swap_inputs}


def apply_submodule_checkpointing(
    layers: list[nn.Module],
    has_kv_sharing: bool,
    enable_compile: bool = False,
    swap_inputs: bool = False,
) -> int:
    """Apply full activation checkpointing to transformer-layer submodules.

    Args:
        layers: Transformer layers whose selected submodules should be wrapped.
        has_kv_sharing: Whether attention submodules must remain outside the
            recomputation regions.
        enable_compile: Whether the wrapped regions will be compiled.
        swap_inputs: Whether checkpoint inputs should be offloaded in eager
            execution. This is ignored in compile mode.
    """
    checkpoint_kwargs = {} if enable_compile else _eager_checkpoint_kwargs(swap_inputs)

    def submodule_checkpoint_wrapper(module: nn.Module) -> nn.Module:
        """Wrap one selected layer submodule with checkpointing."""
        return checkpoint_wrapper(module, **checkpoint_kwargs)

    wrapped_count = 0
    for layer in layers:
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("mlp", "feed_forward", "ffn"),
            submodule_checkpoint_wrapper,
        )
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("self_attn", "attention", "attn", "linear_attn"),
            submodule_checkpoint_wrapper,
            skip=has_kv_sharing,
        )
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("input_layernorm", "attention_norm", "layer_norm1", "norm1"),
            submodule_checkpoint_wrapper,
        )
        wrapped_count += _wrap_first_existing_attr(
            layer,
            ("post_attention_layernorm", "ffn_norm", "layer_norm2", "norm2"),
            submodule_checkpoint_wrapper,
        )
        for attr in (
            "mlp_moe_gen",
            "input_layernorm_moe_gen",
            "post_attention_layernorm_moe_gen",
        ):
            child = getattr(layer, attr, None)
            if isinstance(child, nn.Module) and not _is_checkpoint_wrapped(child):
                setattr(layer, attr, submodule_checkpoint_wrapper(child))
                wrapped_count += 1

    logger.info(
        "Applied submodule activation checkpointing to %d layer(s), wrapping %d submodule(s)",
        len(layers),
        wrapped_count,
    )
    return wrapped_count


def _detect_kv_sharing_and_maybe_disable_cache(model: nn.Module) -> bool:
    """Detect cross-layer KV sharing and disable ordinary model caches."""
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None) or config
    has_kv_sharing = getattr(text_config, "num_kv_shared_layers", 0) > 0
    if has_kv_sharing or config is None:
        return has_kv_sharing

    sub_config_names = getattr(type(config), "sub_configs", None) or {
        "text_config": None,
    }
    sub_configs = (getattr(config, name, None) for name in sub_config_names)
    for sub_config in (config, *sub_configs):
        if sub_config is None or (
            sub_config is not config and not hasattr(sub_config, "use_cache")
        ):
            continue
        if getattr(sub_config, "use_cache", None) is not False:
            _try_disable_use_cache(sub_config)
    return False


def _try_disable_use_cache(sub_config: Any) -> None:
    """Best-effort disable of ``use_cache`` on one config object."""
    try:
        sub_config.use_cache = False
    except Exception:  # pylint: disable=broad-exception-caught
        # Configuration objects may reject assignment with custom errors.
        pass


def _parse_layer_range(key: Union[int, str]) -> tuple[int, int]:
    """Read one plan key: a block index, or an inclusive range ``"first-last"``."""
    if isinstance(key, int) and not isinstance(key, bool):
        if key < 0:
            raise ValueError(
                f"activation_checkpoint.layers keys must not be negative, but got {key}"
            )
        return key, key
    match = _LAYER_RANGE_PATTERN.fullmatch(key) if isinstance(key, str) else None
    if match is None:
        raise ValueError(
            "activation_checkpoint.layers keys must be a layer index or a range "
            f"'first-last', but got {key!r}"
        )
    first = int(match.group(1))
    last = first if match.group(2) is None else int(match.group(2))
    if last < first:
        raise ValueError(
            f"activation_checkpoint.layers range {key!r} ends before it starts"
        )
    return first, last


def _parse_layer_mode(key: Union[int, str], value: Any) -> str:
    """Read the mode one plan entry gives its blocks."""
    # PyYAML reads an unquoted ``off`` as False, as the trainer's literal
    # fields also accept.
    if value is False:
        return "off"
    if isinstance(value, str) and value in _LAYER_MODES:
        return value
    raise ValueError(
        f"activation_checkpoint.layers[{key!r}] must be one of {_LAYER_MODES}, "
        f"but got {value!r}"
    )


def _parse_layer_plan(
    activation_checkpoint: Optional[str],
    layers: Optional[Mapping[Union[int, str], Any]],
) -> tuple[_LayerModeRange, ...]:
    """Read a per-layer plan into ranges ordered by their first block.

    Raises:
        ValueError: The plan is malformed, two of its entries overlap, or the
            mode every other block runs is not one this module knows.
    """
    if not layers:
        return ()
    if not isinstance(layers, Mapping):
        raise ValueError(
            "activation_checkpoint.layers must be a mapping, but got "
            f"{type(layers).__name__}"
        )
    if activation_checkpoint not in (None, *_LAYER_MODES):
        raise ValueError(
            "activation_checkpoint.layers gives some layers another mode than "
            f"activation_checkpoint.mode, which must be one of {_LAYER_MODES} or None, "
            f"but got {activation_checkpoint!r}"
        )
    plan = sorted(
        (
            _LayerModeRange(str(key), *_parse_layer_range(key), _parse_layer_mode(key, value))
            for key, value in layers.items()
        ),
        key=lambda item: item.first,
    )
    for previous, current in zip(plan, plan[1:]):
        if current.first <= previous.last:
            raise ValueError(
                f"activation_checkpoint.layers entries {previous.key!r} and "
                f"{current.key!r} overlap"
            )
    return tuple(plan)


def normalize_activation_checkpoint_layers(
    activation_checkpoint: Optional[str],
    layers: Optional[Mapping[Union[int, str], Any]],
) -> Optional[dict[str, str]]:
    """Validate a per-layer activation checkpoint plan and write it one way.

    The plan gives some transformer blocks another mode than
    ``activation_checkpoint``, which every block it does not name runs, ``None``
    counting as ``"off"``. Each key is a block's index in its repeated-block
    container, or an inclusive range of indices written ``"first-last"``; each
    value is ``"off"``, ``"full"`` or ``"selective"``. Whether the indices exist
    is checked when the plan is applied to a model.

    Args:
        activation_checkpoint: The mode of every block the plan does not name.
        layers: The plan, or ``None`` or an empty mapping for none.

    Returns:
        The plan with string keys in layer order and string modes, or ``None``
        when there is no plan.

    Raises:
        ValueError: The plan is malformed, two of its entries overlap, or
            ``activation_checkpoint`` is not a mode.

    Example:
        normalize_activation_checkpoint_layers("full", {"6-7": False})
        # {"6-7": "off"}
        normalize_activation_checkpoint_layers("off", {"0-2": "full"})
        # {"0-2": "full"}: blocks 0 to 2 are recomputed, the others are not
    """
    plan = _parse_layer_plan(activation_checkpoint, layers)
    return {item.key: item.mode for item in plan} or None


def activation_checkpoint_recomputes(
    activation_checkpoint: Optional[str],
    layers: Optional[Mapping[Union[int, str], Any]] = None,
) -> bool:
    """Whether a mode, with its per-layer plan, recomputes any block.

    A mode that recomputes counts even when the plan leaves every block off,
    because whether the plan names every block is only known on a model.

    Args:
        activation_checkpoint: The mode of every block the plan does not name.
        layers: ``activation_checkpoint.layers``, or ``None``.

    Returns:
        True when ``activation_checkpoint`` is ``"full"`` or ``"selective"``, or
        when the plan runs a block in one of them.

    Raises:
        ValueError: As :func:`normalize_activation_checkpoint_layers`.
    """
    plan = _parse_layer_plan(activation_checkpoint, layers)
    return activation_checkpoint not in (None, "off") or any(item.mode != "off" for item in plan)


def _validate_activation_checkpoint_config(
    activation_checkpoint: Optional[str],
    swap_inputs: bool,
    enable_compile: bool,
) -> None:
    """Validate the activation-checkpoint options and warn about ignored ones.

    Args:
        activation_checkpoint: Requested recomputation mode.
        swap_inputs: Whether checkpoint inputs should be offloaded.
        enable_compile: Whether wrapped regions will be compiled.

    Raises:
        ValueError: The mode or ``swap_inputs`` has an unsupported value.
    """
    if activation_checkpoint not in (None, "off", "full", "selective"):
        raise ValueError(
            "activation_checkpoint.mode must be off, full or selective, but got "
            f"{activation_checkpoint!r}"
        )
    if not isinstance(swap_inputs, bool):
        raise ValueError(
            "activation_checkpoint.swap_inputs must be bool, but got "
            f"{type(swap_inputs).__name__}"
        )

    if swap_inputs and enable_compile:
        logger.warning(
            "activation_checkpoint.swap_inputs is not supported with torch.compile; "
            "input swapping will be disabled."
        )


def _normalize_layer_recompute_ranges(
    layer_ranges: Optional[Sequence[Mapping[str, Any]]],
) -> tuple[_LayerRecomputeRange, ...]:
    """Validate and normalize consecutive per-layer mode overrides."""
    if layer_ranges is None:
        return ()
    if isinstance(layer_ranges, (str, bytes)) or not isinstance(layer_ranges, Sequence):
        raise ValueError("activation_checkpoint.layer_ranges must be a sequence of mappings")

    normalized = []
    previous_end = 0
    for index, raw_range in enumerate(layer_ranges):
        if not isinstance(raw_range, Mapping):
            raise ValueError(f"activation_checkpoint.layer_ranges[{index}] must be a mapping")
        unknown = sorted(set(raw_range) - {"first", "count", "mode"})
        if unknown:
            raise ValueError(
                f"activation_checkpoint.layer_ranges[{index}] has unknown fields {unknown}"
            )
        first = raw_range.get("first", 0)
        count = raw_range.get("count")
        mode = raw_range.get("mode", "off")
        if isinstance(first, bool) or not isinstance(first, int) or first < previous_end:
            raise ValueError(
                "activation_checkpoint.layer_ranges must be ordered, non-overlapping, "
                "and start at non-negative integer indices"
            )
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, int) or count < 1
        ):
            raise ValueError(
                f"activation_checkpoint.layer_ranges[{index}].count must be a positive integer "
                "or null"
            )
        if mode not in ("off", "full", "selective"):
            raise ValueError(
                f"activation_checkpoint.layer_ranges[{index}].mode must be off, full or selective; "
                f"got {mode!r}"
            )
        if count is None and index != len(layer_ranges) - 1:
            raise ValueError(
                "only the final activation_checkpoint.layer_ranges entry may omit count"
            )
        normalized.append(_LayerRecomputeRange(first, count, mode))
        previous_end = first + (count or 0)
    return tuple(normalized)


def _resolve_activation_checkpoint_containers(
    model: nn.Module,
) -> list[_LayerContainerInfo]:
    """Discover the repeated-block containers that activation checkpointing targets.

    Args:
        model: Model to inspect.

    Returns:
        The discovered containers, in registration order.

    Raises:
        ValueError: No container advertises HuggingFace checkpointing support.
    """
    containers = _find_transformer_layer_container_infos(model)
    if not containers:
        raise ValueError(
            f"{type(model).__name__} has no module with a 'gradient_checkpointing' "
            "attribute and a non-empty repeated block container"
        )
    return containers


def _apply_selective_checkpointing(
    containers: list[_LayerContainerInfo],
    ac_layers: list[nn.Module],
    has_kv_sharing: bool,
    *,
    enable_compile: bool,
    swap_inputs: bool,
) -> int:
    """Wrap every discovered layer for selective recomputation.

    KV-shared models fall back to submodule checkpointing so attention does not
    write the cache again during backward recomputation.

    Args:
        containers: The repeated-block containers to wrap.
        ac_layers: Every block selected from ``containers``.
        has_kv_sharing: Whether attention submodules must stay outside the
            recomputation regions.
        enable_compile: Whether the wrapped regions will be compiled.
        swap_inputs: Whether checkpoint inputs should be offloaded in eager
            execution. Ignored in compile mode.

    Returns:
        The number of layers, or submodules in the KV-shared fallback, wrapped.
    """
    if has_kv_sharing:
        logger.warning(
            "Selective activation checkpointing is not supported for KV-shared models; "
            "falling back to submodule activation checkpointing."
        )
        return apply_submodule_checkpointing(
            ac_layers,
            has_kv_sharing,
            enable_compile=enable_compile,
            swap_inputs=swap_inputs,
        )

    if enable_compile:
        def compile_checkpoint_wrapper(layer: nn.Module) -> nn.Module:
            """Wrap one layer with the compile-compatible SAC policy."""
            return checkpoint_wrapper(
                layer,
                policy_fn=compile_selective_checkpoint_policy,
            )

        return _wrap_layer_containers(containers, compile_checkpoint_wrapper)

    def eager_checkpoint_wrapper(layer: nn.Module, **checkpoint_kwargs: Any) -> nn.Module:
        """Wrap one layer with eager selective checkpointing."""
        return checkpoint_wrapper(
            layer,
            swap_inputs=swap_inputs,
            **checkpoint_kwargs,
        )

    return _wrap_layer_containers(
        containers,
        eager_checkpoint_wrapper,
        context_fn=make_selective_checkpoint_context_fn(),
    )


def _mode_for_layer(
    layer_index: int,
    ranges: tuple[_LayerRecomputeRange, ...],
    default_mode: str,
) -> str:
    """Return the layer's override, or the activation-checkpoint default."""
    for layer_range in ranges:
        if layer_index < layer_range.first:
            break
        if layer_range.count is None or layer_index < layer_range.first + layer_range.count:
            return layer_range.mode
    return default_mode


def _apply_layerwise_checkpointing(
    containers: list[_LayerContainerInfo],
    ranges: tuple[_LayerRecomputeRange, ...],
    default_mode: str,
    has_kv_sharing: bool,
    *,
    enable_compile: bool,
    swap_inputs: bool,
) -> int:
    """Wrap transformer layers with their configured checkpoint modes."""
    has_selective_layers = any(
        _mode_for_layer(index, ranges, default_mode) == "selective"
        for index in range(sum(len(container.blocks) for container in containers))
    )
    context_fn = (
        make_selective_checkpoint_context_fn()
        if has_selective_layers and not enable_compile and not has_kv_sharing
        else None
    )
    if has_selective_layers and has_kv_sharing:
        logger.warning(
            "Selective activation checkpointing is not supported for KV-shared models; "
            "selected layers will use submodule activation checkpointing."
        )
    wrapped_count = 0
    layer_index = 0
    for container_info in containers:
        for block_info in container_info.blocks:
            current_block = getattr(block_info.parent, block_info.child_name, None)
            mode = _mode_for_layer(layer_index, ranges, default_mode)
            layer_index += 1
            if not isinstance(current_block, nn.Module) or mode == "off":
                continue
            if _is_checkpoint_wrapped(current_block):
                continue

            if has_kv_sharing:
                wrapped_count += apply_submodule_checkpointing(
                    [current_block],
                    has_kv_sharing=True,
                    enable_compile=enable_compile,
                    swap_inputs=swap_inputs,
                )
                continue

            if mode == "selective":
                if enable_compile:
                    wrapped = checkpoint_wrapper(
                        current_block,
                        policy_fn=compile_selective_checkpoint_policy,
                    )
                else:
                    wrapped = checkpoint_wrapper(
                        current_block,
                        swap_inputs=swap_inputs,
                        context_fn=context_fn,
                    )
            elif enable_compile:
                wrapped = checkpoint_wrapper(current_block)
            else:
                wrapped = checkpoint_wrapper(
                    current_block,
                    **_eager_checkpoint_kwargs(swap_inputs),
                )
            setattr(block_info.parent, block_info.child_name, wrapped)
            wrapped_count += 1
    return wrapped_count


def _apply_full_checkpointing(
    model: nn.Module,
    containers: list[_LayerContainerInfo],
    ac_layers: list[nn.Module],
    has_kv_sharing: bool,
    *,
    enable_compile: bool,
    swap_inputs: bool,
) -> Optional[int]:
    """Wrap discovered layers for full recomputation.

    Prefers the HuggingFace-native implementation when every eligibility check
    passes, and otherwise uses Hyper Parallel's wrappers.

    Args:
        model: Model whose layers should be wrapped.
        containers: The repeated-block containers to wrap.
        ac_layers: Every block selected from ``containers``.
        has_kv_sharing: Whether attention submodules must stay outside the
            recomputation regions.
        enable_compile: Whether the wrapped regions will be compiled.
        swap_inputs: Whether checkpoint inputs should be offloaded in eager
            execution. Ignored in compile mode.

    Returns:
        The number of wrapped submodules, or ``None`` when the HuggingFace-native
        implementation was enabled instead, in which case no per-layer wrapping
        happened and swap prefetch chains must not be registered.
    """
    if _should_use_hf_native_gradient_checkpointing(
        model,
        ac_layers,
        enable_compile=enable_compile,
    ):
        if swap_inputs:
            logger.warning(
                "activation_checkpoint.swap_inputs is not supported by Hugging Face native "
                "gradient checkpointing for now; input swapping will be disabled."
            )
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        logger.info("Using HuggingFace native gradient checkpointing for discovered layers.")
        return None

    if has_kv_sharing:
        return apply_submodule_checkpointing(
            ac_layers,
            has_kv_sharing,
            enable_compile=enable_compile,
            swap_inputs=swap_inputs,
        )

    def full_checkpoint_wrapper(layer: nn.Module) -> nn.Module:
        """Wrap one complete transformer layer for full recomputation."""
        if enable_compile:
            return checkpoint_wrapper(layer)
        return checkpoint_wrapper(layer, **_eager_checkpoint_kwargs(swap_inputs))

    return _wrap_layer_containers(containers, full_checkpoint_wrapper)


def _block_index(block: _TransformerBlockInfo) -> int:
    """Read a block's index from the name its container registers it under."""
    if not block.child_name.isdecimal():
        raise ValueError(
            "activation_checkpoint.layers needs blocks registered under their index, "
            f"but found {block.fqn!r}"
        )
    return int(block.child_name)


def _format_indices(indices: list[int]) -> str:
    """Format block indices as inclusive ranges, such as ``0-5, 8``."""
    ranges: list[list[int]] = []
    for index in sorted(indices):
        if ranges and index == ranges[-1][1] + 1:
            ranges[-1][1] = index
        else:
            ranges.append([index, index])
    return ", ".join(
        str(first) if first == last else f"{first}-{last}" for first, last in ranges
    )


def _block_modes(
    containers: list[_LayerContainerInfo],
    plan: tuple[_LayerModeRange, ...],
    activation_checkpoint: str,
) -> dict[str, str]:
    """Give every discovered block the mode the plan names for it, or the default.

    Returns:
        Each block's mode, by its fully qualified name.

    Raises:
        ValueError: A plan is given for a model with several repeated-block
            containers, whose indices it cannot tell apart, or it names a
            block the container does not hold.
    """
    modes = {
        block.fqn: activation_checkpoint
        for container in containers
        for block in container.blocks
    }
    if not plan:
        return modes
    if len(containers) != 1:
        raise ValueError(
            "activation_checkpoint.layers needs one repeated block container, but "
            f"the model has {len(containers)}: "
            f"{', '.join(container.path for container in containers)}"
        )
    container = containers[0]
    blocks = {_block_index(block): block for block in container.blocks}
    for item in plan:
        for index in range(item.first, item.last + 1):
            if index not in blocks:
                raise ValueError(
                    f"activation_checkpoint.layers[{item.key!r}] names layer {index}, "
                    f"but {container.path} holds layers {_format_indices(list(blocks))}"
                )
            modes[blocks[index].fqn] = item.mode
    return modes


def _describe_block_modes(container: _LayerContainerInfo, modes: dict[str, str]) -> str:
    """Name each mode's blocks as index ranges, such as ``full 0-5; off 6-7``."""
    indices_by_mode: dict[str, list[int]] = {}
    for block in container.blocks:
        indices_by_mode.setdefault(modes[block.fqn], []).append(_block_index(block))
    return "; ".join(
        f"{mode} {_format_indices(indices)}" for mode, indices in indices_by_mode.items()
    )


def _containers_running(
    containers: list[_LayerContainerInfo],
    modes: dict[str, str],
    activation_checkpoint: str,
) -> list[_LayerContainerInfo]:
    """Keep each container's blocks that run one mode, dropping emptied containers."""
    selected = []
    for container in containers:
        blocks = tuple(
            block for block in container.blocks
            if modes[block.fqn] == activation_checkpoint
        )
        if blocks:
            selected.append(
                _LayerContainerInfo(container=container.container, path=container.path, blocks=blocks)
            )
    return selected


def _clear_hf_checkpointing(
    containers: list[_LayerContainerInfo],
    modes: dict[str, str],
) -> None:
    """Turn HuggingFace-native checkpointing off on every block not fully recomputed.

    Enabling it sets the flag on every ``GradientCheckpointingLayer`` of the
    model, and each such block reads its own flag when it is called.
    """
    for container in containers:
        for block in container.blocks:
            if modes[block.fqn] != "full":
                block.module.gradient_checkpointing = False


def _apply_block_modes(
    model: nn.Module,
    containers: list[_LayerContainerInfo],
    modes: dict[str, str],
    has_kv_sharing: bool,
    *,
    enable_compile: bool,
    swap_inputs: bool,
) -> bool:
    """Recompute every block in its mode, the fully recomputed blocks first.

    The fully recomputed blocks go first because the HuggingFace-native path
    enables checkpointing on every block at once, before it is cleared on the
    others; a block left ``"off"`` is not wrapped at all.

    Returns:
        Whether Hyper Parallel's own wrappers wrapped any block, whose swap
        prefetch chains then need registering.
    """
    wrapped_itself = False
    full = _containers_running(containers, modes, "full")
    if full:
        wrapped_count = _apply_full_checkpointing(
            model,
            full,
            _flatten_layer_container_infos(full),
            has_kv_sharing,
            enable_compile=enable_compile,
            swap_inputs=swap_inputs,
        )
        if wrapped_count is None:
            _clear_hf_checkpointing(containers, modes)
        else:
            _report_wrapped(wrapped_count, "full", full)
            wrapped_itself = True
    selective = _containers_running(containers, modes, "selective")
    if selective:
        wrapped_count = _apply_selective_checkpointing(
            selective,
            _flatten_layer_container_infos(selective),
            has_kv_sharing,
            enable_compile=enable_compile,
            swap_inputs=swap_inputs,
        )
        _report_wrapped(wrapped_count, "selective", selective)
        wrapped_itself = True
    return wrapped_itself


def _apply_layer_ranges(
    model: nn.Module,
    activation_checkpoint: Optional[str],
    ranges: tuple[_LayerRecomputeRange, ...],
    *,
    enable_compile: bool,
    swap_inputs: bool,
) -> nn.Module:
    """Recompute each layer in the mode its range gives it, ``activation_checkpoint`` elsewhere.

    Every recomputed layer is wrapped in Hyper Parallel's own checkpoint
    wrapper, whatever its mode.

    Raises:
        ValueError: A range names a layer outside the discovered model.
    """
    containers = _resolve_activation_checkpoint_containers(model)
    ac_layers = _flatten_layer_container_infos(containers)
    for layer_range in ranges:
        end = (
            len(ac_layers)
            if layer_range.count is None
            else layer_range.first + layer_range.count
        )
        if layer_range.first >= len(ac_layers) or end > len(ac_layers):
            raise ValueError(
                "activation_checkpoint.layer_ranges references layers outside the discovered model "
                f"range [0, {len(ac_layers)}): first={layer_range.first}, count={layer_range.count}"
            )
    has_kv_sharing = _detect_kv_sharing_and_maybe_disable_cache(model)

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    default_mode = activation_checkpoint or "off"
    wrapped_count = _apply_layerwise_checkpointing(
        containers,
        ranges,
        default_mode,
        has_kv_sharing,
        enable_compile=enable_compile,
        swap_inputs=swap_inputs,
    )
    has_active_range = any(
        _mode_for_layer(index, ranges, default_mode) != "off"
        for index in range(len(ac_layers))
    )
    if has_active_range:
        _warn_if_nothing_wrapped(wrapped_count, "layerwise", containers)
    logger.info(
        "Layerwise activation checkpointing wrapped %d module(s) in: %s",
        wrapped_count,
        ", ".join(container.path for container in containers),
    )
    if swap_inputs and not enable_compile:
        _register_forward_prefetch_layers(containers)
    return model


def _apply_activation_checkpointing(
    model: nn.Module,
    activation_checkpoint: Optional[str],
    enable_compile: bool = False,
    swap_inputs: bool = False,
    layer_ranges: Optional[Sequence[Mapping[str, Any]]] = None,
    layers: Optional[Mapping[Union[int, str], Any]] = None,
) -> nn.Module:
    """Apply full, selective or layer-specific recomputation to transformer layers.

    Validates the requested options, discovers the repeated-block containers that
    advertise HuggingFace checkpointing support, then delegates to the
    mode-specific wrapper. Swap prefetch chains are registered afterwards for
    every configuration that wrapped the layers itself.

    ``layers`` gives some blocks another mode than ``activation_checkpoint``,
    which every other block runs: see
    :func:`normalize_activation_checkpoint_layers`. With a plan,
    ``activation_checkpoint`` may be ``"off"`` or ``None``, and then only the
    blocks the plan runs full or selective are recomputed. ``layer_ranges``
    gives layers their modes as ranges instead, and wraps every recomputed one
    in Hyper Parallel's own checkpoint wrapper (:func:`_apply_layer_ranges`),
    where ``layers`` keeps HuggingFace's native checkpointing for the fully
    recomputed blocks.

    Raises:
        ValueError: An option is invalid, or ``layers`` and ``layer_ranges``
            are both given.
    """
    normalized_ranges = _normalize_layer_recompute_ranges(layer_ranges)
    plan = _parse_layer_plan(activation_checkpoint, layers)
    if layer_ranges is not None and plan:
        raise ValueError(
            "activation_checkpoint.layers and activation_checkpoint.layer_ranges both give "
            "layers a mode of their own: state one of them"
        )
    _validate_activation_checkpoint_config(
        activation_checkpoint,
        swap_inputs,
        enable_compile,
    )
    if layer_ranges is not None:
        return _apply_layer_ranges(
            model,
            activation_checkpoint,
            normalized_ranges,
            enable_compile=enable_compile,
            swap_inputs=swap_inputs,
        )
    if activation_checkpoint in (None, "off") and not plan:
        return model

    containers = _resolve_activation_checkpoint_containers(model)
    modes = _block_modes(containers, plan, activation_checkpoint or "off")
    has_kv_sharing = _detect_kv_sharing_and_maybe_disable_cache(model)

    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    if plan:
        logger.info(
            "Activation checkpointing per layer in %s: %s",
            containers[0].path,
            _describe_block_modes(containers[0], modes),
        )
    wrapped_itself = _apply_block_modes(
        model,
        containers,
        modes,
        has_kv_sharing,
        enable_compile=enable_compile,
        swap_inputs=swap_inputs,
    )
    # The HuggingFace-native path wraps nothing itself, so it registers no
    # swap prefetch chain.
    if swap_inputs and not enable_compile and wrapped_itself:
        _register_forward_prefetch_layers(containers)
    return model
