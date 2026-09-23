# Copyright 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
"""Common utility functions.

What every part of a distributed checkpoint reaches for and no one part owns: the paths,
the shard geometry, the state dict walk, and the one logger. Checkpoint files are read and
written in :mod:`checkpoint_io`; replicated shards are sent in :mod:`broadcast`.
"""

import time
from functools import wraps
from typing import Any, Optional, Union
from collections.abc import Callable, Collection, Mapping

import torch
import torch.distributed as dist
from torch import Tensor

from hyper_parallel.core.distributed_checkpoint.metadata import (
    ChunkStorageMetadata,
    MetadataIndex,
    CHUNK_INFO,
    ChunkInfo,
)
from hyper_parallel.core.distributed_checkpoint.planner import SavePlan
from hyper_parallel.core.distributed_checkpoint.ragged import compute_ragged_boxes
from hyper_parallel.core.dtensor.layout import infer_slice_area_by_layout
from hyper_parallel.core.dtensor.dtensor import DTensor
from hyper_parallel.tools.logging import get_logger

# The one DCP logger: other distributed_checkpoint modules import this instead of
# registering a component of their own.
logger = get_logger("DCP")


def get_tensor_storage_size(tensor: torch.Tensor) -> int:
    """
    The serialized byte size of one tensor.

    Args:
        tensor (torch.Tensor): The tensor to measure.

    Returns:
        int: Element count times element size, which is what it occupies on disk.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"get_tensor_storage_size expects torch.Tensor, got {type(tensor)!r}")
    return int(tensor.numel()) * int(tensor.element_size())


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """
    Map a ``torch.<type>`` string from checkpoint metadata back to a ``torch.dtype``.

    Args:
        dtype_str (str): The dtype as metadata records it, such as ``torch.float32``.

    Returns:
        torch.dtype: The dtype it names.
    """
    parts = dtype_str.split(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Expected dtype string like 'torch.float32', got {dtype_str!r}.")
    prefix, name = parts
    if prefix != "torch":
        raise ValueError(f"Expected PyTorch dtype string with prefix 'torch', got {dtype_str!r}.")
    dtype = getattr(torch, name, None)
    if isinstance(dtype, torch.dtype):
        return dtype
    raise ValueError(f"{dtype_str!r} does not resolve to a torch.dtype.")


def dcp_timer_decorator(func: Callable) -> Callable:
    """
    Used to collect statistics on the time consumed in each phase of the DCP.

    The timings are per-rank, so the rank is part of the message; enable them
    with ``HP_LOG_CONFIG=DCP:INFO``.

    Args:
        func (Callable): The phase to time.

    Returns:
        Callable: ``func`` wrapped so that each call logs how long it took.
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Log how long one call of the wrapped phase took, and return its result."""
        try:
            rank_id = dist.get_rank()
        except ValueError:
            # No process group yet (offline converters, single-process tools).
            rank_id = 0
        logger.info("[rank=%d] >>> func %s start exec", rank_id, func.__name__)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info("[rank=%d] >>> func %s cost %.4f seconds", rank_id, func.__name__, execution_time)
        return result

    return wrapper


def narrow_tensor_by_index(tensor: Any, offsets: tuple, lengths: tuple) -> Any:
    """
    Narrow the tensor by (offsets, lengths) per dimension.

    Used for resharding operations to extract a slice from a tensor.
    Uses plain slice indexing so any tensor-like object works.

    Args:
        tensor (Any): The tensor to narrow (tensor-like object supporting indexing).
        offsets (tuple): Tuple of offsets per dimension.
        lengths (tuple): Tuple of lengths per dimension.

    Returns:
        Any: The narrowed tensor slice (tensor-like object).
    """
    if not offsets or not lengths:
        return tensor
    slices = tuple(
        slice(int(off), int(off) + int(ln))
        for off, ln in zip(offsets, lengths)
    )
    return tensor[slices]


def chunk_to_area(chunk: ChunkStorageMetadata) -> tuple[tuple[int, int], ...]:
    """
    Convert ChunkStorageMetadata to (start, end) area per dimension.

    Args:
        chunk (ChunkStorageMetadata): ChunkStorageMetadata instance with offsets and sizes.

    Returns:
        tuple[tuple[int, int], ...]: Tuple of (start, end) tuples for each dimension.
    """
    return tuple(
        (chunk.offsets[i], chunk.offsets[i] + chunk.sizes[i])
        for i in range(len(chunk.offsets))
    )


def infer_intersection(
        area_a: tuple[tuple[int, int], ...],
        area_b: tuple[tuple[int, int], ...]
) -> Optional[tuple[tuple[int, int], ...]]:
    """
    Calculates the intersection of two tensor slice areas.

    Args:
        area_a (tuple[tuple[int, int], ...]): First area to intersect.
        area_b (tuple[tuple[int, int], ...]): Second area to intersect.

    Returns:
        Optional[tuple[tuple[int, int], ...]]: Tuple of intersection boundaries or None if no intersection.
    """
    # Validate input formats
    def is_valid_axis_list(axis_list: Any) -> None:
        """Reject an area that is not a sequence of two-element ranges."""
        if not isinstance(axis_list, (tuple, list)):
            raise TypeError("Area must be a tuple of ranges")
        for axis_range in axis_list:
            if (not isinstance(axis_range, (tuple, list)) \
                or len(axis_range) != 2):
                raise TypeError("Each axis range must be a 2-element tuple")

    is_valid_axis_list(area_a)
    is_valid_axis_list(area_b)

    # Check dimension compatibility
    if len(area_a) != len(area_b):
        raise ValueError(
            f"Area dimension mismatch: {len(area_a)} vs {len(area_b)}"
        )

    # Calculate intersection for each dimension
    intersection: list[tuple[int, int]] = []
    for axis_range_a, axis_range_b in zip(area_a, area_b):
        left = max(axis_range_a[0], axis_range_b[0])
        right = min(axis_range_a[1], axis_range_b[1])

        if left >= right:  # No intersection in this dimension
            return None

        intersection.append((left, right))

    return tuple(intersection)


def create_chunk_list_for_tensor(obj: Union[Tensor, DTensor]) -> list[ChunkStorageMetadata]:
    """
    Create list of local chunks for the given object (DTensor or plain tensor).

    Used to determine what this rank needs to load (resharding).

    Args:
        obj (Union[Tensor, DTensor]): hyper DTensor or torch Tensor.

    Returns:
        list[ChunkStorageMetadata]: List of ChunkStorageMetadata representing
            local chunks needed by this rank.
    """
    if isinstance(obj, DTensor):
        layout = obj.layout
        if layout is None:
            shape = obj.shape if hasattr(obj, "shape") else obj.to_local().shape
            return [ChunkStorageMetadata(offsets=(0,) * len(shape), sizes=tuple(shape))]
        if layout.ragged_shard is not None:
            return [
                ChunkStorageMetadata(offsets=box.offsets, sizes=box.sizes)
                for box in compute_ragged_boxes(obj)
            ]

        mesh_shape = getattr(layout, "mesh_shape", None) or getattr(layout, "_mesh", None)
        tensor_map = getattr(layout, "tensor_map", None) or getattr(layout, "_tensor_map", None)
        rank_list = getattr(layout, "rank_list", None) or getattr(layout, "_rank_list", None)

        if mesh_shape is None or tensor_map is None or rank_list is None:
            shape = obj.shape if hasattr(obj, "shape") else obj.to_local().shape
            return [ChunkStorageMetadata(offsets=(0,) * len(shape), sizes=tuple(shape))]

        current_rank = dist.get_rank()
        if current_rank not in rank_list:
            return []

        inner_rank_id = rank_list.index(current_rank)
        full_shape = obj.shape
        slice_area = infer_slice_area_by_layout(
            layout,
            inner_rank_id,
            full_shape,
        )
        offsets = tuple(s for s, _ in slice_area)
        sizes = tuple(e - s for s, e in slice_area)
        return [ChunkStorageMetadata(offsets=offsets, sizes=sizes)]

    if isinstance(obj, Tensor):
        # handle Tensor with shard information
        if hasattr(obj, CHUNK_INFO):
            if not isinstance(getattr(obj, CHUNK_INFO), ChunkInfo):
                raise ValueError("The attr CHUNK_INFO should be a ChunkInfo instance")
            chunk = getattr(obj, CHUNK_INFO).chunk
            return [chunk]
        # torch.Tensor has exactly one chunk in metadata (full tensor)
        shape = tuple(obj.shape)
        return [ChunkStorageMetadata(offsets=(0,) * len(shape), sizes=shape)]

    raise ValueError(f"Not support type {type(obj)} for creating chunk list ")


def plan_ownership_masks(
    all_plans: list[SavePlan],
    save_to_minimum_rank: bool = False,
) -> list[bytearray]:
    """
    Decide which plan writes each item, as one keep-mask per plan.

    An item present in several plans is redundant: only one plan should write it. The owner is
    the plan with the smallest planned storage so far, or the lowest plan index when
    ``save_to_minimum_rank`` is True. Ownership is resolved from the plan order and the item
    sizes alone, so every rank running this over the same gathered plans reaches the same answer.

    Duplicates are assigned largest first (longest-processing-time): placing the big shards while
    the plans are still evenly loaded leaves the small ones to even out the remainder. Assigning
    in arrival order instead lets a late big shard land on an already-heavy plan, and the
    checkpoint's wall time is set by whichever plan writes the most.

    Masks are returned instead of filtered plans because the caller walks ``plan.items`` anyway:
    skipping on a mask costs no hashing and allocates no intermediate copy of every plan.

    Args:
        all_plans (list[SavePlan]): Local plans gathered from all ranks, indexed by rank.
        save_to_minimum_rank (bool): If True, assign duplicates to the lowest plan index; else to
            the plan holding the least data so far. Default False.

    Returns:
        list[bytearray]: One mask per plan, parallel to that plan's ``items``: 1 marks an item the
            plan owns and must write, 0 marks a duplicate another plan took.
    """
    # index -> [write_item, plan_idx, position, plan_idx, position, ...]. One flat list per
    # distinct item, so the common unique item costs a single dict lookup rather than an entry
    # in a duplicate map plus one in a registry plus one in a per-plan set.
    occurrences_by_index: dict[MetadataIndex, list] = {}
    for plan_idx, plan in enumerate(all_plans):
        for position, entry in enumerate(plan.items):
            occurrences = occurrences_by_index.get(entry.index)
            if occurrences is None:
                occurrences_by_index[entry.index] = [entry, plan_idx, position]
            else:
                occurrences.append(plan_idx)
                occurrences.append(position)

    masks = [bytearray(len(plan.items)) for plan in all_plans]
    storage_sizes = [0] * len(all_plans)

    # Unique items are assigned first so that they are all accounted for in storage_sizes before
    # duplicates are balanced against those sizes. Sizes are computed here, once per item.
    duplicates: list[tuple[int, list]] = []
    for occurrences in occurrences_by_index.values():
        item_size = occurrences[0].tensor_storage_size() or 1
        # The layout is [write_item] + (plan_idx, position) * holder_count: drop the slot the
        # item itself takes, then every two remaining slots are one plan holding it.
        holder_count = (len(occurrences) - 1) // 2
        if holder_count > 1:
            duplicates.append((item_size, occurrences))
            continue
        masks[occurrences[1]][occurrences[2]] = 1
        storage_sizes[occurrences[1]] += item_size

    # Largest first, so the shards with room to unbalance the plans are placed while every plan is
    # still a candidate. Python's sort is stable, including with reverse=True, so equally sized
    # duplicates keep their gather order and every rank still agrees on the owner. Sorting is
    # pointless when every duplicate goes to its lowest plan index regardless.
    if not save_to_minimum_rank:
        duplicates.sort(key=lambda pair: pair[0], reverse=True)

    for item_size, occurrences in duplicates:
        # Occurrences were appended in ascending plan order, so slot 1 is the lowest plan index
        # and the storage-size search breaks ties towards it.
        if save_to_minimum_rank:
            owner_slot = 1
        else:
            owner_slot = _least_loaded_owner_slot(occurrences, storage_sizes)
        owner_idx = occurrences[owner_slot]
        masks[owner_idx][occurrences[owner_slot + 1]] = 1
        storage_sizes[owner_idx] += item_size

    return masks


def _least_loaded_owner_slot(occurrences: list, storage_sizes: list[int]) -> int:
    """
    Pick the occurrence slot whose plan currently holds the least data.

    Args:
        occurrences (list): ``[write_item, plan_idx, position, ...]`` for one duplicated item.
        storage_sizes (list[int]): Bytes already assigned to each plan.

    Returns:
        int: Index into ``occurrences`` of the winning ``plan_idx`` (its position follows it).
    """
    # Slot 1 is the first plan_idx and every further holder sits two slots on. Comparing with a
    # strict < keeps the first minimum, which is the lowest plan index.
    best_slot = 1
    best_size = storage_sizes[occurrences[1]]
    for slot in range(3, len(occurrences), 2):
        size = storage_sizes[occurrences[slot]]
        if size < best_size:
            best_slot, best_size = slot, size
    return best_slot


def traverse_state_dict(
    state_dict: Any,
    visitor: Any,
) -> None:
    """
    Invoke ``visitor`` for each value recursively in ``state_dict``.
    Mapping will be traversed and ``visitor`` will be applied to the leaf elements.
    ``visitor`` will only be applied to elements in a list or a tuple, if the
    container contains tensors or mappings.
    """

    def _is_terminal(value: Any) -> bool:
        """Leaf-like container: no nested mappings/lists/tuples/tensors to recurse into."""
        values: Collection
        if isinstance(value, Mapping):
            return False
        if isinstance(value, (list, tuple)):
            values = value
        else:
            return True

        for entry in values:
            if isinstance(entry, (Mapping, list, tuple)) and not _is_terminal(entry):
                return False
            if isinstance(entry, Tensor):
                return False
        return True

    def _traverse_obj(path: tuple[Any, ...], value: Any) -> None:
        if isinstance(value, Mapping):
            for k, v in value.items():
                _traverse_obj(path + (str(k),), v)
        elif _is_terminal(value):
            visitor(path, value)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _traverse_obj(path + (i,), v)

    for key, value in state_dict.items():
        _traverse_obj((str(key),), value)


def flatten_state_dict(state_dict: Any) -> tuple[dict[str, Any], dict[str, tuple[Any, ...]]]:
    """Flatten a nested state dict to dotted FQN keys; returns ``(flat_dict, fqn -> path)``."""
    fqn_names: dict[str, Any] = {}
    mappings: dict[str, tuple[Any, ...]] = {}

    def flat_copy(path: tuple[Any, ...], value: Any) -> None:
        """Record one leaf under its dotted FQN, refusing a name already taken."""
        new_fqn = ".".join(map(str, path))
        if new_fqn in fqn_names:
            raise ValueError(
                f"Duplicate flattened FQN {new_fqn!r} when converting nested state_dict; "
                "two different values map to the same dotted name."
            )
        fqn_names[new_fqn] = value
        mappings[new_fqn] = path

    traverse_state_dict(state_dict, flat_copy)
    return fqn_names, mappings


def set_element(root_dict: Any, path: tuple[Any, ...], value: Any) -> None:
    """Set ``value`` in ``root_dict`` along the ``path`` object path."""
    if not path:
        raise ValueError("path must be non-empty")
    cur_container: Any = root_dict

    def extend_list(lst: list[Any], idx: int) -> None:
        """Pad ``lst`` with None until ``idx`` is a valid position in it."""
        while len(lst) <= idx:
            lst.append(None)

    for i in range(1, len(path)):
        prev_key = path[i - 1]
        next_key = path[i]
        def_val: Any = {} if isinstance(next_key, str) else []

        if isinstance(cur_container, Mapping):
            cur_container = cur_container.setdefault(prev_key, def_val)
        else:
            extend_list(cur_container, prev_key)
            if cur_container[prev_key] is None:
                cur_container[prev_key] = def_val
            cur_container = cur_container[prev_key]

    last_key = path[-1]
    if isinstance(last_key, int):
        extend_list(cur_container, last_key)

    cur_container[last_key] = value


@dcp_timer_decorator
def all_gather_object(
    local_object: Any,
    world_size: int,
    use_collectives: bool,
) -> list[Any]:
    """
    Gather objects from all ranks.

    Args:
        local_object (Any): Local object for current rank.
        world_size (int): Total number of ranks.
        use_collectives (bool): Whether to use collective communication.

    Returns:
        list[Any]: List of all objects from all ranks.
    """
    if use_collectives and world_size > 1:
        all_objects = [None] * world_size
        dist.all_gather_object(all_objects, local_object)
        return all_objects
    return [local_object]
