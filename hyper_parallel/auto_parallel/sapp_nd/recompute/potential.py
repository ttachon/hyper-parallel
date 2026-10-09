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
"""Sort, retain and export recompute plans at fixed parallel dimensions."""

import csv
import json
import math
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from hyper_parallel.auto_parallel.sapp_nd.nd.dimensions import Dimensions
from hyper_parallel.auto_parallel.sapp_nd.recompute.candidate import MODES, RecomputeChoice, to_records, trainer_plan


def sort_recompute_results(results: Sequence[RecomputeChoice]) -> List[RecomputeChoice]:
    """Return recompute results fastest first, using peak memory to break ties.

    Args:
        results: Scored plans to order, without modifying the input sequence.

    Returns:
        Plans in ascending performance-score order.

    Raises:
        ValueError: A plan has no finite performance score.
    """
    if any(result.score is None or not math.isfinite(result.score) for result in results):
        raise ValueError("recompute results must have finite performance scores before sorting")
    return sorted(results, key=lambda result: (result.score, result.memory))


def _layer_settings(choice: RecomputeChoice) -> dict:
    """Expand a named plan independently of how its consecutive ranges are grouped."""
    defined = {}
    for item in choice.ranges:
        if item.first < 0 or item.count <= 0 or item.mode not in MODES:
            raise ValueError("recompute potentials require valid ranges with a named mode per layer")
        for index in range(item.first, item.first + item.count):
            if index in defined:
                raise ValueError("recompute layer ranges must not overlap")
            defined[index] = (item.kind, item.mode, item.option)
    return defined


def keep_recompute_potential(
    current: RecomputeChoice,
    kept: List[RecomputeChoice],
    results: Sequence[RecomputeChoice],
) -> Optional[RecomputeChoice]:
    """Append and return the fastest unused plan adding exactly one FULL layer.

    Args:
        current: Plan whose parallel degrees and other layers must be preserved.
        kept: Retained plans; a qualifying result is appended in place.
        results: Scored recompute alternatives, which need not be sorted.

    Returns:
        The appended plan, or None when no unused qualifying plan remains.

    Raises:
        ValueError: Scores or per-layer plans are invalid.
    """
    defined = _layer_settings(current)
    previous = [(choice.dimensions, _layer_settings(choice)) for choice in kept]
    for result in sort_recompute_results(results):
        if result.dimensions != current.dimensions:
            continue
        proposed = _layer_settings(result)
        if proposed.keys() != defined.keys():
            continue
        changed = [index for index in defined if defined[index] != proposed[index]]
        if len(changed) != 1:
            continue
        index = changed[0]
        kind, mode, _ = defined[index]
        new_kind, new_mode, new_option = proposed[index]
        if (mode == "full" or new_mode != "full" or new_kind != kind
                or new_option.recompute is not None or new_option.link_bandwidth):
            continue
        if any(dimensions == result.dimensions and layers == proposed for dimensions, layers in previous):
            continue
        kept.append(result)
        return result
    return None


def recompute_next_plot_groups(
    scored_space: Sequence[tuple],
    results: Mapping[Dimensions, Sequence[RecomputeChoice]],
    top_num: Optional[int] = None,
) -> List[Tuple[Dimensions, List[RecomputeChoice]]]:
    """Select plans below each next score and include the next best alternative.

    Args:
        scored_space: Parallel configurations in the search's performance order.
        results: Every scored one-layer alternative, including each baseline.
        top_num: Number of parallel configurations to display; defaults to twenty.

    Returns:
        Groups in global configuration order, each sorted by performance.
        Include every plan below the next strategy's score and the first plan
        at or above it. Keep at least two plans when available, so a group
        includes a recompute alternative even when the scores are tied.
        The last group gets the mean size of the preceding groups, rounded
        to the nearest integer with halves rounded up, with a minimum of two
        and limited by the available plans. With only one configuration,
        it gets up to two plans.
    """
    top = scored_space[:max(0, 20 if top_num is None else top_num)]
    if not top:
        return []
    groups = []
    for entry, following in zip(top, top[1:]):
        config = entry[0]
        choices = sort_recompute_results(results.get(config, ()))
        count = sum(choice.score < following[2] for choice in choices)
        groups.append((config, choices[:max(2, count + 1)]))
    count = math.floor(sum(len(choices) for _, choices in groups) / len(groups) + 0.5) if groups else 2
    count = max(2, count)
    config = top[-1][0]
    groups.append((config, sort_recompute_results(results.get(config, ()))[:count]))
    return groups


def recompute_plot_data(
    groups: Sequence[Tuple[Dimensions, Sequence[RecomputeChoice]]],
) -> List[Tuple[Dimensions, List[dict]]]:
    """Prepare the selected plans as chart records without changing their configuration ranks.

    Args:
        groups: Selected plans grouped by fixed parallel configuration.

    Returns:
        Groups with scores, component values, memory, FULL counts and trainer plans.
    """
    prepared = []
    for config, choices in groups:
        records = []
        for choice in choices:
            mode, layers = trainer_plan(choice) if choice.ranges or choice.mode is not None else (None, None)
            records.append({
                "score": choice.score,
                "parts": choice.score_parts,
                "memory": choice.memory,
                "mode": mode,
                "layers": layers,
                "full_layers": sum(item.count for item in choice.ranges if item.mode == "full")
                if choice.ranges else None,
            })
        prepared.append((config, records))
    return prepared


def write_recompute_csv(potentials: Mapping[Dimensions, Sequence[RecomputeChoice]], path: str) -> None:
    """Replace a CSV with each fixed strategy's retained recompute sequence.

    Args:
        potentials: Retained plans in sequence order, grouped by parallel configuration.
        path: Destination CSV, whose parent directory is created if necessary.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    degrees = list(dict.fromkeys(
        name for choices in potentials.values() for choice in choices for name, _ in choice.dimensions
    ))
    columns = ["TOP", "step"] + degrees + [
        "changed_layer", "full_layers", "memory_mb", "score", "recompute_mode", "recompute_layers",
        "layer_recompute", "stage_memory_mb", "stage_savings", "plan_scope",
    ]
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for config, choices in potentials.items():
            previous = {}
            for step, choice in enumerate(choices):
                defined = _layer_settings(choice)
                changed = [index for index in defined if step and defined[index] != previous[index]]
                mode, layers = trainer_plan(choice) if choice.ranges or choice.mode is not None else ("", {})
                scope = "per_layer" if choice.ranges else "whole_model" if choice.mode is not None else "unavailable"
                if any(item.option.link_bandwidth for item in choice.ranges):
                    scope = "offload"
                row = dict(choice.dimensions)
                row.update({
                    "TOP": config.rank,
                    "step": step,
                    "changed_layer": changed[0] if changed else "",
                    "full_layers": sum(item.count for item in choice.ranges if item.mode == "full")
                    if choice.ranges else "",
                    "memory_mb": choice.memory,
                    "score": choice.score,
                    "recompute_mode": mode,
                    "recompute_layers": json.dumps(layers),
                    "layer_recompute": json.dumps(to_records(choice)),
                    "stage_memory_mb": json.dumps(choice.stage_memory),
                    "stage_savings": json.dumps(choice.stage_savings),
                    "plan_scope": scope,
                })
                writer.writerow(row)
                previous = defined
