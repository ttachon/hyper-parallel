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
"""
Graph Parallel Plan - declarative module selection for graph-mode passes.

Declares which modules the graph-mode ``FSDPPass`` should shard and which
modules each pipeline stage owns. ``FSDPPass`` itself owns all the actual
sharding logic (all_gather on parameter placeholders, reduce_scatter on
gradient outputs, in-place live-model sharding), so this is purely a *which
modules* lookup.

Note:
    Earlier revisions stored an ``FSDPModuleConfig`` dataclass per entry and
    exposed ``get_fsdp_config`` / ``merge``. The dataclass carried no field
    beyond the FQN that already keys it, and neither helper had a production
    consumer. They were removed as dead surface; when a real per-module
    option (reshard, CPU offload) lands it can re-add a value type with a
    real consumer.
"""

__all__ = [
    "GraphParallelPlan",
    "create_plan_from_yaml",
    "create_all_fsdp_plan",
]

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

import yaml

DEFAULT_CONFIG_DIR = Path(__file__).parent / "examples"


@dataclass
class GraphParallelPlan:
    """Declare the graph-mode parallel assignment of modules.

    ``fsdp_modules`` holds exact module FQNs to FSDP-shard and
    ``fsdp_patterns`` holds ``fnmatch`` wildcards; :meth:`is_marked_for_fsdp`
    checks the exact set first, then the patterns.

    Pipeline-parallel stage assignment lives in
    ``pp_module_fqns_per_stage``: a list whose i-th entry is the list of
    module FQNs (exact, no wildcards) assigned to stage ``i``. When it is
    ``None`` the ``PpPass`` falls back to an automatic even split of the
    model's layer-like children (see ``pp_pass._auto_stage_split``).

    Example:
        plan = GraphParallelPlan()
        plan.fsdp_mark("tok_embeddings")
        plan.fsdp_mark_pattern("layers.*")
        plan.pp_stage(0, ["tok_embeddings", "layers.0"])
        plan.pp_stage(1, ["layers.1", "norm", "lm_head"])
    """

    fsdp_modules: Set[str] = field(default_factory=set)
    fsdp_patterns: Set[str] = field(default_factory=set)
    pp_module_fqns_per_stage: Optional[List[List[str]]] = None

    def fsdp_mark(self, module_fqn: str) -> "GraphParallelPlan":
        """Mark a specific module for FSDP sharding (exact match).

        Args:
            module_fqn: Module fully qualified name.

        Returns:
            ``self`` (chainable).

        Example:
            plan.fsdp_mark("tok_embeddings")
        """
        self.fsdp_modules.add(module_fqn)
        return self

    def fsdp_mark_pattern(self, pattern: str) -> "GraphParallelPlan":
        """Mark modules for FSDP sharding (wildcard match).

        Args:
            pattern: Module FQN pattern (``fnmatch`` wildcards, e.g. ``*``,
                ``layers.*``).

        Returns:
            ``self`` (chainable).

        Example:
            plan.fsdp_mark_pattern("layers.*")
        """
        self.fsdp_patterns.add(pattern)
        return self

    def is_marked_for_fsdp(self, module_fqn: str) -> bool:
        """Whether ``module_fqn`` is marked for FSDP sharding.

        Pure exact/pattern lookup: an exact mark or any matching pattern
        returns ``True``. Ancestor matching (a mark on ``layers.0`` covering
        ``layers.0.attention.weight``) is NOT done here — callers such as
        ``FSDPPass`` walk a parameter's ancestors and query each level.
        """
        if module_fqn in self.fsdp_modules:
            return True

        return any(
            fnmatch.fnmatch(module_fqn, pattern) for pattern in self.fsdp_patterns
        )

    def pp_stage(self, stage_idx: int, module_fqns: List[str]) -> "GraphParallelPlan":
        """Declare the module FQNs of one pipeline stage (exact match).

        Args:
            stage_idx: Zero-based stage index, ``stage_idx >= 0``. Stages
                may be declared in any order / sparsely; ``PpPass``
                validates completeness (every stage ``0..pp_degree-1``
                declared, no FQN assigned twice) before splitting.
            module_fqns: Module FQNs owned by this stage, in model order.
                Exact FQNs only — wildcards are rejected because a stage
                cut must be unambiguous.

        Returns:
            ``self`` (chainable).

        Raises:
            ValueError: If ``stage_idx`` is negative or a module FQN
                contains wildcard characters.

        Example:
            plan.pp_stage(0, ["tok_embeddings", "layers.0"])
            plan.pp_stage(1, ["layers.1", "norm", "lm_head"])
        """
        if stage_idx < 0:
            raise ValueError(f"stage_idx must be >= 0, got {stage_idx}")
        for fqn in module_fqns:
            if any(ch in fqn for ch in "*?["):
                raise ValueError(
                    f"pp_stage() takes exact module FQNs, got wildcard pattern "
                    f"'{fqn}' — a stage cut must be unambiguous"
                )
        if self.pp_module_fqns_per_stage is None:
            self.pp_module_fqns_per_stage = []
        while len(self.pp_module_fqns_per_stage) <= stage_idx:
            self.pp_module_fqns_per_stage.append([])
        self.pp_module_fqns_per_stage[stage_idx] = list(module_fqns)
        return self


def create_plan_from_yaml(
    config_path: Optional[str] = None,
    model_name: Optional[str] = None,
) -> GraphParallelPlan:
    """Create a ``GraphParallelPlan`` from a YAML configuration file.

    Args:
        config_path: Path to YAML config file.
        model_name: Model name (looks up in ``examples/{model_name}/config.yaml``).

    Returns:
        GraphParallelPlan object.

    Raises:
        ValueError: When neither argument is given, ``model_name`` is empty
            or contains path separators, or the YAML is not a mapping.
        FileNotFoundError: When the resolved config file does not exist.

    Example:
        plan = create_plan_from_yaml(model_name="llama3")
        plan = create_plan_from_yaml(config_path="path/to/config.yaml")
    """
    if config_path is None and model_name is None:
        raise ValueError("Must provide either config_path or model_name")

    if config_path is None:
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
        if ".." in model_name or "/" in model_name or "\\" in model_name:
            raise ValueError(
                f"Invalid model_name '{model_name}': must not contain path "
                "separators or parent directory references"
            )
        config_path = DEFAULT_CONFIG_DIR / model_name / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"YAML config file is empty: {config_path}")
    if not isinstance(config, dict):
        raise ValueError(
            f"YAML config must be a mapping (dict), "
            f"got {type(config).__name__}: {config_path}"
        )

    plan = GraphParallelPlan()

    fsdp_config = _yaml_section(config, "fsdp", config_path)
    _maybe_process_fsdp(plan, fsdp_config)

    pp_config = _yaml_section(config, "pp", config_path)
    if pp_config.get("stages"):
        _process_pp(plan, pp_config)

    return plan


def _yaml_section(config: dict, key: str, config_path: Path) -> dict:
    """Return YAML ``key`` as a mapping; empty/``None`` becomes ``{}``.

    ``or {}``: a YAML key present but empty (``fsdp:`` with only comments
    under it) parses to None, and ``dict.get(key, {})`` then returns
    None instead of the default.
    """
    section = config.get(key) or {}
    if not isinstance(section, dict):
        # A present-but-empty section parses to None and is normalized to {}
        # above, so anything landing here is a real scalar/sequence typo.
        raise ValueError(
            f"YAML '{key}' section must be a mapping (e.g. nested keys or an "
            f"empty section); got {type(section).__name__} in {config_path}"
        )
    return section


def _maybe_process_fsdp(plan: GraphParallelPlan, fsdp_config: dict) -> None:
    """Run the FSDP processor when the section is enabled.

    Enabled defaults to True when explicit modules/patterns are declared.
    """
    has_explicit = bool(fsdp_config.get("modules")) or bool(fsdp_config.get("patterns"))
    if fsdp_config.get("enabled", has_explicit):
        _process_fsdp(plan, fsdp_config)


def _process_fsdp(plan: GraphParallelPlan, fsdp_config: dict) -> None:
    """Process FSDP configuration (modules + patterns) into the plan."""
    for module_config in fsdp_config.get("modules", []):
        plan.fsdp_mark(module_config["name"])

    for pattern_config in fsdp_config.get("patterns", []):
        plan.fsdp_mark_pattern(pattern_config["pattern"])


def _process_pp(plan: GraphParallelPlan, pp_config: dict) -> None:
    """Process PP configuration (explicit per-stage FQN lists) into the plan.

    YAML shape::

        pp:
          stages:
            - [tok_embeddings, layers.0]
            - [layers.1, norm, lm_head]

    An entry may also be a mapping with ``stage`` / ``modules`` keys for
    readability::

        pp:
          stages:
            - stage: 0
              modules: [tok_embeddings, layers.0]
    """
    for idx, stage in enumerate(pp_config["stages"]):
        if isinstance(stage, dict):
            stage_idx = stage.get("stage", idx)
            module_fqns = list(stage.get("modules", []))
        else:
            stage_idx = idx
            module_fqns = list(stage)
        plan.pp_stage(stage_idx, module_fqns)


def create_all_fsdp_plan() -> GraphParallelPlan:
    """Create a plan that FSDP-marks every module (``*`` pattern).

    Convenience for tests / quick demos.
    """
    plan = GraphParallelPlan()
    plan.fsdp_mark_pattern("*")
    return plan
