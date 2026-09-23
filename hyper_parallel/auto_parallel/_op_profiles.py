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
"""Op profiles: the layer kinds of each known architecture family, as data.

Every file under ``op_profiles/`` names a family and declares its layer kinds.
A kind states how many times one layer runs each op the cost model prices,
the numbers the ND arch hooks used to assign in code, and two flavours from a
closed set that say how the layer is shaped:

- ``attention``: ``full`` (the default) or ``linear``, a gated-DeltaNet layer
  priced on the model's linear-attention dimensions;
- ``ffn``: ``dense`` or ``moe``, or absent to keep the model's own
  feed-forward, for families whose layers do not differ in it.

A profile with more than one kind may name a ``default``, the kind a layer
takes when nothing says otherwise; a profile with one kind defaults to it.

A model spec names its profile with ``arch``, and may declare its own counts
in ``ops`` instead.  The cost model reads only what the spec declares.  A
producer that has nothing better than a free-text model name, such as a
MindFormers ``trainer.model_name``, falls back on :func:`infer_arch`.
"""
import functools
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import yaml

from hyper_parallel.auto_parallel._model_spec import ModelSpecError, OpCounts

logger = logging.getLogger(__name__)

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "op_profiles")
DEFAULT_ARCH = "default"

ATTENTION_FLAVOURS = ("full", "linear")
FFN_FLAVOURS = ("dense", "moe")

# Names a model name is matched against, in order, first match wins, with the
# family each one means.  This is the order the cost model itself used to
# match names in; Qwen3.5 comes before the Qwen family it would otherwise
# fall into, under both the Transformers spelling and the release name.
_NAME_ORDER = (
    ("llama2", "llama2"), ("mixtral", "mixtral"), ("t5", "t5"),
    ("pangualpha", "pangualpha"), ("deepseek", "deepseek"),
    ("qwen3_5", "qwen3_5"), ("qwen3.5", "qwen3_5"), ("qwen", "qwen"), ("cm", "cm"),
)


@dataclass(frozen=True)
class LayerKind:
    """One kind of layer: the ops it runs and how it is shaped.

    Attributes:
        name: The kind's name in its profile, which a layer stack refers to.
        ops: How many times one layer of this kind runs each op.
        attention: ``full``, or ``linear`` for a gated-DeltaNet layer.
        ffn: ``dense``, ``moe``, or ``None`` to keep the model's own.
    """

    name: str
    ops: OpCounts
    attention: str = "full"
    ffn: Optional[str] = None


@dataclass(frozen=True)
class OpProfile:
    """The layer kinds of one architecture family.

    Attributes:
        arch: The family, the profile's file name.
        layer_kinds: Every kind of the family, by name.
        default: The kind a layer takes when nothing says otherwise, or
            ``None`` for a family whose stack always states its kinds.
    """

    arch: str
    layer_kinds: Dict[str, LayerKind]
    default: Optional[str] = None

    @property
    def kinds(self) -> Dict[str, OpCounts]:
        """Return each kind's op counts, by kind name."""
        return {name: kind.ops for name, kind in self.layer_kinds.items()}

    def counts(self, kind: str) -> OpCounts:
        """Return one layer kind's counts, or raise naming the kinds there are."""
        return self.kind(kind).ops

    def kind(self, name: str) -> LayerKind:
        """Return one layer kind, or raise naming the kinds there are."""
        if name not in self.layer_kinds:
            raise ModelSpecError(
                f"op profile {self.arch!r} has no layer kind {name!r}; "
                f"it has {sorted(self.layer_kinds)}"
            )
        return self.layer_kinds[name]


def known_archs() -> Tuple[str, ...]:
    """Return the name of every family that has a profile file."""
    return tuple(sorted(
        name[:-len(".yaml")] for name in os.listdir(PROFILE_DIR) if name.endswith(".yaml")
    ))


def _flavour(value: Any, allowed: Tuple[str, ...], where: str, optional: bool) -> Optional[str]:
    """Return a flavour value, refusing one outside *allowed*."""
    if value is None and optional:
        return None
    if value not in allowed:
        raise ModelSpecError(f"{where} must be one of {list(allowed)}, got {value!r}")
    return str(value)


def _layer_kind(arch: str, name: str, data: Any) -> LayerKind:
    """Build one kind of a profile from its ``{ops, attention, ffn}`` mapping."""
    where = f"{arch}.kinds.{name}"
    if not isinstance(data, Mapping):
        raise ModelSpecError(f"{where} must map ops and flavours, got {data!r}")
    unknown = sorted(set(data) - {"ops", "attention", "ffn"})
    if unknown:
        raise ModelSpecError(f"{where} has unknown keys {unknown}; a kind has ops, attention and ffn")
    return LayerKind(
        name=str(name),
        ops=OpCounts.from_dict(data.get("ops"), f"{where}.ops"),
        attention=_flavour(data.get("attention", "full"), ATTENTION_FLAVOURS, f"{where}.attention", False),
        ffn=_flavour(data.get("ffn"), FFN_FLAVOURS, f"{where}.ffn", True),
    )


@functools.lru_cache(maxsize=None)
def load_op_profile(arch: str) -> OpProfile:
    """Read and validate the profile of one family.

    Raises:
        ModelSpecError: If no profile has that name, or the file declares no
            layer kind, an unknown key, an incomplete op vector, an unknown
            flavour, or a default that is not one of its kinds.
    """
    if arch not in known_archs():
        raise ModelSpecError(
            f"unknown arch {arch!r}; the op profiles are {list(known_archs())}"
        )
    with open(os.path.join(PROFILE_DIR, f"{arch}.yaml"), encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    unknown = sorted(set(data) - {"kinds", "default"})
    if unknown:
        raise ModelSpecError(f"op profile {arch!r} has unknown keys {unknown}")
    kinds = data.get("kinds")
    if not isinstance(kinds, Mapping) or not kinds:
        raise ModelSpecError(f"op profile {arch!r} declares no layer kind")
    layer_kinds = {str(name): _layer_kind(arch, name, kind) for name, kind in kinds.items()}
    default = data.get("default")
    if default is None and len(layer_kinds) == 1:
        default = next(iter(layer_kinds))
    if default is not None and default not in layer_kinds:
        raise ModelSpecError(
            f"op profile {arch!r} names default kind {default!r}, but its kinds are {sorted(layer_kinds)}"
        )
    return OpProfile(arch, layer_kinds, default)


def infer_arch(name: Any) -> str:
    """Return the family a free-text model name belongs to.

    The rule the cost model applied to every model before the family became
    data: the first family name in a fixed order that occurs in the
    lower-cased model name, else the default profile.  Only a producer with
    no better source should use it; a spec that declares ``arch`` never gets
    here.
    """
    lowered = str(name).lower()
    for pattern, arch in _NAME_ORDER:
        if pattern in lowered:
            return arch
    logger.warning(
        "no op profile matches model %r, pricing it as %r; declare arch to choose one",
        name, DEFAULT_ARCH,
    )
    return DEFAULT_ARCH


def resolve_ops(arch: str, ops: Optional[Mapping[str, OpCounts]] = None) -> Dict[str, OpCounts]:
    """Return the op counts a model is priced with, per layer kind.

    Declared *ops* replace the profile's, but must name the same layer kinds:
    the family's hooks decide which layers are of which kind, so counts for a
    kind they never assign would be dropped without a word.

    Raises:
        ModelSpecError: If *arch* has no profile, or *ops* names other kinds.
    """
    profile = load_op_profile(arch)
    if ops is None:
        return dict(profile.kinds)
    if set(ops) != set(profile.kinds):
        raise ModelSpecError(
            f"ops declares layer kinds {sorted(ops)}, but arch {arch!r} "
            f"prices {sorted(profile.kinds)}"
        )
    return dict(ops)
