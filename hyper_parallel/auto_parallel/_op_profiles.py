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
"""Op profiles: the op counts of each known architecture family, as data.

Every file under ``op_profiles/`` names a family and declares, per layer kind,
how many times one layer runs each op the cost model prices.  These are the
numbers the ND arch hooks used to assign in code, one callback per family.

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

# Families a model name is matched against, in order, first match wins.  This
# is the order the cost model itself used to match names in.
_NAME_ORDER = ("llama2", "mixtral", "t5", "pangualpha", "deepseek", "qwen", "cm")


@dataclass(frozen=True)
class OpProfile:
    """The op counts of one architecture family, per layer kind."""

    arch: str
    kinds: Dict[str, OpCounts]

    def counts(self, kind: str) -> OpCounts:
        """Return one layer kind's counts, or raise naming the kinds there are."""
        if kind not in self.kinds:
            raise ModelSpecError(
                f"op profile {self.arch!r} has no layer kind {kind!r}; "
                f"it has {sorted(self.kinds)}"
            )
        return self.kinds[kind]


def known_archs() -> Tuple[str, ...]:
    """Return the name of every family that has a profile file."""
    return tuple(sorted(
        name[:-len(".yaml")] for name in os.listdir(PROFILE_DIR) if name.endswith(".yaml")
    ))


@functools.lru_cache(maxsize=None)
def load_op_profile(arch: str) -> OpProfile:
    """Read and validate the profile of one family.

    Raises:
        ModelSpecError: If no profile has that name, or the file declares no
            layer kind, an unknown key, or an incomplete op vector.
    """
    if arch not in known_archs():
        raise ModelSpecError(
            f"unknown arch {arch!r}; the op profiles are {list(known_archs())}"
        )
    with open(os.path.join(PROFILE_DIR, f"{arch}.yaml"), encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    unknown = sorted(set(data) - {"kinds"})
    if unknown:
        raise ModelSpecError(f"op profile {arch!r} has unknown keys {unknown}")
    kinds = data.get("kinds")
    if not isinstance(kinds, Mapping) or not kinds:
        raise ModelSpecError(f"op profile {arch!r} declares no layer kind")
    return OpProfile(arch, {
        str(kind): OpCounts.from_dict(counts, f"{arch}.kinds.{kind}")
        for kind, counts in kinds.items()
    })


def infer_arch(name: Any) -> str:
    """Return the family a free-text model name belongs to.

    The rule the cost model applied to every model before the family became
    data: the first family name in a fixed order that occurs in the
    lower-cased model name, else the default profile.  Only a producer with
    no better source should use it; a spec that declares ``arch`` never gets
    here.
    """
    lowered = str(name).lower()
    for arch in _NAME_ORDER:
        if arch in lowered:
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
