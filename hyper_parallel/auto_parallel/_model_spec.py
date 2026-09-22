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
"""The model IR: a typed, serialisable description of what is being trained.

``memory_estimation/README.md`` already calls ``CostModelConfig`` "the
interface between framework input files and the memory module".  It cannot
be one, because it carries derived values, fuses model shape with parallel
strategy, and has no serialised form -- so the interface gets faked, most
visibly by writing a temporary YAML to disk and parsing it back.

:class:`ModelSpec` is that interface made real for the model half: the facts
about an architecture that hold whatever strategy it is trained under.  It is
a schema, not a grammar.  Three producers fill it -- a framework parser, a
census of an instantiated model, or an engineer writing YAML by hand -- and
every consumer downstream reads the same fields.

Two rules make it safe to hand-fill, both learned from Qwen3.5-35B-A3B, which
reached the cost model with its MTP layer dropped, its shared expert priced at
width zero, and half its attention parameters missing, all silently:

- **Absent is not zero.**  Every optional field defaults to ``None``.  A width
  nothing supplied can then be told apart from a width declared as ``0``,
  which is what let an all-MoE config price its shared expert at no width at
  all and then divide by it.
- **Refuse rather than default.**  :meth:`ModelSpec.validate` rejects a spec
  whose fields contradict each other, at parse time and by name, instead of
  resolving to a number that is wrong further downstream.

The strategy half (degrees, sharding, recompute, precision) is deliberately
absent.  Anything a producer supplies that is not model shape rides in
:attr:`ModelSpec.extra` untouched, so the boundary stays visible until the
execution IR exists to take it.
"""
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping, Optional


class ModelSpecError(ValueError):
    """A model spec is missing a required fact or contradicts itself.

    Raised at parse time, by field name, so a producer learns what it failed
    to supply instead of a consumer inheriting a zero.
    """


def _as_int(value: Any, name: str) -> Optional[int]:
    """Coerce *value* to ``int``, or raise naming the field it came from."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ModelSpecError(f"{name} must be an integer, got {value!r}") from exc


@dataclass(frozen=True)
class VisionSpec:
    """The vision tower of a multimodal model.

    Towers are nested rather than flattened with a prefix, so a spec can carry
    more than one and a consumer can ask whether there is one at all.
    ``max_position_embeddings`` here is the encoder sequence length in merged
    visual tokens, which depends on the images the dataset serves and is
    therefore an input, not a property of the checkpoint.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    name: str = "vision"
    intermediate_size: Optional[int] = None
    out_hidden_size: Optional[int] = None
    patch_size: Optional[int] = None
    spatial_merge_size: Optional[int] = None
    num_position_embeddings: Optional[int] = None
    max_position_embeddings: Optional[int] = None

    def validate(self) -> None:
        """Raise :class:`ModelSpecError` if the tower cannot be costed."""
        for name in ("hidden_size", "num_hidden_layers", "num_attention_heads"):
            value = getattr(self, name)
            if not value or value <= 0:
                raise ModelSpecError(
                    f"vision.{name} is required and must be positive, got {value!r}"
                )


@dataclass(frozen=True)
class ModelSpec:
    """What is being trained, independent of how it is parallelised.

    The four required fields are the ones every cost path dereferences
    unconditionally; the rest are optional because a dense model genuinely has
    no expert count, and ``None`` says so without claiming the count is zero.

    Attributes:
        extra: Producer-supplied keys that are not model shape (precision,
            batch, runtime knobs).  Carried verbatim so nothing is lost while
            the execution IR does not yet exist to receive it.
    """

    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int

    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    intermediate_size: Optional[int] = None
    max_position_embeddings: Optional[int] = None

    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    num_shared_experts: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    shared_expert_intermediate_size: Optional[int] = None
    first_k_dense_replace: Optional[int] = None

    mtp_depth: Optional[int] = None

    multiple_of: Optional[int] = None
    ffn_dim_multiplier: Optional[float] = None
    kv_lora_rank: Optional[int] = None
    q_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    vision: Optional[VisionSpec] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    _REQUIRED = ("hidden_size", "num_hidden_layers", "num_attention_heads", "vocab_size")

    # ---- derived reads -------------------------------------------------

    @property
    def effective_head_dim(self) -> int:
        """Per-head width, explicit when declared and derived when not.

        Costing attention as ``hidden_size x hidden_size`` is only correct
        when ``head_dim == hidden_size / num_attention_heads``.  Qwen3.5
        declares ``head_dim: 256`` against ``h=2048, a=16``, so the two
        differ by 2x and half the projection parameters go missing.
        """
        if self.head_dim:
            return int(self.head_dim)
        return self.hidden_size // self.num_attention_heads

    @property
    def attention_width(self) -> int:
        """Total width of the query projection, ``a * head_dim``."""
        return self.num_attention_heads * self.effective_head_dim

    @property
    def is_moe(self) -> bool:
        """Whether the model routes tokens to more than one expert."""
        return bool(self.num_experts and self.num_experts > 1)

    @property
    def routed_expert_width(self) -> int:
        """Feed-forward width of one routed expert, ``0`` when none resolves.

        Mixtral-style configs run their experts at the dense feed-forward
        width and declare no ``moe_intermediate_size``, so the dense width is
        a real fallback rather than a guess.
        """
        width = self.moe_intermediate_size or self.intermediate_size
        return int(width or 0)

    @property
    def shared_expert_width(self) -> int:
        """Feed-forward width of one shared expert, ``0`` when there is none.

        A config may declare the shared expert's own width or inherit the
        routed one.  An all-MoE config declares no dense ``intermediate_size``
        at all, which is the case that used to price this at zero.
        """
        if not self.num_shared_experts:
            return 0
        width = self.shared_expert_intermediate_size or self.routed_expert_width
        return int(width or 0)

    # ---- validation ----------------------------------------------------

    def validate(self) -> "ModelSpec":
        """Raise :class:`ModelSpecError` unless the spec is internally coherent.

        Returns *self*, so a producer can ``return spec.validate()``.
        """
        for name in self._REQUIRED:
            value = getattr(self, name)
            if not value or value <= 0:
                raise ModelSpecError(
                    f"{name} is required and must be positive, got {value!r}. "
                    "Supply it in the model config or in model.config_overrides."
                )

        kv_heads = self.num_key_value_heads
        if kv_heads and self.num_attention_heads % kv_heads:
            raise ModelSpecError(
                f"num_attention_heads ({self.num_attention_heads}) must be a multiple "
                f"of num_key_value_heads ({kv_heads})"
            )

        if self.is_moe and not self.routed_expert_width:
            raise ModelSpecError(
                f"no feed-forward width resolves for {self.num_experts} routed "
                "experts; declare moe_intermediate_size, or intermediate_size if "
                "the experts run at the dense width"
            )
        if self.num_shared_experts and not self.shared_expert_width:
            raise ModelSpecError(
                "num_shared_experts is set but no width resolves for it; declare "
                "shared_expert_intermediate_size or moe_intermediate_size"
            )
        if self.vision is not None:
            self.vision.validate()
        return self

    # ---- serialised form -----------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the spec as a plain mapping, ready for YAML.

        Unset optional fields are omitted rather than written as ``null``, so
        a hand-written file and a dumped one look alike and a reader cannot
        mistake "not declared" for "declared empty".
        """
        out: Dict[str, Any] = {}
        for spec_field in fields(self):
            if spec_field.name in ("vision", "extra"):
                continue
            value = getattr(self, spec_field.name)
            if value is not None:
                out[spec_field.name] = value
        if self.vision is not None:
            vision: Dict[str, Any] = {}
            for vision_field in fields(self.vision):
                value = getattr(self.vision, vision_field.name)
                if value is not None:
                    vision[vision_field.name] = value
            out["vision"] = vision
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], strict: bool = True) -> "ModelSpec":
        """Build a spec from a mapping, keeping unknown keys in ``extra``.

        Args:
            data: A parsed YAML mapping, or whatever a producer assembled.
            strict: Validate before returning.  Only a test constructing a
                deliberately partial spec should pass ``False``.

        Raises:
            ModelSpecError: If a required field is missing, a field is not an
                integer, or the declared fields contradict each other.
        """
        known = {f.name for f in fields(cls)} - {"vision", "extra"}
        float_fields = {"ffn_dim_multiplier"}
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for key, value in data.items():
            if key == "vision":
                continue
            if key not in known:
                extra[key] = value
            elif key == "name":
                kwargs[key] = str(value)
            elif key in float_fields:
                kwargs[key] = None if value is None else float(value)
            else:
                kwargs[key] = _as_int(value, key)

        for name in cls._REQUIRED:
            if name not in kwargs:
                raise ModelSpecError(
                    f"{name} is required and was not supplied. "
                    "Supply it in the model config or in model.config_overrides."
                )
        kwargs.setdefault("name", "custom")

        raw_vision = data.get("vision")
        if isinstance(raw_vision, Mapping):
            kwargs["vision"] = cls._vision_from_dict(raw_vision)

        spec = cls(extra=extra, **kwargs)
        return spec.validate() if strict else spec

    @staticmethod
    def _vision_from_dict(data: Mapping[str, Any]) -> VisionSpec:
        """Build a :class:`VisionSpec`, dropping keys it does not declare."""
        known = {f.name for f in fields(VisionSpec)}
        kwargs: Dict[str, Any] = {}
        for key, value in data.items():
            if key not in known:
                continue
            kwargs[key] = str(value) if key == "name" else _as_int(value, f"vision.{key}")
        missing = [
            name for name in ("hidden_size", "num_hidden_layers", "num_attention_heads")
            if name not in kwargs
        ]
        if missing:
            raise ModelSpecError(
                f"vision tower is missing required fields: {', '.join(missing)}"
            )
        return VisionSpec(**kwargs)
