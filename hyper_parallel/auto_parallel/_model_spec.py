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
from typing import Any, Dict, Mapping, Optional, Tuple


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


def _as_count(value: Any, name: str) -> int:
    """Coerce *value* to a non-negative whole number, or raise naming it."""
    if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
        raise ModelSpecError(f"{name} must be a whole number, got {value!r}")
    count = _as_int(value, name)
    if count is None or count < 0:
        raise ModelSpecError(f"{name} must be a non-negative count, got {value!r}")
    return count


@dataclass(frozen=True)
class OpCounts:
    """How many times one layer kind runs each op the cost model prices.

    The estimators already price a layer as a count times a per-op cost
    (``perf_estimation/getters.py``); this is the count half, declared as data
    instead of assigned by a per-family callback.  It belongs to a layer
    *kind*, not to a model: an encoder-decoder, or a stack mixing linear and
    full attention, runs different ops in different layers.

    Every op is required.  A count left out would be priced as zero, which is
    exactly the silent failure the model IR exists to refuse, so a layer that
    does not run an op declares ``0``.  The parameter-cast counts are absent on
    purpose: whether parameters are cast depends on optimizer sharding, which
    is strategy, so the consumer derives them from ``attMM`` and ``ffMM``.
    """

    attMM: int     # attention projections: q, k, v, o
    attBMM: int    # attention batched matmuls: QK^T and AV
    ffMM: int      # feed-forward projections, e.g. 3 for a gated MLP
    softmax: int
    dropout: int
    normOp: int
    gather: int    # tensor-parallel gathers
    headCast: int  # casts of the per-head score tensor
    ffAct: int     # feed-forward activation functions
    linrec: int    # linear-attention state updates: the delta rule's recurrence

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str = "ops") -> "OpCounts":
        """Build the vector from a mapping, refusing a missing or unknown op.

        Args:
            data: ``{op name: count}``, one entry per op.
            where: Where the mapping came from, for the error message.

        Raises:
            ModelSpecError: If an op is missing, unknown, or not a count.
        """
        if not isinstance(data, Mapping):
            raise ModelSpecError(f"{where} must map op names to counts, got {data!r}")
        names = [f.name for f in fields(cls)]
        unknown = sorted(set(data) - set(names))
        if unknown:
            raise ModelSpecError(f"{where} declares unknown ops {unknown}; the ops are {names}")
        missing = [name for name in names if name not in data]
        if missing:
            raise ModelSpecError(
                f"{where} is missing counts for {missing}; declare 0 for an op "
                "the layer does not run"
            )
        return cls(**{name: _as_count(data[name], f"{where}.{name}") for name in names})

    def to_dict(self) -> Dict[str, int]:
        """Return the vector as ``{op name: count}``, in declaration order."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


def ops_from_dict(data: Any) -> Dict[str, OpCounts]:
    """Parse ``{layer kind: {op: count}}``, the serialised form of ``ModelSpec.ops``."""
    if not isinstance(data, Mapping) or not data:
        raise ModelSpecError(
            f"ops must map at least one layer kind to its op counts, got {data!r}"
        )
    return {str(kind): OpCounts.from_dict(counts, f"ops.{kind}") for kind, counts in data.items()}


@dataclass(frozen=True)
class LayerGroup:
    """Consecutive layers of one kind, in model order.

    A model's stack is a tuple of groups: DeepSeek-V3 is three dense layers,
    58 MoE layers and one MTP layer; Qwen3.5 alternates groups of linear- and
    full-attention layers.  ``kind`` names a kind of the model's op profile.
    MTP layers are body-shaped layers after the body, marked so that nothing
    has to count them from the end.
    """

    kind: str
    count: int
    mtp: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], where: str = "layers") -> "LayerGroup":
        """Build a group from ``{kind, count, mtp}``, refusing anything else.

        Raises:
            ModelSpecError: If the kind is missing, the count is not a
                positive whole number, or an unknown key is present.
        """
        if not isinstance(data, Mapping):
            raise ModelSpecError(f"{where} must be a mapping of kind and count, got {data!r}")
        unknown = sorted(set(data) - {"kind", "count", "mtp"})
        if unknown:
            raise ModelSpecError(f"{where} declares unknown keys {unknown}; the keys are kind, count, mtp")
        kind = data.get("kind")
        if not isinstance(kind, str) or not kind:
            raise ModelSpecError(f"{where}.kind must name a layer kind, got {kind!r}")
        count = _as_count(data.get("count"), f"{where}.count")
        if count == 0:
            raise ModelSpecError(f"{where}.count must be positive; leave out a group with no layers")
        mtp = data.get("mtp", False)
        if not isinstance(mtp, bool):
            raise ModelSpecError(f"{where}.mtp must be true or false, got {mtp!r}")
        return cls(kind=kind, count=count, mtp=mtp)

    def to_dict(self) -> Dict[str, Any]:
        """Return the group as a mapping, omitting ``mtp`` when it is false."""
        out: Dict[str, Any] = {"kind": self.kind, "count": self.count}
        if self.mtp:
            out["mtp"] = True
        return out


def layers_from_list(data: Any) -> Tuple[LayerGroup, ...]:
    """Parse a list of groups, the serialised form of ``ModelSpec.layers``."""
    if not isinstance(data, (list, tuple)) or not data:
        raise ModelSpecError(f"layers must list at least one group, got {data!r}")
    return tuple(LayerGroup.from_dict(group, f"layers[{index}]") for index, group in enumerate(data))


def check_layer_counts(layers: Tuple[LayerGroup, ...], num_layers: int, mtp_depth: int) -> None:
    """Raise unless *layers* covers exactly the body and the MTP layers, in order.

    Args:
        layers: The groups, in model order.
        num_layers: The number of body layers the model declares.
        mtp_depth: The number of MTP layers the model declares.

    Raises:
        ModelSpecError: If a body group follows an MTP group, or either
            part does not sum to its declared count.
    """
    seen_mtp = False
    for index, group in enumerate(layers):
        if seen_mtp and not group.mtp:
            raise ModelSpecError(f"layers[{index}] is a body group after an MTP group; MTP layers come last")
        seen_mtp = seen_mtp or group.mtp
    body = sum(group.count for group in layers if not group.mtp)
    mtp = sum(group.count for group in layers if group.mtp)
    if body != num_layers:
        raise ModelSpecError(f"layers list {body} body layers, but num_hidden_layers is {num_layers}")
    if mtp != mtp_depth:
        raise ModelSpecError(f"layers list {mtp} MTP layers, but mtp_depth is {mtp_depth}")


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
        arch: The op profile the model is priced with, one of the files in
            ``auto_parallel/op_profiles``.  A producer that knows the family
            declares it; otherwise it is inferred from ``name`` when the spec
            is resolved.
        ops: Per-layer-kind op counts that replace the profile's own, for a
            model no profile describes.  Its kinds must be the profile's.
        layers: The layer stack: groups of consecutive layers of one kind of
            the profile, body first and MTP layers last.  A producer derives
            it from ``layer_types``, ``first_k_dense_replace`` and
            ``mtp_depth`` when the config does not state it.
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

    # Gated DeltaNet linear attention (Qwen3.5), for the layers a linear kind prices.
    linear_num_key_heads: Optional[int] = None
    linear_key_head_dim: Optional[int] = None
    linear_num_value_heads: Optional[int] = None
    linear_value_head_dim: Optional[int] = None
    linear_conv_kernel_dim: Optional[int] = None

    arch: Optional[str] = None
    ops: Optional[Dict[str, OpCounts]] = None
    layers: Optional[Tuple[LayerGroup, ...]] = None

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
        if self.layers is not None:
            check_layer_counts(self.layers, self.num_hidden_layers, self.mtp_depth or 0)
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
            if spec_field.name in ("vision", "extra", "ops", "layers"):
                continue
            value = getattr(self, spec_field.name)
            if value is not None:
                out[spec_field.name] = value
        if self.ops is not None:
            out["ops"] = {kind: counts.to_dict() for kind, counts in self.ops.items()}
        if self.layers is not None:
            out["layers"] = [group.to_dict() for group in self.layers]
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
        kwargs: Dict[str, Any] = {}
        extra: Dict[str, Any] = {}
        for key, value in data.items():
            if key == "vision":
                continue
            if key in known:
                kwargs[key] = cls._field_value(key, value)
            else:
                extra[key] = value

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
    def _field_value(key: str, value: Any) -> Any:
        """Coerce one declared field to the type the schema gives it."""
        if key == "name":
            return str(value)
        if value is None:
            return None
        if key == "arch":
            return str(value)
        if key == "ops":
            return ops_from_dict(value)
        if key == "layers":
            return layers_from_list(value)
        if key == "ffn_dim_multiplier":
            return float(value)
        return _as_int(value, key)

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
