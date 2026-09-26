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
"""Shared Transformers-config resolution for the Auto Parallel cost model.

Both the SAPP-ND parser (``CostModelParserHyperV2``) and the config-adapter
loader need the same model dimensions out of an AutoModels ``model`` section.
This module owns that translation exactly once, so the two callers cannot
drift apart on field names, aliases, or fallback behaviour.

It is also the single place that imports ``transformers``.  The import is
function-local because ``transformers`` is not a hard dependency of
``hyper_parallel`` (``requirements.txt`` only pins numpy) and the non-Hyper
cost-model backends must keep working without it.
"""
import copy
import logging
from typing import Any, Dict, List, Mapping, Optional, Tuple

from hyper_parallel.auto_parallel._layer_census import census_activations, census_output_activations

logger = logging.getLogger(__name__)

# AutoModels root sections that identify the current Trainer schema.
_AUTO_MODELS_SECTIONS = ("training", "accelerator", "fsdp_config")

# ``model`` keys forwarded verbatim to ``AutoConfig.from_pretrained``.
_CONFIG_KWARG_NAMES = (
    "cache_dir", "local_files_only", "revision", "subfolder",
    "token", "trust_remote_code",
)

# Where the model directory is named. A from_config recipe builds the model
# through a factory (``_target_`` plus ``config_path``) and never loads a
# checkpoint, so it carries no pretrained path.
_MODEL_PATH_ALIASES = ("pretrained_model_name_or_path", "config_path")

# Canonical text-tower fields, with the Transformers aliases that carry them.
# First populated alias wins, so Hyper-internal names keep priority over the
# Hugging Face spelling when a config happens to define both.
_TEXT_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "hidden_size": ("hidden_size",),
    "num_hidden_layers": ("num_hidden_layers",),
    "num_attention_heads": ("num_attention_heads",),
    "num_key_value_heads": ("num_key_value_heads",),
    "head_dim": ("head_dim",),
    "intermediate_size": ("intermediate_size",),
    "vocab_size": ("vocab_size",),
    "max_position_embeddings": ("max_position_embeddings",),
    "num_experts": ("num_experts", "n_routed_experts"),
    "num_experts_per_tok": ("num_experts_per_tok",),
    "num_shared_experts": ("n_shared_experts", "num_shared_experts"),
    "moe_intermediate_size": ("moe_intermediate_size",),
    "shared_expert_intermediate_size": ("shared_expert_intermediate_size",),
    "first_k_dense_replace": ("first_k_dense_replace",),
    "mtp_depth": ("mtp_depth", "num_nextn_predict_layers", "mtp_num_hidden_layers"),
    "multiple_of": ("multiple_of",),
    "ffn_dim_multiplier": ("ffn_dim_multiplier",),
    "kv_lora_rank": ("kv_lora_rank",),
    "q_lora_rank": ("q_lora_rank",),
    "qk_rope_head_dim": ("qk_rope_head_dim",),
    "attn_output_gate": ("attn_output_gate",),
    "tie_word_embeddings": ("tie_word_embeddings",),
    # Stated by a few families; Qwen3's config does not state its own.
    "qk_norm": ("qk_norm", "use_qk_norm", "qk_layernorm"),
    # A hybrid stack states its layers; without this every layer is costed
    # as full attention, which is quadratic in the sequence length.
    "layer_types": ("layer_types",),
    "linear_num_key_heads": ("linear_num_key_heads",),
    "linear_key_head_dim": ("linear_key_head_dim",),
    "linear_num_value_heads": ("linear_num_value_heads",),
    "linear_value_head_dim": ("linear_value_head_dim",),
    "linear_conv_kernel_dim": ("linear_conv_kernel_dim",),
}

# Vision towers use their own spelling for the shared concepts.
_VISION_FIELD_ALIASES: Dict[str, Tuple[str, ...]] = {
    "hidden_size": ("hidden_size",),
    "num_hidden_layers": ("depth", "num_hidden_layers"),
    "num_attention_heads": ("num_heads", "num_attention_heads"),
    "intermediate_size": ("intermediate_size",),
    "out_hidden_size": ("out_hidden_size",),
    "patch_size": ("patch_size",),
    "spatial_merge_size": ("spatial_merge_size",),
    "num_position_embeddings": ("num_position_embeddings",),
}

# Fallback when a vision tower declares no positional-embedding grid.
_DEFAULT_VISUAL_SEQ_LEN = 1024


# What a lower-cased model name contains when its attention normalizes each
# head's queries and keys: the Qwen3 generation, Qwen3.5 and the Qwen3
# vision-language models included.
_QK_NORM_NAMES = ("qwen3",)


def infer_qk_norm(name: Any) -> bool:
    """Return whether a model named *name* normalizes each head's queries and keys.

    For a producer whose config does not state it, as a Transformers config
    does not: the Qwen3 generation does, and no other family the cost model
    prices does.
    """
    lowered = str(name).lower()
    return any(pattern in lowered for pattern in _QK_NORM_NAMES)


def _settle_qk_norm(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Settle whether the language model normalizes queries and keys, from its name if nothing states it."""
    if spec.get("qk_norm") is None:
        spec["qk_norm"] = infer_qk_norm(spec.get("name"))
    return spec


def _declares_pretrained_path(mapping: Any) -> bool:
    """Return whether the ``model`` section names a Transformers checkpoint.

    This is the one key only this schema has: the older HyperParallel yamls
    name a checkpoint as ``model.weights_path``. Recognising it means a
    config needs no empty ``training``/``accelerator``/``fsdp_config``
    section just to be identified.
    """
    if isinstance(mapping, Mapping):
        model = mapping.get("model")
    else:
        model = getattr(mapping, "model", None)
    if model is None:
        return False
    if isinstance(model, Mapping):
        return bool(model.get("pretrained_model_name_or_path"))
    return bool(getattr(model, "pretrained_model_name_or_path", None))


def is_auto_models_schema(mapping: Any) -> bool:
    """Return whether *mapping* looks like an AutoModels Trainer config.

    Accepts a plain mapping or any object exposing the root sections as
    attributes, so the SAPP-ND ``Config`` tree and a parsed YAML dict can be
    tested with the same call.
    """
    if _declares_pretrained_path(mapping):
        return True
    if isinstance(mapping, Mapping):
        return any(name in mapping for name in _AUTO_MODELS_SECTIONS)
    holder = getattr(mapping, "__dict__", None)
    if isinstance(holder, dict):
        return any(name in holder for name in _AUTO_MODELS_SECTIONS)
    return False


def _first_attr(config: Any, names: Tuple[str, ...], default: Any = None) -> Any:
    """Return the first populated attribute of *config* among *names*."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


def _text_tower(model_config: Any) -> Any:
    """Return the language-model sub-config of a composite config.

    Multimodal Transformers configs (``Qwen3VLMoeConfig`` and friends) keep
    the language model under ``text_config`` and expose none of its fields at
    the top level, so reading ``hidden_size`` directly raises AttributeError.
    """
    return getattr(model_config, "text_config", None) or model_config


def _spec_from_aliases(config: Any, aliases: Dict[str, Tuple[str, ...]]) -> Dict[str, Any]:
    """Collect the canonical fields declared by *config*."""
    spec = {}
    for canonical, names in aliases.items():
        value = _first_attr(config, names)
        if value is not None:
            spec[canonical] = value
    return spec


def _derive_shared_experts(spec: Dict[str, Any]) -> None:
    """Infer the shared-expert count from its total feed-forward width.

    Qwen2-MoE style configs declare one wide shared expert instead of a
    count.  The cost model only ever uses ``n_shared_exp * hff_exp``, so
    encoding that width as several narrow experts is width-equivalent.
    """
    if spec.get("num_shared_experts"):
        return
    moe_inter = int(spec.get("moe_intermediate_size", 0) or 0)
    shared_inter = int(spec.get("shared_expert_intermediate_size", 0) or 0)
    if moe_inter and shared_inter:
        spec["num_shared_experts"] = max(1, shared_inter // moe_inter)


def _derive_dense_ffn_width(spec: Dict[str, Any]) -> None:
    """Fill the dense feed-forward width for an all-MoE config.

    A model whose every layer is MoE (Qwen3.5-MoE) declares no
    ``intermediate_size``, but the cost model prices the shared expert at
    that width, so leaving it absent drops the shared expert entirely.
    The shared expert's own width is the value meant there.
    """
    if spec.get("intermediate_size"):
        return
    width = spec.get("shared_expert_intermediate_size") or spec.get("moe_intermediate_size")
    if width:
        spec["intermediate_size"] = int(width)


def _visual_seq_len(vision_spec: Dict[str, Any], override: Optional[int]) -> int:
    """Resolve the encoder sequence length in merged visual tokens.

    The true count depends on the image resolution served by the dataset,
    which no configuration we read carries.  The positional-embedding grid
    divided by the spatial merge is the tightest bound available from the
    model alone, so it is the default and *override* exists to correct it.
    """
    if override:
        logger.info("visual sequence length taken from context.visual_seq_len: %d", override)
        return int(override)
    grid = int(vision_spec.get("num_position_embeddings", 0) or 0)
    merge = max(1, int(vision_spec.get("spatial_merge_size", 1) or 1))
    if not grid:
        logger.warning(
            "vision tower declares no num_position_embeddings; assuming %d visual tokens. "
            "Set context.visual_seq_len to the value your dataset produces.",
            _DEFAULT_VISUAL_SEQ_LEN,
        )
        return _DEFAULT_VISUAL_SEQ_LEN
    derived = max(1, grid // (merge * merge))
    logger.info(
        "visual sequence length derived from the vision config: %d "
        "(num_position_embeddings=%d, spatial_merge_size=%d). "
        "Set context.visual_seq_len to override.",
        derived, grid, merge,
    )
    return derived


def _get_hf_config(model_raw: Mapping[str, Any]) -> Any:
    """Call the AutoModels Transformers entry point for *model_raw*."""
    # Transformers is an optional dependency; import only when a config
    # actually needs it so the other cost-model backends stay importable.
    from hyper_parallel.models._transformers.config_resolver import get_hf_config  # pylint: disable=C0415

    config_kwargs = {
        name: model_raw[name]
        for name in _CONFIG_KWARG_NAMES
        if model_raw.get(name) is not None
    }
    return get_hf_config(
        str(_model_path(model_raw)),
        str(model_raw.get("attn_implementation", "sdpa")),
        model_raw.get("torch_dtype", "auto"),
        **config_kwargs,
    )


def _model_path(model_raw: Mapping[str, Any]) -> Optional[str]:
    """Return the model directory the section names, under any accepted key."""
    for alias in _MODEL_PATH_ALIASES:
        value = model_raw.get(alias)
        if value:
            return str(value)
    return None


def checkpoint_configs(model_raw: Mapping[str, Any]) -> Tuple[Any, Any]:
    """The Transformers config of the checkpoint *model_raw* names, and its language model's."""
    config = _get_hf_config(model_raw)
    return config, _text_tower(config)


def _explicit_overrides(model_raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the fields the model section states itself, as a plain dict.

    ``config_overrides`` is the declared form. A from_config recipe instead
    passes the shape it builds as factory arguments beside ``_target_``, such as
    a ``num_hidden_layers`` that crops the released model; those name canonical
    fields, so they are overrides too, and the released config must not win over
    them. ``config_overrides`` keeps priority where both are present.
    """
    overrides = model_raw.get("config_overrides")
    explicit = dict(overrides) if isinstance(overrides, Mapping) else {}
    for field in _TEXT_FIELD_ALIASES:
        value = model_raw.get(field)
        if value is not None:
            explicit.setdefault(field, value)
    return explicit


# The censuses this process has run, by config, layer stack and length: a
# harness pricing several runs of one model runs one.
_CENSUSES: Dict[Tuple[Any, ...], Dict[str, Any]] = {}


def _census(text_config: Any, layers: Any, seq_length: int) -> Dict[str, Any]:
    """The spec's census of each layer kind of *layers* and of the output layer, once per config and length."""
    key = (
        text_config.to_json_string(),
        tuple((group["kind"], int(group["count"])) for group in layers),
        int(seq_length),
    )
    if key not in _CENSUSES:
        logger.info("census of each layer kind and of the output layer at %d tokens", seq_length)
        kinds = census_activations(text_config, layers, seq_length)
        _CENSUSES[key] = {
            "activations": {kind: record.to_dict() for kind, record in kinds.items()},
            "output_activations": census_output_activations(text_config, seq_length).to_dict(),
        }
    return copy.deepcopy(_CENSUSES[key])


def _no_census(census_seq_len: int, explicit: Mapping[str, Any]) -> None:
    """Warn that a census asked for cannot run, unless the overrides state its records.

    A census builds its layers from the checkpoint's config; a search hands
    ND a spec whose overrides state the records its reader measured.
    """
    if census_seq_len and not explicit.get("activations"):
        logger.warning("no census of the layers: it needs the checkpoint's Transformers config")


def _census_layers(spec: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The groups of body layers a census tells apart, in model order, or None.

    The Hyper parser prices a hybrid model's layers by the attention flavour
    its ``layer_types`` states, and any other model's as one kind; None
    where the first layers are dense (``first_k_dense_replace``), a kind
    the parser prices through the model's family.
    """
    if spec.get("first_k_dense_replace"):
        return None
    count = int(spec.get("num_hidden_layers") or 0)
    kinds = [str(kind) for kind in (spec.get("layer_types") or [])[:count]] or ["decoder"] * count
    groups: List[Dict[str, Any]] = []
    for kind in kinds:
        if groups and groups[-1]["kind"] == kind:
            groups[-1]["count"] += 1
        else:
            groups.append({"kind": kind, "count": 1})
    return groups


def resolve_hf_model_spec(
    model_raw: Mapping[str, Any],
    visual_seq_len: Optional[int] = None,
    census_seq_len: int = 0,
) -> Dict[str, Any]:
    """Return canonical cost-model fields for a Trainer ``model`` section.

    Resolves ``model.pretrained_model_name_or_path`` through the same
    Transformers path AutoModels uses.  A composite (vision-language) config
    contributes its language tower under the canonical keys and its vision
    tower under ``"vision"``; a plain config leaves ``"vision"`` absent.

    ``model.config_overrides`` stays supported for standalone cost-model
    search files, and doubles as the fallback when the Transformers config
    cannot be reached (offline node, unreachable repository).  Explicit
    overrides always win over resolved values.  Whether the language model
    normalizes its queries and keys, which a Transformers config does not
    state, is settled from its name when nothing states it.

    Args:
        model_raw: The ``model`` section, as a plain mapping.
        visual_seq_len: Optional override for the encoder sequence length.
        census_seq_len: The tokens to run a census of each layer kind of
            the language model at, and of its output layer
            (:mod:`hyper_parallel.auto_parallel._layer_census`), which the
            spec states as ``"activations"`` and ``"output_activations"``;
            0 runs none.  The census builds its layers from the checkpoint's
            config: a spec from ``config_overrides`` alone gets none, and an
            override of a model field does not reach it.

    Returns:
        A dict of canonical model fields, always carrying ``"name"``.

    Raises:
        ValueError: If neither a pretrained path nor overrides can supply
            the model dimensions.
    """
    explicit = _explicit_overrides(model_raw)
    model_path = _model_path(model_raw)

    if not model_path:
        if explicit:
            explicit.setdefault("name", model_raw.get("name", "custom"))
            _no_census(census_seq_len, explicit)
            return _settle_qk_norm(explicit)
        raise ValueError(
            "AutoModels train.yaml requires model.pretrained_model_name_or_path, "
            "model.config_path or model.config_overrides for Auto Parallel search"
        )

    try:
        model_config = _get_hf_config(model_raw)
    except (ImportError, OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
        if explicit:
            logger.warning(
                "Transformers config resolution failed (%s); "
                "falling back to model.config_overrides", exc,
            )
            explicit.setdefault("name", model_raw.get("name", "custom"))
            _no_census(census_seq_len, explicit)
            return _settle_qk_norm(explicit)
        raise ValueError(
            f"cannot resolve model.pretrained_model_name_or_path '{model_path}'; "
            "install transformers, set model.config_overrides, or make the config "
            "available offline (warm the HF_HOME cache, or pass local_files_only)"
        ) from exc

    spec = _spec_from_aliases(_text_tower(model_config), _TEXT_FIELD_ALIASES)
    _derive_shared_experts(spec)
    _derive_dense_ffn_width(spec)
    spec["name"] = str(getattr(model_config, "model_type", None) or model_path)

    vision_config = getattr(model_config, "vision_config", None)
    if vision_config is not None:
        vision_spec = _spec_from_aliases(vision_config, _VISION_FIELD_ALIASES)
        vision_spec["name"] = f"{spec['name']}_vision"
        vision_spec["max_position_embeddings"] = _visual_seq_len(vision_spec, visual_seq_len)
        spec["vision"] = vision_spec

    spec.update(explicit)
    spec = _settle_qk_norm(spec)
    if census_seq_len:
        layers = _census_layers(spec)
        if layers is None:
            logger.warning("no census of %s: its first layers are dense, a kind a census does not tell apart",
                           spec["name"])
        else:
            spec.update(_census(_text_tower(model_config), layers, census_seq_len))
    return spec
