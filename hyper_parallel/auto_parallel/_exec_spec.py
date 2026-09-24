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
"""The execution IR: how a model is trained, beside what the model is.

:class:`~hyper_parallel.auto_parallel._model_spec.ModelSpec` states the facts
about an architecture that hold whatever strategy it runs under.
:class:`ExecSpec` states the rest: what a search varies (the parallel
degrees, the micro-batching, the sharding and the recompute policy) and the
fixed facts of the run (the sequence length, the precisions, the kernels and
the device's memory).  The cost model derives everything else from the two.

It follows the model spec's two rules.  **Absent is not zero**: every field
defaults to ``None``, which means "not stated", so an ExecSpec can state a
whole run or only what a search candidate changes, and a consumer never
mistakes an unstated degree for a degree of zero.  **Refuse rather than
default**: :meth:`ExecSpec.validate` rejects a stated value that cannot be
right, and :meth:`ExecSpec.from_dict` refuses a key it does not know, by name.
"""
from dataclasses import dataclass, fields
from typing import Any, Dict, Mapping, Optional, Union


class ExecSpecError(ValueError):
    """An execution spec states something it cannot, or something unknown."""


# The rules by which selective recompute picks the ops it recomputes.
SELECTIVE_RULES = ("hyperparallel", "mindformers")


@dataclass(frozen=True)
class ExecSpec:
    """How a model is trained: the strategy a search varies, and the run.

    Every field is optional; ``None`` means the spec does not state it.  A
    byte width, ``grad_accumulation`` or ``shard_activations`` that no spec
    states takes the default of the model's family, from its op profile.

    Attributes:
        dp, tp, pp, vpp, cp, ep: The data, tensor, pipeline, virtual
            pipeline, context and expert parallel degrees.
        etp: The expert tensor-parallel degree; at most 1 means an expert
            layer runs with the dense layers' tensor parallelism.
        sequence_parallel: Whether activations are split along the sequence
            over the tensor-parallel group.
        shard_activations: Whether tensor parallelism shards the activations
            between layers, which a fully recomputed layer keeps as its
            input, and the output layer's.
        micro_batch_size, micro_batch_num, global_batch_size: The batch.
        optimizer_parallel: Whether optimizer states are sharded.
        optimizer_shard: How many ways optimizer states are sharded.
        grad_shard: Whether gradients are sharded too.
        grad_shard_as_params: Whether each gradient is sharded as its
            parameter is, as FSDP holds it; ``grad_shard`` does not apply.
        grad_accumulation: Whether gradients take memory without pipeline
            parallelism too; under it they always do.
        pp_schedule: The pipeline schedule, such as ``1f1b``.
        offset: How many layers each pipeline stage holds beyond an even
            split, per stage or per chunk and stage.
        seq_split: How many parts a sequence is split into for a sequence
            pipeline.
        pp_partition: The layers each stage holds, when stated outright.
        mtp_in_offset, emb_out_in_offset: Whether the MTP layers, and the
            embedding and output layers, count in the offset's layers.
        full_recompute, selective_recompute: Which layers are recomputed in
            full or selectively: true for all, or counts per stage.
        selective_comm_recompute: Which layers recompute their
            sequence-parallel all-gather, under the MindFormers rule.
        selective_rule: Whose selective recompute the run uses, one of
            :data:`SELECTIVE_RULES`.
        recompute_slice_activation: Whether a recomputed layer keeps its
            input sliced over tensor parallelism.
        param_bytes, compute_bytes, softmax_bytes, grad_bytes,
            optimizer_state_bytes, norm_bytes, dropout_bytes: Bytes per
            element of the parameters, activations, softmax outputs,
            gradients, optimizer states, norm activations and dropout masks.
        flash_attention, grad_clip, grouped_gemm, tie_embeddings: Kernels
            and features the run uses.
        vocab_emb_dp: Whether the vocabulary embedding runs data parallel.
        emb_dp_sharded: Whether the embedding table is sharded over data
            parallelism.
        cp_algo: The context-parallel algorithm.
        capacity_factor: The experts' token capacity factor.
        shard_mtp_param: Whether the MTP layers' parameters are sharded.
        frozen: Whether the model's parameters are frozen.
        optimizer: The optimizer's name.
        seq_length: The training sequence length.
        device_memory: The device's memory, such as ``"64GB"``.
    """

    dp: Optional[int] = None
    tp: Optional[int] = None
    pp: Optional[int] = None
    vpp: Optional[int] = None
    cp: Optional[int] = None
    ep: Optional[int] = None
    etp: Optional[int] = None
    sequence_parallel: Optional[bool] = None
    shard_activations: Optional[bool] = None

    micro_batch_size: Optional[int] = None
    micro_batch_num: Optional[int] = None
    global_batch_size: Optional[int] = None

    optimizer_parallel: Optional[bool] = None
    optimizer_shard: Optional[int] = None
    grad_shard: Optional[bool] = None
    grad_shard_as_params: Optional[bool] = None
    grad_accumulation: Optional[bool] = None

    pp_schedule: Optional[str] = None
    offset: Optional[Union[int, list]] = None
    seq_split: Optional[int] = None
    pp_partition: Optional[list] = None
    mtp_in_offset: Optional[bool] = None
    emb_out_in_offset: Optional[bool] = None

    full_recompute: Optional[Union[bool, int, list]] = None
    selective_recompute: Optional[Union[bool, list]] = None
    selective_comm_recompute: Optional[Union[bool, list]] = None
    selective_rule: Optional[str] = None
    recompute_slice_activation: Optional[bool] = None

    param_bytes: Optional[int] = None
    compute_bytes: Optional[int] = None
    softmax_bytes: Optional[int] = None
    grad_bytes: Optional[int] = None
    optimizer_state_bytes: Optional[int] = None
    norm_bytes: Optional[int] = None
    dropout_bytes: Optional[int] = None

    flash_attention: Optional[bool] = None
    grad_clip: Optional[bool] = None
    grouped_gemm: Optional[bool] = None
    vocab_emb_dp: Optional[bool] = None
    emb_dp_sharded: Optional[bool] = None
    tie_embeddings: Optional[bool] = None
    cp_algo: Optional[str] = None
    capacity_factor: Optional[float] = None
    shard_mtp_param: Optional[bool] = None
    frozen: Optional[bool] = None
    optimizer: Optional[str] = None

    seq_length: Optional[int] = None
    device_memory: Optional[Any] = None

    # ---- validation ----------------------------------------------------

    def validate(self) -> "ExecSpec":
        """Raise :class:`ExecSpecError` unless every stated value can be right.

        Returns *self*, so a producer can ``return spec.validate()``.
        """
        for name in _KINDS["count"]:
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ExecSpecError(f"{name} must be at least 1, got {value!r}")
        for name in _KINDS["size"]:
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ExecSpecError(f"{name} must not be negative, got {value!r}")
        if self.selective_rule is not None and self.selective_rule not in SELECTIVE_RULES:
            raise ExecSpecError(
                f"selective_rule must be one of {list(SELECTIVE_RULES)}, got {self.selective_rule!r}"
            )
        if self.capacity_factor is not None and self.capacity_factor <= 0:
            raise ExecSpecError(f"capacity_factor must be positive, got {self.capacity_factor!r}")
        return self

    # ---- serialised form -----------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the stated fields as a plain mapping, ready for YAML."""
        out: Dict[str, Any] = {}
        for spec_field in fields(self):
            value = getattr(self, spec_field.name)
            if value is None:
                continue
            out[spec_field.name] = value if spec_field.name != "device_memory" else str(value)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecSpec":
        """Build a validated spec from a mapping, refusing an unknown key.

        Raises:
            ExecSpecError: If a key is unknown, a value has the wrong type,
                or a stated value cannot be right.
        """
        if not isinstance(data, Mapping):
            raise ExecSpecError(f"an execution spec must be a mapping, got {data!r}")
        known = [spec_field.name for spec_field in fields(cls)]
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ExecSpecError(f"unknown execution spec keys {unknown}; the keys are {known}")
        kwargs = {key: _coerce(key, value) for key, value in data.items()}
        return cls(**kwargs).validate()


# The fields by kind, for coercion from YAML and for validation; a field in
# none of them is kept as it is.
_KINDS: Dict[str, tuple] = {
    "count": ("dp", "tp", "pp", "vpp", "cp", "ep", "micro_batch_size", "micro_batch_num",
              "global_batch_size", "optimizer_shard", "seq_split", "seq_length"),
    "size": ("etp", "param_bytes", "compute_bytes", "softmax_bytes", "grad_bytes",
             "optimizer_state_bytes", "norm_bytes", "dropout_bytes"),
    "flag": ("sequence_parallel", "shard_activations", "optimizer_parallel", "grad_shard",
             "grad_shard_as_params", "grad_accumulation", "mtp_in_offset", "emb_out_in_offset",
             "recompute_slice_activation", "flash_attention", "grad_clip", "grouped_gemm",
             "vocab_emb_dp", "emb_dp_sharded", "tie_embeddings", "shard_mtp_param", "frozen"),
    "name": ("pp_schedule", "selective_rule", "cp_algo", "optimizer", "device_memory"),
}


def _coerce(key: str, value: Any) -> Any:
    """Coerce one field read from YAML to the type the schema gives it."""
    if value is None:
        return None
    if key in _KINDS["count"] or key in _KINDS["size"]:
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ExecSpecError(f"{key} must be a whole number, got {value!r}")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ExecSpecError(f"{key} must be a whole number, got {value!r}") from exc
    if key in _KINDS["flag"]:
        if not isinstance(value, bool):
            raise ExecSpecError(f"{key} must be true or false, got {value!r}")
        return value
    if key in _KINDS["name"]:
        return str(value)
    if key == "capacity_factor":
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ExecSpecError(f"capacity_factor must be a number, got {value!r}") from exc
    return value
