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
"""Verify mode: the parameters ND prices of a model, beside those Transformers builds of it.

``run_nd -f hyper_v2 -y <train.yaml> -V`` builds the first layer of each
kind of the model the train.yaml trains, on fake tensors, and counts its
parameters part by part
(:func:`~hyper_parallel.auto_parallel._layer_census.census_parameters`).
Beside each part it sets what ND's formulas price of it, and the same for
the embedding and the output layer.  It reports each part's two counts and
their difference, rather than a pass or a fail: a fact the parser misread
shows as the part it feeds.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, NamedTuple

import yaml

from hyper_parallel.auto_parallel._hf_model_spec import checkpoint_configs, is_auto_models_schema
from hyper_parallel.auto_parallel._layer_census import census_final_norm, census_parameters
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.head import EvalHead
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.tail import EvalTail
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import check_and_apply_custom_hook
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.comm_time import prepare_context
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.getters import get_layer_custom_configs


class VerifyRow(NamedTuple):
    """One part's parameters: ND's count, the census's, and how many layers hold it."""

    where: str
    part: str
    nd: float
    census: float
    count: int = 1


def nd_parameters(lccfg: Any, ctx: Any) -> Dict[str, float]:
    """The parameters ND prices of a layer of *lccfg*, by part, with the formulas *ctx* names."""
    parts = {"attention": ctx.attn_num_p(lccfg, ctx), "norm": ctx.norm_num_p(lccfg, ctx)}
    if lccfg.n_exp == 1:
        parts["ffn"] = ctx.ffn_num_p(lccfg, ctx)
    else:
        parts["routed"] = ctx.ffn_routed_num_p(lccfg, ctx)
        parts["shared"] = ctx.ffn_shared_num_p(lccfg, ctx)
        parts["router"] = 0.0
    return parts


def _groups(ccfg: Any) -> List[Any]:
    """``(kind, count, config)`` per group of *ccfg*'s layer stack, in model order, as its hooks name the kinds."""
    hooks = ccfg.layer_custom_config or [(ccfg.n_lay, None)]
    configs = get_layer_custom_configs(ccfg)
    if len(configs) != len(hooks):
        hooks = [(count, None) for _, count in configs]
    return [(getattr(hook, "kind", None) or "decoder", count, lccfg)
            for (_, hook), (lccfg, count) in zip(hooks, configs)]


def _text_model(ccfg: Any) -> Any:
    """The config of *ccfg*'s language model: the config itself, or a multimodal config's main submodule's."""
    if not getattr(ccfg, "multimodal", False):
        return ccfg
    return ccfg.mm_ccfgs[getattr(ccfg, "mm_main", None) or ccfg.mm_order[-1]]


def verify_parameters(yaml_path: str) -> List[VerifyRow]:
    """The parameters ND prices of the model a train.yaml trains, part by part, beside the census's.

    Args:
        yaml_path: An AutoModels train.yaml, whose model names a Transformers
            checkpoint.

    Returns:
        A row per part of the first layer of each kind of its language
        model, then the embedding's, the output layer's, and the model's
        whole.  A table the embedding and the output layer share on one
        stage is counted at the output, as ND prices it.

    Raises:
        ValueError: The train.yaml names no Transformers checkpoint.
    """
    with open(yaml_path, encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict) or not is_auto_models_schema(raw) or not isinstance(raw.get("model"), dict):
        raise ValueError(f"{yaml_path}: verify mode needs an AutoModels train.yaml naming a Transformers checkpoint")
    config, text = checkpoint_configs(raw["model"])
    # The family's op counts and fields, as the estimates apply them, on a copy.
    ccfg = copy.deepcopy(_text_model(CostModelConfig(yaml_path, framework="hyper_v2")))
    check_and_apply_custom_hook(ccfg)
    ctx = prepare_context()
    kinds: Dict[str, List[Any]] = {}
    first = 0
    for name, count, lccfg in _groups(ccfg):
        kinds.setdefault(name, [first, 0, lccfg])[1] += count
        first += count
    rows = []
    for name, (index, count, lccfg) in kinds.items():
        nd, census = nd_parameters(lccfg, ctx), census_parameters(text, index)
        rows += [VerifyRow(f"{name} x{count}", part, nd.get(part, 0.0), census.get(part, 0), count)
                 for part in sorted(set(nd) | set(census))]
    table = text.vocab_size * text.hidden_size
    shared = bool(getattr(config, "tie_word_embeddings", False)) and ccfg.p <= 1
    rows.append(VerifyRow("embedding", "table", 0.0 if EvalHead.shares_output_table(ccfg)
                          else EvalHead.num_params_embed(ccfg, ctx), 0 if shared else table))
    rows.append(VerifyRow("output", "table, norm", EvalTail.num_params_output(ccfg, ctx),
                          table + census_final_norm(text)))
    rows.append(VerifyRow("model", "total", sum(row.count * row.nd for row in rows),
                          sum(row.count * row.census for row in rows)))
    return rows


def report(rows: List[VerifyRow]) -> List[str]:
    """The lines of a verify report: each part's two counts, their difference and its share of the census's."""
    lines = [f"{'where':26s} {'part':12s} {'ND':>17s} {'census':>17s} {'ND - census':>15s} {'%':>9s}"]
    for row in rows:
        difference = row.nd - row.census
        share = f"{100 * difference / row.census:+8.3f}%" if row.census else ""
        lines.append(f"{row.where:26s} {row.part:12s} {row.nd:17,.0f} {row.census:17,.0f} "
                     f"{difference:+15,.0f} {share}")
    return lines
