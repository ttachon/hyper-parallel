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
"""Tests for the arch hooks reading their op counts from data.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_arch_hooks.py -v
"""
import copy
import math
import os
import unittest
from types import SimpleNamespace
from typing import Any

import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
from hyper_parallel.auto_parallel._layer_stack import LayerStack, StackGroup, derive_layers, resolve_layers
from hyper_parallel.auto_parallel._model_spec import ModelSpecError, OpCounts
from hyper_parallel.auto_parallel._op_profiles import LayerKind, known_archs, load_op_profile, resolve_ops
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.layer_block import EvalAttn
from hyper_parallel.auto_parallel.sapp_nd.nd.common.arch_hooks import (
    ARCH_HOOKS,
    CWrap,
    apply_layer_kind,
    check_and_apply_custom_hook,
    custom_vision_tower,
    stack_layer_groups,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import CostModelConfig
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import (
    estimate_comp,
    estimate_performance,
    op_table,
)
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.getters import get_layer_custom_configs
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.utils_classes import CustomConfig

_SAPP_ND = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    *[os.pardir] * 5, "hyper_parallel", "auto_parallel", "sapp_nd",
)

_OPS = ("attMM", "attBMM", "ffMM", "softmax", "dropout", "normOp", "gather")

# The literal op counts each arch hook assigned before the counts became
# profiles, in _OPS order, except that mixtral's three expert projections
# were counted as batched matmuls and now count as the matmuls they are.
# headCast and ffAct were pinned to 1 by the estimator for every family.
_LEGACY_OPS = {
    "default": {"decoder": (4, 2, 3, 1, 0, 2, 4)},
    "llama2": {"decoder": (4, 2, 3, 1, 0, 2, 4)},
    "qwen": {"decoder": (4, 2, 3, 1, 0, 2, 4)},
    # Qwen3.5 was priced with the qwen hook; its full-attention kind keeps those counts.
    "qwen3_5": {"full_attention": (4, 2, 3, 1, 0, 2, 4)},
    "deepseek": {"decoder": (4, 2, 3, 1, 0, 2, 4)},
    "cm": {"decoder": (4, 2, 3, 1, 0, 2, 4)},
    "mixtral": {"decoder": (4, 2, 3, 2, 0, 5, 4)},
    "pangualpha": {"decoder": (4, 1, 2, 2, 5, 4, 4)},
    "t5": {"encoder": (4, 1, 2, 2, 5, 2, 4), "decoder": (8, 2, 2, 4, 7, 3, 6)},
    "vision": {"encoder": (4, 2, 2, 1, 0, 2, 4)},
}

# The byte widths the same hooks set: (bytes_grad at p > 1, at p == 1, bytes_dropout).
_LEGACY_BYTES = {
    "default": (4, 0, 0), "llama2": (2, 2, 0), "qwen": (4, 0, 0), "qwen3_5": (4, 0, 0),
    "deepseek": (4, 0, 0), "cm": (4, 0, 0), "mixtral": (2, 0, 0),
    "pangualpha": (4, 0, 1), "t5": (4, 0, 1), "vision": (4, 0, 0),
}


def _counts(**overrides) -> dict:
    """The default decoder's op counts, with *overrides* applied."""
    counts = dict(zip(_OPS, _LEGACY_OPS["default"]["decoder"]), headCast=1, ffAct=1, linrec=0)
    counts.update(overrides)
    return counts


def _hyper_config(name: str = "unit", **model) -> dict:
    """A small dense AutoModels config the hyper_v2 parser reads offline."""
    overrides = {
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "intermediate_size": 2816,
        "vocab_size": 32000,
        "max_position_embeddings": 2048,
    }
    overrides.update(model)
    return {
        "model": {"name": name, "config_overrides": overrides},
        "training": {"global_batch_size": 8, "micro_batch_size": 1},
        "accelerator": {"tp_size": 2, "pp_size": 1},
        "fsdp_config": {"dp_shard_size": 4},
        "dataset": {"data_transform": {"max_seq_len": 2048}},
        "context": {"max_device_memory": "64GB", "device_num": 8},
    }


def _ccfg(name: str = "unit", **model) -> CostModelConfig:
    """Parse :func:`_hyper_config` into a cost-model config."""
    return CostModelConfig(_hyper_config(name, **model), framework="hyper_v2")


def _bare(arch: str, op_counts=None, **fields) -> SimpleNamespace:
    """A config with just what the hooks read, as a test double would build."""
    cfg = SimpleNamespace(
        model_name="unit", arch=arch, op_counts=op_counts, has_op=False, p=2,
        t=2, h=16, dh=4, hff=32, hff_exp=64, n_lay=4, n_mtp=1, n_chosen_exp=2,
        n_exp=4, n_shared_exp=1, ep=2, k_1st_dense=2, config_format="yaml",
        shard_p_os_exp_partial=2, shard_p_os_non_exp=2, shard_embed=1,
        overwrite_eval_functions={},
    )
    for key, value in fields.items():
        setattr(cfg, key, value)
    return cfg


def _state(cfg) -> dict:
    """The op counts and byte widths a hook leaves on *cfg*."""
    names = [f"n_{op}" for op in _OPS] + [
        "n_headCast", "n_ffAct", "n_attParamCast", "n_ffParamCast",
        "bytes_grad", "bytes_os", "bytes_dropout", "bytes_norm",
    ]
    return {name: getattr(cfg, name) for name in names}


def _legacy_state(arch: str, kind: str, has_op: bool, p: int) -> dict:
    """What the literal callback of *arch* left for one layer kind."""
    counts = dict(zip(_OPS, _LEGACY_OPS[arch][kind]))
    grad_pp, grad_no_pp, dropout = _LEGACY_BYTES[arch]
    state = {f"n_{op}": count for op, count in counts.items()}
    state.update(
        n_headCast=1, n_ffAct=1,
        n_attParamCast=0 if has_op else counts["attMM"],
        n_ffParamCast=0 if has_op else counts["ffMM"],
        bytes_grad=grad_pp if p > 1 else grad_no_pp,
        bytes_os=4, bytes_dropout=dropout, bytes_norm=4,
    )
    return state


class TestDispatchReadsTheArch(unittest.TestCase):
    """The hook is chosen by the declared arch, never by the model name."""

    def test_every_profile_has_a_hook(self):
        """
        Feature: arch hook registry.
        Description: A profile without a hook could never be applied, and a
            hook without a profile would have no counts to read.
        Expectation: The registry and the profile files name the same archs.
        """
        self.assertEqual(set(ARCH_HOOKS), set(known_archs()))

    def test_the_model_name_is_not_read(self):
        """
        Feature: dispatch on ccfg.arch.
        Description: A config named like a Qwen model but declaring the
            default arch must not get the Qwen hook's activation sharding.
        Expectation: shard_output_activ stays 1 under tp=2.
        """
        ccfg = _ccfg("qwen2_72b", arch="default")
        check_and_apply_custom_hook(ccfg)
        self.assertEqual(ccfg.arch, "default")
        self.assertEqual(ccfg.shard_output_activ, 1)

    def test_the_declared_arch_is_read(self):
        """
        Feature: dispatch on ccfg.arch.
        Description: A config with a neutral name declaring the qwen arch gets
            the qwen hook.
        Expectation: shard_output_activ follows tp=2.
        """
        ccfg = _ccfg("unit", arch="qwen")
        check_and_apply_custom_hook(ccfg)
        self.assertEqual(ccfg.shard_output_activ, 2)

    def test_a_config_with_no_arch_gets_the_default_counts(self):
        """
        Feature: dispatch fallback.
        Description: A config assembled without a parser carries no arch.
        Expectation: It is priced with the default decoder's counts.
        """
        cfg = _bare(arch=None)
        check_and_apply_custom_hook(CWrap(cfg))
        self.assertEqual(_state(cfg), _legacy_state("default", "decoder", False, 2))


class TestHooksMatchTheirLiterals(unittest.TestCase):
    """Reading counts from profiles leaves each config as the literals did."""

    def _check(self, arch: str, with_table: bool) -> None:
        """Assert *arch*'s hook leaves every layer kind as its literals did."""
        for has_op in (False, True):
            for p in (1, 2):
                with self.subTest(arch=arch, table=with_table, has_op=has_op, p=p):
                    table = resolve_ops(arch) if with_table else None
                    cfg = _bare(arch, table, has_op=has_op, p=p)
                    cfg.layer_stack = resolve_layers(arch, derive_layers(load_op_profile(arch), cfg.n_lay), table)
                    check_and_apply_custom_hook(CWrap(cfg))
                    if arch != "t5":
                        kind = next(iter(_LEGACY_OPS[arch]))
                        self.assertEqual(_state(cfg), _legacy_state(arch, kind, has_op, p))
                        continue
                    # t5 sets its counts and byte widths per layer, from its stack's kinds.
                    for kind, (count, hook) in zip(("encoder", "decoder"), stack_layer_groups(cfg.layer_stack, 4)):
                        layer = copy.deepcopy(cfg)
                        hook(CWrap(layer))
                        self.assertEqual(count, cfg.n_lay // 2)
                        self.assertEqual(_state(layer), _legacy_state(arch, kind, has_op, p))

    def test_counts_from_the_parser(self):
        """
        Feature: hooks read the counts the parser recorded.
        Description: Apply every family's hook to a config carrying its
            resolved profile, for each optimizer-sharding and pipeline case.
        Expectation: Counts and byte widths equal the old literals.
        """
        for arch in known_archs():
            self._check(arch, with_table=True)

    def test_counts_without_a_parser(self):
        """
        Feature: hooks fall back on their own profile.
        Description: Same, on a config that carries no counts at all.
        Expectation: Counts and byte widths equal the old literals.
        """
        for arch in known_archs():
            self._check(arch, with_table=False)

    def test_the_vision_tower_uses_its_own_profile(self):
        """
        Feature: vision tower counts.
        Description: A tower's config carries its language model's arch and
            counts; the tower hook must not price it as that decoder.
        Expectation: The tower's two-projection feed-forward, not three.
        """
        cfg = _bare("qwen", resolve_ops("qwen"))
        custom_vision_tower(cfg)
        self.assertEqual(cfg.n_ffMM, 2)


class TestDeclaredCountsReachTheEstimate(unittest.TestCase):
    """A count the spec declares is the count the estimators price."""

    def _perf(self, ccfg) -> float:
        return estimate_performance(copy.deepcopy(ccfg), device_type=Hard.Device_A2)

    def test_a_declared_count_is_applied(self):
        """
        Feature: counts from the spec.
        Description: Declare a decoder with no softmax in config_overrides.
        Expectation: The hook applies 0, and the FLOP estimate drops.
        """
        base = _ccfg(arch="default")
        declared = _ccfg(arch="default", ops={"decoder": _counts(softmax=0)})
        check_and_apply_custom_hook(declared)
        self.assertEqual(declared.n_softmax, 0)
        self.assertLess(self._perf(declared), self._perf(base))

    def test_head_cast_is_no_longer_pinned(self):
        """
        Feature: no hardcoded op counts in the estimator.
        Description: The estimator used to pin headCast and ffAct to 1 for
            every model; they now come from the vector like every other op.
        Expectation: Declaring headCast 0 lowers the FLOP estimate.
        """
        base = _ccfg(arch="default")
        declared = _ccfg(arch="default", ops={"decoder": _counts(headCast=0)})
        self.assertLess(self._perf(declared), self._perf(base))


class TestTwoKindStack(unittest.TestCase):
    """Each layer of a mixed stack is priced from its own kind's vector."""

    _LINEAR = {"attBMM": 0, "softmax": 0, "headCast": 0}

    def _stack(self, linear_layers: int = 2) -> CostModelConfig:
        """Four layers: full attention, then *linear_layers* without scores."""
        ccfg = _ccfg(arch="default")
        groups = (
            StackGroup(LayerKind("decoder", OpCounts.from_dict(_counts())), 4 - linear_layers),
            StackGroup(LayerKind("scoreless", OpCounts.from_dict(_counts(**self._LINEAR))), linear_layers),
        )
        ccfg.layer_stack = LayerStack("default", tuple(group for group in groups if group.count))
        ccfg.layer_custom_config = stack_layer_groups(ccfg.layer_stack, 4)
        return ccfg

    def test_each_group_gets_its_vector(self):
        """
        Feature: per-kind op counts on the performance path.
        Description: The performance path prices a copy of the config per
            layer group.
        Expectation: The groups carry their own kind's counts.
        """
        groups = get_layer_custom_configs(self._stack())
        self.assertEqual([count for _, count in groups], [2, 2])
        full, linear = groups[0][0], groups[1][0]
        self.assertEqual((full.n_attBMM, full.n_softmax, full.n_headCast), (2, 1, 1))
        self.assertEqual((linear.n_attBMM, linear.n_softmax, linear.n_headCast), (0, 0, 0))

    def test_flops_differ_by_exactly_the_missing_ops(self):
        """
        Feature: per-kind op counts on the performance path.
        Description: Price the stack, and the same stack with every layer
            full, on one pipeline stage.
        Expectation: They differ by two layers' worth of the three ops the
            linear kind does not run, and nothing else.
        """
        layers = [LayerType.NOT_REC_LAYER] * 4
        stages = [[[LayerType.EMBEDDING_LAYER] + layers + [LayerType.OUTPUT_LAYER]]]
        mixed = self._stack(linear_layers=2)
        full = self._stack(linear_layers=0)
        check_and_apply_custom_hook(mixed)
        check_and_apply_custom_hook(full)
        table = op_table(full)
        missing = 2 * (2 * table["n_attBMM"] + table["n_softmax"] + table["n_headCast"])
        saved = (estimate_comp(full, CustomConfig(), stages)[0]
                 - estimate_comp(mixed, CustomConfig(), stages)[0])
        self.assertAlmostEqual(saved / missing, 1.0, places=9)

    def test_memory_path_prices_each_layer_from_its_kind(self):
        """
        Feature: per-kind op counts on the memory path.
        Description: The memory backbone applies each group's hook to the
            layers it covers.  Record the softmax count the attention score
            formula sees at every layer.
        Expectation: Two layers with the full kind's 1, then two with 0.
        """
        seen = {}

        def spy(ccfg: CostModelConfig, ctx: Any) -> float:
            """The real score formula, recording the counts it was given."""
            seen.setdefault(ctx.current_lay_id, set()).add(ccfg.n_softmax)
            return EvalAttn.attn_score_activations(ccfg, ctx)

        evaluator = EvaluatorV2(None, ccfg=self._stack())
        evaluator.set_attn_eval_fun(score=spy)
        evaluator.estimate_peak()
        per_layer = [seen[lay_id] for lay_id in sorted(seen)]
        self.assertEqual(per_layer, [{1}, {1}, {0}, {0}])


_LINEAR_DIMS = {
    "linear_num_key_heads": 8, "linear_key_head_dim": 64, "linear_num_value_heads": 16,
    "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4,
}
_L, _F = "linear_attention", "full_attention"


def _qwen35_model(layer_types, **model) -> dict:
    """The config_overrides of a small Qwen3.5-shaped model with *layer_types*."""
    fields = {
        "num_hidden_layers": len(layer_types), "layer_types": list(layer_types), "head_dim": 128,
        "num_key_value_heads": 2, "attn_output_gate": True, **_LINEAR_DIMS,
    }
    fields.update(model)
    return fields


class TestStackAsData(unittest.TestCase):
    """A stack the parser settled as data prices each layer by its kind."""

    def test_the_stack_comes_from_layer_types(self):
        """
        Feature: Qwen3.5 layer stack.
        Description: layer_types lists [L, L, F, L, F].
        Expectation: The qwen3_5 profile, one group per run, one hook per group.
        """
        ccfg = _ccfg("qwen3_5_moe", **_qwen35_model([_L, _L, _F, _L, _F]))
        self.assertEqual(ccfg.arch, "qwen3_5")
        self.assertEqual([(group.kind.name, group.count) for group in ccfg.layer_stack.groups],
                         [(_L, 2), (_F, 1), (_L, 1), (_F, 1)])
        self.assertEqual([count for count, _ in ccfg.layer_custom_config], [2, 1, 1, 1])

    def test_a_full_layer_restores_the_models_own_attention(self):
        """
        Feature: complete assignments.
        Description: Apply a linear kind then a full one in place, as the
            memory backbone does layer after layer.
        Expectation: Exactly the attention the family hook left.
        """
        ccfg = _ccfg("qwen3_5_moe", **_qwen35_model([_L, _F]))
        check_and_apply_custom_hook(ccfg)
        kinds = {kind.name: kind for kind in ccfg.layer_stack.distinct_kinds()}
        names = ("attn_kind", "a", "dh", "n_kv", "attn_output_gate", "attn_extra_p",
                 "n_attBMM", "n_softmax", "n_headCast", "n_linrec", "lin_n_v")
        before = {name: getattr(ccfg, name) for name in names}
        layer = copy.deepcopy(ccfg)
        apply_layer_kind(layer, kinds[_L])
        apply_layer_kind(layer, kinds[_F])
        self.assertEqual({name: getattr(layer, name) for name in names}, before)

    def test_a_linear_layer_is_priced_on_the_linear_dimensions(self):
        """
        Feature: linear attention.
        Description: Apply the linear kind of a model with 16 value heads of 64.
        Expectation: The value heads carry the q side, the key heads the kv
            side, no score ops, one state update, and the convolution and
            gates as extra parameters.
        """
        ccfg = _ccfg("qwen3_5_moe", **_qwen35_model([_L, _F]))
        check_and_apply_custom_hook(ccfg)
        apply_layer_kind(ccfg, ccfg.layer_stack.groups[0].kind)
        self.assertEqual(
            (ccfg.attn_kind, ccfg.a, ccfg.dh, ccfg.n_kv, ccfg.attn_output_gate,
             ccfg.n_softmax, ccfg.n_linrec, ccfg.attn_extra_p),
            ("linear", 16, 64, 8.0, True, 0, 1, 4 * (2 * 8 * 64 + 16 * 64) + 2 * 1024 * 16),
        )

    def test_a_one_kind_stack_needs_no_hook(self):
        """
        Feature: layer groups.
        Description: A dense model with one MTP layer.
        Expectation: One group covering all five layers, with no hook.
        """
        self.assertEqual(_ccfg("qwen3_moe", mtp_depth=1).layer_custom_config, [(5, None)])

    def test_a_linear_kind_without_dimensions_is_refused(self):
        """
        Feature: linear attention.
        Description: layer_types names linear layers, but no linear dimension
            is declared.
        Expectation: Refused at parse time rather than priced as full attention.
        """
        with self.assertRaises(ModelSpecError) as ctx:
            _ccfg("qwen3_5_moe", num_hidden_layers=4, layer_types=[_L, _F, _L, _F])
        self.assertIn("linear_num_value_heads", str(ctx.exception))

    def test_a_stated_stack_prices_as_the_derived_one(self):
        """
        Feature: layers in the serialised spec.
        Description: The same model three ways: layer_types, the equivalent
            stated layers, and a stack of full attention only.
        Expectation: The first two give the same peak memory and FLOP score,
            and the third does not.
        """
        derived = _qwen35_model([_L, _L, _L, _F])
        stated = dict(derived, arch="qwen3_5", layers=[{"kind": _L, "count": 3}, {"kind": _F, "count": 1}])
        del stated["layer_types"]
        full = dict(stated, layers=[{"kind": _F, "count": 4}])
        results = []
        for model in (derived, stated, full):
            config = _hyper_config("qwen3_5_moe", **model)
            peak = EvaluatorV2(config, framework="hyper_v2", log_level=0).estimate_peak()
            score = estimate_performance(CostModelConfig(config, framework="hyper_v2"),
                                         device_type=Hard.Device_A2)
            results.append((peak, score))
        self.assertEqual(results[0], results[1])
        self.assertNotEqual(results[0][1], results[2][1])


_DEEPSEEK = {
    "num_hidden_layers": 4, "first_k_dense_replace": 1, "mtp_depth": 1, "num_experts": 8,
    "num_experts_per_tok": 2, "num_shared_experts": 1, "moe_intermediate_size": 512,
}


def _deepseek(name: str = "deepseek_v3") -> CostModelConfig:
    """A small DeepSeek-shaped model at ep 2: one dense layer, three MoE and one MTP."""
    config = _hyper_config(name, **_DEEPSEEK)
    config["accelerator"]["ep_size"] = 2
    return CostModelConfig(config, framework="hyper_v2")


class TestDenseThenMoE(unittest.TestCase):
    """DeepSeek's dense and MoE layers, cm's and t5's halves are kinds of the stack."""

    _FFN = ("hff", "n_chosen_exp", "n_exp", "n_shared_exp", "ep")

    def test_the_stack_comes_from_first_k_dense_replace(self):
        """
        Feature: DeepSeek layer stack.
        Description: One dense layer, three MoE layers and one MTP layer.
        Expectation: Three groups, the MTP one repeating the MoE kind, one hook each.
        """
        ccfg = _deepseek()
        self.assertEqual([(group.kind.name, group.count, group.mtp) for group in ccfg.layer_stack.groups],
                         [("dense", 1, False), ("moe", 3, False), ("moe", 1, True)])
        self.assertEqual([count for count, _ in ccfg.layer_custom_config], [1, 3, 1])

    def test_each_kind_assigns_the_whole_feed_forward(self):
        """
        Feature: complete assignments.
        Description: Apply the dense kind, a MoE one, then the dense one
            again, in place, as the memory backbone does.
        Expectation: A dense layer runs one expert at the dense width, no
            shared one and no expert parallelism; a MoE layer runs the model's
            experts at their width and ep; and the dense layers match.
        """
        ccfg = _deepseek()
        check_and_apply_custom_hook(ccfg)
        dense, moe = (group.kind for group in ccfg.layer_stack.groups[:2])
        layer = copy.deepcopy(ccfg)
        states = []
        for kind in (dense, moe, dense):
            apply_layer_kind(layer, kind)
            states.append({name: getattr(layer, name) for name in self._FFN})
        self.assertEqual(states[0], {"hff": 2816, "n_chosen_exp": 1, "n_exp": 1, "n_shared_exp": 0, "ep": 1})
        self.assertEqual(states[1], {"hff": 512, "n_chosen_exp": 2, "n_exp": 8, "n_shared_exp": 1, "ep": 2})
        self.assertEqual(states[2], states[0])

    def test_ep_is_written_past_the_evaluators_guard(self):
        """
        Feature: strategy a kind sets.
        Description: The memory backbone applies kinds through an evaluator,
            whose guard refuses a strategy write inside a hook.
        Expectation: A dense layer still runs at ep 1, and a MoE one at 2.
        """
        evaluator = EvaluatorV2(None, ccfg=_deepseek())
        dense, moe = (group.kind for group in evaluator.ccfg.layer_stack.groups[:2])
        apply_layer_kind(evaluator, dense)
        self.assertEqual(evaluator.ccfg.ep, 1)
        apply_layer_kind(evaluator, moe)
        self.assertEqual(evaluator.ccfg.ep, 2)

    def test_cm_shards_every_layer_as_its_hook_says(self):
        """
        Feature: fields the family gives every layer.
        Description: cm runs DeepSeek's stack with its own optimizer and
            embedding sharding on each layer.
        Expectation: Both kinds assign the sharding cm's hook derived.
        """
        ccfg = _deepseek("cm_llama_moe")
        self.assertEqual(ccfg.arch, "cm")
        check_and_apply_custom_hook(ccfg)
        expected = (ccfg.shard_p_os_exp_partial, math.gcd(ccfg.n_exp, ccfg.shard_p_os_non_exp), ccfg.t)
        for kind in ccfg.layer_stack.distinct_kinds():
            layer = copy.deepcopy(ccfg)
            apply_layer_kind(layer, kind)
            self.assertEqual((layer.shard_p_os_exp, layer.shard_p_os_non_exp_partial, layer.shard_embed),
                             expected, kind.name)

    def test_t5_prices_each_half_as_its_kind(self):
        """
        Feature: t5 layer stack.
        Description: A four-layer t5.
        Expectation: Two encoder then two decoder layers, each with its own
            counts and t5's one-byte dropout mask, which the model does not
            take.
        """
        ccfg = _ccfg("t5_small")
        self.assertEqual([(group.kind.name, group.count) for group in ccfg.layer_stack.groups],
                         [("encoder", 2), ("decoder", 2)])
        check_and_apply_custom_hook(ccfg)
        self.assertEqual(ccfg.bytes_dropout, 0)
        for kind, attention_matmuls in zip(ccfg.layer_stack.distinct_kinds(), (4, 8)):
            layer = copy.deepcopy(ccfg)
            apply_layer_kind(layer, kind)
            self.assertEqual((layer.n_attMM, layer.bytes_dropout), (attention_matmuls, 1), kind.name)


class TestParsersRecordTheArch(unittest.TestCase):
    """Every parser settles the arch before any hook runs."""

    def test_mindformers_infers_from_the_model_name(self):
        """
        Feature: MindFormers parser.
        Description: MindFormers configs name the model in free text only.
        Expectation: deepseekV3 resolves to the deepseek profile.
        """
        ccfg = CostModelConfig(os.path.join(_SAPP_ND, "nd", "yamls", "deepseek.yaml"))
        self.assertEqual(ccfg.arch, "deepseek")
        self.assertEqual(ccfg.op_counts, resolve_ops("deepseek"))

    def test_mindformers_derives_the_stack(self):
        """
        Feature: MindFormers parser.
        Description: The DeepSeek yaml declares first_k_dense_replace and one
            MTP layer.
        Expectation: The dense prefix, the MoE body and the MTP layer, as groups.
        """
        ccfg = CostModelConfig(os.path.join(_SAPP_ND, "nd", "yamls", "deepseek.yaml"))
        groups = [(group.kind.name, group.count, group.mtp) for group in ccfg.layer_stack.groups]
        self.assertEqual(groups, [("dense", ccfg.k_1st_dense, False),
                                  ("moe", ccfg.n_lay - ccfg.k_1st_dense, False),
                                  ("moe", ccfg.n_mtp, True)])
        self.assertEqual([count for count, _ in ccfg.layer_custom_config], [count for _, count, _ in groups])

    def test_hyper_reads_the_spec(self):
        """
        Feature: hyper_v2 parser.
        Description: The resolved spec states its arch, inferred or declared.
        Expectation: The config carries it, with the declared counts if any.
        """
        self.assertEqual(_ccfg("qwen3_moe").arch, "qwen")
        declared = _ccfg("qwen3_moe", arch="default", ops={"decoder": _counts(gather=6)})
        self.assertEqual(declared.arch, "default")
        self.assertEqual(declared.op_counts["decoder"].gather, 6)


if __name__ == "__main__":
    unittest.main()
