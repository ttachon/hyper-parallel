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
"""Tests for applying an execution spec to a cost-model config.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_apply_exec.py -v
"""
import copy
import dataclasses
import os
import tempfile
import unittest
from typing import Any, Dict

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.hook_base import MemEvalHook, hook_runner
from hyper_parallel.auto_parallel._exec_spec import ExecSpec, RecomputeRange
from hyper_parallel.auto_parallel.sapp_nd.nd.common.apply_exec import (
    apply_exec,
    apply_layer_strategy,
    exec_of,
    strategy_exec,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_order import get_model_order
from hyper_parallel.auto_parallel.sapp_nd.nd.common.derive import HYPER_SELECTIVE_REC_OP
from hyper_parallel.auto_parallel.sapp_nd.nd.common.cost_model_preprocess import (
    STRATEGY_GUARDED,
    CostModelConfig,
    arm_strategy_guard,
    strategy_guarded,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_YAMLS = os.path.join(_HERE, *[os.pardir] * 5, "hyper_parallel", "auto_parallel", "sapp_nd", "nd", "yamls")
_SOURCES = (
    (os.path.join(_HERE, "deepseek.yaml"), None),
    (os.path.join(_YAMLS, "hyper_deepseek_v3.yaml"), "hyper_v2"),
    (os.path.join(_YAMLS, "hyper_qwen3_72b.yaml"), "hyper_v2"),
)


# A TorchTitan-style model for the TOML parser: two dense layers, then four
# MoE layers.
_TOML_SOURCE = """def get_train_spec():
    return TrainSpec(model_args=model_args)
model_args = {
    'tiny': ModelArgs(dim=256, inter_dim=1024, hidden_dim=0, vocab_size=4096,
                      n_heads=4, n_layers=6, n_kv_heads=0, kv_lora_rank=0,
                      q_lora_rank=0, qk_rope_head_dim=0, n_dense_layers=2,
                      moe_inter_dim=128, moe_enabled=True,
                      moe_args=MoEArgs(num_experts=8, top_k=2, num_shared_experts=1),
                      enable_weight_tying=False, multiple_of=1, ffn_dim_multiplier=1)
}
"""


def _toml(folder: str) -> CostModelConfig:
    """A TOML config of the model in :data:`_TOML_SOURCE`, on two stages."""
    source = os.path.join(folder, "__init__.py")
    with open(source, "w", encoding="utf-8") as handle:
        handle.write(_TOML_SOURCE)
    return CostModelConfig(Config({
        "model": {"name": "deepseek_v3", "flavor": "tiny"},
        "parallelism": {
            "data_parallel_replicate_degree": 1, "data_parallel_shard_degree": 2,
            "tensor_parallel_degree": 2, "pipeline_parallel_degree": 2, "context_parallel_degree": 1,
            "expert_parallel_degree": 2, "expert_tensor_parallel_degree": 0,
            "pipeline_parallel_schedule": "1F1B",
        },
        "activation_checkpoint": {"mode": "full"},
        "training": {"seq_len": 512, "local_batch_size": 1},
    }), framework="hyperparallel", source_code=source)


def _mindspeed_module(model_id: str, layers: int, **extra: Any) -> Dict[str, Any]:
    """One MindSpeed submodule."""
    module = {
        "model_id": model_id, "freeze": False, "moe_grouped_gemm": False,
        "tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1, "sequence_parallel": False,
        "num_layers": layers, "hidden_size": 256, "ffn_hidden_size": 1024, "vocab_size": 4096,
        "num_attention_heads": 4, "num_query_groups": 0, "kv_channels": 0, "k_lora_rank": 0,
        "q_lora_rank": 0, "qk_rope_head_dim": 0, "num_moe_experts": 1, "moe_router_topk": 1,
        "n_shared_exp": 0, "moe_intermediate_size": 0, "first_k_dense_replace": 0,
        "recompute_num_layers": 1, "params_dtype": "bfloat16", "attention_softmax_in_fp32": True,
        "mtp_num_layers": 0,
    }
    module.update(extra)
    return module


def _mindspeed() -> CostModelConfig:
    """A MindSpeed vision tower and DeepSeek text model, with hooks that change nothing."""
    old_registry = MemEvalHook.hook_registry.copy()
    try:
        MemEvalHook.hook_registry = {}

        class _Hooks(MemEvalHook):
            """One hook per submodule."""

            @staticmethod
            @hook_runner("vit")
            def run_hooks(e: Any) -> None:
                """No change."""
                del e

        class _TextHooks(MemEvalHook):
            """The text model's hook."""

            @staticmethod
            @hook_runner("deepseek_v3")
            def run_hooks(e: Any) -> None:
                """No change."""
                del e

        class _BothHooks(_Hooks, _TextHooks):
            """Both submodules."""

        # The config reads its hooks from the registry when it is built.
        return CostModelConfig(Config({
            "model_id": "multi",
            "tmp": {"pp": 2, "mbs": 1, "dp": 2, "tp": 1, "cp": 1, "vpp": 1, "ep": 2, "seqlen": 512, "etp": 0},
            "image_encoder": _mindspeed_module("vit", 2, pipeline_num_layers=[2, 0]),
            "text_decoder": _mindspeed_module(
                "deepseek_v3", 4, pipeline_num_layers=[1, 3], num_moe_experts=8, moe_router_topk=2,
                n_shared_exp=1, moe_intermediate_size=128, first_k_dense_replace=1,
            ),
        }), hook_cls=_BothHooks(), framework="mindspeed")
    finally:
        MemEvalHook.hook_registry = old_registry


def _state(ccfg: Any) -> Dict[str, Any]:
    """The config's fields, declared or set, with the ones held in objects read as values."""
    state = {}
    names = set(vars(ccfg)) | {spec_field.name for spec_field in dataclasses.fields(type(ccfg))}
    for key in sorted(names):
        value = getattr(ccfg, key)
        if isinstance(value, Config):
            value = vars(value)
        elif isinstance(value, Memory):
            value = (value.size, value.unit)
        state[key] = value
    return state


class TestApplyExec(unittest.TestCase):
    """apply_exec writes what an ExecSpec states, and derives the rest."""

    def _round_trip(self, ccfg: Any, name: str) -> None:
        """Apply *ccfg* its own ExecSpec: only the recompute's form may change."""
        before, layers = _state(ccfg), ccfg.generate_partitions_vpp()
        apply_exec(ccfg, exec_of(ccfg))
        after = _state(ccfg)
        changed = sorted(key for key, value in before.items() if value != after[key])
        self.assertEqual(changed, ["recompute_ranges"], f"{name}: fields changed {changed}")
        self.assertEqual(ccfg.generate_partitions_vpp(), layers, name)

    def test_its_own_spec_leaves_a_config_as_it_is(self):
        """
        Feature: exec_of and apply_exec.
        Description: Read back the ExecSpec of a parsed MindFormers config and
            of two Hyper configs, and apply it to the config it came from.
        Expectation: No field changes but the recompute, which comes back
            stated as ranges, and every layer recomputes as before.
        """
        for path, framework in _SOURCES:
            self._round_trip(CostModelConfig(path, framework=framework), os.path.basename(path))

    def test_every_parser_reads_its_config_back(self):
        """
        Feature: exec_of and apply_exec, for the parsers the yamls do not reach.
        Description: The same round trip on a TOML config of a DeepSeek-shaped
            model, and on each submodule of a MindSpeed vision-language model.
        Expectation: No field changes but the recompute's form, and every
            layer recomputes as before.
        """
        with tempfile.TemporaryDirectory() as folder:
            self._round_trip(_toml(folder), "toml")
        for name, submodule in _mindspeed().mm_ccfgs.items():
            self._round_trip(submodule, f"mindspeed {name}")

    def test_a_partial_spec_changes_what_it_states(self):
        """
        Feature: apply_exec.
        Description: State only a TP degree on the DeepSeek MindFormers
            config, which runs sequence parallelism and slices its
            recompute input.
        Expectation: TP changes, and so do the fields derived from it; the
            data-parallel degree does not.
        """
        ccfg = CostModelConfig(os.path.join(_HERE, "deepseek.yaml"))
        dp = ccfg.d
        apply_exec(ccfg, ExecSpec(tp=2))
        got = (ccfg.t, ccfg.sp, ccfg.shard_recompute_input, ccfg.d)
        self.assertEqual(got, (2, 2, 2, dp), f"t, sp, shard_recompute_input, d={got}")

    def test_a_stated_run_fact_wins_over_the_family(self):
        """
        Feature: apply_exec and the family's defaults.
        Description: The Hyper Qwen config states no byte width and no
            activation sharding.  State two-byte gradients that accumulate
            without pipelining, and whole activations.
        Expectation: Before, the family's 4-byte gradients and Qwen's
            sharded output layer; after, what the spec states.
        """
        ccfg = CostModelConfig(os.path.join(_YAMLS, "hyper_qwen3_72b.yaml"), framework="hyper_v2")
        self.assertEqual((ccfg.bytes_grad, ccfg.shard_output_activ), (4, ccfg.t))
        apply_exec(ccfg, ExecSpec(grad_bytes=2, grad_accumulation=True, shard_activations=False))
        self.assertEqual((ccfg.bytes_grad, ccfg.shard_output_activ), (2, 1))
        self.assertEqual(exec_of(ccfg).grad_bytes, 2)

    def test_device_memory_from_a_string(self):
        """
        Feature: apply_exec.
        Description: An ExecSpec states the device's memory as YAML does.
        Expectation: The config holds it as the cost model's memory size.
        """
        ccfg = CostModelConfig(os.path.join(_HERE, "deepseek.yaml"))
        apply_exec(ccfg, ExecSpec(device_memory="32GB"))
        got = (ccfg.device_capacity.size, str(ccfg.device_capacity.unit))
        self.assertEqual(got, (32.0, "GB"), f"device_capacity={got}")

    def test_strategy_exec_states_what_set_strategy_reads(self):
        """
        Feature: strategy_exec.
        Description: A keyword strategy with integer degrees, a non-integer
            micro-batch size, an optimizer sharding, an offset and a
            recompute list.
        Expectation: Integers are stated and the rest left, sequence
            parallelism is on, and the global batch follows the batching the
            strategy leaves.
        """
        ccfg = CostModelConfig(os.path.join(_HERE, "deepseek.yaml"))
        spec = strategy_exec(ccfg, {"dp": 4, "mp": 2, "mbs": "2", "mb": 8, "op": 1,
                                    "offset": [0, 0], "full_rec": True})
        want = ExecSpec(dp=4, tp=2, micro_batch_num=8, optimizer_shard=1, optimizer_parallel=False,
                        sequence_parallel=True, global_batch_size=ccfg.b * 4 * 8, offset=[0, 0],
                        full_recompute=True)
        self.assertEqual(spec, want, f"strategy_exec gave {spec}")


def _unit(pp: int = 2) -> CostModelConfig:
    """A small dense Hyper model of 8 layers at *pp* stages, fully recomputed."""
    return CostModelConfig({
        "model": {"name": "unit", "config_overrides": {
            "hidden_size": 1024, "num_hidden_layers": 8, "num_attention_heads": 8,
            "num_key_value_heads": 8, "intermediate_size": 2816, "vocab_size": 32000,
            "max_position_embeddings": 2048,
        }},
        "training": {"global_batch_size": 8, "micro_batch_size": 1},
        "accelerator": {"tp_size": 2, "pp_size": pp},
        "fsdp_config": {"dp_shard_size": 2},
        "activation_checkpoint": {"mode": "full"},
        "dataset": {"data_transform": {"max_seq_len": 2048}},
        "context": {"max_device_memory": "64GB", "device_num": 4 * pp},
    }, framework="hyper_v2")


def _layers(ccfg: Any) -> list:
    """Each layer's recompute type, in model order."""
    stages = ccfg.generate_partitions_vpp()
    return [stages[stage][chunk][lay].name[:3] for stage, chunk, lay in get_model_order(ccfg, stages)]


class TestRecomputeRanges(unittest.TestCase):
    """A config's recompute as ranges over its layers, S2's shape."""

    def test_every_config_reads_back_its_recompute_as_ranges(self):
        """
        Feature: recompute_of.
        Description: A model recomputed in full, then only the first layer of
            each of its two stages, as a search's per-stage count states it.
        Expectation: One full range over every layer, then one range per
            stage, at the first layer of each.
        """
        ccfg = _unit()
        layers = int(ccfg.n_lay + ccfg.n_mtp)
        self.assertEqual(exec_of(ccfg).recompute, (RecomputeRange(first=0, count=layers, option="full"),))
        apply_exec(ccfg, ExecSpec(full_recompute=[1, 1]))
        self.assertEqual(exec_of(ccfg).recompute, (
            RecomputeRange(first=0, count=1, option="full"), RecomputeRange(first=4, count=1, option="full"),
        ))

    def test_stated_ranges_give_each_layer_its_option(self):
        """
        Feature: apply_exec and the partition generator.
        Description: State two layers recomputed in full from layer 1, and
            selective recompute of ffAct from layer 5 on; then a per-stage
            form.
        Expectation: Each layer in model order takes its range's option, the
            selective switches are the stated ones, and the per-stage form
            replaces the ranges.
        """
        ccfg = _unit()
        apply_exec(ccfg, ExecSpec(recompute=(
            RecomputeRange(first=1, count=2, option="full"),
            RecomputeRange(first=5, option="selective", ops={"ffAct": "recompute"}),
        )))
        self.assertEqual(_layers(ccfg), ["NOT", "FUL", "FUL", "NOT", "NOT", "SEL", "SEL", "SEL"])
        self.assertEqual(sorted(op for op, keep in vars(ccfg.rec_op).items() if not keep), ["ffAct"])
        apply_exec(ccfg, ExecSpec(full_recompute=True))
        self.assertIsNone(ccfg.recompute_ranges)
        self.assertEqual(_layers(ccfg), ["FUL"] * 8)

    def test_one_range_states_one_mode_for_every_layer(self):
        """
        Feature: one mode for every layer, as HyperParallel's trainer runs.
        Description: One selective range from the first layer on, with the
            rule's switches.
        Expectation: Every layer is selective, with HyperParallel's own
            selective switches.
        """
        ccfg = _unit()
        apply_exec(ccfg, ExecSpec(recompute=(RecomputeRange(option="selective"),)))
        self.assertEqual(_layers(ccfg), ["SEL"] * 8)
        self.assertEqual(vars(ccfg.rec_op), HYPER_SELECTIVE_REC_OP)

    def test_what_a_config_cannot_price_is_refused(self):
        """
        Feature: derive.
        Description: Two selective settings in one config, and a range past
            the model's last layer.
        Expectation: ValueError for each: a config prices one selective
            setting, and every range covers layers the model has.
        """
        ccfg = _unit()
        with self.assertRaises(ValueError):
            apply_exec(ccfg, ExecSpec(recompute=(
                RecomputeRange(first=0, count=4, option="selective", ops={"ffAct": "recompute"}),
                RecomputeRange(first=4, option="selective", ops={"normOp": "recompute"}),
            )))
        with self.assertRaises(ValueError):
            apply_exec(_unit(), ExecSpec(recompute=(RecomputeRange(first=6, count=4, option="full"),)))


class TestStrategyGuard(unittest.TestCase):
    """An armed config takes a degree only through apply_exec."""

    def setUp(self) -> None:
        """The DeepSeek MindFormers config, armed as a search arms it."""
        self.ccfg = CostModelConfig(os.path.join(_HERE, "deepseek.yaml"))
        arm_strategy_guard(self.ccfg)

    def test_a_direct_degree_write_is_refused(self):
        """
        Feature: the strategy guard.
        Description: Write each guarded field of an armed config directly.
        Expectation: AttributeError for each, and the field keeps its value.
        """
        for name in sorted(STRATEGY_GUARDED):
            before = getattr(self.ccfg, name)
            with self.assertRaises(AttributeError, msg=f"{name} was written"):
                setattr(self.ccfg, name, 3)
            self.assertEqual(getattr(self.ccfg, name), before, f"{name} changed")

    def test_the_sanctioned_writers_still_write(self):
        """
        Feature: the strategy guard.
        Description: Change an armed config through set_strategy, apply_exec
            and a layer kind's degrees, and write a field that is not a
            degree, such as the recompute switches a pricer sets.
        Expectation: Every write lands.
        """
        self.ccfg.set_strategy(mp=2)
        apply_exec(self.ccfg, ExecSpec(dp=8))
        apply_layer_strategy(self.ccfg, {"ep": 1})
        self.ccfg.rec_op = Config({"gather": 0})
        got = (self.ccfg.t, self.ccfg.d, self.ccfg.ep, self.ccfg.rec_op.gather)
        self.assertEqual(got, (2, 8, 1, 0), f"t, d, ep, rec_op.gather={got}")

    def test_a_copy_starts_unarmed(self):
        """
        Feature: the strategy guard.
        Description: Copy an armed config, shallow and deep, as an estimator
            does before it prices a layer.
        Expectation: The copies take a direct write; the original still refuses one.
        """
        for copier in (copy.copy, copy.deepcopy):
            clone = copier(self.ccfg)
            clone.t = 8
            self.assertEqual(clone.t, 8, f"{copier.__name__}: t={clone.t}")
        with self.assertRaises(AttributeError):
            self.ccfg.t = 8

    def test_a_hook_is_guarded_for_its_length(self):
        """
        Feature: the strategy guard.
        Description: Run a hook that writes a degree through an evaluator's
            set_ccfg, on an unarmed config, then guard an armed one for a block.
        Expectation: The hook is refused; afterwards the unarmed config takes
            a direct write again, and the armed one still refuses one.
        """
        unarmed = CostModelConfig(os.path.join(_HERE, "deepseek.yaml"))
        evaluator = EvaluatorV2(None, ccfg=unarmed)
        with self.assertRaises(AttributeError):
            evaluator.set_ccfg(lambda cfg: setattr(cfg, "t", 1))
        unarmed.t = 1
        self.assertEqual(unarmed.t, 1, f"t={unarmed.t}")
        with strategy_guarded(self.ccfg):
            pass
        with self.assertRaises(AttributeError):
            self.ccfg.t = 1


if __name__ == "__main__":
    unittest.main()
