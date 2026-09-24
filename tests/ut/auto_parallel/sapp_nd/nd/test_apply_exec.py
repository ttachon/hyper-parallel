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
import os
import unittest
from typing import Any, Dict

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel._exec_spec import ExecSpec
from hyper_parallel.auto_parallel.sapp_nd.nd.common.apply_exec import (
    apply_exec,
    apply_layer_strategy,
    exec_of,
    strategy_exec,
)
from hyper_parallel.auto_parallel.sapp_nd.nd.common.config import Config
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


def _state(ccfg: Any) -> Dict[str, Any]:
    """The config's fields, with the ones held in objects read as values."""
    state = {}
    for key, value in vars(ccfg).items():
        if isinstance(value, Config):
            value = vars(value)
        elif isinstance(value, Memory):
            value = (value.size, value.unit)
        state[key] = value
    return state


class TestApplyExec(unittest.TestCase):
    """apply_exec writes what an ExecSpec states, and derives the rest."""

    def test_its_own_spec_leaves_a_config_as_it_is(self):
        """
        Feature: exec_of and apply_exec.
        Description: Read back the ExecSpec of a parsed MindFormers config and
            of two Hyper configs, and apply it to the config it came from.
        Expectation: No field changes.
        """
        for path, framework in _SOURCES:
            ccfg = CostModelConfig(path, framework=framework)
            before = _state(ccfg)
            apply_exec(ccfg, exec_of(ccfg))
            after = _state(ccfg)
            changed = sorted(key for key, value in before.items() if value != after.get(key))
            self.assertEqual(changed, [], f"{os.path.basename(path)}: fields changed {changed}")

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
