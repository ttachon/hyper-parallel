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
"""Tests for the execution IR's schema.

How to run this:
    pytest tests/ut/auto_parallel/test_exec_spec.py -v
"""
import unittest

from hyper_parallel.auto_parallel._exec_spec import ExecSpec, ExecSpecError, RecomputeRange


class TestExecSpec(unittest.TestCase):
    """An ExecSpec states what it states, and refuses what cannot be right."""

    def test_absent_is_not_zero(self):
        """
        Feature: ExecSpec.
        Description: A spec that states nothing, and its serialised form.
        Expectation: Every field is None, and the mapping is empty.
        """
        spec = ExecSpec.from_dict({})
        self.assertEqual(spec, ExecSpec(), f"spec={spec}")
        self.assertEqual(spec.to_dict(), {}, f"to_dict={spec.to_dict()}")

    def test_from_dict_coerces_the_yaml_types(self):
        """
        Feature: ExecSpec.from_dict.
        Description: A mapping as YAML or a producer might write it.
        Expectation: Counts become integers, the capacity factor a float and
            the device memory a string; the spec round-trips through to_dict.
        """
        spec = ExecSpec.from_dict({
            "dp": "4", "tp": 2.0, "sequence_parallel": True, "capacity_factor": "1.5",
            "selective_rule": "mindformers", "device_memory": "64GB", "offset": [1, -1],
            "full_recompute": [2, 1],
        })
        got = (spec.dp, spec.tp, spec.capacity_factor, spec.device_memory)
        self.assertEqual(got, (4, 2, 1.5, "64GB"), f"dp, tp, capacity_factor, device_memory={got}")
        self.assertEqual(ExecSpec.from_dict(spec.to_dict()), spec, f"round trip of {spec.to_dict()}")

    def test_unknown_key_is_refused(self):
        """
        Feature: ExecSpec.from_dict.
        Description: A key the schema does not know.
        Expectation: ExecSpecError naming it, rather than a silent drop.
        """
        with self.assertRaisesRegex(ExecSpecError, "tensor_parallel"):
            ExecSpec.from_dict({"dp": 4, "tensor_parallel": 2})

    def test_values_that_cannot_be_right_are_refused(self):
        """
        Feature: ExecSpec.validate.
        Description: A zero degree, a negative size, a fractional count, a
            flag that is not a boolean and an unknown selective rule.
        Expectation: ExecSpecError for each.
        """
        for data in ({"dp": 0}, {"micro_batch_size": -1}, {"grad_bytes": -1}, {"tp": 2.5},
                     {"sequence_parallel": "yes"}, {"selective_rule": "megatron"},
                     {"capacity_factor": 0}, {"dropout_bytes": -1}, {"shard_activations": 1},
                     {"grad_accumulation": "no"}, {"grad_shard_as_params": "yes"}):
            with self.assertRaises(ExecSpecError, msg=f"{data} was accepted"):
                ExecSpec.from_dict(data)


class TestRecomputeRanges(unittest.TestCase):
    """How each layer recomputes, as ranges over the layers in model order."""

    def test_ranges_round_trip_through_yaml(self):
        """
        Feature: ExecSpec.recompute.
        Description: Three dense layers recomputed in full, then selective
            recompute of two ops to the last layer.
        Expectation: The ranges read from a mapping, and serialise back to it.
        """
        data = {"recompute": [
            {"first": 0, "count": 3, "option": "full"},
            {"first": 3, "option": "selective", "ops": {"ffAct": "recompute", "normOp": "recompute"}},
        ]}
        spec = ExecSpec.from_dict(data)
        self.assertEqual(spec.recompute, (
            RecomputeRange(first=0, count=3, option="full"),
            RecomputeRange(first=3, option="selective", ops={"ffAct": "recompute", "normOp": "recompute"}),
        ))
        self.assertEqual(spec.to_dict(), data)
        switches = spec.recompute[1].switches()
        self.assertEqual(sorted(op for op, keep in switches.items() if not keep), ["ffAct", "normOp"])

    def test_one_range_is_the_whole_model(self):
        """
        Feature: RecomputeRange.
        Description: A range that states only its option.
        Expectation: It starts at the first layer and runs to the last, and
            takes the rule's switches.
        """
        whole = RecomputeRange(option="full")
        self.assertEqual((whole.first, whole.count, whole.switches()), (0, None, None))

    def test_ranges_that_cannot_be_right_are_refused(self):
        """
        Feature: ExecSpec.validate.
        Description: Overlapping ranges, an open range before another, an
            unknown option, ops on a full range, an unknown op, an unknown
            state, a negative first layer, an empty range and an unknown key.
        Expectation: ExecSpecError for each.
        """
        cases = [
            [{"first": 0, "count": 3, "option": "full"}, {"first": 2, "option": "selective"}],
            [{"first": 0, "option": "full"}, {"first": 5, "option": "selective"}],
            [{"option": "offload"}],
            [{"option": "full", "ops": {"ffAct": "recompute"}}],
            [{"option": "selective", "ops": {"matmul": "recompute"}}],
            [{"option": "selective", "ops": {"ffAct": "offload"}}],
            [{"first": -1, "option": "full"}],
            [{"first": 0, "count": 0, "option": "full"}],
            [{"first": 0, "layers": 3, "option": "full"}],
        ]
        for ranges in cases:
            with self.assertRaises(ExecSpecError, msg=f"{ranges} was accepted"):
                ExecSpec.from_dict({"recompute": ranges})


if __name__ == "__main__":
    unittest.main()
