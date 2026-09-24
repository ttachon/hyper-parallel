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

from hyper_parallel.auto_parallel._exec_spec import ExecSpec, ExecSpecError


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
                     {"capacity_factor": 0}):
            with self.assertRaises(ExecSpecError, msg=f"{data} was accepted"):
                ExecSpec.from_dict(data)


if __name__ == "__main__":
    unittest.main()
