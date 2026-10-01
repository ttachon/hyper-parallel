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
"""Tests for the op records (shared decision S1): their file, their expressions and their checks.

How to run this:
    pytest tests/ut/auto_parallel/test_op_records.py -v
"""
import os
import tempfile
import unittest

import yaml

from hyper_parallel.auto_parallel._op_records import OPS, PARTS, SWITCHES, Expression, load_op_records

_OPS = list(OPS)


def _records(data):
    """The records a file holding *data* states."""
    folder = tempfile.mkdtemp()
    path = os.path.join(folder, "op_records.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return load_op_records(path)


def _minimal(**slots):
    """A records file stating every op and the *slots* given."""
    return {"ops": {op: {} for op in _OPS}, "slots": slots}


class TestTheRecords(unittest.TestCase):
    """The op records the memory model prices a layer with."""

    def test_every_op_has_a_record(self):
        """
        Feature: load_op_records.
        Description: The records the package ships.
        Expectation: Every op of a layer has one; each switch drops
            what the op it names keeps, and gather is communication; every
            slot counts toward a part of the layer.
        """
        records = load_op_records()
        self.assertEqual(sorted(records.ops), sorted(_OPS))
        self.assertEqual(sorted(op.switch for op in records.ops.values() if op.switch), sorted(SWITCHES))
        self.assertEqual([name for name, op in records.ops.items() if op.comm], ["gather"])
        self.assertTrue(all(slot.part in PARTS for slot in records.slots.values()))
        self.assertEqual(sorted(records.keeps("dropout")), ["proj", "score"])

    def test_a_record_is_checked(self):
        """
        Feature: load_op_records' checks.
        Description: Files missing an op, naming an unknown switch or key,
            and slots whose expressions read an attribute, call anything but
            max and min, hold a string or read no total.
        Expectation: Each is refused, naming what is wrong.
        """
        missing = {"ops": {op: {} for op in _OPS[1:]}, "slots": {}}
        cases = [
            (missing, "missing"),
            ({**_minimal(), "ops": {**{op: {} for op in _OPS}, "attMM": {"switch": "attMM"}}}, "switch"),
            ({**_minimal(), "extra": 1}, "unknown keys"),
            (_minimal(qkv={"part": "attention", "formula": "total * ccfg.h", "ops": {"attMM": "h"}}), "Attribute"),
            (_minimal(qkv={"part": "attention", "formula": "total * open(h)", "ops": {"attMM": "h"}}), "max or min"),
            (_minimal(qkv={"part": "attention", "formula": "total", "ops": {"attMM": "'h'"}}), "not a number"),
            (_minimal(qkv={"part": "attention", "formula": "h", "ops": {"attMM": "h"}}), "total"),
            (_minimal(qkv={"part": "brain", "formula": "total", "ops": {"attMM": "h"}}), "part"),
        ]
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                _records(data)

    def test_a_slot_reads_only_its_branch(self):
        """
        Feature: OpRecords.evaluate.
        Description: A slot whose op keeps one field's elements or another's,
            as a flag says, evaluated where the other field does not exist;
            then with the op dropped, and priced alone.
        Expectation: The branch the flag takes is the only one read; max and
            min are calls, not fields; a dropped op keeps nothing, and one
            priced alone keeps its elements whatever the setting.
        """
        records = _records(_minimal(qkv={
            "part": "attention", "formula": "total * max(t, 1) / min(t, 2)",
            "ops": {"attMM": "h if flag else missing", "attBMM": "3"}}))
        fields_ = {"h": 10, "flag": 1, "t": 4}
        value = fields_.__getitem__
        self.assertEqual(records.evaluate("qkv", value, lambda op: 1), (10 + 3) * 4 / 2)
        self.assertEqual(records.evaluate("qkv", value, lambda op: int(op != "attBMM")), 10 * 4 / 2)
        self.assertEqual(records.evaluate("qkv", value, lambda op: 0, only="attBMM"), 3 * 4 / 2)

    def test_an_expression_keeps_python_arithmetic(self):
        """
        Feature: Expression.
        Description: An expression with a conditional, a remainder and an
            or, evaluated on names.
        Expectation: Python's own value, to the bit.
        """
        expression = Expression.parse("(0.5 * h + 0.5 * w if n % 2 == 0 else 1 / 3 * h + 2 / 3 * w) * (b * n)"
                                      " + (d or h / a)", "test")
        names = {"h": 4096, "w": 14336, "n": 3, "b": 2, "d": 0, "a": 28}
        self.assertEqual(expression(dict(names)), (1 / 3 * 4096 + 2 / 3 * 14336) * (2 * 3) + (0 or 4096 / 28))
        self.assertEqual(expression.names, frozenset(names))


if __name__ == "__main__":
    unittest.main()
