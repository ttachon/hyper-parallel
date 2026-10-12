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
"""Unit tests for the offload test runner of the Qwen3.5-MoE demo.

The runner works from a cluster's control node, so it is loaded here from its
path, as the sweep's own tests load the sweep.

How to run this:
    pytest tests/ut/examples/training_demo/test_offload_tests.py
"""
import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from typing import List, Tuple

_RUNNER = (Path(__file__).resolve().parents[4]
           / "examples" / "training_demo" / "offload_tests.py")


def _load_runner():
    """Load the runner by path, registered under its file name."""
    spec = importlib.util.spec_from_file_location("offload_tests", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Runner = _load_runner()
_VERSIONS = "2.7.1+cpu 2.7.1.post4.dev20260402"
# What the kit's exec prints: one node at a time under a header (-S), or every node at once, each line tagged.
_ONE_AT_A_TIME = f"===== node0 (192.168.0.143) =====\n{_VERSIONS}\n===== node1 (192.168.0.245) =====\n{_VERSIONS}\n"
_TOGETHER = f"\x1b[34m[node0]\x1b[0m {_VERSIONS}\n\x1b[32m[node1]\x1b[0m {_VERSIONS}\n"


class _Kit:
    """Stands in for the kit: answers every command with one exit code and one output, and keeps the commands."""

    def __init__(self, code: int, text: str) -> None:
        """The exit code and output every command gets."""
        self.code, self.text, self.commands = code, text, []

    def run(self, *words: str) -> Tuple[int, str]:
        """Record a kit command, and answer it."""
        self.commands.append(list(words))
        return self.code, self.text


class TestEnvironment(unittest.TestCase):
    """The guard launches on nodes that answer one version, whichever way the kit lays the answers out."""

    def _agree(self, code: int, text: str) -> Tuple[bool, List[List[str]]]:
        """What the guard decides for one kit answer, and the commands it sent."""
        kit = _Kit(code, text)
        with contextlib.redirect_stdout(io.StringIO()):
            agreed = Runner.environment(kit)
        return agreed, kit.commands

    def test_one_answer_passes_in_either_layout(self):
        """The same versions from every node pass, under headers or behind the coloured tags."""
        for text in (_ONE_AT_A_TIME, _TOGETHER):
            with self.subTest(text=text):
                agreed, commands = self._agree(0, text)
                self.assertTrue(agreed)
                self.assertEqual(commands, [["exec", "-S", Runner.IMPORTS]])

    def test_a_node_that_differs_or_a_failed_exec_is_refused(self):
        """A node with another answer, or a kit that fails, stops the launch."""
        drifted = _ONE_AT_A_TIME.replace(f"=====\n{_VERSIONS}\n", "=====\nImportError: cannot import name\n", 1)
        for code, text in ((0, drifted), (1, _ONE_AT_A_TIME), (0, "")):
            with self.subTest(code=code, text=text):
                self.assertFalse(self._agree(code, text)[0])

    def test_node_answers_drop_the_kits_headers_tags_and_colours(self):
        """Only what the nodes printed is left."""
        self.assertEqual(Runner.node_answers(_ONE_AT_A_TIME), {_VERSIONS})
        self.assertEqual(Runner.node_answers(_TOGETHER), {_VERSIONS})
        self.assertEqual(Runner.node_answers("[node2] " + _VERSIONS + "\n[node3] other\n"), {_VERSIONS, "other"})


if __name__ == "__main__":
    unittest.main()
