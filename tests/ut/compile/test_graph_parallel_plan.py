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
"""Unit tests for ``hyper_parallel.compile.graph_parallel_plan``.

Covers:

1. ``GraphParallelPlan.fsdp_mark`` / ``fsdp_mark_pattern`` register entries
   and support chaining.
2. ``is_marked_for_fsdp`` exact-match and ``fnmatch`` pattern semantics.
3. ``create_all_fsdp_plan`` marks everything via the ``*`` pattern.
4. ``create_plan_from_yaml`` happy path + validation.
"""

import os
import tempfile
import textwrap
import unittest

import yaml

from hyper_parallel.compile.graph_parallel_plan import (
    GraphParallelPlan,
    create_all_fsdp_plan,
    create_plan_from_yaml,
)


class TestGraphParallelPlanRegistration(unittest.TestCase):
    """``fsdp_mark`` / ``fsdp_mark_pattern`` register entries; chainable."""

    def test_fsdp_mark_registers_exact_match(self):
        """Test ``fsdp_mark`` adds an exact-match entry."""
        plan = GraphParallelPlan()
        result = plan.fsdp_mark("tok_embeddings")
        self.assertIs(result, plan, "fsdp_mark should return self for chaining")
        self.assertIn("tok_embeddings", plan.fsdp_modules)

    def test_fsdp_mark_pattern_registers_pattern(self):
        """Test ``fsdp_mark_pattern`` adds a wildcard-pattern entry."""
        plan = GraphParallelPlan()
        result = plan.fsdp_mark_pattern("layers.*")
        self.assertIs(result, plan)
        self.assertIn("layers.*", plan.fsdp_patterns)

    def test_chaining(self):
        """Test multiple builder calls chain on one plan."""
        plan = (
            GraphParallelPlan()
            .fsdp_mark("tok_embeddings")
            .fsdp_mark("head")
            .fsdp_mark_pattern("layers.*")
        )
        self.assertEqual(len(plan.fsdp_modules), 2)
        self.assertEqual(len(plan.fsdp_patterns), 1)


class TestGraphParallelPlanMatching(unittest.TestCase):
    """``is_marked_for_fsdp`` match semantics."""

    def setUp(self) -> None:
        """Build a plan with one exact-match and one pattern entry."""
        self.plan = (
            GraphParallelPlan()
            .fsdp_mark("tok_embeddings")
            .fsdp_mark_pattern("layers.*")
        )

    def test_exact_match(self):
        """Test exact-match FQN is recognized."""
        self.assertTrue(self.plan.is_marked_for_fsdp("tok_embeddings"))

    def test_pattern_match(self):
        """Test wildcard pattern matches via fnmatch."""
        self.assertTrue(self.plan.is_marked_for_fsdp("layers.0"))
        self.assertTrue(self.plan.is_marked_for_fsdp("layers.42.attention"))

    def test_no_match(self):
        """Test unrelated FQN returns False."""
        self.assertFalse(self.plan.is_marked_for_fsdp("embed"))
        self.assertFalse(self.plan.is_marked_for_fsdp("layers"))  # pattern is layers.*

    def test_nested_fqn_needs_ancestor_walk(self):
        """Exact marks are not prefix matches; the ancestor walk lives in FSDPPass."""
        plan = GraphParallelPlan().fsdp_mark("layers.0")
        self.assertFalse(plan.is_marked_for_fsdp("layers.0.attention.weight"))


class TestCreateAllFsdpPlan(unittest.TestCase):
    """``create_all_fsdp_plan`` marks everything via ``*``."""

    def test_marks_all_modules(self):
        """Test the all-fsdp plan matches any FQN via ``*`` pattern."""
        plan = create_all_fsdp_plan()
        self.assertTrue(plan.is_marked_for_fsdp("anything"))
        self.assertTrue(plan.is_marked_for_fsdp("layers.0.attention.weight"))
        self.assertIn("*", plan.fsdp_patterns)
        self.assertEqual(len(plan.fsdp_modules), 0)


class TestCreatePlanFromYaml(unittest.TestCase):
    """YAML loading happy path + validation rules."""

    def test_loads_modules_and_patterns(self):
        """Test a YAML with both modules and patterns loads both."""
        yaml_text = textwrap.dedent("""
        fsdp:
          enabled: true
          modules:
            - name: tok_embeddings
            - name: head
          patterns:
            - pattern: "layers.*"
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            plan = create_plan_from_yaml(config_path=path)
        finally:
            os.unlink(path)

        self.assertTrue(plan.is_marked_for_fsdp("tok_embeddings"))
        self.assertTrue(plan.is_marked_for_fsdp("head"))
        self.assertTrue(plan.is_marked_for_fsdp("layers.0"))

    def test_enabled_false_skips_loading(self):
        """Test ``enabled: false`` skips loading modules/patterns."""
        yaml_text = textwrap.dedent("""
        fsdp:
          enabled: false
          modules:
            - name: tok_embeddings
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            plan = create_plan_from_yaml(config_path=path)
        finally:
            os.unlink(path)

        self.assertFalse(plan.is_marked_for_fsdp("tok_embeddings"))

    def test_implicit_enable_when_modules_present(self):
        """Test ``enabled`` defaults True when modules are listed."""
        yaml_text = textwrap.dedent("""
        fsdp:
          modules:
            - name: tok_embeddings
        """)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
            path = f.name
        try:
            plan = create_plan_from_yaml(config_path=path)
        finally:
            os.unlink(path)

        self.assertTrue(plan.is_marked_for_fsdp("tok_embeddings"))

    def test_requires_path_or_model_name(self):
        """Test ValueError when neither config_path nor model_name is given."""
        with self.assertRaises(ValueError):
            create_plan_from_yaml()

    def test_missing_file_raises(self):
        """Test FileNotFoundError for a non-existent path."""
        with self.assertRaises(FileNotFoundError):
            create_plan_from_yaml(config_path="/no/such/file.yaml")

    def test_empty_yaml_raises(self):
        """Test an empty YAML file raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                create_plan_from_yaml(config_path=path)
        finally:
            os.unlink(path)

    def test_non_mapping_yaml_raises(self):
        """Test a YAML list (non-mapping) raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(["not", "a", "mapping"], f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                create_plan_from_yaml(config_path=path)
        finally:
            os.unlink(path)

    def test_invalid_model_name_rejected(self):
        """Test ``model_name`` with path separators is rejected."""
        for bad in ("../etc", "foo/bar", "foo\\bar"):
            with self.assertRaises(ValueError):
                create_plan_from_yaml(model_name=bad)


if __name__ == "__main__":
    unittest.main()
