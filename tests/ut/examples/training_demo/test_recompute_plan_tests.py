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
"""Unit tests for the per-layer recompute test runner of the Qwen3.5-MoE demo.

The runner works from a cluster's control node, so it is loaded here from its
path, as the sweep's own tests load the sweep.

How to run this:
    pytest tests/ut/examples/training_demo/test_recompute_plan_tests.py
"""
import contextlib
import importlib.util
import io
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

_RUNNER = (Path(__file__).resolve().parents[4]
           / "examples" / "training_demo" / "recompute_plan_tests.py")


def _load_runner():
    """Load the runner by path, registered under its file name."""
    spec = importlib.util.spec_from_file_location("recompute_plan_tests", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Runner = _load_runner()

_ENV = (
    'NODES=(\n  192.168.0.143\n  192.168.0.169\n)\nSSH_USER="root"\n'
    'REMOTE_ENV_SETUP="source /home/tt/init_env.sh; export HCCL_EXEC_TIMEOUT=900"\n'
    'LOG_DIR="/home/tt/cluster_logs"\n'
)


class TestAllocatorSetting(unittest.TestCase):
    """The allocator setting goes into REMOTE_ENV_SETUP once, and comes out again."""

    def test_setting_is_added_once_and_taken_out(self):
        """Adding twice gives one setting; taking it out gives the config back unchanged."""
        enabled = Runner.with_allocator(Runner.with_allocator(_ENV, True), True)

        self.assertEqual(enabled.count(Runner.ALLOCATOR), 1, f"enabled={enabled!r}")
        self.assertIn(f'HCCL_EXEC_TIMEOUT=900; {Runner.ALLOCATOR}"\n', enabled)
        self.assertEqual(Runner.with_allocator(enabled, False), _ENV)

    def test_line_endings_survive(self):
        """A config written with CRLF keeps CRLF on the line the setting goes into."""
        crlf = _ENV.replace("\n", "\r\n")

        enabled = Runner.with_allocator(crlf, True)

        self.assertIn(f'{Runner.ALLOCATOR}"\r\n', enabled)
        self.assertEqual(Runner.with_allocator(enabled, False), crlf)

    def test_config_it_cannot_edit_is_refused(self):
        """A config that sets the allocator its own way, or has no setup line, is left alone."""
        cases = (
            (_ENV.replace("900", "900; export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512"), "its own way"),
            (_ENV.replace("REMOTE_ENV_SETUP", "# REMOTE_ENV_SETUP"), "no one-line REMOTE_ENV_SETUP"),
        )
        for text, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(SystemExit, message):
                    Runner.with_allocator(text, True)


class TestPlans(unittest.TestCase):
    """Each plan becomes a train yaml and one sweep invocation."""

    def test_plan_over_mode_off_tells_the_sweep_its_mode(self):
        """The sweep's command line states the mode, so a plan over off must say off there too."""
        over_off = Runner.sweep_command("A", "base_off_0-2")
        over_full = Runner.sweep_command("B", "off2-7")

        self.assertEqual(over_off[-2:], ["--activation-checkpoint", "off"], f"command={over_off}")
        self.assertNotIn("--activation-checkpoint", over_full)
        self.assertEqual(over_full[over_full.index("--out") + 1],
                         str(Runner.REPO_ROOT / "output" / "recompute_plan_tests" / "B_off2-7"))
        config = Path(over_full[over_full.index("--config") + 1])
        self.assertTrue(config.is_absolute() and Runner.REPO_ROOT in config.parents, f"config={config}")

    def test_plan_yamls_carry_each_plan_on_the_demo_yaml(self):
        """Every yaml is the demo's train yaml with the plan as its activation_checkpoint section."""
        base = yaml.safe_load(Runner.TRAIN_YAML.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Runner, "plan_yaml", lambda plan: Path(folder) / f"{plan}.yaml"):
                paths = Runner.write_plan_yamls()

            self.assertEqual(len(paths), len(Runner.PLANS))
            for plan, section in Runner.PLANS.items():
                written = yaml.safe_load((Path(folder) / f"{plan}.yaml").read_text(encoding="utf-8"))
                self.assertEqual(written["activation_checkpoint"], section, f"plan={plan}")
                self.assertEqual(dict(written, activation_checkpoint=None),
                                 dict(base, activation_checkpoint=None), f"plan={plan}")

    def test_blocks_name_known_plans(self):
        """Every block and comparison names plans the runner defines."""
        for block, plans in Runner.BLOCKS.items():
            self.assertTrue(set(plans) <= set(Runner.PLANS), f"block={block}")
        for _, first, second in Runner.COMPARISONS:
            for block, plan in (first, second):
                self.assertIn(plan, Runner.BLOCKS[block])


class TestRunBlock(unittest.TestCase):
    """Block B runs with the allocator setting and leaves the config as it found it."""

    def test_block_b_restores_the_config_even_when_stopped(self):
        """The setting is on for each run of block B, and off once the block stops."""
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / "cluster.env"
            env.write_text(_ENV, encoding="utf-8")
            (Path(folder) / "plan.yaml").write_text("{}", encoding="utf-8")
            seen = []

            def fake_stream(command: list, log: Path) -> int:
                """Record each run's output folder and whether the setting was on; stop at the second."""
                del log
                seen.append((command[command.index("--out") + 1], Runner.ALLOCATOR in env.read_text()))
                if len(seen) == 2:
                    raise KeyboardInterrupt
                return 0

            with (
                patch.object(Runner, "CLUSTER_ENV", env),
                patch.object(Runner, "OUT", Path(folder) / "out"),
                patch.object(Runner, "plan_yaml", lambda plan: Path(folder) / "plan.yaml"),
                patch.object(Runner, "stream", fake_stream),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    Runner.run_block("B")

            self.assertEqual([on for _, on in seen], [True, True], f"seen={seen}")
            self.assertTrue(seen[0][0].endswith("B_full"), f"seen={seen}")
            self.assertEqual(env.read_text(encoding="utf-8"), _ENV)

    def test_a_signal_ends_block_b_through_its_cleanup(self):
        """A hangup or a kill becomes SystemExit, which takes the setting out; the handlers come back."""
        with self.assertRaisesRegex(SystemExit, "stopped by signal 1"):
            Runner._stop(1, None)  # pylint: disable=protected-access
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / "cluster.env"
            env.write_text(_ENV, encoding="utf-8")
            (Path(folder) / "plan.yaml").write_text("{}", encoding="utf-8")
            before = signal.getsignal(signal.SIGTERM)

            def killed(command: list, log: Path) -> int:
                """Stand in for a run that a SIGTERM stops, as the runner's handler does."""
                del command, log
                self.assertIs(signal.getsignal(signal.SIGTERM), Runner._stop)  # pylint: disable=protected-access
                raise SystemExit("stopped by signal 15")

            with (
                patch.object(Runner, "CLUSTER_ENV", env),
                patch.object(Runner, "OUT", Path(folder) / "out"),
                patch.object(Runner, "plan_yaml", lambda plan: Path(folder) / "plan.yaml"),
                patch.object(Runner, "stream", killed),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(SystemExit, "signal 15"):
                    Runner.run_block("B", ["off2-7"])

            self.assertEqual(env.read_text(encoding="utf-8"), _ENV)
            self.assertIs(signal.getsignal(signal.SIGTERM), before)

    def test_reset_takes_a_left_setting_out(self):
        """After a block that was cut off, reset clears the setting the results warn about."""
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / "cluster.env"
            env.write_text(Runner.with_allocator(_ENV, True), encoding="utf-8")
            printed = io.StringIO()
            with (
                patch.object(Runner, "CLUSTER_ENV", env),
                patch.object(Runner, "OUT", Path(folder) / "out"),
                contextlib.redirect_stdout(printed),
            ):
                Runner.results({"nodes": [], "log_dir": "/logs", "ssh_user": "root"})
                Runner.main(["reset"])

            self.assertIn("now: ON, left by a block that did not finish; take it out with: "
                          "python examples/training_demo/recompute_plan_tests.py reset", printed.getvalue())
            self.assertEqual(env.read_text(encoding="utf-8"), _ENV)

    def test_unknown_plan_is_refused_before_anything_runs(self):
        """A misspelt plan stops the block before the config is touched."""
        with self.assertRaisesRegex(SystemExit, "unknown plan"):
            Runner.run_block("B", ["off2-8"])

    def test_stream_echoes_and_appends_to_its_log(self):
        """A command's output reaches the terminal and the log, and its exit code comes back."""
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "logs" / "block.log"
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                first = Runner.stream([sys.executable, "-c", "print('one')"], log)
                second = Runner.stream([sys.executable, "-c", "print('two'); raise SystemExit(3)"], log)

            self.assertEqual((first, second), (0, 3))
            self.assertEqual(printed.getvalue().split(), ["one", "two"])
            self.assertEqual(log.read_text(encoding="utf-8").split(), ["one", "two"])

    def test_setup_writes_the_plans_and_calls_the_sweep_stages(self):
        """Setup writes every plan yaml, clears the allocator setting and runs select, mirror and data."""
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / "cluster.env"
            env.write_text(Runner.with_allocator(_ENV, True), encoding="utf-8")
            commands = []
            with (
                patch.object(Runner, "CLUSTER_ENV", env),
                patch.object(Runner, "OUT", Path(folder) / "out"),
                patch.object(Runner, "plan_yaml", lambda plan: Runner.REPO_ROOT / "output" / f"{plan}.yaml"),
                patch.object(Runner, "write_plan_yamls", lambda: []),
                patch.object(Runner, "stream", lambda command, log: commands.append(command) or 0),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                Runner.setup(Path("/home/tt/cluster_all.env"))

            self.assertEqual(env.read_text(encoding="utf-8"), _ENV)
            self.assertEqual(len(commands), 1, f"commands={commands}")
            command = commands[0]
            self.assertEqual(command[command.index("--pool") + 1], str(Path("/home/tt/cluster_all.env")))
            self.assertEqual(command[command.index("--nodes") + 1], "4")
            self.assertEqual([command[i + 1] for i, word in enumerate(command) if word == "--only"],
                             ["select", "mirror", "data"])


class TestNodeCount(unittest.TestCase):
    """Fewer nodes keep dp_shard 32 and write apart from the demo's 4-node tests."""

    def test_each_node_count_has_its_own_folder(self):
        """The demo's 4 nodes keep the first folder; 2 nodes get a folder of their own."""
        self.assertEqual(Runner.out_dir(4), Runner.OUT)
        self.assertEqual(Runner.out_dir(2), Runner.REPO_ROOT / "output" / "recompute_plan_tests_2nodes")

    def test_a_count_dp_shard_32_cannot_cover_is_refused_before_anything_runs(self):
        """An odd count, or none, stops setup before the yamls, the config or the sweep."""
        for nodes in (0, 1, 3):
            with self.subTest(nodes=nodes):
                with (
                    patch.object(Runner, "write_plan_yamls", side_effect=AssertionError("wrote yamls")),
                    patch.object(Runner, "stream", side_effect=AssertionError("ran the sweep")),
                ):
                    with self.assertRaisesRegex(SystemExit, "an even number of 16-die nodes"):
                        Runner.setup(Path("/home/tt/cluster_all.env"), nodes)

    def test_setup_on_two_nodes_asks_the_sweep_for_two_and_logs_apart(self):
        """The sweep's select picks 2 nodes, and the setup log goes to the 2-node folder."""
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / "cluster.env"
            env.write_text(_ENV, encoding="utf-8")
            calls = []
            with (
                patch.object(Runner, "CLUSTER_ENV", env),
                patch.object(Runner, "OUT", Path(folder) / "recompute_plan_tests"),
                patch.object(Runner, "write_plan_yamls", lambda: []),
                patch.object(Runner, "stream", lambda command, log: calls.append((command, log)) or 0),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                Runner.main(["setup", "--nodes", "2"])

            self.assertEqual(len(calls), 1, f"calls={calls}")
            command, log = calls[0]
            self.assertEqual(command[command.index("--nodes") + 1], "2")
            self.assertEqual(command[command.index("--op") + 1], "32")
            self.assertEqual(log, Path(folder) / "recompute_plan_tests_2nodes" / "setup.log")

    def test_run_and_results_follow_the_nodes_setup_selected(self):
        """With 2 nodes in the kit config, a block runs into the 2-node folder and is read from it."""
        env = {"nodes": ["n0", "n1"], "log_dir": "/logs", "ssh_user": "root"}
        calls = []
        with (
            patch.object(Runner, "read_cluster_env", lambda: env),
            patch.object(Runner, "run_block", lambda *args: calls.append(("run", *args))),
            patch.object(Runner, "results", lambda *args: calls.append(("results", *args))),
        ):
            Runner.main(["run", "A", "off3-7", "full"])
            Runner.main(["results"])

        folder = Runner.out_dir(2)
        self.assertEqual(calls, [("run", "A", ["off3-7", "full"], folder), ("results", env, folder)])

    def test_a_block_writes_each_test_under_the_folder_it_is_given(self):
        """Each test's sweep output and the block log go under the folder passed in."""
        with tempfile.TemporaryDirectory() as folder:
            env = Path(folder) / "cluster.env"
            env.write_text(_ENV, encoding="utf-8")
            (Path(folder) / "plan.yaml").write_text("{}", encoding="utf-8")
            two_nodes = Path(folder) / "recompute_plan_tests_2nodes"
            calls = []
            with (
                patch.object(Runner, "CLUSTER_ENV", env),
                patch.object(Runner, "OUT", Path(folder) / "recompute_plan_tests"),
                patch.object(Runner, "plan_yaml", lambda plan: Path(folder) / "plan.yaml"),
                patch.object(Runner, "stream", lambda command, log: calls.append((command, log)) or 0),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                Runner.run_block("A", ["off3-7", "full"], two_nodes)

            self.assertEqual([command[command.index("--out") + 1] for command, _ in calls],
                             [str(two_nodes / "A_off3-7"), str(two_nodes / "A_full")])
            self.assertEqual({log for _, log in calls}, {two_nodes / "block_A.log"})
            self.assertFalse((Path(folder) / "recompute_plan_tests").exists())


class TestResults(unittest.TestCase):
    """The results name every test, each failure's reason and each comparison."""

    @staticmethod
    def _write(out: Path, test: str, **data):
        """Write the sweep's run_states.json for one test."""
        (out / test).mkdir(parents=True)
        record = {"run_id": f"run_{test}", "status": "", "failed": False, **data}
        (out / test / "run_states.json").write_text(json.dumps({"ep2": record}), encoding="utf-8")

    def test_results_list_runs_failures_and_comparisons(self):
        """A run line, a failure's reason from the node log, and the differences of a pair."""
        env = {"nodes": ["n0", "n1"], "log_dir": "/logs", "ssh_user": "root"}

        def fake_grep(env_: dict, node: int, run_id: str, pattern: str) -> str:
            """The plan line from node 0, and an allocator failure from node 1 of the failed run."""
            del env_
            if pattern == Runner.LOGGED_PLAN:
                return "per layer in model.layers: full 0-2; off 3-7"
            return "Tried to allocate 7.58 GiB (NPU 14)" if node == 1 and run_id == "run_B_off2-7" else ""

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self._write(out, "A_base_off_0-2", step_trainer=6051.0, step_trainer_sd=20.0,
                        max_allocated_gb=48.8353, max_reserved_gb=54.5547)
            self._write(out, "A_off3-7", step_trainer=6050.0, step_trainer_sd=21.0,
                        max_allocated_gb=48.8353, max_reserved_gb=54.5547)
            self._write(out, "B_off2-7", failed=True, max_allocated_gb=54.6717, max_reserved_gb=55.4199)
            env_file = out / "cluster.env"
            env_file.write_text(_ENV, encoding="utf-8")
            printed = io.StringIO()
            with (
                patch.object(Runner, "OUT", out),
                patch.object(Runner, "CLUSTER_ENV", env_file),
                patch.object(Runner, "remote_grep", fake_grep),
                contextlib.redirect_stdout(printed),
            ):
                Runner.results(env)

        text = printed.getvalue()
        self.assertIn("A base_off_0-2   ok       6051.0   20.0  48.8353  54.5547   5.72  "
                      "per layer in model.layers: full 0-2; off 3-7", text)
        self.assertIn("B off2-7         FAILED", text)
        self.assertIn("node1: Tried to allocate 7.58 GiB (NPU 14)", text)
        self.assertIn(f"{'A off3_5-7':16s} not run", text)
        self.assertIn("plan over mode off vs the same plan over full: step_trainer +1.000 ms, "
                      "max_allocated_gb +0.000 GiB, max_reserved_gb +0.000 GiB", text)
        self.assertIn("linear layers kept first (0, 1) vs last (5, 6): needs both runs to succeed", text)
        self.assertIn("allocator setting in cluster.env now: off", text)

    def test_a_plan_named_outside_its_block_gets_a_line_and_its_comparison(self):
        """Full run in both blocks on 2 nodes: both lines, the allocator comparison at full, and no
        line for a plan that neither block lists nor ran."""
        env = {"nodes": ["n0", "n1"], "log_dir": "/logs", "ssh_user": "root"}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "recompute_plan_tests_2nodes"
            self._write(out, "A_full", step_trainer=6436.5, step_trainer_sd=4.6,
                        max_allocated_gb=26.1891, max_reserved_gb=34.9453)
            self._write(out, "B_full", step_trainer=6440.0, step_trainer_sd=5.0,
                        max_allocated_gb=26.1891, max_reserved_gb=27.0)
            env_file = Path(folder) / "cluster.env"
            env_file.write_text(_ENV, encoding="utf-8")
            printed = io.StringIO()
            with (
                patch.object(Runner, "CLUSTER_ENV", env_file),
                patch.object(Runner, "remote_grep", lambda *args: ""),
                contextlib.redirect_stdout(printed),
            ):
                Runner.results(env, out)

        text = printed.getvalue()
        self.assertIn(f"2 node(s) in cluster.env, tests in {out}", text)
        self.assertIn("A full           ok       6436.5    4.6  26.1891  34.9453   8.76  no plan line", text)
        self.assertIn("B full           ok       6440.0    5.0  26.1891  27.0000   0.81  no plan line", text)
        self.assertIn("expandable segments vs the default allocator, full: step_trainer +3.500 ms, "
                      "max_allocated_gb +0.000 GiB, max_reserved_gb -7.945 GiB", text)
        self.assertIn("expandable segments vs the default allocator, off3-7: needs both runs to succeed", text)
        self.assertNotIn("A off2-7", text)


if __name__ == "__main__":
    unittest.main()
