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
"""Unit tests for the runner of ND's 16 best strategies, each with its recompute plan, on the Qwen3.5-MoE demo.

The runner works from a cluster's control node, so it is loaded here from its
path, as the sweep's own tests load the sweep.

How to run this:
    pytest tests/ut/examples/training_demo/test_nd_top_plans.py
"""
import contextlib
import csv
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

_DEMO = Path(__file__).resolve().parents[4] / "examples" / "training_demo"


def _load(name: str):
    """Load a module of the demo by path, registered under its file name."""
    spec = importlib.util.spec_from_file_location(name, _DEMO / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Runner = _load("nd_top_plans")
Sweep = _load("sweep_qwen3_5_moe")

_ENV = {"nodes": ["n0", "n1"], "log_dir": "/logs", "ssh_user": "root"}


def _run(name: str):
    """The run of that name."""
    return next(run for run in Runner.RUNS if run.name == name)


class TestRuns(unittest.TestCase):
    """The picks are ND's order of runnable strategies, each launched as the sweep would."""

    def test_picks_are_in_nds_order_and_runnable(self):
        """Places follow ND's ranks and its estimates; every strategy is one the sweep can run on 64 dies."""
        args = Sweep.parse_args([])
        ranks = [run.rank for run in Runner.PICKS]
        self.assertEqual(ranks, sorted(ranks))
        self.assertEqual([run.nd_ms for run in Runner.PICKS], sorted(run.nd_ms for run in Runner.PICKS))
        self.assertEqual(len({(run.ep, run.cp, run.op) for run in Runner.PICKS}), 16)
        for run in Runner.RUNS:
            dp = 64 // run.cp
            point = Sweep.Point(ep=run.ep, cp=run.cp, op=run.op, tp=1, pp=1, dp=dp, mb=64 // dp, mbs=1, gbs=64)
            self.assertIsNone(Sweep._unrunnable(point, args, 64), run.name)  # pylint: disable=protected-access
            self.assertIn(run.mode, ("off", "full"), run.name)

    def test_every_run_launches_in_order_once_and_each_anchor_follows_its_pick(self):
        """ORDER holds every run once, and each anchor comes right after the pick it is the baseline of."""
        self.assertEqual(sorted(Runner.ORDER), sorted(run.name for run in Runner.RUNS))
        for pick, anchor in Runner.PAIRS:
            self.assertEqual(Runner.ORDER.index(anchor), Runner.ORDER.index(pick) + 1)
            self.assertEqual((_run(pick).ep, _run(pick).op), (_run(anchor).ep, _run(anchor).op))
        self.assertEqual([run.name for run in Runner.runs_named([])], list(Runner.ORDER))
        self.assertEqual([run.name for run in Runner.runs_named(["a2", "p08", "a2"])], ["a2", "p08"])
        with self.assertRaisesRegex(SystemExit, "no run p17"):
            Runner.runs_named(["p17"])

    def test_sweep_command_states_the_strategy_and_a_mode_other_than_full(self):
        """Degrees, yaml and folder per run; the mode on the command line only where it is not full."""
        planned, all_off = Runner.sweep_command(_run("p01")), Runner.sweep_command(_run("p08"))

        self.assertNotIn("--activation-checkpoint", planned)
        self.assertEqual(all_off[-2:], ["--activation-checkpoint", "off"], f"command={all_off}")
        self.assertEqual([all_off[all_off.index(flag) + 1] for flag in ("--ep", "--cp", "--op")], ["2", "2", "32"])
        self.assertEqual(planned[planned.index("--out") + 1], str(Runner.OUT / "p01_ep1_cp1_op64_off3-7"))
        config = Path(planned[planned.index("--config") + 1])
        self.assertTrue(config.is_absolute() and Runner.REPO_ROOT in config.parents, f"config={config}")
        self.assertEqual(config.name, "nd_top_plan_off3-7.yaml")

    def test_plan_yamls_are_the_demo_yaml_with_each_recompute(self):
        """One yaml per recompute; each differs from the demo's only in its activation_checkpoint."""
        base = yaml.safe_load(Runner.TRAIN_YAML.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(Runner, "plan_yaml", lambda run: Path(folder) / f"{run.plan}.yaml"):
                paths = Runner.write_plan_yamls()
            written = {path.stem: yaml.safe_load(path.read_text(encoding="utf-8")) for path in paths}

        self.assertEqual(sorted(written), ["full", "off", "off3-7"])
        self.assertEqual(written["off3-7"]["activation_checkpoint"], {"mode": "full", "layers": {"3-7": "off"}})
        self.assertEqual(written["off"]["activation_checkpoint"], {"mode": "off"})
        self.assertEqual(written["full"]["activation_checkpoint"], {"mode": "full"})
        for name, doc in written.items():
            self.assertEqual(dict(doc, activation_checkpoint=None), dict(base, activation_checkpoint=None), name)


class TestLaunching(unittest.TestCase):
    """A round launches what has not run, on nodes that agree, and skips a busy one."""

    def test_nodes_that_disagree_are_refused(self):
        """One answer from every node passes; two answers, or a kit that fails, do not."""
        cases = (("===== node0\n2.7.1 2.7.1\n===== node1\n2.7.1 2.7.1\n", 0, True),
                 ("===== node0\n2.7.1 2.7.1\n===== node1\nImportError: cannot import name\n", 0, False),
                 ("", 1, False))
        for text, code, agree in cases:
            with self.subTest(text=text, code=code):
                answer = subprocess.CompletedProcess([], code, stdout=text)
                with (patch.object(Runner.subprocess, "run", return_value=answer),
                      contextlib.redirect_stdout(io.StringIO())):
                    self.assertIs(Runner.same_environment(), agree)

    def test_a_kept_run_is_skipped_and_a_busy_device_skips_a_launch(self):
        """A run with a record is not launched again; a busy device skips the launch; --again renames."""
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            kept = out / _run("p01").folder
            kept.mkdir()
            (kept / "run_states.json").write_text(json.dumps({"ep1": {"run_id": "r1", "failed": True}}),
                                                  encoding="utf-8")
            launched, printed = [], io.StringIO()
            readiness = iter([1, 0])
            with (
                patch.object(Runner, "OUT", out),
                patch.object(Runner.subprocess, "run",
                             lambda *args, **kwargs: subprocess.CompletedProcess([], next(readiness))),
                patch.object(Runner, "stream", lambda command, log: launched.append(command) or 0),
                contextlib.redirect_stdout(printed),
            ):
                Runner.launch(_run("p01"), again=False)
                Runner.launch(_run("p02"), again=False)
                Runner.launch(_run("p01"), again=True)

            self.assertIn("p01: kept from an earlier run (FAILED); --again runs it anew", printed.getvalue())
            self.assertIn("p02: SKIPPED, a device is busy", printed.getvalue())
            self.assertEqual(len(launched), 1, f"launched={launched}")
            self.assertEqual(launched[0][launched[0].index("--out") + 1], str(kept))
            self.assertFalse(kept.exists())
            self.assertEqual(len(list(out.glob(f"{kept.name}.*"))), 1)

    def test_a_round_needs_setup_first(self):
        """Without the plan yamls, run stops before asking the nodes anything."""
        with tempfile.TemporaryDirectory() as folder:
            with (patch.object(Runner, "plan_yaml", lambda run: Path(folder) / "missing.yaml"),
                  patch.object(Runner, "same_environment", side_effect=AssertionError("asked the nodes"))):
                with self.assertRaisesRegex(SystemExit, "run setup first: missing.yaml missing"):
                    Runner.run_all([], again=False)

    def test_setup_writes_the_yamls_and_calls_the_sweep_stages_on_4_nodes(self):
        """Setup writes the plan yamls, runs select, mirror and data on 4 nodes, then asks for versions."""
        with tempfile.TemporaryDirectory() as folder:
            calls = []
            with (
                patch.object(Runner, "OUT", Path(folder)),
                patch.object(Runner, "write_plan_yamls", lambda: []),
                patch.object(Runner, "stream", lambda command, log: calls.append((command, log)) or 0),
                patch.object(Runner, "same_environment", lambda: True),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                Runner.main(["setup"])

            command, log = calls[0]
            self.assertEqual(command[command.index("--pool") + 1], str(Path("/home/tt/cluster_all.env")))
            self.assertEqual(command[command.index("--nodes") + 1], "4")
            self.assertEqual([command[i + 1] for i, word in enumerate(command) if word == "--only"],
                             ["select", "mirror", "data"])
            self.assertEqual(log, Path(folder) / "setup.log")


class TestResults(unittest.TestCase):
    """The results set each run beside ND's estimate and say what following ND costs."""

    @staticmethod
    def _write(out: Path, name: str, **data):
        """Write the sweep's run_states.json for one run."""
        folder = out / _run(name).folder
        folder.mkdir(parents=True)
        record = {"run_id": f"run_{name}", "status": "", "failed": False, **data}
        (folder / "run_states.json").write_text(json.dumps({"tag": record}), encoding="utf-8")

    def test_results_name_the_cost_of_following_nd_and_each_plan_against_full(self):
        """ND's first pick second of three, the rank correlation, a plan against full, a failure's reason."""
        def fake_grep(env: dict, node: int, run_id: str, pattern: str) -> str:
            """The plan and rule lines from node 0; an allocator failure on node 1 of p03."""
            del env
            if pattern == Runner.LOGGED_PLAN and run_id == "run_p01":
                return "per layer in model.layers: full 0-2; off 3-7"
            if pattern == Runner.LOGGED_RULE:
                return "delta_rule=hyperparallel: 6 linear-attention layers switched"
            return "Tried to allocate 7.58 GiB (NPU 14)" if node == 1 and run_id == "run_p03" else ""

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self._write(out, "p01", step_trainer=6000.0, step_trainer_sd=10.0, max_allocated_gb=47.0,
                        max_reserved_gb=53.0)
            self._write(out, "a1", step_trainer=6400.0, step_trainer_sd=5.0, max_allocated_gb=25.5,
                        max_reserved_gb=35.2)
            self._write(out, "p02", step_trainer=6100.0, step_trainer_sd=9.0, max_allocated_gb=47.5,
                        max_reserved_gb=58.5)
            self._write(out, "p03", failed=True)
            self._write(out, "p08", step_trainer=5900.0, step_trainer_sd=8.0, max_allocated_gb=35.0,
                        max_reserved_gb=40.0)
            printed = io.StringIO()
            with (
                patch.object(Runner, "OUT", out),
                patch.object(Runner, "remote_grep", fake_grep),
                contextlib.redirect_stdout(printed),
            ):
                Runner.results(_ENV, Sweep)
            with open(out / "results.csv", newline="", encoding="utf-8") as handle:
                table = list(csv.DictReader(handle))

        text = printed.getvalue()
        self.assertIn("p01     1  CP1 EP1 OP64    off3-7    5440.5   46.02  ok        6000.0   10.0   47.00"
                      "    53.00   6.00  full 0-2; off 3-7", text)
        self.assertIn("p03     5  CP1 EP1 OP16    off3-7    5460.6   48.14  FAILED", text)
        self.assertIn("node1: Tried to allocate 7.58 GiB (NPU 14)", text)
        self.assertIn("p04     7  CP1 EP2 OP64    off3-7    5469.4   47.27  not run", text)
        self.assertIn("ND's best measured pick, p01 (CP1 EP1 OP64, off3-7, ND's #1), measures 6000.0 ms, 2 of 3 "
                      "picks timed. The fastest is p08 (CP2 EP2 OP32, off) at 5900.0 ms, so following ND costs "
                      "1.7%.", text)
        self.assertIn("Rank correlation of ND's step with the measured one over 3 picks: -0.50", text)
        self.assertIn("CP1 EP1 OP64, off3-7 against full in this round: measured -6.2%, ND -14.9%.", text)
        self.assertIn("CP1 EP2 OP32, off3-7 against full: needs p05 and a2 to run.", text)
        self.assertIn("reserved above 58.00 GiB: p02.", text)
        self.assertEqual([row["run"] for row in table], [run.name for run in Runner.RUNS])
        self.assertEqual(table[0]["step_ms"], "6000.0")

    def test_results_before_any_run(self):
        """With nothing run, every line says so and the verdict waits."""
        with tempfile.TemporaryDirectory() as folder:
            printed = io.StringIO()
            with (
                patch.object(Runner, "OUT", Path(folder)),
                patch.object(Runner, "remote_grep", lambda *args: ""),
                contextlib.redirect_stdout(printed),
            ):
                Runner.results(_ENV, Sweep)

        self.assertEqual(printed.getvalue().count("not run"), len(Runner.RUNS))
        self.assertIn("No pick has a step time yet.", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
