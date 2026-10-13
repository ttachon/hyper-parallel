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
import os
import signal
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
_KIT_CONFIG = (
    'NODES=(\n  192.168.0.143\n  192.168.0.245\n)\nSSH_USER="root"\n'
    'REMOTE_ENV_SETUP="source /home/tt/init_env.sh; export HCCL_EXEC_TIMEOUT=900"\n'
    'LOG_DIR="/home/tt/cluster_logs"\n'
)


def _run(name: str):
    """The run of that name."""
    return next(run for run in Runner.RUNS if run.name == name)


class _Sweep:
    """Stands in for the strategy sweep, which the runner loads for the kit config and the rank correlation."""

    def __init__(self, env: dict) -> None:
        """Answer every kit config with *env*."""
        self.env = env

    def read_cluster_env(self, path: Path) -> dict:
        """The kit config as the sweep reads it."""
        del path
        return self.env

    spearman = staticmethod(Sweep.spearman)


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
        """One answer from every node passes, whichever way the kit lays it out; two answers, or a failure, do not."""
        versions = "2.7.1+cpu 2.7.1.post4.dev20260402"
        one_at_a_time = (f"===== node0 (192.168.0.143) =====\n{versions}\n"
                         f"===== node1 (192.168.0.245) =====\n{versions}\n")
        together = f"\x1b[34m[node0]\x1b[0m {versions}\n\x1b[32m[node1]\x1b[0m {versions}\n"
        cases = ((one_at_a_time, 0, True), (together, 0, True),
                 (one_at_a_time.replace(f"=====\n{versions}\n", "=====\nImportError: cannot import name\n", 1), 0,
                  False),
                 ("", 1, False))
        for text, code, agree in cases:
            with self.subTest(text=text, code=code):
                answer = subprocess.CompletedProcess([], code, stdout=text)
                with (patch.object(Runner.subprocess, "run", return_value=answer) as run,
                      contextlib.redirect_stdout(io.StringIO())):
                    self.assertIs(Runner.same_environment(), agree)
                self.assertEqual(run.call_args.args[0][3:5], ["exec", "-S"])

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


class TestAllocator(unittest.TestCase):
    """A round can run on expandable segments, which always come out of the kit config again."""

    def test_setting_is_added_once_and_taken_out(self):
        """Adding twice gives one setting; taking it out gives the config back; another form is refused."""
        enabled = Runner.with_allocator(Runner.with_allocator(_KIT_CONFIG, True), True)

        self.assertEqual(enabled.count(Runner.ALLOCATOR), 1, f"enabled={enabled!r}")
        self.assertIn(f'HCCL_EXEC_TIMEOUT=900; {Runner.ALLOCATOR}"\n', enabled)
        self.assertEqual(Runner.with_allocator(enabled, False), _KIT_CONFIG)
        own = _KIT_CONFIG.replace("900", "900; export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:512")
        with self.assertRaisesRegex(SystemExit, "its own way"):
            Runner.with_allocator(own, True)

    def test_a_round_on_expandable_segments_takes_the_setting_out_even_when_stopped(self):
        """Every launch sees the setting; once the round stops, by hand here, the config is as it was."""
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "cluster.env"
            config.write_text(_KIT_CONFIG, encoding="utf-8")
            (Path(folder) / "plan.yaml").write_text("{}", encoding="utf-8")
            seen = []

            def fake_stream(command: list, log: Path) -> int:
                """Record whether the setting was on; stop the round at the second launch."""
                del command, log
                seen.append(Runner.ALLOCATOR in config.read_text(encoding="utf-8"))
                if len(seen) == 2:
                    raise KeyboardInterrupt
                return 0

            with (
                patch.object(Runner, "CLUSTER_ENV", config),
                patch.object(Runner, "OUT", Path(folder) / "out"),
                patch.object(Runner, "plan_yaml", lambda run: Path(folder) / "plan.yaml"),
                patch.object(Runner, "same_environment", lambda: True),
                patch.object(Runner, "load_sweep", lambda: _Sweep(_ENV)),
                patch.object(Runner.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess([], 0)),
                patch.object(Runner, "stream", fake_stream),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    Runner.run_all(["p01", "a1", "p02"], again=False, expandable=True)

            self.assertEqual(seen, [True, True])
            self.assertEqual(config.read_text(encoding="utf-8"), _KIT_CONFIG)

    @unittest.skipUnless(hasattr(signal, "SIGHUP"), "no SIGHUP on this platform")
    def test_a_hangup_nohup_ignores_stays_ignored(self):
        """Under nohup a lost terminal must not end the round; a kill still ends it through its cleanup."""
        hangup = getattr(signal, "SIGHUP")
        previous = signal.signal(hangup, signal.SIG_IGN)
        try:
            with Runner._signals_stop_cleanly():  # pylint: disable=protected-access
                self.assertIs(signal.getsignal(hangup), signal.SIG_IGN)
                self.assertIs(signal.getsignal(signal.SIGTERM), Runner._stop)  # pylint: disable=protected-access
        finally:
            signal.signal(hangup, previous)

    def test_results_say_the_allocator_each_run_recorded_and_a_setting_left_behind(self):
        """Runs on two allocators read as mixed; a setting left in the config is named, and reset takes it out."""
        with tempfile.TemporaryDirectory() as folder:
            out, config = Path(folder) / "out", Path(folder) / "cluster.env"
            config.write_text(Runner.with_allocator(_KIT_CONFIG, True), encoding="utf-8")
            for name, allocator in (("p01", "expandable segments"), ("a1", "default")):
                TestResults._write(out, name, step_trainer=6000.0)  # pylint: disable=protected-access
                (out / _run(name).folder / "allocator.txt").write_text(allocator + "\n", encoding="utf-8")
            printed = io.StringIO()
            with (
                patch.object(Runner, "OUT", out),
                patch.object(Runner, "CLUSTER_ENV", config),
                patch.object(Runner, "remote_grep", lambda *args: ""),
                contextlib.redirect_stdout(printed),
            ):
                Runner.results(_ENV, Sweep)
                Runner.main(["reset"])
            reset = config.read_text(encoding="utf-8")

        self.assertIn("Allocator MIXED: expandable segments for p01; default for a1", printed.getvalue())
        self.assertIn("The allocator setting is still ON in cluster.env", printed.getvalue())
        self.assertEqual(reset, _KIT_CONFIG)


class TestClassify(unittest.TestCase):
    """Only the last run of each strategy still has its profile, and only a run that finished is classified."""

    @staticmethod
    def _record(out: Path, name: str, stamp: float, **data):
        """Write one run's record, dated *stamp*, as the sweep would."""
        TestResults._write(out, name, **data)  # pylint: disable=protected-access
        states = out / _run(name).folder / "run_states.json"
        os.utime(states, (stamp, stamp))

    def _round(self, out: Path):
        """p01 then a1 (failed) on one strategy, p05 then a2 on another, p02 alone, p08 failed."""
        self._record(out, "p01", 100.0, step_trainer=3252.6)
        self._record(out, "a1", 200.0, failed=True)
        self._record(out, "p05", 300.0, step_trainer=3083.1)
        self._record(out, "a2", 400.0, step_trainer=3601.8)
        self._record(out, "p02", 500.0, step_trainer=3567.1)
        self._record(out, "p08", 600.0, failed=True)

    def test_a_run_that_kept_its_own_profile_owns_it_whatever_ran_after(self):
        """With its own folder on the node, a pick keeps its profile although its anchor repeated the strategy."""
        listing = [Runner.kept_profile_tag(_run(name)) for name in ("p01", "p05", "a2", "p02")]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self._round(out)
            with patch.object(Runner, "OUT", out):
                owners = Runner.profile_owners(listing)

        self.assertEqual(owners["classify"], ["p01", "p02", "p05", "a2"])
        self.assertEqual(owners["failed, so its profile is not whole"], ["p08", "a1"])
        self.assertEqual(len(owners["not run"]), len(Runner.RUNS) - 6)

    def test_without_its_own_folder_the_last_run_of_a_strategy_owns_the_profile(self):
        """A round run before the runner kept them apart: the strategy's folder holds the newest run's profile."""
        strategies = sorted({Runner.profile_tag(run) for run in Runner.RUNS})
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self._round(out)
            with patch.object(Runner, "OUT", out):
                owners = Runner.profile_owners(strategies)

        self.assertEqual(owners["classify"], ["p02", "a2"])
        replaced = "profile replaced by {}, a later run of the strategy before this script kept them apart"
        self.assertEqual(owners[replaced.format("a1")], ["p01"])
        self.assertEqual(owners[replaced.format("a2")], ["p05"])
        self.assertEqual(owners["failed, so its profile is not whole"], ["p08", "a1"])

    def test_a_profile_the_node_does_not_hold_is_named_rather_than_classified(self):
        """A run whose folder was cleared off the node is left out, not read from another run's."""
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self._record(out, "p02", 100.0, step_trainer=3567.1)
            with patch.object(Runner, "OUT", out):
                owners = Runner.profile_owners([])

        self.assertEqual(owners["classify"], [])
        self.assertEqual(owners["no profile on the node"], ["p02"])

    def test_a_finished_run_moves_its_profile_out_of_the_strategys_folder(self):
        """The run stage renames the profile to the run's own folder, after the sweep and before the next launch."""
        env = {"nodes": ["192.168.0.143"], "ssh_user": "root", "repo_dir": "/home/tt/hyper-parallel"}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self._record(out, "p01", 100.0, step_trainer=3252.6)
            commands, printed = [], io.StringIO()

            def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
                """Record the ssh move, and let the readiness check pass."""
                del kwargs
                commands.append(command)
                return subprocess.CompletedProcess(command, 0)

            def fake_stream(command: list, log: Path) -> int:
                """Stand in for the sweep, which leaves the run's record behind."""
                del log
                self._record(out, "p01", 200.0, step_trainer=3252.6)
                self.assertEqual(command[command.index("--out") + 1], str(out / _run("p01").folder))
                return 0

            with (patch.object(Runner, "OUT", out), patch.object(Runner.subprocess, "run", fake_run),
                  patch.object(Runner, "stream", fake_stream), contextlib.redirect_stdout(printed)):
                Runner.launch(_run("p01"), again=True, env=env)

        moves = [command for command in commands if command[0] == "ssh"]
        self.assertEqual(len(moves), 1, f"commands={commands}")
        self.assertEqual(moves[0][1], "root@192.168.0.143")
        self.assertIn("ep1_cp1_op64_tp1_pp1", moves[0][2])
        self.assertIn("p01_ep1_cp1_op64_tp1_pp1", moves[0][2])
        self.assertIn("p01: profile kept as p01_ep1_cp1_op64_tp1_pp1", printed.getvalue())

    def test_classify_copies_classifies_and_merges_the_owners_only(self):
        """Each owner's folder is copied from the first node and classified; the merge carries step and mode."""
        commands = []

        kept = {"p05": Runner.kept_profile_tag(_run("p05"))}

        def fake_run(command: list, **kwargs: object) -> subprocess.CompletedProcess:
            """Stand in for the node listing, rsync, and the classifier, which writes one row per run."""
            del kwargs
            commands.append(command)
            if command[0] == "ssh":
                # p05 kept its own profile; the other strategies have only their shared folder.
                folders = [kept["p05"]] + sorted({Runner.profile_tag(run) for run in Runner.RUNS})
                return subprocess.CompletedProcess(command, 0, stdout="\n".join(folders) + "\n")
            if command[0] != "rsync":
                labels = dict(item.split("=") for item in command[command.index("--dims") + 1].split(","))
                Path(command[command.index("--csv") + 1]).write_text(
                    ",".join(labels) + ",time,comp,op_wait\n" + ",".join(labels.values()) + ",3700.0,2900.0,300.0\n",
                    encoding="utf-8")
            return subprocess.CompletedProcess(command, 0)

        env = {"nodes": ["192.168.0.143"], "ssh_user": "root", "repo_dir": "/home/tt/hyper-parallel"}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            self._round(out)
            with (patch.object(Runner, "OUT", out), patch.object(Runner.subprocess, "run", fake_run),
                  contextlib.redirect_stdout(io.StringIO()) as printed):
                merged = Runner.classify_round(Sweep, env)
            with open(merged, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        remote = "root@192.168.0.143:/home/tt/hyper-parallel/output/sweep_profiles/"
        rsyncs = [command for command in commands if command[0] == "rsync"]
        # p05 is read from its own folder, p02 and a2 from their strategy's, each into a folder of its own here.
        self.assertEqual([command[-2] for command in rsyncs],
                         [f"{remote}ep1_cp1_op32_tp1_pp1/", f"{remote}p05_ep2_cp1_op32_tp1_pp1/",
                          f"{remote}ep2_cp1_op32_tp1_pp1/"])
        self.assertEqual([Path(command[-1]).name for command in rsyncs],
                         ["p02_ep1_cp1_op32_tp1_pp1", "p05_ep2_cp1_op32_tp1_pp1", "a2_ep2_cp1_op32_tp1_pp1"])
        self.assertTrue(all("--delete" in command for command in rsyncs))
        self.assertEqual([(row["EP"], row["OP"], row["step_trainer"], row["recompute"]) for row in rows],
                         [("1", "32", "3567.100", "full"), ("2", "32", "3083.100", "full"),
                          ("2", "32", "3601.800", "full")])
        self.assertIn("left out, profile replaced by a1, a later run of the strategy", printed.getvalue())
        self.assertIn("3 run(s) classified", printed.getvalue())


if __name__ == "__main__":
    unittest.main()
