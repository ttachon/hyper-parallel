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
"""Unit tests for the Qwen3.5-MoE strategy sweep, the harness that measures a round.

The sweep runs on a cluster's control node and imports nothing but the standard
library, so it is loaded here from its path rather than as a package module.

How to run this:
    pytest tests/ut/examples/training_demo/test_sweep_qwen3_5_moe.py
"""
import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_SWEEP = (Path(__file__).resolve().parents[4]
          / "examples" / "training_demo" / "sweep_qwen3_5_moe.py")


def _load_sweep():
    """Load the sweep script by path, registered so its dataclasses resolve."""
    spec = importlib.util.spec_from_file_location("sweep_qwen3_5_moe", _SWEEP)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Sweep = _load_sweep()


def _log_line(step, seconds, allocated=25.5217, reserved=35.1855):
    """One of the trainer's metric lines: fields sorted after the step, seconds."""
    return (f"[2026-10-02 11:20:{step:02d}] INFO step={step} epoch=0 "
            f"data/consumed_tokens={step * 8192} data/step_tokens=8192 "
            f"memory/device_max_allocated_gb={allocated} "
            f"memory/device_max_reserved_gb={reserved} "
            f"performance/step_time={seconds:.6g} performance/tokens_per_second=1234.5 "
            f"training/grad_norm=1.1 training/lr=3e-05 training/total_loss=7.5")


# One run of the demo as the trainer logs it: ten steps, 3 and 4 profiled, 5
# paying the profile's synchronous parse, the rest the honest ones. The step
# times are EP 1 OP 64 of the 2 October round, whose profiled mean was 6804.6 ms.
_RUN_LOG = "\n".join([
    "torchrun: starting 16 processes",
    _log_line(1, 9.9), _log_line(2, 7.1),
    _log_line(3, 6.8046), _log_line(4, 6.8046),
    _log_line(5, 210.4),
    _log_line(6, 6.2731), _log_line(7, 6.2845), _log_line(8, 6.268),
    _log_line(9, 6.2719), _log_line(10, 6.2825),
])


class TestHarvestStepTimes(unittest.TestCase):
    """The trainer's own step time, read from the log of the run that was profiled."""

    def test_the_profiled_window_and_its_parse_step_are_left_out(self):
        """
        Feature: sweep_qwen3_5_moe.harvest_step_times.
        Description: A ten-step run profiled over steps 3 and 4, read with the
            window's end as the last step to exclude.
        Expectation: The mean of steps 6 to 10 in milliseconds, with the spread
            and the window it covers; step 5, which pays the 210 s parse of the
            profile, is excluded with the profiled steps themselves.
        """
        harvested = Sweep.harvest_step_times(_RUN_LOG, after=5)
        self.assertEqual(harvested["step_trainer"], 6276.0)
        self.assertEqual(harvested["step_trainer_n"], 5)
        self.assertEqual(harvested["step_trainer_steps"], "6-10")
        self.assertAlmostEqual(harvested["step_trainer_sd"], 7.137, places=3)

    def test_the_figure_is_the_honest_total_of_the_same_run(self):
        """
        Feature: sweep_qwen3_5_moe.harvest_step_times.
        Description: The same run's profiled steps against its unprofiled ones.
        Expectation: The profiled steps are the dearer by half a second, which is
            the instrument's cost at EP 1 and the reason a round is not ranked on
            them; both come from one run, so nothing else differs.
        """
        harvested = Sweep.harvest_step_times(_RUN_LOG, after=5)
        profiled = 6804.6
        self.assertGreater(profiled - harvested["step_trainer"], 500.0)

    def test_a_run_with_nothing_to_harvest_yields_nothing(self):
        """
        Feature: sweep_qwen3_5_moe.harvest_step_times.
        Description: An empty log, a log of profiled steps only, and a log whose
            only honest step is one.
        Expectation: No figure rather than a wrong one, except for the single
            step, which is reported with a spread of zero.
        """
        self.assertEqual(Sweep.harvest_step_times("", after=5), {})
        profiled_only = "\n".join([_log_line(3, 6.8), _log_line(4, 6.8), _log_line(5, 210.4)])
        self.assertEqual(Sweep.harvest_step_times(profiled_only, after=5), {})
        one = Sweep.harvest_step_times(_log_line(6, 6.2731), after=5)
        self.assertEqual((one["step_trainer"], one["step_trainer_n"], one["step_trainer_sd"]),
                         (6273.1, 1, 0.0))

    def test_the_peaks_read_from_the_same_text(self):
        """
        Feature: sweep_qwen3_5_moe.harvest_peaks.
        Description: The same log, read for memory.
        Expectation: The peaks the trainer logged, so one read of the log serves
            both figures.
        """
        self.assertEqual(Sweep.harvest_peaks(_RUN_LOG),
                         {"max_allocated_gb": 25.5217, "max_reserved_gb": 35.1855})


class TestMergeAndRank(unittest.TestCase):
    """The column the honest step reaches ND and the comparison by."""

    def setUp(self):
        """Two classified single-row CSVs, as the classify stage writes them."""
        self.out = Path(tempfile.mkdtemp())
        self.header = ["DP", "MP", "PP", "CP", "EP", "MB", "MBS", "OP",
                       "time", "comp", "dp_wait", "mp_wait", "ep_wait",
                       "cp_wait", "pp_wait", "op_wait", "sp_wait"]
        self.rows = {
            "ep1_op64": ["64", "1", "1", "1", "1", "1", "1", "64", "6804.575",
                         "5783.602", "559.399", "0", "0", "0", "0", "398.453", "0"],
            "ep2_op16": ["64", "1", "1", "1", "2", "1", "1", "16", "6535.249",
                         "6187.1", "41.3", "0", "42.8", "0", "0", "168.0", "0"],
        }
        self.parts = []
        for tag, row in self.rows.items():
            path = self.out / f"real_{tag}.csv"
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(self.header)
                writer.writerow(row)
            self.parts.append((tag, path))

    def _merged(self, step_times):
        """Merge the parts with *step_times* and read the result back."""
        merged = self.out / "real_all.csv"
        count = Sweep.merge_csv(self.parts, merged, step_times)
        with open(merged, newline="", encoding="utf-8") as handle:
            return count, list(csv.DictReader(handle))

    def test_the_merged_csv_carries_the_trainer_step(self):
        """
        Feature: sweep_qwen3_5_moe.merge_csv.
        Description: Two strategies merged with the step time harvested for each.
        Expectation: One column beside the profiled one, so ND's reader and the
            comparison take the honest total from the same file; the profiled
            time is kept, because every part is measured against it.
        """
        count, rows = self._merged({"ep1_op64": 6276.0, "ep2_op16": 6430.0})
        self.assertEqual(count, 2)
        self.assertEqual([row[Sweep.TRAINER_STEP_COLUMN] for row in rows],
                         ["6276.000", "6430.000"])
        self.assertEqual([row["time"] for row in rows], ["6804.575", "6535.249"])

    def test_a_strategy_without_a_harvest_leaves_the_cell_empty(self):
        """
        Feature: sweep_qwen3_5_moe.merge_csv.
        Description: A round where only one of the two runs yielded a step time.
        Expectation: An empty cell rather than a guess, which ND's reader and the
            comparison both read as "no honest total for this one".
        """
        _, rows = self._merged({"ep2_op16": 6430.0})
        self.assertEqual([row[Sweep.TRAINER_STEP_COLUMN] for row in rows], ["", "6430.000"])
        self.assertEqual(Sweep._measured_step(rows[0]), 6804.575)
        self.assertEqual(Sweep._measured_step(rows[1]), 6430.0)

    def test_the_round_is_ranked_on_the_honest_total(self):
        """
        Feature: sweep_qwen3_5_moe._measured_step.
        Description: The two strategies of the 2 October round whose order the
            profiler reversed: EP 1 profiled at 6804.6 ms against EP 2 at 6535.2,
            and the trainer's own 6276.0 against 6430.0.
        Expectation: On the profiled column EP 2 is the faster, on the trainer's
            own EP 1 is, so which column a verdict uses decides the verdict.
        """
        _, rows = self._merged({"ep1_op64": 6276.0, "ep2_op16": 6430.0})
        honest = [Sweep._measured_step(row) for row in rows]
        profiled = [float(row["time"]) for row in rows]
        self.assertLess(honest[0], honest[1])
        self.assertGreater(profiled[0], profiled[1])

    def test_the_step_times_are_read_back_from_the_run_states(self):
        """
        Feature: sweep_qwen3_5_moe.load_step_times.
        Description: The run stage's record of a round, then one with no record
            at all, as a directory whose profiles were put there by hand has.
        Expectation: The harvested step per strategy, and nothing where there is
            no record, so classifying such a directory simply has no column.
        """
        states = {"ep1_op64": {"run_id": "a", "EP": 1, "OP": 64, "step_trainer": 6276.0},
                  "ep2_op16": {"run_id": "b", "EP": 2, "OP": 16},
                  "memory_pass": {"ep1_op64": {"run_id": "c", "step_trainer": 9999.0}}}
        (self.out / "run_states.json").write_text(json.dumps(states), encoding="utf-8")

        class _Out:  # only .out is read
            def __init__(self, out):
                self.out = out

        self.assertEqual(Sweep.load_step_times(_Out(self.out)), {"ep1_op64": 6276.0})
        os.remove(self.out / "run_states.json")
        self.assertEqual(Sweep.load_step_times(_Out(self.out)), {})



_ENV = {"nodes": ["n0", "n1", "n2", "n3"], "nproc": 16, "repo_dir": "/repo", "ssh_user": "root",
        "log_dir": "/logs"}


def _ranking_row(rank, ep, op, mode, score):
    """One row of ND's ranking at DP 64 with a recompute column."""
    return {"rank": str(rank), "DP": "64", "MP": "1", "PP": "1", "CP": "1", "EP": str(ep), "MB": "1",
            "MBS": "1", "OP": str(op), "recompute": mode, "memory_mb": "30000", "score": score}


class TestRecomputeDimension(unittest.TestCase):
    """Recompute is a sweep dimension: each strategy runs in each mode, which ND ranks and prices."""

    def test_modes_are_one_a_list_or_auto(self):
        """
        Feature: sweep_qwen3_5_moe.ac_modes.
        Description: The forms --activation-checkpoint takes.
        Expectation: One mode, a list in its order without repeats, all three for auto, and a
            refusal naming the flag for anything else.
        """
        cases = (("full", ("full",)), ("off,full", ("off", "full")), (" off , off ", ("off",)),
                 ("AUTO", ("off", "selective", "full")))
        for stated, expected in cases:
            with self.subTest(stated=stated):
                self.assertEqual(Sweep.ac_modes(Sweep.parse_args(["--activation-checkpoint", stated])),
                                 expected)
        with self.assertRaisesRegex(SystemExit, "--activation-checkpoint off,swap: expected some of"):
            Sweep.ac_modes(Sweep.parse_args(["--activation-checkpoint", "off,swap"]))

    def test_the_grid_runs_each_strategy_in_each_mode(self):
        """
        Feature: sweep_qwen3_5_moe.expand.
        Description: Two EP degrees under two modes, then under one.
        Expectation: Four points named by their mode under two modes; two points named as
            before under one, so a sweep of one mode keeps its earlier names.
        """
        both = Sweep.expand(Sweep.parse_args(["--ep", "1,2", "--activation-checkpoint", "off,full"]), 64)
        self.assertEqual([point.tag for point in both],
                         ["ep1_cp1_op64_tp1_pp1_ac-off", "ep1_cp1_op64_tp1_pp1_ac-full",
                          "ep2_cp1_op64_tp1_pp1_ac-off", "ep2_cp1_op64_tp1_pp1_ac-full"])
        one = Sweep.expand(Sweep.parse_args(["--ep", "1,2", "--activation-checkpoint", "off"]), 64)
        self.assertEqual([point.tag for point in one], ["ep1_cp1_op64_tp1_pp1", "ep2_cp1_op64_tp1_pp1"])

    def test_nds_picks_carry_their_mode(self):
        """
        Feature: sweep_qwen3_5_moe.pick_nd_top.
        Description: ND's ranking of one strategy in two modes, then the same ranking read by a
            sweep of one mode.
        Expectation: Under two modes each pick runs in the mode ND ranked it with; under one,
            the picks are named as before.
        """
        rows = [_ranking_row(1, 2, 32, "off", "1.0"), _ranking_row(2, 2, 32, "full", "2.0")]
        args = Sweep.parse_args(["--activation-checkpoint", "off,full"])
        picks, _ = Sweep.pick_nd_top(rows, args, 64, 2)
        self.assertEqual([(pick.point.ac, pick.point.tag) for pick in picks],
                         [("off", "ep2_cp1_op32_tp1_pp1_ac-off"), ("full", "ep2_cp1_op32_tp1_pp1_ac-full")])
        picks, _ = Sweep.pick_nd_top(rows[1:], Sweep.parse_args(["--activation-checkpoint", "full"]), 64, 1)
        self.assertEqual([pick.point.tag for pick in picks], ["ep2_cp1_op32_tp1_pp1"])

    def test_keys_and_names_follow_the_mode_only_with_several(self):
        """
        Feature: sweep_qwen3_5_moe._strategy_key and _tag_of.
        Description: One row read by a sweep of several modes and by one of a single mode.
        Expectation: The mode is part of the key and of the name only with several.
        """
        row = _ranking_row(1, 2, 32, "off", "1.0")
        self.assertEqual(Sweep._tag_of(Sweep._strategy_key(row, True)), "ep2_cp1_op32_tp1_pp1_ac-off")
        self.assertEqual(Sweep._tag_of(Sweep._strategy_key(row)), "ep2_cp1_op32_tp1_pp1")

    def test_csvs_record_each_runs_mode(self):
        """
        Feature: sweep_qwen3_5_moe.merge_csv and write_peaks_csv.
        Description: Two classified runs merged with their modes, and the run stage's record.
        Expectation: The merged CSV and memory.csv carry the mode each run ran, which ND reads
            to price each measured run in its own mode.
        """
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            parts = []
            for tag in ("a_ac-off", "a_ac-full"):
                part = out / f"real_{tag}.csv"
                part.write_text("DP,EP,time,comp\n64,2,6000,5000\n", encoding="utf-8")
                parts.append((tag, part))
            Sweep.merge_csv(parts, out / "real_all.csv", {}, {"a_ac-off": "off", "a_ac-full": "full"})
            Sweep.write_peaks_csv({"a_ac-off": {"DP": 64, "recompute": "off", "max_allocated_gb": 50.0,
                                                "max_reserved_gb": 55.0}}, out / "memory.csv")
            with open(out / "real_all.csv", newline="", encoding="utf-8") as handle:
                merged = list(csv.DictReader(handle))
            with open(out / "memory.csv", newline="", encoding="utf-8") as handle:
                memory = list(csv.DictReader(handle))
        self.assertEqual([row["recompute"] for row in merged], ["off", "full"])
        self.assertEqual(memory[0]["recompute"], "off")

    def test_runs_and_nd_get_the_modes(self):
        """
        Feature: sweep_qwen3_5_moe.launch, stage_rank and stage_compare.
        Description: A sweep of two modes launches a point of each, ranks and compares.
        Expectation: Each run states its own mode to the trainer, and ND is asked to price
            and rank both modes, in the ranking and in the comparison.
        """
        with tempfile.TemporaryDirectory() as folder:
            args = Sweep.parse_args(["--activation-checkpoint", "off,full", "--out", folder])
            sweep = Sweep.Sweep(args=args, env=_ENV)
            commands = []

            def record(command: list, capture: bool = False, check: bool = True) -> str:
                """Keep the command; a launch's output names its run id."""
                del check
                commands.append(list(command))
                return "run id : r1" if capture else ""

            (Path(folder) / "real_all.csv").write_text("DP\n64\n", encoding="utf-8")
            with patch.object(Sweep, "_run", record), patch.object(Sweep, "require_importable"), \
                    patch.object(Sweep, "load_ranking", return_value=([], "")):
                for point in Sweep.expand(args, 64):
                    Sweep.launch(sweep, point)
                Sweep.stage_rank(sweep)
                Sweep.stage_compare(sweep)
        modes = [part for command in commands for part in command
                 if part.startswith("--activation_checkpoint.mode=")]
        self.assertEqual(modes, ["--activation_checkpoint.mode=off", "--activation_checkpoint.mode=full"])
        nd_calls = [command for command in commands if Sweep.RUN_ND in command]
        self.assertEqual(len(nd_calls), 2)
        for command in nd_calls:
            self.assertEqual(command[command.index("--recompute") + 1:][:2], ["off", "full"])


if __name__ == "__main__":
    unittest.main()
