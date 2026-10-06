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


if __name__ == "__main__":
    unittest.main()
