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
"""Unit tests for SAPP-ND `ratios`, the correction a measured round fits.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_ratios.py
"""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

# The memory model loads before the time model, whose modules import each other through it.
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation import estimate_v2  # pylint: disable=unused-import
from hyper_parallel.auto_parallel.sapp_nd.nd import debug as Debug
from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd import ratios as Ratios
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation import estimate as Estimate
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import apply_regression_coefficients

P = Debug.PerfParts


def _entry(ep, op, step, parts, real):
    """A comparison entry: a strategy at DP 64, ND's *parts* by part, the measured columns *real*."""
    config = Dim.Dimensions([(Dim.DP, 64), (Dim.EP, ep), (Dim.OP, op)], all_dims=[Dim.DP, Dim.EP, Dim.OP])
    values = [parts.get(part, 0.0) for part in Ratios.SCORE_PARTS]
    return (config, 1024, step, sum(values), values, real)


def _round():
    """Three strategies ND prices in units of 100 ms of compute, whose measured busy times it misses."""
    return [
        _entry(1, 64, 700.0, {P.FW_COMPUTE: 2.0, P.BW_COMPUTE: 4.0, P.DP_COMM: 1.0, P.BUBBLE: 1e-18},
               {"comp": 540.0, "op_wait": 90.0, "dp_wait": 0.0, "ep_wait": 0.0, "pp_wait": 0.0}),
        _entry(2, 32, 650.0, {P.FW_COMPUTE: 2.0, P.BW_COMPUTE: 4.0, P.DP_COMM: 0.5, P.DP_REDUCE: 0.1,
                              P.EP_COMM: 1.0},
               {"comp": 600.0, "op_wait": 20.0, "dp_wait": 5.0, "ep_wait": 4.0, "pp_wait": 0.0}),
        _entry(2, 1, 720.0, {P.FW_COMPUTE: 2.0, P.BW_COMPUTE: 4.0, P.DP_COMM: 0.4, P.DP_REDUCE: 0.3,
                             P.EP_COMM: 1.0},
               {"comp": 600.0, "op_wait": 40.0, "dp_wait": 60.0, "ep_wait": 5.0, "pp_wait": 0.0}),
    ]


class TestRatios(unittest.TestCase):
    """What a round measured over what ND estimated, one ratio per part, as run_nd -c reads them."""

    def test_each_part_takes_its_measured_share(self):
        """
        Feature: ratios.fit_ratios.
        Description: The round above: compute 18 units against 1740 ms, FSDP 1.9
            against 150, the all-reduce 0.4 against 65, EP 2 against 9; no TP, CP
            or pipeline, and a bubble that is a rounding residue.
        Expectation: A ratio of sums per part; the parts ND does not price take
            compute's, the bubble P2P's.
        """
        ratios = Ratios.fit_ratios(_round())
        self.assertAlmostEqual(ratios["COMPUTE"], 1740 / 18)
        self.assertAlmostEqual(ratios["DP_COMM"], 150 / 1.9)
        self.assertAlmostEqual(ratios["DP_REDUCE"], 65 / 0.4)
        self.assertAlmostEqual(ratios["EP_COMM"], 9 / 2)
        for key in ("MP_COMM", "CP_COMM", "PP_COMM", "BUBBLE"):
            self.assertAlmostEqual(ratios[key], ratios["COMPUTE"], msg=key)

    def test_the_round_sums_to_what_it_measured(self):
        """
        Feature: ratios.corrected.
        Description: The round corrected by its own ratios.
        Expectation: Its corrected estimates add up to its busy time, as a ratio
            of sums makes them; each strategy keeps its own error.
        """
        entries = _round()
        ratios = Ratios.fit_ratios(entries)
        estimates = [Ratios.corrected(dict(zip(Ratios.SCORE_PARTS, entry[4])), ratios) for entry in entries]
        self.assertAlmostEqual(sum(estimates), sum(Ratios.busy(entry) for entry in entries), places=6)
        self.assertEqual([round(Ratios.busy(entry), 1) for entry in entries], [630.0, 629.0, 705.0])

    def test_run_nd_scores_as_the_ratios_correct(self):
        """
        Feature: the ratio file and estimate.apply_regression_coefficients.
        Description: The round's ratios written, read back and applied to each
            strategy's parts the way run_nd -c applies them.
        Expectation: The file holds every key the reader asks for, and the score
            it gives is the corrected estimate.
        """
        entries = _round()
        ratios = Ratios.fit_ratios(entries)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "ratios.json")
            Ratios.write_ratios(path, ratios, entries, "real_all.csv")
            with open(path, encoding="utf-8") as handle:
                coeffs = json.load(handle)
        self.assertEqual(coeffs["_fitted_on"]["strategies"], ["DP 64 OP 64", "DP 64 EP 2 OP 32", "DP 64 EP 2"])
        for entry in entries:
            debugger = Debug.Debug(entry[0], P, enable=True)
            for part, value in zip(Ratios.SCORE_PARTS, entry[4]):
                debugger.info[part] = value
            score = apply_regression_coefficients(coeffs, debugger, entry[3])
            self.assertAlmostEqual(score, Ratios.corrected(dict(zip(Ratios.SCORE_PARTS, entry[4])), ratios))

    def test_each_strategy_is_predicted_by_the_others(self):
        """
        Feature: ratios.leave_one_out.
        Description: The round, each strategy left out of the fit in turn; and a
            round of one.
        Expectation: A strategy's estimate uses the other two's ratios; alone,
            nothing predicts it.
        """
        entries = _round()
        held = Ratios.leave_one_out(entries)
        others = Ratios.fit_ratios(entries[1:])
        self.assertAlmostEqual(held[0][1], Ratios.corrected(dict(zip(Ratios.SCORE_PARTS, entries[0][4])), others))
        self.assertEqual(Ratios.leave_one_out(entries[:1])[0][1], None)

    def test_the_report_names_the_corrected_pick(self):
        """
        Feature: ratios.report.
        Description: The round's report.
        Expectation: The ratios, one line per strategy in the corrected order, and
            what following the corrected ND costs against the fastest step.
        """
        entries = _round()
        lines = Ratios.report(entries, Ratios.fit_ratios(entries))
        self.assertTrue(lines[0].startswith("Ratios"))
        pick = [line for line in lines if line.startswith("Corrected, ND ranks")]
        self.assertEqual(len(pick), 1)
        self.assertIn("the fastest, DP 64 EP 2 OP 32, 650.0", pick[0])

    def test_the_report_ranks_on_the_trainer_step_where_the_round_has_one(self):
        """
        Feature: ratios.report, ratios.step.
        Description: The round with the trainer's own step time added to each
            strategy's measured parts, ordered against the profiled steps: EP 1
            is the dearest profiled at 700 ms and the cheapest unprofiled at 640,
            which is what the profiler's 0.33 to 0.84 s at EP 1 does to a round.
        Expectation: The report's fastest, its cost of following and its step
            column are the trainer's, and the last line says how many strategies
            it had one for; the busy time, fitted part by part, is untouched.
        """
        entries = _round()
        trainer = {(1, 64): 640.0, (2, 32): 660.0, (2, 1): 700.0}
        for entry in entries:
            key = (int(entry[0].val(Dim.EP)), int(entry[0].val(Dim.OP)))
            entry[5][Debug.TRAINER_STEP] = trainer[key]
        self.assertEqual([Ratios.step(entry) for entry in entries], [640.0, 660.0, 700.0])

        lines = Ratios.report(entries, Ratios.fit_ratios(entries))
        pick = [line for line in lines if line.startswith("Corrected, ND ranks")][0]
        self.assertIn("the fastest, DP 64 OP 64, 640.0", pick)
        self.assertIn("the trainer's own on 3 of 3 strategies", lines[-1])
        self.assertAlmostEqual(Ratios.busy(entries[0]), 630.0)

    def test_the_report_falls_back_to_the_profiled_step(self):
        """
        Feature: ratios.report, ratios.step.
        Description: The same round as every other test here, whose strategies
            carry no trainer step, and one where a single strategy has one.
        Expectation: The profiled step is used where there is nothing better, and
            the last line counts how many strategies each column covered, so a
            mixed round cannot be read as a like for like comparison.
        """
        entries = _round()
        self.assertEqual([Ratios.step(entry) for entry in entries], [700.0, 650.0, 720.0])
        lines = Ratios.report(entries, Ratios.fit_ratios(entries))
        self.assertIn("the trainer's own on 0 of 3 strategies", lines[-1])

        entries[2][5][Debug.TRAINER_STEP] = 600.0
        mixed = Ratios.report(entries, Ratios.fit_ratios(entries))
        self.assertIn("the trainer's own on 1 of 3 strategies", mixed[-1])
        self.assertIn("the fastest, DP 64 EP 2, 600.0",
                      [line for line in mixed if line.startswith("Corrected, ND ranks")][0])
    def test_a_file_missing_ratios_takes_what_the_fit_would_give(self):
        """
        Feature: estimate.apply_regression_coefficients on a partial ratios file.
        Description: A file holding COMPUTE and DP_COMM alone, as one edited by
            hand or fitted before a part existed would, applied twice to a
            strategy that also has an all-reduce, expert traffic and a bubble;
            then a file without COMPUTE.
        Expectation: No crash. The all-reduce takes DP_COMM's ratio, the bubble
            P2P's, itself compute's, and every other part compute's, as
            fit_ratios fills a part it cannot fit; each missing key is named
            once; a file without compute's ratio, the unit every other falls
            back on, is refused.
        """
        parts = {P.FW_COMPUTE: 2.0, P.BW_COMPUTE: 4.0, P.DP_COMM: 0.5, P.DP_REDUCE: 0.1,
                 P.EP_COMM: 1.0, P.BUBBLE: 0.2}
        filled = {"COMPUTE": 100.0, "DP_COMM": 80.0, "DP_REDUCE": 80.0, "MP_COMM": 100.0,
                  "EP_COMM": 100.0, "CP_COMM": 100.0, "PP_COMM": 100.0, "BUBBLE": 100.0}
        config = _entry(2, 32, 0.0, {}, {})[0]
        with patch.object(Estimate, "_MISSING_RATIOS", set()), patch.object(Estimate, "nd_logger") as nd_logger:
            for _ in range(2):
                debugger = Debug.Debug(config, P, enable=True)
                for part in Ratios.SCORE_PARTS:
                    debugger.info[part] = parts.get(part, 0.0)
                score = apply_regression_coefficients({"COMPUTE": 100.0, "DP_COMM": 80.0}, debugger,
                                                      sum(parts.values()))
                self.assertAlmostEqual(score, Ratios.corrected(parts, filled))
            named = sorted(call.args[1] for call in nd_logger.error.call_args_list)
            self.assertEqual(named, ["BUBBLE", "CP_COMM", "DP_REDUCE", "EP_COMM", "MP_COMM", "PP_COMM"])
            with self.assertRaises(ValueError):
                apply_regression_coefficients({"DP_COMM": 80.0}, debugger, 1.0)

    def test_a_round_without_compute_cannot_be_fitted(self):
        """
        Feature: ratios.fit_ratios.
        Description: A round whose strategies ND prices no compute for.
        Expectation: Refused: no ratio has a unit to fall back on.
        """
        entry = _entry(1, 64, 10.0, {P.DP_COMM: 1.0}, {"comp": 5.0, "op_wait": 5.0})
        with self.assertRaises(ValueError):
            Ratios.fit_ratios([entry])


if __name__ == "__main__":
    unittest.main()
