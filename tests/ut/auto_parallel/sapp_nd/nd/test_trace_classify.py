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
"""Unit tests for SAPP-ND `trace_classify`, the profiled-step splitter.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_trace_classify.py
"""
import csv
import gzip
import json
import os
import pathlib
import shutil
import tempfile
import unittest

from hyper_parallel.auto_parallel.sapp_nd.nd import debug as Debug
from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd import trace_classify as TC

_FSDP_PARAM = "hyper_parallel/core/fully_shard/hsdp_param.py"
_C10D = "torch/distributed/distributed_c10d.py"


def _event(name, cat, ts, dur, tid=1, args=None):
    """Chrome trace complete event, times in microseconds."""
    return {"ph": "X", "name": name, "cat": cat, "ts": ts, "dur": dur, "tid": tid, "args": args or {}}


def _py(name, ts, dur):
    return _event(name, "python_function", ts, dur)


def _op(name, ts, dur, dims=None):
    args = {"Input Dims": dims, "Input type": ["float"] * len(dims)} if dims else None
    return _event(name, "cpu_op", ts, dur, args=args)


def _step_events():
    """One 1 ms step touching every rule; expected parts are in _EXPECTED."""
    return [
        _event("ProfilerStep#0", "user_annotation", 0, 1000),
        _py("nn.Module: Model_0", 0, 200),
        _op("aten::mm", 10, 100),
        _py(f"{_FSDP_PARAM}(1027): _get_unsharded_param_data", 120, 60),
        _py(f"{_C10D}(4220): all_gather_into_tensor", 125, 50),
        _op("c10d::_allgather_base_", 126, 4, dims=[[8], [2]]),
        # The post-backward wait runs inside an autograd node: wait, not compute.
        _op("autograd::engine::evaluate_function: MmBackward0", 300, 200),
        _py("hyper_parallel/core/fully_shard/hsdp_state.py(300): post_backward", 400, 50),
        _py("<built-in method wait of PyCapsule object at 0x1>", 410, 30),
        _py(f"{_FSDP_PARAM}(1105): reduce_scatter_grad", 505, 50),
        _py(f"{_C10D}(4745): reduce_scatter_tensor", 510, 40),
        _op("c10d::_reduce_scatter_base_", 511, 38, dims=[[2], [8]]),
        _py("hyper_parallel/distributed/_builder/tp_collective_lowering.py(122): lower", 560, 20),
        _py(f"{_C10D}(3084): all_reduce", 562, 10),
        # The path matches both the sequence and the tensor parallel rules: sequence wins.
        _py("hyper_parallel/distributed/tensor_parallel/sequence_parallel.py(10): gather", 580, 8),
        _py(f"{_C10D}(4220): all_gather_into_tensor", 581, 4),
        _py("hyper_parallel/distributed/expert_parallel/collectives.py(153): dispatch", 590, 22),
        _py(f"{_C10D}(4500): all_to_all_single", 591, 20),
        _py("hyper_parallel/distributed/context_parallel/attention.py(50): ring", 612, 10),
        _py(f"{_C10D}(4220): all_gather_into_tensor", 613, 8),
        _py("hyper_parallel/core/pipeline_parallel/_p2p.py(40): send", 622, 12),
        _py(f"{_C10D}(2000): isend", 623, 10),
        _py("hyper_parallel/core/utils/clip_grad.py(491): _total_norm", 640, 12),
        _py(f"{_C10D}(3084): all_reduce", 641, 10),
        _py("hyper_parallel/trainer/runtime/distributed.py(55): all_reduce", 660, 8),
        _py(f"{_C10D}(3084): all_reduce", 661, 6),
        _py("tools/other.py(1): helper", 670, 6),
        _py(f"{_C10D}(4000): barrier", 671, 4),
        _event("Optimizer.step#AdamW.step", "user_annotation", 700, 100),
        _op("aten::add_", 710, 50),
        _op("aten::copy_", 900, 20),
        # A c10d op with no Python frame above it counts its own tensor-list payload.
        _op("c10d::allreduce_", 950, 5, dims=[[[2, 3]]]),
        _event("aten::mm", "cpu_op", 100, 500, tid=2),
        _op("aten::mm", 1200, 10),
    ]


_EXPECTED_COMP = {"fw": 0.1, "bw": 0.17, "rec": 0.0, "opt": 0.05, "other": 0.02}
_EXPECTED_WAITS = {
    TC.OP_WAIT: 0.05, TC.DP_WAIT: 0.086, TC.MP_WAIT: 0.01, TC.SP_WAIT: 0.004, TC.EP_WAIT: 0.02,
    TC.CP_WAIT: 0.008, TC.PP_WAIT: 0.01, TC.UNCLASSIFIED: 0.009,
}


class TestTraceClassify(unittest.TestCase):
    """Split a synthetic profiled step into ND parts."""

    def setUp(self) -> None:
        """Write the synthetic step to a fresh temporary directory."""
        self.tmpdir = tempfile.mkdtemp()
        self.trace = os.path.join(self.tmpdir, "rank2.pt.trace.json")
        with open(self.trace, "w", encoding="utf-8") as handle:
            json.dump({"traceEvents": _step_events()}, handle)

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _path(self, name):
        return os.path.join(self.tmpdir, name)

    def _classified_csv(self, name, extra_column=None, extra_value=None):
        """A one-row classified CSV of the 2 October EP 1 OP 64 point, plus a column."""
        header = ["DP", "EP", "OP", "time", "comp", "dp_wait", "op_wait", "ep_wait",
                  "mp_wait", "cp_wait", "pp_wait", "sp_wait"]
        row = ["64", "1", "64", "6804.575", "5783.602", "559.399", "398.453",
               "0", "0", "0", "0", "0"]
        if extra_column:
            header.append(extra_column)
            row.append(extra_value)
        path = self._path(name)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerow(row)
        return path

    def test_the_reader_takes_the_trainer_step_beside_the_profiled_one(self):
        """
        Feature: debug.get_comm_classified_data.
        Description: A classified CSV carrying the harness's step_trainer column,
            the trainer's own step time for the same run, and one without it.
        Expectation: The column reaches the parts under its own name while the
            step and every part stay the profile's, and a CSV without it reads
            exactly as before.
        """
        path = self._classified_csv("trainer.csv", Debug.TRAINER_STEP, "6276.0")
        _, time, real = Debug.get_comm_classified_data(path, plot_idle=True)[0]
        self.assertAlmostEqual(time, 6804.575)
        self.assertAlmostEqual(real[Debug.TRAINER_STEP], 6276.0)
        plain = Debug.get_comm_classified_data(self._classified_csv("plain.csv"), plot_idle=True)[0]
        self.assertNotIn(Debug.TRAINER_STEP, plain[2])
        self.assertAlmostEqual(plain[1], time)

    def test_idle_stays_measured_against_the_profiled_step(self):
        """
        Feature: debug.get_comm_classified_data.
        Description: The row above, whose profiled step is 6804.6 ms, whose parts
            sum to 6741.5 and whose trainer step is 6276.0.
        Expectation: Idle is the profiled step less the parts and is NOT the
            trainer's step less them: the parts carry the profiler's cost too, so
            a residual taken across the two columns goes negative, as it does on
            4 of that round's 15 points.
        """
        path = self._classified_csv("trainer.csv", Debug.TRAINER_STEP, "6276.0")
        _, time, real = Debug.get_comm_classified_data(path, plot_idle=True)[0]
        parts = sum(value for name, value in real.items()
                    if name not in {"IDLE", "BUBBLE", Debug.TRAINER_STEP})
        self.assertAlmostEqual(real["IDLE"], time - parts, places=6)
        self.assertGreater(real["IDLE"], 0.0)
        self.assertLess(real[Debug.TRAINER_STEP] - parts, 0.0)

    def test_an_unknown_column_is_still_refused(self):
        """
        Feature: debug.get_comm_classified_data.
        Description: A CSV carrying a column that is neither a dimension, a part,
            nor the trainer's step.
        Expectation: Refused, so a misspelt dimension cannot be read as data; the
            trainer's step is admitted by name and nothing else is.
        """
        path = self._classified_csv("unknown.csv", "step_trainer_sd", "30.0")
        with self.assertRaises(ValueError):
            Debug.get_comm_classified_data(path, plot_idle=True)

    def test_split_trace_assigns_every_part(self):
        """
        Feature: trace_classify.split_trace.
        Description: Split a step that exercises every call-site rule.
        Expectation: Compute per phase, waits per ND column and idle match the synthetic timeline.
        """
        splits = TC.split_trace(self.trace)
        self.assertEqual(len(splits), 1)
        split = splits[0]
        self.assertEqual((split.rank, split.step), (2, "ProfilerStep#0"))
        self.assertAlmostEqual(split.time, 1.0)
        for phase, expected in _EXPECTED_COMP.items():
            self.assertAlmostEqual(split.comp[phase], expected, places=9, msg=f"phase={phase}")
        for column in TC.WAIT_COLUMNS + (TC.UNCLASSIFIED,):
            self.assertAlmostEqual(split.waits[column], _EXPECTED_WAITS.get(column, 0.0), places=9,
                                   msg=f"column={column}")
        self.assertAlmostEqual(split.idle, 1.0 - 0.34 - 0.197, places=9)

    def test_call_sites_count_calls_waits_and_payload(self):
        """
        Feature: trace_classify call-site detail.
        Description: Inspect the per call-site statistics of the synthetic step.
        Expectation: Calls, waits and payload bytes are attributed to the right FSDP and unclassified sites.
        """
        sites = TC.split_trace(self.trace)[0].sites
        gather = sites[(TC.OP_WAIT, "fsdp param all-gather")]
        self.assertEqual((gather.calls, gather.waits, gather.payload_bytes), (1, 0, 32))
        reduce = sites[(TC.DP_WAIT, "fsdp grad reduce-scatter")]
        self.assertEqual((reduce.calls, reduce.waits, reduce.payload_bytes), (1, 1, 32))
        self.assertAlmostEqual(reduce.ms, 0.07, places=9)
        other = sites[(TC.UNCLASSIFIED, TC.UNCLASSIFIED)]
        self.assertEqual((other.calls, other.payload_bytes), (2, 24))

    def test_classify_call_innermost_rule_wins(self):
        """
        Feature: trace_classify.classify_call.
        Description: Classify stacks where several frames match a rule.
        Expectation: The innermost matching frame decides; unknown stacks are unclassified.
        """
        fsdp_under_trainer = ["hyper_parallel/trainer/text_trainer.py(1): train_step",
                              f"{_FSDP_PARAM}(1): unshard", f"{_C10D}(1): all_gather_into_tensor"]
        self.assertEqual(TC.classify_call(fsdp_under_trainer), (TC.OP_WAIT, "fsdp param all-gather"))
        norm_under_trainer = ["hyper_parallel/trainer/base.py(1): step",
                              "hyper_parallel/core/utils/clip_grad.py(1): total_norm", f"{_C10D}(1): all_reduce"]
        self.assertEqual(TC.classify_call(norm_under_trainer), (TC.DP_WAIT, "grad-norm all-reduce"))
        self.assertEqual(TC.classify_call([f"{_C10D}(1): barrier"]), (TC.UNCLASSIFIED, TC.UNCLASSIFIED))

    def test_real_csv_round_trips_through_nd_reader(self):
        """
        Feature: trace_classify.write_real_csv.
        Description: Write the ND real-profiling CSV and read it back with ND's own reader.
        Expectation: ND parses the dimensions and parts, and real_in_parts folds op into DP and sp into MP.
        """
        path = self._path("real.csv")
        TC.write_real_csv(TC.split_trace(self.trace), TC.parse_dims("DP=4,PP=1,MB=2,OP=4"), path)
        dims, time, real = Debug.get_comm_classified_data(path, plot_idle=True)[0]
        self.assertEqual([dims.val(d) for d in (Dim.DP, Dim.PP, Dim.MBN, Dim.OP)], [4, 1, 2, 4])
        self.assertAlmostEqual(time, 1.0)
        self.assertAlmostEqual(real[TC.COMP], 0.34, places=6)
        self.assertAlmostEqual(real["IDLE"], 1.0 - 0.34 - 0.188, places=6)
        parts = Debug.real_in_parts({part: [] for part in Debug.RealParts}, real, time)
        self.assertAlmostEqual(parts[Debug.RealParts.DP_WAIT][-1], 0.136, places=6)
        self.assertAlmostEqual(parts[Debug.RealParts.MP_WAIT][-1], 0.014, places=6)
        self.assertAlmostEqual(parts[Debug.RealParts.EP_WAIT][-1], 0.02, places=6)

    def test_perf_parts_follow_nd_estimate_columns(self):
        """
        Feature: trace_classify.perf_parts.
        Description: Express the step in the columns of ND's debug.csv.
        Expectation: Columns up to TOTAL are PerfParts names and every other column adds up to TOTAL.
        """
        parts = TC.perf_parts(TC.mean_parts(TC.split_trace(self.trace)))
        names = list(parts)
        total = str(Debug.PerfParts.TOTAL)
        nd_names = [str(part) for part in Debug.PerfParts]
        self.assertEqual(names[:names.index(total) + 1], nd_names[:nd_names.index(total) + 1])
        dp_parts = parts[str(Debug.PerfParts.DP_COMM)] + parts[str(Debug.PerfParts.DP_REDUCE)]
        self.assertAlmostEqual(dp_parts, 0.136, places=9)
        self.assertAlmostEqual(sum(value for name, value in parts.items() if name != total), parts[total], places=9)

    def test_parse_dims_validates_names_and_values(self):
        """
        Feature: trace_classify.parse_dims.
        Description: Parse valid and invalid ND dimension strings.
        Expectation: Known dimensions keep their order; unknown, malformed, non-integer and SP entries raise.
        """
        self.assertEqual(TC.parse_dims(" DP=4, MB=2 ,OP=4"), {"DP": "4", "MB": "2", "OP": "4"})
        for text in ("XX=1", "DP", "DP=x", "SP=1"):
            with self.assertRaises(ValueError, msg=f"text={text}"):
                TC.parse_dims(text)

    def test_gzip_trace_and_unnamed_rank(self):
        """
        Feature: trace_classify.load_trace_events.
        Description: Read a gzipped trace whose file name carries no rank.
        Expectation: The step splits as the plain trace does and the rank is -1.
        """
        path = self._path("trace.pt.trace.json.gz")
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"traceEvents": _step_events()}, handle)
        split = TC.split_trace(path)[0]
        self.assertEqual(split.rank, -1)
        self.assertAlmostEqual(split.waits[TC.DP_WAIT], _EXPECTED_WAITS[TC.DP_WAIT], places=9)

    def test_main_writes_all_outputs(self):
        """
        Feature: trace_classify.main.
        Description: Run the command line on the synthetic trace, with and without dimensions.
        Expectation: The three CSVs are written; asking for an ND CSV without --dims exits with an error.
        """
        real, perf, detail = self._path("real.csv"), self._path("perf.csv"), self._path("detail.csv")
        TC.main([self.trace, "--dims", "DP=4,MB=2,OP=4", "--csv", real, "--perf-parts", perf, "--detail", detail])
        with open(detail, encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["op_wait:fsdp param all-gather calls"], "1")
        self.assertTrue(os.path.isfile(real) and os.path.isfile(perf))
        with self.assertRaises(SystemExit):
            TC.main([self.trace, "--csv", real])


_STEP_TRACE = (
    "Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,Communication,Free,Stage,"
    "Bubble,Communication(Not Overlapped and Exclude Receive),Preparing\n"
    "0,1,200000.0,80000.0,20000.0,100000.0,720000.0,1000000.0,0,80000.0,5000.0\n"
)


def _hccl(elapse, wait, transit, sdma, rdma=0.0):
    """One communication.json entry in the shape CANN writes."""
    return {
        "Communication Time Info": {
            "Start Timestamp(us)": 1790181398793803.2, "Elapse Time(ms)": elapse,
            "Transit Time(ms)": transit, "Wait Time(ms)": wait, "Synchronization Time(ms)": wait,
            "Idle Time(ms)": 0.0, "Wait Time Ratio": 1.0, "Synchronization Time Ratio": 1.0,
        },
        "Communication Bandwidth Info": {
            "RDMA": {"Transit Size(MB)": rdma, "Transit Time(ms)": 0.0, "Bandwidth(GB/s)": 0.0},
            "HCCS": {"Transit Size(MB)": 0.0, "Transit Time(ms)": 0.0, "Bandwidth(GB/s)": 0.0},
            "PCIE": {"Transit Size(MB)": 0.0, "Transit Time(ms)": 0.0, "Bandwidth(GB/s)": 0.0},
            "SDMA": {"Transit Size(MB)": sdma, "Transit Time(ms)": transit, "Bandwidth(GB/s)": 148.9},
            # SIO repeats what SDMA already reports and must not be counted again.
            "SIO": {"Transit Size(MB)": sdma, "Transit Time(ms)": transit, "Bandwidth(GB/s)": 148.9},
        },
    }


_COMMUNICATION = {
    "step1": {
        "p2p": {},
        "collective": {
            "hcom_allGather__800_0_1@5862276110395350800": _hccl(30.0, 25.0, 5.0, 100.0),
            "hcom_allGather__800_1_1@5862276110395350800": _hccl(20.0, 15.0, 5.0, 100.0),
            "hcom_alltoallv__900_0_1@1111111111111111111": _hccl(50.0, 10.0, 40.0, 50.0, rdma=50.0),
            # CANN repeats the whole step in a summary entry beside the operators.
            "Total Op Info": _hccl(100.0, 50.0, 50.0, 250.0, rdma=50.0),
        },
    },
}


class TestAscendTraceClassify(unittest.TestCase):
    """Split an Ascend profiling run into ND parts."""

    def setUp(self) -> None:
        """Write a synthetic Ascend run to a fresh temporary directory."""
        self.tmpdir = tempfile.mkdtemp()
        self.run_dir = os.path.join(self.tmpdir, "profiling_dp64_ep16_op2")
        self.output = os.path.join(self.run_dir, "host_1_20260924_ascend_pt", TC.ASCEND_OUTPUT)
        os.makedirs(self.output)
        with open(os.path.join(self.output, "step_trace_time.csv"), "w", encoding="utf-8") as handle:
            handle.write(_STEP_TRACE)
        with open(os.path.join(self.output, "communication.json"), "w", encoding="utf-8") as handle:
            json.dump(_COMMUNICATION, handle)
        self.dims = TC.parse_dims("DP=64,MP=1,PP=1,CP=1,EP=16,MB=1,OP=2")

    def tearDown(self) -> None:
        """Remove the temporary directory."""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_output_dir_found_at_every_level(self):
        """
        Feature: trace_classify.ascend_output_dir.
        Description: Locate ASCEND_PROFILER_OUTPUT from the run, the *_ascend_pt and the output directory.
        Expectation: All three resolve to the same directory, and a plain trace is not mistaken for one.
        """
        for start in (self.run_dir, os.path.dirname(self.output), self.output):
            self.assertEqual(TC.ascend_output_dir(start), pathlib.Path(self.output), msg=f"start={start}")
            self.assertTrue(TC.is_ascend_output(start))
        self.assertFalse(TC.is_ascend_output(os.path.join(self.tmpdir, "absent")))
        with self.assertRaises(ValueError):
            TC.ascend_output_dir(self.tmpdir)

    def test_a_second_run_in_the_same_directory_is_refused(self):
        """
        Feature: trace_classify.ascend_output_dir.
        Description: Re-profiling into a directory that already holds a run leaves both.
        Expectation: The error names both runs rather than silently reading the older one.
        """
        second = os.path.join(self.run_dir, "host_2_20260925_ascend_pt", TC.ASCEND_OUTPUT)
        os.makedirs(second)
        shutil.copy(os.path.join(self.output, "step_trace_time.csv"), second)
        with self.assertRaises(ValueError) as raised:
            TC.ascend_output_dir(self.run_dir)
        message = str(raised.exception)
        self.assertIn("host_1_20260924_ascend_pt", message)
        self.assertIn("host_2_20260925_ascend_pt", message)
        # Naming one of them directly still works.
        self.assertEqual(TC.ascend_output_dir(second), pathlib.Path(second))

    def test_collective_kind_parses_hccl_names(self):
        """
        Feature: trace_classify.collective_kind.
        Description: Split real HCCL operator names into kind and communication group.
        Expectation: The kind is normalised, the group hash is kept, and unknown names are flagged.
        """
        self.assertEqual(TC.collective_kind("hcom_allGather__800_7_1@5862276110395350800"),
                         ("all_gather", "5862276110395350800"))
        self.assertEqual(TC.collective_kind("hcom_reduceScatter__800_0_1@58622")[0], "reduce_scatter")
        self.assertEqual(TC.collective_kind("hcom_alltoallv__900_0_1@11")[0], "all_to_all")
        self.assertEqual(TC.collective_kind("Memcpy HtoD"), ("unknown", ""))

    def test_site_mapping_needs_ranks_when_tp_or_cp_active(self):
        """
        Feature: trace_classify.ascend_site.
        Description: Map collective kinds to ND wait columns with and without TP or CP.
        Expectation: All-to-all is EP and p2p is PP; gathers are FSDP only while TP and CP are 1.
        """
        self.assertEqual(TC.ascend_site("all_to_all", "collective", self.dims)[0], TC.EP_WAIT)
        self.assertEqual(TC.ascend_site("all_gather", "collective", self.dims)[0], TC.OP_WAIT)
        self.assertEqual(TC.ascend_site("all_reduce", "collective", self.dims)[0], TC.DP_WAIT)
        self.assertEqual(TC.ascend_site("p2p", "p2p", self.dims)[0], TC.PP_WAIT)
        with_tp = TC.parse_dims("DP=8,MP=2")
        self.assertEqual(TC.ascend_site("all_gather", "collective", with_tp)[0], TC.UNCLASSIFIED)
        self.assertEqual(TC.ascend_site("broadcast", "collective", self.dims)[0], TC.UNCLASSIFIED)

    def test_context_parallelism_takes_what_only_it_can_send(self):
        """
        Feature: trace_classify.ascend_site under context parallelism.
        Description: p2p and all-to-all with CP 2, with and without pipeline and
            expert parallelism.
        Expectation: p2p is CP at one pipeline stage and PP otherwise; all-to-all is
            CP without expert parallelism and EP otherwise; gathers stay unclassified.
        """
        with_ep = TC.parse_dims("DP=32,PP=1,CP=2,EP=16")
        self.assertEqual(TC.ascend_site("p2p", "p2p", with_ep), (TC.CP_WAIT, "context p2p"))
        self.assertEqual(TC.ascend_site("all_to_all", "collective", with_ep)[0], TC.EP_WAIT)
        self.assertEqual(TC.ascend_site("all_gather", "collective", with_ep)[0], TC.UNCLASSIFIED)
        without_ep = TC.parse_dims("DP=32,PP=1,CP=2,EP=1")
        self.assertEqual(TC.ascend_site("all_to_all", "collective", without_ep),
                         (TC.CP_WAIT, "context all-to-all"))
        with_pp = TC.parse_dims("DP=16,PP=2,CP=2,EP=1")
        self.assertEqual(TC.ascend_site("p2p", "p2p", with_pp)[0], TC.PP_WAIT)

    def test_split_apportions_exposed_communication(self):
        """
        Feature: trace_classify.split_ascend_output.
        Description: Split the synthetic step, whose 100 ms of HCCL time stayed exposed for 80 ms.
        Expectation: Compute is unsplit, each axis keeps its share of the exposed time scaled by 0.8,
            idle equals the free column, SIO is not counted twice in the volume, and the
            summary entry is ignored rather than doubling the step.
        """
        splits = TC.split_ascend_output(self.run_dir, self.dims)
        self.assertEqual(len(splits), 1)
        split = splits[0]
        self.assertEqual((split.rank, split.step), (0, "step1"))
        self.assertAlmostEqual(split.time, 1000.0)
        self.assertAlmostEqual(split.comp["unsplit"], 200.0)
        self.assertAlmostEqual(sum(split.comp.values()), 200.0)
        self.assertAlmostEqual(split.waits[TC.OP_WAIT], 40.0)
        self.assertAlmostEqual(split.waits[TC.EP_WAIT], 40.0)
        self.assertAlmostEqual(sum(split.waits.values()), 80.0)
        self.assertAlmostEqual(split.idle, 720.0)
        gather = split.sites[(TC.OP_WAIT, "fsdp all-gather")]
        self.assertEqual((gather.calls, gather.payload_bytes), (2, int(200.0 * 2 ** 20)))
        # Waiting is rescaled with the elapse it belongs to, so it cannot exceed it.
        self.assertAlmostEqual(gather.wait_ms, 32.0)
        self.assertLessEqual(gather.wait_ms, gather.ms)
        self.assertEqual(split.sites[(TC.EP_WAIT, "expert all-to-all")].payload_bytes, int(100.0 * 2 ** 20))
        self.assertAlmostEqual(split.waits[TC.UNCLASSIFIED], 0.0)
        self.assertNotIn((TC.UNCLASSIFIED, "unknown"), split.sites)

    def test_missing_communication_json_leaves_waits_empty(self):
        """
        Feature: trace_classify.split_ascend_output.
        Description: Split a run whose communication.json is absent.
        Expectation: The top-level split still comes from step_trace_time.csv, with no wait attributed.
        """
        os.remove(os.path.join(self.output, "communication.json"))
        split = TC.split_ascend_output(self.run_dir, self.dims)[0]
        self.assertAlmostEqual(split.comp["unsplit"], 200.0)
        self.assertAlmostEqual(sum(split.waits.values()), 0.0)

    def test_missing_step_trace_column_is_rejected(self):
        """
        Feature: trace_classify.split_ascend_output.
        Description: Split a run whose step_trace_time.csv lacks a needed column.
        Expectation: A ValueError names the missing column instead of a silent zero.
        """
        with open(os.path.join(self.output, "step_trace_time.csv"), "w", encoding="utf-8") as handle:
            handle.write("Device_id,Step,Computing\n0,1,200000.0\n")
        with self.assertRaises(ValueError) as raised:
            TC.split_ascend_output(self.run_dir, self.dims)
        self.assertIn("Stage", str(raised.exception))

    def test_main_reads_a_run_directory_and_nd_reads_it_back(self):
        """
        Feature: trace_classify.main on an Ascend run.
        Description: Run the command line on the run directory and read the CSV with ND's reader.
        Expectation: The run is detected without a flag, ND parses the parts, and --dims is required.
        """
        real = self._path = os.path.join(self.tmpdir, "real.csv")
        TC.main([self.run_dir, "--dims", "DP=64,MP=1,PP=1,CP=1,EP=16,MB=1,OP=2", "--csv", real])
        dims, time, parts = Debug.get_comm_classified_data(real, plot_idle=True)[0]
        self.assertEqual([dims.val(d) for d in (Dim.DP, Dim.EP, Dim.OP)], [64, 16, 2])
        self.assertAlmostEqual(time, 1000.0)
        self.assertAlmostEqual(parts[TC.COMP], 200.0, places=6)
        self.assertAlmostEqual(parts[TC.EP_WAIT], 40.0, places=6)
        self.assertAlmostEqual(parts["IDLE"], 720.0, places=6)
        with self.assertRaises(SystemExit):
            TC.main([self.run_dir])


if __name__ == "__main__":
    unittest.main()
