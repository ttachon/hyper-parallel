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
"""Split profiled training steps into ND's real-profiling parts.

``run_nd --real_csv`` (``ParallelizeLayer.compare_with_csv``) compares ND's
estimate with a measured run given as a CSV of per-part times (read by
``debug.get_comm_classified_data``). This module writes that CSV from
``torch.profiler`` Chrome traces, one per rank.

Each ``ProfilerStep`` is split, on the thread that ran it, into:

- ``<dim>_wait``: time blocked in a ``torch.distributed`` call or a ``Work.wait()``,
  typed by the nearest HyperParallel call site above it. FSDP gathers give
  ``op_wait``; FSDP reductions, gradient clipping and trainer syncs give
  ``dp_wait``; the tensor, sequence, context, expert and pipeline parallel
  modules give ``mp``, ``sp``, ``cp``, ``ep`` and ``pp`` waits.
- ``comp``: time inside top-level ATen ops that are not communication.
- ``idle``: the rest (Python, hooks, data loading).

The call sites are Python frames, so traces need ``with_stack=True``. What is
measured is host-side blocking: exact for host-synchronous backends such as
gloo, while on accelerators a host wait is not the exposed device time.

An Ascend run directory is read instead through ``step_trace_time.csv`` and
``communication.json``, which need neither Python frames nor host blocking; see
``split_ascend_output``. Pass the run directory in place of a trace file.

Example:
    python -m hyper_parallel.auto_parallel.sapp_nd.nd.trace_classify rank*.pt.trace.json.gz \
        --dims DP=4,PP=1,MB=2,MBS=1,OP=4 --csv real.csv

    python -m hyper_parallel.auto_parallel.sapp_nd.nd.trace_classify profiling_dp64_ep16_op2 \
        --dims DP=64,MP=1,PP=1,CP=1,EP=16,MB=1,OP=2 --csv real.csv
"""

import argparse
import csv
import gzip
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import hyper_parallel.auto_parallel.sapp_nd.nd.debug as Debug
import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import logger

TIME = "time"
COMP = str(Debug.RealParts.COMP)
DP_WAIT = str(Debug.RealParts.DP_WAIT)
MP_WAIT = str(Debug.RealParts.MP_WAIT)
EP_WAIT = str(Debug.RealParts.EP_WAIT)
CP_WAIT = str(Debug.RealParts.CP_WAIT)
PP_WAIT = str(Debug.RealParts.PP_WAIT)
# Columns real_in_parts folds into DP_WAIT and MP_WAIT.
OP_WAIT = "op_wait"
SP_WAIT = "sp_wait"
WAIT_COLUMNS = (DP_WAIT, MP_WAIT, EP_WAIT, CP_WAIT, PP_WAIT, OP_WAIT, SP_WAIT)
UNCLASSIFIED = "unclassified"
# "unsplit" carries device compute that the source cannot attribute to a pass.
PHASES = ("fw", "bw", "rec", "opt", "other", "unsplit")

ASCEND_OUTPUT = "ASCEND_PROFILER_OUTPUT"
_STEP_TRACE_CSV = "step_trace_time.csv"
_COMMUNICATION_JSON = "communication.json"
_STEP_TRACE_COLUMNS = {
    "device": "Device_id", "step": "Step", "comp": "Computing", "free": "Free", "stage": "Stage",
    "exposed": "Communication(Not Overlapped)", "comm": "Communication", "bubble": "Bubble",
}
# hcom_allGather__800_0_1@5862276110395350800 -> kind allGather, group 5862276110395350800.
_HCOM_NAME = re.compile(r"^hcom_(?P<kind>[A-Za-z]+)_+\d[\d_]*@(?P<group>\S+)$")
_COLLECTIVE_KINDS = {
    "allgather": "all_gather", "allgatherv": "all_gather", "reducescatter": "reduce_scatter",
    "reducescatterv": "reduce_scatter", "allreduce": "all_reduce", "reduce": "reduce",
    "alltoall": "all_to_all", "alltoallv": "all_to_all", "alltoallvc": "all_to_all",
    "broadcast": "broadcast", "send": "p2p", "receive": "p2p", "batchsendrecv": "p2p",
}
# HCCS, PCIE and SIO are reported again inside SDMA, so only these two are summed.
_LINK_TOTALS = ("RDMA", "SDMA")

_WAIT_PREFIX = "<built-in method wait of "
_C10D_API = re.compile(r"torch/distributed/distributed_c10d\.py\(\d+\): \w+$")
_C10D_OP = re.compile(r"^(c10d::|_c10d_functional::|c10d_functional::)")
_PHASE_PATTERNS = (
    ("bw", re.compile(r"autograd::engine::evaluate_function|torch/autograd/__init__\.py\(\d+\): backward"
                      r"|torch/_tensor\.py\(\d+\): backward")),
    ("rec", re.compile(r"torch/utils/checkpoint\.py|activation_checkpoint")),
    ("opt", re.compile(r"Optimizer\.step|torch/optim/|components/optim|clip_grad|zero_grad")),
)
# (wait column, call-site label, frame pattern): the innermost matching frame decides.
# A None column is FSDP, split by collective kind in _fsdp_site.
_RULES = (
    (SP_WAIT, "sequence parallel", re.compile(r"sequence_parallel")),
    (MP_WAIT, "tensor parallel", re.compile(r"tensor_parallel|tp_collective_lowering")),
    (CP_WAIT, "context parallel", re.compile(r"context_parallel")),
    (EP_WAIT, "expert parallel", re.compile(r"expert_parallel")),
    (PP_WAIT, "pipeline p2p", re.compile(r"pipeline_parallel")),
    (DP_WAIT, "grad-norm all-reduce", re.compile(r"clip_grad")),
    (None, "fsdp", re.compile(r"core/fully_shard/")),
    (DP_WAIT, "trainer all-reduce", re.compile(r"hyper_parallel/trainer/")),
)
_KIND_HINTS = (
    ("all_gather", re.compile(r"all_gather|allgather|unshard")),
    ("reduce_scatter", re.compile(r"reduce_scatter|post_backward|finalize_per_param_reductions")),
    ("all_reduce", re.compile(r"all_reduce|allreduce")),
    ("all_to_all", re.compile(r"all_to_all|alltoall")),
    ("p2p", re.compile(r"isend|irecv|send|recv|p2p")),
)
_DTYPE_BYTES = {
    "float": 4, "double": 8, "c10::BFloat16": 2, "c10::Half": 2, "long int": 8,
    "int": 4, "short int": 2, "bool": 1, "unsigned char": 1, "signed char": 1,
}


@dataclass
class SiteStats:
    """Communication attributed to one call site within one step.

    Attributes:
        ms: Blocked time, in milliseconds.
        calls: ``torch.distributed`` calls issued from the site.
        waits: ``Work.wait()`` calls made outside those calls.
        payload_bytes: Logical tensor bytes of the collectives, largest argument each.
        transit_ms: Device time moving data; Ascend only, 0 from a Chrome trace.
        wait_ms: Device time waiting for peers; Ascend only, 0 from a Chrome trace.
    """

    ms: float = 0.0
    calls: int = 0
    waits: int = 0
    payload_bytes: int = 0
    transit_ms: float = 0.0
    wait_ms: float = 0.0


@dataclass
class StepSplit:
    """One rank's ``ProfilerStep`` split into ND parts, in milliseconds.

    Attributes:
        rank: Rank read from the trace file name, -1 when absent.
        step: Profiler step name.
        time: Step duration.
        comp: Compute time per phase of ``PHASES``.
        waits: Blocked time per column of ``WAIT_COLUMNS``, plus ``UNCLASSIFIED``.
        sites: Per call-site detail, keyed by ``(column, label)``.
    """

    rank: int
    step: str
    time: float
    comp: Dict[str, float]
    waits: Dict[str, float]
    sites: Dict[Tuple[str, str], SiteStats] = field(default_factory=dict)

    @property
    def idle(self) -> float:
        """Step time that is neither compute nor communication."""
        return self.time - sum(self.comp.values()) - sum(self.waits.values())


@dataclass(frozen=True)
class _Frame:
    """An open event and what its children inherit."""

    end: float
    name: str
    comm: Optional[Tuple[str, str]]
    under_op: bool
    phases: FrozenSet[str]


_ROOT = _Frame(float("inf"), "", None, False, frozenset())


def load_trace_events(path: str) -> List[dict]:
    """Return the complete (``"X"``) events of a Chrome trace.

    Args:
        path: ``.json`` or gzipped ``.json.gz`` trace written by ``torch.profiler``.

    Returns:
        The trace events whose phase is ``"X"``.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [event for event in json.load(handle)["traceEvents"] if event.get("ph") == "X"]


def classify_call(stack_names: Sequence[str]) -> Tuple[str, str]:
    """Return the wait column and call-site label of a communication call.

    Args:
        stack_names: Event names from the outermost open frame down to the call itself.

    Returns:
        ``(column, label)`` decided by the innermost frame matching a rule, or
        ``(UNCLASSIFIED, UNCLASSIFIED)`` when no frame matches.
    """
    for name in reversed(stack_names):
        for column, label, pattern in _RULES:
            if pattern.search(name):
                return _fsdp_site(stack_names) if column is None else (column, label)
    return UNCLASSIFIED, UNCLASSIFIED


def _collective_kind(stack_names: Sequence[str]) -> str:
    """Collective kind named by the innermost frame that names one."""
    for name in reversed(stack_names):
        for kind, pattern in _KIND_HINTS:
            if pattern.search(name):
                return kind
    return "unknown"


def _fsdp_site(stack_names: Sequence[str]) -> Tuple[str, str]:
    """FSDP parameter gathers are ND's optimizer-parallel waits, reductions its DP waits."""
    kind = _collective_kind(stack_names)
    if kind == "all_gather":
        return OP_WAIT, "fsdp param all-gather"
    return DP_WAIT, f"fsdp grad {kind.replace('_', '-')}"


def _is_comm_root(name: str, cat: Optional[str]) -> bool:
    """Whether an event blocks on communication: a c10d API frame, a wait, or a c10d op."""
    if cat == "python_function":
        return name.startswith(_WAIT_PREFIX) or _C10D_API.search(name) is not None
    return cat == "cpu_op" and _C10D_OP.match(name) is not None


def _shape_numel(dims: list) -> int:
    """Element count of a shape, or the sum over a tensor list's shapes."""
    if dims and all(isinstance(dim, list) for dim in dims):
        return sum(_shape_numel(dim) for dim in dims)
    if not dims or not all(isinstance(dim, int) for dim in dims):
        return 0
    numel = 1
    for dim in dims:
        numel *= dim
    return numel


def _payload_bytes(event: dict) -> int:
    """Bytes of the largest tensor argument of a c10d op."""
    args = event.get("args", {})
    sizes = [_shape_numel(dims) * _DTYPE_BYTES.get(dtype, 4)
             for dims, dtype in zip(args.get("Input Dims", []), args.get("Input type", [])) if dims]
    return max(sizes, default=0)


def _merge(intervals: Sequence[Tuple[float, float]]) -> List[List[float]]:
    """Sorted, disjoint union of ``[start, end)`` intervals."""
    merged: List[List[float]] = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return merged


def _length_outside(intervals: Sequence[Tuple[float, float]], holes: Sequence[Tuple[float, float]]) -> float:
    """Length of the union of ``intervals`` once the union of ``holes`` is removed."""
    kept, cut = _merge(intervals), _merge(holes)
    overlap, first = 0.0, 0
    for start, stop in kept:
        while first < len(cut) and cut[first][1] <= start:
            first += 1
        index = first
        while index < len(cut) and cut[index][0] < stop:
            overlap += min(stop, cut[index][1]) - max(start, cut[index][0])
            index += 1
    return sum(stop - start for start, stop in kept) - overlap


class _StepSplitter:
    """Walks one thread's events in time order, keeping the open call stack."""

    def __init__(self, stop: float) -> None:
        """Start an empty walk of a step ending at ``stop`` (trace microseconds)."""
        self.stop = stop
        self.stack: List[_Frame] = []
        self.comp: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.waits: Dict[Tuple[str, str], List[Tuple[float, float]]] = defaultdict(list)
        self.sites: Dict[Tuple[str, str], SiteStats] = defaultdict(SiteStats)

    def visit(self, event: dict) -> None:
        """Account one event, given in ``(ts, -dur)`` order."""
        while self.stack and self.stack[-1].end <= event["ts"]:
            self.stack.pop()
        parent = self.stack[-1] if self.stack else _ROOT
        name, cat = event["name"], event.get("cat")
        phases = set(parent.phases) | {phase for phase, pattern in _PHASE_PATTERNS if pattern.search(name)}
        if name.startswith("nn.Module: "):
            phases.add("fw")
        interval = (event["ts"], min(event["ts"] + event["dur"], self.stop))
        comm = parent.comm
        if comm is None and _is_comm_root(name, cat):
            comm = classify_call([frame.name for frame in self.stack] + [name])
            self.waits[comm].append(interval)
            if name.startswith(_WAIT_PREFIX):
                self.sites[comm].waits += 1
            else:
                self.sites[comm].calls += 1
        elif comm is None and cat == "cpu_op" and not parent.under_op:
            self.comp[self._phase(phases)].append(interval)
        if comm is not None and cat == "cpu_op" and _C10D_OP.match(name):
            self.sites[comm].payload_bytes += _payload_bytes(event)
        self.stack.append(_Frame(event["ts"] + event["dur"], name, comm, parent.under_op or cat == "cpu_op",
                                 frozenset(phases)))

    @staticmethod
    def _phase(phases: set) -> str:
        """Compute phase of a top-level op from the phases of its frames."""
        if "bw" in phases:
            return "rec" if "rec" in phases else "bw"
        return next((phase for phase in ("opt", "fw") if phase in phases), "other")

    def split(self, rank: int, step: dict) -> StepSplit:
        """Close the step; a hook blocking inside an autograd node counts as wait only."""
        holes = [interval for intervals in self.waits.values() for interval in intervals]
        comp = {phase: _length_outside(self.comp[phase], holes) / 1e3 for phase in PHASES}
        waits = dict.fromkeys(WAIT_COLUMNS + (UNCLASSIFIED,), 0.0)
        for key, intervals in self.waits.items():
            self.sites[key].ms = sum(stop - start for start, stop in _merge(intervals)) / 1e3
            waits[key[0]] += self.sites[key].ms
        return StepSplit(rank, step["name"], step["dur"] / 1e3, comp, waits, dict(self.sites))


def split_trace(path: str) -> List[StepSplit]:
    """Split every ``ProfilerStep`` of one rank's trace.

    Args:
        path: Trace file; its rank is read from a ``rank<N>`` part of the file name.

    Returns:
        One ``StepSplit`` per profiler step, in time order.
    """
    events = load_trace_events(path)
    match = re.search(r"rank(\d+)", Path(path).name)
    rank = int(match.group(1)) if match else -1
    splits = []
    for step in sorted((e for e in events if e["name"].startswith("ProfilerStep#")), key=lambda e: e["ts"]):
        start, stop = step["ts"], step["ts"] + step["dur"]
        splitter = _StepSplitter(stop)
        tid = step.get("tid")
        thread = (e for e in events if e.get("tid") == tid and e is not step and start <= e["ts"] < stop)
        for event in sorted(thread, key=lambda e: (e["ts"], -e["dur"])):
            splitter.visit(event)
        splits.append(splitter.split(rank, step))
    return splits


def ascend_output_dir(path: str) -> Path:
    """Locate the ``ASCEND_PROFILER_OUTPUT`` directory of a profiling run.

    Args:
        path: That directory, the ``*_ascend_pt`` directory above it, or the
            run directory above that.

    Returns:
        The directory holding ``step_trace_time.csv``.

    Raises:
        ValueError: If no such directory is found under ``path``.
    """
    base = Path(path)
    candidates = [base, base / ASCEND_OUTPUT, *sorted(base.glob(f"*/{ASCEND_OUTPUT}"))]
    for candidate in candidates:
        if (candidate / _STEP_TRACE_CSV).is_file():
            return candidate
    raise ValueError(f"no {ASCEND_OUTPUT}/{_STEP_TRACE_CSV} under {path}")


def is_ascend_output(path: str) -> bool:
    """Whether ``path`` points at an Ascend profiling run rather than a Chrome trace."""
    try:
        ascend_output_dir(path)
    except (ValueError, OSError):
        return False
    return True


def collective_kind(name: str) -> Tuple[str, str]:
    """Split an HCCL operator name into its collective kind and communication group.

    Args:
        name: Operator name such as ``hcom_allGather__800_0_1@5862276110395350800``.

    Returns:
        ``(kind, group)``; ``("unknown", "")`` when the name does not parse.
    """
    match = _HCOM_NAME.match(name)
    if match is None:
        return "unknown", ""
    kind = match.group("kind").lower()
    return _COLLECTIVE_KINDS.get(kind, kind), match.group("group")


def ascend_site(kind: str, section: str, dims: Dict[str, str]) -> Tuple[str, str]:
    """Return the wait column and label of an HCCL operator.

    Ascend traces carry no Python frames, so the axis comes from the collective
    kind: expert parallelism dispatches with all-to-all, FSDP gathers and reduces
    parameters, and the remaining all-reduces are data-parallel. Tensor and
    context parallelism also gather and reduce-scatter, so when either is active
    those operators need the rank sets of ``communication_matrix.json`` to be
    told apart and stay unclassified here.

    Args:
        kind: Collective kind from ``collective_kind``.
        section: ``"collective"`` or ``"p2p"``, the section of ``communication.json``.
        dims: ND dimensions of the run, from ``parse_dims``.

    Returns:
        ``(column, label)``.
    """
    if section == "p2p" or kind == "p2p":
        return PP_WAIT, "pipeline p2p"
    if kind == "all_to_all":
        return EP_WAIT, "expert all-to-all"
    if kind in ("all_gather", "reduce_scatter"):
        if int(dims.get("MP", 1)) > 1 or int(dims.get("CP", 1)) > 1:
            return UNCLASSIFIED, f"{kind} (MP or CP active)"
        return OP_WAIT, f"fsdp {kind.replace('_', '-')}"
    if kind == "all_reduce":
        return DP_WAIT, "all-reduce"
    return UNCLASSIFIED, kind


def _transit_bytes(bandwidth_info: dict) -> int:
    """Bytes moved by one collective, over the link types that do not double count."""
    total = 0.0
    for link in _LINK_TOTALS:
        total += float(bandwidth_info.get(link, {}).get("Transit Size(MB)", 0.0))
    return int(total * 2 ** 20)


def _step_collectives(step_comm: dict, dims: Dict[str, str]) -> Tuple[Dict[Tuple[str, str], SiteStats], float, set]:
    """Aggregate one step's HCCL operators per wait column, with their total elapse.

    Every operator name starts with ``hcom_``. Anything else is one of the
    summary entries CANN writes beside them, such as ``Total Op Info``, which
    repeats the whole step and would double its communication; those are skipped
    and returned so the caller can say what it ignored.
    """
    sites: Dict[Tuple[str, str], SiteStats] = defaultdict(SiteStats)
    total = 0.0
    skipped = set()
    for section, operators in step_comm.items():
        if not isinstance(operators, dict):
            continue
        for name, entry in operators.items():
            if not name.startswith("hcom_"):
                skipped.add(name)
                continue
            info = entry.get("Communication Time Info", {})
            elapse = float(info.get("Elapse Time(ms)", 0.0))
            stats = sites[ascend_site(collective_kind(name)[0], section, dims)]
            stats.ms += elapse
            stats.calls += 1
            stats.wait_ms += float(info.get("Wait Time(ms)", 0.0))
            stats.transit_ms += float(info.get("Transit Time(ms)", 0.0))
            stats.payload_bytes += _transit_bytes(entry.get("Communication Bandwidth Info", {}))
            total += elapse
    return sites, total, skipped


def _read_step_trace(path: Path) -> list:
    """Read ``step_trace_time.csv`` into one dict of floats per step.

    Raises:
        ValueError: If a column the split needs is missing.
    """
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            missing = [column for column in _STEP_TRACE_COLUMNS.values() if column not in row]
            if missing:
                raise ValueError(f"{path}: missing column(s) {', '.join(missing)}")
            rows.append({key: float(row[column]) for key, column in _STEP_TRACE_COLUMNS.items()})
    return rows


def split_ascend_output(path: str, dims: Dict[str, str]) -> List[StepSplit]:
    """Split every step of an Ascend profiling run into ND parts.

    ``step_trace_time.csv`` gives the closed top-level split of each step, in
    microseconds: computing, communication that was not overlapped with it, and
    free device time. ``communication.json`` then apportions the exposed
    communication over the parallelism axes in proportion to each axis's share of
    total HCCL elapse time, which assumes overlap is spread evenly across axes.
    Device compute is not attributed to a pass, so it lands in the ``unsplit``
    phase. The pipeline bubble is added to ``pp_wait`` on top of any p2p time;
    with ``PP=1`` both are zero, and that path is untested against real data.

    Args:
        path: The run directory, its ``*_ascend_pt`` directory, or the
            ``ASCEND_PROFILER_OUTPUT`` directory itself.
        dims: ND dimensions of the run, from ``parse_dims``.

    Returns:
        One ``StepSplit`` per profiled step, in file order.
    """
    base = ascend_output_dir(path)
    communication = {}
    comm_path = base / _COMMUNICATION_JSON
    if comm_path.is_file():
        raw = json.loads(comm_path.read_text(encoding="utf-8"))
        communication = {int(re.sub(r"\D", "", key) or 0): value for key, value in raw.items()}
    splits = []
    ignored = set()
    for row in _read_step_trace(base / _STEP_TRACE_CSV):
        step_number = int(row["step"])
        sites, elapse, skipped = _step_collectives(communication.get(step_number, {}), dims)
        ignored |= skipped
        exposed, bubble = row["exposed"] / 1e3, row["bubble"] / 1e3
        # communication.json times every collective; the CSV says how much stayed exposed.
        # Concurrent operators overlap, so their elapse sums past the step: rescale to what
        # stayed exposed, and keep the wait and transit shares on that same basis.
        scale = exposed / elapse if elapse else 0.0
        waits = dict.fromkeys(WAIT_COLUMNS + (UNCLASSIFIED,), 0.0)
        for key, stats in sites.items():
            stats.ms *= scale
            stats.wait_ms *= scale
            stats.transit_ms *= scale
            waits[key[0]] += stats.ms
        waits[PP_WAIT] += bubble
        comp = dict.fromkeys(PHASES, 0.0)
        comp["unsplit"] = row["comp"] / 1e3
        splits.append(StepSplit(int(row["device"]), f"step{step_number}", row["stage"] / 1e3,
                                comp, waits, dict(sites)))
    if ignored:
        logger.info("%s: ignored %d summary entrie(s): %s", base, len(ignored), ", ".join(sorted(ignored)))
    return splits


def mean_parts(splits: Sequence[StepSplit]) -> Dict[str, float]:
    """Mean over steps and ranks of every part, in milliseconds.

    Args:
        splits: Non-empty sequence of step splits.

    Returns:
        ``time``, ``comp``, ``comp_<phase>``, the wait columns, ``unclassified`` and ``idle``.

    Raises:
        ValueError: If ``splits`` is empty.
    """
    if not splits:
        raise ValueError("no ProfilerStep events to average")
    count = len(splits)
    mean = {TIME: sum(s.time for s in splits) / count, "idle": sum(s.idle for s in splits) / count}
    for phase in PHASES:
        mean[f"comp_{phase}"] = sum(s.comp[phase] for s in splits) / count
    mean[COMP] = sum(mean[f"comp_{phase}"] for phase in PHASES)
    for column in WAIT_COLUMNS + (UNCLASSIFIED,):
        mean[column] = sum(s.waits[column] for s in splits) / count
    return mean


def perf_parts(mean: Dict[str, float]) -> Dict[str, float]:
    """The measured step in the columns of ND's estimate (``debug.csv``).

    Real data cannot split the pipeline bubble from P2P time, so all of ``pp_wait``
    goes to ``PP_COMM``. The columns after ``TOTAL`` hold what ND does not model,
    plus the device compute an Ascend run cannot attribute to a pass.

    Args:
        mean: Output of ``mean_parts``.

    Returns:
        Milliseconds per ``PerfParts`` name up to ``TOTAL``, then the unmodelled rest.
    """
    parts = Debug.PerfParts
    return {
        str(parts.FW_COMPUTE): mean["comp_fw"],
        str(parts.BW_COMPUTE): mean["comp_bw"],
        str(parts.RECOMPUTE): mean["comp_rec"],
        str(parts.DP_COMM): mean[DP_WAIT] + mean[OP_WAIT],
        str(parts.MP_COMM): mean[MP_WAIT] + mean[SP_WAIT],
        str(parts.EP_COMM): mean[EP_WAIT],
        str(parts.CP_COMM): mean[CP_WAIT],
        str(parts.PP_COMM): mean[PP_WAIT],
        str(parts.BUBBLE): 0.0,
        str(parts.TOTAL): mean[TIME],
        "OPTIMIZER": mean["comp_opt"],
        "OTHER_COMPUTE": mean["comp_other"],
        "UNSPLIT_COMPUTE": mean["comp_unsplit"],
        "UNCLASSIFIED_COMM": mean[UNCLASSIFIED],
        "IDLE": mean["idle"],
    }


def parse_dims(text: str) -> Dict[str, str]:
    """Parse ``DP=4,MB=2`` into ND dimension columns.

    Args:
        text: Comma-separated ``NAME=VALUE`` pairs using ``dimensions`` acronyms.

    Returns:
        Column name to value, in the given order.

    Raises:
        ValueError: If a pair is malformed, a name is unknown or a value does not parse.
            ``SP`` is refused: ND reads it with ``bool()``, so any written value is True.
    """
    dims = {}
    for item in (part.strip() for part in text.split(",")):
        if not item:
            continue
        name, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"dimension {item!r} must be written NAME=VALUE")
        dim = Dim.get_dim(name.strip())
        if dim is Dim.SP:
            raise ValueError("SP cannot be written: ND parses it with bool(), so any value reads as True")
        dims[dim.name] = str(dim.from_str(value.strip()))
    return dims


def write_real_csv(splits: Sequence[StepSplit], dims: Dict[str, str], path: str) -> None:
    """Write the mean step in the format ``debug.get_comm_classified_data`` reads.

    Args:
        splits: Step splits to average.
        dims: ND dimension columns of the run, from ``parse_dims``.
        path: Output CSV path.
    """
    mean = mean_parts(splits)
    columns = [TIME, COMP, *WAIT_COLUMNS]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*dims, *columns])
        writer.writerow([*dims.values(), *(f"{mean[column]:.3f}" for column in columns)])


def write_perf_parts_csv(splits: Sequence[StepSplit], dims: Dict[str, str], path: str) -> None:
    """Write the mean step in the columns of ND's ``debug.csv``, see ``perf_parts``.

    Args:
        splits: Step splits to average.
        dims: ND dimension columns of the run, from ``parse_dims``.
        path: Output CSV path.
    """
    parts = perf_parts(mean_parts(splits))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([*dims, *parts])
        writer.writerow([*dims.values(), *(f"{value:.3f}" for value in parts.values())])


def write_detail_csv(splits: Sequence[StepSplit], path: str) -> None:
    """Write every rank and step with its per call-site split.

    Args:
        splits: Step splits to write.
        path: Output CSV path.
    """
    sites = sorted({site for split in splits for site in split.sites})
    site_columns = [f"{column}:{label} {unit}" for column, label in sites for unit in ("ms", "calls", "waits", "MiB")]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "step", TIME, *(f"comp_{p}" for p in PHASES), *WAIT_COLUMNS, UNCLASSIFIED, "idle",
                         *site_columns])
        for split in splits:
            row = [split.rank, split.step, f"{split.time:.3f}", *(f"{split.comp[p]:.3f}" for p in PHASES),
                   *(f"{split.waits[c]:.3f}" for c in WAIT_COLUMNS + (UNCLASSIFIED,)), f"{split.idle:.3f}"]
            for site in sites:
                stats = split.sites.get(site, SiteStats())
                row += [f"{stats.ms:.3f}", stats.calls, stats.waits, f"{stats.payload_bytes / 2**20:.1f}"]
            writer.writerow(row)


def format_report(splits: Sequence[StepSplit]) -> str:
    """Human-readable summary: ND real parts, ND estimate columns, then call sites.

    Args:
        splits: Step splits to summarise.

    Returns:
        The report, one line per part or call site, in milliseconds per step.
    """
    mean = mean_parts(splits)
    total = mean[TIME]
    folded = [
        (COMP, mean[COMP], ""),
        (DP_WAIT, mean[DP_WAIT] + mean[OP_WAIT], f"dp {mean[DP_WAIT]:.1f} + op {mean[OP_WAIT]:.1f}"),
        (MP_WAIT, mean[MP_WAIT] + mean[SP_WAIT], f"mp {mean[MP_WAIT]:.1f} + sp {mean[SP_WAIT]:.1f}"),
        (EP_WAIT, mean[EP_WAIT], ""),
        (CP_WAIT, mean[CP_WAIT], ""),
        (PP_WAIT, mean[PP_WAIT], "bubble included"),
        ("idle", mean["idle"] + mean[UNCLASSIFIED], f"unclassified comm {mean[UNCLASSIFIED]:.1f}"),
        (TIME, total, ""),
    ]
    lines = [f"ND real parts, mean of {len(splits)} rank-steps, ms per step"]
    lines += [f"  {part:17s} {ms:10.1f} {100 * ms / total:6.1f} %  {note}".rstrip() for part, ms, note in folded]
    lines.append("ND estimate columns (debug.csv), ms per step")
    lines += [f"  {part:17s} {ms:10.1f} {100 * ms / total:6.1f} %" for part, ms in perf_parts(mean).items()]
    lines.append("Communication by source, per step: ms, calls, of which waiting, MiB")
    count = len(splits)
    per_site = defaultdict(SiteStats)
    for split in splits:
        for site, stats in split.sites.items():
            per_site[site].ms += stats.ms / count
            per_site[site].calls += stats.calls
            per_site[site].waits += stats.waits
            per_site[site].payload_bytes += stats.payload_bytes
            per_site[site].wait_ms += stats.wait_ms / count
    for (column, label), stats in sorted(per_site.items(), key=lambda item: -item[1].ms):
        # A Chrome trace counts Work.wait() calls; Ascend reports waiting device time.
        waiting = f"{stats.wait_ms:7.1f}" if stats.wait_ms else f"{stats.waits / count:7.1f}"
        lines.append(f"  {column:17s} {label:24s} {stats.ms:10.1f} {stats.calls / count:7.1f} "
                     f"{waiting} {stats.payload_bytes / count / 2**20:9.1f}")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Command-line entry point, see the module docstring.

    Args:
        argv: Arguments, ``sys.argv[1:]`` when None.
    """
    parser = argparse.ArgumentParser(description="Split profiled steps into ND's real-profiling parts.")
    parser.add_argument("traces", nargs="+",
                        help="per-rank torch.profiler traces (*.pt.trace.json[.gz]), "
                        "or Ascend profiling run directories")
    parser.add_argument("--dims", default="", help="ND dimensions of the run, e.g. DP=4,PP=1,MB=2,OP=4")
    parser.add_argument("--csv", help="CSV for ParallelizeLayer.test_from_csv_comm_classified")
    parser.add_argument("--perf-parts", help="CSV in the columns of ND's debug.csv")
    parser.add_argument("--detail", help="CSV of every rank and step, split by call site")
    args = parser.parse_args(argv)
    ascend = [path for path in args.traces if is_ascend_output(path)]
    if (args.csv or args.perf_parts or ascend) and not args.dims:
        parser.error("--dims is required to write an ND CSV, and to classify an Ascend run")
    dims = parse_dims(args.dims)
    splits = []
    for path in args.traces:
        splits += split_ascend_output(path, dims) if path in ascend else split_trace(path)
    if not splits:
        parser.error("no profiled steps found in the given traces")
    logger.output("%s", format_report(splits))
    if args.csv:
        write_real_csv(splits, dims, args.csv)
    if args.perf_parts:
        write_perf_parts_csv(splits, dims, args.perf_parts)
    if args.detail:
        write_detail_csv(splits, args.detail)


if __name__ == "__main__":
    main()
