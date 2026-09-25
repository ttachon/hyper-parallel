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
"""Profile Qwen3.5-MoE across parallel strategies and score ND against them.

Runs the cropped demo once per strategy under the profiler, collects the
Ascend profiling output, classifies each run into ND's parts with
``nd.trace_classify``, and feeds the combined CSV to ``run_nd --real_csv`` so
ND's estimate is printed beside the measurement for every configuration.

Run it on the cluster's control node:

    python examples/training_demo/sweep_qwen3_5_moe.py --ep 1,2,4,8,16,32,64

No axis is swept by default: every degree defaults to 1, so a bare run
profiles a single point and each sweep names the axis it varies. Every
dimension takes a list and the sweep is their cartesian product, so this
compares four strategies:

    ... --ep 2,16 --cp 1,2

Or let ND choose. This runs ND's search at the sweep's shape and profiles the
five strategies it ranks best among those this model and trainer can run,
which tests its ranking where a search relies on it. Naming an axis as well
adds that grid, a known strategy to measure ND's picks against:

    ... --nd-top 5
    ... --nd-top 5 --ep 16

Stages run in order and each can be run alone with ``--only``, so a failed
sweep can be classified without re-running, and a changed cost model can be
re-scored without re-profiling.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent
DEFAULT_ENV = DEMO_DIR / "cluster_qwen3_5_moe.env"
DEFAULT_CONFIG = DEMO_DIR / "train_qwen3_5_moe.yaml"
RUN_ID_PATTERN = re.compile(r"run id\s*:\s*(\S+)")
# One status line per node: "node0   192.168.0.55   DEAD (exit 1) | <last line>".
NODE_STATE_PATTERN = re.compile(r"^node(\d+)\s+(\S+)\s+([^|]+?)\s*\|", re.MULTILINE)
# Rank tag, stripped so one exception raised on 16 ranks collapses to one line.
RANK_PREFIX_PATTERN = re.compile(r"^\[rank\d+\]:\s*")
PEAK_PATTERN = re.compile(
    r"memory/device_max_allocated_gb=([0-9.]+).*?"
    r"memory/device_max_reserved_gb=([0-9.]+)"
)
REMOTE_PROFILES = "output/sweep_profiles"
STAGES = ("rank", "mirror", "data", "run", "fetch", "classify", "compare", "plot")
# The degrees a strategy is named by, as ND's ranking and the classified CSV
# both spell them. SP and VPP are left out: SP only acts with TP and VPP only
# with PP, and neither of those runs on this model.
STRATEGY_DIMS = ("EP", "CP", "OP", "MP", "PP", "MB", "MBS")
RUN_ND = "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd"


def _run(command: Sequence[str], *, capture: bool = False, check: bool = True) -> str:
    """Run a command, echoing it first, and return its stdout when captured."""
    print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)
    result = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture and result.stdout:
        print(result.stdout, end="", flush=True)
    if check and result.returncode != 0:
        raise SystemExit(f"command failed with {result.returncode}")
    return result.stdout or ""


def read_cluster_env(env_path: Path) -> Dict[str, Any]:
    """Return NODES, NPROC_PER_NODE, REPO_DIR, SSH_USER and LOG_DIR from a kit config.

    The file is bash, so bash sources it rather than this parsing it: the node
    list is an array and REPO_DIR may reference other variables.
    """
    script = (
        f'set -euo pipefail; source {shlex.quote(str(env_path))}; '
        'printf "%s\\n" "${NODES[*]:-}" "${NPROC_PER_NODE:-8}" '
        '"${REPO_DIR:-}" "${SSH_USER:-root}" "${LOG_DIR:-}"'
    )
    # Every optional key carries a default: the file is sourced under set -u,
    # where one unset name aborts the shell before printf runs, so a missing
    # LOG_DIR would otherwise read as a truncated answer rather than an error.
    result = subprocess.run(
        ["bash", "-c", script], check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise SystemExit(
            f"could not read {env_path}: {result.stderr.strip() or 'sourcing failed'}")
    values = (result.stdout.splitlines() + [""] * 5)[:5]
    if not values[0].split() or not values[2]:
        raise SystemExit(f"{env_path} must set NODES and REPO_DIR")
    repo_dir = values[2].rstrip("/")
    return {
        "nodes": values[0].split(),
        "nproc": int(values[1]),
        "repo_dir": repo_dir,
        "ssh_user": values[3],
        # The kit's own default when the config leaves it out.
        "log_dir": (values[4] or f"{repo_dir}/scripts/cluster/logs").rstrip("/"),
    }


@dataclass(frozen=True)
class Point:
    """One resolved strategy: every degree concrete, nothing left to infer.

    ``dp`` is the data-parallel width the dataloader splits the batch over,
    ``world / (tp * cp * pp)``. ``op`` is ND's name for the FSDP shard width,
    which covers the data AND context axes, so it divides ``dp * cp``. ``mbs``
    is the micro-batch size and ``mb`` the micro-batches per step.
    """

    ep: int
    cp: int
    op: int
    tp: int
    pp: int
    dp: int
    mb: int
    mbs: int
    gbs: int

    @property
    def tag(self) -> str:
        """Directory-safe name carrying every degree that can vary.

        The micro-batch size appears only above 1, so a sweep that never
        changes it keeps the names its earlier runs were profiled under.
        """
        tag = f"ep{self.ep}_cp{self.cp}_op{self.op}_tp{self.tp}_pp{self.pp}"
        return f"{tag}_mbs{self.mbs}" if self.mbs > 1 else tag

    @property
    def dims(self) -> Dict[str, int]:
        """ND's dimension columns for this strategy."""
        return {"DP": self.dp, "MP": self.tp, "PP": self.pp, "CP": self.cp,
                "EP": self.ep, "MB": self.mb, "MBS": self.mbs, "OP": self.op}


def _split_ints(text: str) -> List[int]:
    """Parse a comma-separated degree list."""
    return [int(part) for part in text.split(",") if part.strip()]


def _unrunnable(point: Point, args: argparse.Namespace, world: int) -> Optional[str]:
    """Say why the trainer or this model could not run a strategy, or None."""
    if point.tp > 1:
        return ("tp_size > 1 shards the Gated DeltaNet conv1d while its groups "
                "and conv_dim stay global, so the forward raises on this model")
    if point.pp > 1:
        return ("pp_size > 1 neither raises nor pipelines: the Trainer has no "
                "pipeline schedule, so every stage group trains a full replica")
    if world % point.ep or args.num_experts % point.ep:
        return (f"ep {point.ep} must divide the world size ({world}) and the "
                f"routed expert count ({args.num_experts})")
    fsdp_width = point.dp * point.cp   # FSDP shards over the data and context axes
    if fsdp_width % point.op:
        return (f"op {point.op} must divide dp*cp ({fsdp_width}); FSDP shards "
                "over the data and context axes together")
    per_step = point.mbs * point.dp
    if point.gbs % per_step:
        return (f"global_batch_size {point.gbs} must be a multiple of "
                f"micro_batch_size*dp ({per_step}) at cp={point.cp}")
    return None


def _axis(args: argparse.Namespace, name: str) -> str:
    """The degrees an axis was given, 1 when it was not named."""
    return getattr(args, name) or "1"


def named_axes(args: argparse.Namespace) -> bool:
    """Whether the command line named any strategy axis."""
    return any(getattr(args, name) for name in ("ep", "cp", "op", "tp", "pp"))


def expand(args: argparse.Namespace, world: int) -> List[Point]:
    """Return the cartesian product of the requested degrees, validated.

    Each combination is checked against the constraints the trainer enforces at
    startup, so an impossible strategy is rejected here rather than after a
    launch and a wait.
    """
    points: List[Point] = []
    grid = itertools.product(*(_split_ints(_axis(args, name))
                               for name in ("ep", "cp", "tp", "pp")))
    for ep, cp, tp, pp in grid:
        non_dp = tp * cp * pp
        if world % non_dp:
            raise SystemExit(
                f"tp*cp*pp ({non_dp}) must divide the world size ({world})")
        dp = world // non_dp
        gbs = args.global_batch_size or world
        for op in (_split_ints(args.op) if args.op else [dp * cp]):
            point = Point(ep=ep, cp=cp, op=op, tp=tp, pp=pp, dp=dp,
                          mb=gbs // (args.micro_batch_size * dp) or 1,
                          mbs=args.micro_batch_size, gbs=gbs)
            reason = _unrunnable(point, args, world)
            if reason:
                raise SystemExit(reason)
            points.append(point)
    return points


@dataclass
class Pick:
    """One strategy of ND's ranking that the sweep will run."""

    rank: int
    score: float
    memory_mb: float
    point: Point
    tied_ops: List[int] = field(default_factory=list)


def _row_point(row: Dict[str, str], args: argparse.Namespace, gbs: int) -> Point:
    """The strategy one row of ND's ranking stands for, as the trainer runs it."""
    def degree(name: str, default: int = 1) -> int:
        """The row's degree *name*, or *default* when the ranking lacks it."""
        return int(row.get(name) or default)

    dp, mbs = degree("DP"), degree("MBS", args.micro_batch_size)
    return Point(ep=degree("EP"), cp=degree("CP"), op=degree("OP"), tp=degree("MP"),
                 pp=degree("PP"), dp=dp, mb=degree("MB", gbs // (mbs * dp) or 1),
                 mbs=mbs, gbs=gbs)


def pick_nd_top(rows: Sequence[Dict[str, str]], args: argparse.Namespace, world: int,
                count: int) -> Tuple[List[Pick], List[Tuple[int, Point, str]]]:
    """Return ND's ``count`` best runnable strategies, and what it ranked above them.

    ND ranks configurations, several of which can be one strategy to the
    trainer: SP on and off at TP 1 run identically. It also ties strategies
    whose difference it does not price, which at EP 1 is every OP. A tie is one
    prediction, so it is run once, at its widest OP: the FSDP default, and the
    one holding the least memory. The other widths are reported with it.

    Returns:
        ``(picks, passed)``: the picks in ND's order, and ``(rank, point,
        reason)`` for every strategy ND ranked above the last pick that this
        model or trainer cannot run.
    """
    gbs = args.global_batch_size or world
    groups: Dict[Any, Dict[str, Any]] = {}
    passed: List[Tuple[int, Point, str]] = []
    refused = set()
    for row in rows:
        point = _row_point(row, args, gbs)
        reason = _unrunnable(point, args, world)
        if reason:
            if point not in refused:
                refused.add(point)
                passed.append((int(row["rank"]), point, reason))
            continue
        # Exact text of the score: ND writes it at full precision, so equal
        # text is a tie and not a near miss.
        key = (point.ep, point.cp, point.tp, point.pp, point.mb, point.mbs, row["score"])
        group = groups.setdefault(key, {"rank": int(row["rank"]), "score": float(row["score"]),
                                        "widths": {}})
        group["widths"].setdefault(point.op, (point, float(row["memory_mb"])))
    picks = []
    for group in list(groups.values())[:count]:
        widest = max(group["widths"])
        point, memory_mb = group["widths"][widest]
        picks.append(Pick(rank=group["rank"], score=group["score"], memory_mb=memory_mb,
                          point=point, tied_ops=sorted(set(group["widths"]) - {widest})))
    last = picks[-1].rank if picks else 0
    return picks, [entry for entry in passed if entry[0] < last]


@dataclass
class Sweep:
    """One sweep's resolved settings, shared by every stage."""

    args: argparse.Namespace
    env: Dict[str, Any]
    points: List[Point] = field(default_factory=list)
    started: float = field(default_factory=time.monotonic)

    @property
    def world(self) -> int:
        """Total device count the sweep runs on."""
        return len(self.env["nodes"]) * self.env["nproc"]

    @property
    def out(self) -> Path:
        """Local directory holding every artifact of this sweep."""
        return self.args.out

    @property
    def profiles(self) -> Path:
        """Local directory the fetched profiling output lands in."""
        return self.out / "profiles"

    @property
    def merged_csv(self) -> Path:
        """The one classified CSV covering every configuration."""
        return self.out / "real_all.csv"

    @property
    def nd_dir(self) -> Path:
        """Where compare writes ND's plots and estimates."""
        return self.out / "nd"

    @property
    def estimates_csv(self) -> Path:
        """ND's estimate of every measured strategy, memory included."""
        return self.nd_dir / f"{self.merged_csv.stem}_estimates.csv"

    @property
    def ranking_csv(self) -> Path:
        """ND's order of every configuration it keeps at this sweep's shape."""
        return self.out / "nd_ranking.csv"

    @property
    def gbs(self) -> int:
        """Global batch size of every strategy, the world size unless given."""
        return self.args.global_batch_size or self.world

    @property
    def shape(self) -> Dict[str, Any]:
        """Everything a strategy leaves fixed and ND's estimate depends on.

        Stored beside ND's ranking, so a ranking made for another shape is
        refused rather than read as this one's: the output directory is shared
        by default, and a stale ranking would otherwise look like a fresh one.
        """
        return {"world": self.world, "layers": self.args.layers,
                "seq_len": self.args.seq_len,
                "activation_checkpoint": self.args.activation_checkpoint,
                "global_batch_size": self.gbs,
                "micro_batch_size": self.args.micro_batch_size,
                "config": self.args.config.name, "arch": self.args.arch}

    @property
    def python(self) -> str:
        """Interpreter for the analysis steps, which must import hyper_parallel."""
        return self.args.python or sys.executable

    @property
    def required_samples(self) -> int:
        """Samples the longest configuration consumes, with a margin."""
        import yaml  # pylint: disable=import-outside-toplevel

        raw = yaml.safe_load(self.args.config.read_text(encoding="utf-8"))
        iters = int(raw["training"]["train_iters"])
        return max(point.gbs for point in self.points) * iters * 2

    def kit(self, *command: str) -> List[str]:
        """Build a cluster-kit invocation bound to this sweep's config."""
        return [self.args.cluster, "-c", str(self.args.cluster_env), *command]


def mirror_code(sweep: Sweep) -> None:
    """Make every node's tree identical to this one, run outputs excluded.

    ``cluster sync`` cannot do this: it honours .gitignore, so it never ships
    the compiled indexed-dataset extension, and it never deletes, so a renamed
    operator YAML survives and the op registry then rejects a duplicate name.
    ``output/`` is excluded so a node's dataset is never removed.
    """
    local = subprocess.run(
        ["hostname", "-I"], check=True, text=True, stdout=subprocess.PIPE
    ).stdout.split()
    repo = sweep.env["repo_dir"]
    for host in sweep.env["nodes"]:
        if host in local:
            print(f"== {host}  (this machine, source) skip", flush=True)
            continue
        print(f"== {host}", flush=True)
        _run([
            "rsync", "-a", "--delete", "--exclude", ".git/", "--exclude", "output/",
            f"{repo}/", f"{sweep.env['ssh_user']}@{host}:{repo}/",
        ])


def stage_data(sweep: Sweep) -> None:
    """Rebuild the Indexed Dataset on every node at the sweep's sequence length.

    Its documents are exactly seq_length long, so raising the sequence without
    rebuilding leaves the reader with samples of the wrong size. The generator
    is deterministic, so every node produces identical files and none has to be
    shipped. cluster exec runs inside REPO_DIR with the environment hook
    sourced, so both the relative path and the interpreter resolve.
    """
    samples = sweep.args.samples or sweep.required_samples
    print(f"rebuilding the dataset: {samples} samples of {sweep.args.seq_len} tokens",
          flush=True)
    _run(sweep.kit("exec",
                   "python -m examples.training_demo.prepare_parallel_data "
                   "--output-dir ./output/training_demo/data "
                   f"--num-samples {samples} --seq-length {sweep.args.seq_len}"),
         check=False)


def _dir_name(point: Point, memory: bool) -> str:
    """Directory name for one strategy's timing or memory pass."""
    return f"{point.tag}_mem" if memory else point.tag


def launch(sweep: Sweep, point: Point, memory: bool = False) -> Optional[str]:
    """Launch one strategy and return the kit's run id.

    ``memory`` adds the allocator history the memory pass needs. It is off for
    the timing pass: recording brackets the profiled window and moves both step
    time and idle, which are the numbers that pass exists to measure.
    """
    start, end = _split_ints(sweep.args.profile_steps)
    command = sweep.kit(
        "torchrun", "-n", str(len(sweep.env["nodes"])),
        "scripts/train_lm.py", str(sweep.args.config.relative_to(REPO_ROOT)),
        f"--model.num_hidden_layers={sweep.args.layers}",
        f"--dataset.data_config.seq_length={sweep.args.seq_len}",
        f"--activation_checkpoint.mode={sweep.args.activation_checkpoint}",
        f"--training.global_batch_size={point.gbs}",
        f"--training.micro_batch_size={point.mbs}",
        f"--accelerator.tp_size={point.tp}",
        f"--accelerator.cp_size={point.cp}",
        f"--accelerator.pp_size={point.pp}",
        f"--accelerator.ep_size={point.ep}",
        f"--fsdp_config.dp_shard_size={point.op}",
        # Expert weights are sharded by EP over the whole device mesh; what is
        # left is sharded over the remaining ranks, so this is world/ep.
        f"--fsdp_config.edp_shard_size={max(1, sweep.world // point.ep)}",
        "--profiling.enabled=true",
        f"--profiling.profile_memory={'true' if memory else 'false'}",
        f"--profiling.start_step={start}",
        f"--profiling.end_step={end}",
        f"--profiling.trace_dir=./{REMOTE_PROFILES}/{_dir_name(point, memory)}",
    )
    # Re-profiling into a directory that already holds a run leaves both, and
    # the classifier then refuses rather than guess which one is fresh.
    _run(sweep.kit("exec", "--no-env",
                   f"rm -rf ./{REMOTE_PROFILES}/{_dir_name(point, memory)}"),
         check=False)
    found = RUN_ID_PATTERN.search(_run(command, capture=True))
    return found.group(1) if found else None


def _node_states(text: str) -> List[Sequence[Any]]:
    """Return ``(index, host, state)`` for every node in a status table."""
    return [(int(m.group(1)), m.group(2), m.group(3).strip())
            for m in NODE_STATE_PATTERN.finditer(text)]


def _failure_reason(sweep: Sweep, run_id: Optional[str],
                    indices: Sequence[int]) -> str:
    """Return the distinct exceptions the failed nodes logged.

    The status table carries each node's LAST log line, which after a crash is
    a stack frame or a shutdown warning rather than the cause. The line that
    says what happened is far above it, so find it here rather than leave the
    reader to open a log on a node they would have to work out first.
    """
    if not run_id:
        return ""
    lines: List[str] = []
    seen = set()
    for index in indices:
        if index >= len(sweep.env["nodes"]) or len(lines) >= 5:
            break
        log = f"{sweep.env['log_dir']}/{run_id}.node{index}.log"
        # Case sensitive on purpose: it keeps out CANN's "[ERROR]" banners and
        # "Inner Error!" lines, which repeat once per rank and say less than
        # the Python exception they accompany.
        found = subprocess.run(
            ["ssh", f"{sweep.env['ssh_user']}@{sweep.env['nodes'][index]}",
             f"grep -aE '(Error|Exception):' {shlex.quote(log)} | head -n 40"],
            check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        ).stdout or ""
        for line in found.splitlines():
            stripped = RANK_PREFIX_PATTERN.sub("", line.strip())
            if not stripped or stripped in seen:
                continue
            seen.add(stripped)
            lines.append(f"  node{index}: {stripped[:200]}")
            if len(lines) >= 5:
                break
    return ("\n" + "\n".join(lines)) if lines else ""


def _stop_run(sweep: Sweep, run_id: Optional[str]) -> None:
    """Stop what is left of a run so the next strategy gets the devices back."""
    if run_id:
        _run(sweep.kit("kill", run_id), check=False)


def wait_for(sweep: Sweep, run_id: Optional[str]) -> str:
    """Block until the run is over, and return the final status text.

    The kit detaches, so a launch returning says nothing about the job. Status
    is decided from the run's rc file before the pid, so a finished run cannot
    read as RUNNING again through pid reuse.

    One dead rank ends the job, but it does not end the other ranks: they wait
    in the collective it never joins until HCCL_EXEC_TIMEOUT, 180 s in the kit
    config here and half an hour by default. Blocking until every node stops
    RUNNING therefore costs that timeout for each failed strategy, and a sweep
    broken the same way at every point pays it at every point. So a DEAD node
    ends the wait at the next poll instead, and the survivors are killed rather
    than left to time out, which is also what frees the devices for the next
    strategy.
    """
    command = sweep.kit("status", *([run_id] if run_id else []))
    deadline = time.monotonic() + sweep.args.timeout
    while True:
        text = subprocess.run(
            command, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout or ""
        dead = [(index, state) for index, _host, state in _node_states(text)
                if state.startswith("DEAD")]
        if dead:
            named = ", ".join(f"node{index} {state}" for index, state in dead)
            reason = _failure_reason(sweep, run_id, [index for index, _ in dead])
            _stop_run(sweep, run_id)
            return f"{text}\nFAILED: {named}{reason}\n"
        if "RUNNING" not in text:
            return text
        if time.monotonic() > deadline:
            _stop_run(sweep, run_id)
            return f"{text}\nTIMED OUT after {sweep.args.timeout}s, run stopped\n"
        time.sleep(sweep.args.poll)


def harvest_peaks(sweep: Sweep, run_id: Optional[str]) -> Dict[str, float]:
    """Return the peak device memory the trainer logged, in GiB.

    Read from the training log rather than the profiler: the trainer reports it
    every step at no cost, so the timing pass yields memory without the
    allocator recording that would distort the very step it is timing.
    """
    if not run_id:
        return {}
    log = f"{sweep.env['log_dir']}/{run_id}.node0.log"
    text = subprocess.run(
        ["ssh", f"{sweep.env['ssh_user']}@{sweep.env['nodes'][0]}", f"cat {log}"],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ).stdout or ""
    found = PEAK_PATTERN.findall(text)
    if not found:
        return {}
    return {"max_allocated_gb": max(float(a) for a, _ in found),
            "max_reserved_gb": max(float(r) for _, r in found)}


def _duration(seconds: float) -> str:
    """Render a duration as 1h02m, 14m05s or 45s."""
    seconds = int(round(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@dataclass
class Progress:
    """How far a run stage has got, to head each launch with.

    ``started`` is when the stage began, so the estimate of what is left
    averages the launches made so far and leaves out the stages before them.
    """

    total: int
    started: float = field(default_factory=time.monotonic)
    done: int = 0

    def header(self, sweep_started: float) -> str:
        """Return ``run 3/12, 14m05s elapsed, about 28m00s left`` for the next launch."""
        now = time.monotonic()
        text = f"run {self.done + 1}/{self.total}, {_duration(now - sweep_started)} elapsed"
        if self.done:
            left = (now - self.started) / self.done * (self.total - self.done)
            text += f", about {_duration(left)} left"
        return text


def run_pass(sweep: Sweep, memory: bool, progress: Progress) -> Dict[str, Any]:
    """Run every strategy once, returning each one's status and peaks."""
    results: Dict[str, Any] = {}
    label = "memory" if memory else "timing"
    for point in sweep.points:
        print(f"\n===== {progress.header(sweep.started)}: {point.tag} ({label}) =====",
              flush=True)
        run_id = launch(sweep, point, memory=memory)
        status = wait_for(sweep, run_id)
        progress.done += 1
        print(status, flush=True)
        results[point.tag] = {
            "run_id": run_id, "status": status,
            "failed": "\nFAILED:" in status or "TIMED OUT" in status,
            **point.dims, **harvest_peaks(sweep, run_id)}
    failed = [tag for tag, data in results.items() if data.get("failed")]
    if failed:
        print(f"\n{len(failed)} of {len(sweep.points)} strategies failed the "
              f"{label} pass: {', '.join(failed)}", flush=True)
    return results


def write_peaks_csv(results: Dict[str, Any], path: Path) -> int:
    """Write the measured peak device memory of every strategy."""
    columns = ["DP", "MP", "PP", "CP", "EP", "MB", "MBS", "OP",
               "max_allocated_gb", "max_reserved_gb"]
    rows = [data for data in results.values()
            if isinstance(data, dict) and "max_allocated_gb" in data]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for data in rows:
            writer.writerow([data.get(name, "") for name in columns])
    return len(rows)


def stage_run(sweep: Sweep) -> None:
    """Run the timing pass, then the memory pass when one is asked for."""
    passes = 2 if sweep.args.profile_memory == "separate" else 1
    progress = Progress(total=passes * len(sweep.points))
    results = run_pass(sweep, memory=sweep.args.profile_memory == "same", progress=progress)
    count = write_peaks_csv(results, sweep.out / "memory.csv")
    print(f"\npeak memory for {count} strategy(ies) in "
          f"{sweep.out / 'memory.csv'}", flush=True)
    if sweep.args.profile_memory == "separate":
        results["memory_pass"] = run_pass(sweep, memory=True, progress=progress)
    (sweep.out / "run_states.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n{progress.done} run(s) in {_duration(time.monotonic() - progress.started)}",
          flush=True)


def stage_fetch(sweep: Sweep) -> None:
    """Copy the profiling output from the node that holds profiling.rank."""
    sweep.profiles.mkdir(parents=True, exist_ok=True)
    remote = f"{sweep.env['repo_dir']}/{REMOTE_PROFILES}"
    _run([
        "rsync", "-a",
        f"{sweep.env['ssh_user']}@{sweep.env['nodes'][0]}:{remote}/",
        f"{sweep.profiles}/",
    ])


def classify(sweep: Sweep, point: Point) -> Optional[Path]:
    """Split one profiled run into ND's parts, returning its CSV.

    The run directory is passed whole rather than a chosen ``*_ascend_pt``
    inside it, so ``trace_classify`` locates the run and refuses a directory
    holding two. Reading either of two silently makes a stale profile look like
    a fresh one, with numbers that are identical for no visible reason.
    """
    config_dir = sweep.profiles / point.tag
    runs = sorted(config_dir.glob("*_ascend_pt"))
    if not runs:
        print(f"{point.tag}: no profiling output, skipped", flush=True)
        return None
    if len(runs) > 1:
        listed = "".join(f"\n  {run.name}" for run in runs)
        print(f"{point.tag}: {len(runs)} profiling runs, refusing to guess:{listed}",
              flush=True)
        return None
    part = sweep.out / f"real_{point.tag}.csv"
    command = [
        sweep.python, "-m", "hyper_parallel.auto_parallel.sapp_nd.nd.trace_classify",
        str(config_dir),
        "--dims", ",".join(f"{name}={value}" for name, value in point.dims.items()),
        "--csv", str(part), "--detail", str(sweep.out / f"detail_{point.tag}.csv"),
    ]
    if subprocess.run(command, check=False, cwd=REPO_ROOT).returncode:
        print(f"{point.tag}: classification failed, skipped", flush=True)
        return None
    return part


def merge_csv(parts: Sequence[Path], merged: Path) -> int:
    """Concatenate single-row classified CSVs, keeping one header."""
    rows: List[List[str]] = []
    header: Optional[List[str]] = None
    for part in parts:
        with open(part, newline="", encoding="utf-8") as handle:
            table = list(csv.reader(handle))
        if len(table) < 2:
            continue
        if header is None:
            header = table[0]
        elif table[0] != header:
            raise SystemExit(f"{part}: header differs from the first CSV")
        rows += table[1:]
    if header is None:
        return 0
    with open(merged, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return len(rows)


def require_importable(sweep: Sweep) -> None:
    """Fail once, early, when the analysis interpreter cannot import the package.

    The sweep itself needs nothing but the standard library, so it runs happily
    under whichever python is on PATH; the classifier and the cost model import
    hyper_parallel and therefore torch. Checking here turns one unusable
    interpreter into a single instruction rather than a traceback per strategy.
    """
    probe = subprocess.run(
        [sweep.python, "-c", "import hyper_parallel"],
        check=False, cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    if probe.returncode:
        last = (probe.stderr or "").strip().splitlines()[-1:]
        raise SystemExit(
            f"{sweep.python} cannot import hyper_parallel: "
            f"{last[0] if last else 'unknown error'}. "
            "Point --python at the interpreter the training runs use, "
            "for example --python /home/tt/envs/hp2/bin/python3.11")


def stage_classify(sweep: Sweep) -> None:
    """Classify every profiled strategy into one combined CSV."""
    require_importable(sweep)
    parts = [part for part in (classify(sweep, p) for p in sweep.points) if part]
    count = merge_csv(parts, sweep.merged_csv)
    print(f"\n{count} configuration(s) in {sweep.merged_csv}", flush=True)


def write_nd_config(sweep: Sweep, nd_yaml: Path) -> None:
    """Write ND's input: the demo config with the shape the sweep runs it at.

    The launch overrides the config's layer count, sequence length, recompute
    mode and batch on the command line, so the config file alone describes a
    different run: 128 tokens without recompute, against a default sweep of
    8192 with full recompute. Those are set here from the same arguments the
    launch uses. The degrees are left as the config states them, since ND
    takes them from the measured CSV or searches them, and the world size is
    stated so that ND derives the data-parallel width the trainer does.
    """
    import yaml  # pylint: disable=import-outside-toplevel

    raw = yaml.safe_load(sweep.args.config.read_text(encoding="utf-8"))
    raw["model"] = dict(raw["model"], num_hidden_layers=sweep.args.layers)
    raw["model"].pop("validate_placement", None)
    dataset = raw.setdefault("dataset", {})
    dataset["data_config"] = dict(dataset.get("data_config") or {},
                                  seq_length=sweep.args.seq_len)
    raw["activation_checkpoint"] = dict(raw.get("activation_checkpoint") or {},
                                        mode=sweep.args.activation_checkpoint)
    raw["training"] = dict(raw.get("training") or {}, global_batch_size=sweep.gbs,
                           micro_batch_size=sweep.args.micro_batch_size)
    raw["context"] = dict(raw.get("context") or {}, device_num=sweep.world)
    nd_yaml.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _shape_file(sweep: Sweep) -> Path:
    """Where the shape ND's ranking was made for is recorded."""
    return sweep.ranking_csv.with_suffix(".shape.json")


def stage_rank(sweep: Sweep) -> None:
    """Rank every configuration ND keeps at the sweep's shape, best first.

    The search covers ND's whole space, TP and PP included, rather than only
    what this model can run: a strategy ND prefers that the trainer cannot run
    is worth seeing, so the choice among the runnable ones is left to
    ``pick_nd_top``. Any ranking already in the directory is removed first, so
    a search that fails cannot leave an older one looking current.
    """
    require_importable(sweep)
    nd_yaml = sweep.out / "nd_model.yaml"
    write_nd_config(sweep, nd_yaml)
    for stale in (sweep.ranking_csv, _shape_file(sweep)):
        stale.unlink(missing_ok=True)
    _run([
        sweep.python, "-m", RUN_ND,
        "-y", str(nd_yaml), "-f", sweep.args.framework,
        "-d", str(sweep.world), "-A", sweep.args.arch, "-b", str(sweep.gbs),
        "-t", str(max(20, 2 * sweep.args.nd_top)),
        "--ranking_csv", str(sweep.ranking_csv), "-o", str(sweep.out / "nd_rank"),
    ])
    _shape_file(sweep).write_text(json.dumps(sweep.shape, indent=2), encoding="utf-8")


def load_ranking(sweep: Sweep) -> Tuple[List[Dict[str, str]], str]:
    """Return ND's ranking for this sweep's shape, or no rows and why not."""
    if not sweep.ranking_csv.is_file() or not _shape_file(sweep).is_file():
        return [], f"no ND ranking in {sweep.out}: run the rank stage"
    stored = json.loads(_shape_file(sweep).read_text(encoding="utf-8"))
    differ = [f"{name} {stored.get(name)} there, {value} here"
              for name, value in sweep.shape.items() if stored.get(name) != value]
    if differ:
        return [], (f"{sweep.ranking_csv} ranks another shape ({'; '.join(differ)}): "
                    "run the rank stage again")
    return _read_rows(sweep.ranking_csv), ""


def _print_picks(picks: Sequence[Pick], passed: Sequence[Tuple[int, Point, str]],
                 wanted: int, kept: int) -> None:
    """Say which of ND's strategies the sweep runs, and which it cannot."""
    print(f"\nND's {len(picks)} best strategies this model can run, "
          f"of the {kept} configurations its search keeps:", flush=True)
    for pick in picks:
        ties = (f"   tied with OP {', '.join(map(str, pick.tied_ops))}"
                if pick.tied_ops else "")
        print(f"  #{pick.rank:<5d} {pick.point.tag:28s} score {pick.score:.4e}  "
              f"{pick.memory_mb / 1024:6.1f} GiB{ties}", flush=True)
    if len(picks) < wanted:
        print(f"  asked for {wanted}: ND keeps no other strategy this model can run",
              flush=True)
    if passed:
        print(f"ND ranks {len(passed)} strategy(ies) above its last pick that "
              "cannot run here:", flush=True)
        by_reason: Dict[str, List[Tuple[int, Point, str]]] = {}
        for entry in passed:
            by_reason.setdefault(entry[2], []).append(entry)
        for reason, entries in by_reason.items():
            rank, point, _ = entries[0]
            print(f"  {len(entries)}, the best #{rank} {point.tag}: {reason}", flush=True)


def choose_points(sweep: Sweep) -> List[Point]:
    """Return the strategies to run: ND's best, the named grid, or both."""
    points: List[Point] = []
    if sweep.args.nd_top:
        rows, why = load_ranking(sweep)
        if not rows:
            raise SystemExit(why)
        picks, passed = pick_nd_top(rows, sweep.args, sweep.world, sweep.args.nd_top)
        _print_picks(picks, passed, sweep.args.nd_top, len(rows))
        points = [pick.point for pick in picks]
    if not sweep.args.nd_top or named_axes(sweep.args):
        points += [point for point in expand(sweep.args, sweep.world) if point not in points]
    return points


def _strategy_key(row: Dict[str, str]) -> Tuple[int, ...]:
    """A strategy's degrees, read alike from ND's ranking and a measured CSV."""
    return tuple(int(row.get(name) or 1) for name in STRATEGY_DIMS)


def _tag_of(key: Tuple[int, ...]) -> str:
    """The directory name of the strategy a key stands for."""
    degree = dict(zip(STRATEGY_DIMS, key))
    tag = (f"ep{degree['EP']}_cp{degree['CP']}_op{degree['OP']}"
           f"_tp{degree['MP']}_pp{degree['PP']}")
    return f"{tag}_mbs{degree['MBS']}" if degree["MBS"] > 1 else tag


def _ranks(values: Sequence[float]) -> List[float]:
    """1-based ranks of *values*, a tie sharing the mean of its ranks."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def spearman(first: Sequence[float], second: Sequence[float]) -> Optional[float]:
    """Rank correlation of two series, None when it is undefined."""
    if len(first) < 3:
        return None
    ranks_a, ranks_b = _ranks(first), _ranks(second)
    mean = (len(first) + 1) / 2
    covariance = sum((a - mean) * (b - mean) for a, b in zip(ranks_a, ranks_b))
    spread = (sum((a - mean) ** 2 for a in ranks_a) * sum((b - mean) ** 2 for b in ranks_b)) ** 0.5
    return covariance / spread if spread else None


def _nd_estimates(sweep: Sweep) -> Dict[Tuple[int, ...], Dict[str, str]]:
    """ND's estimate of each measured strategy, as compare wrote it, by strategy."""
    return {_strategy_key(row): row for row in _read_rows(sweep.estimates_csv)}


def _ranking_table(sweep: Sweep, ranking: Sequence[Dict[str, str]],
                   measured: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """One row per measured strategy with ND's rank, score and memory, in ND's order.

    ND's memory comes from compare's estimates, which cover every measured
    strategy, and from the ranking only for a strategy they lack.
    """
    nd_of: Dict[Tuple[int, ...], Dict[str, str]] = {}
    for row in ranking:
        nd_of.setdefault(_strategy_key(row), row)
    estimates = _nd_estimates(sweep)
    peaks = {_strategy_key(row): row for row in _read_rows(sweep.out / "memory.csv")}
    times = [float(row["time"]) for row in measured]
    table = []
    for row, step, place in zip(measured, times, _ranks(times)):
        key = _strategy_key(row)
        nd_row = nd_of.get(key, {})
        memory_row = estimates.get(key, nd_row)
        table.append({
            "strategy": _tag_of(key), "nd_rank": nd_row.get("rank", ""),
            "nd_score": nd_row.get("score", ""),
            "nd_memory_gib": (f"{float(memory_row['memory_mb']) / 1024:.1f}"
                              if memory_row else ""),
            "measured_ms": f"{step:.1f}", "measured_rank": f"{place:g}",
            "peak_allocated_gib": peaks.get(key, {}).get("max_allocated_gb", ""),
        })
    table.sort(key=lambda entry: int(entry["nd_rank"] or 10 ** 9))
    return table


def _print_verdict(table: Sequence[Dict[str, str]]) -> None:
    """Say what following ND would cost, and how well its order holds."""
    ranked = [entry for entry in table if entry["nd_rank"]]
    if not ranked:
        return
    pick, fastest = ranked[0], min(table, key=lambda entry: float(entry["measured_ms"]))
    cost = float(pick["measured_ms"]) / float(fastest["measured_ms"]) - 1
    print(f"Of these, ND ranks {pick['strategy']} best (its #{pick['nd_rank']}): it "
          f"measures {pick['measured_ms']} ms, {pick['measured_rank']} of {len(table)}. "
          f"The fastest is {fastest['strategy']} at {fastest['measured_ms']} ms, so "
          f"following ND costs {cost:.1%}.", flush=True)
    correlation = spearman([float(entry["nd_score"]) for entry in ranked],
                           [float(entry["measured_ms"]) for entry in ranked])
    if correlation is not None:
        print(f"Rank correlation of ND's score with the measured step over the "
              f"{len(ranked)} it ranks: {correlation:+.2f} (1 is ND's order exactly).",
              flush=True)


def report_ranking(sweep: Sweep, ranking: Sequence[Dict[str, str]]) -> None:
    """Set ND's rank of every measured strategy beside the measured order.

    This is the comparison a top-k sweep exists for: whether the strategy ND
    ranks first is the one that runs fastest, and how much following ND costs
    when it is not. A strategy outside ND's ranking, one its search does not
    generate or one it believes does not fit, has no rank; ``compare`` above
    still prints its estimate.
    """
    measured = _read_rows(sweep.merged_csv)
    if not measured:
        return
    table = _ranking_table(sweep, ranking, measured)
    path = sweep.out / "nd_vs_measured.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"\nND against the measurement, in ND's order of its {len(ranking)} "
          "configurations; '-' is a strategy its search does not generate or "
          "believes does not fit:", flush=True)
    print(f"  {'strategy':28s} {'ND rank':>7s} {'ND score':>10s} {'ND GiB':>7s} "
          f"{'step ms':>9s} {'measured':>8s} {'peak GiB':>8s}", flush=True)
    for entry in table:
        score = f"{float(entry['nd_score']):.3e}" if entry["nd_score"] else "-"
        print(f"  {entry['strategy']:28s} {entry['nd_rank'] or '-':>7s} {score:>10s} "
              f"{entry['nd_memory_gib'] or '-':>7s} {entry['measured_ms']:>9s} "
              f"{entry['measured_rank']:>8s} {entry['peak_allocated_gib'] or '-':>8s}",
              flush=True)
    _print_verdict(table)
    print(f"written to {path}", flush=True)


def stage_compare(sweep: Sweep) -> None:
    """Print ND's estimate beside every measured strategy, and ND's rank of each.

    ``--framework`` must select the AutoModels-aware parser: run_nd defaults to
    ``mindformers``, which reads a different schema, and the deprecated
    ``hyperparallel`` wants a TorchTitan TOML plus a source path.
    """
    require_importable(sweep)
    if not sweep.merged_csv.is_file():
        raise SystemExit(f"nothing to compare: {sweep.merged_csv} does not exist")
    nd_yaml = sweep.out / "nd_model.yaml"
    write_nd_config(sweep, nd_yaml)
    _run([
        sweep.python, "-m", RUN_ND,
        "-y", str(nd_yaml), "-f", sweep.args.framework,
        "-d", str(sweep.world), "-A", sweep.args.arch,
        "--real_csv", str(sweep.merged_csv), "-o", str(sweep.nd_dir),
    ], check=False)
    ranking, why = load_ranking(sweep)
    if ranking:
        report_ranking(sweep, ranking)
    elif sweep.args.nd_top:
        print(why, flush=True)


def _add_strategy_args(parser: argparse.ArgumentParser) -> None:
    """Add the swept parallel degrees and the batch shape."""
    parser.add_argument("--nd-top", type=int, default=0,
                        help="run the N strategies ND ranks best at this shape, of "
                             "those this model and trainer can run; a named axis "
                             "adds its grid beside them")
    parser.add_argument("--ep", default=None,
                        help="expert-parallel degrees; every axis defaults to 1, "
                             "so a sweep varies only what it names")
    parser.add_argument("--cp", default=None, help="context-parallel degrees")
    parser.add_argument("--op", default="",
                        help="FSDP shard widths (fsdp_config.dp_shard_size, ND's "
                             "OP); default is dp*cp, the whole shardable width")
    parser.add_argument("--tp", default=None, help="tensor-parallel degrees")
    parser.add_argument("--pp", default=None, help="pipeline-parallel degrees")
    parser.add_argument("--global-batch-size", type=int, default=0,
                        help="default is the world size, which holds the work "
                             "per step fixed so strategies stay comparable")
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--layers", type=int, default=8,
                        help="decoder layers kept by the crop; a multiple of 4 "
                             "preserves the 3 linear to 1 full attention ratio. "
                             "Communication volume and compute are both linear "
                             "in this, so a crop rescales the step rather than "
                             "changing what the comparison tests")
    parser.add_argument("--seq-len", type=int, default=8192,
                        help="training sequence length; the dataset is rebuilt "
                             "to match, since its documents are exactly this long")
    parser.add_argument("--activation-checkpoint", default="full",
                        choices=("off", "full", "selective"),
                        help="recompute mode; a real run at this size needs full")
    parser.add_argument("--samples", type=int, default=0,
                        help="documents to generate; default covers the longest "
                             "configuration's global batch times train_iters, doubled")
    parser.add_argument("--num-experts", type=int, default=256,
                        help="routed experts, used to reject an ep that cannot divide them")


def _read_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into dicts, empty when the file is absent."""
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _strategy_label(row: Dict[str, str], varying: Sequence[str]) -> str:
    """Name a row by the dimensions that actually change across the sweep."""
    return " ".join(f"{name}{row[name]}" for name in varying) or "single"


def _varying_dims(rows: Sequence[Dict[str, str]]) -> List[str]:
    """Return the dimension columns that take more than one value."""
    names = [n for n in ("EP", "CP", "OP", "DP", "MB", "MBS", "MP", "PP") if n in rows[0]]
    varying = [n for n in names if len({row[n] for row in rows}) > 1]
    return varying or names[:1]


def _draw_time(axis: Any, timing: Sequence[Dict[str, str]]) -> None:
    """Stack each strategy's measured step, split into ND's parts and idle."""
    varying = _varying_dims(timing)
    labels = [_strategy_label(row, varying) for row in timing]
    parts = [c for c in timing[0] if c == "comp" or c.endswith("_wait")]
    parts = [c for c in parts if any(float(row[c]) for row in timing)]
    bottom = [0.0] * len(timing)
    for part in parts:
        values = [float(row[part]) for row in timing]
        axis.bar(labels, values, bottom=bottom, label=part)
        bottom = [b + v for b, v in zip(bottom, values)]
    idle = [float(row["time"]) - b for row, b in zip(timing, bottom)]
    axis.bar(labels, idle, bottom=bottom, label="idle")
    axis.set_ylabel("step (ms)")
    axis.set_title("Measured step, split into ND's parts")
    axis.legend(fontsize="small", ncol=2)
    axis.tick_params(axis="x", rotation=45)


def _draw_memory(axis: Any, memory: Sequence[Dict[str, str]],
                 estimates: Dict[Tuple[int, ...], Dict[str, str]]) -> int:
    """Draw each strategy's peak device memory, and ND's estimate where it has one.

    The trainer logs the maximum over ranks, in GiB; ND models one rank and
    reports MiB, converted here, and its peak includes a 1 GiB safety margin.

    Returns:
        How many strategies have an ND estimate.
    """
    varying = _varying_dims(memory)
    labels = [_strategy_label(row, varying) for row in memory]
    for column, style in (("max_allocated_gb", "o-"), ("max_reserved_gb", "s--")):
        axis.plot(labels, [float(row[column]) for row in memory], style, label=column)
    nd_rows = [estimates.get(_strategy_key(row)) for row in memory]
    drawn = sum(1 for row in nd_rows if row)
    if drawn:
        axis.plot(labels, [float(row["memory_mb"]) / 1024 if row else float("nan")
                           for row in nd_rows], "^:", label="ND estimate")
    axis.set_ylabel("peak per device (GiB)")
    axis.set_title("Peak device memory, measured and ND's" if drawn else "Peak device memory")
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    axis.tick_params(axis="x", rotation=45)
    return drawn


def _save(figure: Any, out: Path) -> None:
    """Write a figure as PDF and PNG."""
    figure.tight_layout()
    figure.savefig(out, bbox_inches="tight")
    figure.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")


def stage_plot(sweep: Sweep) -> None:
    """Draw the sweep: where each step goes, and what it costs in memory.

    ND already plots its estimate against each configuration, with and without
    idle; this is the view across the sweep, which no single configuration
    shows. Memory also gets a figure of its own, the measured peak against
    ND's estimate, which compare writes to ``nd/<csv stem>_estimates.csv``.
    """
    try:
        import matplotlib  # pylint: disable=import-outside-toplevel
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
    except ImportError:
        print("matplotlib not available to this interpreter; skipping the plot. "
              "Run with --python pointing at the training environment.", flush=True)
        return

    timing = _read_rows(sweep.merged_csv)
    memory = _read_rows(sweep.out / "memory.csv")
    if not timing and not memory:
        print("nothing to plot: no classified CSV and no memory.csv", flush=True)
        return
    estimates = _nd_estimates(sweep)

    panels = [name for name, rows in (("time", timing), ("memory", memory)) if rows]
    figure, axes = plt.subplots(len(panels), 1, figsize=(2 + 1.4 * max(
        len(timing), len(memory)), 4 * len(panels)), squeeze=False)
    if timing:
        _draw_time(axes[panels.index("time")][0], timing)
    if memory:
        _draw_memory(axes[panels.index("memory")][0], memory, estimates)
    _save(figure, sweep.out / "sweep.pdf")
    plt.close(figure)
    print(f"sweep plot: {sweep.out / 'sweep.pdf'} (and .png)", flush=True)

    if memory:
        figure, axis = plt.subplots(figsize=(2 + 1.4 * len(memory), 4))
        drawn = _draw_memory(axis, memory, estimates)
        _save(figure, sweep.out / "memory.pdf")
        plt.close(figure)
        missing = ("" if drawn == len(memory) else
                   f"; ND's estimate for {drawn} of {len(memory)}: run compare "
                   f"first, it writes {sweep.estimates_csv.name}")
        print(f"memory plot: {sweep.out / 'memory.pdf'} (and .png){missing}", flush=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the sweep arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cluster-env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "sweep")
    parser.add_argument("--cluster", default="cluster",
                        help="the kit entry point (path or name on PATH)")
    parser.add_argument("--arch", default="A3", help="hardware name passed to ND")
    parser.add_argument("--python", default="",
                        help="interpreter for the classify and compare steps; it "
                             "must import hyper_parallel, so it is the training "
                             "environment's python, not necessarily this one")
    parser.add_argument("--framework", default="hyper_v2",
                        help="run_nd parser; hyper_v2 reads the AutoModels schema")
    parser.add_argument("--profile-memory", choices=("none", "separate", "same"),
                        default="separate",
                        help="'separate' repeats the sweep with the allocator "
                             "history on, keeping it out of the timed run; "
                             "'same' records it in the timed run, which moves "
                             "step time and idle")
    parser.add_argument("--profile-steps", default="3,5",
                        help="start,end of the profiling window (end exclusive)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="seconds to wait for one configuration")
    parser.add_argument("--poll", type=int, default=20)
    parser.add_argument("--only", choices=STAGES, action="append",
                        help="run only these stages (repeatable)")
    _add_strategy_args(parser)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Sweep the requested strategies and score ND against the result."""
    args = parse_args(argv)
    sweep = Sweep(args=args, env=read_cluster_env(args.cluster_env))
    sweep.out.mkdir(parents=True, exist_ok=True)
    stages = set(args.only or STAGES)
    # A grid sweep ranks only when the stage is named: the search needs the
    # analysis interpreter, which a sweep that only runs and fetches does not.
    if "rank" in stages and (args.nd_top or args.only):
        stage_rank(sweep)
    sweep.points = choose_points(sweep)
    print(f"world={sweep.world} nodes={len(sweep.env['nodes'])}"
          f"x{sweep.env['nproc']}  {len(sweep.points)} strategy(ies)", flush=True)
    for point in sweep.points:
        print(f"  {point.tag}  dims={point.dims}", flush=True)

    for name, run_stage in (
            ("mirror", mirror_code),
            ("data", stage_data),
            ("run", stage_run),
            ("fetch", stage_fetch),
            ("classify", stage_classify),
            ("compare", stage_compare),
            ("plot", stage_plot),
    ):
        if name in stages:
            run_stage(sweep)
    print(f"\nsweep done in {_duration(time.monotonic() - sweep.started)}", flush=True)


if __name__ == "__main__":
    main()
