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
from typing import Any, Dict, List, Optional, Sequence

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent
DEFAULT_ENV = DEMO_DIR / "cluster_qwen3_5_moe.env"
DEFAULT_CONFIG = DEMO_DIR / "train_qwen3_5_moe.yaml"
RUN_ID_PATTERN = re.compile(r"run id\s*:\s*(\S+)")
PEAK_PATTERN = re.compile(
    r"memory/device_max_allocated_gb=([0-9.]+).*?"
    r"memory/device_max_reserved_gb=([0-9.]+)"
)
REMOTE_PROFILES = "output/sweep_profiles"
STAGES = ("mirror", "data", "run", "fetch", "classify", "compare", "plot")


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
    which covers the data AND context axes, so it divides ``dp * cp``.
    """

    ep: int
    cp: int
    op: int
    tp: int
    pp: int
    dp: int
    mb: int
    gbs: int

    @property
    def tag(self) -> str:
        """Directory-safe name carrying every degree that can vary."""
        return f"ep{self.ep}_cp{self.cp}_op{self.op}_tp{self.tp}_pp{self.pp}"

    @property
    def dims(self) -> Dict[str, int]:
        """ND's dimension columns for this strategy."""
        return {"DP": self.dp, "MP": self.tp, "PP": self.pp, "CP": self.cp,
                "EP": self.ep, "MB": self.mb, "OP": self.op}


def _split_ints(text: str) -> List[int]:
    """Parse a comma-separated degree list."""
    return [int(part) for part in text.split(",") if part.strip()]


def _reject(point: Point, args: argparse.Namespace, world: int,
            fsdp_width: int) -> None:
    """Raise when a strategy the trainer or this model could not run is asked for."""
    if point.tp > 1:
        raise SystemExit(
            "tp_size > 1 shards the Gated DeltaNet conv1d while its groups and "
            "conv_dim stay global, so the forward raises on this model")
    if point.pp > 1:
        raise SystemExit(
            "pp_size > 1 neither raises nor pipelines: the Trainer has no "
            "pipeline schedule, so every stage group trains a full replica")
    if world % point.ep or args.num_experts % point.ep:
        raise SystemExit(
            f"ep {point.ep} must divide the world size ({world}) and the "
            f"routed expert count ({args.num_experts})")
    if fsdp_width % point.op:
        raise SystemExit(
            f"op {point.op} must divide dp*cp ({fsdp_width}); FSDP shards over "
            "the data and context axes together")
    per_step = args.micro_batch_size * point.dp
    if point.gbs % per_step:
        raise SystemExit(
            f"global_batch_size {point.gbs} must be a multiple of "
            f"micro_batch_size*dp ({per_step}) at cp={point.cp}")


def expand(args: argparse.Namespace, world: int) -> List[Point]:
    """Return the cartesian product of the requested degrees, validated.

    Each combination is checked against the constraints the trainer enforces at
    startup, so an impossible strategy is rejected here rather than after a
    launch and a wait.
    """
    points: List[Point] = []
    grid = itertools.product(_split_ints(args.ep), _split_ints(args.cp),
                             _split_ints(args.tp), _split_ints(args.pp))
    for ep, cp, tp, pp in grid:
        non_dp = tp * cp * pp
        if world % non_dp:
            raise SystemExit(
                f"tp*cp*pp ({non_dp}) must divide the world size ({world})")
        dp = world // non_dp
        fsdp_width = dp * cp           # FSDP shards over the data and context axes
        gbs = args.global_batch_size or world
        for op in (_split_ints(args.op) if args.op else [fsdp_width]):
            point = Point(ep=ep, cp=cp, op=op, tp=tp, pp=pp, dp=dp,
                          mb=gbs // (args.micro_batch_size * dp) or 1, gbs=gbs)
            _reject(point, args, world, fsdp_width)
            points.append(point)
    return points


@dataclass
class Sweep:
    """One sweep's resolved settings, shared by every stage."""

    args: argparse.Namespace
    env: Dict[str, Any]
    points: List[Point] = field(default_factory=list)

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
        f"--training.micro_batch_size={sweep.args.micro_batch_size}",
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


def wait_for(sweep: Sweep, run_id: Optional[str]) -> str:
    """Block until no node reports RUNNING, and return the final status text.

    The kit detaches, so a launch returning says nothing about the job. Status
    is decided from the run's rc file before the pid, so a finished run cannot
    read as RUNNING again through pid reuse.
    """
    command = sweep.kit("status", *([run_id] if run_id else []))
    deadline = time.monotonic() + sweep.args.timeout
    while True:
        text = subprocess.run(
            command, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout or ""
        if "RUNNING" not in text:
            return text
        if time.monotonic() > deadline:
            return f"{text}\nTIMED OUT after {sweep.args.timeout}s\n"
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


def run_pass(sweep: Sweep, memory: bool) -> Dict[str, Any]:
    """Run every strategy once, returning each one's status and peaks."""
    results: Dict[str, Any] = {}
    label = "memory" if memory else "timing"
    for point in sweep.points:
        print(f"\n===== {point.tag} ({label}) =====", flush=True)
        run_id = launch(sweep, point, memory=memory)
        status = wait_for(sweep, run_id)
        print(status, flush=True)
        results[point.tag] = {"run_id": run_id, "status": status,
                              **point.dims, **harvest_peaks(sweep, run_id)}
    return results


def write_peaks_csv(results: Dict[str, Any], path: Path) -> int:
    """Write the measured peak device memory of every strategy."""
    columns = ["DP", "MP", "PP", "CP", "EP", "MB", "OP",
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
    results = run_pass(sweep, memory=sweep.args.profile_memory == "same")
    count = write_peaks_csv(results, sweep.out / "memory.csv")
    print(f"\npeak memory for {count} strategy(ies) in "
          f"{sweep.out / 'memory.csv'}", flush=True)
    if sweep.args.profile_memory == "separate":
        results["memory_pass"] = run_pass(sweep, memory=True)
    (sweep.out / "run_states.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")


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


def write_nd_config(config: Path, nd_yaml: Path) -> None:
    """Write the ND input yaml for the model the demo config builds.

    The sequence length is carried across explicitly: an Indexed Dataset states
    it as ``dataset.data_config.seq_length``, and without it ND falls back to
    the model's context limit, 262144 here against a real 128. The attention
    term is quadratic, so that alone makes compute swamp every other part.
    """
    import yaml  # pylint: disable=import-outside-toplevel

    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    model = dict(raw["model"])
    model.pop("validate_placement", None)
    seq_len = raw["dataset"]["data_config"]["seq_length"]
    nd_yaml.write_text(
        yaml.safe_dump(
            {"model": model, "dataset": {"data_config": {"seq_length": seq_len}}},
            sort_keys=False),
        encoding="utf-8",
    )


def stage_compare(sweep: Sweep) -> None:
    """Print ND's estimate beside every measured strategy.

    ``--framework`` must select the AutoModels-aware parser: run_nd defaults to
    ``mindformers``, which reads a different schema, and the deprecated
    ``hyperparallel`` wants a TorchTitan TOML plus a source path.
    """
    require_importable(sweep)
    if not sweep.merged_csv.is_file():
        raise SystemExit(f"nothing to compare: {sweep.merged_csv} does not exist")
    nd_yaml = sweep.out / "nd_model.yaml"
    write_nd_config(sweep.args.config, nd_yaml)
    _run([
        sweep.python, "-m", "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd",
        "-y", str(nd_yaml), "-f", sweep.args.framework,
        "-d", str(sweep.world), "-A", sweep.args.arch,
        "--real_csv", str(sweep.merged_csv), "-o", str(sweep.out / "nd"),
    ], check=False)


def _add_strategy_args(parser: argparse.ArgumentParser) -> None:
    """Add the swept parallel degrees and the batch shape."""
    parser.add_argument("--ep", default="1",
                        help="expert-parallel degrees; every axis defaults to 1, "
                             "so a sweep varies only what it names")
    parser.add_argument("--cp", default="1", help="context-parallel degrees")
    parser.add_argument("--op", default="",
                        help="FSDP shard widths (fsdp_config.dp_shard_size, ND's "
                             "OP); default is dp*cp, the whole shardable width")
    parser.add_argument("--tp", default="1", help="tensor-parallel degrees")
    parser.add_argument("--pp", default="1", help="pipeline-parallel degrees")
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
    names = [n for n in ("EP", "CP", "OP", "DP", "MB", "MP", "PP") if n in rows[0]]
    varying = [n for n in names if len({row[n] for row in rows}) > 1]
    return varying or names[:1]


def stage_plot(sweep: Sweep) -> None:
    """Draw the sweep: where each step goes, and what it costs in memory.

    ND already plots its estimate against each configuration; this is the view
    across the sweep, which no single configuration shows.
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

    panels = [name for name, rows in (("time", timing), ("memory", memory)) if rows]
    figure, axes = plt.subplots(len(panels), 1, figsize=(2 + 1.4 * max(
        len(timing), len(memory)), 4 * len(panels)), squeeze=False)

    if timing:
        axis = axes[panels.index("time")][0]
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

    if memory:
        axis = axes[panels.index("memory")][0]
        varying = _varying_dims(memory)
        labels = [_strategy_label(row, varying) for row in memory]
        for column, style in (("max_allocated_gb", "o-"), ("max_reserved_gb", "s--")):
            axis.plot(labels, [float(row[column]) for row in memory], style,
                      label=column)
        axis.set_ylabel("peak per device (GiB)")
        axis.set_title("Peak device memory")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
        axis.tick_params(axis="x", rotation=45)

    figure.tight_layout()
    out = sweep.out / "sweep.pdf"
    figure.savefig(out, bbox_inches="tight")
    figure.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"sweep plot: {out} (and .png)", flush=True)


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
    sweep.points = expand(args, sweep.world)
    sweep.out.mkdir(parents=True, exist_ok=True)
    print(f"world={sweep.world} nodes={len(sweep.env['nodes'])}"
          f"x{sweep.env['nproc']}  {len(sweep.points)} strategy(ies)", flush=True)
    for point in sweep.points:
        print(f"  {point.tag}  dims={point.dims}", flush=True)

    stages = set(args.only or STAGES)
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


if __name__ == "__main__":
    main()
