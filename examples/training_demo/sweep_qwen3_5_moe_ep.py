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
"""Profile Qwen3.5-MoE across expert-parallel degrees and score ND against it.

Runs the cropped demo once per ``ep_size`` under the profiler, collects the
Ascend profiling output, classifies each run into ND's parts with
``nd.trace_classify``, and feeds the combined CSV to ``run_nd --real_csv`` so
ND's estimate is printed beside the measurement for every configuration.

Run it on the cluster's control node:

    python examples/training_demo/sweep_qwen3_5_moe_ep.py --ep 1,2,4,8,16,32,64

Stages run in order and each can be run alone with ``--only``, so a failed
sweep can be classified without re-running, and a changed cost model can be
re-scored without re-profiling.
"""

from __future__ import annotations

import argparse
import csv
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
REMOTE_PROFILES = "output/sweep_ep_profiles"
STAGES = ("mirror", "run", "fetch", "classify", "compare")


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
    """Return NODES, NPROC_PER_NODE, REPO_DIR and SSH_USER from a kit config.

    The file is bash, so bash sources it rather than this parsing it: the node
    list is an array and REPO_DIR may reference other variables.
    """
    script = (
        f'set -euo pipefail; source {shlex.quote(str(env_path))}; '
        'printf "%s\\n" "${NODES[*]}" "${NPROC_PER_NODE:-8}" '
        '"${REPO_DIR}" "${SSH_USER:-root}"'
    )
    out = subprocess.run(
        ["bash", "-c", script], check=True, text=True, stdout=subprocess.PIPE
    ).stdout.splitlines()
    return {
        "nodes": out[0].split(),
        "nproc": int(out[1]),
        "repo_dir": out[2].rstrip("/"),
        "ssh_user": out[3],
        "log_dir": out[4].rstrip("/"),
    }


@dataclass
class Sweep:
    """One sweep's resolved settings, shared by every stage."""

    args: argparse.Namespace
    env: Dict[str, Any]
    eps: List[int] = field(default_factory=list)

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


def _tag(ep: int, memory: bool) -> str:
    """Directory name for one configuration's timing or memory pass."""
    return f"ep{ep}_mem" if memory else f"ep{ep}"


PEAK_PATTERN = re.compile(
    r"memory/device_max_allocated_gb=([0-9.]+).*?"
    r"memory/device_max_reserved_gb=([0-9.]+)"
)


def harvest_peaks(sweep: Sweep, run_id: Optional[str]) -> Dict[str, float]:
    """Return the peak device memory the trainer logged, in GiB.

    Read from the training log rather than the profiler: the trainer reports
    it every step at no cost, so the timing pass yields memory without the
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


def launch(sweep: Sweep, ep: int, memory: bool = False) -> Optional[str]:
    """Launch one configuration and return the kit's run id.

    ``memory`` adds the allocator history the memory pass needs. It is off for
    the timing pass: recording brackets the profiled window and moves both step
    time and idle, which are the numbers that pass exists to measure.
    """
    start, end = (int(part) for part in sweep.args.profile_steps.split(","))
    world = sweep.world
    command = sweep.kit(
        "torchrun", "-n", str(len(sweep.env["nodes"])),
        "scripts/train_lm.py", str(sweep.args.config.relative_to(REPO_ROOT)),
        f"--fsdp_config.dp_shard_size={world}",
        f"--training.global_batch_size={world}",
        f"--accelerator.ep_size={ep}",
        # Expert weights are sharded by EP first; what is left is sharded over
        # the remaining ranks, so this is world/ep and never anything else.
        f"--fsdp_config.edp_shard_size={max(1, world // ep)}",
        "--profiling.enabled=true",
        f"--profiling.profile_memory={'true' if memory else 'false'}",
        f"--profiling.start_step={start}",
        f"--profiling.end_step={end}",
        f"--profiling.trace_dir=./{REMOTE_PROFILES}/{_tag(ep, memory)}",
    )
    # Re-profiling into a directory that already holds a run leaves both, and
    # the classifier then refuses rather than guess which one is fresh.
    _run(sweep.kit("exec", "--no-env",
                   f"rm -rf ./{REMOTE_PROFILES}/{_tag(ep, memory)}"), check=False)
    found = RUN_ID_PATTERN.search(_run(command, capture=True))
    return found.group(1) if found else None


def wait_for(sweep: Sweep, run_id: Optional[str]) -> str:
    """Block until no node reports RUNNING, and return the final status text.

    The kit detaches, so a launch returning says nothing about the job. Status
    is decided from the run's rc file before the pid, so a finished run cannot
    read as RUNNING again through pid reuse.
    """
    command = sweep.kit("status", *( [run_id] if run_id else [] ))
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


def run_pass(sweep: Sweep, memory: bool) -> Dict[str, Any]:
    """Run every configuration once, returning each one's status and peaks."""
    results: Dict[str, Any] = {}
    label = "memory" if memory else "timing"
    for ep in sweep.eps:
        print(f"\n===== ep_size {ep} ({label}) =====", flush=True)
        run_id = launch(sweep, ep, memory=memory)
        status = wait_for(sweep, run_id)
        print(status, flush=True)
        results[str(ep)] = {"run_id": run_id, "status": status,
                            **harvest_peaks(sweep, run_id)}
    return results


def write_peaks_csv(results: Dict[str, Any], path: Path) -> int:
    """Write the measured peak device memory of every configuration."""
    rows = [(ep, data) for ep, data in results.items()
            if isinstance(data, dict) and "max_allocated_gb" in data]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["EP", "max_allocated_gb", "max_reserved_gb"])
        for ep, data in rows:
            writer.writerow([ep, data["max_allocated_gb"], data["max_reserved_gb"]])
    return len(rows)


def stage_run(sweep: Sweep) -> None:
    """Run the timing pass, then the memory pass when one is asked for."""
    results = run_pass(sweep, memory=sweep.args.profile_memory == "same")
    count = write_peaks_csv(results, sweep.out / "memory.csv")
    print(f"\npeak memory for {count} configuration(s) in "
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


def classify(sweep: Sweep, ep: int) -> Optional[Path]:
    """Split one profiled run into ND's parts, returning its CSV.

    The run directory is passed whole rather than a chosen ``*_ascend_pt``
    inside it, so ``trace_classify`` locates the run and refuses a directory
    holding two. Reading either of two silently makes a stale profile look
    like a fresh one, with numbers that are identical for no visible reason.
    """
    config_dir = sweep.profiles / f"ep{ep}"
    runs = sorted(config_dir.glob("*_ascend_pt"))
    if not runs:
        print(f"ep{ep}: no profiling output, skipped", flush=True)
        return None
    if len(runs) > 1:
        listed = "".join(f"\n  {run.name}" for run in runs)
        print(f"ep{ep}: {len(runs)} profiling runs, refusing to guess:{listed}",
              flush=True)
        return None
    dims = {"DP": sweep.world, "MP": 1, "PP": 1, "CP": 1,
            "EP": ep, "MB": 1, "OP": sweep.world}
    part = sweep.out / f"real_ep{ep}.csv"
    command = [
        sys.executable, "-m", "hyper_parallel.auto_parallel.sapp_nd.nd.trace_classify",
        str(config_dir),
        "--dims", ",".join(f"{name}={value}" for name, value in dims.items()),
        "--csv", str(part), "--detail", str(sweep.out / f"detail_ep{ep}.csv"),
    ]
    if subprocess.run(command, check=False, cwd=REPO_ROOT).returncode:
        print(f"ep{ep}: classification failed, skipped", flush=True)
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


def stage_classify(sweep: Sweep) -> None:
    """Classify every profiled configuration into one combined CSV."""
    parts = [part for part in (classify(sweep, ep) for ep in sweep.eps) if part]
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
    """Print ND's estimate beside every measured configuration.

    ``--framework`` must select the AutoModels-aware parser: run_nd defaults to
    ``mindformers``, which reads a different schema, and the deprecated
    ``hyperparallel`` wants a TorchTitan TOML plus a source path.
    """
    if not sweep.merged_csv.is_file():
        raise SystemExit(f"nothing to compare: {sweep.merged_csv} does not exist")
    nd_yaml = sweep.out / "nd_model.yaml"
    write_nd_config(sweep.args.config, nd_yaml)
    _run([
        sys.executable, "-m", "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd",
        "-y", str(nd_yaml), "-f", sweep.args.framework,
        "-d", str(sweep.world), "-A", sweep.args.arch,
        "--real_csv", str(sweep.merged_csv), "-o", str(sweep.out / "nd"),
    ], check=False)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse the sweep arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cluster-env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--ep", default="1,2,4,8,16,32,64",
                        help="comma-separated expert-parallel degrees")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "output" / "sweep_ep")
    parser.add_argument("--cluster", default="cluster",
                        help="the kit entry point (path or name on PATH)")
    parser.add_argument("--arch", default="A3", help="hardware name passed to ND")
    parser.add_argument("--framework", default="hyper_v2",
                        help="run_nd parser; hyper_v2 reads the AutoModels schema")
    parser.add_argument("--num-experts", type=int, default=256,
                        help="routed experts, used to reject an ep that cannot divide them")
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Sweep the expert-parallel degrees and score ND against the result."""
    args = parse_args(argv)
    sweep = Sweep(args=args, env=read_cluster_env(args.cluster_env))
    sweep.eps = [int(part) for part in args.ep.split(",") if part.strip()]
    for ep in sweep.eps:
        if sweep.world % ep or args.num_experts % ep:
            raise SystemExit(
                f"ep_size {ep} must divide the world size ({sweep.world}) "
                f"and the routed expert count ({args.num_experts})"
            )
    sweep.out.mkdir(parents=True, exist_ok=True)
    print(f"world={sweep.world} nodes={len(sweep.env['nodes'])}"
          f"x{sweep.env['nproc']} ep={sweep.eps}", flush=True)

    stages = set(args.only or STAGES)
    for name, run_stage in (
            ("mirror", mirror_code),
            ("run", stage_run),
            ("fetch", stage_fetch),
            ("classify", stage_classify),
            ("compare", stage_compare),
    ):
        if name in stages:
            run_stage(sweep)


if __name__ == "__main__":
    main()
