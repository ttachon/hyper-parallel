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
"""Run the offload tests from the cluster's control node, one after another, and read them back.

Two scripts beside this one do the measuring:

- ``offload_probe.py`` trains the Demo 2 crop (Qwen3.5-MoE, 8 layers, 8192
  tokens, 64 dies on four nodes) and copies to and from the host inside the
  step. Here it runs under full recompute at EP 1, 2, 4, 8 and 16: EP 1 runs no
  all_to_all, so it gives what the step's own work takes from the copies, and
  the others give what the all_to_all adds as it grows. Once more at EP 2 with
  8 MiB copies instead of 64, to tell whether the step's host syncs wait behind
  a copy.
- ``bench_host_link.py`` runs on the first node alone: the all_to_all over
  groups of 2, 4, 8 and 16 ranks, plain and as the trainer runs it, beside
  copies; the trainer's exchange beside 64 MiB and 8 MiB copies; and the copy
  speed with 16, 8, 4, 2 and 1 dies of the node copying.

Usage, from the repository root on the control node::

    python examples/training_demo/offload_tests.py list           # the tests and their commands
    python examples/training_demo/offload_tests.py check          # sync, then the scripts' md5 on every node
    python examples/training_demo/offload_tests.py run all        # every test in turn, about 70 minutes
    python examples/training_demo/offload_tests.py run probe      # the probe's six, or bench for its four
    python examples/training_demo/offload_tests.py run probe-ep8 bench-groups
    python examples/training_demo/offload_tests.py results        # rank 0's lines of every test run

Each test waits for the one before it; one that dies is stopped, so the next
gets the devices, and the run goes on. Rank 0's lines of each test go to
``output/offload_tests/<test>.txt`` on this node, and ``results`` prints them
all. The script needs nothing beyond the standard library.
"""
import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent
OUT = REPO_ROOT / "output" / "offload_tests"
PROBE = "examples/training_demo/offload_probe.py"
BENCH = "examples/training_demo/bench_host_link.py"
SCRIPTS = (PROBE, BENCH, "examples/training_demo/offload_tests.py")
TRAIN_YAML = "examples/training_demo/train_qwen3_5_moe.yaml"
CLUSTER_ENV = "examples/training_demo/cluster_qwen3_5_moe.env"
WORLD = 64
RUN_ID_PATTERN = re.compile(r"run id\s*:\s*(\S+)")
# One status line per node: "node0   192.168.0.55   DEAD (exit 1) | <last line>".
NODE_STATE_PATTERN = re.compile(r"^node(\d+)\s+(\S+)\s+([^|]+?)\s*\|", re.MULTILINE)
LINES = "OFFLOAD_PROBE|HOST_LINK"


@dataclass(frozen=True)
class Test:
    """One launch: on how many nodes, which script, its arguments, and what it answers."""

    nodes: int
    script: str
    args: Tuple[str, ...]
    answers: str


def probe(ep: int, *extra: str) -> Test:
    """The probe on the Demo 2 crop at *ep*, every layer fully recomputed, with the probe options *extra*."""
    return Test(4, PROBE, (
        "--probe-ac-off=none", *extra, TRAIN_YAML,
        "--model.num_hidden_layers=8", "--dataset.data_config.seq_length=8192",
        "--training.global_batch_size=64", "--training.micro_batch_size=1", "--training.train_iters=20",
        "--accelerator.tp_size=1", "--accelerator.cp_size=1", "--accelerator.pp_size=1",
        f"--accelerator.ep_size={ep}", "--fsdp_config.dp_shard_size=32",
        f"--fsdp_config.edp_shard_size={WORLD // ep}", "--profiling.enabled=false",
    ), f"the copies' share of the link inside the step at EP {ep}")


def bench(name: str, answers: str, *args: str) -> Test:
    """The host link bench on the first node, its CSV under output/offload_tests on that node."""
    return Test(1, BENCH, (*args, "--out", f"output/offload_tests/{name}.csv"), answers)


# In the order "run all" takes them: what decides the most first.
TESTS: Dict[str, Test] = {
    "probe-ep1": probe(1),
    "probe-ep16": probe(16),
    "bench-groups": bench(
        "bench-groups", "the all_to_all's share over groups of 2, 4, 8 and 16 ranks, plain and the trainer's",
        "--collectives", "all_to_all,all_to_all_hp", "--group-sizes", "2,4,8,16", "--min-kib", "2097152"),
    "probe-ep8": probe(8),
    "probe-ep2": probe(2),
    "probe-chunk8": replace(probe(2, "--probe-chunk-mib=8"),
                            answers="what 8 MiB copies cost the step at EP 2, against probe-ep2's 64 MiB"),
    "bench-chunk64": bench(
        "bench-chunk64", "the trainer's exchange in pairs beside 64 MiB copies",
        "--collectives", "all_to_all_hp", "--group-sizes", "2", "--int-mib", "64", "--min-kib", "2097152"),
    "bench-chunk8": bench(
        "bench-chunk8", "the trainer's exchange in pairs beside 8 MiB copies",
        "--collectives", "all_to_all_hp", "--group-sizes", "2", "--int-mib", "8", "--min-kib", "2097152"),
    "bench-copiers": bench(
        "bench-copiers", "a die's copy speed with 16, 8, 4, 2 and 1 dies of the node copying",
        "--collectives", "none", "--copiers", "16,8,4,2,1", "--min-kib", "262144"),
    "probe-ep4": probe(4),
}
GROUPS = {
    "all": tuple(TESTS),
    "probe": tuple(name for name in TESTS if name.startswith("probe")),
    "bench": tuple(name for name in TESTS if name.startswith("bench")),
}


def read_cluster_env(env_path: Path) -> Dict[str, object]:
    """NODES, SSH_USER and LOG_DIR from the kit config, sourced by bash as the kit sources it."""
    script = (f'set -euo pipefail; source {shlex.quote(str(env_path))}; '
              'printf "%s\\n" "${NODES[*]:-}" "${SSH_USER:-root}" "${LOG_DIR:-}" "${REPO_DIR:-}"')
    result = subprocess.run(["bash", "-c", script], check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise SystemExit(f"could not read {env_path}: {result.stderr.strip() or 'sourcing failed'}")
    nodes, user, log_dir, repo_dir = (result.stdout.splitlines() + [""] * 4)[:4]
    if not nodes.split():
        raise SystemExit(f"{env_path} sets no NODES")
    return {"nodes": nodes.split(), "ssh_user": user,
            "log_dir": (log_dir or f"{repo_dir.rstrip('/')}/scripts/cluster/logs").rstrip("/")}


class Kit:
    """The cluster kit and ssh, bound to one kit config."""

    def __init__(self, cluster: str, env_path: Path, ssh: str) -> None:
        """The kit command *cluster* on *env_path*, and *ssh* to reach a node; both split as a shell would."""
        self.cluster, self.env_path, self.ssh = shlex.split(cluster), env_path, shlex.split(ssh)
        self.env = read_cluster_env(env_path)

    def command(self, *words: str) -> List[str]:
        """A kit invocation."""
        return [*self.cluster, "-c", str(self.env_path), *words]

    def run(self, *words: str) -> Tuple[int, str]:
        """Run a kit command, echoing it and its output; its exit code and output."""
        command = self.command(*words)
        print("+ " + " ".join(shlex.quote(word) for word in command), flush=True)
        result = subprocess.run(command, check=False, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        print(result.stdout or "", end="", flush=True)
        return result.returncode, result.stdout or ""

    def on_node(self, index: int, shell: str) -> str:
        """What *shell* prints on node *index*, empty if it cannot be reached."""
        host = f"{self.env['ssh_user']}@{self.env['nodes'][index]}"
        return subprocess.run([*self.ssh, host, shell], check=False, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout or ""

    def log(self, run_id: str, index: int) -> str:
        """Node *index*'s log of run *run_id*."""
        return f"{self.env['log_dir']}/{run_id}.node{index}.log"


def node_states(text: str) -> List[Tuple[int, str]]:
    """(node, state) for every node of a status table."""
    return [(int(m.group(1)), m.group(3).strip()) for m in NODE_STATE_PATTERN.finditer(text)]


def wait(kit: Kit, run_id: str, nodes: int, timeout: float, poll: float) -> Tuple[str, List[int]]:
    """Block until run *run_id* is over; how it ended, and the nodes that died.

    Only the first *nodes* nodes count, the ones the launch used: the others
    report NO RUN for this run, which would otherwise end up in its status.

    A dead rank ends the job but leaves the others waiting in a collective until
    HCCL's timeout, so a node reported DEAD stops the whole run at once, which
    also frees the devices for the next test.
    """
    deadline = time.monotonic() + timeout
    while True:
        text = subprocess.run(kit.command("status", run_id), check=False, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout or ""
        states = [(index, state) for index, state in node_states(text) if index < nodes]
        dead = [(index, state) for index, state in states if state.startswith("DEAD")]
        if dead:
            kit.run("kill", run_id)
            return "FAILED: " + ", ".join(f"node{index} {state}" for index, state in dead), [i for i, _ in dead]
        if states and not any(state == "RUNNING" for _, state in states):
            return ", ".join(sorted({state for _, state in states})), []
        if time.monotonic() > deadline:
            kit.run("kill", run_id)
            return f"TIMED OUT after {timeout:.0f} s, stopped", []
        time.sleep(poll)


def run_test(kit: Kit, name: str, test: Test, timeout: float, poll: float,
             guard: bool = True) -> Dict[str, str]:
    """Launch one test, wait for it, and keep rank 0's lines; its record.

    With *guard*, the kit's own readiness check runs first and the test is
    skipped rather than launched where any device of the config is busy.
    """
    print(f"\n=== {name}: {test.answers}", flush=True)
    if len(kit.env["nodes"]) < test.nodes:
        return {"test": name, "status": f"SKIPPED: needs {test.nodes} nodes, the config has "
                                         f"{len(kit.env['nodes'])}"}
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    if guard and kit.run("npu")[0]:
        print(f"=== {name}: SKIPPED, a device is busy or a node is unreachable", flush=True)
        return {"test": name, "status": "SKIPPED: a device is busy or a node is unreachable",
                "started": started}
    code, text = kit.run("torchrun", "-n", str(test.nodes), test.script, *test.args)
    found = RUN_ID_PATTERN.search(text)
    if code or not found:
        return {"test": name, "status": f"NOT LAUNCHED (kit exit {code})", "started": started}
    run_id = found.group(1)
    status, dead = wait(kit, run_id, test.nodes, timeout, poll)
    lines = [line for line in kit.on_node(0, f"grep -aE {shlex.quote(LINES)} {shlex.quote(kit.log(run_id, 0))}")
             .splitlines() if line.strip()]
    for index in dead[:2]:
        lines += [f"errors on node{index}:"] + kit.on_node(
            index, f"grep -aE '(Error|Exception):' {shlex.quote(kit.log(run_id, index))} | sort -u | head -n 8"
        ).splitlines()
    record = {"test": name, "run_id": run_id, "status": status, "started": started,
              "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
              "command": " ".join(shlex.quote(word) for word in kit.command("torchrun", "-n", str(test.nodes),
                                                                             test.script, *test.args))}
    OUT.mkdir(parents=True, exist_ok=True)
    header = [f"# {name}: {test.answers}", f"# run {run_id}, {status}, {started} to {record['ended']}",
              f"# {record['command']}"]
    (OUT / f"{name}.txt").write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print(f"=== {name}: run {run_id}, {status}, {len(lines)} lines kept in {OUT / (name + '.txt')}", flush=True)
    return record


def chosen(names: Sequence[str]) -> List[str]:
    """The tests *names* stands for, groups expanded, each once, in the order given."""
    picked: List[str] = []
    for name in names:
        for test in GROUPS.get(name, (name,)):
            if test not in TESTS:
                raise SystemExit(f"no test {test!r}: choose from {', '.join([*GROUPS, *TESTS])}")
            if test not in picked:
                picked.append(test)
    return picked


def run(kit: Kit, names: Sequence[str], timeout: float, poll: float, guard: bool = True) -> None:
    """Run the tests named one after another, each record added to output/offload_tests/runs.json.

    A run stopped by hand ends the sequence: whoever stopped it wants the
    devices, and the next test would take them straight back.
    """
    index = OUT / "runs.json"
    for name in chosen(names):
        record = run_test(kit, name, TESTS[name], timeout, poll, guard)
        records = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
        OUT.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(records + [record], indent=2) + "\n", encoding="utf-8")
        if "KILLED" in record["status"]:
            print(f"\n{name} was stopped by hand, so the rest are left unrun.", flush=True)
            break
    print(f"\ndone: python {Path(__file__).relative_to(REPO_ROOT).as_posix()} results", flush=True)


def check(kit: Kit) -> None:
    """Sync the tree to every node, then print each script's md5 here and on every node."""
    kit.run("sync")
    for script in SCRIPTS:
        print(f"here   {hashlib.md5((REPO_ROOT / script).read_bytes()).hexdigest()}  {script}")
    kit.run("exec", "md5sum " + " ".join(SCRIPTS))


def show_list() -> None:
    """Every test, what it answers, and the kit command it runs."""
    for group, names in GROUPS.items():
        print(f"{group}: {' '.join(names)}")
    for name, test in TESTS.items():
        command = ["cluster", "-c", CLUSTER_ENV, "torchrun", "-n", str(test.nodes), test.script, *test.args]
        print(f"\n{name}: {test.answers}\n  " + " ".join(shlex.quote(word) for word in command))


def results() -> None:
    """Print every test's kept lines, the latest run of each, for pasting back."""
    for name in TESTS:
        path = OUT / f"{name}.txt"
        if path.exists():
            print(path.read_text(encoding="utf-8"), end="")
    if not any((OUT / f"{name}.txt").exists() for name in TESTS):
        print(f"no results under {OUT}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run one step of the offload tests."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cluster", default="cluster", help="the kit command")
    parser.add_argument("--env", type=Path, default=REPO_ROOT / CLUSTER_ENV, help="the kit config")
    parser.add_argument("--ssh", default="ssh", help="how to reach a node")
    steps = parser.add_subparsers(dest="step", required=True)
    steps.add_parser("list", help="the tests and their commands")
    steps.add_parser("check", help="sync, then the scripts' md5 here and on every node")
    run_step = steps.add_parser("run", help="the tests or groups named, one after another")
    run_step.add_argument("names", nargs="+", help=f"of {', '.join([*GROUPS, *TESTS])}")
    run_step.add_argument("--timeout", type=float, default=1800, help="seconds one test may take")
    run_step.add_argument("--poll", type=float, default=20, help="seconds between status checks")
    run_step.add_argument("--no-guard", action="store_true",
                          help="launch without the kit's readiness check first")
    steps.add_parser("results", help="rank 0's lines of every test run")
    args = parser.parse_args(argv)
    if args.step == "results":
        results()
    elif args.step == "list":
        show_list()
    elif args.step == "check":
        check(Kit(args.cluster, args.env, args.ssh))
    else:
        run(Kit(args.cluster, args.env, args.ssh), args.names, args.timeout, args.poll,
            not args.no_guard)


if __name__ == "__main__":
    main(sys.argv[1:])
