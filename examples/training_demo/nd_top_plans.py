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
"""Run ND's 16 best strategies for the Qwen3.5-MoE demo, each with its recompute plan, and read them back.

ND's search chose a recompute mode for every layer of every strategy, off or full, the fastest its memory model
fits in the budget it was given, and ranked the strategies with their plans (SEARCH says how). PICKS are the 16
it ranks best that this model and trainer can run, chosen as the sweep's --nd-top chooses, in ND's order, each
with its plan as the trainer states it: activation_checkpoint.mode, and .layers for the layers that run the
other mode. The budget is Demo 2's: the 58.00 GiB PyTorch can hold on an A3 die, less the 8.76 GiB the default
allocator reserved beyond allocation at full recompute.

Each pick runs once through the strategy sweep's run stage, on 4 nodes at 8192 tokens, 8 layers and one sequence
a die, and the trainer times steps 6 to 10 unprofiled. Two anchors run ND's first pick and Demo 2's strategy fully
recomputed, each right after its pick: the baseline each plan is read against in this round. The yamls keep the
demo's delta rule, hyperparallel, which no earlier round of this demo ran, so no earlier time is a baseline.
Before each launch the kit's readiness check runs, and a busy device skips that run until the next ``run``.

The script needs nothing beyond the standard library, apart from PyYAML to write the plan yamls.

Usage, from the repository root on the control node:
    python examples/training_demo/nd_top_plans.py setup        # plan yamls, 4 nodes from the pool, mirror, data
    python examples/training_demo/nd_top_plans.py run          # every pick, then the anchors; a run kept is skipped
    python examples/training_demo/nd_top_plans.py run p08 a1   # only these; --again runs a kept one anew
    python examples/training_demo/nd_top_plans.py results
"""
import argparse
import csv
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Set

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent
SWEEP = DEMO_DIR / "sweep_qwen3_5_moe.py"
TRAIN_YAML = DEMO_DIR / "train_qwen3_5_moe.yaml"
CLUSTER_ENV = DEMO_DIR / "cluster_qwen3_5_moe.env"
OUT = REPO_ROOT / "output" / "nd_top_plans"
NODES = 4
# What PyTorch could hold on an A3 die on 7 October; a run whose reserved peak passes it ran out of memory.
CEILING_GIB = 58.00
SEARCH = ("nd/main d2d162b3 with Transformers 5.14.1, 12 October 2026: run_nd -f hyper_v2 -y <the sweep's "
          "nd_model.yaml: 8 layers, 8192 tokens, batch 64, census, expert shard group> -d 64 -A A3 -b 64 "
          "-M 49.25GB -ar --recompute off full -c <the ratios the 30 September round fitted>; picks by the "
          "sweep's pick_nd_top, ranks in ND's whole order")
IMPORTS = 'python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"'
# The kit's exec runs the nodes together by default, every line tagged [nodeN] in the node's colour; with -S it runs
# them one at a time, each under a "===== nodeN (host) =====" header. What follows the tag is the node's answer.
COLOUR = re.compile(r"\x1b\[[0-9;]*m")
NODE_TAG = re.compile(r"^\[node[0-9]+\]\s*")
LOGGED_PLAN = "per layer in model.layers: .*"
LOGGED_RULE = "delta_rule=hyperparallel: [0-9]+ linear-attention layers switched"
FAILURE = r"Tried to allocate [^)]*\)|EL0004[^.]*|cannot import name '[^']*'"
OFF_3_7 = {"3-7": "off"}


@dataclass(frozen=True)
class Run:
    """One launch: a strategy, the recompute it runs, and ND's estimate of it.

    ``rank`` is the strategy's place in ND's whole order, ``nd_ms`` ND's step
    with the 30 September ratios and ``nd_mib`` its peak, under the recompute
    the run states: its plan for a pick, full recompute for an anchor.
    """

    name: str
    ep: int
    cp: int
    op: int
    mode: str
    rank: int
    nd_ms: float
    nd_mib: int
    layers: Dict[str, str] = field(default_factory=dict)

    @property
    def plan(self) -> str:
        """The recompute in a few characters: full, off, or the layers a full run leaves off."""
        if not self.layers:
            return self.mode
        return "off" + "_".join(key for key, mode in self.layers.items() if mode == "off")

    @property
    def strategy(self) -> str:
        """The degrees that vary across this round."""
        return f"CP{self.cp} EP{self.ep} OP{self.op}"

    @property
    def folder(self) -> str:
        """The sweep's output folder for this run, under OUT."""
        return f"{self.name}_ep{self.ep}_cp{self.cp}_op{self.op}_{self.plan}"


PICKS = (
    Run("p01", 1, 1, 64, "full", 1, 5440.5, 47126, OFF_3_7),
    Run("p02", 1, 1, 32, "full", 3, 5447.2, 47849, OFF_3_7),
    Run("p03", 1, 1, 16, "full", 5, 5460.6, 49295, OFF_3_7),
    Run("p04", 2, 1, 64, "full", 7, 5469.4, 48406, OFF_3_7),
    Run("p05", 2, 1, 32, "full", 9, 5471.0, 48553, OFF_3_7),
    Run("p06", 2, 1, 16, "full", 11, 5474.0, 48847, OFF_3_7),
    Run("p07", 2, 1, 8, "full", 13, 5480.2, 49434, OFF_3_7),
    Run("p08", 2, 2, 32, "off", 15, 5546.8, 36396),
    Run("p09", 2, 2, 16, "off", 16, 5547.1, 36690),
    Run("p10", 2, 2, 8, "off", 17, 5547.6, 37277),
    Run("p11", 2, 2, 64, "off", 18, 5547.7, 36249),
    Run("p12", 2, 2, 4, "off", 19, 5548.8, 38452),
    Run("p13", 2, 2, 2, "off", 20, 5551.2, 40801),
    Run("p14", 4, 2, 32, "off", 21, 5573.0, 36908),
    Run("p15", 4, 2, 16, "off", 22, 5573.3, 37202),
    Run("p16", 4, 2, 8, "off", 23, 5573.9, 37789),
)
# Full recompute at ND's first pick and at Demo 2's strategy; rank and estimate from ND's full-recompute search.
ANCHORS = (
    Run("a1", 1, 1, 64, "full", 1, 6395.3, 27446),
    Run("a2", 2, 1, 32, "full", 15, 6604.2, 27593),
)
# Each anchor and the pick it is the baseline of.
PAIRS = (("p01", "a1"), ("p05", "a2"))
RUNS = PICKS + ANCHORS
# The order a round launches in: each anchor right after its pick, so a round cut short keeps its baseline.
ORDER = ("p01", "a1", "p02", "p03", "p04", "p05", "a2") + tuple(run.name for run in PICKS[5:])


def runs_named(names: Sequence[str]) -> List[Run]:
    """The runs *names* stands for, in the order given; every run in ORDER when none is named.

    Raises:
        SystemExit: A name no run has.
    """
    by_name = {run.name: run for run in RUNS}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise SystemExit(f"no run {', '.join(unknown)}: the runs are p01 to p{len(PICKS):02d}, a1 and a2")
    return [by_name[name] for name in dict.fromkeys(names or ORDER)]


def plan_yaml(run: Run) -> Path:
    """The train yaml a run's recompute runs from."""
    return DEMO_DIR / f"nd_top_plan_{run.plan}.yaml"


def write_plan_yamls() -> List[Path]:
    """Write one train yaml per recompute: the demo's own, with it as the activation_checkpoint section."""
    import yaml  # pylint: disable=import-outside-toplevel
    base = yaml.safe_load(TRAIN_YAML.read_text(encoding="utf-8"))
    written: Dict[Path, None] = {}
    for run in RUNS:
        path = plan_yaml(run)
        if path in written:
            continue
        section: Dict[str, Any] = {"mode": run.mode}
        if run.layers:
            section["layers"] = dict(run.layers)
        path.write_text(yaml.safe_dump(dict(base, activation_checkpoint=section), sort_keys=False),
                        encoding="utf-8")
        written[path] = None
    return list(written)


def sweep_command(run: Run) -> List[str]:
    """The sweep invocation that runs one strategy once with its recompute, writing under OUT."""
    command = [sys.executable, str(SWEEP), "--ep", str(run.ep), "--cp", str(run.cp), "--op", str(run.op),
               "--profile-memory", "none", "--config", str(plan_yaml(run)), "--out", str(OUT / run.folder),
               "--only", "run"]
    if run.mode != "full":
        # The sweep states the mode on the command line, which overrides the yaml's.
        command += ["--activation-checkpoint", run.mode]
    return command


def stream(command: Sequence[str], log: Path) -> int:
    """Run a command from the repository root, echoing its output and appending it to a log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a", encoding="utf-8") as handle, subprocess.Popen(
            command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}) as process:
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
    return process.returncode


def kit(*words: str) -> List[str]:
    """A cluster kit invocation on the demo's kit config."""
    return ["cluster", "-c", str(CLUSTER_ENV), *words]


def same_environment() -> bool:
    """Whether every node answers the same torch and torch_npu versions, the answers printed.

    A node whose sympy had drifted ended every rank on it with exit 2 and no
    traceback on 12 October, which reads as a defect in the run.
    """
    result = subprocess.run(kit("exec", "-S", IMPORTS), check=False, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout or "", end="", flush=True)
    return result.returncode == 0 and len(node_answers(result.stdout or "")) == 1


def node_answers(text: str) -> Set[str]:
    """The distinct lines the nodes printed to a kit exec, without the kit's headers, tags and colours."""
    lines = (NODE_TAG.sub("", COLOUR.sub("", line)).strip() for line in text.splitlines())
    return {line for line in lines if line and not line.startswith(("=====", "+ "))}


def setup(pool: Path) -> None:
    """Write the plan yamls, then select the nodes, copy the code to them, build the dataset and check them."""
    for path in write_plan_yamls():
        print(f"wrote {path.relative_to(REPO_ROOT)}", flush=True)
    command = [sys.executable, str(SWEEP), "--pool", str(pool), "--nodes", str(NODES), "--out",
               str(OUT / "setup"), "--only", "select", "--only", "mirror", "--only", "data"]
    if stream(command, OUT / "setup.log"):
        raise SystemExit("the sweep's select, mirror and data stages failed: see the lines above")
    if not same_environment():
        raise SystemExit("the nodes do not answer the same torch versions, so a launch would fail on the odd one: "
                         "fix that node, or run setup again to select others")


def read_run(run: Run) -> Optional[Dict[str, Any]]:
    """The sweep's record of one run, or None where it has not run."""
    try:
        states = json.loads((OUT / run.folder / "run_states.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    records = [data for data in states.values() if isinstance(data, dict) and "run_id" in data]
    return records[0] if records else None


def _state(record: Dict[str, Any]) -> str:
    """A recorded run in a few words: its state, step and peak."""
    if record.get("failed"):
        return "FAILED"
    return (f"ok, {_number(record.get('step_trainer'), 1)} ms a step, "
            f"{_number(record.get('max_allocated_gb'), 2)} GiB allocated, "
            f"{_number(record.get('max_reserved_gb'), 2)} reserved")


def launch(run: Run, again: bool) -> None:
    """Run one strategy through the sweep, unless it has run or a device is busy."""
    folder = OUT / run.folder
    kept = read_run(run)
    if kept is not None and not again:
        print(f"{run.name}: kept from an earlier run ({_state(kept)}); --again runs it anew", flush=True)
        return
    if subprocess.run(kit("npu"), check=False).returncode:
        print(f"{run.name}: SKIPPED, a device is busy or a node is unreachable; run it later", flush=True)
        return
    if folder.exists():
        folder.rename(folder.with_name(f"{folder.name}.{time.strftime('%Y%m%d_%H%M%S')}"))
    if stream(sweep_command(run), OUT / "runs.log"):
        print(f"{run.name}: the sweep exited with an error; going on with the next run", flush=True)
    record = read_run(run)
    print(f"{run.name}: {_state(record) if record else 'no record, so it did not run'}", flush=True)


def run_all(names: Sequence[str], again: bool) -> None:
    """Launch the runs named, or all, one after the other, once the nodes agree on their environment."""
    chosen = runs_named(names)
    missing = sorted({plan_yaml(run).name for run in chosen if not plan_yaml(run).exists()})
    if missing:
        raise SystemExit(f"run setup first: {', '.join(missing)} missing")
    if not same_environment():
        raise SystemExit("the nodes do not answer the same torch versions: fix the odd node, or run setup again")
    for index, run in enumerate(chosen, 1):
        print(f"\n##### {index} of {len(chosen)}: {run.name}, {run.strategy}, recompute {run.plan} #####",
              flush=True)
        launch(run, again)
    print(f"\ndone: python {Path(__file__).relative_to(REPO_ROOT).as_posix()} results", flush=True)


def load_sweep() -> ModuleType:
    """The strategy sweep, loaded by path, for its kit config reader and rank correlation."""
    spec = importlib.util.spec_from_file_location("sweep_qwen3_5_moe", SWEEP)
    sweep = importlib.util.module_from_spec(spec)
    # Registered before it runs, so the sweep's dataclasses resolve their own module.
    sys.modules[spec.name] = sweep
    spec.loader.exec_module(sweep)
    return sweep


def remote_grep(env: Dict[str, Any], node: int, run_id: str, pattern: str) -> str:
    """The first match of an extended regex in one node's log of a run, or an empty string."""
    log = f"{env['log_dir']}/{run_id}.node{node}.log"
    command = f"grep -a -m1 -hoE {shlex.quote(pattern)} {shlex.quote(log)}"
    return subprocess.run(
        ["ssh", f"{env['ssh_user']}@{env['nodes'][node]}", command],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    ).stdout.strip()


def _number(value: Any, digits: int) -> str:
    """A measured value at fixed precision, or a dash where there is none."""
    return "-" if value is None else f"{float(value):.{digits}f}"


def measure(env: Dict[str, Any], run: Run) -> Dict[str, Any]:
    """One run's row: ND's estimate, what the sweep recorded, and what the trainer logged."""
    record = read_run(run) or {}
    row: Dict[str, Any] = {"run": run.name, "nd_rank": run.rank, "strategy": run.strategy, "recompute": run.plan,
                           "nd_ms": run.nd_ms, "nd_gib": round(run.nd_mib / 1024, 2),
                           "state": "not run" if not record else ("FAILED" if record.get("failed") else "ok"),
                           "step_ms": record.get("step_trainer"), "step_sd": record.get("step_trainer_sd"),
                           "alloc_gib": record.get("max_allocated_gb"), "reserved_gib": record.get("max_reserved_gb"),
                           "logged_plan": "", "logged_rule": "", "failure": ""}
    if record.get("run_id"):
        row["logged_plan"] = remote_grep(env, 0, record["run_id"], LOGGED_PLAN)
        row["logged_rule"] = remote_grep(env, 0, record["run_id"], LOGGED_RULE)
        for node in range(len(env["nodes"]) if record.get("failed") else 0):
            reason = remote_grep(env, node, record["run_id"], FAILURE)
            if reason:
                row["failure"] = f"node{node}: {reason}"
                break
    return row


def _line(row: Dict[str, Any]) -> str:
    """One run's line of the table."""
    allocated, reserved = row["alloc_gib"], row["reserved_gib"]
    gap = None if allocated is None or reserved is None else float(reserved) - float(allocated)
    logged = row["logged_plan"].replace("per layer in model.layers: ", "") or "-"
    return (f"{row['run']:4s} {row['nd_rank']:>4d}  {row['strategy']:15s} {row['recompute']:7s} "
            f"{row['nd_ms']:8.1f} {row['nd_gib']:7.2f}  {row['state']:7s} {_number(row['step_ms'], 1):>8s} "
            f"{_number(row['step_sd'], 1):>6s} {_number(allocated, 2):>7s} {_number(reserved, 2):>8s} "
            f"{_number(gap, 2):>6s}  {logged}{'' if row['logged_rule'] else ' (no rule line)'}")


def _timed(row: Dict[str, Any]) -> bool:
    """Whether a run finished with a step time."""
    return row["state"] == "ok" and bool(row["step_ms"])


def _following_nd(timed: Sequence[Dict[str, Any]], sweep: ModuleType) -> List[str]:
    """Where ND's best timed pick lands, what following it costs, and how well ND's order holds."""
    fastest = min(timed, key=lambda row: float(row["step_ms"]))
    first = timed[0]
    place = sorted(float(row["step_ms"]) for row in timed).index(float(first["step_ms"])) + 1
    lines = [f"ND's best measured pick, {first['run']} ({first['strategy']}, {first['recompute']}, ND's "
             f"#{first['nd_rank']}), measures {float(first['step_ms']):.1f} ms, {place} of {len(timed)} picks timed. "
             f"The fastest is {fastest['run']} ({fastest['strategy']}, {fastest['recompute']}) at "
             f"{float(fastest['step_ms']):.1f} ms, so following ND costs "
             f"{float(first['step_ms']) / float(fastest['step_ms']) - 1:.1%}."]
    correlation = sweep.spearman([row["nd_ms"] for row in timed], [float(row["step_ms"]) for row in timed])
    if correlation is not None:
        lines.append(f"Rank correlation of ND's step with the measured one over {len(timed)} picks: "
                     f"{correlation:+.2f} (1 is ND's order exactly).")
    return lines


def _against_full(rows: Dict[str, Dict[str, Any]]) -> List[str]:
    """Each anchored plan's step against the same strategy fully recomputed in this round, measured and ND's."""
    lines = []
    for pick, anchor in PAIRS:
        plan, full = rows[pick], rows[anchor]
        if _timed(plan) and _timed(full):
            lines.append(f"{plan['strategy']}, {plan['recompute']} against full in this round: measured "
                         f"{float(plan['step_ms']) / float(full['step_ms']) - 1:+.1%}, ND "
                         f"{plan['nd_ms'] / full['nd_ms'] - 1:+.1%}.")
        else:
            lines.append(f"{plan['strategy']}, {plan['recompute']} against full: needs {pick} and {anchor} to run.")
    return lines


def _memory(rows: Dict[str, Dict[str, Any]]) -> List[str]:
    """ND's peak against the trainer's, and the runs whose reserved peak passed what PyTorch can hold."""
    measured = [row for row in rows.values() if row["alloc_gib"] is not None]
    if not measured:
        return []
    error = sum(abs(row["nd_gib"] / float(row["alloc_gib"]) - 1) for row in measured) / len(measured)
    over = [row["run"] for row in measured if float(row["reserved_gib"] or 0) > CEILING_GIB]
    return [f"ND's peak against the trainer's max allocated over {len(measured)} runs: mean |error| {error:.1%}; "
            f"reserved above {CEILING_GIB:.2f} GiB: {', '.join(over) or 'none'}."]


def verdict(rows: Dict[str, Dict[str, Any]], sweep: ModuleType) -> List[str]:
    """What following ND costs, how well its order holds, each plan against full, and memory."""
    timed = [rows[run.name] for run in PICKS if _timed(rows[run.name])]
    if not timed:
        return ["No pick has a step time yet."]
    return _following_nd(timed, sweep) + _against_full(rows) + _memory(rows)


def results(env: Dict[str, Any], sweep: ModuleType) -> None:
    """Print one line per run in ND's order, each failure's reason, then the verdict; write results.csv."""
    rows = {run.name: measure(env, run) for run in RUNS}
    print(f"{len(PICKS)} picks and {len(ANCHORS)} anchors in {OUT}, {len(env['nodes'])} node(s) in "
          f"{CLUSTER_ENV.name}\nranked by {SEARCH}")
    print(f"{'run':4s} {'ND#':>4s}  {'strategy':15s} {'recomp':7s} {'ND ms':>8s} {'ND GiB':>7s}  {'state':7s} "
          f"{'step ms':>8s} {'sd':>6s} {'alloc':>7s} {'reserved':>8s} {'gap':>6s}  plan the trainer logged")
    for run in RUNS:
        if run is ANCHORS[0]:
            print("anchors, fully recomputed; their rank and estimate are from ND's full-recompute search:")
        print(_line(rows[run.name]))
        if rows[run.name]["failure"]:
            print(f"{'':10s}{rows[run.name]['failure']}")
    print()
    print("\n".join(verdict(rows, sweep)))
    table = OUT / "results.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    with open(table, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[RUNS[0].name]))
        writer.writeheader()
        writer.writerows(rows[run.name] for run in RUNS)
    print(f"written to {table}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run one step of the round."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    setup_step = steps.add_parser("setup", help="plan yamls, then the sweep's select, mirror and data, then versions")
    setup_step.add_argument("--pool", type=Path, default=Path("/home/tt/cluster_all.env"))
    run_step = steps.add_parser("run", help="every pick then the anchors, or the runs named")
    run_step.add_argument("names", nargs="*", help="p01 to p16, a1, a2")
    run_step.add_argument("--again", action="store_true", help="run a kept run anew, its old folder renamed")
    steps.add_parser("results", help="one line per run, then the verdict")
    args = parser.parse_args(argv)
    if args.step == "setup":
        setup(args.pool)
    elif args.step == "run":
        run_all(args.names, args.again)
    else:
        sweep = load_sweep()
        results(sweep.read_cluster_env(CLUSTER_ENV), sweep)


if __name__ == "__main__":
    main()
