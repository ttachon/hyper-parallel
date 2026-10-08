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
"""Run the per-layer recompute tests from the cluster's control node, and read them back.

Each test is one recompute plan of the demo's 8-layer Qwen3.5-MoE crop (layers 3
and 7 full attention, the others linear attention) at EP 2, dp_shard 32 and 8192
tokens, one sequence a die, run once by the strategy sweep's run stage. The
demo's 4 nodes give edp_shard 32. On 2 nodes edp_shard is 16, which doubles each
die's share of the expert state (bf16 weight, gradient and Muon momentum, about
0.56 GiB more a die) and changes nothing else a die holds; their results go to
output/recompute_plan_tests_2nodes, apart from the demo's. Block A
runs with the default allocator. Block B runs with
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True, which this script adds to the
kit config's REMOTE_ENV_SETUP for block B only and always takes out again. The
script needs nothing beyond the standard library, apart from PyYAML to write the
plan yamls.

Usage, from the repository root on the control node:
    python examples/training_demo/recompute_plan_tests.py setup             # or: setup --nodes 2
    python examples/training_demo/recompute_plan_tests.py unit
    python examples/training_demo/recompute_plan_tests.py run A
    python examples/training_demo/recompute_plan_tests.py run B
    python examples/training_demo/recompute_plan_tests.py results
    python examples/training_demo/recompute_plan_tests.py reset     # only after a block was cut off
"""
import argparse
import contextlib
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

DEMO_DIR = Path(__file__).resolve().parent
REPO_ROOT = DEMO_DIR.parent.parent
SWEEP = DEMO_DIR / "sweep_qwen3_5_moe.py"
TRAIN_YAML = DEMO_DIR / "train_qwen3_5_moe.yaml"
CLUSTER_ENV = DEMO_DIR / "cluster_qwen3_5_moe.env"
OUT = REPO_ROOT / "output" / "recompute_plan_tests"
DP_SHARD = 32
STRATEGY = ("--ep", "2", "--op", str(DP_SHARD))
DEMO_NODES = 4
DIES_PER_NODE = 16
ALLOCATOR = "export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True"

# Each plan is the activation_checkpoint section of a train yaml.
PLANS: Dict[str, Dict[str, Any]] = {
    "full": {"mode": "full"},
    "off3-7": {"mode": "full", "layers": {"3-7": "off"}},
    "off3_5-7": {"mode": "full", "layers": {"3": "off", "5-7": "off"}},
    "off0-1_3_7": {"mode": "full", "layers": {"0-1": "off", "3": "off", "7": "off"}},
    "off2_4-7": {"mode": "full", "layers": {"2": "off", "4-7": "off"}},
    "off2-7": {"mode": "full", "layers": {"2-7": "off"}},
    "base_off_0-2": {"mode": "off", "layers": {"0-2": "full"}},
}
BLOCKS = {
    "A": ("full", "base_off_0-2", "off3-7", "off3_5-7", "off0-1_3_7"),
    "B": ("full", "off3-7", "off2_4-7", "off2-7"),
}
# What each pair of runs tests, and the two runs: the second is the reference.
COMPARISONS = (
    ("plan over mode off vs the same plan over full", ("A", "base_off_0-2"), ("A", "off3-7")),
    ("linear layers kept first (0, 1) vs last (5, 6)", ("A", "off0-1_3_7"), ("A", "off3_5-7")),
    ("expandable segments vs the default allocator, off3-7", ("B", "off3-7"), ("A", "off3-7")),
    ("expandable segments vs the default allocator, full", ("B", "full"), ("A", "full")),
)
UNIT_TESTS = (
    "tests/ut/auto_models/distributed/test_activation_checkpoint.py",
    "tests/ut/trainer/test_config_overrides.py",
    "tests/ut/trainer/test_base.py",
    "tests/ut/examples/training_demo/test_recompute_plan_tests.py",
)
LOGGED_PLAN = "per layer in model.layers: .*|Using HuggingFace native"
FAILURE = r"Tried to allocate [^)]*\)|EL0004[^.]*"


def out_dir(nodes: int) -> Path:
    """The folder the tests on this many nodes write to: the demo's 4 nodes keep the first one."""
    return OUT if nodes == DEMO_NODES else OUT.with_name(f"{OUT.name}_{nodes}nodes")


def check_nodes(nodes: int) -> None:
    """Refuse a node count whose dies dp_shard 32 cannot cover a whole number of times.

    Raises:
        SystemExit: The node count is below 1, or odd.
    """
    if nodes < 1 or nodes * DIES_PER_NODE % DP_SHARD:
        raise SystemExit(f"--nodes {nodes}: dp_shard {DP_SHARD} needs a multiple of {DP_SHARD} dies, "
                         f"so an even number of {DIES_PER_NODE}-die nodes")


def plan_yaml(plan: str) -> Path:
    """The train yaml one plan runs from."""
    return DEMO_DIR / f"plan_test_{plan}.yaml"


def write_plan_yamls() -> List[Path]:
    """Write one train yaml per plan: the demo's own, with the plan as its activation_checkpoint."""
    import yaml  # pylint: disable=import-outside-toplevel
    base = yaml.safe_load(TRAIN_YAML.read_text(encoding="utf-8"))
    paths = []
    for plan, section in PLANS.items():
        path = plan_yaml(plan)
        path.write_text(yaml.safe_dump(dict(base, activation_checkpoint=section), sort_keys=False),
                        encoding="utf-8")
        paths.append(path)
    return paths


def with_allocator(text: str, enabled: bool) -> str:
    """Return a kit config with the allocator setting added to its REMOTE_ENV_SETUP, or taken out.

    Raises:
        SystemExit: The config sets the allocator some other way, or has no
            one-line REMOTE_ENV_SETUP to add the setting to.
    """
    setting = f"; {ALLOCATOR}"
    stripped = text.replace(setting, "")
    if "PYTORCH_NPU_ALLOC_CONF" in stripped:
        raise SystemExit(f"{CLUSTER_ENV} sets PYTORCH_NPU_ALLOC_CONF its own way: take that out by hand first")
    if not enabled:
        return stripped
    lines = stripped.splitlines(keepends=True)
    for index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        if body.startswith("REMOTE_ENV_SETUP=") and body.endswith('"'):
            lines[index] = body[:-1] + setting + '"' + line[len(body):]
            return "".join(lines)
    raise SystemExit(f'{CLUSTER_ENV} has no one-line REMOTE_ENV_SETUP="..." to add the allocator setting to')


def set_allocator(enabled: bool) -> None:
    """Add the allocator setting to the kit config, or take it out, and say which."""
    text = CLUSTER_ENV.read_text(encoding="utf-8")
    updated = with_allocator(text, enabled)
    if updated != text:
        CLUSTER_ENV.write_text(updated, encoding="utf-8")
    print(f"expandable segments {'ON' if enabled else 'off'} in {CLUSTER_ENV.name}", flush=True)


def sweep_command(block: str, plan: str, out: Optional[Path] = None) -> List[str]:
    """The sweep invocation that runs one plan once, writing under ``out`` (default OUT)."""
    out = out or OUT
    command = [sys.executable, str(SWEEP), *STRATEGY, "--profile-memory", "none",
               "--config", str(plan_yaml(plan)), "--out", str(out / f"{block}_{plan}"), "--only", "run"]
    if PLANS[plan].get("mode", "off") == "off":
        # The sweep states the mode on the command line, which overrides the yaml's.
        command += ["--activation-checkpoint", "off"]
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


def setup(pool: Path, nodes: int = DEMO_NODES) -> None:
    """Write the plan yamls, then select the nodes, copy the code to them and build the dataset."""
    check_nodes(nodes)
    for path in write_plan_yamls():
        print(f"wrote {path.relative_to(REPO_ROOT)}", flush=True)
    set_allocator(False)
    command = [sys.executable, str(SWEEP), "--pool", str(pool), "--nodes", str(nodes), *STRATEGY,
               "--only", "select", "--only", "mirror", "--only", "data"]
    if stream(command, out_dir(nodes) / "setup.log"):
        raise SystemExit("the sweep's select, mirror and data stages failed: see the lines above")


def unit() -> int:
    """Run the unit tests of the per-layer plan, where pytest is installed."""
    if importlib.util.find_spec("pytest") is None:
        print("pytest is not installed in this environment: skip this step", flush=True)
        return 0
    return stream([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *UNIT_TESTS],
                  OUT / "unit.log")


def _stop(signum: int, frame: Any) -> None:
    """Turn a hangup or a termination into SystemExit, so cleanup still runs."""
    del frame
    raise SystemExit(f"stopped by signal {signum}")


@contextlib.contextmanager
def _signals_stop_cleanly() -> Iterator[None]:
    """Make a lost terminal or a kill end the block through its cleanup, then restore the handlers."""
    previous = {}
    for name in ("SIGHUP", "SIGTERM"):
        if hasattr(signal, name):
            number = getattr(signal, name)
            previous[number] = signal.signal(number, _stop)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def run_block(block: str, plans: Sequence[str] = (), out: Optional[Path] = None) -> None:
    """Run a block's plans one after the other, the allocator set as the block needs."""
    out = out or OUT
    plans = tuple(plans) or BLOCKS[block]
    unknown = [plan for plan in plans if plan not in PLANS]
    if unknown:
        raise SystemExit(f"unknown plan(s) {unknown}; the plans are {list(PLANS)}")
    missing = [plan_yaml(plan).name for plan in plans if not plan_yaml(plan).exists()]
    if missing:
        raise SystemExit(f"run setup first: {', '.join(missing)} missing")
    with _signals_stop_cleanly():
        set_allocator(block == "B")
        try:
            for index, plan in enumerate(plans, 1):
                folder = out / f"{block}_{plan}"
                if folder.exists():
                    folder.rename(folder.with_name(f"{folder.name}.{time.strftime('%Y%m%d_%H%M%S')}"))
                print(f"\n##### block {block}, test {index} of {len(plans)}: {plan} #####", flush=True)
                code = stream(sweep_command(block, plan, out), out / f"block_{block}.log")
                if code:
                    print(f"the sweep exited with {code}; going on with the next test", flush=True)
        finally:
            if block == "B":
                set_allocator(False)


def read_run(block: str, plan: str, out: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The sweep's record of one test's run under ``out`` (default OUT), or None where it has not run."""
    try:
        states = json.loads(((out or OUT) / f"{block}_{plan}" / "run_states.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    runs = [data for data in states.values() if isinstance(data, dict) and "run_id" in data]
    return runs[0] if runs else None


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


def _row(name: str, run: Dict[str, Any], logged: str) -> str:
    """One test's line: state, step time, peaks and the plan the trainer logged."""
    allocated, reserved = run.get("max_allocated_gb"), run.get("max_reserved_gb")
    gap = None if allocated is None or reserved is None else float(reserved) - float(allocated)
    return (f"{name:16s} {'FAILED' if run.get('failed') else 'ok':6s} "
            f"{_number(run.get('step_trainer'), 1):>8s} {_number(run.get('step_trainer_sd'), 1):>6s} "
            f"{_number(allocated, 4):>8s} {_number(reserved, 4):>8s} {_number(gap, 2):>6s}  "
            f"{logged or 'no plan line'}")


def _difference(first: Dict[str, Any], second: Dict[str, Any], key: str, unit_name: str) -> str:
    """The first run's value less the second's, for one recorded figure."""
    if first.get(key) is None or second.get(key) is None:
        return f"{key} -"
    return f"{key} {float(first[key]) - float(second[key]):+.3f} {unit_name}"


def results(env: Dict[str, Any], out: Optional[Path] = None) -> None:
    """Print one line per test, the reason each failure gave, then each comparison.

    A plan named on the command line outside its block's list gets a line too,
    when it ran.
    """
    out = out or OUT
    print(f"{len(env['nodes'])} node(s) in {CLUSTER_ENV.name}, tests in {out}")
    print(f"{'test':16s} {'state':6s} {'step ms':>8s} {'sd':>6s} {'alloc':>8s} {'reserved':>8s} "
          f"{'gap':>6s}  plan the trainer logged")
    measured = {}
    for block, plans in BLOCKS.items():
        for plan in PLANS:
            run = read_run(block, plan, out)
            name = f"{block} {plan}"
            if run is None:
                if plan in plans:
                    print(f"{name:16s} not run")
                continue
            measured[(block, plan)] = run
            print(_row(name, run, remote_grep(env, 0, run["run_id"], LOGGED_PLAN)))
            for node in range(len(env["nodes"]) if run.get("failed") else 0):
                reason = remote_grep(env, node, run["run_id"], FAILURE)
                if reason:
                    print(f"{'':16s} node{node}: {reason}")
    print()
    for label, first, second in COMPARISONS:
        runs = measured.get(first), measured.get(second)
        if None in runs or any(run.get("failed") for run in runs):
            print(f"{label}: needs both runs to succeed")
            continue
        print(f"{label}: " + ", ".join(_difference(*runs, key, unit_name) for key, unit_name in (
            ("step_trainer", "ms"), ("max_allocated_gb", "GiB"), ("max_reserved_gb", "GiB"))))
    if ALLOCATOR in CLUSTER_ENV.read_text(encoding="utf-8"):
        print(f"\nallocator setting in {CLUSTER_ENV.name} now: ON, left by a block that did not finish; "
              f"take it out with: python {Path(__file__).relative_to(REPO_ROOT).as_posix()} reset")
    else:
        print(f"\nallocator setting in {CLUSTER_ENV.name} now: off")


def read_cluster_env() -> Dict[str, Any]:
    """The kit config's nodes, ssh user and log directory, read the way the sweep reads them."""
    spec = importlib.util.spec_from_file_location("sweep_qwen3_5_moe", SWEEP)
    sweep = importlib.util.module_from_spec(spec)
    # Registered before it runs, so the sweep's dataclasses resolve their own module.
    sys.modules[spec.name] = sweep
    spec.loader.exec_module(sweep)
    return sweep.read_cluster_env(CLUSTER_ENV)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Run one step of the tests."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    steps = parser.add_subparsers(dest="step", required=True)
    setup_step = steps.add_parser("setup", help="plan yamls, then the sweep's select, mirror and data")
    setup_step.add_argument("--pool", type=Path, default=Path("/home/tt/cluster_all.env"))
    setup_step.add_argument("--nodes", type=int, default=DEMO_NODES,
                            help="nodes to select; the tests on any count but 4 write to their own folder")
    steps.add_parser("unit", help="the per-layer plan's unit tests")
    run_step = steps.add_parser("run", help="one block of tests, or the plans named")
    run_step.add_argument("block", choices=sorted(BLOCKS))
    run_step.add_argument("plans", nargs="*")
    steps.add_parser("results", help="one line per test, then the comparisons")
    steps.add_parser("reset", help="take the allocator setting out of the kit config")
    args = parser.parse_args(argv)
    if args.step == "setup":
        setup(args.pool, args.nodes)
    elif args.step == "unit":
        sys.exit(unit())
    elif args.step == "reset":
        set_allocator(False)
    else:
        # The tests run on, and are read from, the nodes the last setup selected.
        env = read_cluster_env()
        if args.step == "run":
            run_block(args.block, args.plans, out_dir(len(env["nodes"])))
        else:
            results(env, out_dir(len(env["nodes"])))


if __name__ == "__main__":
    main()
