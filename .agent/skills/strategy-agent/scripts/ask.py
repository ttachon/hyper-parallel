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
"""Ask ND for a layout: describe the run, price every valid layout once, keep the run, print the answer.

Run from the HyperParallel checkout whose ND should answer, with the Python that has its dependencies:

    python .agent/skills/strategy-agent/scripts/ask.py --family qwen3_5_moe --cluster a3 \\
        --layers 8 --seq 8192 --global-batch 64 --micro-batch 1 --dies 64 --recompute full \\
        --request "the request, in the asker's words"

``--yaml <trainer yaml>`` describes the run with the asker's own config instead of a template. Everything
lands in ``output/strategy-agent/<time>/``: the request, the yaml ND read, the command, the versions,
ND's log and ranking, and the answer, so that any answer can be checked or run again.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

import answer

SKILL = Path(__file__).resolve().parents[1]
MODES = ("off", "selective", "full")
# Request field -> where the trainer yaml states it.
FIELDS = {
    "layers": ("model", "num_hidden_layers"),
    "seq": ("dataset", "data_config", "seq_length"),
    "global_batch": ("training", "global_batch_size"),
    "micro_batch": ("training", "micro_batch_size"),
    "dies": ("context", "device_num"),
}
# ND log lines the answer must pass on: a part it did not price, a field it priced as 0, what it refused.
NOTICES = ("MTP layer", "no parser set", "priced fully recomputed", "refus", "Error", "WARNING")


def repo_root() -> Path:
    """The checkout this skill sits in: ND is run from it."""
    for folder in SKILL.parents:
        if (folder / "hyper_parallel" / "__init__.py").is_file():
            return folder
    raise SystemExit(f"no HyperParallel checkout above {SKILL}")


def _set(doc: Dict[str, Any], path: Sequence[str], value: Any) -> None:
    """Set ``doc[path[0]][path[1]]...`` to value, making the missing levels."""
    for key in path[:-1]:
        doc = doc.setdefault(key, {})
    doc[path[-1]] = value


def _get(doc: Dict[str, Any], path: Sequence[str]) -> Any:
    """``doc[path[0]][path[1]]...``, None when a level is missing."""
    for key in path:
        if not isinstance(doc, dict) or key not in doc:
            return None
        doc = doc[key]
    return doc


def describe(args: argparse.Namespace) -> Dict[str, Any]:
    """The trainer yaml ND reads: the template or the asker's yaml, with every field the request states."""
    if args.yaml:
        doc = yaml.safe_load(Path(args.yaml).read_text(encoding="utf-8"))
    else:
        template = SKILL / "templates" / f"{args.family}.yaml"
        if not template.is_file():
            known = sorted(path.stem for path in (SKILL / "templates").glob("*.yaml"))
            raise SystemExit(f"no template for {args.family!r} (templates: {known}); "
                             "pass the asker's trainer yaml with --yaml, after the model reader checked it")
        doc = yaml.safe_load(template.read_text(encoding="utf-8"))
        doc["model"]["config_path"] = (template.parent / doc["model"]["config_path"]).as_posix()
    for name, path in FIELDS.items():
        if getattr(args, name) is not None:
            _set(doc, path, getattr(args, name))
    if len(args.recompute) == 1 and args.recompute[0] in MODES:
        _set(doc, ("activation_checkpoint", "mode"), args.recompute[0])
    return doc


def _status(family: str, cluster: str) -> str:
    """The status table's line for this family on this cluster, or the line for any other pairing."""
    rows = []
    for line in (SKILL / "references" / "status.md").read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if line.startswith("|") and len(cells) >= 5 and not set(cells[0]) <= set("-: "):
            rows.append(cells)
    for wanted in ((family, cluster), (family, "*"), ("*", "*")):
        for cells in rows[1:]:
            if (cells[0].strip("`"), cells[1].strip("`")) == wanted:
                return f"{cells[2]}. {cells[4]}"
    return "No status line for this pairing: treat the ranking as not validated."


def _versions(repo: Path, ratios: Optional[Path]) -> Dict[str, Any]:
    """What the answer depends on besides the request: the ND tree, the libraries, the ratios."""
    def git(*command: str) -> str:
        """What a git command prints in the checkout, empty when git fails."""
        return subprocess.run(["git", "-C", str(repo), *command], capture_output=True, text=True,
                              check=False).stdout.strip()

    def version(name: str) -> str:
        """The installed version of a distribution, or "not installed"."""
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    return {
        "nd_commit": git("rev-parse", "HEAD"),
        "nd_tree_changed": bool(git("status", "--porcelain", "--", "hyper_parallel", "examples")),
        "transformers": version("transformers"),
        "torch": version("torch"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ratios": ratios.as_posix() if ratios else None,
        "ratios_sha256": hashlib.sha256(ratios.read_bytes()).hexdigest() if ratios else None,
    }


def _arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """The request's values, from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default=None,
                        help="template name, for example qwen3_5_moe; with --yaml it only picks the status line")
    parser.add_argument("--yaml", default=None, help="the asker's trainer yaml, instead of the family's template")
    parser.add_argument("--cluster", default="a3", help="cluster key in calibration/clusters.json")
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None, help="tokens a sequence")
    parser.add_argument("--global-batch", dest="global_batch", type=int, default=None, help="sequences a step")
    parser.add_argument("--micro-batch", dest="micro_batch", type=int, default=None, help="sequences a die")
    parser.add_argument("--dies", type=int, default=None, help="total dies")
    parser.add_argument("--recompute", nargs="+", default=["auto"], choices=MODES + ("auto",),
                        help="the activation checkpoint mode the run uses, several to compare, auto for all three")
    parser.add_argument("--budget", default=None, help="memory budget a die, for example 58GB")
    parser.add_argument("--request", default="", help="the request in the asker's words, kept with the run")
    parser.add_argument("--out", default=None, help="run folder (default output/strategy-agent/<time>)")
    parser.add_argument("--timeout", type=int, default=1800, help="seconds ND may take")
    args = parser.parse_args(argv)
    if not (args.family or args.yaml):
        parser.error("give --family, or --yaml with the asker's trainer yaml")
    return args


def _cluster(name: str) -> Dict[str, Any]:
    """The cluster's record, its ratios file as a path; a bare record for a cluster with no calibration."""
    clusters = json.loads((SKILL / "calibration" / "clusters.json").read_text(encoding="utf-8"))
    record = dict(clusters.get(name, {}))
    record["ratios"] = SKILL / "calibration" / record["ratios"] if record.get("ratios") else None
    record.setdefault("device", name.upper())
    return record


def _command(doc: Dict[str, Any], run: Path, recompute: Sequence[str], cluster: Dict[str, Any],
             budget: Optional[str]) -> List[str]:
    """The ND command that prices the run ``run/run.yaml`` describes."""
    command = [
        sys.executable, "-m", "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd", "-f", "hyper_v2",
        "-y", str(run / "run.yaml"), "-d", str(_get(doc, FIELDS["dies"])), "-A", cluster["device"],
        "-b", str(_get(doc, FIELDS["global_batch"])), "-v", "2", "-o", str(run / "nd_out"),
        "--ranking_csv", str(run / "nd_rank.csv"), "--recompute", *recompute,
    ]
    if budget:
        command += ["-M", budget]
    if cluster["ratios"]:
        command += ["-c", str(cluster["ratios"])]
    return command


def _price(command: Sequence[str], repo: Path, run: Path, timeout: int) -> Tuple[int, float, List[str]]:
    """Run ND from the checkout, its log kept in the run folder: exit code, seconds, log lines."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["PYTHONPATH"] = os.pathsep.join([str(repo)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    started = time.monotonic()
    with open(run / "nd.log", "w", encoding="utf-8") as log:
        result = subprocess.run(list(command), cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=timeout, check=False)
    lines = (run / "nd.log").read_text(encoding="utf-8", errors="replace").splitlines()
    return result.returncode, time.monotonic() - started, lines


def _report(args: argparse.Namespace, run: Path, versions: Dict[str, Any], cluster: Dict[str, Any],
            budget: Optional[str], seconds: float, lines: Sequence[str]) -> List[str]:
    """The answer: the status line, what priced it, the card, ND's notices and where the run is kept."""
    notices = [line.strip() for line in lines if any(word in line for word in NOTICES)]
    card = answer.card(answer.Ranking(str(run / "nd_rank.csv")), calibrated=cluster["ratios"] is not None,
                       spread=cluster.get("remeasure_spread_pct"), budget=budget)
    changed = " (with local changes)" if versions["nd_tree_changed"] else ""
    scale = "calibrated" if cluster["ratios"] else "no calibration: relative scores only"
    return [
        f"Status: {_status(args.family or '*', args.cluster)}",
        f"Priced by ND {versions['nd_commit'][:8]}{changed}, Transformers {versions['transformers']}, "
        f"cluster {args.cluster} ({scale}), in {seconds:.0f} s.",
        "",
        card,
        "ND's notices, verbatim:" if notices else "ND printed no notice.",
        *[f"- {line}" for line in notices],
        "",
        f"Run kept in {run}",
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Describe the run, price it with ND, keep everything, print the answer."""
    args = _arguments(argv)
    repo = repo_root()
    cluster = _cluster(args.cluster)
    budget = args.budget or cluster.get("budget")
    doc = describe(args)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run = Path(args.out) if args.out else repo / "output" / "strategy-agent" / stamp
    run.mkdir(parents=True, exist_ok=False)
    (run / "run.yaml").write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (run / "request.md").write_text(args.request + "\n", encoding="utf-8")
    command = _command(doc, run, args.recompute, cluster, budget)
    versions = _versions(repo, cluster["ratios"])
    (run / "versions.json").write_text(json.dumps(versions, indent=2) + "\n", encoding="utf-8")
    (run / "command.txt").write_text(f"cd {repo}\n" + " ".join(command) + "\n", encoding="utf-8")

    code, seconds, lines = _price(command, repo, run, args.timeout)
    if code != 0 or not (run / "nd_rank.csv").is_file():
        sys.stdout.write(f"ND failed (exit {code}) after {seconds:.0f} s; the last lines of {run / 'nd.log'}:\n")
        sys.stdout.write("\n".join(lines[-25:]) + "\n")
        return 1
    report = _report(args, run, versions, cluster, budget, seconds, lines)
    (run / "answer.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    sys.stdout.write("\n".join(report) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
