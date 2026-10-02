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
"""ND's estimate of measured Qwen3.5-MoE runs, one row each.

Reads the first seven columns of a measurement sheet, ``seq TP CP EP dp_shard
edp_shard recompute`` (``no``, ``off``, ``selective`` or ``full``), separated
by tabs or spaces with a header line allowed, and prints for every run ND's
peak memory in GiB, its step in seconds and its raw score, as columns to set
beside the measured ones.

Each run is priced alone. Its ND input is the demo config as the sweep's rank
stage writes it, with the run's degrees, expert shard and recompute mode, and
``run_nd -mppb`` prices the mode the yaml states. The score turns into seconds
as ND's parts times ratios the sweep's compare stage fitted on the 2026-09-30
round, 8192 tokens with full recompute; they do not hold at short sequences,
whose steps are launch overhead and idle. A run is cached in ``--out`` until
its ND input changes, so adding rows prices only the new ones.

Run it with the trainer's python::

    python examples/training_demo/nd_rows.py --rows rows.tsv
"""
import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional

import yaml

REPO = Path(__file__).resolve().parents[2]
RUN_ND = "hyper_parallel.auto_parallel.sapp_nd.nd.run_nd"
# ND's parts measured over estimated, in ms per unit, as the compare stage fitted
# them on the round of 2026-09-30; a part that round did not exercise takes compute's.
RATIOS_0930 = {"COMPUTE": 6.914483249342451e-11, "DP_COMM": 2.933292889996305e-11,
               "DP_REDUCE": 6.857542887100095e-11, "EP_COMM": 5.393909911314646e-12}
COMPUTE_PARTS = ("FW_COMPUTE", "BW_COMPUTE", "RECOMPUTE")
COMM_PARTS = ("DP_COMM", "DP_REDUCE", "MP_COMM", "EP_COMM", "CP_COMM", "PP_COMM", "BUBBLE")
MODES = {"no": "off", "off": "off", "selective": "selective", "full": "full"}
CSV_HEADER = "DP,MP,PP,CP,EP,MB,MBS,OP,time,comp,dp_wait,mp_wait,ep_wait,cp_wait,pp_wait,op_wait,sp_wait"
TABLE_HEADER = "seq\tTP\tCP\tEP\tFSDP.dp\tFSDP.edp\trecompute\tND memory GiB\tND step s\tND score"


class Run(NamedTuple):
    """One measured run, as the first seven columns of the sheet state it."""

    seq: int
    tp: int
    cp: int
    ep: int
    dp_shard: int
    edp_shard: int
    mode: str


def read_runs(path: Path) -> List[Run]:
    """Read a sheet's runs, skipping any line that does not start with a number.

    Args:
        path: Text file, one run per line, its first seven cells tab or space separated.

    Returns:
        The runs in file order, each recompute mode as the trainer names it.

    Raises:
        ValueError: A run states a recompute mode other than no, off, selective or full.
    """
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = line.split()
        if len(cells) < 7 or not cells[0].isdigit():
            continue
        mode = MODES.get(cells[6].lower())
        if mode is None:
            raise ValueError(f"recompute must be no, off, selective or full, not {cells[6]!r}: {line}")
        runs.append(Run(*(int(cell) for cell in cells[:6]), mode))
    return runs


def nd_config(base: dict, args: argparse.Namespace, run: Run) -> str:
    """ND's input for one run: the config as the rank stage writes it, at the run's degrees and mode.

    The rank stage states ``context.expert_shard: group`` because the sweep's
    launch shards each strategy's experts over their whole group; a run
    measured elsewhere states its own ``edp_shard_size`` instead.

    Args:
        base: The train yaml the runs used, as loaded.
        args: The command line, for the world size, the layers and the batch.
        run: The run to price.

    Returns:
        The yaml text.
    """
    raw = json.loads(json.dumps(base))
    raw["model"] = dict(raw["model"], num_hidden_layers=args.layers)
    raw["model"].pop("validate_placement", None)
    dataset = raw.setdefault("dataset", {})
    dataset["data_config"] = dict(dataset.get("data_config") or {}, seq_length=run.seq)
    raw["activation_checkpoint"] = dict(raw.get("activation_checkpoint") or {}, mode=run.mode)
    raw["training"] = dict(raw.get("training") or {}, global_batch_size=args.gbs, micro_batch_size=args.mbs)
    raw["accelerator"] = dict(raw.get("accelerator") or {}, tp_size=run.tp, cp_size=run.cp, ep_size=run.ep,
                              pp_size=1)
    raw["fsdp_config"] = dict(raw.get("fsdp_config") or {}, dp_shard_size=run.dp_shard,
                              edp_shard_size=run.edp_shard)
    context = dict(raw.get("context") or {}, device_num=args.world, census=True)
    context.pop("expert_shard", None)
    raw["context"] = context
    return yaml.safe_dump(raw, sort_keys=False)


def _price(args: argparse.Namespace, run: Run, out: Path) -> None:
    """Run ``run_nd`` on the ND input in ``out`` for one run, its log beside it."""
    dp = args.world // (run.tp * run.cp)
    micro_batches = max(1, args.gbs // (args.mbs * dp))
    (out / "real_all.csv").write_text(
        f"{CSV_HEADER}\n{dp},{run.tp},1,{run.cp},{run.ep},{micro_batches},{args.mbs},{run.dp_shard},"
        "1000,1000,0,0,0,0,0,0,0\n", encoding="utf-8")
    repo = str(Path(args.repo).resolve())
    pythonpath = os.pathsep.join(path for path in (repo, os.environ.get("PYTHONPATH")) if path)
    cmd = [sys.executable, "-m", RUN_ND, "-y", str(out / "nd_model.yaml"), "-f", "hyper_v2",
           "-d", str(args.world), "-A", args.arch, "-mppb",
           "--real_csv", str(out / "real_all.csv"), "-o", str(out)]
    with open(out / "log.txt", "w", encoding="utf-8") as log:
        subprocess.run(cmd, cwd=repo, env=dict(os.environ, MPLBACKEND="Agg", PYTHONPATH=pythonpath),
                       stdout=log, stderr=subprocess.STDOUT, check=False)


def estimate(args: argparse.Namespace, base: dict, run: Run) -> Optional[Dict[str, str]]:
    """ND's estimate of one run, priced unless ``--out`` already holds it for the same input.

    Args:
        args: The command line.
        base: The train yaml the runs used, as loaded.
        run: The run to price.

    Returns:
        The run's row of ``run_nd``'s ``real_all_estimates.csv``, or None when
        ND could not cost it, in which case its log says why.
    """
    out = Path(args.out).resolve() / (f"s{run.seq}_tp{run.tp}_cp{run.cp}_ep{run.ep}"
                                      f"_dp{run.dp_shard}_edp{run.edp_shard}_{run.mode}")
    out.mkdir(parents=True, exist_ok=True)
    config, estimates = out / "nd_model.yaml", out / "real_all_estimates.csv"
    text = nd_config(base, args, run)
    if not (estimates.is_file() and config.is_file() and config.read_text(encoding="utf-8") == text):
        config.write_text(text, encoding="utf-8")
        estimates.unlink(missing_ok=True)
        _price(args, run, out)
    if not estimates.is_file():
        print(f"ND could not cost {out.name}: see {out / 'log.txt'}", file=sys.stderr)
        return None
    with open(estimates, encoding="utf-8") as table:
        return next(csv.DictReader(table), None)


def step_seconds(row: Dict[str, str], ratios: Dict[str, float]) -> float:
    """ND's step in seconds: each part of its estimate times that part's ratio.

    Args:
        row: A row of ``real_all_estimates.csv``.
        ratios: Milliseconds per unit of ND's estimate, per part.

    Returns:
        The step in seconds.
    """
    ms = sum(float(row[part]) for part in COMPUTE_PARTS) * ratios["COMPUTE"]
    ms += sum(float(row[part]) * ratios.get(part, ratios["COMPUTE"]) for part in COMM_PARTS if part in row)
    return ms / 1000


def _parse_args() -> argparse.Namespace:
    """The command line, with the sweep's own world, layers and batch as defaults."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--rows", type=Path, required=True,
                        help="the runs: seq TP CP EP dp_shard edp_shard recompute, one per line")
    parser.add_argument("--repo", default=str(REPO), help="root of the tree whose ND prices the runs")
    parser.add_argument("--config", default="examples/training_demo/train_qwen3_5_moe.yaml",
                        help="the train yaml the runs used, relative to --repo unless it exists as given")
    parser.add_argument("--world", type=int, default=64)
    parser.add_argument("--arch", default="A3")
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--gbs", type=int, default=64, help="global batch size of the runs")
    parser.add_argument("--mbs", type=int, default=1, help="micro-batch size of the runs")
    parser.add_argument("--ratios", type=Path, help="an nd_ratios.json the compare stage wrote")
    parser.add_argument("--out", default="nd_rows_out")
    return parser.parse_args()


def main() -> None:
    """Print ND's estimate of every run in ``--rows`` and write the table to ``<out>/nd_rows.tsv``."""
    args = _parse_args()
    ratios = RATIOS_0930
    if args.ratios:
        ratios = {name: value for name, value in json.loads(args.ratios.read_text(encoding="utf-8")).items()
                  if not name.startswith("_")}
    config = Path(args.config) if Path(args.config).is_file() else Path(args.repo) / args.config
    base = yaml.safe_load(config.read_text(encoding="utf-8"))
    lines = [TABLE_HEADER]
    print(TABLE_HEADER, flush=True)
    for run in read_runs(args.rows):
        row = estimate(args, base, run)
        cells = "\t".join(str(value) for value in run)
        if row is None:
            lines.append(f"{cells}\tnot costed\t\t")
        else:
            lines.append(f"{cells}\t{int(row['memory_mb']) / 1024:.2f}\t{step_seconds(row, ratios):.3f}\t"
                         f"{float(row['score']):.4e}")
        print(lines[-1], flush=True)
    table = Path(args.out) / "nd_rows.tsv"
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"written {table}")


if __name__ == "__main__":
    main()
