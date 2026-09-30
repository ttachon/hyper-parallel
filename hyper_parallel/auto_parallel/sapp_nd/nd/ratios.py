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
"""Ratios that correct ND's parts from a measured round, in the file ``run_nd -c`` reads.

A round of strategies measured on a cluster and classified into the parts ND
estimates (``nd.trace_classify``) gives, for each part, the ratio of what the
cluster measured to what ND estimated, summed over the round. ND reads the
ratios from a file with ``-c`` (``estimate.apply_regression_coefficients``)
and re-scores every candidate with them: each part times its ratio, summed,
in milliseconds.

``run_nd --real_csv <csv> --write_ratios <json>`` fits them on a comparison,
writes the file, and prints how well they predict each measured strategy
when fitted on the others.
"""
import json
from typing import Dict, List, Optional, Sequence, Tuple

from hyper_parallel.auto_parallel.sapp_nd.nd.debug import PerfParts

# Each ratio of the file: the ND parts it scales and the measured columns it
# is fitted to. The classifier's Ascend path files FSDP's gathers and
# reduce-scatters under op_wait and the all-reduces under dp_wait, and real
# data cannot tell the bubble from P2P time, so one ratio covers both.
FITS: Tuple[Tuple[str, Tuple[PerfParts, ...], Tuple[str, ...]], ...] = (
    ("COMPUTE", (PerfParts.FW_COMPUTE, PerfParts.BW_COMPUTE, PerfParts.RECOMPUTE), ("comp",)),
    ("DP_COMM", (PerfParts.DP_COMM,), ("op_wait",)),
    ("DP_REDUCE", (PerfParts.DP_REDUCE,), ("dp_wait",)),
    ("MP_COMM", (PerfParts.MP_COMM,), ("mp_wait", "sp_wait")),
    ("EP_COMM", (PerfParts.EP_COMM,), ("ep_wait",)),
    ("CP_COMM", (PerfParts.CP_COMM,), ("cp_wait",)),
    ("PP_COMM", (PerfParts.PP_COMM, PerfParts.BUBBLE), ("pp_wait",)),
)

# The parts a score splits into, in the order a comparison lists them.
SCORE_PARTS = [part for part in PerfParts if part not in {PerfParts.TOTAL, PerfParts.MEMORY}]

# The share of ND's whole estimate below which a part is not priced at all:
# a bubble without a pipeline leaves a rounding residue of 1e-16 of it.
NEGLIGIBLE = 1e-9


def label(config) -> str:
    """A strategy's degrees above 1, DP always: ``DP 64 EP 2 OP 32``."""
    return " ".join(f"{dim} {value}" for dim, value in zip(config.keys(), config.values())
                    if value not in ("1", "False") or str(dim) == "DP")


def _estimated(entry: tuple) -> Dict[PerfParts, float]:
    """ND's parts of one comparison entry, by part."""
    return dict(zip(SCORE_PARTS, entry[4]))


def _measured(entry: tuple, columns: Sequence[str]) -> float:
    """The milliseconds one comparison entry measured in *columns*."""
    return sum(entry[5].get(column) or 0.0 for column in columns)


def busy(entry: tuple) -> float:
    """The measured step of a comparison entry less idle: every column a ratio is fitted to."""
    return sum(_measured(entry, columns) for _, _, columns in FITS)


def fit_ratios(entries: Sequence[tuple]) -> Dict[str, float]:
    """Fit one ratio per part, measured over estimated, summed over *entries*.

    A part ND prices at next to nothing over the round, TP where no strategy
    splits tensors or the rounding residue of a bubble without a pipeline,
    takes compute's ratio: it then only turns ND's units into milliseconds, as
    ND weighs every part in one unit. The bubble takes P2P's.

    Args:
        entries: ``(config, peak_mem, real_time, score, parts, real_parts)``, as
            ``ParallelizeLayer.compare_with_csv`` returns them.

    Returns:
        The ratio of each key ``apply_regression_coefficients`` reads.

    Raises:
        ValueError: When ND prices no compute over the round, the one part every
            ratio can fall back on.
    """
    fitted: Dict[str, Optional[float]] = {}
    whole = sum(sum(_estimated(entry).values()) for entry in entries)
    for key, parts, columns in FITS:
        estimated = sum(sum(_estimated(entry)[part] for part in parts) for entry in entries)
        measured = sum(_measured(entry, columns) for entry in entries)
        fitted[key] = measured / estimated if estimated > NEGLIGIBLE * whole else None
    unit = fitted["COMPUTE"]
    if unit is None:
        raise ValueError("ND prices no compute over the round: no ratio can be fitted")
    ratios = {key: unit if value is None else value for key, value in fitted.items()}
    ratios[PerfParts.BUBBLE.name] = ratios[PerfParts.PP_COMM.name]
    return ratios


def corrected(parts: Dict[PerfParts, float], ratios: Dict[str, float]) -> float:
    """ND's estimate of one strategy with *ratios*, in milliseconds, as ``run_nd -c`` scores it."""
    total = 0.0
    for key, keyed_parts, _ in FITS:
        for part in keyed_parts:
            ratio = ratios[PerfParts.BUBBLE.name] if part == PerfParts.BUBBLE else ratios[key]
            total += (parts.get(part) or 0.0) * ratio
    return total


def leave_one_out(entries: Sequence[tuple]) -> List[Tuple[tuple, Optional[float]]]:
    """Each entry with ND's estimate of it corrected by ratios fitted on the others, None alone."""
    entries = list(entries)
    if len(entries) < 2:
        return [(entry, None) for entry in entries]
    return [(entry, corrected(_estimated(entry), fit_ratios(entries[:i] + entries[i + 1:])))
            for i, entry in enumerate(entries)]


def write_ratios(path: str, ratios: Dict[str, float], entries: Sequence[tuple], source: str) -> None:
    """Write *ratios* where ``run_nd -c`` reads them, with the round they were fitted on."""
    record = dict(ratios)
    record["_fitted_on"] = {"csv": source, "strategies": [label(entry[0]) for entry in entries]}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)


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


def report(entries: Sequence[tuple], ratios: Dict[str, float]) -> List[str]:
    """The ratios, each measured strategy corrected in and out of sample, and the corrected ND's pick.

    The corrected estimate leaves out idle, which ND does not price, so it is
    set beside the busy time; the cost of following the corrected ND is
    measured on the whole step.
    """
    lines = ["Ratios, measured milliseconds per ND unit:"]
    lines += [f"  {key:10s} {value:.4e}" for key, value in ratios.items()]
    rows = sorted(((corrected(_estimated(entry), ratios), out, entry) for entry, out in leave_one_out(entries)),
                  key=lambda row: row[0])
    lines.append(f"  {'strategy':22s} {'corrected':>9s} {'left out':>9s} {'busy':>8s} {'error':>7s} {'step':>8s}")
    for estimate, out, entry in rows:
        held = f"{out:9.1f} {busy(entry):8.1f} {out / busy(entry) - 1:+7.1%}" if out is not None else \
            f"{'-':>9s} {busy(entry):8.1f} {'-':>7s}"
        lines.append(f"  {label(entry[0]):22s} {estimate:9.1f} {held} {entry[2]:8.1f}")
    pick, fastest = rows[0][2], min(entries, key=lambda entry: entry[2])
    lines.append(f"Corrected, ND ranks {label(pick[0])} first: it measures {pick[2]:.1f} ms; the fastest, "
                 f"{label(fastest[0])}, {fastest[2]:.1f}: following the corrected ND costs "
                 f"{pick[2] / fastest[2] - 1:.1%}.")
    correlation = spearman([estimate for estimate, _, _ in rows], [entry[2] for _, _, entry in rows])
    if correlation is not None:
        lines.append(f"Rank correlation of the corrected estimate with the measured step: {correlation:+.2f}.")
    return lines
