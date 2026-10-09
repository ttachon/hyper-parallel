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
"""Read the three measurements the next ND round needs, and summarise them in one block.

Each subcommand reads saved output and writes one small JSON result beside it.
Nothing here launches a training run and nothing here ranks a strategy against
ND: the launches are the two shell scripts named in the runbook, and the
comparison with ND belongs off the device.

``indexer``
    Test 1. The same strategy run twice with one environment variable changed,
    so the kernel launch count moves and nothing else does. It answers whether
    device idle follows dispatch pressure, which is the one idle mechanism with
    a measured correlation on both models. The check that makes it a controlled
    experiment is the log line the fused path prints about itself: a reference
    leg that still prints it did not test anything.

``fallback``
    Test 2. Whether the host operator fallback is on the critical path. The
    kernel table says the four fallback operators cost about 2870 ms of an
    8089 ms step, and a table of durations cannot say whether the device was
    waiting for them. This takes the start times as well, and measures how much
    of the fallback runs while no device kernel does. It needs no new run.

``ep16``
    Test 3. The one hole in the compute model's evidence: every honest step time
    is at expert degree 1 to 8, and degree 16 is where the routing carries
    4270 ms of a 9411 ms step. It splits the instrument between the parts and
    the residual, in the shape of nd_golden/sweeps/b8_rederived_1002.csv. One
    sweep run is enough, because the harness harvests the trainer's own step
    over the steps after the profiling window, which is a same-run pair.

``summary``
    One compact block over whatever results exist, ready to paste back, naming
    what is missing rather than leaving a gap. Run it last, or at any point to
    see how far the window got.

Two rules this script holds to, both of them measured rather than preferred. A
quantity's LEVEL is read against the profiler's own columns and its RANKING
against the trainer's own step time, and the two are never mixed inside one
number. And a profiled measurement is paired with an unprofiled one only inside
one round, because the profiled side alone moves 4.6% between rounds.
"""
import argparse
import collections
import csv
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Counter, Dict, Iterable, List, Optional, Sequence, Tuple

# The fused Lightning-Indexer states this once per run, in the run's own log.
# Its absence is what identifies the reference leg of test 1, and its presence
# in both legs is what says the test did not run.
_ENGAGED = "fused lightning-indexer path engaged"

# Device idle against the kernel launch count, fitted over the seven V4.1
# points of 8 October: Free = _IDLE_INTERCEPT_MS + _IDLE_PER_LAUNCH_US x launches.
# Test 1 moves the launch count on one strategy and checks the slope locally.
_IDLE_INTERCEPT_MS = -1275.0
_IDLE_PER_LAUNCH_US = 28.7

# How a kernel table's accelerator core value is classified. Anything that is
# neither the host queue nor a collective counts as device compute, so a core
# name nobody has seen yet is included rather than dropped in silence.
_HOST_CORE = "ai_cpu"
_COMM_CORE = "comm"

# Directory names every Ascend profile ends with, which say nothing about which
# run produced it. Stripped before a profile is labelled.
_GENERIC_PROFILE_DIRS = ("ascend_profiler_output", "profile", "profiles")

# The sweep's own dimension columns, which together identify a strategy.
_KEY = ("DP", "MP", "PP", "CP", "EP", "MB", "MBS", "OP")

# The run roots the runbook tells you to use for test 1, relative to --root.
# They exist so that reading a finished test is one short command: a line that
# wraps when it is copied off a page is a line that will be run wrong.
_LEG_FUSED = "t1_fused"
_LEG_REFERENCE = "t1_ref"
_LEG_FUSED_PROFILE = "t1_fp"
_LEG_REFERENCE_PROFILE = "t1_rp"

# A parsed kernel: start, end, lower-case core, name. Starts and ends are in
# milliseconds and relative to the first kernel of the table.
_Span = Tuple[float, float, str, str]


def _pick(names: Sequence[str], *words: str) -> Optional[str]:
    """Return the first column whose name contains every word, ignoring case.

    Ascend spells these columns differently between releases, so every reader
    here selects by substring and reports what it selected.

    Args:
        names: The column names available.
        *words: Lower-case words that must all appear in the name.

    Returns:
        The matching column name, or None when nothing matches.
    """
    return next((n for n in names if all(w in n.lower() for w in words)), None)


def _to_ms(column: str) -> float:
    """Return the divisor that converts the named column's unit to milliseconds.

    Args:
        column: A column name, which normally carries its unit, as in
            ``Duration(us)``.

    Returns:
        The divisor. Microseconds are assumed when the name states no unit,
        which is what every table seen so far uses.
    """
    lowered = column.lower()
    if "(ns" in lowered or "_ns" in lowered:
        return 1e6
    if "(ms" in lowered or "_ms" in lowered:
        return 1.0
    return 1e3


def _read_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV into a list of dictionaries, tolerating a byte order mark."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _find_one(root: Path, name: str) -> Optional[Path]:
    """Find a single named file anywhere under a directory.

    Args:
        root: The directory to search, or the file itself.
        name: The file name to look for.

    Returns:
        The newest match, or None. The newest is taken rather than the first so
        that a directory reused across runs gives the current profile, which is
        the trap that once classified a stale profile as if it were fresh.
    """
    if root.is_file():
        return root if root.name == name else None
    matches = sorted(root.rglob(name), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _float(value: Any) -> Optional[float]:
    """Parse a float, returning None for anything that is not one."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) else parsed


def _merge(intervals: Iterable[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Merge overlapping intervals into a sorted, disjoint cover.

    Several streams run at once, so device compute intervals overlap each other
    freely. Merging first is what makes the uncovered time a measure of the
    device being idle rather than a sum of per stream gaps.

    Args:
        intervals: Pairs of start and end, in any order.

    Returns:
        Disjoint intervals in increasing order of start.
    """
    merged: List[Tuple[float, float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _uncovered(start: float, end: float, cover: Sequence[Tuple[float, float]]) -> float:
    """Return how much of one interval no interval of the cover overlaps.

    Args:
        start: The interval's start.
        end: The interval's end.
        cover: Disjoint intervals in increasing order, as ``_merge`` returns.

    Returns:
        The uncovered duration, in the units the inputs use.
    """
    exposed = 0.0
    position = start
    for low, high in cover:
        if high <= position:
            continue
        if low >= end:
            break
        if low > position:
            exposed += min(low, end) - position
        position = max(position, high)
        if position >= end:
            return exposed
    return exposed + max(0.0, end - position)


def _span_total(intervals: Sequence[Tuple[float, float]]) -> float:
    """Sum the lengths of intervals, as given and without merging."""
    return sum(end - start for start, end in intervals)


def _spread_pct(values: Sequence[float]) -> float:
    """Return the spread across values as a percentage of their mean."""
    if len(values) < 2:
        return 0.0
    return round(100 * (max(values) - min(values)) / statistics.fmean(values), 2)


def _write_result(root: Path, name: str, payload: Dict[str, Any]) -> Path:
    """Write one test's result as JSON, and say where it went.

    Args:
        root: The results directory, created when missing.
        name: The test's name, which becomes the file name.
        payload: What to record.

    Returns:
        The path written.
    """
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {path}")
    return path


# --------------------------------------------------------------------------- #
# Test 1: the indexer path
# --------------------------------------------------------------------------- #

def _engaged_states(root: Path, runs: Sequence[Dict[str, str]]) -> Tuple[List[Optional[bool]], List[str]]:
    """Report which runs printed the fused indexer's own log line.

    Args:
        root: The leg's run root.
        runs: The successful rows of its runs.csv.

    Returns:
        One state per run, True, False, or None where the log was unreadable,
        and the log paths looked at.
    """
    engaged: List[Optional[bool]] = []
    logs: List[str] = []
    for row in runs:
        log = root / row["tag"] / "run.log"
        if not log.is_file():
            # The repeat script records the path relative to the tree root.
            log = Path(row.get("log", ""))
        logs.append(str(log))
        if log.is_file():
            engaged.append(_ENGAGED in log.read_text(encoding="utf-8", errors="replace"))
        else:
            engaged.append(None)
    return engaged, logs


def _leg_stats(runs: Sequence[Dict[str, str]], steps: Sequence[float]) -> Dict[str, Any]:
    """Summarise one leg's run means, step times and peaks.

    Args:
        runs: The successful rows of a leg's runs.csv.
        steps: Every clean step time the leg recorded.

    Returns:
        The step mean over all repeats, the range and spread of the run means,
        and the highest peak any run reported.
    """
    run_means = [mean for row in runs if (mean := _float(row.get("mean_ms")))]
    peaks = [peak for row in runs if (peak := _float(row.get("peak_alloc_gb")))]
    return {
        "runs": len(runs),
        "steps": len(steps),
        "mean_ms": round(statistics.fmean(steps), 1),
        "run_mean_min_ms": round(min(run_means), 1) if run_means else None,
        "run_mean_max_ms": round(max(run_means), 1) if run_means else None,
        "spread_pct": _spread_pct(run_means),
        "peak_alloc_gb": round(max(peaks), 3) if peaks else None,
    }


def _read_repeat_leg(root: Path) -> Dict[str, Any]:
    """Read one leg of the repeat script's output.

    Args:
        root: A RUN_ROOT the repeat script wrote, holding runs.csv and steps.csv.

    Returns:
        The leg's step times, their spread, the peaks, and which runs printed
        the fused indexer's own log line.
    """
    runs_csv, steps_csv = root / "runs.csv", root / "steps.csv"
    if not runs_csv.is_file() or not steps_csv.is_file():
        raise SystemExit(f"not a repeat-script run root, runs.csv or steps.csv is missing: {root}")

    runs = [row for row in _read_rows(runs_csv) if row.get("status") == "ok"]
    steps = [value for row in _read_rows(steps_csv) if (value := _float(row.get("step_time_ms")))]
    if not runs or not steps:
        raise SystemExit(f"no successful run in {root}: nothing to compare")

    engaged, logs = _engaged_states(root, runs)
    return {"root": str(root), **_leg_stats(runs, steps), "fused_engaged": engaged, "logs": logs}


def _profiled_steps(trace: Path) -> Optional[int]:
    """Count the steps a profile actually covers, from its own step trace.

    The window is ``[start_step, end_step)``, so it is one step on the V4.1
    launcher's defaults and two on the Qwen3.5 sweep's. Guessing it halves or
    doubles every per-step figure derived from a kernel count, so it is counted
    here rather than passed in.

    Args:
        trace: A step_trace_time.csv.

    Returns:
        The number of distinct steps in it, or None when there is no step
        column to count.
    """
    rows = _read_rows(trace)
    names = list(rows[0].keys()) if rows else []
    column = next((n for n in names if n.strip().lower() == "step"), None) or _pick(names, "step")
    if column is None:
        return None
    steps = {row[column] for row in rows if row.get(column) not in (None, "")}
    return len(steps) or None


def _stage_columns(trace: Path) -> Dict[str, Any]:
    """Read the Computing, Free and Stage columns of a step trace, per step."""
    rows = _read_rows(trace)
    names = list(rows[0].keys()) if rows else []
    out: Dict[str, Any] = {"step_trace_time": str(trace)}
    for key in ("computing", "free", "stage"):
        column = _pick(names, key)
        if column is None:
            out[f"{key}_note"] = f"no {key} column in {trace.name}: {names}"
            continue
        values = [value for row in rows if (value := _float(row.get(column))) is not None]
        if values:
            out[f"{key}_ms"] = round(statistics.fmean(values) / _to_ms(column), 1)
    return out


def _read_profile_parts(profile: Optional[Path], steps: int) -> Dict[str, Any]:
    """Read the launch count and the stage decomposition of one profile.

    Args:
        profile: A directory holding an Ascend profiler output, or None.
        steps: How many steps the profile covers, used to report per step.

    Returns:
        The launch count and the Computing, Free and Stage columns per step, or
        an explanation of what was missing.
    """
    if profile is None:
        return {"state": "not given"}
    out: Dict[str, Any] = {"state": "read", "profile": str(profile)}

    trace = _find_one(profile, "step_trace_time.csv")
    counted = _profiled_steps(trace) if trace is not None else None
    out["steps_covered"] = counted or steps
    out["steps_source"] = "counted in step_trace_time.csv" if counted else "--steps, not counted"

    kernels = _find_one(profile, "kernel_details.csv")
    if kernels is None:
        out["launches"] = None
        out["launches_note"] = f"no kernel_details.csv under {profile}"
    else:
        out["launches"] = round(len(_read_rows(kernels)) / out["steps_covered"])
        out["kernel_details"] = str(kernels)

    if trace is None:
        out["stage_note"] = f"no step_trace_time.csv under {profile}"
    else:
        out.update(_stage_columns(trace))
    return out


def _path_verdict(fused: Dict[str, Any],
                  reference: Dict[str, Any]) -> Tuple[str, str, List[bool], List[bool]]:
    """Judge whether the two legs really took different indexer paths.

    Args:
        fused: The leg expected to use the fused indexer.
        reference: The leg expected to have been pinned to the reference path.

    Returns:
        The verdict, a note explaining anything but ``ok``, and the readable
        states of each leg.
    """
    fused_says = [state for state in fused["fused_engaged"] if state is not None]
    ref_says = [state for state in reference["fused_engaged"] if state is not None]
    if not fused_says or not ref_says:
        return ("unproven", "a run's log could not be read, so the path taken is not established",
                fused_says, ref_says)
    if not all(fused_says):
        return ("void", "the fused leg did not print the engaged line in every run, so it fell back "
                        "to the reference path and both legs are the same measurement",
                fused_says, ref_says)
    if any(ref_says):
        return ("void", "the reference leg printed the engaged line, so V41_DISABLE_FUSED_INDEXER did "
                        "not reach the workers and both legs are the same measurement",
                fused_says, ref_says)
    return "ok", "", fused_says, ref_says


def _report_slope(fused_parts: Dict[str, Any], ref_parts: Dict[str, Any],
                  verdict: str) -> Dict[str, Any]:
    """Print the launch count against idle, and return what it adds to the result.

    The slope is only a measurement when the two legs really ran different
    code. On a void pair it would divide a real idle difference by a launch
    difference that is pure noise, and the quotient looks like a finding. So
    the numbers are printed either way and the slope is computed only when the
    experiment actually happened.
    """
    launches = [parts.get("launches") for parts in (fused_parts, ref_parts)]
    idles = [parts.get("free_ms") for parts in (fused_parts, ref_parts)]
    if any(value is None for value in launches + idles):
        print("\nno profiled pair given, so the launch count and idle are not available.")
        print("  Add --profile-fused and --profile-reference once the two profiled runs exist:")
        print("  the honest step above is the ranking half, the parts are the level half, "
              "and neither substitutes for the other.")
        return {}

    launch_delta = launches[1] - launches[0]
    idle_delta = idles[1] - idles[0]
    for label, parts in (("fused", fused_parts), ("reference", ref_parts)):
        print(f"\n{label} profile: {parts['steps_covered']} step(s), {parts['steps_source']}")
    print(f"launches a step: fused {launches[0]:,}, reference {launches[1]:,}, "
          f"{launch_delta:+,} ({100 * launch_delta / launches[0]:+.1f}%)")
    print(f"device idle a step: fused {idles[0]:.1f} ms, reference {idles[1]:.1f} ms, {idle_delta:+.1f} ms")
    added: Dict[str, Any] = {"launch_delta": launch_delta, "idle_delta_ms": round(idle_delta, 1)}

    if verdict != "ok":
        print("\n  NO SLOPE. The two legs ran the same code, so the launch difference above is "
              "run to run\n  variation and a slope computed from it would be an artefact, not a "
              "result. The idle\n  difference is real and unexplained, but this pair cannot say "
              "what explains it.")
        return added
    if abs(launch_delta) < 0.01 * launches[0]:
        print("\n  NO SLOPE. The launch count moved by under 1%, which is inside run to run "
              "variation,\n  so this pair cannot test the slope whatever the idle did.")
        return added

    local = 1e3 * idle_delta / launch_delta
    predicted = _IDLE_PER_LAUNCH_US * launch_delta / 1e3
    print(f"\n  locally that is {local:.1f} us of idle a launch, against "
          f"{_IDLE_PER_LAUNCH_US:.1f} us from the fit over the seven points")
    print(f"  the fit predicts {predicted:+.1f} ms of the {idle_delta:+.1f} ms measured")
    if local * _IDLE_PER_LAUNCH_US <= 0:
        print("  the local slope has the OPPOSITE sign to the fit, which is the outcome that would "
              "retire the dispatch reading rather than confirm it")
    added["idle_us_per_launch"] = round(local, 1)
    added["idle_predicted_ms"] = round(predicted, 1)
    return added


def _indexer_legs(args: argparse.Namespace) -> Tuple[Path, Path, Optional[Path], Optional[Path]]:
    """Resolve the four directories test 1 reads.

    ``--root`` is the runbook's layout, so the command that reads a finished
    test is short enough to copy off a page without wrapping. The explicit
    flags override it one at a time, for a run that was put somewhere else.

    Args:
        args: The parsed arguments.

    Returns:
        The fused leg, the reference leg, and the two profiles, which are None
        when neither given nor present under the root.

    Raises:
        SystemExit: When neither a root nor both legs are given.
    """
    root = Path(args.root) if args.root else None
    if root is None and not (args.fused and args.reference):
        raise SystemExit("give --root, or both --fused and --reference")

    def leg(explicit: Optional[str], name: str, required: bool) -> Optional[Path]:
        """Resolve one directory: the flag if given, else the root's own name."""
        if explicit:
            return Path(explicit)
        candidate = root / name if root else None
        if candidate is not None and candidate.is_dir():
            return candidate
        if required:
            raise SystemExit(f"no {name} directory under {root}: give --{name.split('_')[1]} instead")
        return None

    return (leg(args.fused, _LEG_FUSED, True), leg(args.reference, _LEG_REFERENCE, True),
            leg(args.profile_fused, _LEG_FUSED_PROFILE, False),
            leg(args.profile_reference, _LEG_REFERENCE_PROFILE, False))


def cmd_indexer(args: argparse.Namespace) -> None:
    """Compare the fused and reference indexer legs of one strategy."""
    fused_dir, reference_dir, fused_profile, ref_profile = _indexer_legs(args)
    fused = _read_repeat_leg(fused_dir)
    reference = _read_repeat_leg(reference_dir)
    fused_parts = _read_profile_parts(fused_profile, args.steps)
    ref_parts = _read_profile_parts(ref_profile, args.steps)

    print("=" * 78)
    print("Test 1: the indexer path, one strategy, one environment variable changed")
    print("=" * 78)
    print(f"\n{'leg':12s} {'runs':>5s} {'steps':>6s} {'mean ms':>10s} {'min':>9s} {'max':>9s} "
          f"{'spread':>8s} {'peak GiB':>9s}")
    for label, leg in (("fused", fused), ("reference", reference)):
        print(f"{label:12s} {leg['runs']:5d} {leg['steps']:6d} {leg['mean_ms']:10.1f} "
              f"{leg['run_mean_min_ms']:9.1f} {leg['run_mean_max_ms']:9.1f} "
              f"{leg['spread_pct']:7.2f}% {leg['peak_alloc_gb'] or 0:9.3f}")

    # The experiment is void unless the two legs really took different paths, so
    # this is settled before any number derived from them is printed.
    verdict, note, fused_says, ref_says = _path_verdict(fused, reference)
    print(f"\npath check: {verdict}")
    print(f"  fused leg printed '{_ENGAGED}' in {sum(1 for s in fused_says if s)}/{len(fused_says)} runs")
    print(f"  reference leg printed it in {sum(1 for s in ref_says if s)}/{len(ref_says)} runs")
    if note:
        print(f"  {note}")

    step_delta = reference["mean_ms"] - fused["mean_ms"]
    worst_spread = max(fused["spread_pct"], reference["spread_pct"])
    print(f"\nhonest step: reference is {step_delta:+.1f} ms against fused, "
          f"{100 * step_delta / fused['mean_ms']:+.2f}%")
    print(f"  the worse of the two spreads is {worst_spread:.2f}%, so a delta inside that is not resolved")

    payload: Dict[str, Any] = {
        "test": "indexer", "verdict": verdict, "note": note,
        "fused": fused, "reference": reference,
        "fused_profile": fused_parts, "reference_profile": ref_parts,
        "step_delta_ms": round(step_delta, 1),
        "step_delta_pct": round(100 * step_delta / fused["mean_ms"], 2),
        "worst_spread_pct": worst_spread,
    }
    payload.update(_report_slope(fused_parts, ref_parts, verdict))
    _write_result(Path(args.results), "indexer", payload)


# --------------------------------------------------------------------------- #
# Test 2: the host operator fallback
# --------------------------------------------------------------------------- #

def _parse_kernels(rows: Sequence[Dict[str, str]], start_col: str, dur_col: str,
                   core_col: str, name_col: Optional[str]) -> List[_Span]:
    """Parse the columns the overlap needs, dropping rows that lack them.

    Starts are shifted to the first kernel before any arithmetic, because these
    timestamps are sixteen digits of microseconds, where a float keeps
    microsecond precision only once the common part is removed.
    """
    start_scale, dur_scale = _to_ms(start_col), _to_ms(dur_col)
    raw: List[Tuple[float, float, str, str]] = []
    for row in rows:
        start, duration = _float(row.get(start_col)), _float(row.get(dur_col))
        if start is None or duration is None:
            continue
        raw.append((start / start_scale, duration / dur_scale,
                    (row.get(core_col) or "").lower(), row.get(name_col or "") or ""))
    if not raw:
        return []
    origin = min(item[0] for item in raw)
    return [(start - origin, start - origin + duration, core, name)
            for start, duration, core, name in raw]


def _classify(spans: Sequence[_Span]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]],
                                               float, Counter[str]]:
    """Split kernels into host fallback, device compute and collectives.

    Returns:
        The host intervals, the device compute intervals, the summed collective
        milliseconds, and the host milliseconds by kernel name.
    """
    host: List[Tuple[float, float]] = []
    compute: List[Tuple[float, float]] = []
    comm_ms = 0.0
    by_name: Counter[str] = collections.Counter()
    for start, end, core, name in spans:
        if _HOST_CORE in core:
            host.append((start, end))
            by_name[name] += end - start
        elif _COMM_CORE in core:
            comm_ms += end - start
        else:
            compute.append((start, end))
    return host, compute, comm_ms, by_name


def _find_all(root: Path, name: str) -> List[Path]:
    """Find every named file under a directory, newest first.

    Test 2 takes a directory and measures everything in it rather than asking
    for one path: a profile picked by hand is a profile that can be the wrong
    one, and every result is labelled with the file it came from anyway.
    """
    if root.is_file():
        return [root] if root.name == name else []
    return sorted(root.rglob(name), key=lambda p: p.stat().st_mtime, reverse=True)


def _fallback_one(kernels: Path, steps: int) -> Optional[Dict[str, Any]]:
    """Measure one kernel table's exposed host fallback, printing as it goes."""
    profile = kernels.parent
    rows = _read_rows(kernels)
    names = list(rows[0].keys()) if rows else []
    start_col = _pick(names, "start") or _pick(names, "timestamp")
    dur_col, core_col = _pick(names, "duration"), _pick(names, "core")
    name_col = _pick(names, "name")
    print(f"\nfile:    {kernels}")
    print(f"columns: start={start_col!r} duration={dur_col!r} core={core_col!r} name={name_col!r}")
    if not start_col or not dur_col or not core_col:
        print("  REFUSED. This measurement needs a start time, a duration and a core per kernel.")
        print(f"  The columns in the file are: {names}")
        print("  Report this rather than working around it: the arithmetic is void without start "
              "times, and the existing per core sums already cover durations alone.")
        return None

    spans = _parse_kernels(rows, start_col, dur_col, core_col, name_col)
    if not spans:
        print("  REFUSED. No row carried both a start time and a duration.")
        return None

    host, compute, comm_ms, by_name = _classify(spans)
    cover = _merge(compute)
    host_wall = _span_total(_merge(host))
    exposed = sum(_uncovered(start, end, cover) for start, end in host)
    # The window is [start_step, end_step), which is ONE step on the V4.1
    # launcher's defaults and two on the Qwen3.5 sweep's. Counting it from the
    # profile's own trace is what stops every figure here being halved.
    trace = _find_one(profile, "step_trace_time.csv")
    counted = _profiled_steps(trace) if trace is not None else None
    used = counted or steps
    stage = _stage_columns(trace) if trace is not None else {}
    result = {
        "profile": str(profile), "kernel_details": str(kernels), "kernels": len(rows),
        "steps_covered": used,
        "steps_source": "counted in step_trace_time.csv" if counted else "--steps, not counted",
        "host_summed_ms": round(_span_total(host) / used, 1),
        "host_wall_ms": round(host_wall / used, 1),
        "host_exposed_ms": round(exposed / used, 1),
        "host_exposed_share_pct": round(100 * exposed / host_wall, 1) if host_wall else 0.0,
        "device_compute_wall_ms": round(_span_total(cover) / used, 1),
        "comm_summed_ms": round(comm_ms / used, 1),
        "computing_ms": stage.get("computing_ms"),
        "free_ms": stage.get("free_ms"),
        "top_operators": {name: round(value / used, 1) for name, value in by_name.most_common(6)},
    }
    _print_fallback(result)
    return result


def _print_fallback(result: Dict[str, Any]) -> None:
    """Print one profile's fallback measurement, against the right denominator."""
    print(f"\n  over {result['kernels']:,} kernels and {result['steps_covered']} step(s)"
          f" ({result['steps_source']}), per step:")
    print(f"    host fallback, summed        {result['host_summed_ms']:9.1f} ms  (occupancy)")
    print(f"    host fallback, wall clock    {result['host_wall_ms']:9.1f} ms  (its own union)")
    print(f"    of that, no cube or vector   {result['host_exposed_ms']:9.1f} ms  <- the exposed share")
    print(f"    cube and vector, wall clock  {result['device_compute_wall_ms']:9.1f} ms")
    print(f"    collectives, summed          {result['comm_summed_ms']:9.1f} ms")
    print(f"\n    {result['host_exposed_share_pct']:.0f}% of the fallback runs with no cube or "
          "vector kernel beside it")

    computing = result.get("computing_ms")
    if computing:
        print(f"    against Computing of {computing:.1f} ms a step, that is "
              f"{100 * result['host_exposed_ms'] / computing:.0f}% of it")
    if result.get("free_ms"):
        print(f"    Free is {result['free_ms']:.1f} ms a step and is NOT the denominator: the")
        print("    profiler counts this fallback inside Computing, so it is device work that")
        print("    starves the matrix and vector units, not device idle.")

    print("\n  top fallback operators, summed ms a step:")
    for name, value in result["top_operators"].items():
        print(f"    {name[:52]:52s} {value:9.1f}")


def cmd_fallback(args: argparse.Namespace) -> None:
    """Measure how much of the host operator fallback runs with the device idle."""
    print("=" * 78)
    print("Test 2: is the host operator fallback on the critical path")
    print("=" * 78)

    tables: List[Path] = []
    for raw in args.profile:
        found = _find_all(Path(raw), "kernel_details.csv")
        if not found:
            print(f"\n{raw}: no kernel_details.csv anywhere under it")
        tables.extend(path for path in found if path not in tables)
    if not tables:
        raise SystemExit("\nno kernel table was found, so nothing was measured")
    print(f"\n{len(tables)} kernel table(s) found, measuring every one of them.")

    results = [result for table in tables
               if (result := _fallback_one(table, args.steps)) is not None]
    if not results:
        raise SystemExit("\nno kernel table could be read, so nothing was measured")

    print("\nRead it as evidence and not as proof: a host operator with no device kernel beside it is "
          "\nconsistent with the device waiting for it, and only the timeline proves an ordering.")
    _write_result(Path(args.results), "fallback", {"test": "fallback", "profiles": results})


# --------------------------------------------------------------------------- #
# Test 3: the expert degree 16 pair
# --------------------------------------------------------------------------- #

def _strategy_key(row: Dict[str, str]) -> Optional[str]:
    """Build a strategy's identity from the sweep's own dimension columns."""
    if any(column not in row for column in _KEY):
        return None
    return " ".join(f"{column}{row[column]}" for column in _KEY)


def _label(key: str) -> str:
    """Shorten a strategy key to the expert and optimizer degrees a reader knows."""
    found = dict(re.findall(r"([A-Z]+)(\d+)", key))
    return f"EP{found.get('EP', '?')}/OP{found.get('OP', '?')}"


def _exposed_total(row: Dict[str, str]) -> float:
    """Sum a sweep row's wait columns.

    Every wait the classifier files is exposed communication by construction:
    what overlapped the compute is not in these columns.
    """
    total = 0.0
    for column, raw in row.items():
        if column.endswith("_wait"):
            total += _float(raw) or 0.0
    return total


def _pair_row(prof: Dict[str, str], unprof: Dict[str, str], key: str) -> Optional[Dict[str, Any]]:
    """Pair one strategy's profiled and unprofiled rows into the split's shape."""
    profiled_ms = _float(prof.get("time"))
    # The honest clock is the separate unprofiled run's own step. Where the sweep
    # recorded the trainer's step inside the profiled run too, that is a weaker
    # reading of the same thing, so it is only a fallback.
    honest_ms = _float(unprof.get("step_trainer")) or _float(unprof.get("time"))
    comp_ms = _float(prof.get("comp"))
    if profiled_ms is None or honest_ms is None or comp_ms is None:
        return None
    exposed_ms = _exposed_total(prof)
    return {
        "strategy": _label(key), "key": key,
        "profiled_ms": round(profiled_ms, 1), "trainer_ms": round(honest_ms, 1),
        "instrument_ms": round(profiled_ms - honest_ms, 1), "comp_ms": round(comp_ms, 1),
        "exposed_ms": round(exposed_ms, 1),
        "residual_profiled_ms": round(profiled_ms - comp_ms - exposed_ms, 1),
        "residual_trainer_ms": round(honest_ms - comp_ms - exposed_ms, 1),
    }


def _honest_rows(profiled: Dict[str, Dict[str, str]], unprofiled_path: Optional[str],
                 profiled_path: str) -> Tuple[Dict[str, Dict[str, str]], str]:
    """Resolve the rows carrying the honest clock, and say where they came from.

    One sweep run carries both clocks, because the harness harvests the
    trainer's own step over the steps AFTER the profiling window. That is a
    same-run pair, which is the only sound kind: the one strategy measured both
    ways reads 6276.0 against 6275.2 ms, 0.013% apart. A second file is accepted
    for a round that really was launched twice.
    """
    if unprofiled_path:
        rows = {key: row for row in _read_rows(Path(unprofiled_path)) if (key := _strategy_key(row))}
        return rows, unprofiled_path
    return profiled, f"{profiled_path} (its own step_trainer column)"


def cmd_ep16(args: argparse.Namespace) -> None:
    """Split the profiler's own cost between the parts and the residual."""
    profiled = {key: row for row in _read_rows(Path(args.profiled)) if (key := _strategy_key(row))}
    unprofiled, source = _honest_rows(profiled, args.unprofiled, args.profiled)
    shared = sorted((key for key in profiled if key in unprofiled), key=_label)
    if not shared:
        raise SystemExit("no strategy appears in both files: the two sweeps did not cover the same points")

    print("=" * 78)
    print("Test 3: the instrument's split between the parts and the residual")
    print("=" * 78)
    print(f"\nprofiled:   {args.profiled}")
    print(f"honest:     {source}")
    print(f"{len(shared)} strategy(ies) in both\n")
    print(f"{'strategy':12s} {'profiled':>9s} {'honest':>9s} {'instrument':>10s} "
          f"{'comp':>9s} {'exposed':>9s} {'resid prof':>10s} {'resid honest':>12s}")

    rows_out = []
    for key in shared:
        row = _pair_row(profiled[key], unprofiled[key], key)
        if row is None:
            print(f"{_label(key):12s}  incomplete row, skipped")
            continue
        rows_out.append(row)
        print(f"{row['strategy']:12s} {row['profiled_ms']:9.1f} {row['trainer_ms']:9.1f} "
              f"{row['instrument_ms']:10.1f} {row['comp_ms']:9.1f} {row['exposed_ms']:9.1f} "
              f"{row['residual_profiled_ms']:10.1f} {row['residual_trainer_ms']:12.1f}")
    if not rows_out:
        raise SystemExit("no strategy had every column needed")

    negative = [row for row in rows_out if row["residual_trainer_ms"] < 0]
    print(f"\n{len(negative)} of {len(rows_out)} strategies have a NEGATIVE residual against the honest "
          "step,\nwhich is the instrument sitting inside the parts and not only in the remainder.")
    if negative:
        print("  " + ", ".join(f"{row['strategy']} {row['residual_trainer_ms']:+.1f} ms"
                               for row in negative))
    print("\nDo not substitute the honest total into the profiled decomposition. The parts exist only "
          "\nunder the profiler, so only the total can be replaced, which is as far as this goes.")

    out_csv = Path(args.results) / "ep16_pairs.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"\nwrote {out_csv}")
    _write_result(Path(args.results), "ep16",
                  {"test": "ep16", "profiled": args.profiled, "unprofiled": args.unprofiled,
                   "pairs": rows_out})


# --------------------------------------------------------------------------- #
# The summary
# --------------------------------------------------------------------------- #

def _summarise_indexer(data: Dict[str, Any], out: List[str]) -> None:
    """Render test 1 into the summary block."""
    out.append(f"TEST 1  indexer path, dispatch pressure against idle   [{data['verdict']}]")
    if data["verdict"] != "ok":
        out.append(f"  VOID OR UNPROVEN: {data['note']}")
    fused, reference = data["fused"], data["reference"]
    out.append(f"  fused      {fused['mean_ms']:8.1f} ms over {fused['runs']} runs, "
               f"spread {fused['spread_pct']:.2f}%")
    out.append(f"  reference  {reference['mean_ms']:8.1f} ms over {reference['runs']} runs, "
               f"spread {reference['spread_pct']:.2f}%")
    resolved = abs(data["step_delta_pct"]) > data["worst_spread_pct"]
    out.append(f"  step delta {data['step_delta_ms']:+8.1f} ms ({data['step_delta_pct']:+.2f}%), "
               f"{'OUTSIDE' if resolved else 'inside'} the {data['worst_spread_pct']:.2f}% spread")
    if "launch_delta" not in data:
        out.append("  no profiled pair yet, so the launch count and idle are missing")
        return
    out.append(f"  launches   {data['launch_delta']:+8,} a step, idle {data['idle_delta_ms']:+.1f} ms")
    if "idle_us_per_launch" in data:
        out.append(f"  local slope {data['idle_us_per_launch']:.1f} us a launch against "
                   f"{_IDLE_PER_LAUNCH_US} fitted, predicted {data['idle_predicted_ms']:+.1f} ms")


def _profile_label(profile: str) -> str:
    """Name a profile by the run it came from, not by the profiler's own folder.

    Every Ascend profile ends in the same two or three directory names, so a
    label taken from the leaf makes two different runs look identical. This
    walks up to the first names that say which run it was.
    """
    parts = [part for part in Path(profile).parts
             if part.lower() not in _GENERIC_PROFILE_DIRS and not part.lower().endswith("_ascend_pt")]
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return parts[-1] if parts else profile


def _summarise_fallback(data: Dict[str, Any], out: List[str]) -> None:
    """Render test 2 into the summary block."""
    out.append("TEST 2  host operator fallback, is it on the critical path")
    for entry in data["profiles"]:
        label = _profile_label(entry["profile"])
        out.append(f"  {label[:26]:26s} fallback {entry['host_wall_ms']:7.1f} ms a step, "
                   f"exposed {entry['host_exposed_ms']:7.1f} ({entry['host_exposed_share_pct']:.0f}%)")
        computing = entry.get("computing_ms")
        if computing:
            # Free is NOT the denominator: this work is counted inside Computing.
            out.append(f"  {'':26s} {100 * entry['host_exposed_ms'] / computing:.0f}% of Computing "
                       f"({computing:.1f} ms a step), which is the denominator, not Free")


def _summarise_ep16(data: Dict[str, Any], out: List[str]) -> None:
    """Render test 3 into the summary block."""
    out.append("TEST 3  expert degree 16, the instrument split")
    for row in data["pairs"]:
        out.append(f"  {row['strategy']:12s} profiled {row['profiled_ms']:7.1f}  "
                   f"honest {row['trainer_ms']:7.1f}  instrument {row['instrument_ms']:+7.1f}  "
                   f"residual {row['residual_trainer_ms']:+7.1f}")


_RENDERERS = (("indexer", _summarise_indexer, "test 1, the indexer path"),
              ("fallback", _summarise_fallback, "test 2, the host fallback"),
              ("ep16", _summarise_ep16,
               "test 3, the expert degree 16 pair (Qwen3.5, a separate round)"))


def cmd_summary(args: argparse.Namespace) -> None:
    """Print one compact block over every result that exists."""
    root = Path(args.results)
    out: List[str] = ["=" * 78, "ND window tests, summary", "=" * 78]
    missing = []
    for name, render, description in _RENDERERS:
        path = root / f"{name}.json"
        if not path.is_file():
            missing.append(description)
            continue
        out.append("")
        try:
            render(json.loads(path.read_text(encoding="utf-8")), out)
        except (KeyError, ValueError, TypeError) as error:
            out.append(f"  {name}.json could not be read: {error}")

    out.append("")
    out.append("-" * 78)
    out.append("NOT RUN: " + "; ".join(missing) if missing else "All three tests have results.")
    out.append(f"results directory: {root}")
    out.append("Every figure above is derived from saved output in that directory. Nothing here is "
               "ranked against ND, which is done off the device.")

    block = "\n".join(out)
    print(block)
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(block + "\n", encoding="utf-8")
        print(f"\nwrote {destination}")
        print("Send that one file back, or paste the block above.")


def _positive(value: str) -> int:
    """Parse a step count, refusing zero so nothing is divided by it."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("a profile covers at least one step")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="output/nd_window_tests",
                        help="where each test writes its JSON result (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("indexer", help="test 1: the fused against the reference indexer")
    one.add_argument("--root", default=None,
                     help=f"directory holding {_LEG_FUSED}, {_LEG_REFERENCE} and, when they exist, "
                          f"{_LEG_FUSED_PROFILE} and {_LEG_REFERENCE_PROFILE}")
    one.add_argument("--fused", default=None, help="RUN_ROOT of the leg with the fused indexer")
    one.add_argument("--reference", default=None,
                     help="RUN_ROOT of the leg with V41_DISABLE_FUSED_INDEXER=1")
    one.add_argument("--profile-fused", default=None, help="a profiled run of the fused leg")
    one.add_argument("--profile-reference", default=None, help="a profiled run of the reference leg")
    one.add_argument("--steps", type=_positive, default=2,
                     help="steps each profile covers (default: %(default)s)")
    one.set_defaults(func=cmd_indexer)

    two = sub.add_parser("fallback", help="test 2: how much of the host fallback is exposed")
    two.add_argument("profile", nargs="+", help="directories holding Ascend profiler output")
    two.add_argument("--steps", type=_positive, default=2,
                     help="steps each profile covers (default: %(default)s)")
    two.set_defaults(func=cmd_fallback)

    three = sub.add_parser("ep16", help="test 3: pair a profiled sweep with an unprofiled one")
    three.add_argument("--profiled", required=True, help="real_all.csv of the sweep's timing pass")
    three.add_argument("--unprofiled", default=None,
                       help="real_all.csv of a separate unprofiled round; omit it to take the honest "
                            "clock from the step_trainer column of the file above")
    three.set_defaults(func=cmd_ep16)

    four = sub.add_parser("summary", help="one block over every result that exists")
    four.add_argument("--out", default=None, help="also write the block to this file")
    four.set_defaults(func=cmd_summary)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one subcommand."""
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
