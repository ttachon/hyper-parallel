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
"""Time the host link with every rank copying at once, and its interference with collectives.

Launch on one node with all of its devices, in the trainer's environment::

    torchrun --nproc_per_node 16 bench_host_link.py --out host_link.csv

or through the cluster kit from the repository root, on the first node only::

    cluster -c examples/training_demo/cluster_qwen3_5_moe.env torchrun -n 1 \\
        examples/training_demo/bench_host_link.py --out output/offload_bench/host_link.csv

Every rank does the same thing at the same moment, as in a training step:

- copy: D2H and H2D between device memory and pinned host memory, sizes from
  ``--min-kib`` to ``--max-mib``, on a side stream as an offload would. With
  ``--copiers``, once for each count of ranks a node that copy while the others
  idle, which tells a die's own link from the node's shared host side.
- alone: at ``--int-mib``, each copy direction and each of ``--collectives`` alone.
- together: a copy timed while a collective runs throughout, and a collective
  timed while copies run throughout, every rank copying.

The collectives span the whole job, or with ``--group-sizes`` groups of that
many consecutive ranks, every group at once, as the trainer lays out its expert
groups (``ep`` is the innermost mesh dimension). ``all_to_all_hp`` is the
trainer's expert exchange rather than the plain one: a counts all_to_all, both
counts read on the host, then a list all_to_all split by them, as
``_prepare_ep_dispatch`` and ``_EPAllToAllUneven`` run it.

A sample times back-to-back calls with device events: no barrier and no host
clock inside the window, a barrier before it aligns the ranks. A sample's time
is the max over ranks, the rank a step waits for. The first ``--warmup`` samples
are dropped and the median over ``--samples`` is reported. Bytes are counted
from the tensors, GiB = 2**30, and a collective row counts its full buffer
(algbw). skew = slowest rank over the median rank; cover = share of a together
window the background part spanned (below 0.95, raise ``--margin``). Rank 0
prints every row and the summary on lines that start with ``HOST_LINK``.

Per collective and copy direction, from the alone and together times,
``f = 2 - (copy_alone / copy_together + comm_alone / comm_together)``
is 0 when the two do not interfere and 1 when they share one resource.
With no ``--group-sizes`` and no ``--copiers`` it runs what the 7 October 2026
runs ran (``nd_golden/offload/bench_host_link.py``, md5 6d66127d).
"""
import argparse
import contextlib
import csv
import datetime
import math
import os
import socket
import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import

MIB, GIB = 2 ** 20, 2 ** 30
PREFIX = "HOST_LINK"
COLLECTIVES = ("all_gather", "reduce_scatter", "all_to_all", "all_to_all_hp")

Run = Callable[[], None]
Part = Tuple[Run, int, Any]
Row = Dict[str, Any]
Pair = Tuple[str, str, float, float, float, float]
Buffers = Tuple[torch.Tensor, torch.Tensor]
Add = Callable[..., Optional[Row]]


def say(line: str) -> None:
    """Print one of rank 0's lines, prefixed so that a node's log can be grepped for them."""
    print(f"{PREFIX} {line}", flush=True)


class HostEvent:
    """A timing event on the host clock, for a dry run on CPU."""

    def __init__(self) -> None:
        """An event not yet recorded."""
        self.t = 0.0

    def record(self) -> None:
        """Note the time."""
        self.t = time.perf_counter()

    def elapsed_time(self, other: "HostEvent") -> float:
        """Milliseconds from this event to *other*."""
        return (other.t - self.t) * 1e3


class Device:
    """The few calls the benchmark needs, on npu or cuda, or on the host for a CPU dry run."""

    def __init__(self, kind: str, local_rank: int) -> None:
        """The device of this rank, *kind* one of npu, cuda and cpu."""
        self.kind, self.acc, self.dev = kind, None, torch.device("cpu")
        if kind != "cpu":
            if kind == "npu":
                import torch_npu  # noqa: F401  pylint: disable=import-outside-toplevel,unused-import
            self.acc = getattr(torch, kind)
            self.acc.set_device(local_rank)
            self.dev = torch.device(kind, local_rank)

    def stream(self) -> Any:
        """A side stream, or none on the host."""
        return self.acc.Stream() if self.acc else None

    def on(self, stream: Any) -> Any:
        """A context that runs on *stream*."""
        return self.acc.stream(stream) if stream is not None else contextlib.nullcontext()

    def event(self) -> Any:
        """A timing event."""
        return self.acc.Event(enable_timing=True) if self.acc else HostEvent()

    def sync(self) -> None:
        """Wait for the device."""
        if self.acc:
            self.acc.synchronize()


def sample(dev: Device, parts: Sequence[Part]) -> List[Tuple[float, float]]:
    """One aligned sample of parts = [(run, reps, stream)], enqueued in that order.

    Returns each part's (start, end) in ms after an anchor event that every part waits for.
    """
    dist.barrier()
    dev.sync()
    anchor = dev.event()
    anchor.record()
    marks = []
    for run, reps, stream in parts:
        with dev.on(stream):
            if stream is not None:
                stream.wait_event(anchor)
            start, end = dev.event(), dev.event()
            start.record()
            for _ in range(reps):
                run()
            end.record()
        marks.append((start, end))
    dev.sync()
    return [(anchor.elapsed_time(s), anchor.elapsed_time(e)) for s, e in marks]


def measure(dev: Device, parts: Sequence[Part], args: argparse.Namespace) -> Optional[List[Any]]:
    """Warmup plus samples; on rank 0 returns [sample][rank][part] = (start, end) in ms."""
    kept = [sample(dev, parts) for _ in range(args.warmup + args.samples)][args.warmup:]
    mine = torch.tensor(kept, dtype=torch.float32, device=dev.dev)
    every = [torch.empty_like(mine) for _ in range(dist.get_world_size())]
    dist.all_gather(every, mine)
    if dist.get_rank():
        return None
    by_rank = torch.stack(every).cpu().tolist()
    return [[ranks[s] for ranks in by_rank] for s in range(len(kept))]


def agreed_seconds(dev: Device, run: Run, stream: Any) -> float:
    """Seconds of one call after a warmup call, the max over ranks, so every rank agrees."""
    sample(dev, [(run, 1, stream)])
    start, end = sample(dev, [(run, 1, stream)])[0]
    t = torch.tensor([(end - start) / 1e3], dtype=torch.float32, device=dev.dev)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return max(t.item(), 1e-6)


def row(test: str, op: str, nbytes: int, reps: int, windows: List[Any], part: int = 0,
        cover_by: Optional[int] = None, only: Optional[Sequence[int]] = None) -> Row:
    """One part over the samples: per sample the max over ranks of seconds a call, then the median.

    Where *only* is given, only those ranks count: the ones that ran the part.
    """
    worst, skew, cover = [], [], []
    for ranks in windows:
        ranks = ranks if only is None else [ranks[r] for r in only]
        secs = [max(r[part][1] - r[part][0], 1e-6) / 1e3 / reps for r in ranks]
        worst.append(max(secs))
        skew.append(max(secs) / statistics.median(secs))
        if cover_by is not None:
            for r in ranks:
                (s, e), (bs, be) = r[part], r[cover_by]
                cover.append(max(0.0, min(e, be) - max(s, bs)) / max(e - s, 1e-6))
    med = statistics.median(worst)
    return {"test": test, "op": op, "bytes": nbytes, "reps": reps, "median_s": med, "min_s": min(worst),
            "max_s": max(worst), "GiB_s": nbytes / med / GIB, "rank_skew": statistics.median(skew),
            "cover": min(cover) if cover else ""}


def fit(rows: Sequence[Row]) -> Tuple[float, float, float]:
    """t = a + size / b by least squares on relative error: a in s, b in GiB/s, and the max error."""
    suu = suv = svv = su = sv = 0.0
    for r in rows:
        u, v = 1 / r["median_s"], r["bytes"] / GIB / r["median_s"]
        suu, suv, svv, su, sv = suu + u * u, suv + u * v, svv + v * v, su + u, sv + v
    det = suu * svv - suv * suv
    a, g = (su * svv - sv * suv) / det, (suu * sv - suv * su) / det
    return a, 1 / g, max(abs((a + g * r["bytes"] / GIB) / r["median_s"] - 1) for r in rows)


def group_of(size: int, world: int) -> Tuple[Any, int]:
    """This rank's group of *size* consecutive ranks, and the size; ``None`` and the world for 0.

    Every rank creates every group, in the same order, as ``new_group`` requires.
    """
    if size in (0, world):
        return None, world
    if size < 1 or world % size:
        raise ValueError(f"a group size must divide the {world} ranks, not {size}")
    rank, mine = dist.get_rank(), None
    for start in range(0, world, size):
        group = dist.new_group(list(range(start, start + size)))
        if start <= rank < start + size:
            mine = group
    return mine, size


def copying_ranks(count: int, world: int) -> List[int]:
    """The ranks that copy when *count* ranks of each node do, the first local ranks; every rank for 0."""
    per_node = int(os.environ.get("LOCAL_WORLD_SIZE", world))
    if count <= 0 or count >= per_node:
        return list(range(world))
    return [rank for rank in range(world) if rank % per_node < count]


def expert_exchange(dev: Device, full: torch.Tensor, size: int, group: Any) -> Run:
    """The trainer's expert exchange over *full*: counts all_to_all, both counts on the host, list all_to_all."""
    rows = full.numel() // size
    send_counts = torch.full((size,), rows, dtype=torch.int64, device=dev.dev)
    recv_counts = torch.empty_like(send_counts)
    out = torch.empty_like(full)

    def run() -> None:
        """One exchange, as one MoE layer's dispatch runs it."""
        dist.all_to_all_single(recv_counts, send_counts, group=group)
        send, recv = send_counts.tolist(), recv_counts.tolist()
        dist.all_to_all(list(out.split(recv)), list(full.split(send)), group=group)
    return run


def collective(dev: Device, name: str, nbytes: int, size: int, group: Any) -> Tuple[Run, int]:
    """A collective over *group* of *size* ranks on a buffer of about *nbytes*, and that buffer's bytes."""
    dtype = torch.float32 if dev.kind == "cpu" else torch.bfloat16
    n = nbytes // torch.empty(0, dtype=dtype).element_size() // size * size
    full = torch.zeros(n, dtype=dtype, device=dev.dev)
    part = torch.zeros(n // size, dtype=dtype, device=dev.dev)
    if name == "all_gather":
        def run() -> None:
            """One all_gather."""
            dist.all_gather_into_tensor(full, part, group=group)
    elif name == "reduce_scatter":
        def run() -> None:
            """One reduce_scatter."""
            dist.reduce_scatter_tensor(part, full, group=group)
    elif name == "all_to_all":
        out = torch.empty_like(full)

        def run() -> None:
            """One all_to_all, equal splits."""
            dist.all_to_all_single(out, full, group=group)
    elif name == "all_to_all_hp":
        run = expert_exchange(dev, full, size, group)
    else:
        raise ValueError(f"unknown collective {name!r}")
    return run, full.numel() * full.element_size()


def copier(buffers: Buffers, direction: str, nbytes: int) -> Tuple[Run, int]:
    """A copy of *nbytes* between the device buffer and the pinned host one, and the bytes it moves."""
    dev_buf, host_buf = buffers
    d, h = dev_buf[:nbytes], host_buf[:nbytes]
    src, dst = (d, h) if direction == "d2h" else (h, d)

    def run() -> None:
        """One copy, not waited for."""
        dst.copy_(src, non_blocking=True)
    return run, d.numel() * d.element_size()


def idle() -> None:
    """What a rank that does not copy runs in the copy's place."""


def copy_sweep(dev: Device, args: argparse.Namespace, buffers: Buffers, side: Any, add: Add) -> None:
    """Every size from --min-kib to --max-mib, each direction, once for each count of copying ranks."""
    world, rank = dist.get_world_size(), dist.get_rank()
    for count in args.copiers:
        copying = copying_ranks(count, world)
        label = "" if len(copying) == world else f" {count}/node"
        for direction in ("d2h", "h2d"):
            n = args.min_kib * 1024
            while n <= args.max_mib * MIB:
                run, moved = copier(buffers, direction, n)
                reps = min(args.max_reps, max(1, math.ceil(args.window_mib * MIB / moved)))
                parts = [(run if rank in copying else idle, reps, side)]
                add("copy", direction + label, moved, reps, measure(dev, parts, args), only=copying)
                n *= 2


def interfere(dev: Device, args: argparse.Namespace, comm: Tuple[Run, int, str], buffers: Buffers,
              side: Any, add: Add) -> List[Pair]:
    """One collective alone, then beside each copy direction: the share of its speed alone each side keeps."""
    crun, cbytes, label = comm
    creps = max(1, math.ceil(args.int_window_ms / 1e3 / agreed_seconds(dev, crun, None)))
    comm_alone = add("alone", label, cbytes, creps, measure(dev, [(crun, creps, None)], args))
    pairs = []
    for direction in ("d2h", "h2d"):
        run, moved = copier(buffers, direction, args.int_mib * MIB)
        reps = max(1, math.ceil(args.int_window_ms / 1e3 / agreed_seconds(dev, run, side)))
        copy_alone = add("alone", direction, moved, reps, measure(dev, [(run, reps, side)], args))
        busy_comm = [(run, reps, side), (crun, math.ceil(creps * args.margin), None)]
        copy_under = add("together", f"{direction} under {label}", moved, reps,
                         measure(dev, busy_comm, args), 0, 1)
        busy_copy = [(run, math.ceil(reps * args.margin), side), (crun, creps, None)]
        comm_under = add("together", f"{label} under {direction}", cbytes, creps,
                         measure(dev, busy_copy, args), 1, 0)
        if comm_alone is not None:
            keep_copy = copy_alone["median_s"] / copy_under["median_s"]
            keep_comm = comm_alone["median_s"] / comm_under["median_s"]
            pairs.append((label, direction, keep_copy, keep_comm, 2 - keep_copy - keep_comm,
                          min(copy_under["cover"], comm_under["cover"])))
    return pairs


def interference(dev: Device, args: argparse.Namespace, buffers: Buffers, side: Any, add: Add) -> List[Pair]:
    """Every collective in every group size, alone and beside the copies; rank 0 gets the pairs."""
    world, pairs = dist.get_world_size(), []
    for asked in args.group_sizes:
        group, size = group_of(asked, world)
        for name in args.collectives:
            crun, cbytes = collective(dev, name, args.comm_mib * MIB, size, group)
            label = name if size == world else f"{name}/{size}"
            pairs += interfere(dev, args, (crun, cbytes, label), buffers, side, add)
    return pairs


def summary(rows: Sequence[Row], interference_pairs: Sequence[Pair], args: argparse.Namespace) -> List[str]:
    """The lines an offload model needs: plateau and fit per copy, interference per pair."""
    lines = []
    for op in dict.fromkeys(r["op"] for r in rows if r["test"] == "copy"):
        mine = [r for r in rows if r["test"] == "copy" and r["op"] == op]
        flat = [r["GiB_s"] for r in mine if r["bytes"] >= args.plateau_mib * MIB]
        fitted = [r for r in mine if r["bytes"] >= args.fit_min_mib * MIB]
        line = (f"{op}: plateau {statistics.median(flat):.2f} GiB/s from {args.plateau_mib} MiB" if flat
                else f"{op}: no copy of {args.plateau_mib} MiB or more")
        if len(fitted) >= 2:
            a, b, err = fit(fitted)
            line += (f"; latency + bandwidth from {args.fit_min_mib} MiB: {a * 1e6:.1f} us + size / {b:.2f}"
                     f" GiB/s, max error {err:.1%}")
        lines.append(line)
    for name, d, keep_copy, keep_comm, f, cover in interference_pairs:
        warn = "" if cover >= 0.95 else ", BACKGROUND ENDED EARLY: raise --margin"
        lines.append(f"{d} with {name}: copy keeps {keep_copy:.0%}, {name} keeps {keep_comm:.0%},"
                     f" f = {f:.2f}, cover {cover:.3f}{warn}")
    return lines


def provenance(dev: Device, args: argparse.Namespace, world: int, host_buf: torch.Tensor) -> str:
    """One line saying where and how the numbers were taken."""
    try:
        pinned = host_buf.is_pinned()
    except Exception as exc:  # pylint: disable=broad-except  # a note, not a requirement
        pinned = f"unknown ({type(exc).__name__})"
    versions = f"torch {torch.__version__}"
    if dev.kind == "npu":
        import torch_npu  # pylint: disable=import-outside-toplevel
        versions += f", torch_npu {torch_npu.__version__}"
    name = dev.acc.get_device_name() if dev.acc else "cpu"
    return (f"host {socket.gethostname()}, {world} ranks, {os.environ.get('LOCAL_WORLD_SIZE', '?')} a node,"
            f" {name}, {versions}, pinned {pinned}, {datetime.datetime.now().isoformat(timespec='seconds')},"
            f" args {vars(args)}")


def write(rows: Sequence[Row], lines: Sequence[str], args: argparse.Namespace, header: str) -> None:
    """Write every row and the summary to --out, its folder made if missing."""
    folder = os.path.dirname(args.out)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# {header}\n")
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        fh.writelines(f"# {line}\n" for line in lines)


def _ints(text: str) -> List[int]:
    """A comma separated list of integers."""
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    """The benchmark's options."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="npu", choices=("npu", "cuda", "cpu"), help="cpu: dry run on gloo")
    p.add_argument("--min-kib", type=int, default=64, help="smallest copy")
    p.add_argument("--max-mib", type=int, default=2048, help="largest copy, and the pinned buffer a rank holds")
    p.add_argument("--window-mib", type=int, default=1024, help="bytes a copy sample moves back to back")
    p.add_argument("--max-reps", type=int, default=1000, help="most calls in one sample")
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--int-mib", type=int, default=256, help="copy size in the interference test")
    p.add_argument("--comm-mib", type=int, default=512, help="collective buffer size")
    p.add_argument("--collectives", default="all_gather,reduce_scatter,all_to_all",
                   help=f"comma separated, of {', '.join(COLLECTIVES)}; or none")
    p.add_argument("--group-sizes", type=_ints, default=[0],
                   help="comma separated: run the collectives over groups of this many consecutive ranks, "
                        "every group at once, once for each size; 0 for the whole job")
    p.add_argument("--copiers", type=_ints, default=[0],
                   help="comma separated: run the copy sweep once for each count of ranks a node that "
                        "copy, the first local ranks, the others idle; 0 for every rank")
    p.add_argument("--int-window-ms", type=float, default=200.0, help="timed window of a part run alone")
    p.add_argument("--margin", type=float, default=3.0, help="how much longer the background part runs")
    p.add_argument("--plateau-mib", type=int, default=256, help="sizes the plateau is the median of")
    p.add_argument("--fit-min-mib", type=int, default=1, help="smallest size in the latency + bandwidth fit")
    p.add_argument("--out", default="host_link.csv")
    args = p.parse_args()
    args.collectives = [c for c in args.collectives.split(",") if c and c != "none"]
    unknown = [c for c in args.collectives if c not in COLLECTIVES]
    if unknown:
        p.error(f"--collectives takes {', '.join(COLLECTIVES)} or none; got {unknown}")
    return args


def main() -> None:
    """Measure, print rank 0's rows and summary, and write them to --out."""
    args = parse_args()
    dev = Device(args.device, int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group({"npu": "hccl", "cuda": "nccl", "cpu": "gloo"}[args.device])
    rank, world = dist.get_rank(), dist.get_world_size()
    top = args.max_mib * MIB
    buffers = (torch.empty(top, dtype=torch.uint8, device=dev.dev),
               torch.zeros(top, dtype=torch.uint8, pin_memory=args.device != "cpu"))
    side = dev.stream()
    rows: List[Row] = []
    header = provenance(dev, args, world, buffers[1])
    if rank == 0:
        say(f"# {header}")

    def add(test: str, op: str, nbytes: int, reps: int, windows: Optional[List[Any]], part: int = 0,
            cover_by: Optional[int] = None, only: Optional[Sequence[int]] = None) -> Optional[Row]:
        """Keep and print rank 0's row of one part; ``None`` on every other rank."""
        if windows is None:
            return None
        r = row(test, op, nbytes, reps, windows, part, cover_by, only)
        rows.append(r)
        cover = "" if r["cover"] == "" else f"  cover {r['cover']:.3f}"
        say(f"{test:8s} {op:26s} {nbytes / MIB:10.3f} MiB {r['GiB_s']:8.2f} GiB/s  x{reps:<5d}"
            f" skew {r['rank_skew']:.2f}{cover}")
        return r

    copy_sweep(dev, args, buffers, side, add)
    pairs = interference(dev, args, buffers, side, add)
    if rank == 0:
        lines = summary(rows, pairs, args)
        for line in lines:
            say(line)
        write(rows, lines, args, header)
        say(f"wrote {args.out}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
