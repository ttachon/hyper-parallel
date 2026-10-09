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
"""Measure what host copies get inside the training step, and what they cost it.

Trains as ``scripts/train_lm.py`` does, with a probe that copies a dummy device
buffer to pinned host memory and back on a side stream, at the points an
offload would, and times every step with device events. Each step runs one
mode, cycling after a warmup of ``base`` steps:

- ``base``: no copies, the step as it runs today.
- ``d2h``: copies to the host from the end of decoder layer 0's forward,
  queued to outlast the forward.
- ``h2d``: copies from the host from the start of the layers' backward, up to
  ``--probe-max-gib``, about the backward's first second.
- ``both``: one offload's worth, ``--probe-offload-gib``, to the host from the
  end of layer 0's forward, and back from the start of layer 1's backward,
  which is when layer 0's activations would have to start coming back.

Every die copies at once, as an offload would. The copies are also timed alone,
every die at once, before training and after it, so that each die's share of
the link inside the step reads against its own speed alone. No activation
moves and nothing is freed: the probe measures the link inside the step and
what the copies cost the step, not the memory an offload would save.

Run it as the Demo 2 sweep runs the trainer, through the cluster kit from the
repository root, with this file in place of ``scripts/train_lm.py`` and the
probe's options added. Options that start with ``--probe-`` are the probe's;
``--probe-help`` lists them. Every other argument goes to the trainer as it
would to ``scripts/train_lm.py``. The Demo 2 configuration, EP 2 with a
32-wide shard on four nodes at 8192 tokens, under the plan ND picks::

    cluster -c examples/training_demo/cluster_qwen3_5_moe.env torchrun -n 4 \\
        examples/training_demo/offload_probe.py examples/training_demo/train_qwen3_5_moe.yaml \\
        --model.num_hidden_layers=8 --dataset.data_config.seq_length=8192 \\
        --training.global_batch_size=64 --training.micro_batch_size=1 --training.train_iters=20 \\
        --accelerator.tp_size=1 --accelerator.cp_size=1 --accelerator.pp_size=1 --accelerator.ep_size=2 \\
        --fsdp_config.dp_shard_size=32 --fsdp_config.edp_shard_size=32 --profiling.enabled=false \\
        --probe-ac-off=3-7

Twenty steps are four of warmup and four of each mode, and need 1280 samples
of 8192 tokens in the dataset. Rank 0 prints a line starting with
``OFFLOAD_PROBE`` after every step and the report at the end, in node 0's
log, and writes every number to ``--probe-out``.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import json
import math
import socket
import statistics
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch  # pylint: disable=forbidden-backend-import
import torch.distributed as dist  # pylint: disable=forbidden-backend-import
import yaml

MIB = 2 ** 20
GIB = 2 ** 30
MODES = ("base", "d2h", "h2d", "both")
PREFIX = "OFFLOAD_PROBE"
# A window used to size the copies before any base step has measured one, in ms.
_DEFAULT_WINDOW_MS = 1500.0


def parse_probe_options(tokens: Sequence[str]) -> argparse.Namespace:
    """Parse the probe's own options, the arguments that start with ``--probe-``.

    Args:
        tokens: The probe's arguments, each ``--probe-name=value`` or a flag.

    Returns:
        The options.
    """
    parser = argparse.ArgumentParser(prog="offload_probe", description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("--probe-ac-off", default=None,
                        help="Run full recompute but for these layers, which run off, as the Demo 2 plans "
                             "state them: 3-7 for the plan ND picks, none for every layer full. Unset keeps "
                             "the config's activation_checkpoint as it is.")
    parser.add_argument("--probe-modes", default=",".join(MODES),
                        help="The modes to cycle through after the warmup, comma separated, of "
                             f"{', '.join(MODES)}.")
    parser.add_argument("--probe-warmup", type=int, default=4, help="Base steps before the cycle starts.")
    parser.add_argument("--probe-chunk-mib", type=int, default=64,
                        help="The size of one copy, and of the device and pinned host buffers, in MiB.")
    parser.add_argument("--probe-offload-gib", type=float, default=4.0,
                        help="What the both mode moves each way, in GiB: what ND offloads from layer 0 on "
                             "the Demo 2 search is 2.3 to 4.05.")
    parser.add_argument("--probe-outlast", type=float, default=1.3,
                        help="How far the d2h and h2d queues outlast their window, as a share of what the "
                             "copies alone would move in the last base step's window.")
    parser.add_argument("--probe-max-gib", type=float, default=16.0,
                        help="The most one queue holds, in GiB. The backward outlasts any queue this size, so "
                             "the h2d mode times the first second or so of it; a larger queue takes longer to "
                             "queue in the backward's own hook, which delays the step it measures.")
    parser.add_argument("--probe-alone-gib", type=float, default=2.0,
                        help="What each direction moves to time the copies alone, in GiB.")
    parser.add_argument("--probe-alone-reps", type=int, default=3, help="How many times the copies alone run.")
    parser.add_argument("--probe-out", default=None,
                        help="The JSON rank 0 writes; output/offload_probe/probe_<time>.json when unset.")
    parser.add_argument("--probe-help", action="help", help="Show the probe's options and exit.")
    options = parser.parse_args(list(tokens))
    modes = tuple(mode.strip() for mode in options.probe_modes.split(",") if mode.strip())
    unknown = [mode for mode in modes if mode not in MODES]
    if not modes or unknown:
        parser.error(f"--probe-modes takes {', '.join(MODES)}; got {options.probe_modes!r}")
    if options.probe_chunk_mib <= 0 or options.probe_offload_gib <= 0 or options.probe_outlast <= 0:
        parser.error("--probe-chunk-mib, --probe-offload-gib and --probe-outlast must be positive")
    return argparse.Namespace(
        ac_off=options.probe_ac_off,
        modes=modes,
        warmup=max(0, options.probe_warmup),
        chunk=options.probe_chunk_mib * MIB,
        offload=int(options.probe_offload_gib * GIB),
        outlast=options.probe_outlast,
        max_bytes=int(options.probe_max_gib * GIB),
        alone=int(options.probe_alone_gib * GIB),
        alone_reps=max(1, options.probe_alone_reps),
        out=options.probe_out,
    )


def split_arguments(argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Split a command line into the probe's arguments and the trainer's.

    Returns:
        ``(probe, trainer)``, each in the order given.
    """
    probe = [token for token in argv if token.startswith("--probe-")]
    trainer = [token for token in argv if not token.startswith("--probe-")]
    return probe, trainer


def plan_checkpoint(ac_off: Optional[str]) -> Optional[Dict[str, Any]]:
    """The ``activation_checkpoint`` section ``--probe-ac-off`` stands for, as the Demo 2 plan yamls state it.

    Args:
        ac_off: The layers that run off under full recompute, such as ``3-7``,
            ``none`` for every layer full, or ``None`` to keep the config's.

    Returns:
        The section, or ``None`` to keep the config's.
    """
    if ac_off is None:
        return None
    if ac_off.strip().lower() == "none":
        return {"mode": "full"}
    layers = {part.strip(): "off" for part in ac_off.split(",") if part.strip()}
    return {"mode": "full", "layers": layers}


def apply_plan(trainer_args: List[str], ac_off: Optional[str]) -> List[str]:
    """The trainer's arguments with the config file rewritten to run the plan ``--probe-ac-off`` names.

    The rewritten config goes to a file of its own, as the Demo 2 plan yamls
    did, since a plan's mapping cannot ride the cluster kit's command line.
    """
    checkpoint = plan_checkpoint(ac_off)
    if checkpoint is None:
        return list(trainer_args)
    position = next((index for index, token in enumerate(trainer_args) if not token.startswith("-")), None)
    if position is None:
        raise SystemExit(f"{PREFIX}: --probe-ac-off needs the config file among the trainer's arguments")
    with open(trainer_args[position], encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["activation_checkpoint"] = checkpoint
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="offload_probe_", delete=False,
                                     encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return trainer_args[:position] + [handle.name] + trainer_args[position + 1:]


class _HostEvent:
    """A timing event on the host clock, for a dry run where every op runs as it is called."""

    def __init__(self) -> None:
        """An event not yet recorded."""
        self.time = 0.0

    def record(self) -> None:
        """Take the time."""
        self.time = time.perf_counter()

    def elapsed_time(self, other: "_HostEvent") -> float:
        """The milliseconds from this event to *other*."""
        return (other.time - self.time) * 1e3


class _Device:
    """The few device calls the probe makes: on an npu or a cuda device, or on the host for a dry run."""

    def __init__(self, device: torch.device) -> None:
        """Bind the calls to *device*'s accelerator module, where it has one."""
        self.device = device
        self.module = getattr(torch, device.type, None) if device.type in ("npu", "cuda") else None

    def stream(self) -> Any:
        """A side stream, or ``None`` on the host."""
        return self.module.Stream() if self.module is not None else None

    def on(self, stream: Any) -> Any:
        """A context that makes *stream* current."""
        return self.module.stream(stream) if stream is not None else contextlib.nullcontext()

    def event(self) -> Any:
        """A timing event."""
        return self.module.Event(enable_timing=True) if self.module is not None else _HostEvent()

    def sync(self) -> None:
        """Wait for every stream of the device."""
        if self.module is not None:
            self.module.synchronize()


def find_layers(model: torch.nn.Module) -> Tuple[List[torch.nn.Module], torch.nn.Module]:
    """The decoder layers and the final norm of a Hugging Face style decoder.

    The module holding the most layers in a ``layers`` module list, beside a
    ``norm``.

    Raises:
        ValueError: For a model with no such module.
    """
    best = None
    for module in model.modules():
        layers = getattr(module, "layers", None)
        norm = getattr(module, "norm", None)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) and isinstance(norm, torch.nn.Module):
            if best is None or len(layers) > len(best.layers):
                best = module
    if best is None:
        raise ValueError(f"{PREFIX}: found no module with decoder layers and a final norm in {type(model).__name__}")
    return list(best.layers), best.norm


def _hidden(args: Sequence[Any], kwargs: Dict[str, Any]) -> Optional[torch.Tensor]:
    """The hidden states a decoder layer or a norm is called with."""
    if args and isinstance(args[0], torch.Tensor):
        return args[0]
    hidden = kwargs.get("hidden_states")
    return hidden if isinstance(hidden, torch.Tensor) else None


class OffloadProbe:
    """Times every step and copies a dummy buffer at the points an offload would; a trainer callback.

    Args:
        model: The model the trainer calls.
        device: The training device.
        options: The probe's options (:func:`parse_probe_options`).
        plan: What the run recomputes, for the report.
        log: Where the probe's lines go.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device, options: argparse.Namespace,
                 plan: str = "", log: Callable[[str], None] = print) -> None:
        """Allocate the copy buffers and hook the model's forward, its layers and its final norm."""
        self.options = options
        self.plan = plan
        self.log = log
        self.dev = _Device(device)
        self.stream = self.dev.stream()
        self.chunk = options.chunk
        self.device_buffer = torch.empty(self.chunk, dtype=torch.uint8, device=device)
        self.host_buffer = torch.empty(self.chunk, dtype=torch.uint8, pin_memory=self.dev.module is not None)
        self.rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.layers, norm = find_layers(model)
        self.records: List[Dict[str, Any]] = []
        self.alone: Dict[str, Dict[str, List[float]]] = {}
        self.windows = {"after0": None, "backward": None}
        self.errors: List[str] = []
        self.steps = 0
        self.current: Optional[Dict[str, Any]] = None
        self.phase = "idle"
        self.handles = [model.register_forward_pre_hook(self._model_pre, with_kwargs=True)]
        for index, layer in enumerate(self.layers):
            self.handles.append(
                layer.register_forward_pre_hook(functools.partial(self._layer_pre, index), with_kwargs=True))
        self.handles.append(norm.register_forward_pre_hook(self._norm_pre, with_kwargs=True))

    # Hooks. A failure is reported and turns the probe off for the step, never the training run.

    def _guard(self, name: str, body: Callable[[], None]) -> None:
        """Run a hook's body; on an exception, record it and stop probing the step."""
        try:
            body()
        except Exception:  # pylint: disable=broad-except
            if len(self.errors) < 20:
                self.errors.append(f"step {self.steps} {name}: {traceback.format_exc(limit=3)}")
            if self.current is not None:
                self.current["failed"] = True

    def _mark(self, name: str) -> None:
        """Record a timing event on the current stream, once a step."""
        if self.current is not None and name not in self.current["events"]:
            event = self.dev.event()
            event.record()
            self.current["events"][name] = event

    def _model_pre(self, module: torch.nn.Module, args: Any, kwargs: Any) -> None:
        """The model's forward begins: the step's first micro-batch is the one probed."""
        del module, args, kwargs
        if self.current is not None and self.phase == "pending":
            self.phase = "forward"
            self._guard("forward", lambda: self._mark("forward"))

    def _layer_pre(self, index: int, module: torch.nn.Module, args: Any, kwargs: Any) -> None:
        """Layer *index*'s forward begins, the end of the one before it; a recomputation is ignored."""
        del module
        if self.current is None or self.phase != "forward":
            if self.current is not None:
                self.current["ignored"] += 1
            return

        def body() -> None:
            """Mark the layer, watch its input's gradient, and after layer 0 start the copies out."""
            self._mark(f"layer{index}")
            hidden = _hidden(args, kwargs)
            if hidden is not None and hidden.requires_grad:
                hidden.register_hook(functools.partial(self._gradient, f"grad_in{index}"))
            if index == 1 and self.current["mode"] in ("d2h", "both"):
                size = self.options.offload if self.current["mode"] == "both" else self._outlasting("d2h", "after0")
                self._queue("d2h", "d2h", size, "layer1")

        self._guard(f"layer{index}", body)

    def _norm_pre(self, module: torch.nn.Module, args: Any, kwargs: Any) -> None:
        """The final norm begins: the layers' forward has ended."""
        del module
        if self.current is None or self.phase != "forward":
            return

        def body() -> None:
            """Mark the norm and watch its input's gradient, where the layers' backward begins."""
            self._mark("norm")
            hidden = _hidden(args, kwargs)
            if hidden is not None and hidden.requires_grad:
                hidden.register_hook(functools.partial(self._gradient, "grad_norm_in"))

        self._guard("norm", body)
        self.phase = "head"

    def _gradient(self, name: str, grad: torch.Tensor) -> None:
        """A gradient the probe watches has arrived: a layer's backward begins or ends."""
        del grad
        if self.current is None:
            return
        if name == "grad_norm_in":
            self.phase = "backward"

        def body() -> None:
            """Mark the gradient and start the copies back that the step's mode calls for."""
            self._mark(name)
            mode = self.current["mode"]
            if name == "grad_norm_in" and mode == "h2d":
                self._queue("h2d", "h2d", self._outlasting("h2d", "backward"), name)
            if name == "grad_in2" and mode == "both":
                self._queue("h2d", "h2d", self.options.offload, name)

        self._guard(name, body)

    # Copies.

    def _outlasting(self, direction: str, window: str) -> int:
        """Bytes that outlast the last base step's *window*, at the speed this die copied alone."""
        rates = self.alone.get("before", {}).get(direction)
        rate = statistics.median(rates) if rates else 10.0  # GiB/s
        window_ms = self.windows[window] or _DEFAULT_WINDOW_MS
        return min(self.options.max_bytes, int(rate * GIB * window_ms / 1e3 * self.options.outlast))

    def _enqueue(self, direction: str, size: int, after: Any) -> Tuple[Any, List[Any], float]:
        """Queue *size* bytes of copies on the side stream, behind *after*; one event at each copy's end.

        Returns:
            ``(start, ends, cpu_ms)``: the queue's start event, each copy's
            end event, and the host time the queueing took.
        """
        began = time.perf_counter()
        count = max(1, math.ceil(size / self.chunk))
        ends = []
        with self.dev.on(self.stream):
            if self.stream is not None:
                self.stream.wait_event(after)
            start = self.dev.event()
            start.record()
            for _ in range(count):
                if direction == "d2h":
                    self.host_buffer.copy_(self.device_buffer, non_blocking=True)
                else:
                    self.device_buffer.copy_(self.host_buffer, non_blocking=True)
                end = self.dev.event()
                end.record()
                ends.append(end)
        return start, ends, (time.perf_counter() - began) * 1e3

    def _queue(self, name: str, direction: str, size: int, after: str) -> None:
        """Queue copies for the step behind its event *after*."""
        start, ends, cpu_ms = self._enqueue(direction, size, self.current["events"][after])
        self.current["copies"][name] = {"direction": direction, "start": start, "ends": ends, "cpu_ms": cpu_ms}

    def _time_alone(self, direction: str) -> List[float]:
        """This die's copy speed with nothing else running, every die copying at once, in GiB/s."""
        rates = []
        for _ in range(self.options.alone_reps):
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            self.dev.sync()
            anchor = self.dev.event()
            anchor.record()
            start, ends, _ = self._enqueue(direction, self.options.alone, anchor)
            self.dev.sync()
            rates.append(len(ends) * self.chunk / GIB / (start.elapsed_time(ends[-1]) / 1e3))
        return rates

    # Trainer callback.

    def on_train_begin(self, state: Any = None, **kwargs: Any) -> None:
        """Time the copies alone, every die at once."""
        del state, kwargs
        self.alone["before"] = {direction: self._time_alone(direction) for direction in ("d2h", "h2d")}
        if self.rank == 0:
            self.log(f"{PREFIX} rank 0 alone: to the host {statistics.median(self.alone['before']['d2h']):.2f} "
                     f"GiB/s, from it {statistics.median(self.alone['before']['h2d']):.2f}")

    def mode_of(self, step: int) -> str:
        """The mode of the probe's *step*-th step, counting from 1."""
        if step <= self.options.warmup:
            return "base"
        return self.options.modes[(step - self.options.warmup - 1) % len(self.options.modes)]

    def on_step_begin(self, state: Any = None, **kwargs: Any) -> None:
        """Pick the step's mode and mark its start."""
        del state, kwargs
        self.steps += 1
        self.current = {"step": self.steps, "mode": self.mode_of(self.steps),
                        "warmup": self.steps <= self.options.warmup, "events": {}, "copies": {}, "ignored": 0,
                        "failed": False, "wall": time.perf_counter()}
        self.phase = "pending"
        self._guard("step_begin", lambda: self._mark("step_begin"))

    def on_step_end(self, state: Any = None, **kwargs: Any) -> None:
        """Mark the step's end, wait for every stream, and read the step's times."""
        del state, kwargs
        current = self.current
        if current is None:
            return
        self._guard("step_end", lambda: self._mark("step_end"))
        self.dev.sync()
        wall = (time.perf_counter() - current["wall"]) * 1e3
        self.current, self.phase = None, "idle"
        try:
            record = resolve(current, self.chunk, wall)
        except Exception:  # pylint: disable=broad-except
            self.errors.append(f"step {current['step']} resolve: {traceback.format_exc(limit=3)}")
            return
        self.records.append(record)
        times = record["t"]
        if record["mode"] == "base" and not record["failed"]:
            if "layer1" in times and "norm" in times:
                self.windows["after0"] = times["norm"] - times["layer1"]
            if "grad_norm_in" in times and "grad_in0" in times:
                self.windows["backward"] = times["grad_in0"] - times["grad_norm_in"]
        if self.rank == 0:
            self.log(step_line(record, len(self.layers)))

    def on_train_end(self, state: Any = None, **kwargs: Any) -> None:
        """Time the copies alone again, gather every rank's numbers on rank 0, report and write them."""
        del state, kwargs
        self.alone["after"] = {direction: self._time_alone(direction) for direction in ("d2h", "h2d")}
        for handle in self.handles:
            handle.remove()
        payload = {"rank": self.rank, "host": socket.gethostname(), "records": self.records, "alone": self.alone,
                   "errors": self.errors, "layers": len(self.layers)}
        if dist.is_available() and dist.is_initialized():
            payloads: List[Any] = [None] * dist.get_world_size()
            dist.all_gather_object(payloads, payload)
        else:
            payloads = [payload]
        if self.rank != 0:
            return
        report = summarize(payloads, self.options, self.plan)
        for line in report["lines"]:
            self.log(line)
        out = Path(self.options.out or f"output/offload_probe/probe_{time.strftime('%Y%m%d_%H%M%S')}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"options": dict(vars(self.options)),
                                   "plan": self.plan, "summary": report["summary"], "ranks": payloads},
                                  indent=1, default=str), encoding="utf-8")
        self.log(f"{PREFIX} every number: {out.resolve()}")

    def on_epoch_begin(self, state: Any = None, **kwargs: Any) -> None:
        """Nothing to do."""

    def on_epoch_end(self, state: Any = None, **kwargs: Any) -> None:
        """Nothing to do."""

    def on_micro_step_begin(self, state: Any = None, micro_batch: Any = None, **kwargs: Any) -> None:
        """Nothing to do: the step's first micro-batch is the one probed."""

    def on_micro_step_end(self, state: Any = None, **kwargs: Any) -> None:
        """Nothing to do."""


def resolve(current: Dict[str, Any], chunk: int, wall: float) -> Dict[str, Any]:
    """A step's events as milliseconds after its start, once every stream has finished.

    Returns:
        The step's record: its mode, every mark's time, each queue's start and
        copy ends, the host's step time and how many recomputing calls the
        probe ignored.
    """
    events = current["events"]
    anchor = events["step_begin"]
    times = {name: round(anchor.elapsed_time(event), 3) for name, event in events.items()}
    copies = {}
    for name, copy in current["copies"].items():
        copies[name] = {
            "direction": copy["direction"],
            "chunk": chunk,
            "start": round(anchor.elapsed_time(copy["start"]), 3),
            "ends": [round(anchor.elapsed_time(end), 3) for end in copy["ends"]],
            "cpu_ms": round(copy["cpu_ms"], 3),
        }
    return {"step": current["step"], "mode": current["mode"], "warmup": current["warmup"],
            "failed": current["failed"], "ignored": current["ignored"], "wall_ms": round(wall, 3),
            "t": times, "copies": copies}


def copy_stats(copy: Dict[str, Any], window_end: Optional[float]) -> Dict[str, Any]:
    """What a queue of copies moved by *window_end*, and how fast it copied while it ran.

    The speed is taken over the copies that ended inside the window, from the
    queue's start to the last of them, so a copy still running at the window's
    end does not count.

    Returns:
        ``queued_gib``, ``done_gib``, ``gibps`` (``None`` when no copy ended in
        the window), ``late_ms`` (how long after the window the last copy
        ended, 0 when it ended inside) and ``cpu_ms``.
    """
    chunk, start, ends = copy["chunk"], copy["start"], copy["ends"]
    end_of_window = math.inf if window_end is None else window_end
    done = [end for end in ends if end <= end_of_window]
    rate = len(done) * chunk / GIB / ((done[-1] - start) / 1e3) if done and done[-1] > start else None
    return {
        "queued_gib": len(ends) * chunk / GIB,
        "done_gib": len(done) * chunk / GIB,
        "gibps": rate,
        "late_ms": max(0.0, ends[-1] - end_of_window) if ends and window_end is not None else 0.0,
        "cpu_ms": copy.get("cpu_ms", 0.0),
    }


def phases(record: Dict[str, Any], layers: int) -> Dict[str, Any]:
    """A step's phases in milliseconds, those its marks allow.

    ``forward``: the layers' forward, from layer 0's start to the final norm;
    ``after0``: what of it follows layer 0, the window ND gives layer 0's
    copies; ``head``: from the final norm to the gradient at its input, the
    norm, the output layer and the loss, forward and backward; ``backward``:
    the layers' backward, to the gradient at layer 0's input; ``step``: the
    whole step on the device; ``layer``: each layer's forward.
    """
    t = record["t"]

    def span(first: str, last: str) -> Optional[float]:
        """The milliseconds from mark *first* to mark *last*, where both were recorded."""
        return t[last] - t[first] if first in t and last in t else None

    marks = [f"layer{index}" for index in range(layers)] + ["norm"]
    return {
        "forward": span("layer0", "norm"),
        "after0": span("layer1", "norm"),
        "head": span("norm", "grad_norm_in"),
        "backward": span("grad_norm_in", "grad_in0"),
        "step": span("step_begin", "step_end"),
        "layer": [span(first, last) for first, last in zip(marks, marks[1:])],
    }


def window_of(record: Dict[str, Any], name: str) -> Optional[float]:
    """The time a queue's copies must end by: the forward's end to the host, the backward's from it.

    In the both mode, the copies back must end before layer 0's backward begins.
    """
    t = record["t"]
    if name == "d2h":
        return t.get("norm")
    if record["mode"] == "both":
        return t.get("grad_in1")
    return t.get("grad_in0")


def step_line(record: Dict[str, Any], layers: int) -> str:
    """One rank's step in one line, for the training log."""
    parts = phases(record, layers)

    def ms(value: Optional[float]) -> str:
        """A time in milliseconds, or ``?`` where it is unknown."""
        return "?" if value is None else f"{value:.1f}"

    line = (f"{PREFIX} step {record['step']} {record['mode']}{' (warmup)' if record['warmup'] else ''}: "
            f"step {ms(parts['step'])} ms, forward {ms(parts['forward'])} (after layer 0 {ms(parts['after0'])}), "
            f"head {ms(parts['head'])}, backward {ms(parts['backward'])}")
    for name, copy in record["copies"].items():
        stats = copy_stats(copy, window_of(record, name))
        rate = "?" if stats["gibps"] is None else f"{stats['gibps']:.2f}"
        line += (f"; {name} {stats['done_gib']:.2f} of {stats['queued_gib']:.2f} GiB in its window at {rate} "
                 f"GiB/s, {stats['late_ms']:.1f} ms late")
    if record["failed"]:
        line += "; PROBE FAILED this step"
    return line


def _median(values: Sequence[Optional[float]]) -> Optional[float]:
    """The median of the values that are known."""
    known = [value for value in values if value is not None]
    return statistics.median(known) if known else None


def _fmt(value: Optional[float], digits: int = 1) -> str:
    """A number, or ``?`` when it is unknown."""
    return "?" if value is None else f"{value:.{digits}f}"


def summarize(payloads: Sequence[Dict[str, Any]], options: argparse.Namespace, plan: str = "") -> Dict[str, Any]:
    """Every rank's steps reduced to what the job saw, mode by mode.

    A step waits for its slowest die, so its times are the most any rank
    measured; a copy speed is the slowest die's and the median die's, each
    against that die's own speed alone. Over a mode's steps, the median.

    Returns:
        ``lines``, the report, and ``summary``, the same numbers as data.
    """
    ranks = len(payloads)
    layers = payloads[0]["layers"]
    hosts = sorted({payload["host"] for payload in payloads})
    alone = {}
    for when in ("before", "after"):
        for direction in ("d2h", "h2d"):
            per_rank = [_median(payload["alone"].get(when, {}).get(direction, [])) for payload in payloads]
            known = [rate for rate in per_rank if rate is not None]
            alone[when, direction] = (min(known) if known else None, _median(known), per_rank)
    by_step: Dict[int, List[Dict[str, Any]]] = {}
    for payload in payloads:
        for record in payload["records"]:
            by_step.setdefault(record["step"], []).append(dict(record, rank=payload["rank"]))
    steps = []
    for step, records in sorted(by_step.items()):
        every = [phases(record, layers) for record in records]
        # "index" is the step's number; "step", set below with the phases, its time.
        entry = {"index": step, "mode": records[0]["mode"], "warmup": records[0]["warmup"],
                 "failed": sum(record["failed"] for record in records),
                 "ignored": max(record["ignored"] for record in records)}
        for key in ("forward", "after0", "head", "backward", "step"):
            known = [parts[key] for parts in every if parts[key] is not None]
            entry[key] = max(known) if known else None
        entry["layer"] = [_median([parts["layer"][index] for parts in every]) for index in range(layers)]
        for name in ("d2h", "h2d"):
            shares, rates, late, done, cpu = [], [], [], [], []
            for record in records:
                copy = record["copies"].get(name)
                if copy is None:
                    continue
                stats = copy_stats(copy, window_of(record, name))
                own = alone["before", name][2][[p["rank"] for p in payloads].index(record["rank"])]
                rates.append(stats["gibps"])
                shares.append(stats["gibps"] / own if stats["gibps"] is not None and own else None)
                late.append(stats["late_ms"])
                done.append(stats["done_gib"])
                cpu.append(stats["cpu_ms"])
            if rates:
                known_rates = [rate for rate in rates if rate is not None]
                known_shares = [share for share in shares if share is not None]
                entry[name] = {
                    "slowest_gibps": min(known_rates) if known_rates else None,
                    "median_gibps": _median(known_rates),
                    "least_share": min(known_shares) if known_shares else None,
                    "median_share": _median(known_shares),
                    "least_done_gib": min(done),
                    "late_ranks": sum(value > 0 for value in late),
                    "worst_late_ms": max(late),
                    "most_cpu_ms": max(cpu),
                }
        steps.append(entry)
    measured = [entry for entry in steps if not entry["warmup"]]
    modes = {}
    for mode in options.modes:
        of_mode = [entry for entry in measured if entry["mode"] == mode]
        if not of_mode:
            continue
        summary = {"steps": len(of_mode)}
        for key in ("forward", "after0", "head", "backward", "step"):
            summary[key] = _median([entry[key] for entry in of_mode])
        for name in ("d2h", "h2d"):
            copies = [entry[name] for entry in of_mode if name in entry]
            if copies:
                summary[name] = {key: _median([copy[key] for copy in copies]) for key in copies[0]}
                summary[name]["late_ranks"] = max(copy["late_ranks"] for copy in copies)
                summary[name]["worst_late_ms"] = max(copy["worst_late_ms"] for copy in copies)
        summary["layer"] = [_median([entry["layer"][index] for entry in of_mode]) for index in range(layers)]
        modes[mode] = summary
    lines = _report_lines(ranks, hosts, layers, plan, options, alone, modes, steps)
    errors = [error for payload in payloads for error in payload["errors"]]
    if errors:
        lines.append(f"{PREFIX} {len(errors)} probe errors, the first: {errors[0].strip().splitlines()[-1]}")
    return {"lines": lines, "summary": {"modes": modes, "steps": steps,
                                        "alone": {f"{when}_{direction}": value[:2]
                                                  for (when, direction), value in alone.items()},
                                        "errors": errors}}


def _report_lines(ranks: int, hosts: Sequence[str], layers: int, plan: str, options: argparse.Namespace,
                  alone: Dict[Tuple[str, str], Any], modes: Dict[str, Dict[str, Any]],
                  steps: Sequence[Dict[str, Any]]) -> List[str]:
    """The report's lines."""
    counts = ", ".join(f"{mode} {summary['steps']}" for mode, summary in modes.items())
    lines = [f"{PREFIX} {ranks} ranks on {len(hosts)} nodes, {layers} layers, {plan or 'the config recompute'}; "
             f"copies of {options.chunk // MIB} MiB; steps measured: {counts} (after {options.warmup} warmup)"]
    for when in ("before", "after"):
        lines.append(
            f"{PREFIX} alone {when} training, every die at once: to the host slowest die "
            f"{_fmt(alone[when, 'd2h'][0], 2)} GiB/s, median {_fmt(alone[when, 'd2h'][1], 2)}; from it slowest "
            f"{_fmt(alone[when, 'h2d'][0], 2)}, median {_fmt(alone[when, 'h2d'][1], 2)}")
    base = modes.get("base", {})
    for mode, summary in modes.items():
        delta = ""
        if mode != "base" and base.get("step") and summary.get("step"):
            change = summary["step"] - base["step"]
            delta = f" ({change:+.1f} ms, {change / base['step']:+.2%} against base)"
        lines.append(f"{PREFIX} {mode}: step {_fmt(summary.get('step'))} ms{delta}; layers' forward "
                     f"{_fmt(summary.get('forward'))} (after layer 0 {_fmt(summary.get('after0'))}), head "
                     f"{_fmt(summary.get('head'))}, layers' backward {_fmt(summary.get('backward'))}")
        for name, where in (("d2h", "to the host in the forward"), ("h2d", "from the host in the backward")):
            copy = summary.get(name)
            if not copy:
                continue
            window = "before layer 0's backward" if mode == "both" and name == "h2d" else (
                "by the forward's end" if name == "d2h" else "by the backward's end")
            lines.append(
                f"{PREFIX}   {name} {where}: slowest die {_fmt(copy['slowest_gibps'], 2)} GiB/s "
                f"(share of its speed alone {_fmt(copy['least_share'], 2)}), median die "
                f"{_fmt(copy['median_gibps'], 2)} ({_fmt(copy['median_share'], 2)}); {window} the slowest die "
                f"moved {_fmt(copy['least_done_gib'], 2)} GiB, {copy['late_ranks']} of {ranks} ranks late, worst "
                f"{_fmt(copy['worst_late_ms'])} ms; queueing took at most {_fmt(copy['most_cpu_ms'])} ms of host")
    if base.get("layer"):
        lines.append(f"{PREFIX} base, each layer's forward in ms: "
                     + ", ".join(f"{index} {_fmt(value)}" for index, value in enumerate(base["layer"])))
    ignored = max((entry["ignored"] for entry in steps), default=0)
    failed = sum(entry["failed"] for entry in steps)
    lines.append(f"{PREFIX} recomputing forwards ignored a step: up to {ignored}; rank-steps the probe failed: "
                 f"{failed}")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Build the trainer as scripts/train_lm.py does, add the probe, and train."""
    probe_args, trainer_args = split_arguments(sys.argv[1:] if argv is None else argv)
    options = parse_probe_options(probe_args)
    trainer_args = apply_plan(trainer_args, options.ac_off)
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.trainer.config.parser import parse_training_args
    from hyper_parallel.trainer.text_trainer import TextTrainer

    config = parse_training_args(trainer_args)
    trainer = TextTrainer(config)
    base = trainer.base
    checkpoint = config.activation_checkpoint
    layers = getattr(checkpoint, "layers", None)
    plan = f"activation checkpoint {checkpoint.mode}" + (f", layers {layers}" if layers else "")
    # Flushed, so that the lines reach the node's log as the run goes.
    log = functools.partial(print, flush=True)
    if getattr(base, "num_micro_batches", 1) > 1 and base.global_rank == 0:
        log(f"{PREFIX} {base.num_micro_batches} micro-batches a step: the first of each step is probed")
    probe = OffloadProbe(base.model, base.device, options, plan=plan, log=log)
    base._callbacks.append(probe)  # pylint: disable=protected-access
    trainer.train()


if __name__ == "__main__":
    main()
