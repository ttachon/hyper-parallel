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
"""Price a search's candidates in worker processes, for run_nd -j.

A worker builds its own runner once, from the command-line values run_nd built
the main process's from (the recipe), with the censuses the main process ran
handed over, so that no worker runs one.  The main process walks the loop nest,
refuses and sets each candidate as a single process does, and sends the rest
as plain degrees in chunks; a worker sets each on its own config, prices it and
returns the numbers and the debug parts.  The main process writes them in the
search's order, so every output reads as one process writes it.
"""
import multiprocessing
import os
from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

from hyper_parallel.auto_parallel import _hf_model_spec
import hyper_parallel.auto_parallel.sapp_nd.nd.debug as Debug
from hyper_parallel.auto_parallel.sapp_nd.nd.common import cost_model_preprocess
from hyper_parallel.auto_parallel.sapp_nd.perf_estimation.estimate import estimate_performance

# This worker's runner, built on its first task.
_RUNNER = None


def default_workers(verbosity: int) -> int:
    """The workers a search uses when run_nd -j states none: ND_JOBS, else half the logical cores, at most 4.

    On Windows a worker is a new process that imports torch and parses the run
    again, about 7 s of CPU, so on an 8-core laptop four workers took the
    Qwen3.5 example to 25.5 s, eight to 27.0 and sixteen to 35.7.  Above
    verbosity 2 a search prints a line per estimate, which only the main
    process can, so it prices in one process.
    """
    stated = os.environ.get("ND_JOBS")
    if stated:
        return max(1, int(stated))
    if verbosity > 2:
        return 1
    return max(1, min(4, (os.cpu_count() or 2) // 2))


class Workers:
    """The processes that price a search's candidates, and the recipe each rebuilds the search's runner from.

    run_nd starts them before it parses the run, so that they import while it
    does.  The search hands them its candidates as plain degrees and gets back
    each candidate's number and debug parts, in the order it sent them.
    """

    def __init__(self, count: int, recipe: Dict[str, Any], pool: Any = None) -> None:
        """Start *count* spawned workers for *recipe*, or use *pool*, which a test passes in their place."""
        self.count = count
        self.recipe = recipe
        self._pool = pool if pool is not None else multiprocessing.get_context("spawn").Pool(
            count, initializer=_warm_up)

    def memory(self, items: List[Tuple[int, tuple]]) -> List[Tuple[Any, Any]]:
        """Each item's peak and debug parts, in order; an item is its position and the candidate's loop degrees."""
        return self._run(memory_task, items)

    def performance(self, items: List[Tuple[int, tuple, Any]], cache_file: Optional[str]) -> List[Tuple[Any, Any]]:
        """Each item's score and debug parts, in order; an item adds the peak its generation found."""
        return self._run(partial(perf_task, cache_file=cache_file), items)

    def _run(self, func: Callable, items: list) -> List[Tuple[Any, Any]]:
        """*func* over *items* in chunks, each with the recipe and the censuses this process ran."""
        censuses = dict(_hf_model_spec._CENSUSES)  # pylint: disable=protected-access
        tasks = [(self.recipe, censuses, chunk) for chunk in chunks(items, self.count)]
        priced = [None] * len(items)
        for out, reads in self._pool.imap(func, tasks):
            merge_unset_reads(reads)
            for index, value, info in out:
                priced[index] = (value, info)
        return priced

    def close(self, abort: bool = False) -> None:
        """Stop the workers: once their tasks are done, or at once where the search failed."""
        if abort:
            self._pool.terminate()
        else:
            self._pool.close()
        self._pool.join()


def _warm_up() -> None:
    """A worker's start: its output silenced, the search's modules imported; the runner comes with a task."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    # pylint: disable=import-outside-toplevel,unused-import
    import hyper_parallel.auto_parallel.sapp_nd.nd.parallelize  # noqa: F401


def build_runner(recipe: Dict[str, Any]) -> Any:
    """The search runner run_nd builds from *recipe*, the command-line values a pooled search takes."""
    # pylint: disable=import-outside-toplevel
    from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.size import Memory
    from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level
    import hyper_parallel.auto_parallel.sapp_nd.nd.common.hardware as Hard
    import hyper_parallel.auto_parallel.sapp_nd.nd.dimensions as Dim
    import hyper_parallel.auto_parallel.sapp_nd.nd.parallelize as Par

    cost_model_preprocess.set_strict(recipe["strict"])
    set_verbose_level(recipe["verbosity"])
    Debug.set_output_dir(recipe["output_dir"])
    machine = Hard.Machine(recipe["devices"], recipe["device_type"])
    max_mem = Memory.from_string(recipe["max_mem"]) if recipe["max_mem"] is not None else None
    return Par.Parallelize(
        recipe["framework"], recipe["yaml_config"], machine,
        global_batch_size=recipe["global_batch_size"], dimensions=Dim.get_dims(recipe["dimensions"]),
        swap_os=recipe["swap_os"], mppb=recipe["mppb"], auto_recompute=False, auto_offload=False,
        host_link=None, model=recipe["model"], max_mem=max_mem,
        mem_for_ppb=Memory.from_string(recipe["mem_for_ppb"]),
    ).instance


def _runner(recipe: Dict[str, Any], censuses: Dict[Any, Any]) -> Any:
    """This worker's runner, built on its first task with the main process's censuses so that it runs none."""
    global _RUNNER  # pylint: disable=global-statement
    if _RUNNER is None:
        _hf_model_spec._CENSUSES.update(censuses)  # pylint: disable=protected-access
        _RUNNER = build_runner(recipe)
        # The main process reports the fields its own parse read; a worker reports what its estimates read.
        cost_model_preprocess.UNSET_READS.clear()
    return _RUNNER


def _unset_reads() -> Dict[str, Dict[str, int]]:
    """The unset fields this worker's estimates read since its last task, which the main process reports."""
    reads = {attr: dict(readers) for attr, readers in cost_model_preprocess.UNSET_READS.items()}
    cost_model_preprocess.UNSET_READS.clear()
    return reads


def memory_task(task: Tuple[dict, dict, List[Tuple[int, tuple]]]) -> Tuple[list, dict]:
    """Each candidate's peak, and its debug parts where the search writes them, as inside_loop_nest prices it."""
    recipe, censuses, chunk = task
    runner = _runner(recipe, censuses)
    out = []
    for index, dims in chunk:
        config = runner.config.make_parallel_config(*dims)
        runner.config.set_parallel_config(config)
        if runner.enable_debug:
            debugger = Debug.Debug(config, info_type=Debug.MemParts, enable=True, output_file="debug_mem.csv")
            out.append((index, runner.memory_estim(debugger), dict(debugger.info)))
        else:
            out.append((index, runner.memory_estim(), None))
    return out, _unset_reads()


def perf_task(task: Tuple[dict, dict, List[Tuple[int, tuple, Any]]], cache_file: Optional[str]) -> Tuple[list, dict]:
    """Each candidate's score, and its debug parts where the search writes them, as order_search_space prices it."""
    recipe, censuses, chunk = task
    runner = _runner(recipe, censuses)
    out = []
    for index, dims, memory in chunk:
        config = runner.config.make_parallel_config(*dims)
        runner.config.set_parallel_config(config)
        if runner.enable_debug:
            debugger = Debug.Debug(config, info_type=Debug.PerfParts, enable=True)
            score = estimate_performance(runner.priced(), debugger=debugger, device_type=runner.machine.device,
                                         memory=memory, cache_file=cache_file, stage_savings=None)
            out.append((index, score, dict(debugger.info)))
        else:
            score = estimate_performance(runner.priced(), device_type=runner.machine.device, memory=memory,
                                         cache_file=cache_file, stage_savings=None)
            out.append((index, score, None))
    return out, _unset_reads()


def merge_unset_reads(reads: Dict[str, Dict[str, int]]) -> None:
    """Add a worker's unset reads to this process's, which the search reports under its ranking."""
    for attr, readers in reads.items():
        cost_model_preprocess.UNSET_READS[attr].update(Counter(readers))


def chunks(items: list, workers: int) -> List[list]:
    """*items* in contiguous chunks, about eight a worker, so that a slow chunk does not hold the others."""
    size = max(1, len(items) // (workers * 8))
    return [items[i:i + size] for i in range(0, len(items), size)]
