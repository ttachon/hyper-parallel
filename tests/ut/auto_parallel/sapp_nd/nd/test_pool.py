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
"""Unit tests for a search priced in worker processes (run_nd -j)."""
import json
import os
import pickle
import tempfile
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

import yaml

from hyper_parallel.auto_parallel.sapp_nd.nd import _pool
from hyper_parallel.auto_parallel.sapp_nd.nd import debug as Debug
from hyper_parallel.auto_parallel.sapp_nd.nd import dimensions as Dim
from hyper_parallel.auto_parallel.sapp_nd.nd import parallelize as Par
from hyper_parallel.auto_parallel.sapp_nd.nd.logger import set_verbose_level

DEEPSEEK_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deepseek.yaml")
RATIOS = {"COMPUTE": 2.0, "DP_COMM": 0.5, "DP_REDUCE": 1.5, "MP_COMM": 1.0,
          "EP_COMM": 0.25, "CP_COMM": 1.0, "PP_COMM": 1.0, "BUBBLE": 1.0}


class _InlinePool:
    """A pool that runs each task in this process, the task and its function through pickle as a worker gets them."""

    def __init__(self) -> None:
        """No task run yet, and not stopped."""
        self.tasks = 0
        self.stopped = None

    def imap(self, func: Any, tasks: List[Any]) -> Any:
        """Each task's result, in order."""
        for task in tasks:
            self.tasks += 1
            yield pickle.loads(pickle.dumps(func))(pickle.loads(pickle.dumps(task)))

    def close(self) -> None:
        """Stopped once its tasks are done."""
        self.stopped = "close"

    def terminate(self) -> None:
        """Stopped at once."""
        self.stopped = "terminate"

    def join(self) -> None:
        """Nothing to wait for."""


class TestPool(unittest.TestCase):
    """A search priced in workers ranks, scores and records as one process does."""

    def setUp(self) -> None:
        """A scratch folder, verbosity 2 so that the search writes its debug CSVs, and no worker runner yet."""
        set_verbose_level(2)
        folder = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        self.addCleanup(folder.cleanup)
        self.folder = folder.name
        for patcher in (patch.object(_pool, "_RUNNER", None), patch.dict(os.environ, {"ND_CENSUS_CACHE": "off"})):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._reset)
        self.ratios = os.path.join(self.folder, "ratios.json")
        with open(self.ratios, "w", encoding="utf-8") as handle:
            json.dump(RATIOS, handle)

    @staticmethod
    def _reset() -> None:
        """No bounds and no output folder left over for the next test."""
        for dim in Dim.ALL_DIMS:
            dim.reset_bound()
        Debug.set_output_dir(None)
        set_verbose_level(1)

    def _recipe(self, name: str) -> Dict[str, Any]:
        """run_nd's values for the DeepSeek test config cut to 7 layers and 16 experts, on 16 devices."""
        with open(DEEPSEEK_YAML, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        config["model"]["model_config"].update(num_layers=7, offset=0)
        config["parallel_config"].update(data_parallel=4, model_parallel=2, pipeline_stage=2,
                                         expert_parallel=2, micro_batch_num=4)
        config["moe_config"]["expert_num"] = 16
        path = os.path.join(self.folder, "deepseek_small.yaml")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle)
        output = os.path.join(self.folder, name)
        os.makedirs(output, exist_ok=True)
        return {
            "framework": "mindformers", "yaml_config": path, "devices": 16, "device_type": "A2",
            "global_batch_size": 16, "dimensions": ["DP", "MP", "PP", "EP", "OP"], "swap_os": False,
            "mppb": False, "model": None, "max_mem": None, "mem_for_ppb": "0GB", "verbosity": 2,
            "output_dir": output, "strict": False,
        }

    def _search(self, recipe: Dict[str, Any], pool: Any = None, count: int = 4, pool_from: int = 1) -> tuple:
        """The search's ranking and the two debug CSVs it wrote, priced here or by *count* workers.

        *pool* is the processes the workers run in, an inline pool, or ``True`` for spawned ones;
        *pool_from* is the fewest candidates they price, one so that they price this small search.
        """
        runner = _pool.build_runner(recipe)
        if pool is not None:
            runner.workers = _pool.Workers(count, recipe, pool=None if pool is True else pool)
        try:
            with patch.object(Par, "POOL_FROM", pool_from):
                scored, parts, _, _ = runner._search_and_order(  # pylint: disable=protected-access
                    None, count if pool is not None else None, self.ratios)
        finally:
            runner.close_pool(abort=True)
        ranking = [(config.unique_name(), memory, score, values) for config, memory, score, values in scored]
        csvs = []
        for name in ("debug.csv", "debug_mem.csv"):
            with open(os.path.join(recipe["output_dir"], name), encoding="utf-8") as handle:
                csvs.append(handle.read())
        return ranking, parts, csvs

    def test_a_pooled_search_ranks_as_one_process_does(self):
        """
        Feature: run_nd -j, the search priced in workers.
        Description: Search the small DeepSeek with a ratios file in one process, then again with its
            candidates priced by workers, each task and function passed through pickle as a worker
            gets them.
        Expectation: The same ranking, scores and parts, and the same debug.csv and debug_mem.csv,
            row for row.
        """
        alone = self._search(self._recipe("alone"))
        pool = _InlinePool()
        pooled = self._search(self._recipe("pooled"), pool)
        self.assertGreater(pool.tasks, 2)
        self.assertTrue(alone[0])
        self.assertEqual(pooled, alone)

    def test_workers_in_their_own_processes_price_as_this_one(self):
        """
        Feature: run_nd -j, the search priced in workers.
        Description: The same search with two spawned workers, as run_nd starts them.
        Expectation: The same ranking and debug CSVs as in one process.
        """
        alone = self._search(self._recipe("alone"))
        self.assertEqual(self._search(self._recipe("pooled"), True, 2), alone)

    def test_a_small_search_prices_here_and_stops_the_workers(self):
        """
        Feature: POOL_FROM.
        Description: The small DeepSeek, whose candidates are fewer than the workers' start pays for,
            searched with workers at the default threshold.
        Expectation: No task reaches the workers, which are stopped at once, and the ranking and
            debug CSVs are those of one process.
        """
        alone = self._search(self._recipe("alone"))
        pool = _InlinePool()
        self.assertEqual(self._search(self._recipe("pooled"), pool, pool_from=Par.POOL_FROM), alone)
        self.assertEqual((pool.tasks, pool.stopped), (0, "terminate"))

    def test_a_search_the_workers_cannot_rebuild_prices_here(self):
        """
        Feature: _pooled.
        Description: A search with workers, then asked for one, without workers, choosing the
            recompute per layer, and with a recompute dimension.
        Expectation: Only the first prices in the workers.
        """
        recipe = self._recipe("rules")
        runner = _pool.build_runner(recipe)
        runner.workers = _pool.Workers(4, recipe, pool=_InlinePool())
        self.assertTrue(runner._pooled(4))  # pylint: disable=protected-access
        self.assertFalse(runner._pooled(1))  # pylint: disable=protected-access
        for name, value in (("workers", None), ("auto_recompute", True), ("recompute_dimension", ("full",))):
            with patch.object(runner, name, value):
                self.assertFalse(runner._pooled(4), name)  # pylint: disable=protected-access

    def test_the_default_number_of_workers(self):
        """
        Feature: _pool.default_workers.
        Description: No ND_JOBS on 16 logical cores and on 2, the same above verbosity 2, and ND_JOBS
            stated.
        Expectation: 4, then 1, then 1, then what ND_JOBS states.
        """
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ND_JOBS", None)
            with patch.object(os, "cpu_count", return_value=16):
                self.assertEqual(_pool.default_workers(2), 4)
                self.assertEqual(_pool.default_workers(3), 1)
            with patch.object(os, "cpu_count", return_value=2):
                self.assertEqual(_pool.default_workers(2), 1)
        with patch.dict(os.environ, {"ND_JOBS": "6"}):
            self.assertEqual(_pool.default_workers(3), 6)

    def test_chunks_cover_every_candidate_in_order(self):
        """
        Feature: _pool.chunks.
        Description: 100 candidates for four workers, and 3 for four.
        Expectation: Every candidate once and in order, about eight chunks a worker for the first.
        """
        items = list(range(100))
        chunks = _pool.chunks(items, 4)
        self.assertEqual([item for chunk in chunks for item in chunk], items)
        self.assertGreaterEqual(len(chunks), 25)
        self.assertEqual(_pool.chunks([1, 2, 3], 4), [[1], [2], [3]])


if __name__ == "__main__":
    unittest.main()
