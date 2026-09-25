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
"""Tests for the memory of a schedule that splits the sequence (Seq1F1B).

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/memory_estimation/test_seq1f1b.py -v
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import yaml

from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.estimate_v2 import EvaluatorV2
from hyper_parallel.auto_parallel.sapp_nd.memory_estimation.evaluators.utils import EvalUtils

_DEEPSEEK = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "nd", "deepseek.yaml")


def _seqpipe(folder: str, chunks: int) -> str:
    """The DeepSeek yaml under the seqpipe schedule, its sequence in *chunks*."""
    with open(_DEEPSEEK, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    data.setdefault("parallel", {})["pipeline_config"] = {
        "pipeline_interleave": True, "pipeline_scheduler": "seqpipe",
    }
    data["parallel_config"]["seq_split_num"] = chunks
    path = os.path.join(folder, "seqpipe.yaml")
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)
    return path


class TestSeq1F1B(unittest.TestCase):
    """Every layer of a Seq1F1B run is priced per chunk of the same sequence."""

    def test_every_layer_sees_the_whole_sequence(self):
        """
        Feature: the seq1f1b micro factor and the memory walk.
        Description: Estimate the DeepSeek yaml under seqpipe with its
            sequence in 4 chunks, recording the sequence length each layer's
            micro factor sees.
        Expectation: Every layer sees the config's whole sequence, which no
            layer before it has shrunk.
        """
        seen = []
        factor = EvalUtils.pp_seq1f1b_micro_factor

        def record(ccfg, ctx):
            seen.append(ccfg.s)
            return factor(ccfg, ctx)

        with tempfile.TemporaryDirectory() as folder, \
                patch.object(EvalUtils, "pp_seq1f1b_micro_factor", staticmethod(record)):
            evaluator = EvaluatorV2(_seqpipe(folder, 4), framework="mindformers", log_level=0)
            evaluator.estimate_peak()
        self.assertGreater(len(seen), 1)
        self.assertEqual(set(seen), {evaluator.ccfg.s}, f"sequence lengths seen: {sorted(set(seen))[:4]}")


if __name__ == "__main__":
    unittest.main()
