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
"""Unit tests for the model order of a pipeline partition's layers.

How to run this:
    pytest tests/ut/auto_parallel/sapp_nd/nd/test_layer_order.py -v
"""
import unittest
from types import SimpleNamespace

from hyper_parallel.auto_parallel._exec_spec import RecomputeRange
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_order import get_model_order, layer_recompute_types
from hyper_parallel.auto_parallel.sapp_nd.nd.common.layer_type import LayerType

_EMB, _OUT, _LAY = LayerType.EMBEDDING_LAYER, LayerType.OUTPUT_LAYER, LayerType.NOT_REC_LAYER


class TestModelOrder(unittest.TestCase):
    """Model order runs chunk by chunk across the stages."""

    def test_interleaved_stages(self):
        """
        Feature: get_model_order.
        Description: Two stages of two chunks of one layer, the embedding
            on the first and the output on the last.
        Expectation: Chunk 0 of stages 0 and 1, then chunk 1 of both.
        """
        stages = [[[_EMB, _LAY], [_LAY]], [[_LAY], [_LAY, _OUT]]]
        cfg = SimpleNamespace(p=2, vp=2, pp_sched="1f1b")
        self.assertEqual(get_model_order(cfg, stages), [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 0)])

    def test_a_v_schedule_comes_back_up_the_stages(self):
        """
        Feature: get_model_order.
        Description: The same stages under a V schedule.
        Expectation: The second chunk runs from the last stage back to the
            first.
        """
        stages = [[[_EMB, _LAY], [_LAY, _OUT]], [[_LAY], [_LAY]]]
        cfg = SimpleNamespace(p=2, vp=2, pp_sched="zero_bubble_v")
        self.assertEqual(get_model_order(cfg, stages), [(0, 0, 1), (1, 0, 0), (1, 1, 0), (0, 1, 0)])


class TestLayerRecomputeTypes(unittest.TestCase):
    """Each layer takes the option of the range that covers it."""

    def test_ranges_and_the_layers_they_leave(self):
        """
        Feature: layer_recompute_types.
        Description: Six layers, the second and third recomputed in full and
            the fifth on selectively.
        Expectation: The uncovered layers are not recomputed.
        """
        ranges = (RecomputeRange(first=1, count=2, option="full"), RecomputeRange(first=4, option="selective"))
        got = [kind.name[:3] for kind in layer_recompute_types(ranges, 6)]
        self.assertEqual(got, ["NOT", "FUL", "FUL", "NOT", "SEL", "SEL"])


if __name__ == "__main__":
    unittest.main()
