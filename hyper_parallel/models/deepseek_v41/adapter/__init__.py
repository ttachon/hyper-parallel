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
"""DeepSeek-V4.1 model integration providers.

Providers are grouped by conversion, data, distributed execution, policies, and
validation. ``registration`` is the only discovery entry point.

Importing this package runs that entry point, as ``qwen3_moe``'s adapter package
does, so a recipe naming a ``_target_`` factory inside it installs the adapter and
the fused Lightning-Indexer provider. Without this the provider stayed unset: the
registry reaches ``registration`` only through its lazy scan by model family, which
a ``_target_`` factory never triggers, so every run took the reference selection
and said nothing about it.
"""

from hyper_parallel.models.deepseek_v41.adapter import registration  # noqa: F401  pylint: disable=unused-import
