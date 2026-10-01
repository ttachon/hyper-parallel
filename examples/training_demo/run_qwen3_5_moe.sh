#!/bin/bash
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

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
OUTPUT_DIR="${PROJECT_ROOT}/output/training_demo"
DATA_ROOT="${OUTPUT_DIR}/data"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"

# Shared cluster layout: every node mounts the assets at the same path, so the
# default needs no argument. Pass a directory as the first argument, or set
# MODEL_PATH, to run against another copy.
DEFAULT_MODEL_PATH="/home/tt/models/Qwen3.5-35B-A3B-Base"

# A leading argument is the model directory unless it is a trainer override.
if [[ $# -ge 1 && "$1" != --* ]]; then
    MODEL_PATH=$1
    shift
fi
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"

MODEL_PATH=$(cd "${MODEL_PATH}" 2>/dev/null && pwd) || {
    echo "Model directory does not exist: ${MODEL_PATH}" >&2
    echo "Pass one as the first argument or set MODEL_PATH." >&2
    exit 1
}
if [[ ! -s "${MODEL_PATH}/config.json" ]]; then
    echo "Qwen3.5-MoE config.json is missing: ${MODEL_PATH}/config.json" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${DATA_ROOT}"
# global_batch_size 16 x train_iters 10 needs 160 samples, so generate 256.
if [[ ! -s "${DATA_ROOT}/parallel_offline_text_document.bin" \
        || ! -s "${DATA_ROOT}/parallel_offline_text_document.idx" ]]; then
    python -m examples.training_demo.prepare_parallel_data \
        --output-dir "${DATA_ROOT}" \
        --num-samples 256 \
        --seq-length 128
fi

torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    scripts/train_lm.py \
    "${SCRIPT_DIR}/train_qwen3_5_moe.yaml" \
    --model.config_path="${MODEL_PATH}" \
    --dataset.model_assets.tokenizer.pretrained_model_name_or_path="${MODEL_PATH}" \
    --dataset.data_path="${DATA_ROOT}/parallel_offline_text_document" \
    "$@" \
    2>&1 | tee "${OUTPUT_DIR}/run_qwen3_5_moe.log"
