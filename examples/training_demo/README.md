# Training demos

The root keeps the shared `train_text.py` entry point. Model-owned builders,
recipes, data preparation, launchers, and validation manifests are grouped by
family:

- `qwen3_moe/`: cropped and full-model Qwen3-MoE text demos;
- `deepseek_v41/`: DeepSeek-V4.1 text/VLM crops and validation tooling.

## Qwen3-MoE cropped Hugging Face demos

These examples build a layer-cropped Qwen3-30B-A3B model with
`HyperAutoModelForCausalLM.from_config`. They read the complete Hugging Face
configuration, keep all original hidden, attention, vocabulary, and expert
dimensions, and change only `num_hidden_layers` (four by default). No model
checkpoint tensor is loaded, so model parameters are randomly initialized.

## Placement validation from YAML

YAML can directly control placement validation:

```yaml
model:
  _target_: examples.training_demo.qwen3_moe.cropped_qwen3_moe.build_cropped_qwen3_moe
  validate_placement: true
```

The value is retained by `Target`, passed through `BaseTrainer._build_model`,
forwarded by `build_cropped_qwen3_moe` to `HyperAutoModelForCausalLM.from_config`,
and finally used as the infrastructure `validate_mode` value. The typed CLI
override `--model.validate_placement=true` follows the same path. The launch
scripts do not override this field, so editing the YAML remains effective.

## Data modes

- Offline uses the Indexed Dataset format. Its targets are already shifted,
  so `labels_are_shifted: true`; the implicit-mask CP wrapper owns the causal
  mask.
- Online tokenizes and packs JSONL text at runtime. Its transform emits
  pre-shifted labels, and packed document boundaries use a global block mask accepted by
  `qwen3_moe_flash_attention_cp_mask_wrapper`.

Both launchers automatically generate small deterministic local datasets when
the expected files are absent. This process does not access or download an
external dataset.

## Run

Pass a local Qwen3-30B-A3B Hugging Face directory containing `config.json` and
tokenizer assets as the first argument. The examples deliberately set
`local_files_only: true`: missing assets cause an explicit error instead of an
implicit model or weight download. Prepare the runtime according to the
project installation guide before launching the example.

```bash
bash examples/training_demo/qwen3_moe/run_parallel_offline.sh /path/to/Qwen3-30B-A3B
bash examples/training_demo/qwen3_moe/run_parallel_online.sh /path/to/Qwen3-30B-A3B
```

The default topology uses eight devices with TP=2, CP=2, EP=2, and FSDP. To
enable placement validation without editing YAML:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_offline.sh \
    /path/to/Qwen3-30B-A3B \
    --model.validate_placement=true
```

Additional typed overrides are forwarded to the Trainer. For example, a
one-step smoke test is:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_offline.sh \
    /path/to/Qwen3-30B-A3B \
    --training.train_iters=1
```

Logs and generated data are stored under `output/training_demo`.

## Full pretrained model

Both full-model launchers load all 48 layers and the complete Hugging Face
checkpoint through `HyperAutoModelForCausalLM.from_pretrained`. Online uses the
packaged `hyper_parallel/models/qwen3_moe/recipes/train.yaml`;
Offline uses `examples/training_demo/qwen3_moe/train_parallel_full_offline.yaml` because
the two data paths instantiate different Dataset, DataLoader, and collate
targets. Ordinary values can be overridden on the command line, but the typed
configuration interface intentionally does not replace `_target_` values.

The Online launcher tokenizes and packs a deterministic local JSONL file at
runtime. It generates that file under `output/training_demo/data` when needed
and uses a 128-token smoke-test length by default; an appended typed override
can increase the sequence length:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_full_online.sh \
    /path/to/Qwen3-30B-A3B
```

The Offline launcher requires an existing Indexed Dataset and never generates
or downloads one implicitly. Pass the dataset prefix without the `.bin` or
`.idx` suffix:

```bash
bash examples/training_demo/qwen3_moe/run_parallel_full_offline.sh \
    /path/to/Qwen3-30B-A3B \
    /path/to/offline_text_document
```

Both launchers validate the local model `config.json` before starting and force
model/tokenizer loading into `local_files_only` mode. The Offline launcher also
validates both Indexed Dataset files. Missing local assets therefore fail
explicitly rather than triggering a network download. Additional typed Trainer
overrides may be appended to either command.

## DeepSeek-V4.1 Engram and shared compressed attention

`deepseek_v41/train_deepseek_v41_online.yaml` is a four-layer, randomly initialized
DeepSeek-V4.1 text crop. Four layers are the minimum that execute all requested
paths: layer 1 owns Engram, layer 2 publishes compressed KV and Lightning
Indexer selections plus compact hierarchical candidate blocks, and layer 3
performs a fresh Reindex over that candidate pool. The crop also enables the
PanGu-style selected-TopK Indexer KL training path. Vision, DSpark, the second
Engram layer, later compressed-attention source groups, PP-stage shadow
indexers, runtime KV-cache decode, and FP4 QAT remain outside this validation
crop. Raw KV, shared compressed KV, and compressed index K use PanGu-style
asynchronous KV-all-gather CP.

The Engram table is scaled consistently instead of truncating a checkpoint
table. `deepseek_v41/prepare_deepseek_v41_assets.py` changes every active hash bucket to a
different prime near 4096, then recomputes offsets and the embedding row count.
For the active layer-1 table this changes 384,006,168 rows to 100,776 rows while
retaining 3 n-gram orders, 8 hash heads, and a 256-wide embedding. The tokenizer
normalization and hash multipliers remain the V4.1 values.

Online packing emits compact sample boundaries instead of a dense `[S,S]`
attention mask. Each packed sample is aligned to the encoder compression ratio,
so neither CSA2 compressor groups nor Engram n-grams cross sample boundaries.

Prepare the shell using the project
[`installation guide`](../../docs/installation.md), then run TP1 first. A
successful TP1 run creates a marker required by TP2:

```bash
bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
    /path/to/DeepSeek-V4.1-Flash tp1

bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
    /path/to/DeepSeek-V4.1-Flash tp2

bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
    /path/to/DeepSeek-V4.1-Flash cp2
```

All modes use 16 processes, Online tokenization, a 4096-token sequence, EP=16,
and the mandatory FP32-main-parameter policy. TP1 uses FSDP=16; TP2 uses
FSDP=8 and enables sequence parallel so Engram's replicated fusion projections
operate on disjoint token slices, matching PanGu's `SequenceParallelLinear`
contract. CP2 keeps TP=1 and uses asynchronous Colossal/KV-all-gather CP; it
does not use Ulysses sequence-to-head exchange. The launcher reads only local
config/tokenizer files and creates the scaled Engram metadata plus deterministic
Online JSONL under `output/training_demo/deepseek_v41`.

The model-owned Engram, CSA2, TP/CP/EP, checkpoint, and validation declarations
live together in the
[`deepseek_v41` adapter](../../hyper_parallel/models/deepseek_v41/adapter/).
The generic workflow is documented in the
[`model-integration validation guide`](../../docs/guide/trainer/model_integration_validation.md).

## DeepSeek-V4.1 multimodal Online smoke

The custom validation crop adds the V4.1 ViT, 3x3 aligner, image-boundary
embeddings, image-aware MoE routing, and OpenAI-messages image data transform.
It uses one vision block and 16 routed experts for the validation crop while
retaining the released model dimensions. The model adapter declares per-vision
block, aligner, and mixed-mesh Engram FSDP units plus the actual forward order;
the generic FSDP manager contains no DeepSeek-specific branches.

Prepare the environment and the local image JSONL, then run:

```bash
bash examples/training_demo/deepseek_v41/run_deepseek_v41_vlm_online.sh \
    /path/to/DeepSeek-V4.1-Flash \
    /path/to/train.jsonl
```

The default data path is
`output/training_demo/deepseek_v41/mm_data/deepseek_v41_messages/train.jsonl`.
The launcher removes a stale success marker before starting and recreates it
only after all 16 ranks complete. Use
[`deepseek_v41_validation.yaml`](deepseek_v41/deepseek_v41_validation.yaml) to
generate reproducible structure, module-parity, precision, checkpoint, and
performance evidence for the selected local environment.

## DeepSeek-V4.1 fused operators on the nd/dsv41-fused branch

This branch merges the three open fused-operator pull requests on top of
`trainer_dev`, so one checkout carries all of them:

- GitCode !1435 and GitHub #990, the CANN compressed attention adapter plus the
  halo context-parallel gather, which starts the raw KV bank at `raw_start`;
- GitHub #1016, the native LI V2, SMLA and fused KL operators;
- GitHub #953, the pluggable Lightning Indexer provider.

All three implement the same fused Indexer. Each keeps its own contract, and
`compressed_causal_topk` offers them in a fixed order: the native operators,
which `use_fused` demands and which raise rather than fall back; then a
registered selection provider, which declines what it cannot serve; then the
CANN adapter where it is available; then the reference chunk loop. PR #953
arrived with its flag named `use_fused`, already taken by PR #1016 with the
opposite default and the opposite failure behaviour, so on this branch the
provider flag is `use_provider`.

Two of these paths are active without asking:

- the provider, which `adapter/registration.py` installs at import and which
  disables itself for the rest of the process after an operator error;
- the CANN adapter, which tests the installed interface and the input shapes.

Only the native operators need opting in. Pass `use_fused_kernels: true` to the
`SharedCompressedDSAAttention` entry under `plan_overrides` in
`train_deepseek_v41_online.yaml`. That also selects fused SMLA attention and
fused KL, and raises on an unsupported call instead of running the baseline.

A run that silently takes the reference path looks like a run that takes a fused
path, so the provider states the positive fact in the run's own log. Read the log
rather than inferring a path from the absence of a warning.

Fused attention and the halo gather are refused together. The native kernel
addresses the whole global sequence, while a halo gather starts the raw bank at
`raw_start`, and the two have never run together. `_apply_sparse_attention`
raises on that combination instead of reading the bank at the wrong offsets.

### The model directory

The crop loads no checkpoint tensor, so the model weights are never read and
three small files are enough:

| File | Why |
|------|-----|
| `config.json` | released architecture, and `model_type` must be `deepseek_v41` |
| `tokenizer.json` | the Engram hash is defined over this exact tokenizer |
| `tokenizer_config.json` | carries the end-of-sentence and padding tokens |

A quantized repack of the released repository serves equally well: its
architecture fields are the released ones, and only the unread
`engram_rotation_config` is added.

Do not leave out `tokenizer_config.json`. Without it the tokenizer still loads
and still produces the same ids, but `eos_token_id` and `pad_token_id` are None,
and `data/text/text_transform.py` appends the end-of-sentence token only when it
is not None. Documents would then run together with no separator and nothing
would report it. This check states the positive fact instead:

```bash
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('/home/tt/models/DeepSeek-V4.1-Flash', local_files_only=True, use_fast=True)
assert t.eos_token_id is not None and t.pad_token_id is not None, 'tokenizer_config.json missing or broken'
print('ok', len(t), t.eos_token_id, t.pad_token_id)"
```

It prints `ok 129280 1 1` for the released tokenizer, whose `tokenizer.json` is
6,367,257 bytes with md5 `8a8245dc7f6c6bfcb0684a4be4e17217`. Both the released
repository and its w8a8 repack carry that same file.

### Running the text crop on a 16-device host

Point the launcher at the directory holding those three files:

```bash
bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
    /home/tt/models/DeepSeek-V4.1-Flash tp1
```

The launcher refuses to start when the config or the tokenizer is missing, writes
the scaled Engram metadata and the deterministic Online JSONL under
`output/training_demo/deepseek_v41`, and reuses both on later runs. Preparing the
metadata needs `tokenizers`, `numpy` and a Transformers that carries
`models.deepseek_v4`. It also tees the log, so read the log rather than the exit
status.

### Comparing activation-checkpoint modes

`train_deepseek_v41_online.yaml` ships `mode: "off"` and one training iteration.
Override both to time the three modes at one topology:

```bash
for MODE in '"off"' selective full; do
    bash examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh \
        /home/tt/models/DeepSeek-V4.1-Flash tp1 \
        --activation_checkpoint.mode=$MODE --training.train_iters=10
done
```

Override values are parsed as YAML, where the bare word `off` is the boolean
false. The configuration normalizes that back to `"off"`, so both spellings
select the same mode, and the quoted form says what it means.
