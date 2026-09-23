# Cropped Qwen3-MoE training demos

These examples build a layer-cropped Qwen3-30B-A3B model with
`HyperAutoModelForCausalLM.from_config`. They read the complete Hugging Face
configuration, keep all original hidden, attention, vocabulary, and expert
dimensions, and change only `num_hidden_layers` (four by default). No model
checkpoint tensor is loaded, so model parameters are randomly initialized.

## Placement validation from YAML

YAML can directly control placement validation:

```yaml
model:
  _target_: examples.training_demo.cropped_qwen3_moe.build_cropped_qwen3_moe
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
bash examples/training_demo/run_parallel_offline.sh /path/to/Qwen3-30B-A3B
bash examples/training_demo/run_parallel_online.sh /path/to/Qwen3-30B-A3B
```

The default topology uses eight devices with TP=2, CP=2, EP=2, and FSDP. To
enable placement validation without editing YAML:

```bash
bash examples/training_demo/run_parallel_offline.sh \
    /path/to/Qwen3-30B-A3B \
    --model.validate_placement=true
```

Additional typed overrides are forwarded to the Trainer. For example, a
one-step smoke test is:

```bash
bash examples/training_demo/run_parallel_offline.sh \
    /path/to/Qwen3-30B-A3B \
    --training.train_iters=1
```

Logs and generated data are stored under `output/training_demo`.

## Full pretrained model

Both full-model launchers load all 48 layers and the complete Hugging Face
checkpoint through `HyperAutoModelForCausalLM.from_pretrained`. Online uses the
packaged `hyper_parallel/models/qwen3_moe/recipes/train.yaml`;
Offline uses `examples/training_demo/train_parallel_full_offline.yaml` because
the two data paths instantiate different Dataset, DataLoader, and collate
targets. Ordinary values can be overridden on the command line, but the typed
configuration interface intentionally does not replace `_target_` values.

The Online launcher tokenizes and packs a deterministic local JSONL file at
runtime. It generates that file under `output/training_demo/data` when needed
and uses a 128-token smoke-test length by default; an appended typed override
can increase the sequence length:

```bash
bash examples/training_demo/run_parallel_full_online.sh \
    /path/to/Qwen3-30B-A3B
```

The Offline launcher requires an existing Indexed Dataset and never generates
or downloads one implicitly. Pass the dataset prefix without the `.bin` or
`.idx` suffix:

```bash
bash examples/training_demo/run_parallel_full_offline.sh \
    /path/to/Qwen3-30B-A3B \
    /path/to/offline_text_document
```

Both launchers validate the local model `config.json` before starting and force
model/tokenizer loading into `local_files_only` mode. The Offline launcher also
validates both Indexed Dataset files. Missing local assets therefore fail
explicitly rather than triggering a network download. Additional typed Trainer
overrides may be appended to either command.

## Cropped Qwen3.5-MoE

`cropped_qwen3_5_moe.py` and `train_qwen3_5_moe.yaml` build a layer-cropped
Qwen3.5-35B-A3B text tower the same way, from configuration only. Qwen3.5-MoE
alternates three `linear_attention` (Gated DeltaNet) layers with one
`full_attention` layer, so `num_hidden_layers` must stay a multiple of four;
the builder truncates `layer_types` with it. Only the text tower is built: the
vision tower and the MTP layer are not part of the causal-LM class.

```bash
bash examples/training_demo/run_qwen3_5_moe.sh
```

The launcher generates the Indexed Dataset when absent and defaults to the
shared cluster model path. Pass a directory as the first argument, or set
`MODEL_PATH`, to use another copy. `--model.num_experts=32` makes a first pass
cheaper.

### Topology

The config ships one node of 16 dies with `ep_size: 8`. When scaling out:

| field | rule |
|---|---|
| `accelerator.ep_size` | divides `num_experts` (256) and the world size |
| `fsdp_config.dp_shard_size` | the world size |
| `fsdp_config.edp_shard_size` | world size / `ep_size`; 1 leaves expert optimizer state unsharded |
| `training.global_batch_size` | a multiple of `micro_batch_size` x `dp_world_size` |

`tp_size` and `pp_size` stay at 1. TP breaks the Gated DeltaNet grouped
convolution, because `conv1d` is sharded while `groups` and `conv_dim` stay
global. The Trainer has no pipeline schedule, so `pp_size` above 1 does not
raise and does not pipeline: each stage group trains a full replica.

### Multinode

`cluster_qwen3_5_moe.env` is a cluster-kit run config for a four node pool.

```bash
cluster -c examples/training_demo/cluster_qwen3_5_moe.env torchrun -n 4 \
    scripts/train_lm.py examples/training_demo/train_qwen3_5_moe.yaml \
    --fsdp_config.dp_shard_size=64 \
    --fsdp_config.edp_shard_size=8 \
    --training.global_batch_size=64
```

Every node needs the repository, the generated dataset and the compiled
`_indexed_helpers_cpp` extension. `cluster sync` delivers none of the last two:
it honours `.gitignore`, which excludes `output/` and `*.so`, and it never
deletes, so files removed or renamed upstream survive on the targets. A stale
`hyper_parallel/core/shard/ops/yaml/` is fatal rather than subtle, because the
op registry globs that directory and rejects any duplicate operator name.
Mirror the tree instead:

```bash
for h in <nodes>; do
    rsync -a --delete --exclude '.git/' \
        /path/to/hyper-parallel/ root@$h:/path/to/hyper-parallel/
done
```

Read the kit's node logs first when a multinode launch fails. A node whose
environment setup fails exits before training starts and reports it only there.
