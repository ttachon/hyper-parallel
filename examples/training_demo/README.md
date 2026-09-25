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

Every node needs the repository, the compiled `_indexed_helpers_cpp`
extension and the generated dataset. `cluster sync` delivers neither of the
last two:
it honours `.gitignore`, which excludes `output/` and `*.so`, and it never
deletes, so files removed or renamed upstream survive on the targets. A stale
`hyper_parallel/core/shard/ops/yaml/` is fatal rather than subtle, because the
op registry globs that directory and rejects any duplicate operator name.

Mirror the code, excluding run outputs so the mirror can never delete a node's
dataset:

```bash
for h in <nodes>; do
    rsync -a --delete --exclude '.git/' --exclude 'output/' \
        /path/to/hyper-parallel/ root@$h:/path/to/hyper-parallel/
done
```

Then generate the dataset once on every node. It is deterministic, so every
node produces identical files and nothing has to ship it:

```bash
cluster -c examples/training_demo/cluster_qwen3_5_moe.env exec \
    'python -m examples.training_demo.prepare_parallel_data \
        --output-dir ./output/training_demo/data --num-samples 1024 --seq-length 128'
```

`cluster exec` runs inside `REPO_DIR` with the environment hook sourced, so both
the relative path and the interpreter resolve correctly.

Read the kit's node logs first when a multinode launch fails. A node whose
environment setup fails exits before training starts and reports it only there.

### Sweeping expert parallelism against ND

`sweep_qwen3_5_moe_ep.py` profiles one run per `ep_size`, classifies each into
ND's parts and prints ND's estimate beside the measurement:

```bash
python examples/training_demo/sweep_qwen3_5_moe_ep.py --ep 1,2,4,8,16,32,64
```

Every swept dimension takes a list and the sweep is their cartesian product,
so this runs four strategies, and `--op` holds the FSDP shard width fixed
across all of them:

```bash
python examples/training_demo/sweep_qwen3_5_moe_ep.py --ep 2,16 --cp 1,2 --op 16
```

| flag | sets | default |
|---|---|---|
| `--ep` | `accelerator.ep_size` | `1,2,4,8,16,32,64` |
| `--cp` | `accelerator.cp_size` | `1` |
| `--op` | `fsdp_config.dp_shard_size`, ND's `OP` | `dp * cp` |
| `--tp`, `--pp` | refused above 1 on this model, see below | `1` |
| `--global-batch-size` | `training.global_batch_size` | the world size |

The derived degrees follow the trainer's own arithmetic, so a strategy is
labelled in the classified CSV as it actually ran:

- `dp = world / (tp * cp * pp)`, which is what the dataloader splits the batch
  over and what ND calls `DP`. Raising `cp` therefore lowers `dp`.
- `MB = global_batch_size / (micro_batch_size * dp)`. The default global batch
  is the world size, which holds the work per step fixed, so raising `cp`
  raises `MB` rather than shrinking the step.
- `op` divides `dp * cp`, not `dp`: FSDP shards over the data and context axes
  together.
- `edp_shard_size = world / ep`, since expert weights are sharded by EP over
  the whole device mesh.

`--tp` and `--pp` above 1 are refused rather than run: TP shards the Gated
DeltaNet `conv1d` while its `groups` and `conv_dim` stay global, so the forward
raises, and `pp_size` above 1 neither raises nor pipelines because the Trainer
has no pipeline schedule. Every other combination is validated against the
trainer's startup constraints before anything launches.

Stages run in order and any can be run alone, so a sweep can be re-classified
without re-profiling and a changed cost model re-scored without re-running:

```bash
python examples/training_demo/sweep_qwen3_5_moe_ep.py --only classify --only compare
```

| stage | what it does |
|---|---|
| `mirror` | makes every node's tree identical, excluding `output/` |
| `run` | launches each degree and waits on the kit's rc file, not the pid |
| `fetch` | copies the profiles from the node holding `profiling.rank` |
| `classify` | `nd.trace_classify` per run, merged into one CSV |
| `compare` | `run_nd --real_csv`, printing measured against estimated shares |

Memory is measured in its own pass. Recording the allocator history brackets
the profiled window and moves both step time and idle, so `--profile-memory`
defaults to `separate`: the timing pass runs with it off, then the whole sweep
repeats with it on into `ep<N>_mem` directories. Use `same` to fold it into one
pass when the step time does not matter, or `none` to skip it.

Peak device memory needs neither pass: the trainer reports it every step, so
the sweep harvests `memory/device_max_allocated_gb` and
`memory/device_max_reserved_gb` from the timing run's log into `memory.csv`, at
no cost to the numbers being timed. The memory pass is for the detail that the
peak alone does not give: `operator_memory.csv`, `memory_record.csv` and the
allocator snapshot `.pkl`, which opens at pytorch.org/memory_viz.

The ND input yaml is generated rather than reused: ND reads the training
sequence length from `data.max_seq_len`, while the demo feeds an Indexed
Dataset whose length lives in `dataset.data_config.seq_length`. Without
carrying it across, the model would be costed at its 262144 context limit.
`--framework` defaults to `hyper_v2`, the parser that reads this schema;
run_nd's own default reads a different one.
