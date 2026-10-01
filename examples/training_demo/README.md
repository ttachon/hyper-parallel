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

### Sweeping parallel strategies against ND

`sweep_qwen3_5_moe.py` profiles one run per strategy, classifies each into
ND's parts and prints ND's estimate beside the measurement. No axis is swept
unless it is named, so a sweep varies only what it says it varies:

```bash
python examples/training_demo/sweep_qwen3_5_moe.py --ep 1,2,4,8,16,32,64
```

Every dimension takes a list and the sweep is their cartesian product, so this
runs four strategies, and `--op` holds the FSDP shard width fixed across all
of them:

```bash
python examples/training_demo/sweep_qwen3_5_moe.py --ep 2,16 --cp 1,2 --op 16
```

One axis at a time is usually what a cost model needs. Sweeping EP alone does
not isolate EP: expert weights are all-gathered by FSDP in blocks of
`num_experts / ep`, so EP moves the data-parallel volume too, and a
disagreement along that axis cannot be attributed to either term. Sweeping OP
at fixed EP changes the shard width without changing that volume, which
separates them.

| flag | sets | default |
|---|---|---|
| `--pool` | starts by picking the nodes, see below | off |
| `--nodes` | how many nodes `--pool` picks | as many as `--cluster-env` lists |
| `--nd-top` | runs ND's N best runnable strategies, see below | off |
| `--ep` | `accelerator.ep_size` | `1` |
| `--cp` | `accelerator.cp_size` | `1` |
| `--op` | `fsdp_config.dp_shard_size`, ND's `OP` | `dp * cp` |
| `--tp`, `--pp` | refused above 1 on this model, see below | `1` |
| `--global-batch-size` | `training.global_batch_size` | the world size |
| `--layers` | `model.num_hidden_layers` | `8` |
| `--seq-len` | `dataset.data_config.seq_length`, and rebuilds the dataset | `8192` |
| `--activation-checkpoint` | `activation_checkpoint.mode` | `full` |

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

The sweep defaults to sequence 8192 with full recompute, which the
single-node demo's 128 is too small to stand in for, but keeps the crop at 8
layers so a sweep finishes in a usable time. Layer count rescales the step
rather than changing what the comparison tests, since both communication
volume and compute are linear in it; sequence length is not substitutable that
way, because attention is quadratic in it and the step is launch-bound until
it is long. Pass `--layers 32` when the absolute numbers matter more than the
turnaround. Two consequences worth knowing:

- The dataset is rebuilt by the `data` stage because its documents are exactly
  `seq_length` long, so raising the sequence without rebuilding would leave the
  reader with samples of the wrong size.
- The un-sharded cross-entropy over a 248320 vocabulary is the dominant memory
  term at long sequence and is immune to recompute. If a configuration runs out
  of memory, that is the first place to look: raising `--cp` shards the sequence
  and therefore the logits, which is why ND's own search reaches for it.

`--tp` and `--pp` above 1 are refused rather than run: TP shards the Gated
DeltaNet `conv1d` while its `groups` and `conv_dim` stay global, so the forward
raises, and `pp_size` above 1 neither raises nor pipelines because the Trainer
has no pipeline schedule. Every other combination is validated against the
trainer's startup constraints before anything launches.

Stages run in order and any can be run alone, so a sweep can be re-classified
without re-profiling and a changed cost model re-scored without re-running:

```bash
python examples/training_demo/sweep_qwen3_5_moe.py --only classify --only compare
```

| stage | what it does |
|---|---|
| `select` | with `--pool` only: picks the nodes that pass the kit's census and writes them into `--cluster-env` |
| `rank` | ND's search at the sweep's shape into `nd_ranking.csv`; runs by default only with `--nd-top` |
| `mirror` | makes every node's tree identical, excluding `output/` |
| `data` | rebuilds the Indexed Dataset on every node at `--seq-len` |
| `run` | launches each strategy, waits on the kit's rc file, stops on the first dead node |
| `fetch` | copies the profiles from the node holding `profiling.rank` |
| `classify` | `nd.trace_classify` per run, merged into one CSV |
| `compare` | `run_nd --real_csv`, printing measured against estimated shares, then ND's rank of each strategy when a ranking exists |
| `plot` | `sweep.pdf`/`.png`: the step split and the peak memory across the sweep; `memory.pdf`/`.png`: the measured peak against ND's estimate |

On a shared pool, an idle NPU is not a working one: a node can have a link
down, memory held by a process `npu-smi` does not list, an environment that
fails to set up, or a die that fails its first op. `--pool` makes the sweep
start from nodes that pass all of that now:

```bash
python examples/training_demo/sweep_qwen3_5_moe.py --pool /home/tt/cluster_all.env ...
```

The `select` stage runs the kit's `cluster select --auto N --census` over the
pool's nodes, `N` from `--nodes` or as many as `--cluster-env` lists. It runs
it under `--cluster-env` with the pool's nodes swapped in, not under the pool's
own config, so the census tests this sweep's repository directory, interpreter
and HCCL settings, and the config it writes is `--cluster-env` with the nodes
that passed. That file then replaces `--cluster-env`, so every later stage, and
every later `--only` run, uses them; the one it replaced is kept in `--out` as
`<name>.before_select`. When too few pass, the kit prints why for each node and
the sweep waits, re-probing every five minutes, until enough do. The census
runs inside the repository directory, so the stage first creates it, empty, on
every pool node that lacks it; `mirror` then fills it on the nodes picked. The
kit's `select` must have `--census`.

A strategy that dies is not waited out. One dead rank ends the job, but it
does not end the other ranks: they wait in the collective it never joins until
`HCCL_EXEC_TIMEOUT`, which `cluster_qwen3_5_moe.env` sets to 180 s against a
default of half an hour. Even three minutes per failed strategy adds up over a
sweep broken the same way at every point, so the sweep watches for a
node the kit reports as `DEAD` rather than for every node to stop running,
kills what is left of that run, which is also what frees the devices for the
next strategy, and moves on. It prints the exception it found in the failed
nodes' logs, deduplicated across ranks, since the status line carries only
each node's last log line and after a crash that is a stack frame rather than
the cause. Failed strategies are listed together at the end of the pass and
marked in `run_states.json`; the classify stage already skips a strategy that
produced no profile.

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

`run_nd` writes its own per-configuration comparison to `nd/`:
`real_all.pdf`, measured against estimated parts with the idle remainder;
`real_all_no_idle.pdf`, the same without idle and ordered by the step less
idle, since idle is half of a short step and does not reproduce between runs;
and `real_all_estimates.csv`, ND's estimate of every measured strategy with its
peak memory. The `plot` stage adds the view across the sweep that no single
configuration shows: the measured step split into ND's parts, and peak device
memory with ND's estimate beside it, with the x axis labelled by whichever
dimensions actually vary. Memory also gets `memory.pdf` on its own. The trainer
logs the maximum over ranks in GiB while ND models one rank, in MiB converted
to GiB, with a 1 GiB safety margin in its peak.

The ND input yaml, `nd_model.yaml`, is generated rather than reused. The launch
overrides the config's layer count, sequence length, recompute mode and batch
on the command line, so the config file alone describes another run, 128
tokens without recompute against a default sweep of 8192 with full recompute.
The generated yaml is the config with the values the launch uses, and it
states the world size so that ND derives the data-parallel width the trainer
does.

### Profiling the strategies ND ranks best

A sweep can also let ND choose what to run. `--nd-top N` runs ND's search at
the sweep's shape and profiles the N strategies it ranks best among those this
model and trainer can run, which tests the part of its ranking a search relies
on. The named degrees keep their meaning. A degree named with one value holds
for ND's picks as for the grid, so `--op 16` runs every strategy at OP 16, and
ND's best there; naming axes also adds their grid beside the picks, so this
runs ND's best at OP 16 and EP 4 to 64 at OP 16:

```bash
python examples/training_demo/sweep_qwen3_5_moe.py --nd-top 5 --ep 4,8,16,32,64 --op 16 \
    --python /home/tt/envs/hp2/bin/python3.11
python examples/training_demo/sweep_qwen3_5_moe.py --nd-top 5 --only rank   # preview only
```

An axis named with a list only adds its grid: ND's picks keep their own value
on it. To measure ND's picks against a strategy already known to be good, name
that strategy's axis with its value and another, `--ep 1,16`, rather than
`--ep 16`, which would hold ND's picks to EP 16.

The `rank` stage writes ND's whole order to `nd_ranking.csv`, through
`run_nd --ranking_csv`, and the shape it was made for beside it. A later stage
refuses a ranking made for another shape rather than read it as this one's. The
search covers ND's whole space, TP and PP included, and the choice among its
configurations follows three rules:

- A configuration this model or trainer cannot run, or one away from a degree
  the sweep fixes, is passed over and listed with the reason, so a preference of
  ND's that is not tested is visible.
- Configurations that are one strategy to the trainer are run once: SP on and
  off at TP 1 run identically.
- A tie is one prediction, so it is run once. At any EP, ND gives every OP
  above 1 the same time, since its estimate depends on whether FSDP shards and
  not on how widely; the tie is run at its widest OP, the FSDP default and the
  one holding the least memory, or at the OP the sweep fixes, and the other
  widths are printed with it.

At PP 1, ND's search keeps the micro-batch count at 1 and grows the
micro-batch size instead, so a CP 2 strategy runs two sequences per
micro-batch where the grid would accumulate two micro-batches of one. Each
strategy therefore carries its own micro-batch size to the launch and to the
classified CSV as `MBS`. A strategy directory gains an `_mbs<N>` suffix only
when that size is above 1.

After `compare`, the sweep sets ND's rank of every measured strategy beside the
measured order, whether or not ND chose it, and writes the table to
`nd_vs_measured.csv`: ND's rank, score and memory, the measured step and its
place, and the peak memory the trainer logged. It then says how much slower
the strategy ND ranks best measures than the fastest one measured, and gives
the rank correlation of ND's score with the step. A strategy with no ND rank
is one its search does not generate or believes does not fit. With the Muon
optimizer ND caps OP at `dp / ep`, a bound taken from MindSpeed's Muon, so it
never generates OP 16 at EP 8 or above although the trainer runs it.
`--framework` defaults to `hyper_v2`, the parser that reads this schema;
run_nd's own default reads a different one.
