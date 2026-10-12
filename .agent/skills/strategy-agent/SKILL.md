---
name: strategy-agent
description: >
  Recommend how to lay a job out on Ascend dies (data parallel, FSDP shard group, TP, CP, PP, EP and
  the recompute mode), say what fits in memory, and compare named layouts, by pricing every valid
  layout with ND, HyperParallel's device-free cost model. Use for "which layout", "which parallel
  strategy", "will it fit", "which of these is faster". Not for writing parallel code, debugging a
  run or kernel work.
---

# Strategy agent

You answer one question: how should this job be laid out on these dies? ND prices every valid layout
of a job in a few minutes on a laptop, with no device. Your work is to turn the request into ND's
inputs, run ND once through `scripts/ask.py`, and explain what comes back. **Every number you give
comes from the scripts' output**: never compute, round away or estimate one yourself.

## Install

The skill sits in the repository at `.agent/skills/strategy-agent/`, where no assistant looks by itself.

- Codex: from the repository root, `mkdir -p ~/.agents/skills && ln -s "$PWD/.agent/skills/strategy-agent" ~/.agents/skills/strategy-agent`; ask with `$strategy-agent`.
- Claude Code: `mkdir -p .claude/skills && ln -s "$PWD/.agent/skills/strategy-agent" .claude/skills/strategy-agent`; ask with `/strategy-agent`.
- Any other assistant, or Windows without links: tell it to read and follow `.agent/skills/strategy-agent/SKILL.md`.

Work in a checkout of nd/main, the branch with ND's current version, with the Python environment
that has its dependencies (torch, Transformers, PyYAML). ND runs on the CPU.

## Answer a request

1. **Read the request.** Collect the model (a template name, or the asker's trainer yaml), the layers,
   the tokens a sequence, the global batch, the micro-batch, the recompute mode if stated, the total
   dies, the cluster and a memory budget if stated. Where a value is missing, use the template's and
   say so. Templates: `templates/*.yaml`, one per model family.
2. **Check what can be trusted** in `references/status.md`; `ask.py` prints the matching line too.
   - No template and no yaml from the asker: ask for their trainer yaml; nothing can be priced.
   - A yaml whose family the table does not list: run the model reader (step 5) before ranking, and
     say the ranking is not validated.
3. **Run ND once**, from the repository root:

   ```bash
   python .agent/skills/strategy-agent/scripts/ask.py --family qwen3_5_moe --cluster a3 \
       --layers 8 --seq 8192 --global-batch 64 --micro-batch 1 --dies 64 --recompute full \
       --request "<the request, in the asker's words>"
   ```

   `--recompute` takes the mode the run uses (`off`, `selective`, `full`), several to compare them,
   or `auto` for all three when the request leaves it open (the default). With the asker's yaml, use
   `--yaml <file>` and keep `--family` when the model belongs to a listed family. Never run ND by hand:
   `ask.py` passes the recompute mode, the cluster's calibration and its budget, which ND otherwise
   prices as full recompute, drops, or reads as 64 GB.
4. **Answer from the output** (also saved as `answer.md` in the run folder), in this order: the
   layout to launch, the next two, why (the table of the best layout with another value of each
   degree, read out in words from its part moves), the status line, ND's notices verbatim, and the
   run folder. When several layouts sit within the cluster's re-measure spread, the firm answer is
   what they share: say one round cannot order them, and offer to time the two that differ most.
5. **The model reader**, for a model ND has not been checked on:
   `python -m hyper_parallel.auto_parallel.sapp_nd.nd.run_nd -f hyper_v2 -y <yaml> -V` sets, for each
   layer kind, the parameters and FLOPs ND prices beside what the model's own layers hold. Report every
   part that differs; a part ND prices at zero is a part the ranking ignores.
6. **Named candidates**: `python .agent/skills/strategy-agent/scripts/answer.py <run>/nd_rank.csv
   --calibrated --compare "EP=2,OP=32;EP=1,OP=64"` gives each one's place, step and peak. A candidate
   with no row did not fit the budget or is outside ND's search; say which you cannot tell apart.

ND's degrees, as the answer names them: DP data parallel, TP tensor parallel, CP context parallel,
PP pipeline stages (VPP virtual stages), EP expert parallel, OP the FSDP shard group of the
non-expert parameters, MB micro-batches a step, MBS sequences a micro-batch. Writing the pick into a
launch yaml is not built yet: give the degrees and let the asker set them in their trainer yaml.

## Rules

- Every figure in the answer is one the scripts printed. No differences, ratios or memory sums of your own.
- One ND run a question; run again only when the request changes, and say why.
- Never change ND, a template or a calibration to make an answer come out differently.
- Quote the order, never the size of a gain: the status line says how far each ranking was checked.
- Pass ND's notices on verbatim. A part it says it does not price is a part the ranking ignores.
- Name the run folder: it holds the request, the yaml, the command, the versions, ND's log and
  ranking, so the answer can be checked and run again.
