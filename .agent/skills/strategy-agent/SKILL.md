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
that has its dependencies (torch, Transformers, PyYAML). ND runs on the CPU. For DeepSeek-V4.1, work
in a checkout of nd/strategy-agent-v41, which adds the V4.1 model to the same ND.

## Answer a request

1. **Read the request.** Collect the model (a template name, or the asker's trainer yaml), the layers,
   the tokens a sequence, the global batch, the micro-batch, the recompute mode if stated, the total
   dies, the cluster and a memory budget if stated. Where a value is missing, use the template's and
   say so. Templates: `templates/qwen3_5_moe.yaml` (Qwen3.5-35B-A3B and its layer crops) and
   `templates/deepseek_v41.yaml` (the DeepSeek-V4.1 validation crop as its launcher builds it, 40
   layers, which a request cannot change; ND measures its memory with the census). In a checkout without
   the V4.1 model, `ask.py` prices the crop described by hand instead, by formulas, and says so.
2. **Check what can be trusted.** `references/trust.json` says, for each template on each cluster, how
   far ND was checked against timed runs; `references/status.md` explains it. `ask.py` applies it: it
   prints the status line, uses the cluster's calibration only where it holds for the family, leaves
   out of an open search the recompute modes ND misprices, and checks the pick against the region
   that was timed.
   - No template and no yaml from the asker: ask for their trainer yaml; nothing can be priced.
   - A yaml of a family with no record: check the model (step 5) before ranking, and say the
     ranking is not validated.
3. **Run ND once**, through the script beside this file. It finds the checkout it belongs to and runs ND
   there, so it works from any folder, also through a link to this skill:

   ```bash
   python <this skill's folder>/scripts/ask.py --family qwen3_5_moe --cluster a3 \
       --layers 8 --seq 8192 --global-batch 64 --micro-batch 1 --dies 64 --recompute full \
       --request "<the request, in the asker's words>"
   ```

   `--recompute` takes the mode the run uses (`off`, `selective`, `full`), several to compare them,
   or `auto` when the request leaves it open (the default): every mode the family's record does not
   exclude, and the answer says which were left out and why. With the asker's yaml, use
   `--yaml <file>` and keep `--family` when the model belongs to a listed family. Never run ND by hand:
   `ask.py` passes the recompute mode, the cluster's calibration and its budget, which ND otherwise
   prices as full recompute, drops, or reads as 64 GB.
4. **Answer from the output** (also saved as `answer.md` in the run folder), in this order: the
   layout to launch, the next two, why (the table of the best layout with another value of each
   degree, read out in words from its part moves), the status line, ND's notices verbatim, and the
   run folder. When several layouts sit within the cluster's re-measure spread, the firm answer is
   what they share: say one round cannot order them, and offer to time the two that differ most.
   **When the answer says ND's pick lies outside the region that was timed, the layout to launch is
   the best one inside it**, which the answer names; give ND's pick as the candidate worth timing, with
   the reason the answer quotes. With no region at all, recommend nothing ND alone decides: say what
   fits, and which layouts are worth timing.
5. **Check the model**, for a model ND has not been checked on, or when asked what ND prices of it:
   `python <this skill's folder>/scripts/check.py` with `--family` or `--yaml` as for `ask.py`. It sets
   the parameters ND prices for each part of a layer beside what the model's own layers hold (DeepSeek-V4.1
   through its own tool, against the crop's tensors, and the blocks no ND field carries; any other model
   through ND's verify mode, FLOPs too), and keeps the check in a run folder. Report every part that
   differs; a part ND prices at zero is a part the ranking ignores.
6. **Named candidates**: `python <this skill's folder>/scripts/answer.py <run>/nd_rank.csv
   --calibrated --compare "EP=2,OP=32;EP=1,OP=64"` gives each one's place, step and peak. A candidate
   with no row did not fit the budget or is outside ND's search; say which you cannot tell apart.
7. **Which layers can skip recompute**, for a layout already chosen: `python <this skill's folder>/scripts/plan.py`
   with the same run flags plus `--layout "EP=2,OP=32"` (TP, CP and PP are 1 unless given). It states the
   budget from the cluster's measured reserve, since ND prices allocated memory and a run dies on reserved,
   lets ND choose each layer's mode, and writes the plan into `launch.yaml` (`activation_checkpoint.layers`).
   Give the plan, the budget and how it was derived, and the allocator note it prints; never the gain.

ND's degrees, as the answer names them: DP data parallel, TP tensor parallel, CP context parallel,
PP pipeline stages (VPP virtual stages), EP expert parallel, OP the FSDP shard group of the
non-expert parameters, MB micro-batches a step, MBS sequences a micro-batch.

The run folder also holds the trainer yaml to launch: `launch.yaml` for the layout the answer recommends,
and `launch_nd_pick.yaml` for ND's own pick when that differs or nothing validates it. They are the
request's yaml with the degrees written where the trainer reads them, the expert shard group included
(`edp_shard_size`, the whole expert group, as ND priced it and as every timed run was launched), and
without ND's `context` section, which the trainer refuses. Tell the asker to set the tokenizer and data
paths for their cluster; a virtual pipeline degree, when one is picked, is not written.

## Rules

- Every figure in the answer is one the scripts printed. No differences, ratios or memory sums of your own.
- One ND run a question; run again only when the request changes, and say why.
- Never change ND, a template or a calibration to make an answer come out differently.
- Quote the order, never the size of a gain: the status line says how far each ranking was checked.
- Pass ND's notices on verbatim. A part it says it does not price is a part the ranking ignores.
- Name the run folder: it holds the request, the yaml, the command, the versions, ND's log and
  ranking, so the answer can be checked and run again.
