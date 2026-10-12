# How far each answer can be trusted

ND ranks layouts well where its ranking was checked against runs timed on the same cluster, and
is unproven elsewhere. `scripts/ask.py` prints the line of this table that matches the request
(template and cluster; `*` matches any) at the top of every answer. Quote it; do not soften it.

| Template | Cluster | Status | Evidence | What the answer may say |
|---|---|---|---|---|
| `qwen3_5_moe` | `a3` | Back-tested | 16 layouts of Qwen3.5-35B-A3B cut to 8 layers (8192 tokens, global batch 64, full recompute, 64 dies) timed on the job's own step time: ND's pick ran 2nd, 0.54% behind the fastest, rank correlation +0.91. Six per-layer recompute plans: the fastest that fits was ND's, peaks within 2.9%. Timed with the Transformers delta rule; the HyperParallel delta rule, the trainer's default since 9 October 2026, is not timed yet | The order of the layouts and which fit. Not the size of a gain: ND said 14.4% for a recompute plan that measured 6.0%. Selective recompute runs 37 to 45% slower than ND prices it, so never recommend selective over full on ND's word alone |
| `deepseek_v41` | `a3` | Timed, not validated | 7 layouts timed on 16 dies: ND's order -0.39. It prices the expert degree backwards: EP2 +22% and EP16 +41% over EP1, where the runs measured -4.7% and -1.2% | Which layouts fit. No recommendation on the expert degree. There is no template yet: the asker's yaml goes through `--yaml` |
| `qwen3_5_moe` | `*` | No calibration | No run timed on this cluster | The order only, as relative scores, and no step time. About five launches on the cluster give it a calibration |
| `*` | `*` | Never timed | Nothing timed for this family | A ranking labelled not validated, after the model reader has checked what ND prices for each layer kind |
