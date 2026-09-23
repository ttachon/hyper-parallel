# Simple Model Training Example

FSDP training with HyperParallel Graph Mode.

## Quick Start

```bash
# Single card
python train.py --config config.yaml

# 8 cards
torchrun --nproc_per_node=8 train.py --config config.yaml
```

## Files

- `train.py` - Training entry script
- `config.yaml` - Training and sharding configuration

## Configuration

```yaml
parallel:
  enable_overlap: true

sharding:
  fsdp:
    enabled: true
    patterns:
      - pattern: "*"
```

## Code

```python
from hyper_parallel.compile import GraphParallelPlan, GraphTrainer, PassConfig

# Model
model = DummyModel(vocab_size, dim)

# Parallel plan
parallel_plan = GraphParallelPlan()
parallel_plan.fsdp_mark_pattern("*")

# Trainer
trainer = GraphTrainer(
    model=model,
    train_fn=train_fn,
    pass_config=PassConfig(enable_overlap=True),
    parallel_plan=parallel_plan,
)

# Train
trainer.train(dataloader, max_steps=1000)
```

## Limitations

- FSDP only (TP/EP/PP planned)
- Parameters must have dim 0 divisible by world_size
