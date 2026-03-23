# MIL-MLP-loss-Landscaper

A MIL/MLP-aware loss-landscape framework for PyTorch, built on top of the
[Landscaper](https://github.com/Vis4SciML/Landscaper) design.

The framework lets you visualise the loss landscape of **Multiple Instance
Learning (MIL)** and **MLP** models using the same engine, with first-class
support for bag-level data, pre-computed instance embeddings, and flexible
parameter filtering.

---

## Features

| Capability | Details |
|---|---|
| **Task adapters** | `StandardClassificationAdapter`, `MILClassificationAdapter`, `MLPAdapter` |
| **Opaque batch handling** | No shape assumptions inside the core engine |
| **Parameter filters** | Restrict landscape to head / aggregator / encoder sub-modules |
| **1-D & 2-D probes** | `sample_1d`, `sample_2d`, `run_landscape` |
| **MIL model** | `AttentionMILMLP` – attention-based MIL on pre-computed embeddings |
| **Data utilities** | `EmbeddingBagDataset`, `variable_length_collate`, `padded_bag_collate` |
| **Checkpoint I/O** | `load_model_from_checkpoint`, `load_model_and_adapter` |
| **Reproducibility** | `LandscapeConfig(stochastic_eval=False, fixed_seed=42)` |

---

## Installation

```bash
pip install -e .          # from repo root
pip install -e ".[dev]"   # + pytest
```

Requires **Python >= 3.8** and **PyTorch >= 2.0.0**.

---

## Quick-start

### Standard MLP

```python
from landscaper import run_landscape, MLPAdapter, LandscapeConfig

results = run_landscape(
    model=mlp,
    adapter=MLPAdapter(),
    dataloader=dataloader,
    mode="2d",
    config=LandscapeConfig(n_points_2d_alpha=21, n_points_2d_beta=21, fixed_seed=0),
    device="cpu",
)
Z = results["losses"]   # torch.Tensor [21, 21]
```

### MIL on pre-computed embeddings

```python
from landscaper import (
    run_landscape, MILClassificationAdapter,
    AttentionMILMLP, EmbeddingBagDataset, variable_length_collate,
    LandscapeConfig,
)
from torch.utils.data import DataLoader

model   = AttentionMILMLP(d_in=512, d_hidden=256, n_classes=2)
adapter = MILClassificationAdapter()
dataset = EmbeddingBagDataset(bags=bags, labels=labels)
loader  = DataLoader(dataset, batch_size=1, collate_fn=variable_length_collate)

results = run_landscape(
    model=model, adapter=adapter, dataloader=loader,
    mode="1d",
    config=LandscapeConfig(n_points_1d=51, fixed_seed=42),
    device="cuda",
)
```

### Region-specific landscape (head-only)

```python
results = run_landscape(
    model=mil_model,
    adapter=MILClassificationAdapter(),
    dataloader=loader,
    param_filter=lambda n, p: "attn" in n or "cls" in n,
    config=LandscapeConfig(n_points_2d_alpha=21, n_points_2d_beta=21),
)
```

---

## Package structure

```
src/landscaper/
  config.py               # LandscapeConfig dataclass
  core/
    directions.py         # random_direction, filter_normalize_direction,
                          #   apply_perturbation, ParamFilter
    evaluator.py          # LandscapeEvaluator
    sampler.py            # sample_1d, sample_2d
  tasks/
    base.py               # TaskAdapter Protocol
    classification.py     # StandardClassificationAdapter
    mil.py                # MILClassificationAdapter
    mlp.py                # MLPAdapter
  models/
    wrappers.py           # AttentionMILMLP, load_model_from_checkpoint,
                          #   load_model_and_adapter
  data/
    collate.py            # EmbeddingBagDataset, variable_length_collate,
                          #   padded_bag_collate
  experiments/
    run_landscape.py      # run_landscape
tests/
  test_landscaper.py      # 40 unit tests
```

---

## LandscapeConfig reference

| Field | Default | Description |
|---|---|---|
| `n_points_1d` | 51 | Grid points for a 1-D probe |
| `alpha_range` | `(-1.0, 1.0)` | Axis range for alpha |
| `beta_range` | `(-1.0, 1.0)` | Axis range for beta |
| `n_points_2d_alpha` | 21 | Grid points along alpha for 2-D probe |
| `n_points_2d_beta` | 21 | Grid points along beta for 2-D probe |
| `max_batches` | `None` | Limit batches per loss evaluation |
| `normalize_directions` | `True` | Filter-normalise random directions |
| `stochastic_eval` | `False` | Keep `model.eval()` during probing |
| `fixed_seed` | `None` | Seed for reproducible directions |

---

## Running tests

```bash
pytest tests/ -v
```
