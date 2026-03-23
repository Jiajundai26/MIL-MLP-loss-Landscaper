"""High-level ``run_landscape`` entry point.

This module provides the :func:`run_landscape` function which ties together
all components of the framework:

* random direction generation (with optional parameter filter),
* 1-D and 2-D loss surface probing,
* loss surface visualization.

Typical usage::

    from torch.utils.data import DataLoader
    from landscaper.experiments.run_landscape import run_landscape
    from landscaper.tasks import MILClassificationAdapter
    from landscaper.models import AttentionMILMLP

    model = AttentionMILMLP(d_in=512, d_hidden=256, n_classes=2)
    adapter = MILClassificationAdapter()
    dataloader = DataLoader(dataset, batch_size=1, ...)

    results = run_landscape(model=model, adapter=adapter,
                            dataloader=dataloader, mode="2d")

    # Standard MLP
    results = run_landscape(model=mlp, adapter=MLPAdapter(), ...)

    # Head-only landscape (attention + classifier only)
    results = run_landscape(
        model=mil_model,
        adapter=MILClassificationAdapter(),
        param_filter=lambda n, p: "attn" in n or "cls" in n,
    )
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from landscaper.config import LandscapeConfig
from landscaper.core.directions import (
    Direction,
    ParamFilter,
    filter_normalize_direction,
    random_direction,
)
from landscaper.core.evaluator import LandscapeEvaluator
from landscaper.core.sampler import sample_1d, sample_2d
from landscaper.tasks.base import TaskAdapter

logger = logging.getLogger(__name__)


def run_landscape(
    model: nn.Module,
    adapter: TaskAdapter,
    dataloader: Any,
    *,
    mode: Literal["1d", "2d"] = "2d",
    param_filter: Optional[ParamFilter] = None,
    config: Optional[LandscapeConfig] = None,
    device: Optional[Union[str, torch.device]] = None,
    dir1: Optional[Direction] = None,
    dir2: Optional[Direction] = None,
) -> Dict[str, Any]:
    """Run a loss-landscape experiment and return the results.

    Parameters
    ----------
    model:
        The PyTorch model to probe.  The model is *not* mutated — a deep
        copy is used for each perturbation.
    adapter:
        Task adapter describing how to compute the loss for each batch.
    dataloader:
        Iterable of batches.  The adapter controls how each batch is
        interpreted.
    mode:
        ``"1d"`` for a 1-D landscape probe along a single random direction,
        or ``"2d"`` for a 2-D surface over two random directions.
    param_filter:
        Optional predicate ``(name, param) -> bool`` restricting which
        parameters are perturbed.  Useful for region-specific landscapes
        (e.g. head-only, aggregator-only).
    config:
        :class:`~landscaper.config.LandscapeConfig` controlling grid size,
        axis ranges, etc.  Defaults to ``LandscapeConfig()``.
    device:
        Compute device.  Defaults to CUDA if available, else CPU.
    dir1 / dir2:
        Pre-computed directions.  If not provided, random Gaussian directions
        are generated.  ``dir2`` is only used when ``mode="2d"``.

    Returns
    -------
    dict
        Keys depend on ``mode``:

        * ``"mode"`` : ``"1d"`` or ``"2d"``
        * ``"alphas"`` : 1-D grid (and ``"betas"`` for 2-D)
        * ``"losses"`` : loss values (list for 1-D, tensor for 2-D)
        * ``"dir1"`` (and ``"dir2"`` for 2-D) : direction dicts used
    """
    cfg = config if config is not None else LandscapeConfig()

    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = model.to(device)

    # Reproducibility
    if cfg.fixed_seed is not None:
        torch.manual_seed(cfg.fixed_seed)
        random.seed(cfg.fixed_seed)

    # Directions
    if dir1 is None:
        dir1 = random_direction(model, param_filter=param_filter,
                                normalize=cfg.normalize_directions)
        if cfg.normalize_directions:
            dir1 = filter_normalize_direction(dir1, model)

    if mode == "2d" and dir2 is None:
        dir2 = random_direction(model, param_filter=param_filter,
                                normalize=cfg.normalize_directions)
        if cfg.normalize_directions:
            dir2 = filter_normalize_direction(dir2, model)

    evaluator = LandscapeEvaluator(
        model=model,
        adapter=adapter,
        dataloader=dataloader,
        device=device,
    )

    # Grid
    alpha_min, alpha_max = cfg.alpha_range
    beta_min, beta_max = cfg.beta_range

    if mode == "1d":
        alphas = torch.linspace(alpha_min, alpha_max, cfg.n_points_1d).tolist()
        losses = sample_1d(model, dir1, alphas, evaluator)
        logger.info("1-D landscape complete: %d points", len(alphas))
        return {"mode": "1d", "alphas": alphas, "losses": losses, "dir1": dir1}

    # mode == "2d"
    alphas = torch.linspace(alpha_min, alpha_max, cfg.n_points_2d_alpha).tolist()
    betas = torch.linspace(beta_min, beta_max, cfg.n_points_2d_beta).tolist()
    Z = sample_2d(model, dir1, dir2, alphas, betas, evaluator)
    logger.info(
        "2-D landscape complete: %dx%d grid", cfg.n_points_2d_alpha, cfg.n_points_2d_beta
    )
    return {
        "mode": "2d",
        "alphas": alphas,
        "betas": betas,
        "losses": Z,
        "dir1": dir1,
        "dir2": dir2,
    }
