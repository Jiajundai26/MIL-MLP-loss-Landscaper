"""1-D and 2-D loss-landscape samplers.

The samplers perturb a base model along one or two directions and record
the loss at each grid point.  They rely on:

* :func:`~landscaper.core.directions.apply_perturbation` to move the model,
* :class:`~landscaper.core.evaluator.LandscapeEvaluator` to compute the loss.

Both sampler functions operate on *copies* of the base model and never
mutate the original weights.
"""

from __future__ import annotations

import copy
from typing import Any, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from landscaper.core.directions import Direction, apply_perturbation
from landscaper.core.evaluator import LandscapeEvaluator


def sample_1d(
    base_model: nn.Module,
    direction: Direction,
    alphas: Sequence[float],
    evaluator: LandscapeEvaluator,
) -> List[float]:
    """Probe the loss along a single direction.

    For each scalar ``alpha`` in ``alphas``, a model copy is created with
    parameters ``base + alpha * direction`` and the loss is evaluated.

    Parameters
    ----------
    base_model:
        Un-perturbed starting model.
    direction:
        Direction dict (as returned by
        :func:`~landscaper.core.directions.random_direction`).
    alphas:
        Sequence of step sizes to probe.
    evaluator:
        Evaluator that computes the loss for a given model.

    Returns
    -------
    List[float]
        Loss values, one per ``alpha``.
    """
    base_state = copy.deepcopy(base_model.state_dict())
    losses: List[float] = []

    for alpha in alphas:
        model = copy.deepcopy(base_model)
        apply_perturbation(model, base_state, [(direction, alpha)])
        losses.append(evaluator.evaluate_loss(model_override=model))

    return losses


def sample_2d(
    base_model: nn.Module,
    dir1: Direction,
    dir2: Direction,
    alphas: Sequence[float],
    betas: Sequence[float],
    evaluator: LandscapeEvaluator,
) -> torch.Tensor:
    """Probe the loss on a 2-D plane spanned by two directions.

    For each ``(alpha, beta)`` pair, a model copy is created with parameters
    ``base + alpha * dir1 + beta * dir2`` and the loss is evaluated.

    Parameters
    ----------
    base_model:
        Un-perturbed starting model.
    dir1 / dir2:
        Two direction dicts spanning the plane.
    alphas / betas:
        Grid coordinates along each axis.
    evaluator:
        Evaluator that computes the loss.

    Returns
    -------
    torch.Tensor
        Shape ``[len(alphas), len(betas)]`` — the loss grid.
    """
    base_state = copy.deepcopy(base_model.state_dict())
    Z = torch.zeros(len(alphas), len(betas))

    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            model = copy.deepcopy(base_model)
            apply_perturbation(model, base_state, [(dir1, a), (dir2, b)])
            Z[i, j] = evaluator.evaluate_loss(model_override=model)

    return Z
