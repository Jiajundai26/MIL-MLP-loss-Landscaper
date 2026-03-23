"""LandscapeConfig: configuration object for landscape experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence


@dataclass
class LandscapeConfig:
    """Configuration for a loss-landscape experiment.

    Parameters
    ----------
    n_points_1d:
        Number of grid points for a 1-D landscape probe.
    alpha_range:
        ``(min, max)`` range for the 1-D / 2-D ``alpha`` axis.
    beta_range:
        ``(min, max)`` range for the 2-D ``beta`` axis.
    n_points_2d_alpha / n_points_2d_beta:
        Number of grid points along each axis for a 2-D landscape probe.
    max_batches:
        Maximum number of dataloader batches to evaluate per loss
        computation.  ``None`` means the full dataloader.
    normalize_directions:
        If ``True``, apply filter-normalisation to random directions.
    stochastic_eval:
        If ``False`` (default), the model is set to ``eval()`` mode before
        each loss computation so that dropout and batch-norm behave
        deterministically.
    fixed_seed:
        If set, the random seed is fixed before generating directions to
        make experiments reproducible.
    """

    n_points_1d: int = 51
    alpha_range: tuple = (-1.0, 1.0)
    beta_range: tuple = (-1.0, 1.0)
    n_points_2d_alpha: int = 21
    n_points_2d_beta: int = 21
    max_batches: Optional[int] = None
    normalize_directions: bool = True
    stochastic_eval: bool = False
    fixed_seed: Optional[int] = None
