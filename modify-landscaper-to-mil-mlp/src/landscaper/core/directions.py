"""Direction generation for loss-landscape probes.

Directions are random (or filter-constrained) perturbation vectors in the
parameter space of a PyTorch model.  They are used by the samplers to move
away from a base checkpoint along meaningful axes.

A *parameter filter* is a callable:

    def param_filter(name: str, param: torch.nn.Parameter) -> bool:
        ...

that returns ``True`` for parameters that should be included in the
direction vector (and perturbed).  This makes it easy to restrict the
landscape to a specific sub-module:

    # head only
    lambda n, p: n.startswith("cls")

    # attention block only
    lambda n, p: "attention" in n or "aggregator" in n

    # exclude frozen encoder
    lambda n, p: not n.startswith("encoder")

When no filter is supplied, all parameters are included.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

# Type alias for a parameter filter predicate
ParamFilter = Callable[[str, nn.Parameter], bool]

# Type alias for a direction: mapping from parameter name to tensor
Direction = Dict[str, torch.Tensor]

def _default_filter(n: str, p: "nn.Parameter") -> bool:  # noqa: F821
    return True


_DEFAULT_FILTER: ParamFilter = _default_filter


def _filter_params(
    model: nn.Module,
    param_filter: Optional[ParamFilter],
) -> List[tuple]:
    """Return ``[(name, param)]`` for parameters accepted by the filter."""
    filt = param_filter if param_filter is not None else _DEFAULT_FILTER
    return [(n, p) for n, p in model.named_parameters() if filt(n, p)]


def random_direction(
    model: nn.Module,
    param_filter: Optional[ParamFilter] = None,
    normalize: bool = True,
) -> Direction:
    """Generate a random Gaussian direction in parameter space.

    Parameters
    ----------
    model:
        The model whose parameter shapes define the direction space.
    param_filter:
        Optional predicate to restrict which parameters contribute to the
        direction.  Parameters excluded by the filter will have a *zero*
        perturbation tensor (so they remain unchanged during probing).
    normalize:
        If ``True`` (default), each per-layer direction tensor is divided by
        its Frobenius norm so that directions have unit norm per layer
        (filter-normalisation, following Li et al. 2018).

    Returns
    -------
    Direction
        A dict mapping parameter name -> perturbation tensor of the same
        shape as the parameter.
    """
    direction: Direction = {}
    included = {n for n, _ in _filter_params(model, param_filter)}

    for name, param in model.named_parameters():
        if name in included:
            d = torch.randn_like(param.data)
            if normalize:
                norm = d.norm()
                if norm > 0:
                    d = d / norm
            direction[name] = d
        else:
            direction[name] = torch.zeros_like(param.data)

    return direction


def filter_normalize_direction(
    direction: Direction,
    model: nn.Module,
) -> Direction:
    """Normalise a direction so each layer vector has the same norm as the
    corresponding parameter tensor (filter-normalisation).

    This is the normalisation strategy recommended by Li et al. (2018) to
    make directions scale-invariant across layers.
    """
    normalised: Direction = {}
    param_dict = dict(model.named_parameters())

    for name, d in direction.items():
        if name in param_dict:
            param_norm = param_dict[name].data.norm()
            d_norm = d.norm()
            if d_norm > 0 and param_norm > 0:
                normalised[name] = d * (param_norm / d_norm)
            else:
                normalised[name] = d.clone()
        else:
            normalised[name] = d.clone()

    return normalised


def apply_perturbation(
    model: nn.Module,
    base_state: Dict[str, torch.Tensor],
    perturbations: List[tuple],
) -> None:
    """In-place: set ``model`` parameters to ``base + sum(alpha_i * dir_i)``.

    Parameters
    ----------
    model:
        The model to perturb.
    base_state:
        ``state_dict`` of the un-perturbed model (as returned by
        ``model.state_dict()``).
    perturbations:
        List of ``(direction, alpha)`` pairs.  Each ``direction`` is a
        :class:`Direction` dict and ``alpha`` is a scalar step size.
    """
    new_state = {k: v.clone() for k, v in base_state.items()}
    for direction, alpha in perturbations:
        for name, delta in direction.items():
            if name in new_state:
                new_state[name] = new_state[name] + alpha * delta.to(new_state[name].device)
    model.load_state_dict(new_state)
