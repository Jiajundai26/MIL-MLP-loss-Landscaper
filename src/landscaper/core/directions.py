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

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from landscaper.tasks.base import TaskAdapter

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


def _build_hessian_param_list(
    model: nn.Module,
    param_filter: Optional[ParamFilter],
) -> tuple:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    filt = param_filter if param_filter is not None else _DEFAULT_FILTER
    included = [(n, p) for n, p in named if filt(n, p)]
    return named, included


def _mean_loss_for_hessian(
    model: nn.Module,
    adapter: TaskAdapter,
    dataloader: Any,
    device: torch.device,
    max_batches: Optional[int],
) -> torch.Tensor:
    total_loss = torch.tensor(0.0, device=device)
    total_count = 0
    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break
        loss = adapter.compute_loss(model, batch, device)
        bs = int(adapter.batch_size(batch))
        total_loss = total_loss + loss * bs
        total_count += bs
    if total_count == 0:
        raise ValueError("Cannot estimate Hessian directions on an empty dataloader.")
    return total_loss / float(total_count)


def dominant_hessian_directions(
    model: nn.Module,
    adapter: TaskAdapter,
    dataloader: Any,
    device: torch.device,
    *,
    top_n: int = 2,
    max_batches: Optional[int] = None,
    power_iters: int = 20,
    tol: float = 1e-6,
    param_filter: Optional[ParamFilter] = None,
    normalize: bool = True,
) -> List[Direction]:
    """Estimate dominant Hessian-eigenvector directions via power iteration."""
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    named_all, named_included = _build_hessian_param_list(model, param_filter)
    if not named_included:
        raise ValueError("No parameters selected for Hessian direction estimation.")

    params = [p for _, p in named_included]
    sizes = [int(p.numel()) for p in params]
    total_dim = int(sum(sizes))

    def split_vec(vec: torch.Tensor) -> List[torch.Tensor]:
        parts = []
        cursor = 0
        for sz, p in zip(sizes, params):
            parts.append(vec[cursor: cursor + sz].view_as(p))
            cursor += sz
        return parts

    def hvp(vec: torch.Tensor) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        loss = _mean_loss_for_hessian(model, adapter, dataloader, device, max_batches)
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        vec_parts = split_vec(vec)
        g_dot_v = torch.tensor(0.0, device=device)
        for g, v in zip(grads, vec_parts):
            g_dot_v = g_dot_v + (g * v).sum()
        hv_parts = torch.autograd.grad(g_dot_v, params, retain_graph=False, create_graph=False)
        return torch.cat([h.reshape(-1) for h in hv_parts]).detach()

    eigvecs: List[torch.Tensor] = []
    for _ in range(top_n):
        v = torch.randn(total_dim, device=device)
        v = v / (v.norm() + 1e-12)
        for _ in range(power_iters):
            hv = hvp(v)
            for q in eigvecs:
                hv = hv - torch.dot(hv, q) * q
            norm = hv.norm()
            if norm <= tol:
                break
            v_new = hv / norm
            if (v_new - v).norm() <= tol:
                v = v_new
                break
            v = v_new
        eigvecs.append(v.detach())

    directions: List[Direction] = []
    included_names = {n for n, _ in named_included}
    for v in eigvecs:
        parts = split_vec(v)
        part_map = {n: p.detach().clone() for (n, _), p in zip(named_included, parts)}
        direction: Direction = {}
        for name, param in named_all:
            if name in included_names:
                direction[name] = part_map[name].view_as(param).to(param.device)
            else:
                direction[name] = torch.zeros_like(param.data)
        if normalize:
            direction = filter_normalize_direction(direction, model)
        directions.append(direction)
    return directions
