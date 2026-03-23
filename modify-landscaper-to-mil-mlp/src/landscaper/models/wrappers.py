"""Model wrappers and checkpoint utilities.

This module provides:

* :class:`AttentionMILMLP` — a compact Attention-MIL model that operates
  on pre-computed instance embeddings.  It supports both pure MIL workloads
  and plain MLP tabular workloads (set ``n_instances=1``).

* :func:`load_model_from_checkpoint` — load any ``nn.Module`` from a
  ``torch.save`` checkpoint file.

* :func:`load_model_and_adapter` — convenience factory that creates a model
  *and* the matching :class:`~landscaper.tasks.base.TaskAdapter` from a
  config dict and checkpoint path.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionMILMLP(nn.Module):
    """Attention-based MIL model on pre-computed instance embeddings.

    The forward pass:

    1. Projects each instance embedding through a shallow MLP ``phi``.
    2. Computes a soft-max attention score over instances via ``attn``.
    3. Aggregates instances into a bag-level representation ``z``.
    4. Classifies ``z`` with a linear head ``cls``.

    The model returns a dict with keys ``"logits"``, ``"attention"``, and
    ``"bag_embedding"`` so that downstream adapters can access both the
    classification output and intermediate representations.

    Parameters
    ----------
    d_in:
        Dimensionality of the input instance embeddings.
    d_hidden:
        Hidden dimensionality used by ``phi`` and ``attn``.
    n_classes:
        Number of output classes.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        n_classes: int,
    ) -> None:
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
        )
        self.attn = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )
        self.cls = nn.Linear(d_hidden, n_classes)

    def forward(self, bag: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        bag:
            Instance embedding tensor of shape ``[N, d_in]`` (single bag)
            or ``[B, N, d_in]`` (batched — **not** natively supported; pass
            individual bags via the MIL adapter).

        Returns
        -------
        dict
            ``logits``      : ``[1, n_classes]``
            ``attention``   : ``[N]`` — per-instance attention weights
            ``bag_embedding``: ``[d_hidden]`` — aggregated bag vector
        """
        h = self.phi(bag)  # [N, H]
        raw_attn = self.attn(h).squeeze(-1)  # [N]
        a = torch.softmax(raw_attn, dim=0)  # [N]
        z = (a.unsqueeze(-1) * h).sum(dim=0)  # [H]
        logits = self.cls(z.unsqueeze(0))  # [1, C]
        return {"logits": logits, "attention": a, "bag_embedding": z}


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------


def load_model_from_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    map_location: Optional[Any] = None,
    strict: bool = True,
) -> nn.Module:
    """Load ``state_dict`` from a checkpoint file into ``model``.

    The checkpoint may be:
    * a raw ``state_dict`` (dict of tensors), or
    * a dict containing a ``"state_dict"`` key (common Lightning / custom
      training loop convention).

    Parameters
    ----------
    model:
        The model instance to load weights into.
    ckpt_path:
        Path to the ``.pt`` / ``.pth`` checkpoint file.
    map_location:
        Passed directly to :func:`torch.load`.  Useful for loading GPU
        checkpoints on CPU.
    strict:
        Passed to :meth:`~torch.nn.Module.load_state_dict`.

    Returns
    -------
    nn.Module
        The same ``model`` object with weights loaded.
    """
    checkpoint = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=strict)
    return model


def load_model_and_adapter(
    model_cfg: Dict[str, Any],
    ckpt_path: str,
    task_type: str = "mil",
    map_location: Optional[Any] = None,
) -> Tuple[nn.Module, Any]:
    """Factory: build a model + adapter pair from a config dict and checkpoint.

    Parameters
    ----------
    model_cfg:
        Dict with at least a ``"type"`` key.  Currently supported types:

        * ``"AttentionMILMLP"`` — requires ``"d_in"``, ``"d_hidden"``,
          ``"n_classes"`` keys.

    ckpt_path:
        Path to the model checkpoint.
    task_type:
        One of ``"mil"``, ``"mlp"``, ``"classification"``.
    map_location:
        Passed to :func:`torch.load`.

    Returns
    -------
    (model, adapter)
    """
    from landscaper.tasks import (
        MILClassificationAdapter,
        MLPAdapter,
        StandardClassificationAdapter,
    )

    model_type = model_cfg.get("type", "AttentionMILMLP")
    if model_type == "AttentionMILMLP":
        model: nn.Module = AttentionMILMLP(
            d_in=model_cfg["d_in"],
            d_hidden=model_cfg["d_hidden"],
            n_classes=model_cfg["n_classes"],
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")

    load_model_from_checkpoint(model, ckpt_path, map_location=map_location)
    model.eval()

    if task_type == "mil":
        adapter = MILClassificationAdapter()
    elif task_type == "mlp":
        adapter = MLPAdapter()
    elif task_type == "classification":
        adapter = StandardClassificationAdapter()
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")

    return model, adapter
