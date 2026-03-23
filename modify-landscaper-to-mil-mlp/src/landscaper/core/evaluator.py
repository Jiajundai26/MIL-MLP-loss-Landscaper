"""Core landscape evaluator.

The evaluator owns the model + adapter + dataloader triple and exposes a
single ``evaluate_loss`` method that the samplers call.  It is deliberately
opaque to the contents of each batch — all task-specific knowledge lives in
the adapter.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from landscaper.tasks.base import TaskAdapter


class LandscapeEvaluator:
    """Evaluate aggregate loss over a dataloader using a task adapter.

    Parameters
    ----------
    model:
        The model to evaluate.
    adapter:
        Task adapter that knows how to compute the loss for a single batch.
    dataloader:
        PyTorch-compatible dataloader (iterable of batches).
    device:
        Device to run evaluation on.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter: TaskAdapter,
        dataloader: Any,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.dataloader = dataloader
        self.device = device or torch.device("cpu")

    @torch.no_grad()
    def evaluate_loss(
        self,
        model_override: Optional[nn.Module] = None,
        max_batches: Optional[int] = None,
    ) -> float:
        """Return the mean loss over (up to ``max_batches`` of) the dataloader.

        Parameters
        ----------
        model_override:
            If given, evaluate this model instead of ``self.model``.  Useful
            for probing perturbed copies without mutating the base model.
        max_batches:
            Stop after this many batches.  ``None`` means the full dataloader.

        Returns
        -------
        float
            Weighted average loss (weighted by batch size).
        """
        model = model_override if model_override is not None else self.model
        model.eval()

        total_loss = 0.0
        total_count = 0

        for i, batch in enumerate(self.dataloader):
            if max_batches is not None and i >= max_batches:
                break
            loss = self.adapter.compute_loss(model, batch, self.device)
            bs = self.adapter.batch_size(batch)
            total_loss += loss.item() * bs
            total_count += bs

        return total_loss / max(total_count, 1)
