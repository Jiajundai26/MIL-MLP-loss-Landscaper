"""Base abstractions for task adapters."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class TaskAdapter(Protocol):
    """Protocol that every task adapter must satisfy.

    An adapter encapsulates all task-specific logic (loss computation,
    output extraction, device movement) so the core landscape engine can
    remain agnostic to model architecture and batch format.
    """

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a scalar loss tensor for ``batch`` evaluated on ``model``."""
        ...

    def compute_outputs(
        self,
        model: torch.nn.Module,
        batch: Any,
        device: torch.device,
    ) -> Any:
        """Return raw model outputs (logits, dict, tuple, …) for ``batch``."""
        ...

    def move_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        """Move ``batch`` to ``device`` and return the result."""
        ...

    def batch_size(self, batch: Any) -> int:
        """Return the number of samples in ``batch``."""
        ...
