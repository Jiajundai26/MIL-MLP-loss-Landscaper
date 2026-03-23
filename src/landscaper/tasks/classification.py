"""Standard image/tabular classification adapter."""

from __future__ import annotations

from typing import Any, Tuple, Union

import torch
import torch.nn.functional as F


class StandardClassificationAdapter:
    """Adapter for the classic ``(inputs, targets)`` batch convention.

    Works out-of-the-box for image classifiers and tabular models whose
    dataloader yields 2-tuples or 2-element lists ``(x, y)``.

    Parameters
    ----------
    criterion:
        Loss callable ``(logits, targets) -> scalar``.  Defaults to
        cross-entropy.
    input_key / target_key:
        If the dataloader yields *dict* batches, use these keys to extract
        inputs and targets instead of positional indexing.
    """

    def __init__(
        self,
        criterion: Any = None,
        input_key: str = "input",
        target_key: str = "label",
    ) -> None:
        self.criterion = criterion if criterion is not None else F.cross_entropy
        self.input_key = input_key
        self.target_key = target_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpack(
        self, batch: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(inputs, targets)`` from a batch."""
        if isinstance(batch, dict):
            return batch[self.input_key], batch[self.target_key]
        # tuple / list
        x, y = batch[0], batch[1]
        return x, y

    # ------------------------------------------------------------------
    # TaskAdapter interface
    # ------------------------------------------------------------------

    def move_batch_to_device(
        self, batch: Any, device: torch.device
    ) -> Any:
        if isinstance(batch, dict):
            return {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        x, y = self._unpack(batch)
        return (x.to(device), y.to(device))

    def compute_outputs(
        self,
        model: torch.nn.Module,
        batch: Any,
        device: torch.device,
    ) -> Any:
        batch = self.move_batch_to_device(batch, device)
        x, _ = self._unpack(batch)
        return model(x)

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
        device: torch.device,
    ) -> torch.Tensor:
        batch = self.move_batch_to_device(batch, device)
        x, y = self._unpack(batch)
        outputs = model(x)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return self.criterion(logits, y)

    def batch_size(self, batch: Any) -> int:
        x, _ = self._unpack(batch)
        return int(x.shape[0])
