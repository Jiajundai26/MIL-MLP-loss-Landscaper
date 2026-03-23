"""MIL (Multiple Instance Learning) task adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


class MILClassificationAdapter:
    """Adapter for MIL models that operate on *bags* of instances.

    The dataloader is expected to yield dicts with at least the keys
    ``bag_key`` (default ``"bag"``) and ``label_key`` (default ``"label"``).

    Supported bag shapes
    --------------------
    * ``[N, D]``       — pre-computed instance embeddings (single bag)
    * ``[B, N, D]``    — batched pre-computed embeddings
    * ``[B, N, C, H, W]`` — batched raw image bags

    Variable-length bags (list of tensors) are also accepted: the adapter
    iterates over the list and accumulates the loss.

    Parameters
    ----------
    bag_key / label_key:
        Dict keys used to extract the bag tensor and label from each batch.
    criterion:
        Loss callable ``(logits, targets) -> scalar``.  Defaults to
        cross-entropy.
    attn_reg_weight / inst_loss_weight:
        Optional coefficients for auxiliary losses returned by the model
        under the ``"attn_reg"`` / ``"inst_loss"`` keys of the output dict.
    """

    def __init__(
        self,
        bag_key: str = "bag",
        label_key: str = "label",
        criterion: Any = None,
        attn_reg_weight: float = 0.0,
        inst_loss_weight: float = 0.0,
    ) -> None:
        self.bag_key = bag_key
        self.label_key = label_key
        self.criterion = criterion if criterion is not None else F.cross_entropy
        self.attn_reg_weight = attn_reg_weight
        self.inst_loss_weight = inst_loss_weight

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _forward_single_bag(
        self,
        model: torch.nn.Module,
        bag: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Forward one bag and return its loss."""
        # Ensure bag has a batch dimension for models that expect [B, N, D]
        if bag.dim() == 2:
            # [N, D] -> pass as-is; models typically handle this
            outputs = model(bag)
        else:
            outputs = model(bag)

        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        # Ensure label has correct shape: always [1] for single-bag forward
        if label.dim() == 0:
            label = label.unsqueeze(0)
        # Ensure logits have a batch dimension
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        loss = self.criterion(logits, label)

        if isinstance(outputs, dict):
            if self.attn_reg_weight and "attn_reg" in outputs:
                loss = loss + self.attn_reg_weight * outputs["attn_reg"]
            if self.inst_loss_weight and "inst_loss" in outputs:
                loss = loss + self.inst_loss_weight * outputs["inst_loss"]

        return loss

    # ------------------------------------------------------------------
    # TaskAdapter interface
    # ------------------------------------------------------------------

    def move_batch_to_device(
        self, batch: Any, device: torch.device
    ) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                moved[k] = v.to(device)
            elif isinstance(v, list) and all(isinstance(t, torch.Tensor) for t in v):
                moved[k] = [t.to(device) for t in v]
            else:
                moved[k] = v
        return moved

    def compute_outputs(
        self,
        model: torch.nn.Module,
        batch: Any,
        device: torch.device,
    ) -> Any:
        batch = self.move_batch_to_device(batch, device)
        bag = batch[self.bag_key]
        if isinstance(bag, list):
            return [model(b) for b in bag]
        return model(bag)

    def compute_loss(
        self,
        model: torch.nn.Module,
        batch: Any,
        device: torch.device,
    ) -> torch.Tensor:
        batch = self.move_batch_to_device(batch, device)
        bag = batch[self.bag_key]
        label = batch[self.label_key]

        # Variable-length bags: list of tensors, accumulate mean loss
        if isinstance(bag, list):
            losses = []
            for b, lbl in zip(bag, label if label.dim() > 0 else [label]):
                losses.append(self._forward_single_bag(model, b, lbl))
            return torch.stack(losses).mean()

        # Batched bags [B, N, D]: iterate over batch dimension
        if bag.dim() == 3:
            losses = []
            for i in range(bag.shape[0]):
                lbl = label[i] if label.dim() > 0 else label
                losses.append(self._forward_single_bag(model, bag[i], lbl))
            return torch.stack(losses).mean()

        # Single bag [N, D]
        return self._forward_single_bag(model, bag, label)

    def batch_size(self, batch: Any) -> int:
        bag = batch[self.bag_key]
        if isinstance(bag, list):
            return len(bag)
        if isinstance(bag, torch.Tensor):
            # [B, N, D] -> B; [N, D] -> 1
            return bag.shape[0] if bag.dim() == 3 else 1
        return 1
