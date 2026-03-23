"""Data utilities for MIL/MLP landscapes.

Provides:

* :class:`EmbeddingBagDataset` — a ``Dataset`` for pre-computed instance
  embeddings.  Each sample is a dict ``{"bag": Tensor[N, D], "label": Tensor}``.

* :func:`variable_length_collate` — a collate function that handles
  variable-length bags (i.e. bags with different numbers of instances) by
  returning a list of tensors rather than stacking them into a single array.

* :func:`padded_bag_collate` — a collate function that pads variable-length
  bags to the length of the longest bag in a mini-batch and returns a
  ``[B, N_max, D]`` tensor together with a mask ``[B, N_max]``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset


class EmbeddingBagDataset(Dataset):
    """Dataset of pre-computed instance-embedding bags.

    Parameters
    ----------
    bags:
        Sequence of tensors, each of shape ``[N_i, D]`` where ``N_i`` is the
        (possibly variable) number of instances in the i-th bag.
    labels:
        Sequence of integer or float labels, one per bag.
    """

    def __init__(
        self,
        bags: Sequence[torch.Tensor],
        labels: Sequence[Any],
    ) -> None:
        if len(bags) != len(labels):
            raise ValueError(
                f"bags and labels must have the same length "
                f"({len(bags)} vs {len(labels)})"
            )
        self.bags = bags
        self.labels = labels

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        bag = self.bags[idx]
        label = self.labels[idx]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label)
        return {"bag": bag, "label": label}


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------


def variable_length_collate(
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collate a list of bag samples into a batch.

    Unlike the default PyTorch collator, this function does **not** try to
    stack bags of different lengths.  Instead it returns:

    * ``"bag"``   : list of tensors ``[N_i, D]``
    * ``"label"`` : 1-D tensor of labels ``[B]``

    This is the expected format for :class:`~landscaper.tasks.mil.MILClassificationAdapter`
    when bags have variable length.
    """
    bags = [s["bag"] for s in samples]
    labels = torch.stack([s["label"] for s in samples])
    return {"bag": bags, "label": labels}


def padded_bag_collate(
    samples: List[Dict[str, Any]],
    pad_value: float = 0.0,
) -> Dict[str, Any]:
    """Collate variable-length bags into a padded tensor.

    Returns:

    * ``"bag"``   : ``[B, N_max, D]`` — zero-padded bag tensor
    * ``"label"`` : ``[B]`` — label tensor
    * ``"mask"``  : ``[B, N_max]`` bool tensor, ``True`` for real instances
    """
    bags = [s["bag"] for s in samples]
    labels = torch.stack([s["label"] for s in samples])

    n_max = max(b.shape[0] for b in bags)
    d = bags[0].shape[1]
    B = len(bags)

    padded = bags[0].new_full((B, n_max, d), fill_value=pad_value)
    mask = torch.zeros(B, n_max, dtype=torch.bool)

    for i, b in enumerate(bags):
        n = b.shape[0]
        padded[i, :n] = b
        mask[i, :n] = True

    return {"bag": padded, "label": labels, "mask": mask}
