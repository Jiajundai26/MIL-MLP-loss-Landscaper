"""tasks sub-package."""

from landscaper.tasks.base import TaskAdapter
from landscaper.tasks.classification import StandardClassificationAdapter
from landscaper.tasks.mil import MILClassificationAdapter
from landscaper.tasks.mlp import MLPAdapter

__all__ = [
    "TaskAdapter",
    "StandardClassificationAdapter",
    "MILClassificationAdapter",
    "MLPAdapter",
]
