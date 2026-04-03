"""landscaper: MIL/MLP-aware loss-landscape framework for PyTorch.

Top-level public API:

    from landscaper import (
        LandscapeConfig,
        LandscapeEvaluator,
        run_landscape,
        # adapters
        TaskAdapter,
        StandardClassificationAdapter,
        MILClassificationAdapter,
        MLPAdapter,
        # direction helpers
        random_direction,
        filter_normalize_direction,
        apply_perturbation,
        # samplers
        sample_1d,
        sample_2d,
        # models
        AttentionMILMLP,
        load_model_from_checkpoint,
        load_model_and_adapter,
        # data
        EmbeddingBagDataset,
        variable_length_collate,
        padded_bag_collate,
    )
"""

from landscaper.config import LandscapeConfig
from landscaper.core import (
    Direction,
    LandscapeEvaluator,
    ParamFilter,
    apply_perturbation,
    dominant_hessian_directions,
    filter_normalize_direction,
    random_direction,
    sample_1d,
    sample_2d,
)
from landscaper.data import (
    EmbeddingBagDataset,
    padded_bag_collate,
    variable_length_collate,
)
from landscaper.experiments import run_landscape
from landscaper.models import (
    AttentionMILMLP,
    load_model_and_adapter,
    load_model_from_checkpoint,
)
from landscaper.tasks import (
    MILClassificationAdapter,
    MLPAdapter,
    StandardClassificationAdapter,
    TaskAdapter,
)

__all__ = [
    "LandscapeConfig",
    "LandscapeEvaluator",
    "run_landscape",
    "TaskAdapter",
    "StandardClassificationAdapter",
    "MILClassificationAdapter",
    "MLPAdapter",
    "Direction",
    "ParamFilter",
    "random_direction",
    "dominant_hessian_directions",
    "filter_normalize_direction",
    "apply_perturbation",
    "sample_1d",
    "sample_2d",
    "AttentionMILMLP",
    "load_model_from_checkpoint",
    "load_model_and_adapter",
    "EmbeddingBagDataset",
    "variable_length_collate",
    "padded_bag_collate",
]
