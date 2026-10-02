"""Trainers (TRL LoRA DPO; labeled fixture), preference rendering and checkpoint publication.

Importing this package does not import torch; `dpo`/`modeling` need the `train` extra.
"""

from .common import NoTrainableExamples, TrainingRequestError, derive_checkpoint_id, load_preference_dataset
from .fixture import FixtureTrainer, base_checkpoint_ref


def get_trainer(name: str, **kwargs):
    """Lazy import so `python -m learning_loop.training.run` does not pre-import its own module."""
    from .run import get_trainer as _get

    return _get(name, **kwargs)


__all__ = [
    "FixtureTrainer",
    "NoTrainableExamples",
    "TrainingRequestError",
    "base_checkpoint_ref",
    "derive_checkpoint_id",
    "get_trainer",
    "load_preference_dataset",
]
