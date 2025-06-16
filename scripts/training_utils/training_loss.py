from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from scripts.constant_utils import TaskType
from tabpfn.model.bar_distribution import FullSupportBarDistribution

if TYPE_CHECKING:
    from torch.nn.modules.loss import _Loss


def get_loss(*, task_type: TaskType, borders: torch.Tensor) -> _Loss:
    """Get the trainings loss function based on the task type.

    Arguments:
    ----------
    task_type: TaskType
        The task type of the competition.
    borders: int
        The buckets for the regression task (from the model).

    Returns:
    --------
    loss: _Loss
        The loss function to use.

    """
    match task_type:
        case TaskType.REGRESSION:
            loss = FullSupportBarDistribution(
                borders=borders,
                ignore_nan_targets=True,
            )
        case TaskType.BINARY_CLASSIFICATION:
            loss = torch.nn.BCEWithLogitsLoss()
        case TaskType.MULTICLASS_CLASSIFICATION:
            loss = torch.nn.CrossEntropyLoss()
        case _:
            raise ValueError(f"Loss for task type {task_type} not supported.")

    return loss


def compute_loss(
    *,
    loss_fn: _Loss,
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute the loss of the model.

    This methods selects the correct elements from the logits and target tensors based on the loss function.
    This is essentially an adaptor for the output of the TabPFN model.

    Arguments:
    ----------
    loss_fn: _Loss
        The loss function to use.
    logits: torch.Tensor
        The logits of the model.
            * For classification: (n_samples, batch_size, n_classes)
            * For regression: (n_samples, batch_size, ?TODO?)
    target: torch.Tensor (n_samples, batch_size, 1)
        The target values.

    Returns:
    --------
    loss: torch.Tensor

        The loss tensor.
    """
    match loss_fn:
        case torch.nn.modules.BCEWithLogitsLoss():
            logits = logits[:, :, 1]  # select positive class logits only
            target = target[:, :, 0]
        case torch.nn.modules.CrossEntropyLoss():
            logits = logits.reshape(-1, logits.shape[-1])
            target = target.long().flatten()
        case _ if isinstance(loss_fn, FullSupportBarDistribution):
            return loss_fn(logits=logits, y=target[:, :, 0]).mean()
        case _:
            raise ValueError(f"Loss of type {type(loss_fn)} not supported.")

    return loss_fn(input=logits, target=target)
