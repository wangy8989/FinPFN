from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import random
from scripts.constant_utils import SupportedDevice, TaskType
import numpy as np
import pandas as pd
import time

if TYPE_CHECKING:
    from scripts.metric_utils.ag_metrics import Scorer
    from tabpfn.model.transformer import PerFeatureTransformer


def validate_tabpfn(
    *,
    X_train: torch.Tensor,  # (n_samples, batch_size, n_features)
    y_train: torch.Tensor,  # (n_samples, batch_size, 1)
    X_val: torch.Tensor,    # (n_samples, batch_size, n_features)
    y_val: torch.Tensor,    # (n_samples, batch_size, 1)
    chunk_size: int,      # adjust to fit your GPU
    validation_metric: Scorer,
    model: PerFeatureTransformer,
    model_forward_fn: Callable,
    task_type: TaskType,
    device: SupportedDevice,
    is_data_parallel: bool,
) -> float:
    """
    Validate the TabPFN model and return a loss (lower is better).
    Now supports batch_size > 1 by looping through batch elements.
    """
    
    # Move tensors to device
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    scores = []
    batch_size = X_train.shape[1]
    batch_indices = torch.arange(batch_size, device=device)  # full validation set
    loss = model.module.criterion if is_data_parallel else model.criterion

    # Perform many iterations, each with batches
    for i in range(0, batch_size, chunk_size):
        chunk = batch_indices[i:i + chunk_size]

        # Perform the forward pass for all selected batches at once
        # if logits for bar distribution: shape (n_samples, chunk_size, 5000), if prediction: (n_samples, chunk_size)
        pred_logits = model_forward_fn(
            model=model,
            X_train=X_train[:, chunk],  # shape (n_samples, chunk_size, n_features)
            y_train=y_train[:, chunk],  # shape (n_samples, chunk_size, 1)
            X_test=X_val[:, chunk],     # shape (n_samples, chunk_size, n_features)
            forward_for_validation=False,      # If True, This means that a regression model will return predictions instead of logits for the bar distribution.
        )

        # Handle each task type without a loop
        if task_type == TaskType.REGRESSION:
            # Flatten predictions and true values for regression
            # shape (n_samples, chunk_size, 5000) -> (n_samples*chunk_size, 5000)
            y_pred = pred_logits.float().flatten(start_dim=0, end_dim=1)#.cpu().detach().numpy()  
            y_true = y_val[:, chunk].float().flatten()#.cpu().detach().numpy()  # shape: (n_samples*chunk_size,)

        elif task_type == TaskType.BINARY_CLASSIFICATION:
            if validation_metric.needs_threshold or validation_metric.needs_proba:
                # Binary class probabilities, shape: (n_samples*chunk_size,)
                y_pred = torch.sigmoid(pred_logits[:, :, 1]).flatten().cpu().detach().numpy()
            else:
                y_pred = torch.softmax(pred_logits, dim=-1).flatten(start_dim=0, end_dim=1).cpu().detach().numpy()  # Raw class scores
            y_true = y_val[:, chunk].long().flatten().cpu().detach().numpy()

        elif task_type == TaskType.MULTICLASS_CLASSIFICATION:
            y_pred = torch.softmax(pred_logits, dim=-1).flatten(start_dim=0, end_dim=1).cpu().detach().numpy()  # Multiclass probabilities
            y_true = y_val[:, chunk].long().flatten().cpu().detach().numpy()

        else:
            raise ValueError(f"Task type {task_type} not supported.")
                
        # ce: (5000=500*10, 5000), (5000, )
#         score = validation_metric(y_true=y_true, y_pred=y_pred)
        score = loss(logits=y_pred, y=y_true).cpu().detach().numpy().mean()  # bar distribution loss
        scores.append(score)

    # Average across batch
    mean_score = np.mean(scores)
    
    X_train.cpu()
    y_train.cpu()
    X_val.cpu()
    y_val.cpu()

    return mean_score #validation_metric.convert_score_to_error(score=mean_score)  # convert to negative

