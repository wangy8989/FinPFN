from __future__ import annotations

from enum import Enum


class TaskType(str, Enum):
    """Task type of the data for fine-tuning."""

    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary"
    MULTICLASS_CLASSIFICATION = "multiclass"


class SupportedValidationMetric(str, Enum):
    """Supported validation metrics for fine-tuning."""

    RMSE = "rmse"
    R2 = "r2"

    ROC_AUC = "roc_auc"
    ROC_AUC_MULTICLASS = "roc_auc_ovo_macro"
    ACCURACY = "accuracy"
    BALANCED_ACC = "balanced_accuracy"

    MCC = "mcc"


class SupportedDevice(str, Enum):
    """Supported device for fine-tuning."""

    CPU = "cpu"
    GPU = "cuda"
