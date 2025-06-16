from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.training_utils.ag_early_stopping import AdaptiveES
    from torch.optim.optimizer import Optimizer


@dataclass
class FineTuneStepResults:
    """Dataclass to store the results of a fine-tuning step."""

    # Minimal step results
    training_loss: float
    """The training loss of the current step."""
    device_utilization: float
    """The device utilization after the current step."""
    step_with_update: bool
    """Whether the optimizer, lr scheduler, and loss made a step."""
    optimizer_step_skipped: bool
    """Whether the optimizer step was skipped due to NaNs before grad scaling."""
    grad_norm_before_clip: float
    """The gradient norm before clipping."""

    # Optionally set by the loop
    step_index: int | None = None
    """The index of the current step."""
    best_validation_loss: float | None = None
    """The best validation loss seen so far."""
    best_validation_score: float | None = None
    """The best validation score seen so far."""
    patience_left: int | None = None
    """The remaining patience left for early stopping."""
    time_left: float | None = None
    """The remaining time left for fine-tuning."""
    validation_loss: float | None = None
    """The validation loss of the current step."""

    def register_meta_state(
        self,
        *,
        step_index: int,
        best_validation_loss: float,
        best_validation_score: float,
        patience_left: int,
        time_left: float,
        validation_loss: float,
        training_loss: float,
    ) -> FineTuneStepResults:
        self.step_index = step_index
        self.best_validation_loss = best_validation_loss
        self.best_validation_score = best_validation_score
        self.patience_left = patience_left
        self.time_left = time_left
        self.validation_loss = validation_loss
        self.training_loss = training_loss
        return self

    def to_results_dict(self) -> dict:
        return {
            "Best Val. Loss": f"{self.best_validation_loss:.6f}",
            "Best Val. Score": f"{self.best_validation_score:.6f}",
            "Training Loss": f"{self.training_loss:.5f}",
            "Val. Loss": f"{self.validation_loss:.6f}",
            "Patience": self.patience_left,
            "Time": self.time_left,
            "Utilization": self.device_utilization,
            "Grad Norm": self.grad_norm_before_clip,
        }


@dataclass
class FineTuneSetup:
    """Configuration for fine-tuning a model."""

    optimizer: Optimizer
    """The optimizer object."""
    loss_fn: callable
    """The loss function used to compute the training loss."""

    batch_size: int
    """The batch size of the fine-tuning."""
    max_steps: int
    """The maximum number of steps for the fine-tuning."""

    adaptive_es: AdaptiveES
    """The configured adaptive early stopping object."""

    update_every_n_steps: int
    """The number of steps to update the model before validation"""
    validate_every_n_steps: int
    """The number of steps to validate the model"""

    data_loader_workers: int
    """The number of workers for the data loader."""

    @property
    def report_str(self):
        return f"""
        === Learning HPs ===
            \tBatch Size: {self.batch_size}
            \tLr: {self.optimizer.defaults["lr"]} | Weight Decay: {self.optimizer.defaults["weight_decay"]}
            \tMax Steps: {self.max_steps}
            \tAdaptiveES: Adaptive Rate {self.adaptive_es.adaptive_rate} | Min Patience {self.adaptive_es.min_patience} | Max Patience {self.adaptive_es.max_patience}
            \tUpdate Every N Steps: {self.update_every_n_steps} | Validate Every N Steps: {self.validate_every_n_steps}
        """
