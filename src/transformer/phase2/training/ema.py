"""
Exponential Moving Average (EMA) of model weights.

Maintains a shadow copy of model parameters that is updated as:
    shadow = decay * shadow + (1 - decay) * param

EMA weights often give 0.5-1% better test performance at zero training cost.
"""

from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average of model weights.

    Usage:
        ema = EMA(model, decay=0.999)
        # After each optimizer step:
        ema.update()
        # For evaluation:
        with ema.apply():
            predictions = model(batch)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay

        # Shadow parameters: deep copy of initial model weights
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update shadow weights: shadow = decay * shadow + (1-decay) * param."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    @contextmanager
    def apply(self):
        """
        Context manager that swaps model weights with shadow weights.

        Restores original weights on exit.
        """
        # Save current weights
        original = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

        try:
            yield
        finally:
            # Restore original weights
            for name, param in self.model.named_parameters():
                if name in original:
                    param.data.copy_(original[name])

    def state_dict(self) -> dict:
        """Serialize EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict):
        """Restore EMA state from checkpoint."""
        self.decay = state_dict["decay"]
        for name, tensor in state_dict["shadow"].items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)
