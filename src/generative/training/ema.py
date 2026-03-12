"""Exponential Moving Average (EMA) of model weights.

Maintains a shadow copy of model parameters updated as:
    shadow = decay * shadow + (1 - decay) * param

EMA weights often yield modest generalisation improvements at zero training cost.
Uses a context-manager interface (``with ema.apply():``) for clean eval swaps.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters.

    Usage::

        ema = EMA(model, decay=0.999)
        # After each optimiser step:
        ema.update()
        # For evaluation:
        with ema.apply():
            predictions = model(batch)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay

        # Shadow parameters: clone of initial trainable weights
        self.shadow: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow weights: shadow = decay * shadow + (1 - decay) * param."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    @contextmanager
    def apply(self):
        """Context manager that swaps model weights with shadow weights.

        Restores original weights on exit, even if an exception is raised.
        """
        original: dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
        try:
            yield
        finally:
            for name, param in self.model.named_parameters():
                if name in original:
                    param.data.copy_(original[name])

    def state_dict(self) -> dict:
        """Serialise EMA state for checkpointing."""
        return {
            "decay": self.decay,
            "shadow": {k: v.clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore EMA state from a checkpoint."""
        self.decay = state_dict["decay"]
        for name, tensor in state_dict["shadow"].items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)
