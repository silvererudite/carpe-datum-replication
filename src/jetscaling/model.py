"""The one model recipe. Size is the ONLY thing that varies across the sweep.

STATUS: contract only. Implement in build step 3 (docs/orientation.md 6.2).

Architecture (fixed by the pre-registration):

    per-particle Linear(F -> d)          F = 7 features
      -> n_layers x pre-LN encoder block  (MHSA + MLP, NO positional embedding)
      -> masked mean-pool over constituents
      -> LayerNorm -> Linear(d -> 1)      one logit
    loss: BCEWithLogits

Why no positional embedding: a jet is a *set* of constituents. The pT ordering in the
file is an artefact of how the data was written, not physics. Making the model
permutation-invariant is the whole reason this is a "set transformer" and not a
sequence model -- and it means the ordering of the 64 kept constituents cannot leak
information the network shouldn't have.

Masking is load-bearing in two places: the attention logits (padded keys must get
-inf before softmax) and the mean-pool (divide by the true constituent count, not by
max_constituents). Get either wrong and the loss silently depends on how many pad
slots a jet happens to have -- which correlates with jet type. That is a leak.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    """Pre-LN block: x + attn(LN(x)); x + mlp(LN(x)).

    Pre-LN (not post-LN) because these runs have no LR tuning budget and post-LN
    needs a warmup schedule tuned per depth to stay stable -- exactly the kind of
    per-run tuning the pre-registration forbids.
    """

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        raise NotImplementedError("build step 3")

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, C, d); key_padding_mask: (B, C) bool, True where the slot is PADDING."""
        raise NotImplementedError("build step 3")


class SetTransformer(nn.Module):
    """Permutation-invariant jet tagger. Returns one logit per jet."""

    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        raise NotImplementedError("build step 3")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: (B, C, F) float; mask: (B, C) bool, True where the constituent is REAL.

        Returns (B,) logits.
        """
        raise NotImplementedError("build step 3")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Exact parameter count, embeddings and biases INCLUDED.

    This number -- not the target from configs/model_sizes.yaml -- is the N that goes
    into the scaling fit. A 10% error in N shifts a fitted exponent by roughly
    0.1*alpha in log space; with only four N values on the grid, that is not noise.
    """
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def build_model(size_cfg: dict, n_features: int) -> SetTransformer:
    """Instantiate from one entry of configs/model_sizes.yaml.

    Log a warning (do not fail) if |count_parameters - target_params| / target > 0.25:
    the ladder should stay roughly log-spaced, and a badly-off rung distorts the fit.
    """
    raise NotImplementedError("build step 3")


def permutation_invariance_check(model: SetTransformer, x: torch.Tensor, mask: torch.Tensor) -> float:
    """Max |logit(x) - logit(shuffle(x))| over a batch. Must be ~1e-5 or smaller.

    Run this in notebook 01. If it is not tiny, something (a positional embedding, a
    mask bug, a reshape that mixes the constituent axis) has broken the set property.
    """
    raise NotImplementedError("build step 3")
